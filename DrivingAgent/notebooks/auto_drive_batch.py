#!/usr/bin/env python3
"""
Utility to collect multiple successful CARLA driving runs automatically.

The script repeatedly launches the simulator Docker container and the
drive_and_infer_async.py process, watches both processes, and restarts the
pipeline whenever either part fails. It stops once the requested number of
successful runs is reached or when the maximum attempt count is exceeded.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest


SCRIPT_DIR = Path(__file__).resolve().parent
DRIVING_AGENT_DIR = SCRIPT_DIR.parent
CARLA_RUNNER_ROOT = DRIVING_AGENT_DIR.parent
SIMULATOR_DIR = CARLA_RUNNER_ROOT / "CarlaSimulator"
DEFAULT_SLACK_WEBHOOK = (
    "https://hooks.slack.com/services/"
    "T719J8DT9/B09SH3VD27N/FydPAs5MuxnsWIdLT0RbS68T"
)
NFS_DIR = Path("/home/tsuruoka/nfs/BEV/CarlaRunner/20251213_OccupancyAD_SystemEval_result")


DEFAULT_DRIVER_COMMAND: Sequence[str] = [
    "python3",
    "DrivingAgent/notebooks/drive_and_infer_async.py",
    "--spawn-index",
    "361",
    "--duration-seconds",
    "10",
    "--traffic-manager-port",
    "60055",
    "--driver-debug",
    "--timeout",
    "120",
    "--port",
    "{sim_port}",
    "--camera-output-root",
    str(NFS_DIR),
    "--max-pending-frames",
    "128",
    "--tick-timeout-seconds",
    "120",
    "--target-speed-mps",
    "4.0",
    "--use-lane-yaw",
    "--use-lane-following",
    "--inference-mode",
    "sync",
    "--log-control-debug",
    "--class-weight-json",
    str(SCRIPT_DIR / "class_weight.json"),
    "--use-occ-planner",
    "--enable-planner-bumper-offset",
    "--planner-weight-speed",
    "2.0",
    "--acceleration-gain",
    "1.0",
    "--planner-horizon-s",
    "3.0",
    "--enable-planner-bumper-offset",
    "--cost-aggregate",
    "max",
    "--planner-forward-axis",
    "col",
    "--planner-weight-occ",
    "1.0",
    "--attack-config",
    "./DrivingAgent/config/attack_appearing.json",
    #"--test-car-distance",
    #"12",
]


def build_default_driver_command(sim_port: int) -> List[str]:
    return [token.format(sim_port=sim_port) for token in DEFAULT_DRIVER_COMMAND]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple CARLA driving attempts with automatic recovery."
    )
    parser.add_argument(
        "--success-target",
        type=int,
        default=5,
        help="Number of successful driving runs required before stopping.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Stop after this many attempts even if successes < target.",
    )
    parser.add_argument(
        "--sim-port",
        type=int,
        default=5555,
        help="Port used by the CARLA simulator and driver (--port).",
    )
    parser.add_argument(
        "--sim-startup-delay",
        type=float,
        default=10.0,
        help="Seconds to wait after starting the simulator before launching the driver.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between simulator/driver health checks.",
    )
    parser.add_argument(
        "--between-runs-delay",
        type=float,
        default=5.0,
        help="Seconds to wait between attempts after cleanup.",
    )
    parser.add_argument(
        "--driver-command",
        type=str,
        default="",
        help=(
            "Override the driving command. Provide a full shell-style string; "
            "it will be split with shlex.split(). If empty the default "
            "drive_and_infer_async invocation is used."
        ),
    )
    parser.add_argument(
        "--attack-config-path",
        type=Path,
        help="Override the --attack-config argument passed to the driver process.",
    )
    parser.add_argument(
        "--camera-output-root-path",
        type=Path,
        help="Override the --camera-output-root argument passed to the driver process.",
    )
    parser.add_argument(
        "--sim-script",
        type=Path,
        default=SIMULATOR_DIR / "run_carla_simulator.sh",
        help="Path to run_carla_simulator.sh.",
    )
    parser.add_argument(
        "--carla-container",
        type=str,
        default="carla-server",
        help="Docker container name to remove between attempts.",
    )
    parser.add_argument(
        "--pkill-pattern",
        type=str,
        default="async",
        help="Process pattern supplied to `pkill -f -9` for straggler cleanup.",
    )
    parser.add_argument(
        "--disable-pkill",
        action="store_true",
        help="Skip the pkill cleanup step.",
    )
    parser.add_argument(
        "--slack-webhook",
        type=str,
        default=os.environ.get("SLACK_URL", DEFAULT_SLACK_WEBHOOK),
        help="Send progress notifications to this Slack incoming webhook URL.",
    )
    return parser.parse_args()


class AutoDriveController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sim_script = args.sim_script.resolve()
        self.sim_cwd = self.sim_script.parent
        self.driver_cwd = CARLA_RUNNER_ROOT
        self.driver_command = self._resolve_driver_command()
        self.sim_process: Optional[subprocess.Popen[bytes]] = None
        self.driver_process: Optional[subprocess.Popen[bytes]] = None
        self.slack_webhook: Optional[str] = args.slack_webhook

    def _resolve_driver_command(self) -> List[str]:
        if self.args.driver_command:
            return shlex.split(self.args.driver_command)
        command = build_default_driver_command(self.args.sim_port)
        if self.args.attack_config_path:
            command = self._override_arg(command, "--attack-config", str(self.args.attack_config_path))
        if self.args.camera_output_root_path:
            command = self._override_arg(
                command, "--camera-output-root", str(self.args.camera_output_root_path)
            )
        return command

    def _override_arg(self, command: List[str], flag: str, value: str) -> List[str]:
        command = list(command)
        try:
            idx = command.index(flag)
        except ValueError:
            command.extend([flag, value])
            return command
        if idx + 1 < len(command):
            command[idx + 1] = value
        else:
            command.append(value)
        return command

    def run(self) -> None:
        successes = 0
        attempt = 0
        while successes < self.args.success_target:
            attempt += 1
            if attempt > self.args.max_attempts:
                print(
                    f"Reached max attempts ({self.args.max_attempts}) "
                    f"with {successes} successes.",
                    file=sys.stderr,
                )
                break
            print(f"\n=== Attempt {attempt} (current successes: {successes}) ===")
            try:
                result = self._run_single_attempt(attempt, successes)
            except KeyboardInterrupt:
                print("Interrupted by user, shutting down...")
                self._emergency_cleanup()
                raise

            if result:
                successes += 1
                print(f"Attempt {attempt} finished successfully ({successes} total).")
                self._notify_slack(
                    f":white_check_mark: Attempt {attempt} succeeded "
                    f"({successes}/{self.args.success_target} successes)."
                )
            else:
                print(f"Attempt {attempt} failed, restarting everything.")

            if successes < self.args.success_target:
                time.sleep(self.args.between_runs_delay)

        print(
            f"\nCompleted automation with {successes} successful runs "
            f"after {attempt} attempts."
        )
        self._notify_slack(
            f"Automation finished with {successes}/{self.args.success_target} "
            f"successes after {attempt} attempts."
        )

    def _run_single_attempt(self, attempt: int, successes: int) -> bool:
        del attempt, successes  # currently unused but kept for potential logging.
        self.sim_process = self._launch_simulator()
        time.sleep(self.args.sim_startup_delay)
        if self.sim_process.poll() is not None:
            raise RuntimeError("Simulator exited before driver launch.")

        self.driver_process = self._launch_driver()
        actor, exit_code = self._wait_for_exit()
        success = actor == "driver" and exit_code == 0
        if success:
            print("Driver process reported success (exit code 0).")
        else:
            print(
                f"{actor.capitalize()} exited with code {exit_code}; "
                "marking attempt as failed."
            )

        self._cleanup_process(self.driver_process, "driver")
        self._cleanup_process(self.sim_process, "simulator")
        self.driver_process = None
        self.sim_process = None

        self._post_attempt_cleanup()
        return success

    def _launch_simulator(self) -> subprocess.Popen[bytes]:
        cmd = ["bash", str(self.sim_script), str(self.args.sim_port)]
        print(f"Starting simulator: {' '.join(cmd)} (cwd={self.sim_cwd})")
        return subprocess.Popen(cmd, cwd=self.sim_cwd)

    def _launch_driver(self) -> subprocess.Popen[bytes]:
        print(
            "Starting driver command:\n  "
            + " ".join(self.driver_command)
            + f"\n(cwd={self.driver_cwd})"
        )
        return subprocess.Popen(self.driver_command, cwd=self.driver_cwd)

    def _wait_for_exit(self) -> Tuple[str, int]:
        while True:
            if self.sim_process and self.sim_process.poll() is not None:
                return "simulator", self.sim_process.returncode or 0
            if self.driver_process and self.driver_process.poll() is not None:
                return "driver", self.driver_process.returncode or 0
            time.sleep(self.args.poll_interval)

    def _cleanup_process(
        self, process: Optional[subprocess.Popen[bytes]], name: str
    ) -> None:
        if not process or process.poll() is not None:
            return
        print(f"Terminating {name} process...")
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print(f"Forcing {name} process to exit...")
            process.kill()
            process.wait(timeout=5)

    def _post_attempt_cleanup(self) -> None:
        if not self.args.disable_pkill and self.args.pkill_pattern:
            self._run_command(
                ["pkill", "-f", "-9", self.args.pkill_pattern],
                "pkill cleanup",
                check=False,
            )
        if self.args.carla_container:
            self._run_command(
                ["docker", "rm", "-f", self.args.carla_container],
                "docker cleanup",
                check=False,
            )

    def _run_command(
        self, command: Sequence[str], label: str, check: bool = True
    ) -> None:
        try:
            subprocess.run(command, check=check)
        except subprocess.CalledProcessError as exc:
            print(f"{label} failed with code {exc.returncode} (ignored).", file=sys.stderr)

    def _emergency_cleanup(self) -> None:
        self._cleanup_process(self.driver_process, "driver")
        self._cleanup_process(self.sim_process, "simulator")
        self.driver_process = None
        self.sim_process = None

    def _notify_slack(self, text: str) -> None:
        if not self.slack_webhook:
            return
        data = json.dumps({"text": text}).encode("utf-8")
        req = urlrequest.Request(
            self.slack_webhook,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=10):
                pass
        except urlerror.URLError as exc:
            print(f"Failed to send Slack notification: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    controller = AutoDriveController(args)
    try:
        controller.run()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
