import argparse
import importlib.util
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

import cv2

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TMP_DIR = SCRIPT_DIR / "tmp"
DRIVING_AGENT_ROOT = SCRIPT_DIR.parents[1]

PATH_MAPPINGS = [
    (Path("/home/tsuruoka/hdd/BEV"), Path("/workspace")),
    (Path("/data/hdd/tsuruoka/BEV"), Path("/workspace")),
]


def find_camera_images(
    base_dir: Path, frame_id: Optional[str], template: Optional[str]
) -> Dict[str, Path]:
    exts = [".jpg", ".jpeg", ".png"]
    results = {}
    for cam in CAMERAS:
        candidate = resolve_with_template(base_dir, cam, frame_id, template, exts)
        if candidate is None:
            candidate = search_by_name(base_dir, cam, exts)
        results[cam] = candidate
    return results


def resolve_with_template(
    base_dir: Path, cam: str, frame_id: Optional[str], template: Optional[str], exts: List[str]
) -> Optional[Path]:
    if not template:
        return None
    context = {
        "cam": cam,
        "cam_lower": cam.lower(),
        "cam_short": cam.replace("CAM_", "").lower(),
        "frame": frame_id or "",
    }
    try:
        rel = template.format(**context)
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in template: {exc}")
    target = base_dir / rel
    if target.exists():
        return target
    if target.suffix == "":
        for ext in exts:
            candidate = target.with_suffix(ext)
            if candidate.exists():
                return candidate
    return None


def search_by_name(base_dir: Path, cam: str, exts: List[str]) -> Path:
    tokens = [
        cam,
        cam.lower(),
        cam.replace("CAM_", ""),
        cam.replace("CAM_", "").lower(),
    ]
    matches: List[Path] = []
    for ext in exts:
        for file in base_dir.rglob(f"*{ext}"):
            stem_lower = file.stem.lower()
            if any(token.lower() in stem_lower for token in tokens):
                matches.append(file)
    if not matches:
        raise FileNotFoundError(f"No image found for {cam} under {base_dir}")
    unique = []
    seen = set()
    for m in matches:
        resolved = m.resolve()
        if resolved not in seen:
            unique.append(m)
            seen.add(resolved)
    if len(unique) > 1:
        raise ValueError(
            f"Multiple candidates for {cam}: {', '.join(str(p) for p in unique[:5])}"
        )
    return unique[0]


def convert_and_save(images: Dict[str, Path], dest_dir: Path) -> Dict[str, Dict[str, Path]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Dict[str, Path]] = {}
    for cam, src in images.items():
        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {src}")
        dest = dest_dir / f"{cam}.jpg"
        if not cv2.imwrite(str(dest), img):
            raise IOError(f"Failed to save image: {dest}")
        host_path = dest.resolve()
        container_path = convert_to_container_path(host_path)
        saved[cam] = {"host": host_path, "container": container_path}
    return saved


def convert_to_container_path(path: Path) -> Path:
    original = Path(path)
    original_str = original.as_posix()
    resolved = original.resolve()
    resolved_str = resolved.as_posix()

    for host_root, container_root in PATH_MAPPINGS:
        host_prefix = host_root.as_posix()
        container_prefix = container_root.as_posix()

        if original_str.startswith(host_prefix):
            suffix = original_str[len(host_prefix):]
            return Path(container_prefix + suffix)

        if resolved_str.startswith(host_prefix):
            suffix = resolved_str[len(host_prefix):]
            return Path(container_prefix + suffix)

    return original


def convert_to_host_path(path: Path, base_dir: Optional[Path] = None) -> Path:
    path_obj = Path(path)
    if not path_obj.is_absolute() and base_dir is not None:
        candidate = (base_dir / path_obj).resolve()
        if candidate.exists():
            return candidate

    path_str = path_obj.as_posix()
    resolved_str = path_obj.resolve().as_posix() if path_obj.exists() else path_str

    for host_root, container_root in PATH_MAPPINGS:
        container_prefix = container_root.as_posix()
        host_prefix = host_root.as_posix()

        if path_str.startswith(container_prefix):
            suffix = path_str[len(container_prefix):]
            candidate = Path(host_prefix + suffix)
            if candidate.exists():
                return candidate.resolve()

        if resolved_str.startswith(container_prefix):
            suffix = resolved_str[len(container_prefix):]
            candidate = Path(host_prefix + suffix)
            if candidate.exists():
                return candidate.resolve()

    if path_obj.is_absolute():
        return path_obj.resolve()

    if base_dir is not None:
        return (base_dir / path_obj).resolve()

    return path_obj.resolve()


def update_pkl(template_pkl: Path, output_pkl: Path, image_paths: Dict[str, Dict[str, Path]]) -> None:
    with open(template_pkl, "rb") as f:
        data = pickle.load(f)
    for sample in data["infos"]:
        for cam, paths in image_paths.items():
            if cam not in sample["cams"]:
                raise KeyError(f"{cam} not in PKL sample.")
            sample["cams"][cam]["data_path"] = str(paths["container"])
    with open(output_pkl, "wb") as f:
        pickle.dump(data, f)


def _load_payload_builder():
    module_path = SCRIPT_DIR / "2b_build_openocc_payload.py"
    spec = importlib.util.spec_from_file_location("openocc_payload_builder", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load payload builder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["openocc_payload_builder"] = module
    spec.loader.exec_module(module)
    return module


def capture_with_carla(
    template_pkl: Path,
    template_index: int,
    spawn_index: int,
    record_seconds: float,
) -> Tuple[Dict[str, Any], Path]:
    module = _load_payload_builder()
    builder = module.OpenOccPayloadBuilder(module.BASE_DIR, module.CONFIG)
    template_info = module.load_template_info(template_pkl, template_index)
    payload = builder.build_payload(spawn_index, record_seconds, template_info=template_info)
    return payload, builder.base_dir


def copy_payload_images(payload: Dict[str, Any], dest_dir: Path, base_dir: Path) -> Dict[str, Dict[str, Path]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Path]] = {}
    sample = payload["infos"][0]
    for cam, entry in sample["cams"].items():
        src = convert_to_host_path(Path(entry["data_path"]), base_dir)
        if not src.exists():
            raise FileNotFoundError(f"Camera image not found: {src}")
        dest = dest_dir / f"{cam}.jpg"
        shutil.copy2(src, dest)
        host_path = dest.resolve()
        container_path = convert_to_container_path(host_path)
        results[cam] = {"host": host_path, "container": container_path}
    return results


def save_payload(payload: Dict[str, Any], output_pkl: Path) -> None:
    with output_pkl.open("wb") as f:
        pickle.dump(payload, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Carla single frame into OpenOccupancy PKL."
    )
    parser.add_argument(
        "--carla-dir",
        type=Path,
        default=None,
        help="Directory containing Carla camera images for manual mode. "
        "If omitted, images will be captured from a running CARLA simulator.",
    )
    parser.add_argument(
        "--template-pkl",
        default=Path("/home/tsuruoka/hdd/BEV/OpenOccupancy/notebooks/tmp/_my_single_infer.pkl"),
        type=Path,
        help="Template PKL file to duplicate.",
    )
    parser.add_argument(
        "--output-pkl",
        default=DEFAULT_TMP_DIR / "_carla_infer.pkl",
        type=Path,
        help="Output PKL file path.",
    )
    parser.add_argument(
        "--output-image-dir",
        default=DEFAULT_TMP_DIR / "carla_frame",
        type=Path,
        help="Directory to store converted images.",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default=None,
        help="Optional frame identifier for template placeholder {frame}.",
    )
    parser.add_argument(
        "--filename-template",
        type=str,
        default=None,
        help=(
            "Optional template for locating files (placeholders: {cam}, {cam_lower},"
            " {cam_short}, {frame}). Example: 'frames/{frame}/{cam}.png'"
        ),
    )
    parser.add_argument(
        "--template-index",
        type=int,
        default=0,
        help="Index of the template sample to use when loading template-pkl "
        "and when merging metadata for CARLA captures.",
    )
    parser.add_argument(
        "--spawn-index",
        type=int,
        default=361,
        help="CARLA spawn index to use when capturing a frame (ignored in manual mode).",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=1.0,
        help="How long to wait after spawning sensors before capturing images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.carla_dir:
        if not args.carla_dir.exists():
            raise FileNotFoundError(f"Directory not found: {args.carla_dir}")
        camera_images = find_camera_images(args.carla_dir, args.frame_id, args.filename_template)
        saved_paths = convert_and_save(camera_images, args.output_image_dir)
        update_pkl(args.template_pkl, args.output_pkl, saved_paths)

        print("Generated:", args.output_pkl)
        for cam, paths in saved_paths.items():
            print(f"{cam}: host={paths['host']} container={paths['container']}")
    else:
        payload, base_dir = capture_with_carla(
            args.template_pkl,
            args.template_index,
            args.spawn_index,
            args.record_seconds,
        )
        host_base = base_dir if base_dir is not None else DRIVING_AGENT_ROOT
        saved_paths = copy_payload_images(payload, args.output_image_dir, host_base)
        update_pkl(args.template_pkl, args.output_pkl, saved_paths)

        print(f"Generated: {args.output_pkl}")
        for cam, paths in saved_paths.items():
            print(f"{cam}: host={paths['host']} container={paths['container']}")


if __name__ == "__main__":
    main()
