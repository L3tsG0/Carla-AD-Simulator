from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunLogger:
    """Collect structured run events and write them to a JSONL file."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.start_time = time.time()
        self._file_initialized = False

    def log(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        entry: Dict[str, Any] = {
            "event": event,
            "timestamp": time.time(),
        }
        entry.update(payload)
        with self._lock:
            self.events.append(entry)
            self._write_entry(entry)

    def flush(self) -> None:
        """No-op when writing incrementally; kept for API compatibility."""
        return

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of buffered events (for debugging/testing)."""
        with self._lock:
            return {"events": list(self.events)}

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        mode = "a"
        if not self._file_initialized:
            mode = "w"
            self._file_initialized = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open(mode, encoding="utf-8") as fh:
            json.dump(entry, fh)
            fh.write("\n")
            fh.flush()
