from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


class EnvConfig:
    """Simple .env loader with OS overrides."""

    def __init__(self, env_path: Path) -> None:
        self.env_path = env_path
        self._values = self._load_file()

    def _load_file(self) -> Dict[str, str]:
        if not self.env_path.exists():
            return {}
        values: Dict[str, str] = {}
        with self.env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip()
        return values

    def get(self, key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, self._values.get(key, default))

    def get_int(self, key: str, default: int) -> int:
        value = self.get(key, None)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def get_float(self, key: str, default: float) -> float:
        value = self.get(key, None)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
