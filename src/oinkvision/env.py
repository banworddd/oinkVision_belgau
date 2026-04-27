"""Small helpers for reading local .env files and applying path overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_under_root(root: Path, value: str | Path | None) -> Path:
    if value is None:
        return root
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def load_local_env(env_path: str | Path | None = None) -> None:
    path = Path(env_path) if env_path is not None else PROJECT_ROOT / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def get_data_root() -> Path:
    load_local_env()
    return Path(os.environ.get("OINK_DATA_ROOT", str(PROJECT_ROOT.parent / "data")))


def get_output_root() -> Path:
    load_local_env()
    return Path(os.environ.get("OINK_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs")))


def apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    load_local_env()

    data_root = get_data_root()
    output_root = get_output_root()

    config.setdefault("paths", {})
    configured_output_dir = config["paths"].get("output_dir", "default_run")
    resolved_output_dir = _resolve_under_root(output_root, configured_output_dir)

    config["paths"]["data_root"] = str(data_root)
    config["paths"]["train_annotation_dir"] = str(data_root / "train" / "annotation")
    config["paths"]["output_dir"] = str(resolved_output_dir)

    config.setdefault("data", {})
    frame_cache_dir = config["data"].get("frame_cache_dir")
    if frame_cache_dir is not None:
        config["data"]["frame_cache_dir"] = str(_resolve_under_root(output_root, frame_cache_dir))

    return config
