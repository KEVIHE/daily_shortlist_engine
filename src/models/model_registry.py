"""Model artifact paths and lightweight persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODEL_FILES = {
    "ranker": "ranker.txt",
    "classifier": "classifier.txt",
    "regressor_up": "regressor_up.txt",
    "regressor_down": "regressor_down.txt",
    "metadata": "metadata.json",
}


def ensure_model_dir(project_root: Path) -> Path:
    model_dir = project_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def model_path(project_root: Path, name: str) -> Path:
    return ensure_model_dir(project_root) / MODEL_FILES[name]


def save_metadata(project_root: Path, payload: dict[str, Any]) -> Path:
    path = model_path(project_root, "metadata")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def load_metadata(project_root: Path) -> dict[str, Any]:
    path = model_path(project_root, "metadata")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
