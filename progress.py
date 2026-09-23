from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError):
        return default


def _save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(data, target, indent=2)
            target.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_completed(path: Path) -> set[str]:
    data = _load(path, {"completed": []})
    return set(data.get("completed", [])) if isinstance(data, dict) else set()


def save_completed(path: Path, completed: set[str]) -> None:
    _save(path, {"completed": sorted(completed)})


def load_failed(path: Path) -> list[dict[str, Any]]:
    data = _load(path, {"failed": []})
    if not isinstance(data, dict) or not isinstance(data.get("failed"), list):
        return []
    return data["failed"]


def save_failed(path: Path, failed: list[dict[str, Any]]) -> None:
    _save(path, {"failed": failed})


def load_metadata(path: Path) -> list[dict[str, Any]]:
    data = _load(path, [])
    return data if isinstance(data, list) else []


def save_metadata(path: Path, metadata: list[dict[str, Any]]) -> None:
    _save(path, metadata)
