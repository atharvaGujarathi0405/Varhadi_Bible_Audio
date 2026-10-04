from __future__ import annotations

import csv
from pathlib import Path

from config import ROOT_DIR


def repo_path(path) -> str:
    """Repo-relative POSIX path when possible, so the same file always has one spelling
    (resume keys, CSVs readable from both Windows and WSL)."""
    try:
        return Path(path).resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return Path(path).as_posix()


def strip_punctuation(text: str) -> str:
    """Drop Unicode punctuation/symbols (categories P*/S*) and collapse whitespace. Keeps
    letters, Devanagari matras/virama and digits (a regex \w would strip combining marks).
    The ONLY normalisation applied before scoring: no spelling or dialect changes."""
    import unicodedata

    chars = (" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join("".join(chars).split())


def read_asr_metadata(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row.setdefault("reference_text", "")
        row.setdefault("speaker_id", "")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
