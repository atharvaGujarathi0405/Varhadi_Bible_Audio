from __future__ import annotations

from pathlib import Path

from config import ROOT_DIR  # reuse the downloader's root; do not duplicate

DATA_DIR = ROOT_DIR / "data"

# Relative to a data dir, so tests can point everything at a tmp dir.
MANIFEST = Path("metadata/manifest.csv")
GOLD = Path("transcripts/gold_transcripts.csv")
GOLD_EVAL = Path("evaluation/gold_eval.csv")  # asr_metadata.csv format, verified rows only
GOLD_EVAL_AUDIO = Path("evaluation/audio")
SPLITS = Path("splits")
RAW_RECORDINGS = Path("raw/recordings")  # original uploads, preserved byte-for-byte
PROCESSED_RECORDINGS = Path("processed/recordings")  # 16 kHz mono WAV used by ASR
VAHNT_TEXT = Path("transcripts/vahnt_text")  # published chapter text + provenance (JSON)
ALIGNMENTS = Path("transcripts/alignments")  # candidate aligned segments, not verified
REGISTRY = Path("private/speaker_registry.csv")  # gitignored; pseudonym <-> keyed hash only

SUBDIRS = ["raw", "processed", "transcripts", "metadata", "splits", "evaluation", "private"]


def rel(path: Path) -> str:
    """Repo-relative POSIX path when possible, so CSVs work from both Windows and WSL."""
    try:
        return Path(path).resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return Path(path).as_posix()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate
