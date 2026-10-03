"""Dataset manifest: one row per utterance/file, with provenance, consent and split.

Integrity rules enforced by validate_row:
- transcripts are never assumed: `not_available` rows must have an empty transcript,
  `verified` rows must have one;
- native recordings require consent_status == "given";
- speaker IDs are pseudonymous (VH_S###) or "unknown", never a name or phone number.
"""
from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from asr_baseline.utils import write_csv

from .config import DATA_DIR, MANIFEST, SPLITS, rel

FIELDS = [
    "utterance_id",
    "audio_path",
    "speaker_id",
    "district",
    "age_group",
    "gender",
    "duration_sec",
    "language",
    "dialect",
    "domain",
    "transcript",
    "transcript_status",
    "recording_quality",
    "consent_status",
    "source",
    "license",
    "split",
]

ALLOWED = {
    "domain": {"general", "scripture", "agriculture"},
    "transcript_status": {"not_available", "pending_review", "verified"},
    "recording_quality": {"good", "fair", "poor", "unchecked"},
    "consent_status": {"given", "not_applicable_published"},
    "split": {"train", "validation", "test", "unassigned"},
}
NATIVE_SOURCE = "native_recording"
SPEAKER_ID_RE = re.compile(r"^(VH_S\d{3,}|unknown)$")
SPLIT_NAMES = ["train", "validation", "test"]

VAHNT_SOURCE = "bible.com VAHNT audio Bible (version 3451)"
VAHNT_LICENSE = "unverified: local research use only, see README Dataset provenance"


def validate_row(row: dict) -> list[str]:
    uid = row.get("utterance_id", "?")
    errors = [f"{uid}: missing field {field}" for field in FIELDS if field not in row]
    if errors:
        return errors
    for field, allowed in ALLOWED.items():
        if row[field] not in allowed:
            errors.append(f"{uid}: {field}={row[field]!r} not in {sorted(allowed)}")
    if not SPEAKER_ID_RE.match(row["speaker_id"]):
        errors.append(f"{uid}: speaker_id {row['speaker_id']!r} must be VH_S### or unknown")
    if row["transcript_status"] == "not_available" and row["transcript"].strip():
        errors.append(f"{uid}: transcript present but status is not_available")
    if row["transcript_status"] == "verified" and not row["transcript"].strip():
        errors.append(f"{uid}: verified status with empty transcript")
    if row["source"] == NATIVE_SOURCE and row["consent_status"] != "given":
        errors.append(f"{uid}: native recording without consent")
    return errors


def read_manifest(data_dir: Path = DATA_DIR) -> list[dict]:
    path = data_dir / MANIFEST
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_manifest(rows: list[dict], data_dir: Path = DATA_DIR) -> None:
    errors = [error for row in rows for error in validate_row(row)]
    if errors:
        raise ValueError("invalid manifest rows:\n" + "\n".join(errors))
    write_csv(data_dir / MANIFEST, sorted(rows, key=lambda r: r["utterance_id"]), FIELDS)


def upsert(rows: list[dict], new_rows: list[dict]) -> list[dict]:
    """Insert or replace by utterance_id. Existing split assignments are preserved."""
    by_id = {row["utterance_id"]: row for row in rows}
    for row in new_rows:
        old = by_id.get(row["utterance_id"])
        by_id[row["utterance_id"]] = {**row, "split": old["split"]} if old else row
    return list(by_id.values())


def vahnt_rows(audio_dir: Path, duration_fn) -> list[dict]:
    """VAHNT chapters as general Varhadi speech. Speaker unknown, reference NOT AVAILABLE."""
    return [
        {
            "utterance_id": f"VAHNT_{path.stem}",
            "audio_path": rel(path),
            "speaker_id": "unknown",
            "district": "unknown",
            "age_group": "",
            "gender": "",
            "duration_sec": f"{duration_fn(path):.3f}",
            "language": "mr",
            "dialect": "varhadi",
            "domain": "scripture",
            "transcript": "",
            "transcript_status": "not_available",
            "recording_quality": "unchecked",
            "consent_status": "not_applicable_published",
            "source": VAHNT_SOURCE,
            "license": VAHNT_LICENSE,
            "split": "unassigned",
        }
        for path in sorted(audio_dir.glob("*.mp3"))
    ]


def assign_splits(rows: list[dict], ratios=(0.8, 0.1, 0.1), seed: int = 13) -> list[dict]:
    """Split by SPEAKER so no speaker appears in two splits.

    Rows with speaker_id "unknown" (e.g. VAHNT) always go to train: their speakers cannot
    be told apart, so leakage into validation/test could not be ruled out.
    Known speakers are shuffled with a fixed seed and assigned greedily by duration: test
    first, then validation, the rest train. With few speakers the realised ratios will
    differ from the targets; they are reported, not forced.
    """
    if abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("ratios must sum to 1")
    duration = defaultdict(float)
    for row in rows:
        if row["speaker_id"] != "unknown":
            duration[row["speaker_id"]] += float(row["duration_sec"] or 0) or 1.0
    speakers = sorted(duration)
    random.Random(seed).shuffle(speakers)
    total = sum(duration.values())
    targets = {"test": ratios[2] * total, "validation": ratios[1] * total}
    filled = {"test": 0.0, "validation": 0.0}
    split_of = {}
    for speaker in speakers:
        split = next((s for s in ("test", "validation") if filled[s] < targets[s]), "train")
        if split != "train":
            filled[split] += duration[speaker]
        split_of[speaker] = split
    return [{**row, "split": split_of.get(row["speaker_id"], "train")} for row in rows]


def speaker_leaks(rows: list[dict]) -> dict[str, set[str]]:
    """Speakers that appear in more than one of train/validation/test."""
    splits = defaultdict(set)
    for row in rows:
        if row["split"] in SPLIT_NAMES and row["speaker_id"] != "unknown":
            splits[row["speaker_id"]].add(row["split"])
    return {speaker: found for speaker, found in splits.items() if len(found) > 1}


def write_splits(rows: list[dict], data_dir: Path = DATA_DIR) -> dict[str, dict]:
    leaks = speaker_leaks(rows)
    if leaks:
        raise ValueError(f"speaker leakage across splits: {leaks}")
    summary = {}
    for split in SPLIT_NAMES:
        subset = [row for row in rows if row["split"] == split]
        write_csv(data_dir / SPLITS / f"{split}.csv", subset, FIELDS)
        summary[split] = {
            "utterances": len(subset),
            "speakers": len({row["speaker_id"] for row in subset}),
            "minutes": round(sum(float(row["duration_sec"] or 0) for row in subset) / 60, 2),
        }
    return summary
