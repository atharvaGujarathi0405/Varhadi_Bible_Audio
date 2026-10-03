"""Gold (manually verified) transcript workflow.

    seed  ->  pending_review  --review-->  verified | rejected  --export-->  evaluation set

- The original audio is never modified; a segment is (audio_path, start_sec, end_sec).
- `raw_asr` keeps the untouched baseline output; `corrected` is what a human reviewer
  heard. Seeding from ASR output is a convenience that risks anchoring the reviewer to
  the model's errors, so reviewers must transcribe what they hear, not polish the ASR.
- Only `verified` rows with a reviewer and a non-empty `corrected` text are exported, so
  WER/CER is only ever computed against human-verified references.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from asr_baseline.config import MODEL_NAME, SAMPLE_RATE
from asr_baseline.utils import write_csv

from .config import DATA_DIR, GOLD, GOLD_EVAL, GOLD_EVAL_AUDIO, rel, resolve

FIELDS = [
    "segment_id",
    "audio_path",
    "speaker_id",
    "start_sec",
    "end_sec",
    "raw_asr",
    "corrected",
    "reviewer",
    "review_status",
    "reviewed_at",
    "provenance",
]
STATUSES = {"pending_review", "verified", "rejected"}
EVAL_FIELDS = ["audio_path", "reference_text", "speaker_id"]  # == asr_metadata.csv


def new_row(segment_id, audio_path, speaker_id, start="", end="", raw_asr="", corrected="", provenance=""):
    return {
        "segment_id": segment_id,
        "audio_path": audio_path,
        "speaker_id": speaker_id,
        "start_sec": start,
        "end_sec": end,
        "raw_asr": raw_asr,
        "corrected": corrected,
        "reviewer": "",
        "review_status": "pending_review",
        "reviewed_at": "",
        "provenance": provenance,
    }


def read_gold(data_dir: Path = DATA_DIR) -> list[dict]:
    path = data_dir / GOLD
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def validate_gold(rows: list[dict]) -> list[str]:
    errors, seen = [], set()
    for row in rows:
        sid = row["segment_id"]
        if sid in seen:
            errors.append(f"{sid}: duplicate segment_id")
        seen.add(sid)
        if row["review_status"] not in STATUSES:
            errors.append(f"{sid}: review_status {row['review_status']!r} not in {sorted(STATUSES)}")
        if row["review_status"] == "verified" and not (row["corrected"].strip() and row["reviewer"].strip()):
            errors.append(f"{sid}: verified requires corrected text and reviewer")
        if bool(row["start_sec"]) != bool(row["end_sec"]):
            errors.append(f"{sid}: start_sec and end_sec must both be set or both empty")
        elif row["start_sec"] and float(row["end_sec"]) <= float(row["start_sec"]):
            errors.append(f"{sid}: end_sec must be after start_sec")
    return errors


def write_gold(rows: list[dict], data_dir: Path = DATA_DIR) -> None:
    errors = validate_gold(rows)
    if errors:
        raise ValueError("invalid gold rows:\n" + "\n".join(errors))
    write_csv(data_dir / GOLD, rows, FIELDS)


def seed_from_chunks(rows: list[dict], chunks_dir: Path, audio_dir: Path) -> list[dict]:
    """Add one pending segment per successful ASR chunk (asr_outputs/chunks/<stem>.csv).
    Existing segment_ids are left untouched, so reseeding never overwrites a review."""
    existing = {row["segment_id"] for row in rows}
    added = []
    for chunk_csv in sorted(chunks_dir.glob("*.csv")):
        with chunk_csv.open(encoding="utf-8", newline="") as source:
            latest = {int(r["chunk_index"]): r for r in csv.DictReader(source)}
        for index, chunk in sorted(latest.items()):
            segment_id = f"{chunk_csv.stem}_c{index:03d}"
            if chunk["status"] != "ok" or segment_id in existing:
                continue
            added.append(
                new_row(
                    segment_id,
                    rel(audio_dir / f"{chunk_csv.stem}.mp3"),
                    "unknown",
                    chunk["start_sec"],
                    chunk["end_sec"],
                    raw_asr=chunk["prediction"].strip(),
                    provenance=f"asr_chunk_raw:{MODEL_NAME}",
                )
            )
    return rows + added


def review(rows: list[dict], segment_id: str, corrected: str, reviewer: str, status: str) -> list[dict]:
    matches = [row for row in rows if row["segment_id"] == segment_id]
    if not matches:
        raise KeyError(f"unknown segment_id {segment_id}")
    matches[0].update(
        corrected=corrected.strip(),
        reviewer=reviewer.strip(),
        review_status=status,
        reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    errors = validate_gold(matches)
    if errors:
        raise ValueError("; ".join(errors))
    return rows


def export_gold(data_dir: Path = DATA_DIR) -> int:
    """Write verified segments as 16 kHz WAV clips + an asr_metadata.csv-format file that
    scripts/run_asr_baseline.py --metadata can consume. Returns the number exported."""
    import librosa
    import soundfile as sf

    verified = [row for row in read_gold(data_dir) if row["review_status"] == "verified"]
    audio_dir = data_dir / GOLD_EVAL_AUDIO
    audio_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for row in verified:
        start = float(row["start_sec"]) if row["start_sec"] else 0.0
        duration = float(row["end_sec"]) - start if row["end_sec"] else None
        audio, sr = librosa.load(str(resolve(row["audio_path"])), sr=SAMPLE_RATE, mono=True, offset=start, duration=duration)
        clip = audio_dir / f"{row['segment_id']}.wav"
        sf.write(str(clip), audio, sr)
        out.append({"audio_path": rel(clip), "reference_text": row["corrected"], "speaker_id": row["speaker_id"]})
    write_csv(data_dir / GOLD_EVAL, out, EVAL_FIELDS)
    return len(out)
