"""Gold (manually verified) transcript workflow.

    import/seed -> pending_review --human review--> verified | rejected | needs_realignment
                                                       |
                                                       +--export--> evaluation set

- The original audio is never modified; a segment is (audio_path, start_sec, end_sec).
- `candidate_text` is the text proposed BEFORE review (published VAHNT text for aligned
  segments, the read prompt for native recordings, empty for ASR-seeded chunks). It is
  never modified. `raw_asr` keeps the untouched baseline output.
- A reviewer listens and records a decision:
      correct           -> verified, reference = candidate_text
      needs_correction  -> verified, reference = corrected_text (what they actually heard)
      rejected          -> rejected (unusable: wrong audio, noise, bad boundaries beyond fixing)
      needs_realignment -> needs_realignment (text is fine but the time span is wrong)
- Only `verified` rows are exported, so WER/CER is only ever computed against
  human-verified references. The review UI (scripts/review_server.py) only accepts a
  verifying decision after the clip has actually been played through.
"""
from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

from asr_baseline.config import MODEL_NAME, SAMPLE_RATE
from asr_baseline.utils import write_csv

from .config import DATA_DIR, GOLD, GOLD_EVAL, GOLD_EVAL_AUDIO, rel, resolve

FIELDS = [
    "segment_id",
    "audio_path",
    "speaker_id",
    "source_chapter",
    "start_sec",
    "end_sec",
    "source_ref",
    "source_url",
    "align_score",
    "candidate_text",
    "raw_asr",
    "decision",
    "corrected_text",
    "reviewer",
    "review_status",
    "reviewed_at",
    "provenance",
    # human notes (optional)
    "review_notes",
    # automatic triage (scripts/triage_gold_candidates.py) - ordering/pseudo-label eligibility only, NEVER gold
    "alignment_score",
    "agreement_score",
    "audio_quality_score",
    "duration_score",
    "speaking_rate_score",
    "artifact_score",
    "confidence_score",
    "triage_status",
    "triage_flags",
    "triage_version",
]
TRIAGE_FIELDS = FIELDS[FIELDS.index("alignment_score"):]
STATUS_FOR_DECISION = {
    "correct": "verified",
    "needs_correction": "verified",
    "rejected": "rejected",
    "needs_realignment": "needs_realignment",
}
STATUSES = {"pending_review", *STATUS_FOR_DECISION.values()}
# prompt03 names for the same decisions (stored values stay the canonical lower-case ones)
DECISION_ALIASES = {"VERIFIED": "correct", "CORRECTED": "needs_correction", "REJECTED": "rejected",
                    "NEEDS_REALIGNMENT": "needs_realignment"}
REVIEWER_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{1,19}$")  # initials/pseudonym, never a name with spaces or a phone number
EVAL_FIELDS = ["audio_path", "reference_text", "speaker_id", "segment_id", "source_chapter", "duration_sec"]


def new_row(segment_id, audio_path, speaker_id, start="", end="", candidate="", raw_asr="", provenance="", **extra):
    row = dict.fromkeys(FIELDS, "")
    row.update(
        segment_id=segment_id,
        audio_path=audio_path,
        speaker_id=speaker_id,
        start_sec=start,
        end_sec=end,
        candidate_text=candidate,
        raw_asr=raw_asr,
        review_status="pending_review",
        provenance=provenance,
        **extra,
    )
    return row


def reference(row: dict) -> str:
    """The human-verified reference text of a verified row."""
    return row["corrected_text"] if row["decision"] == "needs_correction" else row["candidate_text"]


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
        status, decision = row["review_status"], row["decision"]
        if status not in STATUSES:
            errors.append(f"{sid}: review_status {status!r} not in {sorted(STATUSES)}")
        if decision and STATUS_FOR_DECISION.get(decision) != status:
            errors.append(f"{sid}: decision {decision!r} does not match status {status!r}")
        if status != "pending_review" and not (decision and row["reviewer"].strip()):
            errors.append(f"{sid}: reviewed rows need a decision and a reviewer")
        if decision == "needs_correction" and not row["corrected_text"].strip():
            errors.append(f"{sid}: needs_correction requires corrected_text")
        if decision == "correct" and not row["candidate_text"].strip():
            errors.append(f"{sid}: 'correct' needs a candidate_text; type the transcript (needs_correction)")
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
    """Add one pending segment per successful ASR chunk (asr_outputs/chunks/<stem>.csv),
    with NO candidate text: a reviewer must type what they hear. Existing segment_ids are
    left untouched, so reseeding never overwrites a review."""
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
                    source_chapter=chunk_csv.stem,
                )
            )
    return rows + added


def import_alignments(rows: list[dict], alignments_dir: Path, text_dir: Path) -> list[dict]:
    """Queue aligned VAHNT segments (scripts/vahnt_align.py) for human review, with the
    PUBLISHED text as candidate_text. Existing segment_ids are left untouched, so
    re-importing never overwrites a review."""
    import json

    existing = {row["segment_id"] for row in rows}
    added = []
    for alignment_csv in sorted(alignments_dir.glob("*.csv")):
        source = json.loads((text_dir / f"{alignment_csv.stem}.json").read_text(encoding="utf-8"))
        with alignment_csv.open(encoding="utf-8", newline="") as handle:
            for seg in csv.DictReader(handle):
                if seg["segment_id"] in existing:
                    continue
                added.append(
                    new_row(
                        seg["segment_id"],
                        seg["audio_path"],
                        "unknown",
                        seg["start_sec"],
                        seg["end_sec"],
                        candidate=seg["text"],
                        raw_asr=seg["raw_asr"],
                        provenance=f"vahnt_text_align:{MODEL_NAME}:ctc_viterbi",
                        source_chapter=alignment_csv.stem,
                        source_ref=seg["units"],
                        source_url=source["url"],
                        align_score=seg["align_score"],
                    )
                )
    return rows + added


def review(rows: list[dict], segment_id: str, decision: str, reviewer: str, corrected_text: str = "",
           notes: str = "") -> list[dict]:
    """Record a human decision. candidate_text is never touched. Re-submitting replaces the
    previous decision on the same row (one row per segment, no duplicates)."""
    decision = DECISION_ALIASES.get(decision, decision)
    if decision not in STATUS_FOR_DECISION:
        raise ValueError(f"decision must be one of {sorted(STATUS_FOR_DECISION)} (or {sorted(DECISION_ALIASES)})")
    if not REVIEWER_ID_RE.match(reviewer.strip()):
        raise ValueError("reviewer id must be 2-20 letters/digits/_/- starting with a letter (initials or pseudonym)")
    matches = [row for row in rows if row["segment_id"] == segment_id]
    if not matches:
        raise KeyError(f"unknown segment_id {segment_id}")
    row = matches[0]
    updated = {
        **row,
        "decision": decision,
        "corrected_text": corrected_text.strip() if decision == "needs_correction" else "",
        "reviewer": reviewer.strip(),
        "review_status": STATUS_FOR_DECISION[decision],
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "review_notes": " ".join(notes.split()),
    }
    errors = validate_gold([updated])
    if errors:
        raise ValueError("; ".join(errors))
    row.update(updated)
    return rows


def export_gold(data_dir: Path = DATA_DIR, chapters: list[str] | None = None) -> int:
    """Write verified segments (optionally only from `chapters`) as 16 kHz WAV clips plus a
    CSV in the asr_metadata format that scripts/run_asr_baseline.py --metadata consumes.
    Returns the number exported."""
    import librosa
    import soundfile as sf

    verified = [
        row
        for row in read_gold(data_dir)
        if row["review_status"] == "verified" and (chapters is None or row["source_chapter"] in chapters)
    ]
    audio_dir = data_dir / GOLD_EVAL_AUDIO
    audio_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for row in verified:
        start = float(row["start_sec"]) if row["start_sec"] else 0.0
        duration = float(row["end_sec"]) - start if row["end_sec"] else None
        audio, sr = librosa.load(str(resolve(row["audio_path"])), sr=SAMPLE_RATE, mono=True, offset=start, duration=duration)
        clip = audio_dir / f"{row['segment_id']}.wav"
        sf.write(str(clip), audio, sr)
        out.append(
            {
                "audio_path": rel(clip),
                "reference_text": reference(row),
                "speaker_id": row["speaker_id"],
                "segment_id": row["segment_id"],
                "source_chapter": row["source_chapter"],
                "duration_sec": f"{len(audio) / sr:.3f}",
            }
        )
    write_csv(data_dir / GOLD_EVAL, out, EVAL_FIELDS)
    return len(out)
