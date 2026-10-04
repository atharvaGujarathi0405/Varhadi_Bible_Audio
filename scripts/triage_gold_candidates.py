#!/usr/bin/env python3
"""Automatic triage of aligned candidate segments (review ORDER + pseudo-label eligibility).

    python scripts/triage_gold_candidates.py --chapter JHN_001 --report
    python scripts/triage_gold_candidates.py --all --export-csv --report

Writes ONLY the triage columns of data/transcripts/gold_transcripts.csv (confidence and
component scores, triage_status, triage_flags, triage_version). It never changes review
fields, candidate text or audio, never deletes candidates and never marks anything
verified. See varhadi_data/triage.py for the formula and docs/gold_review.md for the policy.

--export-csv  data/evaluation/triage_report.csv  (every signal, per segment)
--report      data/evaluation/triage_summary.json (counts, distributions, config used)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_baseline.config import ASR_AUDIO_DIR
from asr_baseline.utils import write_csv
from varhadi_data import gold
from varhadi_data.config import ALIGNMENTS, DATA_DIR
from varhadi_data.triage import CONFIG_PATH, STATUSES, audio_signals, load_config, score_segment

REPORT_CSV = Path("evaluation/triage_report.csv")
SUMMARY_JSON = Path("evaluation/triage_summary.json")
COMPONENTS = ["alignment_score", "agreement_score", "audio_quality_score", "duration_score", "speaking_rate_score"]


def heading_counts(alignments_dir: Path) -> dict[str, int]:
    counts = {}
    for path in alignments_dir.glob("*.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            counts.update({row["segment_id"]: int(row.get("n_headings") or 0) for row in csv.DictReader(handle)})
    return counts


def segment_audio(row: dict, audio_dir: Path):
    """The segment's samples from the cached 16 kHz chapter WAV (read-only), or None."""
    import soundfile as sf

    wav = audio_dir / f"{row['source_chapter']}.wav"
    if not wav.exists() or not row.get("start_sec"):
        return None
    with sf.SoundFile(str(wav)) as handle:
        sr = handle.samplerate
        start = int(float(row["start_sec"]) * sr)
        handle.seek(start)
        audio = handle.read(int(float(row["end_sec"]) * sr) - start, dtype="float32")
    return audio_signals(audio, sr)


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    q = statistics.quantiles(values, n=10) if len(values) > 1 else [values[0]] * 9
    return {"min": round(min(values), 4), "p10": round(q[0], 4), "median": round(statistics.median(values), 4),
            "p90": round(q[8], 4), "max": round(max(values), 4)}


def triage(rows: list[dict], chapters: set[str] | None, config: dict, data_dir: Path, audio_dir: Path) -> list[dict]:
    """Score selected rows in place (triage columns only); return the full report rows."""
    headings = heading_counts(data_dir / ALIGNMENTS)
    report = []
    for row in rows:
        if chapters is not None and row["source_chapter"] not in chapters:
            continue
        result = score_segment(row, segment_audio(row, audio_dir), headings.get(row["segment_id"], 0), config)
        row.update({
            "alignment_score": result["alignment_score"],
            "agreement_score": result["agreement_score"],
            "audio_quality_score": result["audio_quality_score"],
            "duration_score": result["duration_score"],
            "speaking_rate_score": result["speaking_rate_score"],
            "artifact_score": result["artifact_penalty"],
            "confidence_score": result["confidence_score"],
            "triage_status": result["triage_status"],
            "triage_flags": result["triage_flags"],
            "triage_version": config["version"],
        })
        report.append({"segment_id": row["segment_id"], "chapter_id": row["source_chapter"],
                       "start_sec": row["start_sec"], "end_sec": row["end_sec"], "review_status": row["review_status"],
                       **result, "candidate_transcript": row["candidate_text"], "baseline_asr": row["raw_asr"]})
    return report


def summarise(report: list[dict], config: dict) -> dict:
    by_chapter = {}
    for row in report:
        by_chapter.setdefault(row["chapter_id"], Counter())[row["triage_status"]] += 1
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_version": config["version"],
        "config": config,
        "total_candidates": len(report),
        "counts": {status: sum(r["triage_status"] == status for r in report) for status in STATUSES},
        "human_review_status_counts": dict(Counter(r["review_status"] for r in report)),
        "flag_counts": dict(Counter(f for r in report for f in r["triage_flags"].split(";") if f)),
        "distributions": {key: quantiles([float(r[key]) for r in report])
                          for key in ["confidence_score", *COMPONENTS, "duration_sec", "chars_per_sec", "cer_vs_asr"]},
        "by_chapter": {chapter: dict(counts) for chapter, counts in sorted(by_chapter.items())},
        "note": "Triage is NOT verification. HIGH_CONFIDENCE_CANDIDATE segments are unverified.",
    }


def main(argv=None, data_dir: Path = DATA_DIR, audio_dir: Path = ASR_AUDIO_DIR) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--chapter", nargs="+", help="chapter ids, e.g. JHN_001")
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--threshold", type=float, help="override thresholds.high_confidence for this run")
    parser.add_argument("--review-threshold", type=float, help="override thresholds.review for this run")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.threshold is not None:
        config["thresholds"]["high_confidence"] = args.threshold
    if args.review_threshold is not None:
        config["thresholds"]["review"] = args.review_threshold
    if not 0 <= config["thresholds"]["review"] <= config["thresholds"]["high_confidence"] <= 1:
        parser.error("need 0 <= review threshold <= high-confidence threshold <= 1")

    rows = gold.read_gold(data_dir)
    before = {r["segment_id"]: {k: r[k] for k in r if k not in gold.TRIAGE_FIELDS} for r in rows}
    report = triage(rows, None if args.all else set(args.chapter), config, data_dir, audio_dir)
    if not report:
        print("No candidates matched.")
        return 1
    after = {r["segment_id"]: {k: r[k] for k in r if k not in gold.TRIAGE_FIELDS} for r in rows}
    assert before == after, "triage must only change triage columns"
    gold.write_gold(rows, data_dir)

    summary = summarise(report, config)
    if args.export_csv:
        write_csv(data_dir / REPORT_CSV, report, list(report[0]))
    if args.report:
        (data_dir / SUMMARY_JSON).parent.mkdir(parents=True, exist_ok=True)
        (data_dir / SUMMARY_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = summary["counts"]
    print(f"Total candidates: {summary['total_candidates']}")
    print(f"High-confidence candidates: {counts['HIGH_CONFIDENCE_CANDIDATE']}  (UNVERIFIED)")
    print(f"Needs review: {counts['NEEDS_REVIEW']}")
    print(f"Rejected: {counts['REJECT_CANDIDATE']}")
    print(f"Human-verified (from the review workflow, not triage): "
          f"{summary['human_review_status_counts'].get('verified', 0)}")
    print(f"Confidence: {summary['distributions']['confidence_score']}")
    print(f"Flags: {summary['flag_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
