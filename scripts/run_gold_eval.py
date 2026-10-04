#!/usr/bin/env python3
"""Score an ASR model on the human-verified gold test set (WSL; needs NeMo for step 2).

    python scripts/run_gold_eval.py --test-chapters JHN_001                    # baseline
    python scripts/run_gold_eval.py --name varhadi_adapted --checkpoint x.nemo # adapted

1. export verified segments of the test chapters -> data/evaluation/gold_eval.csv + clips
2. transcribe every clip with the model (raw output, no correction)
3. WER / CER / substitutions / deletions / insertions + per-sample + error pairs + report
   -> experiments/<name>/<name>_metrics.json, _per_sample.csv, _error_pairs.csv, _report.md

Only verified references are ever scored. With zero verified segments nothing is computed.
Results are labelled PILOT EVALUATION unless --final is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_baseline.config import MODEL_NAME
from asr_baseline.error_analysis import OP_FIELDS, edit_operations
from asr_baseline.evaluate import PER_SAMPLE_FIELDS, compute_metrics, per_sample_metrics
from asr_baseline.utils import write_csv
from config import ROOT_DIR
from varhadi_data.config import DATA_DIR, GOLD_EVAL

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
LIMITATIONS = [
    "Scripture speech (VAHNT New Testament), not agricultural speech: says nothing about "
    "crop/pest/scheme vocabulary.",
    "Speaker identity is unknown (VAHNT narrator(s)); test segments may share speakers "
    "with any VAHNT training data, so speaker-independent generalisation is NOT measured. "
    "Test chapters are held out at chapter level only.",
    "References are the published VAHNT text as confirmed or corrected by a listener; the "
    "verification depends on that reviewer's Varhadi knowledge.",
    "Segments come from forced alignment; reviewers rejected or flagged mis-aligned ones, "
    "which are excluded (see counts below).",
    "Each segment is decoded on its own (<= ~20 s), not inside the full chapter.",
]


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def score_and_report(
    eval_csv: Path,
    predictions_csv: Path,
    out_dir: Path,
    name: str,
    model: str,
    test_chapters: list[str],
    final: bool = False,
    review_counts: dict | None = None,
) -> dict:
    """Pure scoring/reporting step (no model). Returns the metrics dict."""
    eval_rows = {row["audio_path"]: row for row in _read(eval_csv)}
    # select by gold membership, not by the predictions file's own reference column
    scoreable = [row for row in _read(predictions_csv) if row["status"] == "ok" and row["audio_path"] in eval_rows]
    if not scoreable:
        raise RuntimeError("no verified references with successful predictions; nothing scored")
    for row in scoreable:  # the reference comes from the verified gold export, never from elsewhere
        row["reference"] = eval_rows[row["audio_path"]]["reference_text"]
        row["duration_sec"] = eval_rows[row["audio_path"]]["duration_sec"]

    label = "FINAL EVALUATION" if final else "PILOT EVALUATION"
    speakers = sorted({eval_rows[r["audio_path"]]["speaker_id"] for r in scoreable})
    metrics = {
        **compute_metrics(scoreable),
        "model": model,
        "label": label,
        "date": date.today().isoformat(),
        "test_chapters": test_chapters,
        "evaluated_minutes": round(sum(float(r["duration_sec"]) for r in scoreable) / 60, 2),
        "speakers": speakers,
        "normalisation": "punctuation/symbols stripped on both sides (asr_baseline.utils.strip_punctuation); no spelling or dialect normalisation",
        "references": "human-verified gold transcripts only (review_status=verified)",
        "review_counts": review_counts or {},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    per_sample = per_sample_metrics(scoreable, {r["audio_path"]: eval_rows[r["audio_path"]]["speaker_id"] for r in scoreable})
    for row in per_sample:
        row["segment_id"] = eval_rows[row["audio_path"]]["segment_id"]
    write_csv(out_dir / f"{name}_per_sample.csv", per_sample, ["segment_id", *PER_SAMPLE_FIELDS])
    ops = edit_operations(scoreable)
    write_csv(out_dir / f"{name}_error_pairs.csv", ops, OP_FIELDS)
    (out_dir / f"{name}_report.md").write_text(_report(name, metrics, per_sample, ops), encoding="utf-8")
    return metrics


def _report(name: str, m: dict, per_sample: list[dict], ops: list[dict]) -> str:
    subs = [o for o in ops if o["type"] == "substitution"][:25]
    similar = sum(o["count"] for o in ops if o["heuristic_note"].startswith("similar"))
    all_subs = sum(o["count"] for o in ops if o["type"] == "substitution")
    lines = [
        f"# {name}: {m['label']}",
        "",
        f"Date: {m['date']} · Model: `{m['model']}`",
        "",
    ]
    if m["label"].startswith("PILOT"):
        lines += ["> **PILOT EVALUATION.** Small, single-domain, speaker-unknown test set: use it to "
                  "validate the pipeline and spot error patterns, not as the project's final result.", ""]
    lines += [
        "## Metrics",
        "",
        "| WER | CER | Substitutions | Deletions | Insertions | Reference words |",
        "|---|---|---|---|---|---|",
        f"| {m['wer']:.4f} | {m['cer']:.4f} | {m['substitutions']} | {m['deletions']} | {m['insertions']} | {m['reference_words']} |",
        "",
        "## Test set",
        "",
        f"- Evaluated segments: {m['evaluated_samples']} ({m['evaluated_minutes']} min)",
        f"- Test chapters (held out): {', '.join(m['test_chapters'])}",
        f"- Speakers: {', '.join(m['speakers'])} (VAHNT narrator identity unknown)",
        "- Domain: general Varhadi scripture speech (VAHNT), NOT agricultural speech",
        f"- Review status counts in these chapters: {json.dumps(m['review_counts'])}",
        "",
        "## Verification procedure",
        "",
        "Candidate segments came from CTC forced alignment of the published VAHNT text. A human "
        "reviewer played each clip in full in `scripts/review_server.py` (the tool refuses to "
        "verify a clip that has not been played through) and marked it *correct* (published text "
        "matches the speech) or *needs correction* (typed what was actually said). Rejected and "
        "needs-realignment segments are excluded. Only `verified` segments are scored.",
        "",
        f"Scoring normalisation: {m['normalisation']}.",
        "",
        "## Error analysis",
        "",
        f"Substitutions tagged *similar form* by the string-similarity heuristic: {similar} of {all_subs}. "
        "This tag is a heuristic, not a diagnosis; `manual_category` in the error-pairs CSV is for human labelling.",
        "",
        "| Reference | Hypothesis | Count | Heuristic |",
        "|---|---|---|---|",
        *[f"| {o['reference_word']} | {o['hypothesis_word']} | {o['count']} | {o['heuristic_note']} |" for o in subs],
        "",
        f"Full tables: `{name}_per_sample.csv`, `{name}_error_pairs.csv`.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in LIMITATIONS],
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-chapters", nargs="+", default=["JHN_001"])
    parser.add_argument("--name", default="baseline", help="experiment name -> experiments/<name>/")
    parser.add_argument("--checkpoint", help="local .nemo to evaluate instead of the pretrained baseline")
    parser.add_argument("--final", action="store_true", help="label as FINAL (only for an adequate held-out set)")
    args = parser.parse_args()

    from asr_baseline.transcribe import run_batch
    from asr_baseline.model import load_model
    from varhadi_data import gold

    gold_rows = [r for r in gold.read_gold() if r["source_chapter"] in args.test_chapters]
    review_counts = {s: sum(r["review_status"] == s for r in gold_rows) for s in sorted(gold.STATUSES)}
    exported = gold.export_gold(chapters=args.test_chapters)
    if exported == 0:
        print(f"No verified segments in {args.test_chapters} (status counts: {review_counts}). "
              "WER/CER NOT computed. Review segments first: python scripts/review_server.py --chapter "
              f"{args.test_chapters[0]}")
        return 1

    out_dir = EXPERIMENTS_DIR / args.name
    predictions = out_dir / f"{args.name}_predictions.csv"
    loaded = load_model(args.checkpoint)
    run_batch(metadata_path=DATA_DIR / GOLD_EVAL, predictions_path=predictions, force=True,
              chunks_dir=out_dir / "chunks", loaded=loaded)
    metrics = score_and_report(DATA_DIR / GOLD_EVAL, predictions, out_dir, args.name,
                               args.checkpoint or MODEL_NAME, args.test_chapters, args.final, review_counts)
    print(json.dumps({k: metrics[k] for k in ("label", "evaluated_samples", "evaluated_minutes", "wer", "cer",
                                              "substitutions", "deletions", "insertions")}, indent=2))
    print(f"Report: {out_dir / (args.name + '_report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
