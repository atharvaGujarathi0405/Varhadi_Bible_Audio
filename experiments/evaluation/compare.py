#!/usr/bin/env python3
"""Experiment A (pretrained baseline) vs Experiment B (Varhadi-adapted) on the SAME test set.

    python experiments/evaluation/compare.py [--a baseline] [--b varhadi_adapted]

Reads experiments/<name>/<name>_metrics.json and _per_sample.csv written by
scripts/run_gold_eval.py and writes experiments/evaluation/comparison.{csv,json,md}.
Refuses to compare unless both runs scored exactly the same segment IDs. Reports measured
differences only; it never labels a model "better".
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from asr_baseline.utils import write_csv  # noqa: E402

METRICS = ["wer", "cer", "substitutions", "deletions", "insertions", "reference_words", "evaluated_samples", "evaluated_minutes"]


def load(experiments: Path, name: str) -> tuple[dict, set[str]]:
    metrics = json.loads((experiments / name / f"{name}_metrics.json").read_text(encoding="utf-8"))
    with (experiments / name / f"{name}_per_sample.csv").open(encoding="utf-8", newline="") as source:
        segments = {row["segment_id"] for row in csv.DictReader(source)}
    return metrics, segments


def compare(experiments: Path, a: str, b: str, out_dir: Path) -> dict:
    (ma, sa), (mb, sb) = load(experiments, a), load(experiments, b)
    if sa != sb:
        raise ValueError(f"different test sets: {len(sa ^ sb)} segment IDs differ; re-run both on the same export")
    rows = []
    for key in METRICS:
        va, vb = ma[key], mb[key]
        rows.append({"metric": key, a: va, b: vb, "difference_b_minus_a": round(vb - va, 6)})
    result = {
        "experiment_a": {"name": a, "model": ma["model"], "label": ma["label"]},
        "experiment_b": {"name": b, "model": mb["model"], "label": mb["label"]},
        "test_chapters": ma["test_chapters"],
        "segments": len(sa),
        "metrics": rows,
        "note": "Measured differences on one shared test set; lower WER/CER = fewer errors. "
                "No significance test has been run.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "comparison.csv", rows, ["metric", a, b, "difference_b_minus_a"])
    label = "PILOT" if "PILOT" in (ma["label"] + mb["label"]) else "FINAL"
    md = [
        f"# {label} comparison: {a} vs {b}",
        "",
        f"Same test set: {len(sa)} verified segments from {', '.join(ma['test_chapters'])}.",
        f"- A `{a}`: `{ma['model']}`",
        f"- B `{b}`: `{mb['model']}`",
        "",
        f"| Metric | {a} | {b} | B − A |",
        "|---|---|---|---|",
        *[f"| {r['metric']} | {r[a]} | {r[b]} | {r['difference_b_minus_a']} |" for r in rows],
        "",
        result["note"],
        "",
    ]
    (out_dir / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", default="baseline")
    parser.add_argument("--b", default="varhadi_adapted")
    args = parser.parse_args()
    result = compare(ROOT / "experiments", args.a, args.b, ROOT / "experiments" / "evaluation")
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
