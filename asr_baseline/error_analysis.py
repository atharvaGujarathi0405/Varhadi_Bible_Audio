from __future__ import annotations

import csv
from pathlib import Path

from .config import ERROR_ANALYSIS_PATH, PER_SAMPLE_METRICS_PATH
from .utils import write_csv

FIELDS = ["audio_path", "reference", "prediction", "wer", "cer", "error_summary", "manual_category", "notes"]

HIGH_WER_THRESHOLD = 0.5

# Filled in later by a human reviewing error_summary + reference/prediction pairs.
# Never auto-assigned: an error must not be labeled "Varhadi vocabulary" etc. without review.
MANUAL_CATEGORIES = [
    "pronunciation",
    "vocabulary",
    "code_switching",
    "agricultural_term",
    "proper_noun",
    "noise",
    "segmentation",
    "transcript_issue",
    "unknown",
]


def summarize_errors(reference: str, prediction: str) -> str:
    import jiwer

    output = jiwer.process_words(reference, prediction)
    return (
        f"substitutions={output.substitutions} "
        f"deletions={output.deletions} "
        f"insertions={output.insertions} "
        f"hits={output.hits}"
    )


def build_error_analysis(per_sample_rows: list[dict]) -> list[dict]:
    rows = []
    for row in per_sample_rows:
        wer = float(row["wer"])
        summary = summarize_errors(row["reference"], row["prediction"])
        if wer >= HIGH_WER_THRESHOLD:
            summary += " [HIGH_WER]"
        rows.append(
            {
                "audio_path": row["audio_path"],
                "reference": row["reference"],
                "prediction": row["prediction"],
                "wer": row["wer"],
                "cer": row["cer"],
                "error_summary": summary,
                "manual_category": "",
                "notes": "",
            }
        )
    return rows


def run(per_sample_path: Path = PER_SAMPLE_METRICS_PATH, out_path: Path = ERROR_ANALYSIS_PATH) -> None:
    with per_sample_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    write_csv(out_path, build_error_analysis(rows), FIELDS)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
