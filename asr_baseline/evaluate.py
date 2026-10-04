from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import ASR_METADATA_PATH, METRICS_PATH, MODEL_NAME, PER_SAMPLE_METRICS_PATH, PREDICTIONS_PATH
from .utils import strip_punctuation, write_csv

PER_SAMPLE_FIELDS = [
    "audio_path", "reference", "prediction", "wer", "cer",
    "substitutions", "deletions", "insertions", "reference_words", "duration_sec", "speaker_id",
]
EMPTY = {"wer": None, "cer": None, "substitutions": None, "deletions": None, "insertions": None, "reference_words": 0}


def load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def evaluable_rows(rows: list[dict]) -> list[dict]:
    """Only rows with a non-empty ground-truth reference and a successful prediction.
    WER/CER is never computed against missing or unaligned references."""
    return [row for row in rows if row.get("reference", "").strip() and row.get("status") == "ok"]


def _scores(references: list[str], predictions: list[str]) -> dict:
    """Corpus-level WER/CER and edit counts. Both sides get the same punctuation stripping
    (strip_punctuation) and nothing else."""
    import jiwer

    references = [strip_punctuation(text) for text in references]
    predictions = [strip_punctuation(text) for text in predictions]
    words = jiwer.process_words(references, predictions)
    return {
        "wer": words.wer,
        "cer": jiwer.process_characters(references, predictions).cer,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "reference_words": words.hits + words.substitutions + words.deletions,
    }


def compute_metrics(rows: list[dict]) -> dict:
    base = {"model": MODEL_NAME, "dataset_size": len(rows), "evaluated_samples": len(rows)}
    if not rows:
        return {**base, **EMPTY}
    return {**base, **_scores([row["reference"] for row in rows], [row["prediction"] for row in rows])}


def _speaker_lookup(metadata_path: Path) -> dict[str, str]:
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8", newline="") as source:
        return {row["audio_path"]: row.get("speaker_id", "") for row in csv.DictReader(source)}


def per_sample_metrics(rows: list[dict], speaker_by_path: dict[str, str]) -> list[dict]:
    """Raw reference and prediction are kept as-is; scores use the normalised text."""
    return [
        {
            "audio_path": row["audio_path"],
            "reference": row["reference"],
            "prediction": row["prediction"],
            **_scores([row["reference"]], [row["prediction"]]),
            "duration_sec": row.get("duration_sec", ""),
            "speaker_id": speaker_by_path.get(row["audio_path"], ""),
        }
        for row in rows
    ]


def run_evaluation(
    predictions_path: Path = PREDICTIONS_PATH,
    metrics_path: Path = METRICS_PATH,
    per_sample_path: Path = PER_SAMPLE_METRICS_PATH,
    metadata_path: Path = ASR_METADATA_PATH,
) -> dict:
    rows = load_predictions(predictions_path)
    scoreable = evaluable_rows(rows)
    metrics = compute_metrics(scoreable)
    metrics["total_predictions"] = len(rows)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(per_sample_path, per_sample_metrics(scoreable, _speaker_lookup(metadata_path)), PER_SAMPLE_FIELDS)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute WER/CER for predictions that have ground truth")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--metadata", type=Path, default=ASR_METADATA_PATH, help="for speaker_id lookup")
    args = parser.parse_args()
    metrics = run_evaluation(predictions_path=args.predictions, metadata_path=args.metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
