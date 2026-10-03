#!/usr/bin/env python3
"""Run the raw Marathi ASR baseline over Varhadi audio. No correction, no post-processing."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_baseline.audio_utils import validate_audio
from asr_baseline.chunking import ChunkSettings
from asr_baseline.config import ASR_AUDIO_DIR, ASR_METADATA_PATH, MODEL_NAME, PREDICTIONS_PATH
from config import LOG_DIR
from asr_baseline.model import load_model
from asr_baseline.transcribe import PREDICTIONS_FIELDS, run_batch, transcribe_sample
from asr_baseline.utils import append_csv_row


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "asr_baseline.log", encoding="utf-8"), logging.StreamHandler()],
    )


def run_single(audio_path: Path, settings: ChunkSettings) -> int:
    valid, reason = validate_audio(audio_path)
    if not valid:
        print(f"Invalid audio: {reason}")
        return 1

    loaded = load_model()
    prediction, duration, inference_time = transcribe_sample(loaded, audio_path, ASR_AUDIO_DIR, settings)

    append_csv_row(
        PREDICTIONS_PATH,
        {
            "audio_path": str(audio_path),
            "reference": "",
            "prediction": prediction,
            "duration_sec": f"{duration:.3f}",
            "inference_time_sec": f"{inference_time:.3f}",
            "status": "ok",
            "error_message": "",
        },
        PREDICTIONS_FIELDS,
    )

    print("=" * 40)
    print("KISAANDOST ASR BASELINE")
    print("=" * 40)
    print(f"Model:    {MODEL_NAME} ({loaded.decoder})")
    print(f"Device:   {loaded.device}")
    print(f"Audio:    {audio_path}")
    print(f"Duration: {duration:.2f}s")
    if duration > settings.max_sec:
        print(f"Chunked:  target {settings.target_sec}s, max {settings.max_sec}s, overlap {settings.overlap_sec}s")
    print()
    print("RAW ASR OUTPUT:")
    print(prediction)
    print()
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Saved to: {PREDICTIONS_PATH}")
    print("=" * 40)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, help="Transcribe a single audio file")
    parser.add_argument("--force", action="store_true", help="Re-run samples already recorded in predictions.csv")
    parser.add_argument("--chunk-sec", type=float, help="Target chunk length (env ASR_CHUNK_SEC, default 20)")
    parser.add_argument("--max-chunk-sec", type=float, help="Max chunk length; longer audio is chunked (env ASR_CHUNK_MAX_SEC, default 30)")
    parser.add_argument("--overlap-sec", type=float, help="Chunk overlap (env ASR_CHUNK_OVERLAP_SEC, default 0)")
    parser.add_argument("--metadata", type=Path, default=ASR_METADATA_PATH, help="audio_path,reference_text,speaker_id CSV (e.g. data/evaluation/gold_eval.csv)")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH, help="Output/resume CSV for batch mode")
    args = parser.parse_args()
    configure_logging()
    settings = ChunkSettings.from_env(
        target_sec=args.chunk_sec, max_sec=args.max_chunk_sec, overlap_sec=args.overlap_sec
    )

    if args.audio:
        return run_single(args.audio, settings)

    run_batch(metadata_path=args.metadata, predictions_path=args.predictions, force=args.force, settings=settings)
    print(f"Predictions written to {args.predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
