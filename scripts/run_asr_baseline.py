#!/usr/bin/env python3
"""Run the raw Marathi ASR baseline over Varhadi audio. No correction, no post-processing."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_baseline.audio_utils import ensure_wav_copy, get_duration, validate_audio
from asr_baseline.config import ASR_AUDIO_DIR, MODEL_NAME, PREDICTIONS_PATH
from asr_baseline.model import load_model, transcribe_one
from asr_baseline.transcribe import PREDICTIONS_FIELDS, run_batch
from asr_baseline.utils import append_csv_row


def run_single(audio_path: Path) -> int:
    valid, reason = validate_audio(audio_path)
    if not valid:
        print(f"Invalid audio: {reason}")
        return 1

    duration = get_duration(audio_path)
    loaded = load_model()
    wav_path = ensure_wav_copy(audio_path, ASR_AUDIO_DIR)
    start = time.monotonic()
    prediction = transcribe_one(loaded, str(wav_path))
    inference_time = time.monotonic() - start

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
    args = parser.parse_args()

    if args.audio:
        return run_single(args.audio)

    run_batch(force=args.force)
    print(f"Predictions written to {PREDICTIONS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
