#!/usr/bin/env python3
"""Inspect audio/ and asr_metadata.csv before running ASR. Read-only, no model needed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_baseline.audio_utils import get_duration, validate_audio
from asr_baseline.config import ASR_METADATA_PATH
from asr_baseline.transcribe import discover_samples
from asr_baseline.utils import read_asr_metadata


def main() -> int:
    samples = discover_samples()
    total_duration = 0.0
    valid_count = 0
    with_reference = 0
    for sample in samples:
        path = Path(sample["audio_path"])
        valid, reason = validate_audio(path)
        if valid:
            valid_count += 1
            total_duration += get_duration(path)
        else:
            print(f"INVALID {path}: {reason}")
        if sample.get("reference_text", "").strip():
            with_reference += 1

    print("=" * 40)
    print("ASR DATASET VALIDATION")
    print("=" * 40)
    using_metadata = ASR_METADATA_PATH.exists() and bool(read_asr_metadata(ASR_METADATA_PATH))
    source = "asr_metadata.csv" if using_metadata else "audio/ directory scan (no reference rows)"
    print(f"Metadata source : {source}")
    print(f"Total samples   : {len(samples)}")
    print(f"Valid audio     : {valid_count}")
    print(f"Invalid audio   : {len(samples) - valid_count}")
    print(f"With reference  : {with_reference} (eligible for WER/CER)")
    print(f"Without ref     : {len(samples) - with_reference} (transcription-only)")
    print(f"Total duration  : {total_duration / 60:.1f} minutes")
    print("=" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
