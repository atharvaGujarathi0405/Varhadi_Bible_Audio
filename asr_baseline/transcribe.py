from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from .audio_utils import ensure_wav_copy, get_duration, validate_audio
from .chunking import ChunkSettings, transcribe_chunked
from .config import ASR_AUDIO_DIR, ASR_METADATA_PATH, AUDIO_DIR, CHUNK_WORK_DIR, CHUNKS_DIR, PREDICTIONS_PATH
from .model import load_model, transcribe_one
from .utils import append_csv_row, read_asr_metadata, repo_path

PREDICTIONS_FIELDS = [
    "audio_path",
    "reference",
    "prediction",
    "duration_sec",
    "inference_time_sec",
    "status",
    "error_message",
]


def discover_samples(
    metadata_path: Path = ASR_METADATA_PATH, audio_dir: Path = AUDIO_DIR
) -> list[dict]:
    """asr_metadata.csv rows when it has data; otherwise every MP3 under audio/,
    with an empty reference (transcription-only)."""
    if metadata_path.exists():
        rows = read_asr_metadata(metadata_path)
        if rows:
            return rows
    return [
        {"audio_path": str(path), "reference_text": "", "speaker_id": ""}
        for path in sorted(audio_dir.glob("*.mp3"))
    ]


def already_done(predictions_path: Path) -> set[str]:
    """ASR's own resume state, kept in predictions.csv. Independent of the downloader's progress.json."""
    if not predictions_path.exists():
        return set()
    with predictions_path.open(encoding="utf-8", newline="") as source:
        return {
            repo_path(row["audio_path"])  # normalise legacy absolute/relative spellings
            for row in csv.DictReader(source)
            if row.get("status") == "ok"
        }


def transcribe_sample(
    loaded,
    audio_path: Path,
    cache_dir: Path,
    settings: ChunkSettings | None = None,
    chunks_dir: Path = CHUNKS_DIR,
) -> tuple[str, float, float]:
    """Audio up to `settings.max_sec` goes through transcribe_one unchanged; longer audio is
    chunked (see chunking.py) because a full chapter OOM-kills the encoder on CPU."""
    settings = settings or ChunkSettings.from_env()
    valid, reason = validate_audio(audio_path)
    if not valid:
        raise RuntimeError(reason)
    duration = get_duration(audio_path)
    wav_path = ensure_wav_copy(audio_path, cache_dir)
    start = time.monotonic()
    if duration <= settings.max_sec:
        prediction = transcribe_one(loaded, str(wav_path))
    else:
        prediction = transcribe_chunked(
            loaded, wav_path, chunks_dir / f"{audio_path.stem}.csv", cache_dir / CHUNK_WORK_DIR.name, settings
        )
    inference_time = time.monotonic() - start
    return prediction, duration, inference_time


def run_batch(
    metadata_path: Path = ASR_METADATA_PATH,
    audio_dir: Path = AUDIO_DIR,
    predictions_path: Path = PREDICTIONS_PATH,
    cache_dir: Path = ASR_AUDIO_DIR,
    force: bool = False,
    settings: ChunkSettings | None = None,
    chunks_dir: Path = CHUNKS_DIR,
    loaded=None,
) -> None:
    samples = discover_samples(metadata_path, audio_dir)
    if force and predictions_path.exists():
        predictions_path.unlink()
    if force:
        for chunk_csv in chunks_dir.glob("*.csv"):
            chunk_csv.unlink()
    done = set() if force else already_done(predictions_path)

    loaded = loaded or load_model()
    for sample in samples:
        audio_path = Path(sample["audio_path"])
        if repo_path(audio_path) in done:
            logging.info("Skipping already-transcribed %s", audio_path)
            continue
        row = {
            "audio_path": repo_path(audio_path),
            "reference": sample.get("reference_text", ""),
            "prediction": "",
            "duration_sec": "",
            "inference_time_sec": "",
            "status": "ok",
            "error_message": "",
        }
        try:
            prediction, duration, inference_time = transcribe_sample(loaded, audio_path, cache_dir, settings, chunks_dir)
            row.update(
                prediction=prediction,
                duration_sec=f"{duration:.3f}",
                inference_time_sec=f"{inference_time:.3f}",
            )
        except Exception as error:
            row.update(status="failed", error_message=str(error))
            logging.exception("Failed to transcribe %s", audio_path)
        append_csv_row(predictions_path, row, PREDICTIONS_FIELDS)
