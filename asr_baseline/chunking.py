"""Chunked inference for long audio (P0).

A full ~8-minute chapter OOM-kills the Conformer encoder on CPU (full self-attention over
one long utterance). We split the cached 16 kHz mono WAV into short chunks, transcribe each
with the unchanged `transcribe_one`, and stitch the raw outputs back in order.

Stitching strategy (documented in docs/asr_baseline.md):
  1. Cut points are chosen at the lowest-energy 25 ms frame inside [target, max] seconds
     from the chunk start, so cuts usually fall in pauses between words.
  2. With overlap = 0 (default) chunks are disjoint and joined with a single space; no
     word can be duplicated because no audio is transcribed twice.
  3. With overlap > 0 each chunk starts `overlap` seconds before the previous cut, so the
     overlap audio is transcribed twice. Words at a chunk edge are heard with truncated
     context and often differ between the two chunks (observed on JHN_001: "बनच्या" vs
     "ह वन्याच्या"), so an exact suffix/prefix match is too brittle. Instead we take the
     longest common run of >= MIN_MATCH_WORDS words (difflib) between the last
     MAX_OVERLAP_WORDS words of the stitched text and the first MAX_OVERLAP_WORDS words of
     the next chunk, keep the previous text up to the end of that run, and continue with
     the next chunk after it. The edge words on both sides of the run are discarded in
     favour of the shared, better-supported run. If no such run exists, the texts are
     joined unchanged and the seam is logged, rather than guessing which words to drop.
No text is corrected or normalized: the stitched output is still raw ASR output.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .config import SAMPLE_RATE
from .utils import append_csv_row

CHUNK_FIELDS = ["chunk_index", "start_sec", "end_sec", "prediction", "status", "error_message"]
FRAME_SEC = 0.025
MAX_OVERLAP_WORDS = 30
MIN_MATCH_WORDS = 2  # a single shared word (e.g. "होता") is too likely to be coincidental


@dataclass(frozen=True)
class ChunkSettings:
    target_sec: float = 20.0
    max_sec: float = 30.0
    overlap_sec: float = 0.0

    def __post_init__(self):
        if not 0 < self.target_sec <= self.max_sec:
            raise ValueError("need 0 < target_sec <= max_sec")
        if not 0 <= self.overlap_sec < self.target_sec:
            raise ValueError("need 0 <= overlap_sec < target_sec")

    @classmethod
    def from_env(cls, **overrides) -> "ChunkSettings":
        values = {
            "target_sec": float(os.environ.get("ASR_CHUNK_SEC", cls.target_sec)),
            "max_sec": float(os.environ.get("ASR_CHUNK_MAX_SEC", cls.max_sec)),
            "overlap_sec": float(os.environ.get("ASR_CHUNK_OVERLAP_SEC", cls.overlap_sec)),
        }
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


def plan_chunks(audio, sr: int, settings: ChunkSettings) -> list[tuple[int, int]]:
    """Return ordered (start_sample, end_sample) ranges covering the whole signal."""
    import numpy as np

    total = len(audio)
    target, max_len = int(settings.target_sec * sr), int(settings.max_sec * sr)
    overlap, frame = int(settings.overlap_sec * sr), max(1, int(FRAME_SEC * sr))
    chunks, start = [], 0
    while start < total:
        if total - start <= max_len:
            chunks.append((start, total))
            break
        window = audio[start + target : start + max_len]
        usable = len(window) // frame * frame
        if usable:
            energy = np.sqrt(np.mean(window[:usable].reshape(-1, frame) ** 2, axis=1))
            cut = start + target + int(np.argmin(energy)) * frame + frame // 2
        else:
            cut = start + max_len
        chunks.append((start, cut))
        start = max(cut - overlap, start + 1)
    return chunks


def stitch(texts: list[str], overlapping: bool) -> str:
    words: list[str] = []
    for index, text in enumerate(texts):
        new = text.split()
        if overlapping and words and new:
            tail_start = max(0, len(words) - MAX_OVERLAP_WORDS)
            tail, head = words[tail_start:], new[:MAX_OVERLAP_WORDS]
            match = SequenceMatcher(None, tail, head, autojunk=False).find_longest_match(0, len(tail), 0, len(head))
            if match.size >= MIN_MATCH_WORDS:
                del words[tail_start + match.a + match.size :]
                new = new[match.b + match.size :]
            else:
                logging.info("No overlap match at seam %d; joined unchanged", index)
        words.extend(new)
    return " ".join(words)


def _completed_chunks(chunk_csv: Path) -> dict[int, dict]:
    """Latest row per chunk index wins, so a retried chunk replaces its earlier failure."""
    if not chunk_csv.exists():
        return {}
    with chunk_csv.open(encoding="utf-8", newline="") as source:
        latest = {int(row["chunk_index"]): row for row in csv.DictReader(source)}
    return {index: row for index, row in latest.items() if row["status"] == "ok"}


def transcribe_chunked(
    loaded,
    wav_path: Path,
    chunk_csv: Path,
    work_dir: Path,
    settings: ChunkSettings,
    transcribe_fn=None,
) -> str:
    """Transcribe a 16 kHz mono WAV chunk by chunk with per-chunk resume.

    Every chunk's raw output is appended to `chunk_csv` (chunk metadata + status). Chunks
    already recorded as ok are skipped on rerun; failed chunks are retried. Raises if any
    chunk still fails, so the file is marked failed and retried on the next run.
    """
    import soundfile as sf

    if transcribe_fn is None:
        from .model import transcribe_one as transcribe_fn

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if sr != SAMPLE_RATE or audio.ndim != 1:
        raise RuntimeError(f"expected {SAMPLE_RATE} Hz mono WAV, got {sr} Hz, {audio.ndim}D")

    ranges = plan_chunks(audio, sr, settings)
    done = _completed_chunks(chunk_csv)
    work_dir.mkdir(parents=True, exist_ok=True)
    texts, failures = [], 0
    for index, (start, end) in enumerate(ranges):
        if index in done:
            texts.append(done[index]["prediction"])
            continue
        row = {
            "chunk_index": index,
            "start_sec": f"{start / sr:.3f}",
            "end_sec": f"{end / sr:.3f}",
            "prediction": "",
            "status": "ok",
            "error_message": "",
        }
        chunk_path = work_dir / f"{wav_path.stem}_c{index:03d}.wav"
        try:
            sf.write(str(chunk_path), audio[start:end], sr)
            row["prediction"] = transcribe_fn(loaded, str(chunk_path))
            logging.info("%s chunk %d/%d done", wav_path.stem, index + 1, len(ranges))
        except Exception as error:
            row.update(status="failed", error_message=str(error))
            failures += 1
            logging.exception("%s chunk %d failed", wav_path.stem, index)
        finally:
            chunk_path.unlink(missing_ok=True)
        append_csv_row(chunk_csv, row, CHUNK_FIELDS)
        texts.append(row["prediction"])

    if failures:
        raise RuntimeError(f"{failures}/{len(ranges)} chunks failed; rerun to retry them")
    return stitch(texts, overlapping=settings.overlap_sec > 0)
