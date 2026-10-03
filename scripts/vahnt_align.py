#!/usr/bin/env python3
"""Option b: published VAHNT text + CTC forced alignment -> candidate segment pairs.

  fetch-text  (Windows .venv, needs Playwright)  -> data/transcripts/vahnt_text/<BOOK>_<NNN>.json
  align       (WSL .venv-asr-wsl, needs NeMo)     -> data/transcripts/alignments/<BOOK>_<NNN>.csv

Aligned segments are candidates, not verified ground truth. See docs/dataset.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from varhadi_data.config import ALIGNMENTS, DATA_DIR, VAHNT_TEXT, rel

SEGMENT_FIELDS = ["segment_id", "audio_path", "start_sec", "end_sec", "units", "n_headings",
                  "text", "align_score", "raw_asr", "cer_vs_raw_asr"]


def cmd_fetch_text(args):
    from varhadi_data.vahnt_text import fetch_chapter_text

    for index, chapter_id in enumerate(args.chapter):
        if index:
            time.sleep(args.delay)  # polite, sequential
        out = fetch_chapter_text(chapter_id, DATA_DIR / VAHNT_TEXT, headless=not args.headed)
        units = json.loads(out.read_text(encoding="utf-8"))["units"]
        print(f"{chapter_id}: {sum(u['kind'] == 'verse' for u in units)} verses, "
              f"{sum(u['kind'] == 'heading' for u in units)} headings -> {rel(out)}")


def cmd_align(args):
    from asr_baseline.audio_utils import ensure_wav_copy
    from asr_baseline.chunking import ChunkSettings
    from asr_baseline.config import ASR_AUDIO_DIR, AUDIO_DIR
    from asr_baseline.model import load_model
    from asr_baseline.utils import write_csv
    from varhadi_data.align import align_chapter

    loaded = load_model()
    for stem in args.chapter:
        text = json.loads((DATA_DIR / VAHNT_TEXT / f"{stem}.json").read_text(encoding="utf-8"))
        audio = AUDIO_DIR / f"{stem}.mp3"
        wav = ensure_wav_copy(audio, ASR_AUDIO_DIR)
        start = time.monotonic()
        segments = align_chapter(loaded, wav, text["units"], ChunkSettings.from_env(), max_sec=args.max_sec)
        rows = [{"segment_id": f"{stem}_a{i:03d}", "audio_path": rel(audio), **seg} for i, seg in enumerate(segments)]
        out = DATA_DIR / ALIGNMENTS / f"{stem}.csv"
        write_csv(out, rows, SEGMENT_FIELDS)
        print(f"{stem}: {len(rows)} segments in {time.monotonic() - start:.1f}s -> {rel(out)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-text", help="fetch published chapter text, e.g. JHN.1.VAHNT")
    fetch.add_argument("chapter", nargs="+")
    fetch.add_argument("--delay", type=float, default=2.0)
    fetch.add_argument("--headed", action="store_true")
    fetch.set_defaults(fn=cmd_fetch_text)
    align = sub.add_parser("align", help="align fetched text to audio, e.g. JHN_001")
    align.add_argument("chapter", nargs="+")
    align.add_argument("--max-sec", type=float, default=20.0, help="max segment length")
    align.set_defaults(fn=cmd_align)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
