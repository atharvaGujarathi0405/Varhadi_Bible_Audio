"""Run the original MAT.1 single-chapter proof of concept."""

from __future__ import annotations

import argparse
import logging
from logging import FileHandler, StreamHandler

from browser import BrowserAudioDownloader, validate_mp3
from chapters import Chapter
from config import AUDIO_DIR, LOG_DIR, LOG_PATH


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[StreamHandler(), FileHandler(LOG_PATH, encoding="utf-8")],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    configure_logging()
    output = AUDIO_DIR / "MAT_001.mp3"
    try:
        resource_url = BrowserAudioDownloader(args.headless).download_chapter(
            Chapter("MAT", 1).url, output
        )
    except Exception as error:
        logging.exception("Audio download failed: %s", error)
        return 1
    valid, reason = validate_mp3(output)
    print(f"Discovered audio resource: {resource_url}")
    print(f"Saved: {output} ({output.stat().st_size} bytes)")
    print(f"Validation: {'successful' if valid else 'failed'} ({reason})")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())