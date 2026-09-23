from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

from browser import BrowserAudioDownloader, validate_mp3
from chapters import BOOK_CHAPTERS, Chapter, all_chapters, chapter_from_id
from config import (
    AUDIO_DIR,
    DEFAULT_DELAY_SECONDS,
    FAILED_PATH,
    LOG_DIR,
    LOG_PATH,
    MAX_ATTEMPTS,
    METADATA_PATH,
    PROGRESS_PATH,
    RETRY_DELAY_SECONDS,
)
from progress import (
    load_completed,
    load_failed,
    load_metadata,
    save_completed,
    save_failed,
    save_metadata,
)


@dataclass
class Summary:
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH, encoding="utf-8")],
        force=True,
    )


def remove_failure(failed: list[dict], chapter_id: str) -> None:
    failed[:] = [entry for entry in failed if entry.get("chapter") != chapter_id]


def record_metadata(metadata: list[dict], chapter: Chapter) -> None:
    entry = {
        "chapter_id": chapter.chapter_id,
        "book": chapter.book,
        "chapter": chapter.number,
        "url": chapter.url,
        "audio_file": f"audio/{chapter.filename}",
    }
    metadata[:] = [item for item in metadata if item.get("chapter_id") != chapter.chapter_id]
    metadata.append(entry)


def process_chapter(
    chapter: Chapter,
    browser: BrowserAudioDownloader,
    completed: set[str],
    failed: list[dict],
    metadata: list[dict],
    delay_seconds: float,
) -> str:
    output_path = AUDIO_DIR / chapter.filename
    chapter_id = chapter.chapter_id
    logging.info("Processing %s", chapter_id)
    if chapter_id in completed or output_path.exists():
        valid, reason = validate_mp3(output_path)
        logging.info("Validation result for existing file: %s (%s)", valid, reason)
        if valid:
            completed.add(chapter_id)
            record_metadata(metadata, chapter)
            save_completed(PROGRESS_PATH, completed)
            save_metadata(METADATA_PATH, metadata)
            remove_failure(failed, chapter_id)
            save_failed(FAILED_PATH, failed)
            logging.info("Skipped %s", chapter_id)
            return "skipped"
        output_path.unlink(missing_ok=True)
        logging.warning("Invalid existing file removed for %s: %s", chapter_id, reason)

    last_reason = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        logging.info("%s attempt %d/%d URL=%s", chapter_id, attempt, MAX_ATTEMPTS, chapter.url)
        try:
            browser.download_chapter(chapter.url, output_path)
            valid, reason = validate_mp3(output_path)
            logging.info("Validation result for %s: %s (%s)", chapter_id, valid, reason)
            if not valid:
                raise RuntimeError(reason)
            completed.add(chapter_id)
            remove_failure(failed, chapter_id)
            record_metadata(metadata, chapter)
            save_completed(PROGRESS_PATH, completed)
            save_failed(FAILED_PATH, failed)
            save_metadata(METADATA_PATH, metadata)
            logging.info("Downloaded %s", chapter.filename)
            return "downloaded"
        except Exception as error:
            last_reason = str(error)
            output_path.unlink(missing_ok=True)
            logging.exception("Error processing %s on attempt %d: %s", chapter_id, attempt, error)
            if attempt < MAX_ATTEMPTS:
                logging.info("Retrying %s after %.1f seconds", chapter_id, RETRY_DELAY_SECONDS)
                time.sleep(RETRY_DELAY_SECONDS)

    failed.append({"chapter": chapter_id, "reason": last_reason, "attempts": MAX_ATTEMPTS})
    remove_failure_duplicates(failed)
    save_failed(FAILED_PATH, failed)
    logging.error("Failed %s after %d attempts", chapter_id, MAX_ATTEMPTS)
    if delay_seconds:
        time.sleep(delay_seconds)
    return "failed"


def remove_failure_duplicates(failed: list[dict]) -> None:
    unique: dict[str, dict] = {}
    for entry in failed:
        unique[entry.get("chapter", "")] = entry
    failed[:] = [entry for entry in unique.values() if entry.get("chapter")]


def print_status() -> None:
    chapters = all_chapters()
    valid_count = sum(
        validate_mp3(AUDIO_DIR / chapter.filename)[0]
        for chapter in chapters
    )
    failed = load_failed(FAILED_PATH)
    print(f"Completed and valid: {valid_count}/{len(chapters)}")
    print(f"Remaining: {len(chapters) - valid_count}")
    print(f"Failed records: {len(failed)}")


def print_summary(summary: Summary, failed: list[dict]) -> None:
    print("\n" + "=" * 40)
    print("VAHNT AUDIO DOWNLOAD COMPLETE")
    print(f"Total chapters : {summary.total}")
    print(f"Downloaded     : {summary.downloaded}")
    print(f"Skipped        : {summary.skipped}")
    print(f"Failed         : {summary.failed}")
    print(f"Remaining      : {summary.total - summary.downloaded - summary.skipped}")
    if failed:
        print("\nFailed chapters:")
        for entry in failed:
            print(f"- {entry['chapter']}")
    print("=" * 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all VAHNT New Testament chapter audio")
    parser.add_argument("--book", choices=list(BOOK_CHAPTERS))
    parser.add_argument("--chapter", help="Download exactly one chapter, e.g. MAT.1.VAHNT")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    args = parser.parse_args()
    selected = sum(bool(value) for value in (args.book, args.chapter, args.retry_failed, args.status))
    if selected > 1:
        parser.error("--book, --chapter, --retry-failed, and --status are mutually exclusive")
    if args.delay < 0:
        parser.error("--delay must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    configure_logging()
    if args.status:
        print_status()
        return 0

    completed = load_completed(PROGRESS_PATH)
    failed = load_failed(FAILED_PATH)
    metadata = load_metadata(METADATA_PATH)
    if args.chapter:
        chapters = [chapter_from_id(args.chapter)]
    elif args.book:
        chapters = [chapter for chapter in all_chapters() if chapter.book == args.book]
    elif args.retry_failed:
        chapters = [chapter_from_id(entry["chapter"]) for entry in failed]
    else:
        chapters = all_chapters()

    summary = Summary(total=len(chapters))
    browser = BrowserAudioDownloader(headless=args.headless)
    for index, chapter in enumerate(chapters, start=1):
        print(f"[{index:03d}/{summary.total:03d}] {chapter.chapter_id} ", end="", flush=True)
        result = process_chapter(
            chapter, browser, completed, failed, metadata, args.delay
        )
        summary.__dict__[result] += 1
        symbols = {"downloaded": "✓ Downloaded", "skipped": "✓ Skipped", "failed": "✗ Failed"}
        print(symbols[result])
        if result != "failed" and index < summary.total and args.delay:
            time.sleep(args.delay)

    print_summary(summary, failed)
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())