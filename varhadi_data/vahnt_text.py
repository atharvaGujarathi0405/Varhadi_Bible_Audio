"""Fetch the published VAHNT chapter text from bible.com (option b: text + forced alignment).

Uses a real Chromium page via Playwright, the same way the existing downloader does,
because the site serves a JavaScript challenge to plain HTTP clients. One page per call,
no parallelism.

Parsing keeps what a narrator reads: section headings (`s`) and verse `content`.
It drops verse number labels, cross-references (`r`) and footnotes (`note`/`fr`/`fq`/`body`).
The text is stored verbatim. Normalisation for alignment/evaluation happens later and is
applied to both sides.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from config import TRANSLATION_CODE, VERSION_ID

TEXT_URL = "https://www.bible.com/bible/{version}/{chapter_id}"
TEXT_LICENSE = "unverified: published VAHNT text, local research use only"
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "wbr", "source"}
SKIP = {"label", "r", "note", "fr", "fq", "body"}


def _suffixes(attrs: dict) -> set[str]:
    return {cls.rsplit("__", 1)[-1] for cls in (attrs.get("class") or "").split()}


class _ChapterParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, set[str], str | None]] = []
        self.units: list[dict] = []

    def handle_starttag(self, tag, attrs):
        if tag in VOID_TAGS:
            if tag == "br" and self.units:
                self.units[-1]["text"] += " "
            return
        attrs = dict(attrs)
        suffixes = _suffixes(attrs)
        self.stack.append((tag, suffixes, attrs.get("data-usfm")))
        if "content" in suffixes and self.units:  # a new text run never glues onto the previous word
            self.units[-1]["text"] += " "
        if "s" in suffixes and not self._inside(SKIP):
            self.units.append({"kind": "heading", "ref": "", "text": ""})

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):  # tolerate unclosed tags
            if self.stack[index][0] == tag:
                del self.stack[index:]
                return

    def _inside(self, names: set[str]) -> bool:
        return any(suffixes & names for _, suffixes, _ in self.stack)

    def handle_data(self, data):
        if not data.strip() or self._inside(SKIP):
            return
        if self._inside({"s"}):
            self.units[-1]["text"] += data
            return
        if not self._inside({"content"}):
            return
        ref = next((usfm for _, suffixes, usfm in reversed(self.stack) if "verse" in suffixes and usfm), None)
        if ref is None:
            return
        last = self.units[-1] if self.units else None
        if last and last["kind"] == "verse" and last["ref"] == ref:
            last["text"] += data
        else:
            self.units.append({"kind": "verse", "ref": ref, "text": data})


def parse_chapter_html(html: str) -> list[dict]:
    parser = _ChapterParser()
    parser.feed(html)
    units = [{**unit, "text": re.sub(r"\s+", " ", unit["text"]).strip()} for unit in parser.units]
    return [unit for unit in units if unit["text"]]


def fetch_chapter_text(chapter_id: str, out_dir: Path, headless: bool = True) -> Path:
    """Render one chapter page, parse it, and save JSON with provenance. Returns the path."""
    from playwright.sync_api import sync_playwright

    from chapters import chapter_from_id

    chapter = chapter_from_id(chapter_id)
    url = TEXT_URL.format(version=VERSION_ID, chapter_id=chapter.chapter_id)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            # any verse of this chapter; not an exact "X.N.1" match, which misses a bridged first verse ("X.N.1+X.N.2")
            page.wait_for_selector(f"[data-usfm^='{chapter.book}.{chapter.number}.']", timeout=30_000)
            html = page.content()
        finally:
            browser.close()

    units = parse_chapter_html(html)
    if not any(unit["kind"] == "verse" for unit in units):
        raise RuntimeError(f"no verse text parsed from {url}")
    out = out_dir / f"{Path(chapter.filename).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "chapter_id": chapter.chapter_id,
                "translation": TRANSLATION_CODE,
                "url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "license": TEXT_LICENSE,
                "units": units,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return out
