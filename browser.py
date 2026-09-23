from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from playwright.sync_api import Page, Response, TimeoutError, sync_playwright

from config import MIN_AUDIO_BYTES, PAGE_TIMEOUT_MS, PLAYBACK_WAIT_SECONDS


DIRECT_AUDIO_TYPES = {"audio/mpeg", "audio/mp3", "audio/mpeg3"}
STREAM_TYPES = {
    "application/dash+xml",
    "application/vnd.apple.mpegurl",
    "application/x-mpegurl",
}


def media_kind(response: Response) -> tuple[str, str]:
    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    url = response.url.lower()
    if content_type in DIRECT_AUDIO_TYPES or urlparse(url).path.endswith(".mp3"):
        return "direct-mp3", content_type
    if content_type in STREAM_TYPES or url.endswith((".m3u8", ".mpd")):
        return "stream-manifest", content_type
    if content_type.startswith("audio/"):
        return "direct-audio", content_type
    return "other", content_type


def choose_resource(resources: list[dict[str, str]]) -> dict[str, str] | None:
    return next(
        (resource for resource in resources if resource["kind"] == "direct-mp3"), None
    ) or next(
        (resource for resource in resources if resource["kind"] == "direct-audio"), None
    )


def validate_mp3(path: Path, minimum_bytes: int = MIN_AUDIO_BYTES) -> tuple[bool, str]:
    if not path.is_file():
        return False, "file does not exist"
    try:
        size = path.stat().st_size
        if size <= minimum_bytes:
            return False, f"file is too small ({size} bytes)"
        with path.open("rb") as source:
            header = source.read(3)
        is_id3 = header == b"ID3"
        is_frame = len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0
        if is_id3 or is_frame:
            return True, f"valid MP3 header, {size} bytes"
        return False, "file does not have an MP3 header"
    except OSError as error:
        return False, str(error)


class BrowserAudioDownloader:
    """Download audio using the same page/network discovery as the proof of concept."""

    def __init__(self, headless: bool = False) -> None:
        self.headless = headless

    def download_chapter(self, chapter_url: str, output_path: Path) -> str:
        resources: list[dict[str, str]] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self.headless)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            page.on("response", lambda response: self._record_response(response, resources))
            try:
                logging.info("Opening %s", chapter_url)
                page.goto(chapter_url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
                try:
                    page.wait_for_load_state("networkidle", timeout=15_000)
                except TimeoutError:
                    logging.info("Page did not become network idle; continuing")
                self._inspect_audio_elements(page)
                self._start_playback(page)
                time.sleep(PLAYBACK_WAIT_SECONDS)
                self._inspect_audio_elements(page)
                resource = choose_resource(resources)
                if resource is None:
                    manifests = [item for item in resources if item["kind"] == "stream-manifest"]
                    raise RuntimeError(f"No direct MP3 resource detected; streams={manifests}")
                logging.info("Audio resource discovered: %s", resource["url"])
                self._download_resource(resource, page, output_path)
                return resource["url"]
            finally:
                browser.close()

    @staticmethod
    def _record_response(response: Response, resources: list[dict[str, str]]) -> None:
        kind, content_type = media_kind(response)
        if kind == "other":
            return
        resource = {
            "kind": kind,
            "url": response.url,
            "content_type": content_type or "unknown",
            "status": str(response.status),
        }
        resources.append(resource)
        logging.info(
            "Audio candidate: kind=%s status=%s content_type=%s url=%s",
            kind,
            response.status,
            resource["content_type"],
            response.url,
        )

    @staticmethod
    def _inspect_audio_elements(page: Page) -> None:
        for index, element in enumerate(page.locator("audio").all(), start=1):
            logging.info(
                "Audio element %d: src=%s current_src=%s paused=%s ready_state=%s",
                index,
                element.get_attribute("src"),
                element.evaluate("element => element.currentSrc"),
                element.evaluate("element => element.paused"),
                element.evaluate("element => element.readyState"),
            )

    @staticmethod
    def _start_playback(page: Page) -> None:
        if page.locator("audio").count():
            logging.info("Detected audio element; requesting playback")
            page.locator("audio").first.evaluate(
                "element => element.play().catch(error => console.debug(error))"
            )
            return
        for selector in (
            "button[aria-label*='Play' i]",
            "button[title*='Play' i]",
            "[role='button'][aria-label*='Play' i]",
        ):
            button = page.locator(selector).first
            if button.count():
                try:
                    button.click(timeout=3_000)
                    return
                except TimeoutError:
                    logging.debug("Playback control was not clickable: %s", selector)
        logging.warning("No audio element or recognizable play button was found")

    @staticmethod
    def _download_resource(resource: dict[str, str], page: Page, output_path: Path) -> None:
        if resource["kind"] != "direct-mp3":
            raise RuntimeError(f"Observed resource is not a direct MP3: {resource}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.part")
        cookies = page.context.cookies()
        cookie_header = "; ".join(f"{cookie['name']}={cookie['value']}" for cookie in cookies)
        headers = {"User-Agent": page.evaluate("navigator.userAgent")}
        if cookie_header:
            headers["Cookie"] = cookie_header
        logging.info("Download started: %s", resource["url"])
        try:
            with requests.get(resource["url"], headers=headers, stream=True, timeout=60) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                if content_type not in DIRECT_AUDIO_TYPES and not re.search(
                    r"\.mp3(?:$|[?#])", resource["url"], re.IGNORECASE
                ):
                    raise RuntimeError(f"Download response is not audio/mpeg: {content_type}")
                with temporary_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            output.write(chunk)
            valid, reason = validate_mp3(temporary_path)
            logging.info("Validation result: %s (%s)", valid, reason)
            if not valid:
                raise RuntimeError(reason)
            temporary_path.replace(output_path)
            logging.info("Download completed: %s bytes=%d", output_path, output_path.stat().st_size)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
