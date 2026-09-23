from __future__ import annotations

from dataclasses import dataclass

from config import BASE_URL, TRANSLATION_CODE, VERSION_ID


BOOK_CHAPTERS = {
    "MAT": 28, "MRK": 16, "LUK": 24, "JHN": 21, "ACT": 28, "ROM": 16,
    "1CO": 16, "2CO": 13, "GAL": 6, "EPH": 6, "PHP": 4, "COL": 4,
    "1TH": 5, "2TH": 3, "1TI": 6, "2TI": 4, "TIT": 3, "PHM": 1,
    "HEB": 13, "JAS": 5, "1PE": 5, "2PE": 3, "1JN": 5, "2JN": 1,
    "3JN": 1, "JUD": 1, "REV": 22,
}
EXPECTED_CHAPTER_COUNT = 260


@dataclass(frozen=True)
class Chapter:
    book: str
    number: int

    @property
    def chapter_id(self) -> str:
        return f"{self.book}.{self.number}.{TRANSLATION_CODE}"

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{VERSION_ID}/{self.chapter_id}"

    @property
    def filename(self) -> str:
        return f"{self.book}_{self.number:03d}.mp3"


def all_chapters() -> list[Chapter]:
    chapters = [
        Chapter(book, number)
        for book, count in BOOK_CHAPTERS.items()
        for number in range(1, count + 1)
    ]
    if len(chapters) != EXPECTED_CHAPTER_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_CHAPTER_COUNT} chapters, got {len(chapters)}")
    return chapters


def chapter_from_id(chapter_id: str) -> Chapter:
    parts = chapter_id.split(".")
    if len(parts) != 3 or parts[2] != TRANSLATION_CODE:
        raise ValueError(f"Invalid chapter ID: {chapter_id}")
    book, number_text, _ = parts
    if book not in BOOK_CHAPTERS:
        raise ValueError(f"Unknown New Testament book: {book}")
    try:
        number = int(number_text)
    except ValueError as error:
        raise ValueError(f"Invalid chapter number: {chapter_id}") from error
    if number < 1 or number > BOOK_CHAPTERS[book]:
        raise ValueError(f"Chapter is outside the configured range: {chapter_id}")
    return Chapter(book, number)
