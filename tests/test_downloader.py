import json

from browser import choose_resource, media_kind, validate_mp3
from chapters import EXPECTED_CHAPTER_COUNT, all_chapters, chapter_from_id
from progress import load_completed, load_failed, save_completed, save_failed


class FakeResponse:
    def __init__(self, content_type: str, url: str):
        self.headers = {"content-type": content_type}
        self.url = url


def test_total_chapter_count_and_order():
    chapters = all_chapters()
    assert len(chapters) == EXPECTED_CHAPTER_COUNT == 260
    assert chapters[0].chapter_id == "MAT.1.VAHNT"
    assert chapters[-1].chapter_id == "REV.22.VAHNT"


def test_url_and_filename_generation():
    chapter = chapter_from_id("MAT.2.VAHNT")
    assert chapter.url == "https://www.bible.com/audio-bible/3451/MAT.2.VAHNT"
    assert chapter.filename == "MAT_002.mp3"


def test_audio_resource_classification_without_network():
    response = FakeResponse("audio/mpeg; charset=binary", "https://cdn.example/audio.mp3")
    assert media_kind(response) == ("direct-mp3", "audio/mpeg")
    resource = {"kind": "direct-mp3", "url": response.url}
    assert choose_resource([resource]) == resource


def test_progress_and_failed_state_round_trip(tmp_path):
    progress_path = tmp_path / "progress.json"
    failed_path = tmp_path / "failed.json"
    completed = {"MAT.1.VAHNT", "MAT.2.VAHNT"}
    failures = [{"chapter": "MAT.3.VAHNT", "reason": "timeout", "attempts": 3}]
    save_completed(progress_path, completed)
    save_failed(failed_path, failures)
    assert load_completed(progress_path) == completed
    assert load_failed(failed_path) == failures
    assert json.loads(progress_path.read_text())["completed"]


def test_failed_records_can_be_deduplicated_by_chapter(tmp_path):
    failed_path = tmp_path / "failed.json"
    save_failed(
        failed_path,
        [
            {"chapter": "MAT.3.VAHNT", "reason": "first", "attempts": 3},
            {"chapter": "MAT.3.VAHNT", "reason": "latest", "attempts": 3},
        ],
    )
    failures = load_failed(failed_path)
    latest = {entry["chapter"]: entry for entry in failures}
    assert latest["MAT.3.VAHNT"]["reason"] == "latest"


def test_mp3_validation(tmp_path):
    path = tmp_path / "sample.mp3"
    path.write_bytes(b"ID3" + b"0" * 20_000)
    assert validate_mp3(path)[0]
    path.write_bytes(b"not audio")
    assert not validate_mp3(path)[0]