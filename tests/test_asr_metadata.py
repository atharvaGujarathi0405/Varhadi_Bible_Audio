from asr_baseline.transcribe import discover_samples
from asr_baseline.utils import read_asr_metadata


def test_read_asr_metadata_defaults_missing_columns(tmp_path):
    path = tmp_path / "asr_metadata.csv"
    path.write_text("audio_path,reference_text,speaker_id\naudio/a.mp3,,\n", encoding="utf-8")
    rows = read_asr_metadata(path)
    assert rows == [{"audio_path": "audio/a.mp3", "reference_text": "", "speaker_id": ""}]


def test_discover_samples_uses_metadata_rows_when_present(tmp_path):
    metadata_path = tmp_path / "asr_metadata.csv"
    metadata_path.write_text(
        "audio_path,reference_text,speaker_id\naudio/a.mp3,hello,S01\n", encoding="utf-8"
    )
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "b.mp3").write_bytes(b"ID3" + b"0" * 20_000)

    samples = discover_samples(metadata_path=metadata_path, audio_dir=audio_dir)
    assert samples == [{"audio_path": "audio/a.mp3", "reference_text": "hello", "speaker_id": "S01"}]


def test_discover_samples_falls_back_to_directory_scan_when_metadata_empty(tmp_path):
    metadata_path = tmp_path / "asr_metadata.csv"
    metadata_path.write_text("audio_path,reference_text,speaker_id\n", encoding="utf-8")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "a.mp3").write_bytes(b"ID3" + b"0" * 20_000)
    (audio_dir / "b.mp3").write_bytes(b"ID3" + b"0" * 20_000)

    samples = discover_samples(metadata_path=metadata_path, audio_dir=audio_dir)
    assert len(samples) == 2
    assert all(sample["reference_text"] == "" for sample in samples)


def test_already_done_treats_absolute_and_relative_spellings_as_one_file(tmp_path):
    from asr_baseline.transcribe import already_done
    from asr_baseline.utils import repo_path
    from config import ROOT_DIR

    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "audio_path,reference,prediction,duration_sec,inference_time_sec,status,error_message\n"
        f"{ROOT_DIR / 'audio' / 'JHN_001.mp3'},,x,1,1,ok,\n",
        encoding="utf-8",
    )
    assert repo_path("audio/JHN_001.mp3") in already_done(predictions)
    assert repo_path(ROOT_DIR / "audio" / "JHN_001.mp3") == "audio/JHN_001.mp3"
