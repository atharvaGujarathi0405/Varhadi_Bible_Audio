import csv

import numpy as np
import pytest
import soundfile as sf

from asr_baseline.chunking import ChunkSettings, plan_chunks, stitch, transcribe_chunked
from asr_baseline.transcribe import transcribe_sample

SR = 16000


def tone_with_gaps(seconds: float, gaps: list[float]) -> np.ndarray:
    """A loud tone with 0.4 s silences centred on each time in `gaps`."""
    t = np.arange(int(seconds * SR)) / SR
    audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    for gap in gaps:
        audio[int((gap - 0.2) * SR) : int((gap + 0.2) * SR)] = 0
    return audio


def write_wav(path, audio):
    sf.write(str(path), audio, SR)
    return path


def test_settings_reject_invalid_values():
    with pytest.raises(ValueError):
        ChunkSettings(target_sec=40, max_sec=30)
    with pytest.raises(ValueError):
        ChunkSettings(overlap_sec=20)


def test_settings_from_env_and_cli_override(monkeypatch):
    monkeypatch.setenv("ASR_CHUNK_SEC", "10")
    monkeypatch.setenv("ASR_CHUNK_MAX_SEC", "15")
    settings = ChunkSettings.from_env(overlap_sec=2.0, max_sec=None)
    assert (settings.target_sec, settings.max_sec, settings.overlap_sec) == (10, 15, 2.0)


def test_plan_chunks_cuts_in_silence_and_covers_audio():
    audio = tone_with_gaps(70, gaps=[25, 52])
    ranges = plan_chunks(audio, SR, ChunkSettings(20, 30, 0))
    assert ranges[0][0] == 0 and ranges[-1][1] == len(audio)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))  # contiguous, no gaps/overlap
    for _, end in ranges[:-1]:
        assert audio[end] == 0  # cut landed inside a silence
    assert all(end - start <= 30 * SR for start, end in ranges)


def test_plan_chunks_with_overlap_and_no_silence():
    audio = tone_with_gaps(70, gaps=[])
    ranges = plan_chunks(audio, SR, ChunkSettings(20, 30, 2))
    assert all(b[0] == a[1] - 2 * SR for a, b in zip(ranges, ranges[1:]))
    assert ranges[-1][1] == len(audio)


def test_short_audio_is_a_single_chunk():
    audio = tone_with_gaps(12, gaps=[])
    assert plan_chunks(audio, SR, ChunkSettings()) == [(0, len(audio))]


def test_stitch_removes_overlap_duplicates_only_when_overlapping():
    texts = ["अ ब क ड", "क ड इ ई", "इ ई उ"]
    assert stitch(texts, overlapping=True) == "अ ब क ड इ ई उ"
    assert stitch(texts, overlapping=False) == "अ ब क ड क ड इ ई इ ई उ"
    assert stitch(["अ ब", "क ड"], overlapping=True) == "अ ब क ड"  # no match: joined unchanged


def test_stitch_handles_differently_heard_edge_words():
    # Real seam from JHN_001 with 2 s overlap: edge words differ, shared run is "पहिले शब्द होता".
    prev = "ह्या सगळ्या जगाच्या बनच्या पहिले शब्द होता"
    nxt = "ह वन्याच्या पहिले शब्द होता अन् हा शब्द"
    assert stitch([prev, nxt], overlapping=True) == "ह्या सगळ्या जगाच्या बनच्या पहिले शब्द होता अन् हा शब्द"
    # a single shared word is not trusted as an overlap
    assert stitch(["अ होता", "होता ब"], overlapping=True) == "अ होता होता ब"


def test_transcribe_chunked_orders_and_records_metadata(tmp_path):
    wav = write_wav(tmp_path / "ch.wav", tone_with_gaps(70, gaps=[25, 52]))
    chunk_csv = tmp_path / "chunks" / "ch.csv"
    seen = []

    def fake(_, path):
        seen.append(path)
        return f"w{len(seen)}"

    text = transcribe_chunked(None, wav, chunk_csv, tmp_path / "work", ChunkSettings(20, 30, 0), fake)
    assert text == "w1 w2 w3"
    with chunk_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["chunk_index"] for r in rows] == ["0", "1", "2"]
    assert rows[0]["start_sec"] == "0.000" and float(rows[-1]["end_sec"]) == 70.0
    assert not any((tmp_path / "work").iterdir())  # temp chunk WAVs cleaned up


def test_transcribe_chunked_resumes_and_retries_only_failed_chunks(tmp_path):
    wav = write_wav(tmp_path / "ch.wav", tone_with_gaps(70, gaps=[25, 52]))
    chunk_csv = tmp_path / "ch.csv"
    calls = []

    def flaky(_, path):
        calls.append(path)
        if path.endswith("c001.wav") and len(calls) <= 3:
            raise MemoryError("simulated OOM")
        return path[-8:-4]

    settings = ChunkSettings(20, 30, 0)
    with pytest.raises(RuntimeError, match="1/3 chunks failed"):
        transcribe_chunked(None, wav, chunk_csv, tmp_path / "w", settings, flaky)
    assert len(calls) == 3

    text = transcribe_chunked(None, wav, chunk_csv, tmp_path / "w", settings, flaky)
    assert len(calls) == 4 and calls[-1].endswith("c001.wav")  # only the failed chunk re-ran
    assert text == "c000 c001 c002"


def test_transcribe_chunked_rejects_wrong_sample_rate(tmp_path):
    wav = tmp_path / "8k.wav"
    sf.write(str(wav), np.zeros(8000 * 40, dtype="float32"), 8000)
    with pytest.raises(RuntimeError, match="16000 Hz mono"):
        transcribe_chunked(None, wav, tmp_path / "c.csv", tmp_path / "w", ChunkSettings(), lambda *_: "x")


def test_transcribe_sample_rejects_malformed_audio(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(RuntimeError, match="could not decode"):
        transcribe_sample(None, bad, tmp_path / "cache", ChunkSettings())


def test_transcribe_sample_routes_long_audio_to_chunking(tmp_path, monkeypatch):
    import asr_baseline.transcribe as transcribe

    src = write_wav(tmp_path / "long.wav", tone_with_gaps(45, gaps=[25]))
    monkeypatch.setattr(transcribe, "transcribe_one", lambda *_: pytest.fail("long audio must be chunked"))
    monkeypatch.setattr("asr_baseline.model.transcribe_one", lambda _, p: "x")
    text, duration, _ = transcribe_sample(None, src, tmp_path / "cache", ChunkSettings(20, 30, 0), tmp_path / "chunks")
    assert text == "x x" and duration == 45
    assert (tmp_path / "chunks" / "long.csv").exists()
