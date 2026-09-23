import wave
from pathlib import Path

from asr_baseline.audio_utils import ensure_wav_copy, get_duration, validate_audio


def make_silent_wav(path: Path, seconds: float = 0.5, sample_rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * int(seconds * sample_rate))


def test_get_duration_matches_written_length(tmp_path):
    path = tmp_path / "sample.wav"
    make_silent_wav(path, seconds=0.5)
    assert get_duration(path) == 0.5


def test_validate_audio_accepts_real_file(tmp_path):
    path = tmp_path / "sample.wav"
    make_silent_wav(path, seconds=0.2)
    valid, reason = validate_audio(path)
    assert valid
    assert "0.20" in reason


def test_validate_audio_rejects_missing_file(tmp_path):
    valid, reason = validate_audio(tmp_path / "missing.wav")
    assert not valid
    assert "does not exist" in reason


def test_ensure_wav_copy_caches_and_resamples(tmp_path):
    source = tmp_path / "source.wav"
    make_silent_wav(source, seconds=0.3, sample_rate=8000)
    cache_dir = tmp_path / "cache"

    wav_path = ensure_wav_copy(source, cache_dir)
    assert wav_path.exists()
    first_mtime = wav_path.stat().st_mtime_ns

    # Second call should reuse the cached copy, not regenerate it.
    ensure_wav_copy(source, cache_dir)
    assert wav_path.stat().st_mtime_ns == first_mtime
