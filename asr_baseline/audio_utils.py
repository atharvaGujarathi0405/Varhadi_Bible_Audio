from __future__ import annotations

from pathlib import Path

from .config import SAMPLE_RATE


def get_duration(path: Path) -> float:
    import librosa

    return float(librosa.get_duration(path=str(path)))


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE):
    """Load audio resampled to `sample_rate`, mono, float32. In-memory; does not touch `path`."""
    import librosa

    audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio, sr


def resample_audio(audio, orig_sr: int, target_sr: int = SAMPLE_RATE):
    if orig_sr == target_sr:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def convert_to_mono(audio):
    import librosa

    return librosa.to_mono(audio) if audio.ndim > 1 else audio


def validate_audio(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "file does not exist"
    try:
        duration = get_duration(path)
    except Exception as error:
        return False, f"could not decode audio: {error}"
    if duration <= 0:
        return False, "zero-length audio"
    return True, f"valid audio, {duration:.2f}s"


def ensure_wav_copy(source_path: Path, cache_dir: Path) -> Path:
    """Return a cached 16kHz mono WAV copy of `source_path` under `cache_dir`.

    Converts once and reuses the copy on later runs. The original file is never modified.
    """
    import soundfile as sf

    wav_path = cache_dir / f"{source_path.stem}.wav"
    if not wav_path.exists():
        audio, sr = load_audio(source_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_path), audio, sr)
    return wav_path
