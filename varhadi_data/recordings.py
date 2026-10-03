"""Native Varhadi recording intake: consent gate, pseudonymous speaker IDs, quality checks.

Privacy:
- The speaker key (name/phone/anything that identifies the person) is never stored. Only
  an HMAC-SHA256 of it, keyed with SPEAKER_ID_SALT from the environment, is kept in the
  gitignored registry, so the same person maps to the same VH_S### on later sessions.
- Withdrawing consent deletes the speaker's audio, manifest and gold rows, and blanks the
  registry hash. The pseudonym is retired, never reused.
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import os
import shutil
from datetime import date
from pathlib import Path

from asr_baseline.audio_utils import load_audio, validate_audio
from asr_baseline.utils import append_csv_row, write_csv

from . import gold
from .config import DATA_DIR, GOLD_EVAL, GOLD_EVAL_AUDIO, PROCESSED_RECORDINGS, RAW_RECORDINGS, REGISTRY, rel
from .manifest import NATIVE_SOURCE, read_manifest, upsert, write_manifest

REGISTRY_FIELDS = ["speaker_id", "key_hash", "status", "registered_on"]
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 600.0
CLIP_LEVEL = 0.999
CLIP_RATIO_POOR = 0.001  # >0.1% clipped samples
QUIET_DBFS = -40.0
LOUD_FAIR_DBFS = -30.0


def _read_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def speaker_id_for(speaker_key: str, data_dir: Path = DATA_DIR, salt: str | None = None) -> str:
    salt = salt if salt is not None else os.environ.get("SPEAKER_ID_SALT", "")
    if not salt:
        raise RuntimeError("SPEAKER_ID_SALT is not set; refusing to derive speaker IDs without a secret")
    digest = hmac.new(salt.encode(), speaker_key.strip().lower().encode(), hashlib.sha256).hexdigest()
    path = data_dir / REGISTRY
    registry = _read_registry(path)
    for row in registry:
        if row["key_hash"] == digest and row["status"] == "active":
            return row["speaker_id"]
    next_number = max((int(row["speaker_id"][4:]) for row in registry), default=0) + 1
    speaker_id = f"VH_S{next_number:03d}"
    append_csv_row(
        path,
        {"speaker_id": speaker_id, "key_hash": digest, "status": "active", "registered_on": date.today().isoformat()},
        REGISTRY_FIELDS,
    )
    return speaker_id


def quality_check(audio, sr: int) -> tuple[str, str]:
    """Return (recording_quality, notes). Flags, never silently fixes, the audio."""
    import numpy as np

    duration = len(audio) / sr
    clip_ratio = float(np.mean(np.abs(audio) >= CLIP_LEVEL)) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    dbfs = 20 * np.log10(rms) if rms > 0 else -np.inf
    notes = f"duration={duration:.2f}s clip_ratio={clip_ratio:.4f} rms_dbfs={dbfs:.1f}"
    if clip_ratio > CLIP_RATIO_POOR or dbfs < QUIET_DBFS:
        return "poor", notes
    if dbfs < LOUD_FAIR_DBFS:
        return "fair", notes
    return "good", notes


def register_recording(
    source: Path,
    speaker_key: str,
    district: str,
    domain: str,
    consent: str,
    age_group: str = "",
    gender: str = "",
    prompt_text: str = "",
    data_dir: Path = DATA_DIR,
    salt: str | None = None,
) -> dict:
    """Validate, pseudonymise and store one recording. Returns the manifest row.

    age_group/gender are stored only when the speaker consented to share them (caller's
    responsibility to pass "" otherwise). `prompt_text` is what the speaker was asked to
    read; it goes to the gold workflow as pending_review because people do not read
    prompts verbatim, so it is never treated as a verified transcript.
    """
    import soundfile as sf

    if consent != "given":
        raise ValueError("recording rejected: informed consent not given")
    valid, reason = validate_audio(source)
    if not valid:
        raise ValueError(f"recording rejected: {reason}")
    audio, sr = load_audio(source)
    duration = len(audio) / sr
    if not MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC:
        raise ValueError(f"recording rejected: duration {duration:.2f}s outside [{MIN_DURATION_SEC}, {MAX_DURATION_SEC}]")

    speaker_id = speaker_id_for(speaker_key, data_dir, salt)
    rows = read_manifest(data_dir)
    taken = [int(r["utterance_id"].rsplit("_", 1)[1]) for r in rows if r["speaker_id"] == speaker_id]
    utterance_id = f"{speaker_id}_{max(taken, default=0) + 1:04d}"

    raw_path = data_dir / RAW_RECORDINGS / speaker_id / f"{utterance_id}{source.suffix.lower()}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, raw_path)
    wav_path = data_dir / PROCESSED_RECORDINGS / speaker_id / f"{utterance_id}.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), audio, sr)

    quality, _ = quality_check(audio, sr)
    row = {
        "utterance_id": utterance_id,
        "audio_path": rel(wav_path),
        "speaker_id": speaker_id,
        "district": district,
        "age_group": age_group,
        "gender": gender,
        "duration_sec": f"{duration:.3f}",
        "language": "mr",
        "dialect": "varhadi",
        "domain": domain,
        "transcript": "",
        "transcript_status": "pending_review" if prompt_text else "not_available",
        "recording_quality": quality,
        "consent_status": "given",
        "source": NATIVE_SOURCE,
        "license": "project consent form",
        "split": "unassigned",
    }
    write_manifest(upsert(rows, [row]), data_dir)
    if prompt_text:
        gold_rows = gold.read_gold(data_dir)
        gold_rows.append(
            gold.new_row(utterance_id, row["audio_path"], speaker_id, corrected=prompt_text, provenance="read_prompt")
        )
        gold.write_gold(gold_rows, data_dir)
    return row


def withdraw_speaker(speaker_id: str, data_dir: Path = DATA_DIR) -> int:
    """Delete everything stored for a speaker who withdrew consent. Returns rows removed."""
    rows = read_manifest(data_dir)
    keep = [row for row in rows if row["speaker_id"] != speaker_id]
    for directory in (RAW_RECORDINGS, PROCESSED_RECORDINGS):
        shutil.rmtree(data_dir / directory / speaker_id, ignore_errors=True)
    gold_rows = gold.read_gold(data_dir)
    for row in gold_rows:  # exported eval clips of this speaker
        if row["speaker_id"] == speaker_id:
            (data_dir / GOLD_EVAL_AUDIO / f"{row['segment_id']}.wav").unlink(missing_ok=True)
    gold.write_gold([row for row in gold_rows if row["speaker_id"] != speaker_id], data_dir)
    write_manifest(keep, data_dir)
    if (data_dir / GOLD_EVAL).exists():
        gold.export_gold(data_dir)  # rebuild so no eval row points at deleted audio

    registry = _read_registry(data_dir / REGISTRY)
    for row in registry:
        if row["speaker_id"] == speaker_id:
            row.update(key_hash="", status="withdrawn")
    write_csv(data_dir / REGISTRY, registry, REGISTRY_FIELDS)
    return len(rows) - len(keep)

