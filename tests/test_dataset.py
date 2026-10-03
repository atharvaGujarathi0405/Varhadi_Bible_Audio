import csv

import numpy as np
import pytest
import soundfile as sf

from varhadi_data import gold
from varhadi_data.config import GOLD_EVAL, PROCESSED_RECORDINGS, RAW_RECORDINGS, REGISTRY
from varhadi_data.manifest import assign_splits, read_manifest, speaker_leaks, vahnt_rows, validate_row, write_splits
from varhadi_data.recordings import quality_check, register_recording, speaker_id_for, withdraw_speaker

SR = 16000
SALT = "test-salt"


def speech_like(path, seconds=2.0, amplitude=0.3, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * 200 * t)).astype("float32"), sr)
    return path


def register(tmp_path, key="9876543210", name="a.wav", **kw):
    args = dict(district="Amravati", domain="agriculture", consent="given", data_dir=tmp_path / "data", salt=SALT)
    args.update(kw)
    return register_recording(speech_like(tmp_path / name), key, **args)


def test_vahnt_rows_never_fabricate_speaker_or_transcript(tmp_path):
    (tmp_path / "JHN_001.mp3").write_bytes(b"")
    [row] = vahnt_rows(tmp_path, lambda _: 473.5)
    assert row["speaker_id"] == "unknown" and row["transcript"] == ""
    assert row["transcript_status"] == "not_available" and row["domain"] == "scripture"
    assert validate_row(row) == []


def test_validate_row_rejects_integrity_violations(tmp_path):
    (tmp_path / "X.mp3").write_bytes(b"")
    [row] = vahnt_rows(tmp_path, lambda _: 1.0)
    assert validate_row({**row, "transcript": "made up"})  # text while not_available
    assert validate_row({**row, "transcript_status": "verified"})  # verified without text
    assert validate_row({**row, "speaker_id": "9876543210"})  # phone number as speaker id
    assert validate_row({**row, "source": "native_recording"})  # native without consent


def test_speaker_ids_are_pseudonymous_stable_and_need_salt(tmp_path):
    data = tmp_path / "data"
    first = speaker_id_for("Ramesh 9876543210", data, SALT)
    assert first == "VH_S001"
    assert speaker_id_for("  ramesh 9876543210 ", data, SALT) == first  # normalised key
    assert speaker_id_for("someone else", data, SALT) == "VH_S002"
    registry = (data / REGISTRY).read_text(encoding="utf-8")
    assert "9876543210" not in registry and "Ramesh" not in registry
    with pytest.raises(RuntimeError, match="SPEAKER_ID_SALT"):
        speaker_id_for("x", data, "")


def test_register_recording_requires_consent(tmp_path):
    with pytest.raises(ValueError, match="consent"):
        register(tmp_path, consent="refused")
    assert not (tmp_path / "data" / REGISTRY).exists()  # no ID allocated for a refusal


def test_register_recording_rejects_malformed_audio(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="rejected"):
        register_recording(bad, "k", "Akola", "general", "given", data_dir=tmp_path / "data", salt=SALT)


def test_register_recording_preserves_original_and_writes_16k(tmp_path):
    row = register(tmp_path, prompt_text="माझ्या कपाशीवर बोंडअळी आली आहे")
    data = tmp_path / "data"
    assert row["utterance_id"] == "VH_S001_0001" and row["recording_quality"] == "good"
    original = data / RAW_RECORDINGS / "VH_S001" / "VH_S001_0001.wav"
    assert original.read_bytes() == (tmp_path / "a.wav").read_bytes()
    info = sf.info(str(data / PROCESSED_RECORDINGS / "VH_S001" / "VH_S001_0001.wav"))
    assert (info.samplerate, info.channels) == (SR, 1)
    # prompt text is queued for review, never auto-verified
    assert row["transcript_status"] == "pending_review" and row["transcript"] == ""
    [g] = gold.read_gold(data)
    assert g["review_status"] == "pending_review" and g["provenance"] == "read_prompt"
    assert register(tmp_path, name="b.wav")["utterance_id"] == "VH_S001_0002"


def test_quality_check_flags_clipping_and_quiet_audio():
    t = np.arange(SR) / SR
    assert quality_check((0.3 * np.sin(2 * np.pi * 200 * t)).astype("float32"), SR)[0] == "good"
    assert quality_check(np.clip(2 * np.sin(2 * np.pi * 200 * t), -1, 1).astype("float32"), SR)[0] == "poor"
    assert quality_check((0.001 * np.sin(2 * np.pi * 200 * t)).astype("float32"), SR)[0] == "poor"


def manifest_rows(speakers):
    base = {"duration_sec": "10", "split": "unassigned"}
    return [{**base, "utterance_id": f"{s}_{i}", "speaker_id": s} for s in speakers for i in range(3)]


def test_split_is_by_speaker_deterministic_and_unknown_goes_to_train():
    rows = manifest_rows([f"VH_S{i:03d}" for i in range(1, 11)] + ["unknown"])
    first = assign_splits(rows, seed=1)
    assert first == assign_splits(rows, seed=1)
    assert speaker_leaks(first) == {}
    assert {r["split"] for r in first if r["speaker_id"] == "unknown"} == {"train"}
    assert {r["split"] for r in first} == {"train", "validation", "test"}


def test_leakage_is_detected_and_blocks_split_export(tmp_path):
    rows = manifest_rows(["VH_S001"])
    rows[0]["split"], rows[1]["split"] = "train", "test"
    assert speaker_leaks(rows) == {"VH_S001": {"train", "test"}}
    with pytest.raises(ValueError, match="leakage"):
        write_splits(rows, tmp_path)


def test_gold_seed_from_chunks_keeps_raw_and_skips_failed(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "JHN_002.csv").write_text(
        "chunk_index,start_sec,end_sec,prediction,status,error_message\n"
        "0,0.000,24.688, योहान अध्याय दोन,ok,\n1,24.688,52.775,,failed,oom\n",
        encoding="utf-8",
    )
    rows = gold.seed_from_chunks([], chunks, tmp_path)
    assert [r["segment_id"] for r in rows] == ["JHN_002_c000"]
    assert rows[0]["raw_asr"] == "योहान अध्याय दोन" and rows[0]["corrected"] == ""
    rows[0]["review_status"] = "verified"  # pretend it was reviewed...
    assert gold.seed_from_chunks(rows, chunks, tmp_path) == rows  # ...reseed must not overwrite


def test_gold_review_requires_text_and_reviewer():
    rows = [gold.new_row("s1", "a.wav", "VH_S001")]
    with pytest.raises(ValueError, match="verified requires"):
        gold.review(rows, "s1", "", "AG", "verified")
    gold.review(rows, "s1", "माझ्या कपाशीवर", "AG", "verified")
    assert rows[0]["review_status"] == "verified" and rows[0]["reviewed_at"]


def test_export_includes_only_verified_segments(tmp_path):
    data = tmp_path / "data"
    src = speech_like(tmp_path / "long.wav", seconds=6, sr=SR)
    rows = [
        gold.new_row("seg_ok", str(src), "VH_S001", "1.000", "3.500"),
        gold.new_row("seg_pending", str(src), "VH_S001", "3.500", "5.000", corrected="draft"),
    ]
    gold.review(rows, "seg_ok", "खरा मजकूर", "AG", "verified")
    gold.write_gold(rows, data)
    assert gold.export_gold(data) == 1
    with (data / GOLD_EVAL).open(encoding="utf-8", newline="") as f:
        [row] = list(csv.DictReader(f))
    assert row["reference_text"] == "खरा मजकूर" and row["speaker_id"] == "VH_S001"
    assert sf.info(str(data / "evaluation" / "audio" / "seg_ok.wav")).duration == pytest.approx(2.5, abs=0.01)


def test_withdraw_deletes_all_speaker_data_and_retires_id(tmp_path):
    data = tmp_path / "data"
    register(tmp_path, key="alice", prompt_text="text")
    register(tmp_path, key="bob", name="b.wav")
    assert withdraw_speaker("VH_S001", data) == 1
    assert [r["speaker_id"] for r in read_manifest(data)] == ["VH_S002"]
    assert not (data / RAW_RECORDINGS / "VH_S001").exists()
    assert not (data / PROCESSED_RECORDINGS / "VH_S001").exists()
    assert gold.read_gold(data) == []
    # pseudonym retired, not reused, even if the same person registers again later
    assert register(tmp_path, key="alice", name="c.wav")["speaker_id"] == "VH_S003"
