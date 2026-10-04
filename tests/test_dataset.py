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
    assert rows[0]["raw_asr"] == "योहान अध्याय दोन" and rows[0]["candidate_text"] == ""
    assert rows[0]["source_chapter"] == "JHN_002"
    rows[0]["review_status"] = "verified"  # pretend it was reviewed...
    assert gold.seed_from_chunks(rows, chunks, tmp_path) == rows  # ...reseed must not overwrite


def test_gold_review_decisions_map_to_status_and_keep_candidate():
    rows = [gold.new_row("s1", "a.wav", "unknown", candidate="पयले शब्द होता")]
    gold.review(rows, "s1", "needs_correction", "AG", corrected_text="पयले शब्दच होता")
    row = rows[0]
    assert row["review_status"] == "verified" and row["reviewed_at"]
    assert row["candidate_text"] == "पयले शब्द होता"  # never overwritten
    assert gold.reference(row) == "पयले शब्दच होता"
    gold.review(rows, "s1", "correct", "AG")
    assert gold.reference(row) == "पयले शब्द होता" and row["corrected_text"] == ""
    gold.review(rows, "s1", "needs_realignment", "AG")
    assert row["review_status"] == "needs_realignment"
    gold.review(rows, "s1", "rejected", "AG")
    assert row["review_status"] == "rejected"


def test_gold_review_rejects_invalid_decisions_without_changing_the_row():
    rows = [gold.new_row("s1", "a.wav", "unknown")]  # no candidate (ASR-seeded chunk)
    before = dict(rows[0])
    with pytest.raises(ValueError, match="needs a candidate_text"):
        gold.review(rows, "s1", "correct", "AG")
    with pytest.raises(ValueError, match="requires corrected_text"):
        gold.review(rows, "s1", "needs_correction", "AG", corrected_text="  ")
    with pytest.raises(ValueError, match="reviewer id"):
        gold.review(rows, "s1", "rejected", "")
    with pytest.raises(ValueError, match="decision must be"):
        gold.review(rows, "s1", "verified", "AG")
    assert rows[0] == before
    with pytest.raises(KeyError):
        gold.review(rows, "nope", "rejected", "AG")


def test_validate_gold_rejects_hand_edited_verified_without_review():
    row = gold.new_row("s1", "a.wav", "unknown", candidate="x")
    row["review_status"] = "verified"  # e.g. edited in a spreadsheet, no decision/reviewer
    assert any("decision and a reviewer" in e for e in gold.validate_gold([row]))


def test_export_includes_only_verified_segments_from_selected_chapters(tmp_path):
    data = tmp_path / "data"
    src = speech_like(tmp_path / "long.wav", seconds=6, sr=SR)
    rows = [
        gold.new_row("seg_ok", str(src), "unknown", "1.000", "3.500", candidate="खरा", source_chapter="JHN_001"),
        gold.new_row("seg_pending", str(src), "unknown", "3.500", "5.000", candidate="draft", source_chapter="JHN_001"),
        gold.new_row("seg_other", str(src), "unknown", "0.000", "1.000", candidate="इतर", source_chapter="JHN_002"),
    ]
    gold.review(rows, "seg_ok", "needs_correction", "AG", corrected_text="खरा मजकूर")
    gold.review(rows, "seg_other", "correct", "AG")
    gold.write_gold(rows, data)
    assert gold.export_gold(data, chapters=["JHN_001"]) == 1
    with (data / GOLD_EVAL).open(encoding="utf-8", newline="") as f:
        [row] = list(csv.DictReader(f))
    assert row["reference_text"] == "खरा मजकूर" and row["segment_id"] == "seg_ok"
    assert float(row["duration_sec"]) == pytest.approx(2.5, abs=0.01)
    assert sf.info(str(data / "evaluation" / "audio" / "seg_ok.wav")).duration == pytest.approx(2.5, abs=0.01)
    assert gold.export_gold(data) == 2  # no chapter filter: every verified segment


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


def test_import_alignments_queues_published_text_for_review(tmp_path):
    (tmp_path / "align").mkdir()
    (tmp_path / "text").mkdir()
    (tmp_path / "text" / "JHN_001.json").write_text('{"url": "https://www.bible.com/bible/3451/JHN.1.VAHNT"}', encoding="utf-8")
    (tmp_path / "align" / "JHN_001.csv").write_text(
        "segment_id,audio_path,start_sec,end_sec,units,n_headings,text,align_score,raw_asr,cer_vs_raw_asr\n"
        "JHN_001_a000,audio/JHN_001.mp3,6.320,19.620,JHN.1.1..JHN.1.2,1,पयले शब्द होता,-0.2854,पहिले शब्द होता,0.04\n",
        encoding="utf-8",
    )
    rows = gold.import_alignments([], tmp_path / "align", tmp_path / "text")
    [row] = rows
    assert row["review_status"] == "pending_review" and row["decision"] == ""  # never auto-verified
    assert row["candidate_text"] == "पयले शब्द होता" and row["raw_asr"] == "पहिले शब्द होता"
    assert row["source_chapter"] == "JHN_001" and row["source_ref"] == "JHN.1.1..JHN.1.2"
    assert row["source_url"] == "https://www.bible.com/bible/3451/JHN.1.VAHNT" and row["align_score"] == "-0.2854"
    gold.review(rows, "JHN_001_a000", "correct", "AG")
    assert gold.import_alignments(rows, tmp_path / "align", tmp_path / "text") == rows  # no overwrite


def test_session_sheet_filters_crop_scope_and_skips_unrewritten_read_prompts():
    from varhadi_data.prompts import session_sheet

    base = {"topic": "t", "text": "मराठी मसुदा", "varhadi_text": ""}
    prompts = [
        {**base, "prompt_id": "E03", "type": "elicitation", "crop": "cotton"},
        {**base, "prompt_id": "R01", "type": "read", "crop": "cotton"},  # no Varhadi rewrite yet
        {**base, "prompt_id": "R02", "type": "read", "crop": "soybean", "varhadi_text": "वऱ्हाडी वाक्य"},
        {**base, "prompt_id": "R03", "type": "read", "crop": "orange", "varhadi_text": "x"},  # out of scope
        {**base, "prompt_id": "E01", "type": "elicitation", "crop": "general"},
    ]
    sheet, skipped = session_sheet(prompts, ["cotton", "soybean"], ["general"])
    assert [r["prompt_id"] for r in sheet] == ["E03", "R02", "E01"] and skipped == ["R01"]
    assert sheet[1]["show_text"] == sheet[1]["expected_read_text"] == "वऱ्हाडी वाक्य"
    assert sheet[0]["expected_read_text"] == ""  # free answer: transcribed after recording


def test_real_prompt_file_is_complete_and_unrewritten():
    from varhadi_data.config import DATA_DIR
    from varhadi_data.prompts import crop_scope, read_prompts

    prompts = read_prompts(DATA_DIR)
    crops, always = crop_scope(DATA_DIR)
    assert {p["crop"] for p in prompts} <= set(crops) | set(always)
    assert all(not p["varhadi_text"] for p in prompts)  # nobody has written Varhadi text for us
