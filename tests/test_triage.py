import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import triage_gold_candidates as cli  # noqa: E402
from config import ROOT_DIR  # noqa: E402
from varhadi_data import gold  # noqa: E402
from varhadi_data.triage import audio_signals, load_config, plateau, ramp, score_segment  # noqa: E402

SR = 16000
CONFIG = load_config()
GOOD_AUDIO = {"silence_ratio": 0.1, "clip_ratio": 0.0, "rms_dbfs": -18.0}
TEXT = "ह्या सगळ्या जगाच्या बन्याच्या पयले शब्द होता अन् हा शब्द देवासोबत होता अन् हा शब्दचं देव होता"


def row(candidate=TEXT, asr=TEXT, start="0.000", end="11.000", align="-0.3"):
    r = gold.new_row("s1", "audio/JHN_001.mp3", "unknown", start, end, candidate=candidate, raw_asr=asr,
                     source_chapter="JHN_001")
    r["align_score"] = align
    return r


def test_ramp_and_plateau_normalise_to_unit_interval():
    assert ramp(-0.5, bad=-2.0, good=-0.5) == 1.0 and ramp(-3.0, bad=-2.0, good=-0.5) == 0.0
    assert ramp(-1.25, bad=-2.0, good=-0.5) == pytest.approx(0.5)
    assert ramp(0.12, bad=0.45, good=0.12) == 1.0 and ramp(0.6, bad=0.45, good=0.12) == 0.0  # reversed direction
    assert ramp(None, 0, 1) == 0.0 and ramp(float("nan"), 0, 1) == 0.0
    limits = {"hard_min": 1.0, "min_ok": 3.0, "max_ok": 25.0, "hard_max": 45.0}
    assert plateau(10, limits) == 1.0 and plateau(2, limits) == 0.5 and plateau(0.5, limits) == 0.0
    assert plateau(35, limits) == 0.5 and plateau(50, limits) == 0.0


def test_confidence_is_weighted_components_minus_penalties():
    result = score_segment(row(), GOOD_AUDIO, 0, CONFIG)
    weights = CONFIG["weights"]
    expected = sum(weights[name] * result[f"{name}_score"] for name in weights) / sum(weights.values())
    assert result["base_confidence"] == pytest.approx(expected, abs=1e-3)
    assert result["confidence_score"] == pytest.approx(expected - result["artifact_penalty"], abs=1e-3)
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert result["triage_status"] == "HIGH_CONFIDENCE_CANDIDATE"


def test_triage_classification_paths():
    # translator note in parentheses: correctable -> review, never auto-accepted, never auto-rejected
    noted = score_segment(row(candidate=TEXT + " (जवळपास तीन हजार लिटर)", align="-2.5"), GOOD_AUDIO, 0, CONFIG)
    assert "parentheses" in noted["triage_flags"] and noted["triage_status"] == "NEEDS_REVIEW"
    # text crammed into too little audio (misalignment) -> reject candidate
    crammed = score_segment(row(end="2.500"), GOOD_AUDIO, 0, CONFIG)
    assert "speaking_rate_out_of_range" in crammed["triage_flags"] and crammed["triage_status"] == "REJECT_CANDIDATE"
    # mostly silence -> reject candidate
    silent = score_segment(row(), {"silence_ratio": 0.9, "clip_ratio": 0.0, "rms_dbfs": -45.0}, 0, CONFIG)
    assert silent["triage_status"] == "REJECT_CANDIDATE"
    # weak alignment but fine otherwise -> needs review
    weak = score_segment(row(align="-1.6"), GOOD_AUDIO, 0, CONFIG)
    assert weak["triage_status"] == "NEEDS_REVIEW"


def test_malformed_and_missing_scores_never_crash_or_auto_accept():
    for bad in (row(align=""), row(align="not-a-number"), row(start="", end=""), row(asr=""), row(candidate="")):
        result = score_segment(bad, None, 0, CONFIG)
        assert result["triage_status"] in ("NEEDS_REVIEW", "REJECT_CANDIDATE")
        assert 0.0 <= result["confidence_score"] <= 1.0
    assert "missing_align_score" in score_segment(row(align=""), GOOD_AUDIO, 0, CONFIG)["triage_flags"]
    assert score_segment(row(candidate=""), GOOD_AUDIO, 0, CONFIG)["triage_status"] == "REJECT_CANDIDATE"


def test_audio_signals_detect_silence_clipping_and_loudness():
    t = np.arange(4 * SR) / SR
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype("float32")
    assert audio_signals(tone, SR)["silence_ratio"] == 0.0
    half_silent = tone.copy()
    half_silent[: 2 * SR] = 0
    assert audio_signals(half_silent, SR)["silence_ratio"] == pytest.approx(0.5, abs=0.01)
    assert audio_signals(np.clip(5 * tone, -1, 1), SR)["clip_ratio"] > 0.1
    assert audio_signals(np.zeros(0, dtype="float32"), SR)["silence_ratio"] == 1.0


def _hashes(paths):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def test_triage_script_only_writes_triage_columns_and_never_verifies(tmp_path):
    data, audio_dir = tmp_path / "data", tmp_path / "audio"
    audio_dir.mkdir()
    t = np.arange(12 * SR) / SR
    sf.write(str(audio_dir / "JHN_001.wav"), (0.3 * np.sin(2 * np.pi * 200 * t)).astype("float32"), SR)
    rows = [row(), row(candidate=TEXT + " (टीप)", align="-2.5")]
    rows[1]["segment_id"] = "s2"
    gold.review(rows, "s2", "rejected", "AG", notes="human said no")
    gold.write_gold(rows, data)
    downloader_files = [ROOT_DIR / name for name in
                        ("progress.json", "failed.json", "chapter_metadata.json", "downloader.py", "browser.py")]
    before_hashes = _hashes(downloader_files)
    before = {r["segment_id"]: {k: v for k, v in r.items() if k not in gold.TRIAGE_FIELDS} for r in gold.read_gold(data)}

    assert cli.main(["--all", "--export-csv", "--report"], data_dir=data, audio_dir=audio_dir) == 0

    after_rows = gold.read_gold(data)
    after = {r["segment_id"]: {k: v for k, v in r.items() if k not in gold.TRIAGE_FIELDS} for r in after_rows}
    assert before == after  # review fields, candidate text etc. untouched
    assert {r["segment_id"]: r["review_status"] for r in after_rows} == {"s1": "pending_review", "s2": "rejected"}
    assert after_rows[0]["triage_status"] == "HIGH_CONFIDENCE_CANDIDATE"  # triaged, but still NOT verified
    assert all(r["triage_version"] == CONFIG["version"] for r in after_rows)
    summary = json.loads((data / cli.SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary["total_candidates"] == 2 and summary["human_review_status_counts"] == {"pending_review": 1, "rejected": 1}
    assert (data / cli.REPORT_CSV).exists()
    assert _hashes(downloader_files) == before_hashes  # downloader data byte-identical


def test_cli_threshold_override_and_validation(tmp_path):
    data, audio_dir = tmp_path / "data", tmp_path / "audio"
    audio_dir.mkdir()
    gold.write_gold([row()], data)
    assert cli.main(["--chapter", "JHN_001", "--threshold", "1.0"], data_dir=data, audio_dir=audio_dir) == 0
    assert gold.read_gold(data)[0]["triage_status"] == "NEEDS_REVIEW"  # nothing reaches 1.0 without audio
    with pytest.raises(SystemExit):
        cli.main(["--all", "--threshold", "0.3", "--review-threshold", "0.6"], data_dir=data, audio_dir=audio_dir)
