import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_gold_eval import score_and_report  # noqa: E402

from asr_baseline.utils import write_csv  # noqa: E402


def write(path, rows):
    write_csv(path, rows, list(rows[0]))
    return path


def test_score_and_report_uses_verified_references_and_labels_pilot(tmp_path):
    eval_csv = write(tmp_path / "gold_eval.csv", [
        {"audio_path": "data/evaluation/audio/s1.wav", "reference_text": "पयले शब्द होता, अन् हा.",
         "speaker_id": "unknown", "segment_id": "s1", "source_chapter": "JHN_001", "duration_sec": "12.0"},
        {"audio_path": "data/evaluation/audio/s2.wav", "reference_text": "तो देव होता",
         "speaker_id": "unknown", "segment_id": "s2", "source_chapter": "JHN_001", "duration_sec": "6.0"},
    ])
    predictions = write(tmp_path / "pred.csv", [
        # the predictions file's own reference column is ignored in favour of the gold export
        {"audio_path": "data/evaluation/audio/s1.wav", "reference": "IGNORED", "prediction": "पहिले शब्द होता हा",
         "duration_sec": "12", "inference_time_sec": "1", "status": "ok", "error_message": ""},
        {"audio_path": "data/evaluation/audio/s2.wav", "reference": "", "prediction": "तो देव होता रे",
         "duration_sec": "6", "inference_time_sec": "1", "status": "ok", "error_message": ""},
        {"audio_path": "data/other.wav", "reference": "x", "prediction": "y",  # not in the gold export
         "duration_sec": "1", "inference_time_sec": "1", "status": "ok", "error_message": ""},
    ])
    metrics = score_and_report(eval_csv, predictions, tmp_path / "out", "baseline", "model-x", ["JHN_001"])
    assert metrics["label"] == "PILOT EVALUATION"
    assert metrics["evaluated_samples"] == 2 and metrics["evaluated_minutes"] == 0.3
    assert (metrics["substitutions"], metrics["deletions"], metrics["insertions"]) == (1, 1, 1)
    assert metrics["wer"] == pytest.approx(3 / 8)
    saved = json.loads((tmp_path / "out" / "baseline_metrics.json").read_text(encoding="utf-8"))
    assert saved["wer"] == metrics["wer"]
    report = (tmp_path / "out" / "baseline_report.md").read_text(encoding="utf-8")
    assert "PILOT EVALUATION" in report and "NOT agricultural" in report
    assert (tmp_path / "out" / "baseline_error_pairs.csv").exists()
    assert (tmp_path / "out" / "baseline_per_sample.csv").exists()


def test_score_and_report_refuses_without_verified_references(tmp_path):
    eval_csv = tmp_path / "gold_eval.csv"
    eval_csv.write_text("audio_path,reference_text,speaker_id,segment_id,source_chapter,duration_sec\n", encoding="utf-8")
    predictions = write(tmp_path / "pred.csv", [
        {"audio_path": "a.wav", "reference": "x", "prediction": "y", "duration_sec": "1",
         "inference_time_sec": "1", "status": "ok", "error_message": ""},
    ])
    with pytest.raises(RuntimeError, match="nothing scored"):
        score_and_report(eval_csv, predictions, tmp_path / "out", "baseline", "m", ["JHN_001"])
    assert not (tmp_path / "out").exists()


def test_compare_requires_identical_test_sets_and_reports_differences(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "evaluation"))
    from compare import compare

    def run(name, wer, segments):
        d = tmp_path / name
        d.mkdir()
        metrics = {"model": name, "label": "PILOT EVALUATION", "test_chapters": ["JHN_001"], "wer": wer, "cer": 0.1,
                   "substitutions": 3, "deletions": 1, "insertions": 0, "reference_words": 20,
                   "evaluated_samples": len(segments), "evaluated_minutes": 1.0}
        (d / f"{name}_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        write(d / f"{name}_per_sample.csv", [{"segment_id": s} for s in segments])

    run("baseline", 0.4, ["s1", "s2"])
    run("adapted", 0.25, ["s1", "s2"])
    result = compare(tmp_path, "baseline", "adapted", tmp_path / "out")
    wer = next(r for r in result["metrics"] if r["metric"] == "wer")
    assert wer["difference_b_minus_a"] == -0.15
    md = (tmp_path / "out" / "comparison.md").read_text(encoding="utf-8")
    assert "PILOT" in md and "better" not in md.lower()

    run("other", 0.2, ["s1", "s3"])
    with pytest.raises(ValueError, match="different test sets"):
        compare(tmp_path, "baseline", "other", tmp_path / "out2")


def test_gold_export_excludes_high_confidence_and_other_unverified_segments(tmp_path):
    import numpy as np
    import soundfile as sf

    from varhadi_data import gold

    wav = tmp_path / "JHN_001.wav"
    sf.write(str(wav), np.zeros(16000 * 8, dtype="float32"), 16000)
    rows = []
    for sid, triage in [("hi", "HIGH_CONFIDENCE_CANDIDATE"), ("nr", "NEEDS_REVIEW"), ("rj", "REJECT_CANDIDATE"),
                        ("ok", "NEEDS_REVIEW")]:
        row = gold.new_row(sid, str(wav), "unknown", "1.000", "3.000", candidate="पयले शब्द", source_chapter="JHN_001")
        row["triage_status"], row["confidence_score"] = triage, "0.99"
        rows.append(row)
    gold.review(rows, "ok", "CORRECTED", "AG", corrected_text="पयले शब्दच")
    data = tmp_path / "data"
    gold.write_gold(rows, data)
    assert gold.export_gold(data, chapters=["JHN_001"]) == 1  # only the human-reviewed segment
    exported = list(csv.DictReader(open(data / "evaluation" / "gold_eval.csv", encoding="utf-8")))
    assert [(r["segment_id"], r["reference_text"]) for r in exported] == [("ok", "पयले शब्दच")]
