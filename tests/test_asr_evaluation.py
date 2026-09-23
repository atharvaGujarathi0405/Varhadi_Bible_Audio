from asr_baseline.evaluate import compute_metrics, evaluable_rows, per_sample_metrics
from asr_baseline.error_analysis import build_error_analysis


def test_evaluable_rows_excludes_missing_reference_and_failed_status():
    rows = [
        {"audio_path": "a.mp3", "reference": "text", "prediction": "text", "status": "ok"},
        {"audio_path": "b.mp3", "reference": "", "prediction": "text", "status": "ok"},
        {"audio_path": "c.mp3", "reference": "text", "prediction": "", "status": "failed"},
    ]
    assert evaluable_rows(rows) == [rows[0]]


def test_compute_metrics_empty_when_no_ground_truth():
    metrics = compute_metrics([])
    assert metrics["evaluated_samples"] == 0
    assert metrics["wer"] is None
    assert metrics["cer"] is None


def test_compute_metrics_perfect_prediction_has_zero_wer_cer():
    rows = [{"audio_path": "a.mp3", "reference": "hello world", "prediction": "hello world"}]
    metrics = compute_metrics(rows)
    assert metrics["evaluated_samples"] == 1
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0


def test_per_sample_metrics_joins_speaker_id():
    rows = [{"audio_path": "a.mp3", "reference": "hi", "prediction": "hi there", "duration_sec": "1.0"}]
    result = per_sample_metrics(rows, speaker_by_path={"a.mp3": "S01"})
    assert result[0]["speaker_id"] == "S01"
    assert result[0]["wer"] > 0


def test_error_analysis_flags_high_wer():
    rows = [{"audio_path": "a.mp3", "reference": "one two three", "prediction": "zero", "wer": "1.0", "cer": "0.9"}]
    result = build_error_analysis(rows)
    assert "[HIGH_WER]" in result[0]["error_summary"]
    assert result[0]["manual_category"] == ""
