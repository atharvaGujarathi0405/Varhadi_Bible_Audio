from asr_baseline.evaluate import compute_metrics, evaluable_rows, per_sample_metrics
from asr_baseline.error_analysis import build_error_analysis, edit_operations


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


def test_metrics_report_edit_counts_and_ignore_punctuation_only():
    rows = [
        {"audio_path": "a.wav", "reference": "पयले शब्द होता, अन् हा.", "prediction": "पहिले शब्द होता हा"},
        {"audio_path": "b.wav", "reference": "तो देव होता", "prediction": "तो देव होता रे"},
    ]
    metrics = compute_metrics(rows)
    # a: पयले->पहिले substituted, अन् deleted; b: रे inserted; punctuation is not an error
    assert (metrics["substitutions"], metrics["deletions"], metrics["insertions"]) == (1, 1, 1)
    assert metrics["reference_words"] == 8
    assert metrics["wer"] == 3 / 8
    assert compute_metrics([{"audio_path": "b", "reference": "अ, ब.", "prediction": "अ ब"}])["wer"] == 0.0
    # dialect spelling is NOT normalised away: पयले vs पहिले stays an error
    assert compute_metrics([{"audio_path": "c", "reference": "पयले", "prediction": "पहिले"}])["wer"] == 1.0


def test_edit_operations_aggregates_pairs_with_heuristic_tags_only():
    rows = [
        {"reference": "पयले शब्द होता अन् हा", "prediction": "पहिले शब्द होता हा"},
        {"reference": "पयले देव.", "prediction": "पहिले देव रे"},
    ]
    ops = edit_operations(rows)
    top = ops[0]
    assert (top["type"], top["reference_word"], top["hypothesis_word"], top["count"]) == ("substitution", "पयले", "पहिले", 2)
    assert top["heuristic_note"] == "similar form (heuristic)" and top["manual_category"] == ""
    kinds = {(o["type"], o["reference_word"], o["hypothesis_word"]) for o in ops}
    assert ("deletion", "अन्", "") in kinds and ("insertion", "", "रे") in kinds
