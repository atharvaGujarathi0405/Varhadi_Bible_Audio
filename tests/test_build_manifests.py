import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "varhadi_adaptation"))
from build_manifests import PSEUDO, manifest_line, split_rows  # noqa: E402

from varhadi_data import gold  # noqa: E402


def seg(sid, chapter, decision=None, triage="NEEDS_REVIEW", candidate="पयले शब्द, होता."):
    row = gold.new_row(sid, f"audio/{chapter}.mp3", "unknown", "6.320", "19.620", candidate=candidate,
                       source_chapter=chapter)
    row["triage_status"] = triage
    if decision:
        gold.review([row], sid, decision, "AG", corrected_text="पयले शब्दच होता." if decision == "CORRECTED" else "")
    return row


def ids(rows):
    return [r["segment_id"] for r in rows]


def test_default_uses_only_human_gold_and_holds_out_chapters():
    rows = [
        seg("t1", "JHN_001", "VERIFIED"), seg("t2", "JHN_001", triage="HIGH_CONFIDENCE_CANDIDATE"),
        seg("v1", "JHN_002", "CORRECTED"), seg("v2", "JHN_002", triage="HIGH_CONFIDENCE_CANDIDATE"),
        seg("a1", "MAT_001", "VERIFIED"), seg("a2", "MAT_001", triage="HIGH_CONFIDENCE_CANDIDATE"),
        seg("r1", "MAT_001", "REJECTED"), seg("n1", "MAT_001", "NEEDS_REALIGNMENT"),
    ]
    strict = split_rows(rows, ["JHN_001"], ["JHN_002"])
    assert (ids(strict["test"]), ids(strict["val"]), ids(strict["train"])) == (["t1"], ["v1"], ["a1"])
    assert strict["train_pseudo_labeled"] == [] and strict["val_pseudo_labeled"] == []


def test_pseudo_labels_are_separate_high_confidence_only_and_never_in_test():
    rows = [
        seg("t2", "JHN_001", triage="HIGH_CONFIDENCE_CANDIDATE"),  # test chapter: never pseudo
        seg("a1", "MAT_001", "VERIFIED"),
        seg("a2", "MAT_001", triage="HIGH_CONFIDENCE_CANDIDATE"),
        seg("a3", "MAT_001", triage="NEEDS_REVIEW"),  # not eligible
        seg("a4", "MAT_001", triage="REJECT_CANDIDATE"),  # not eligible
        seg("v2", "JHN_002", triage="HIGH_CONFIDENCE_CANDIDATE"),
    ]
    weak = split_rows(rows, ["JHN_001"], ["JHN_002"], include_unverified=True)
    assert ids(weak["train"]) == ["a1"]  # gold manifest stays gold-only
    assert ids(weak["train_pseudo_labeled"]) == ["a2"]
    assert ids(weak["val_pseudo_labeled"]) == ["v2"]
    assert weak["test"] == []
    assert manifest_line(weak["train_pseudo_labeled"][0], "audio")["label_source"] == PSEUDO


def test_manifest_line_uses_offset_lang_and_scoring_normalisation():
    corrected = seg("t1", "JHN_001", "CORRECTED")
    line = manifest_line(corrected, "audio")
    assert line == {"audio_filepath": "audio/JHN_001.wav", "offset": 6.32, "duration": 13.3,
                    "text": "पयले शब्दच होता", "lang": "mr", "segment_id": "t1", "label_source": "GOLD_CORRECTED"}
    assert corrected["candidate_text"] == "पयले शब्द, होता."  # candidate untouched
    assert manifest_line(seg("t3", "JHN_001", "VERIFIED"), "audio")["label_source"] == "GOLD_VERIFIED"
