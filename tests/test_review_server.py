import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import review_server  # noqa: E402
from varhadi_data import gold  # noqa: E402

SR = 16000
gold_read, gold_write = gold.read_gold, gold.write_gold


@pytest.fixture
def server(tmp_path, monkeypatch):
    wav = tmp_path / "JHN_001.wav"
    sf.write(str(wav), (0.3 * np.sin(np.arange(10 * SR) / 10)).astype("float32"), SR)
    data = tmp_path / "data"
    triage = [("seg_high", "HIGH_CONFIDENCE_CANDIDATE", "0.97"), ("seg_review_b", "NEEDS_REVIEW", "0.88"),
              ("seg_reject", "REJECT_CANDIDATE", "0.30"), ("seg_review_a", "NEEDS_REVIEW", "0.61")]
    rows = []
    for sid, status, conf in triage:
        row = gold.new_row(sid, str(wav), "unknown", "2.000", "5.500", candidate="पयले शब्द", source_chapter="JHN_001")
        row.update(triage_status=status, confidence_score=conf)
        rows.append(row)
    gold.write_gold(rows, data)
    monkeypatch.setattr(gold, "read_gold", lambda data_dir=data: gold_read(data))
    monkeypatch.setattr(gold, "write_gold", lambda rows, data_dir=data: gold_write(rows, data))
    httpd = HTTPServer(("127.0.0.1", 0), review_server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", data
    httpd.shutdown()


def post(base, payload):
    request = urllib.request.Request(f"{base}/api/review", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def queue(base):
    with urllib.request.urlopen(f"{base}/api/queue") as response:
        return json.loads(response.read())


def by_id(data, sid):
    return next(r for r in gold_read(data) if r["segment_id"] == sid)


def test_queue_orders_needs_review_then_rejects_then_high_confidence(server):
    base, _ = server
    order = [item["segment_id"] for item in queue(base)["items"]]
    assert order == ["seg_review_a", "seg_review_b", "seg_reject", "seg_high"]


def test_audio_clip_has_segment_duration(server, tmp_path):
    base, _ = server
    with urllib.request.urlopen(f"{base}/audio/seg_high") as response:
        (tmp_path / "clip.wav").write_bytes(response.read())
    assert sf.info(str(tmp_path / "clip.wav")).duration == pytest.approx(3.5, abs=0.01)
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"{base}/audio/..%2F..%2Fsecret")


def test_verify_requires_full_listen_and_high_confidence_is_not_verified(server):
    base, data = server
    status, body = post(base, {"segment_id": "seg_high", "decision": "VERIFIED", "reviewer": "AG", "played_fraction": 0.5})
    assert status == 400 and "listen" in body["error"]
    assert by_id(data, "seg_high")["review_status"] == "pending_review"  # triage alone never verifies
    assert queue(base)["progress"]["verified"] == 0


def test_correction_is_stored_separately_and_persists_across_refresh(server):
    base, data = server
    status, _ = post(base, {"segment_id": "seg_review_a", "decision": "CORRECTED", "reviewer": "AG",
                            "corrected_text": "पयले शब्दच", "notes": "extra च spoken", "played_fraction": 1.0})
    assert status == 200
    row = by_id(data, "seg_review_a")
    assert row["candidate_text"] == "पयले शब्द"  # original candidate preserved
    assert (row["corrected_text"], row["decision"], row["review_status"]) == ("पयले शब्दच", "needs_correction", "verified")
    assert row["review_notes"] == "extra च spoken" and row["reviewer"] == "AG" and row["reviewed_at"]
    # a "refresh" is just a new request: state comes from the CSV, not from memory
    refreshed = {item["segment_id"]: item for item in queue(base)["items"]}
    assert refreshed["seg_review_a"]["decision"] == "needs_correction"


def test_progress_counts_and_rejected_excluded_from_verified(server):
    base, data = server
    post(base, {"segment_id": "seg_review_a", "decision": "VERIFIED", "reviewer": "AG", "played_fraction": 1.0})
    post(base, {"segment_id": "seg_review_b", "decision": "CORRECTED", "reviewer": "AG",
                "corrected_text": "नवा मजकूर", "played_fraction": 1.0})
    post(base, {"segment_id": "seg_reject", "decision": "REJECTED", "reviewer": "AG"})
    p = queue(base)["progress"]
    assert (p["reviewed"], p["verified"], p["corrected"], p["rejected"], p["remaining"]) == (3, 1, 1, 1, 1)
    assert p["triage"] == {"HIGH_CONFIDENCE_CANDIDATE": 1, "NEEDS_REVIEW": 2, "REJECT_CANDIDATE": 1}
    assert p["verified_minutes"] == pytest.approx(2 * 3.5 / 60, abs=0.01)  # rejected not counted
    assert by_id(data, "seg_reject")["review_status"] == "rejected"


def test_duplicate_submission_updates_one_row(server):
    base, data = server
    for _ in range(2):
        status, _ = post(base, {"segment_id": "seg_review_a", "decision": "VERIFIED", "reviewer": "AG", "played_fraction": 1.0})
        assert status == 200
    rows = gold_read(data)
    assert len(rows) == 4 and sum(r["segment_id"] == "seg_review_a" for r in rows) == 1
    assert queue(base)["progress"]["verified"] == 1


@pytest.mark.parametrize("reviewer", ["", "A", "Ramesh Patil", "9876543210", "<script>"])
def test_invalid_reviewer_ids_are_rejected(server, reviewer):
    base, data = server
    status, body = post(base, {"segment_id": "seg_reject", "decision": "REJECTED", "reviewer": reviewer})
    assert status == 400 and "reviewer id" in body["error"]
    assert by_id(data, "seg_reject")["review_status"] == "pending_review"
