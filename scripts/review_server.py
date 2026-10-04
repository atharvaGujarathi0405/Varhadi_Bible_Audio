#!/usr/bin/env python3
"""Local listening/review tool for gold transcripts, ordered by automatic triage.

    python scripts/triage_gold_candidates.py --chapter JHN_001 --report   # optional, orders the queue
    python scripts/review_server.py [--chapter JHN_001] [--port 8765]
    then open http://127.0.0.1:8765

Queue order: NEEDS_REVIEW, then REJECT_CANDIDATE, then HIGH_CONFIDENCE_CANDIDATE, each by
lowest confidence first (untriaged segments count as NEEDS_REVIEW). Triage only orders the
queue: a HIGH_CONFIDENCE_CANDIDATE is still unverified until a human decides.

Each decision is written to data/transcripts/gold_transcripts.csv immediately, through
varhadi_data.gold (same validation as the CLI), so a refresh or a crash loses nothing.
candidate_text is never modified; corrections go to corrected_text. A verifying decision
(VERIFIED / CORRECTED) is only accepted after the clip has been played through (>= 95%),
so "verified" always means a human listened. Binds to 127.0.0.1 only.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_baseline.audio_utils import ensure_wav_copy
from asr_baseline.config import ASR_AUDIO_DIR
from varhadi_data import gold
from varhadi_data.config import resolve

MIN_PLAYED_FRACTION = 0.95
VERIFYING = {"correct", "needs_correction", "VERIFIED", "CORRECTED"}
TRIAGE_RANK = {"NEEDS_REVIEW": 0, "": 0, "REJECT_CANDIDATE": 1, "HIGH_CONFIDENCE_CANDIDATE": 2}
LOCK = threading.Lock()  # ponytail: one global lock; fine for a single local reviewer


def clip_wav(row: dict) -> bytes:
    """WAV bytes for one segment, cut from the cached 16 kHz copy; source files untouched."""
    import soundfile as sf

    source = resolve(row["audio_path"])
    wav = source if source.suffix.lower() == ".wav" else ensure_wav_copy(source, ASR_AUDIO_DIR)
    with sf.SoundFile(str(wav)) as handle:
        sr = handle.samplerate
        start = int(float(row["start_sec"]) * sr) if row["start_sec"] else 0
        stop = int(float(row["end_sec"]) * sr) if row["end_sec"] else handle.frames
        handle.seek(start)
        audio = handle.read(stop - start, dtype="float32")
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _confidence(row: dict) -> float:
    try:
        return float(row.get("confidence_score") or 0.0)
    except ValueError:
        return 0.0


def ordered(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (TRIAGE_RANK.get(r.get("triage_status", ""), 0), _confidence(r), r["segment_id"]))


def public(row: dict) -> dict:
    keys = ["segment_id", "source_chapter", "source_ref", "source_url", "start_sec", "end_sec", "align_score",
            "candidate_text", "raw_asr", "decision", "corrected_text", "reviewer", "review_status", "reviewed_at",
            "review_notes", "confidence_score", "alignment_score", "agreement_score", "triage_status", "triage_flags"]
    return {key: row.get(key, "") for key in keys}


def progress(rows: list[dict]) -> dict:
    decisions = [row["decision"] for row in rows]
    reviewed = sum(row["review_status"] != "pending_review" for row in rows)
    minutes = sum(float(r["end_sec"]) - float(r["start_sec"]) for r in rows
                  if r["review_status"] == "verified" and r["end_sec"]) / 60
    return {
        "total": len(rows),
        "reviewed": reviewed,
        "verified": decisions.count("correct"),
        "corrected": decisions.count("needs_correction"),
        "rejected": decisions.count("rejected"),
        "needs_realignment": decisions.count("needs_realignment"),
        "remaining": len(rows) - reviewed,
        "verified_minutes": round(minutes, 2),
        "triage": {status: sum(r.get("triage_status") == status for r in rows)
                   for status in ("HIGH_CONFIDENCE_CANDIDATE", "NEEDS_REVIEW", "REJECT_CANDIDATE")},
    }


class Handler(BaseHTTPRequestHandler):
    chapter: str | None = None

    def _send(self, body: bytes, content_type: str, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status=HTTPStatus.OK):
        self._send(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def _rows(self) -> list[dict]:
        rows = gold.read_gold()  # read fresh every request: the CSV is the only state
        return [row for row in rows if not self.chapter or row["source_chapter"] == self.chapter]

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if url.path == "/api/queue":
            rows = self._rows()
            return self._json({"items": [public(r) for r in ordered(rows)], "progress": progress(rows),
                               "chapter": self.chapter or "all"})
        if url.path.startswith("/audio/"):
            segment_id = url.path.removeprefix("/audio/")
            row = next((r for r in gold.read_gold() if r["segment_id"] == segment_id), None)  # lookup by id only
            if row is None:
                return self._json({"error": "unknown segment"}, HTTPStatus.NOT_FOUND)
            return self._send(clip_wav(row), "audio/wav")
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if urlparse(self.path).path != "/api/review":
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 100_000:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(length) or b"{}")
            decision = body.get("decision", "")
            if decision in VERIFYING and float(body.get("played_fraction", 0)) < MIN_PLAYED_FRACTION:
                raise ValueError("listen to the whole clip before verifying it")
            with LOCK:
                rows = gold.read_gold()
                gold.review(rows, body.get("segment_id", ""), decision, body.get("reviewer", ""),
                            body.get("corrected_text", ""), body.get("notes", ""))
                gold.write_gold(rows)
        except (ValueError, KeyError, TypeError) as error:
            return self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        self._json({"ok": True})

    def log_message(self, fmt, *args):  # keep the console quiet; audio requests are frequent
        if not self.path.startswith("/audio/"):
            super().log_message(fmt, *args)


PAGE = """<!doctype html>
<html lang="mr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold Review</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#666;--card:#fff;--line:#ddd;--ok:#1b7f3b;--warn:#a15c00;--bad:#b3261e;--acc:#2b59c3}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--muted:#aaa;--card:#222;--line:#3a3a3a}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:12px}
.meta{color:var(--muted);font-size:14px}
.bar{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:14px;margin-top:8px}
.bar b{font-variant-numeric:tabular-nums}
.cand{font-size:22px;line-height:1.7;margin:6px 0}
.asr{font-size:17px;line-height:1.6;color:var(--muted);margin:4px 0 0}
textarea{width:100%;box-sizing:border-box;font-size:20px;line-height:1.6;padding:8px;min-height:100px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px}
input{font-size:16px;padding:8px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
#notes{width:100%;box-sizing:border-box;margin-top:6px}
audio{width:100%;margin:8px 0}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
button{font-size:16px;padding:12px 16px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button:disabled{opacity:.4;cursor:not-allowed}
.ok{border-color:var(--ok);color:var(--ok)}.warn{border-color:var(--warn);color:var(--warn)}.bad{border-color:var(--bad);color:var(--bad)}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:0 8px;font-size:13px;margin-right:4px}
.flag{border-color:var(--warn);color:var(--warn)}
.done{border-color:var(--ok);color:var(--ok)}
#msg{min-height:1.5em;margin-top:8px}
kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
</style></head><body><main>
<header><div><b>Gold review</b> <span class="meta" id="scope"></span></div>
<label>Reviewer ID <input id="reviewer" size="8" placeholder="initials"></label></header>
<div class="bar" id="progress"></div>
<div class="bar meta" id="triage"></div>
<div class="card" id="card"><p>Loading…</p></div>
<p class="meta">Transcribe what is <b>spoken</b>, keeping Varhadi forms (काई, नाई, त्याले…). Do not convert them to standard Marathi.
Unclear audio: reject with a note, don't guess. The verify buttons unlock once the clip has played to the end.<br>
Keys: <kbd>V</kbd> verify · <kbd>E</kbd> edit (then <kbd>Ctrl</kbd>+<kbd>Enter</kbd> saves the correction) · <kbd>R</kbd> reject ·
<kbd>A</kbd> needs realignment · <kbd>Space</kbd> replay · <kbd>N</kbd>/<kbd>P</kbd> next/previous</p>
</main><script>
const $ = id => document.getElementById(id);
let items = [], idx = 0, maxPlayed = 0;
try { $('reviewer').value = localStorage.getItem('reviewer') || ''; } catch (e) {}
$('reviewer').onchange = () => { try { localStorage.setItem('reviewer', $('reviewer').value.trim()); } catch (e) {} };
function esc(s){ return String(s || '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
const pending = i => items[i] && items[i].review_status === 'pending_review';
async function load(goToId){
  const d = await (await fetch('/api/queue')).json();
  items = d.items;
  const p = d.progress, t = p.triage;
  $('scope').textContent = 'chapter: ' + d.chapter;
  $('progress').innerHTML = `<span>Reviewed: <b>${p.reviewed} / ${p.total}</b></span><span>Verified: <b>${p.verified}</b></span>
    <span>Corrected: <b>${p.corrected}</b></span><span>Rejected: <b>${p.rejected}</b></span>
    <span>Needs realignment: <b>${p.needs_realignment}</b></span><span>Remaining: <b>${p.remaining}</b></span>
    <span>Verified audio: <b>${p.verified_minutes} min</b></span>`;
  $('triage').innerHTML = `Triage (automatic, NOT verification): high-confidence candidates ${t.HIGH_CONFIDENCE_CANDIDATE} ·
    needs review ${t.NEEDS_REVIEW} · reject candidates ${t.REJECT_CANDIDATE}`;
  const at = goToId ? items.findIndex(x => x.segment_id === goToId) : -1;
  idx = at >= 0 ? at : Math.max(0, items.findIndex(x => x.review_status === 'pending_review'));
  show();
}
function show(){
  const s = items[idx];
  if (!s){ $('card').innerHTML = '<p>Nothing to review here.</p>'; return; }
  maxPlayed = 0;
  const dur = (s.end_sec - s.start_sec).toFixed(1);
  const flags = (s.triage_flags || '').split(';').filter(Boolean).map(f => `<span class="pill flag">${esc(f)}</span>`).join('');
  const state = s.review_status === 'pending_review' ? '' :
    `<span class="pill done">${esc(s.review_status)} · ${esc(s.decision)} by ${esc(s.reviewer)}</span>`;
  const shown = s.decision === 'needs_correction' ? s.corrected_text : s.candidate_text;
  $('card').innerHTML = `
    <div class="meta">${idx + 1} / ${items.length} · <b>${esc(s.segment_id)}</b> · ${esc(s.source_ref)} · ${s.start_sec}–${s.end_sec}s (<b>${dur}s</b>)
      ${s.source_url ? ` · <a href="${esc(s.source_url)}" target="_blank" rel="noopener">source</a>` : ''}</div>
    <div class="meta"><span class="pill">${esc(s.triage_status || 'untriaged')}</span>
      confidence <b>${esc(s.confidence_score || 'n/a')}</b> · alignment score ${esc(s.align_score || 'n/a')}
      (component ${esc(s.alignment_score || 'n/a')}) · ASR agreement ${esc(s.agreement_score || 'n/a')} ${flags} ${state}</div>
    <audio id="player" controls preload="auto" src="/audio/${encodeURIComponent(s.segment_id)}"></audio>
    <div class="meta">Candidate transcript (published text, automatic alignment):</div>
    <div class="cand">${esc(s.candidate_text || '(none: type what you hear below)')}</div>
    <div class="meta">Baseline ASR (pretrained Marathi; its spellings are not the target):</div>
    <div class="asr">${esc(s.raw_asr)}</div>
    <div class="meta" style="margin-top:10px">What is actually spoken (edit only if it differs):</div>
    <textarea id="corr">${esc(shown)}</textarea>
    <input id="notes" placeholder="notes (optional), e.g. 'translator note not read', 'unclear word'" value="${esc(s.review_notes)}">
    <div class="btns">
      <button class="ok" id="bV" disabled>V ✓ Verify as written</button>
      <button class="warn" id="bE" disabled>Ctrl+Enter ✎ Save correction</button>
      <button class="bad" id="bR">R ✗ Reject</button>
      <button id="bA">A ⟷ Needs realignment</button>
      <button id="bP">P ◀ Prev</button><button id="bN">N Next ▶</button>
    </div>
    <div id="msg"></div>`;
  const p = $('player');
  const unlock = () => { $('bV').disabled = !s.candidate_text; $('bE').disabled = false; };
  p.ontimeupdate = () => { if (p.duration){ maxPlayed = Math.max(maxPlayed, p.currentTime / p.duration); if (maxPlayed >= 0.95) unlock(); } };
  p.onended = () => { maxPlayed = 1; unlock(); };
  p.play().catch(() => {});  // autoplay where the browser allows it
  $('bV').onclick = () => send('VERIFIED');
  $('bE').onclick = () => send('CORRECTED');
  $('bR').onclick = () => send('REJECTED');
  $('bA').onclick = () => send('NEEDS_REALIGNMENT');
  $('bN').onclick = () => move(1);
  $('bP').onclick = () => move(-1);
}
function move(step){ const n = idx + step; if (n >= 0 && n < items.length){ idx = n; show(); } }
async function send(decision){
  const s = items[idx], reviewer = $('reviewer').value.trim();
  if (!reviewer){ $('msg').textContent = 'Enter your reviewer ID (initials) first.'; return; }
  const text = $('corr').value.trim();
  if (decision === 'CORRECTED' && text === (s.candidate_text || '').trim()){
    $('msg').textContent = 'Text unchanged from the candidate: use V (verify as written).'; return; }
  const r = await fetch('/api/review', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({segment_id: s.segment_id, decision, reviewer, corrected_text: text,
                          notes: $('notes').value, played_fraction: maxPlayed})});
  const d = await r.json();
  if (!r.ok){ $('msg').textContent = 'Not saved: ' + d.error; return; }
  // saved on the server; move on to the next PENDING segment after this one
  let next = items.findIndex((x, i) => i > idx && x.review_status === 'pending_review');
  const nextId = next >= 0 ? items[next].segment_id : null;
  await load(nextId);
}
document.addEventListener('keydown', e => {
  const typing = e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT';
  if (e.target.id === 'corr' && e.key === 'Enter' && e.ctrlKey){ e.preventDefault(); if (!$('bE').disabled) send('CORRECTED'); return; }
  if (typing) return;
  const p = $('player'); if (!p) return;
  const k = e.key.toLowerCase();
  if (e.key === ' '){ e.preventDefault(); p.currentTime = 0; p.play(); }
  else if (k === 'v' && !$('bV').disabled) send('VERIFIED');
  else if (k === 'e'){ e.preventDefault(); $('corr').focus(); }
  else if (k === 'r') send('REJECTED');
  else if (k === 'a') send('NEEDS_REALIGNMENT');
  else if (k === 'n') move(1);
  else if (k === 'p') move(-1);
});
load();
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chapter", help="only review segments from this chapter, e.g. JHN_001")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    Handler.chapter = args.chapter
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Review server on http://127.0.0.1:{args.port}  (chapter: {args.chapter or 'all'}; Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
