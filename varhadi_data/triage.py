"""Transparent, rule-based triage of aligned candidate segments.

Triage ranks candidates for HUMAN review and marks which ones could be used as
PSEUDO-LABELED training data. It NEVER marks a segment verified: only a human decision in
the review workflow (varhadi_data.gold.review) does that.

    confidence = sum(weight_i * component_i) - sum(artifact penalties), clipped to [0, 1]

Components, each in [0, 1] (ramps and weights in triage_config.json):
  alignment      ramp(align_score): mean CTC log-prob of the aligned tokens
  agreement      mean of ramp(CER candidate vs baseline ASR) and ramp((ins+del)/ref words)
  audio_quality  mean of ramps for silence ratio, clipping ratio and loudness
  duration       1 inside [min_ok, max_ok] s, linear down to 0 at the hard limits
  speaking_rate  same shape on candidate characters per second (a boundary-quality proxy:
                 too fast = text crammed into too little audio, too slow = extra audio)

Triage status:
  REJECT_CANDIDATE           base confidence (before artifact penalties) < thresholds.review,
                             or a hard failure (empty text, duration/rate outside hard limits,
                             mostly silence). Text artifacts never cause rejection: they are
                             correctable by a reviewer, so they only block HIGH_CONFIDENCE.
  HIGH_CONFIDENCE_CANDIDATE  confidence >= thresholds.high_confidence and no blocking flag
  NEEDS_REVIEW               everything else

Known bias: `agreement` compares the candidate with the *baseline* model's output, so high
confidence partly means "the baseline already transcribes this well". Never use triage to
skip human review of TEST data; it only orders the queue.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from asr_baseline.utils import strip_punctuation

CONFIG_PATH = Path(__file__).with_name("triage_config.json")
STATUSES = ("HIGH_CONFIDENCE_CANDIDATE", "NEEDS_REVIEW", "REJECT_CANDIDATE")
FRAME_SEC = 0.025
SILENCE_DBFS = -40.0


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ramp(value: float, bad: float, good: float) -> float:
    """Linear map: value at `bad` -> 0, at `good` -> 1, clipped. Works for either direction."""
    if value is None or value != value:  # None or NaN
        return 0.0
    position = (value - bad) / (good - bad)
    return max(0.0, min(1.0, position))


def plateau(value: float, limits: dict) -> float:
    """1 inside [min_ok, max_ok], linear to 0 at hard_min / hard_max."""
    if value < limits["min_ok"]:
        return ramp(value, limits["hard_min"], limits["min_ok"])
    if value > limits["max_ok"]:
        return ramp(value, limits["hard_max"], limits["max_ok"])
    return 1.0


def audio_signals(audio, sr: int) -> dict:
    import numpy as np

    if len(audio) == 0:
        return {"silence_ratio": 1.0, "clip_ratio": 0.0, "rms_dbfs": -120.0}
    frame = max(1, int(FRAME_SEC * sr))
    usable = len(audio) // frame * frame or len(audio)
    frames = audio[:usable].reshape(-1, frame) if usable >= frame else audio[None, :]
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    with np.errstate(divide="ignore"):
        frame_db = 20 * np.log10(np.maximum(frame_rms, 1e-12))
    rms = float(np.sqrt(np.mean(audio**2)))
    return {
        "silence_ratio": float(np.mean(frame_db < SILENCE_DBFS)),
        "clip_ratio": float(np.mean(np.abs(audio) >= 0.999)),
        "rms_dbfs": 20 * float(np.log10(max(rms, 1e-12))),
    }


def text_signals(candidate: str, asr: str) -> dict:
    import jiwer

    ref, hyp = strip_punctuation(candidate), strip_punctuation(asr)
    if not ref or not hyp:
        return {"cer_vs_asr": 1.0, "insert_delete_rate": 1.0, "substitutions": 0, "deletions": 0, "insertions": 0}
    words = jiwer.process_words(ref, hyp)
    n = words.hits + words.substitutions + words.deletions
    return {
        "cer_vs_asr": jiwer.cer(ref, hyp),
        "insert_delete_rate": (words.insertions + words.deletions) / max(n, 1),
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
    }


def artifact_flags(candidate: str, asr: str, n_headings: int) -> list[str]:
    flags = []
    if not candidate.strip():
        flags.append("empty_candidate")
    if not asr.strip():
        flags.append("empty_asr")
    if re.search(r"[()\[\]{}]", candidate):
        flags.append("parentheses")  # translator notes / footnotes usually not read aloud
    if re.search(r"[0-9A-Za-z]", candidate):
        flags.append("digits_or_latin")
    if re.search(r"(?:^|\s)(\S+)(?:\s+\1){2,}(?:\s|$)", strip_punctuation(candidate)):
        flags.append("repeated_word")
    if n_headings > 0:
        flags.append("contains_heading")  # section headings: usually read, but not always
    return flags


def score_segment(row: dict, audio_sig: dict | None, n_headings: int, config: dict) -> dict:
    """All signals, components, confidence and triage status for one gold row."""
    ramps, weights = config["ramps"], config["weights"]
    candidate, asr = row.get("candidate_text", ""), row.get("raw_asr", "")
    try:
        duration = float(row["end_sec"]) - float(row["start_sec"])
    except (KeyError, TypeError, ValueError):
        duration = 0.0
    try:
        align = float(row.get("align_score", ""))
    except (TypeError, ValueError):
        align = None  # missing score -> component 0, flagged

    text = text_signals(candidate, asr)
    chars = len(strip_punctuation(candidate).replace(" ", ""))
    rate = chars / duration if duration > 0 else 0.0
    flags = artifact_flags(candidate, asr, n_headings)
    if align is None:
        flags.append("missing_align_score")
    if audio_sig is None:
        flags.append("audio_unavailable")

    components = {
        "alignment": ramp(align, **ramps["align_score"]),
        "agreement": (ramp(text["cer_vs_asr"], **ramps["cer_vs_asr"])
                      + ramp(text["insert_delete_rate"], **ramps["insert_delete_rate"])) / 2,
        "audio_quality": 0.0 if audio_sig is None else (
            ramp(audio_sig["silence_ratio"], **ramps["silence_ratio"])
            + ramp(audio_sig["clip_ratio"], **ramps["clip_ratio"])
            + min(ramp(audio_sig["rms_dbfs"], **ramps["rms_dbfs_low"]),
                  ramp(audio_sig["rms_dbfs"], **ramps["rms_dbfs_high"]))) / 3,
        "duration": plateau(duration, config["duration_sec"]),
        "speaking_rate": plateau(rate, config["chars_per_sec"]),
    }
    if components["duration"] == 0.0:
        flags.append("duration_out_of_range")
    if components["speaking_rate"] == 0.0:
        flags.append("speaking_rate_out_of_range")
    if audio_sig and audio_sig["silence_ratio"] >= ramps["silence_ratio"]["bad"]:
        flags.append("mostly_silence")

    penalty = sum(config["artifact_penalties"].get(flag, 0.0) for flag in flags)
    weighted = sum(weights[name] * value for name, value in components.items()) / sum(weights.values())
    base = max(0.0, min(1.0, weighted))
    confidence = max(0.0, min(1.0, weighted - penalty))

    hard_fail = {"empty_candidate", "duration_out_of_range", "speaking_rate_out_of_range", "mostly_silence"} & set(flags)
    thresholds = config["thresholds"]
    correctable = set(flags) & set(config.get("never_reject_flags", []))
    if hard_fail or (base < thresholds["review"] and not correctable):
        status = "REJECT_CANDIDATE"
    elif confidence >= thresholds["high_confidence"] and not set(flags) & set(config["blocking_flags"]):
        status = "HIGH_CONFIDENCE_CANDIDATE"
    else:
        status = "NEEDS_REVIEW"

    return {
        "duration_sec": round(duration, 3),
        "chars_per_sec": round(rate, 2),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in text.items()},
        **({k: round(v, 4) for k, v in audio_sig.items()} if audio_sig else {}),
        **{f"{name}_score": round(value, 4) for name, value in components.items()},
        "base_confidence": round(base, 4),
        "artifact_penalty": round(penalty, 4),
        "confidence_score": round(confidence, 4),
        "triage_flags": ";".join(flags),
        "triage_status": status,
    }
