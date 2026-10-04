"""CTC forced alignment of published VAHNT text to chapter audio (option b).

Produces CANDIDATE segment-level (audio span, text) pairs. They are not verified ground
truth: they enter the gold workflow as pending_review and need a human check.

Method:
1. Frame-level CTC log-probs from the pretrained model (Marathi softmax of the
   multilingual head), computed chunk by chunk with the same silence cuts as inference,
   so a full chapter fits in memory. Each frame keeps its absolute time.
2. Viterbi CTC alignment of the whole chapter's token sequence (ctc_align below).
   The audio may contain speech that is not in the text: the book/chapter announcement at
   the start, and anything after the last verse. Frames before the first token and after
   the last token may stay unaligned, scored as the best token at that frame, which makes
   them neutral, so the text lands where it actually matches. Speech missing from the
   text in the MIDDLE of a chapter is not modelled; it lowers that segment's score.
3. Units (section headings, verses) are grouped into segments of at most `max_sec`,
   cutting between units at the midpoint of the gap between their aligned tokens.

Scores, and the bias they can introduce:
- `align_score`: mean log-prob of the aligned tokens over their frames (closer to 0 =
  more confident). Use it to prioritise human review, NOT to silently filter the test set.
- `raw_asr` / `cer_vs_raw_asr`: greedy baseline decode of the span and its CER against
  the published text. Informational only. Filtering on it would keep exactly the segments
  the baseline already transcribes well and bias the baseline-vs-adapted comparison.
"""
from __future__ import annotations

import numpy as np

from asr_baseline.utils import strip_punctuation

NEG = -1e30


clean_for_alignment = strip_punctuation  # same normalisation as scoring (asr_baseline.utils)


def ctc_align(log_probs: np.ndarray, tokens: list[int], blank: int) -> list[tuple[int, int]]:
    """Viterbi CTC alignment with free (unaligned) audio before and after the text.

    log_probs: (T, V) frame log-probabilities. Returns an inclusive (start_frame, end_frame)
    span per token.
    """
    T, L = len(log_probs), len(tokens)
    if L == 0:
        raise ValueError("no tokens to align")
    ext = np.full(2 * L + 1, blank)
    ext[1::2] = tokens
    S = len(ext)
    can_skip = np.zeros(S, bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]  # token -> next token directly, unless repeated
    needed = L + int(np.sum(ext[3::2] == ext[1:-2:2]))
    if T < needed:
        raise ValueError(f"audio too short: {T} frames for {L} tokens")

    garbage = log_probs.max(axis=1)
    back = np.zeros((T, S), np.int8)  # 0 stay, 1 from s-1, 2 from s-2, 3 entered from free prefix
    alpha = np.full(S, NEG)
    alpha[:2] = log_probs[0, ext[:2]]
    back[0, :2] = 3
    prefix = 0.0  # score of leaving frames 0..t-1 unaligned
    suffix, suffix_entry = NEG, None  # best score with frames after the text unaligned
    for t in range(1, T):
        prefix += garbage[t - 1]
        exit_score = max(alpha[-1], alpha[-2])
        if exit_score + garbage[t] > suffix + garbage[t]:
            suffix, suffix_entry = exit_score + garbage[t], (t - 1, S - 1 if alpha[-1] >= alpha[-2] else S - 2)
        else:
            suffix += garbage[t]
        one = np.concatenate(([NEG], alpha[:-1]))
        two = np.where(can_skip, np.concatenate(([NEG, NEG], alpha[:-2])), NEG)
        candidates = np.stack([alpha, one, two])
        choice = candidates.argmax(axis=0)
        best = candidates[choice, np.arange(S)]
        for s in (0, 1):
            if prefix > best[s]:
                best[s], choice[s] = prefix, 3
        back[t] = choice
        alpha = best + log_probs[t, ext]

    end_t, end_s = T - 1, S - 1 if alpha[-1] >= alpha[-2] else S - 2
    if suffix_entry is not None and suffix > max(alpha[-1], alpha[-2]):
        end_t, end_s = suffix_entry

    spans: list[list[int]] = [[-1, -1] for _ in range(L)]
    s, t = end_s, end_t
    while True:
        if s % 2:  # odd states are tokens
            k = (s - 1) // 2
            spans[k][0] = t
            if spans[k][1] < 0:
                spans[k][1] = t
        step = int(back[t, s])  # plain int: NumPy 2 would keep s as int8 and overflow
        if step == 3:
            break
        s -= step
        t -= 1
    return [tuple(span) for span in spans]


def group_units(unit_spans: list[tuple[float, float]], max_sec: float) -> list[tuple[int, int, float, float]]:
    """Group consecutive units into segments <= max_sec (a single longer unit stays alone).

    unit_spans: aligned (start_sec, end_sec) per unit, in order. Returns
    (first_unit, last_unit, seg_start_sec, seg_end_sec), with cuts at the midpoint of
    the gap between neighbouring units, so no audio of a unit is cut off.
    """
    groups, first = [], 0
    for i in range(1, len(unit_spans) + 1):
        if i == len(unit_spans) or unit_spans[i][1] - unit_spans[first][0] > max_sec:
            groups.append([first, i - 1])
            first = i
    out = []
    for index, (a, b) in enumerate(groups):
        start = unit_spans[a][0] if index == 0 else (unit_spans[a - 1][1] + unit_spans[a][0]) / 2
        end = unit_spans[b][1] if index == len(groups) - 1 else (unit_spans[b][1] + unit_spans[b + 1][0]) / 2
        out.append((a, b, start, end))
    return out


def chapter_log_probs(loaded, wav_path, settings, language: str = "mr"):
    """(log_probs (T, V) numpy, frame_start_times (T,), frame_sec) for a 16 kHz mono WAV,
    computed per silence-cut chunk so long chapters fit in memory."""
    import soundfile as sf
    import torch

    from asr_baseline.chunking import plan_chunks

    model = loaded.model
    frame_sec = model.cfg.preprocessor.window_stride * model.cfg.encoder.subsampling_factor
    audio, sr = sf.read(str(wav_path), dtype="float32")
    parts, times = [], []
    for start, end in plan_chunks(audio, sr, settings):
        signal = torch.tensor(audio[start:end]).unsqueeze(0)
        with torch.no_grad():
            features, lengths = model.preprocessor(input_signal=signal, length=torch.tensor([end - start]))
            encoded, enc_lengths = model.encoder(audio_signal=features, length=lengths)
            log_probs = model.ctc_decoder(encoder_output=encoded, language_ids=[language])
        frames = int(enc_lengths[0])
        parts.append(log_probs[0, :frames].numpy())
        times.append(start / sr + np.arange(frames) * frame_sec)
    return np.concatenate(parts), np.concatenate(times), frame_sec


def greedy_text(log_probs: np.ndarray, blank: int, ids_to_text) -> str:
    best = log_probs.argmax(axis=1)
    ids = [int(t) for i, t in enumerate(best) if t != blank and (i == 0 or t != best[i - 1])]
    return ids_to_text(ids) if ids else ""


def align_chapter(loaded, wav_path, units: list[dict], settings, max_sec: float = 20.0, language: str = "mr"):
    """Return segment dicts for one chapter. `units` come from vahnt_text.parse_chapter_html."""
    import jiwer

    tokenizer = loaded.model.tokenizer
    log_probs, times, frame_sec = chapter_log_probs(loaded, wav_path, settings, language)
    blank = log_probs.shape[1] - 1

    unit_tokens = [tokenizer.text_to_ids(clean_for_alignment(unit["text"]), language) for unit in units]
    tokens = [tok for ids in unit_tokens for tok in ids]
    spans = ctc_align(log_probs, tokens, blank)

    unit_spans, unit_scores, offset = [], [], 0
    for ids in unit_tokens:
        mine = spans[offset : offset + len(ids)]
        unit_spans.append((float(times[mine[0][0]]), float(times[mine[-1][1]] + frame_sec)))
        unit_scores.append([float(log_probs[a : b + 1, tok].mean()) for (a, b), tok in zip(mine, ids)])
        offset += len(ids)

    segments = []
    for a, b, start, end in group_units(unit_spans, max_sec):
        text = " ".join(unit["text"] for unit in units[a : b + 1])
        frames = (times >= start) & (times < end)
        raw = greedy_text(log_probs[frames], blank, lambda ids: tokenizer.ids_to_text(ids, language)).strip()
        reference = clean_for_alignment(text)
        refs = [unit["ref"] for unit in units[a : b + 1] if unit["ref"]]
        segments.append(
            {
                "start_sec": f"{start:.3f}",
                "end_sec": f"{end:.3f}",
                "units": f"{refs[0]}..{refs[-1]}" if refs else "heading",
                "n_headings": sum(unit["kind"] == "heading" for unit in units[a : b + 1]),
                "text": text,
                "align_score": f"{np.mean([s for scores in unit_scores[a : b + 1] for s in scores]):.4f}",
                "raw_asr": raw,
                "cer_vs_raw_asr": f"{jiwer.cer(reference, clean_for_alignment(raw)):.4f}" if raw else "",
            }
        )
    return segments
