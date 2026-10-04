from __future__ import annotations

import csv
from pathlib import Path

from .config import ERROR_ANALYSIS_PATH, PER_SAMPLE_METRICS_PATH
from .utils import strip_punctuation, write_csv

FIELDS = ["audio_path", "reference", "prediction", "wer", "cer", "error_summary", "manual_category", "notes"]

HIGH_WER_THRESHOLD = 0.5

# Filled in later by a human reviewing error_summary + reference/prediction pairs.
# Never auto-assigned: an error must not be labeled "Varhadi vocabulary" etc. without review.
MANUAL_CATEGORIES = [
    "varhadi_to_standard_marathi",  # Varhadi form recognised as its standard-Marathi equivalent
    "pronunciation",
    "vocabulary",  # unseen/rare vocabulary
    "grammatical_dialectal",
    "code_switching",
    "agricultural_term",
    "proper_noun",
    "deletion",
    "insertion",
    "noise",
    "segmentation",
    "transcript_issue",
    "unknown",
]

OP_FIELDS = [
    "type", "reference_word", "hypothesis_word", "count",
    "similarity", "heuristic_note", "manual_category", "notes",
]
SIMILAR_FORM = 0.6  # heuristic threshold on difflib character similarity, not a validated cut-off


def summarize_errors(reference: str, prediction: str) -> str:
    import jiwer

    output = jiwer.process_words(reference, prediction)
    return (
        f"substitutions={output.substitutions} "
        f"deletions={output.deletions} "
        f"insertions={output.insertions} "
        f"hits={output.hits}"
    )


def build_error_analysis(per_sample_rows: list[dict]) -> list[dict]:
    rows = []
    for row in per_sample_rows:
        wer = float(row["wer"])
        summary = summarize_errors(row["reference"], row["prediction"])
        if wer >= HIGH_WER_THRESHOLD:
            summary += " [HIGH_WER]"
        rows.append(
            {
                "audio_path": row["audio_path"],
                "reference": row["reference"],
                "prediction": row["prediction"],
                "wer": row["wer"],
                "cer": row["cer"],
                "error_summary": summary,
                "manual_category": "",
                "notes": "",
            }
        )
    return rows


def edit_operations(rows: list[dict]) -> list[dict]:
    """Aggregate word-level substitutions/deletions/insertions over (reference, prediction)
    pairs, most frequent first. Uses the same punctuation stripping as scoring.

    `heuristic_note` is a deterministic string-similarity tag, NOT a diagnosis: a
    "similar form" substitution (e.g. पयले->पहिले) suggests a spelling/dialect/pronunciation
    variant; "different word" suggests vocabulary or a misrecognition. `manual_category`
    is left for a human (see MANUAL_CATEGORIES).
    """
    from collections import Counter
    from difflib import SequenceMatcher

    import jiwer

    counts = Counter()
    for row in rows:
        out = jiwer.process_words(strip_punctuation(row["reference"]), strip_punctuation(row["prediction"]))
        ref, hyp = out.references[0], out.hypotheses[0]
        for chunk in out.alignments[0]:
            refs = ref[chunk.ref_start_idx : chunk.ref_end_idx]
            hyps = hyp[chunk.hyp_start_idx : chunk.hyp_end_idx]
            if chunk.type == "substitute":
                counts.update(("substitution", r, h) for r, h in zip(refs, hyps))
            elif chunk.type == "delete":
                counts.update(("deletion", r, "") for r in refs)
            elif chunk.type == "insert":
                counts.update(("insertion", "", h) for h in hyps)
    out_rows = []
    for (kind, ref_word, hyp_word), count in counts.most_common():
        similarity = SequenceMatcher(None, ref_word, hyp_word).ratio() if kind == "substitution" else None
        note = ""
        if similarity is not None:
            note = "similar form (heuristic)" if similarity >= SIMILAR_FORM else "different word (heuristic)"
        out_rows.append({
            "type": kind,
            "reference_word": ref_word,
            "hypothesis_word": hyp_word,
            "count": count,
            "similarity": f"{similarity:.2f}" if similarity is not None else "",
            "heuristic_note": note,
            "manual_category": "",
            "notes": "",
        })
    return out_rows


def run(per_sample_path: Path = PER_SAMPLE_METRICS_PATH, out_path: Path = ERROR_ANALYSIS_PATH) -> None:
    with per_sample_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    write_csv(out_path, build_error_analysis(rows), FIELDS)


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
