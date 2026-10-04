#!/usr/bin/env python3
"""Build NeMo train/val/test manifests for Varhadi adaptation (P2 data prep; CPU, no model).

    python experiments/varhadi_adaptation/build_manifests.py \
        --test-chapters JHN_001 --val-chapters JHN_002 [--include-unverified] [--copy-audio]

Source of truth: data/transcripts/gold_transcripts.csv (candidates, triage, human reviews).

GOLD (human-verified: review decision VERIFIED/correct or CORRECTED/needs_correction) is the
only data used by default. Splits are held out at CHAPTER level (VAHNT speaker identity is
unknown, so a speaker-level split is impossible):
  test  = gold segments of --test-chapters (never anything else, under any flag)
  val   = gold segments of --val-chapters
  train = gold segments of every other chapter
Rejected / needs_realignment segments are never used.

--include-unverified adds PSEUDO_LABELED data for semi-supervised experiments: pending
segments that triage marked HIGH_CONFIDENCE_CANDIDATE, labelled with their candidate text.
They go to SEPARATE files (train_pseudo_labeled_manifest.json, val_pseudo_labeled_manifest.json)
and are never mixed into the gold manifests: to train on both, pass both files explicitly
(NeMo accepts a list of manifests) and report it.

Each JSONL line: {"audio_filepath", "offset", "duration", "text", "lang": "mr", "segment_id",
"label_source"}. `offset` points into the chapter's 16 kHz WAV; `lang` is required by the
AI4Bharat fork (return_language_id). Text = punctuation stripped (same as scoring), nothing else.

Output (gitignored, contains transcripts): data/processed/p2_bundle/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from asr_baseline.config import ASR_AUDIO_DIR, MODEL_NAME
from asr_baseline.utils import strip_punctuation
from varhadi_data import gold
from varhadi_data.config import DATA_DIR

BUNDLE = DATA_DIR / "processed" / "p2_bundle"
LANG = "mr"  # the checkpoint's Marathi softmax; Varhadi is written in Devanagari Marathi script
PSEUDO = "PSEUDO_LABELED"
GOLD_LABELS = {"correct": "GOLD_VERIFIED", "needs_correction": "GOLD_CORRECTED"}


def is_gold(row: dict) -> bool:
    return row["review_status"] == "verified" and row["decision"] in GOLD_LABELS


def is_pseudo_eligible(row: dict) -> bool:
    return (row["review_status"] == "pending_review" and row.get("triage_status") == "HIGH_CONFIDENCE_CANDIDATE"
            and bool(row["candidate_text"].strip()))


def split_rows(rows, test_chapters, val_chapters, include_unverified=False) -> dict[str, list[dict]]:
    out = {"train": [], "val": [], "test": [], "train_pseudo_labeled": [], "val_pseudo_labeled": []}
    for row in rows:
        if not row["start_sec"]:
            continue
        chapter = row["source_chapter"]
        if chapter in test_chapters:
            if is_gold(row):
                out["test"].append(row)  # test = gold only, regardless of flags
            continue
        split = "val" if chapter in val_chapters else "train"
        if is_gold(row):
            out[split].append(row)
        elif include_unverified and is_pseudo_eligible(row):
            out[f"{split}_pseudo_labeled"].append(row)
    held_out = set(test_chapters) | set(val_chapters)
    leaked = {r["source_chapter"] for key in ("train", "train_pseudo_labeled") for r in out[key]} & held_out
    assert not leaked, f"held-out chapters leaked into train: {leaked}"
    return out


def manifest_line(row: dict, audio_prefix: str) -> dict:
    if is_gold(row):
        text, label = gold.reference(row), GOLD_LABELS[row["decision"]]
    else:
        text, label = row["candidate_text"], PSEUDO
    start, end = float(row["start_sec"]), float(row["end_sec"])
    return {
        "audio_filepath": f"{audio_prefix}/{row['source_chapter']}.wav",
        "offset": round(start, 3),
        "duration": round(end - start, 3),
        "text": strip_punctuation(text),
        "lang": LANG,
        "segment_id": row["segment_id"],
        "label_source": label,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-chapters", nargs="+", default=["JHN_001"])
    parser.add_argument("--val-chapters", nargs="+", default=["JHN_002"])
    parser.add_argument("--include-unverified", "--allow-unverified-train", dest="include_unverified",
                        action="store_true",
                        help="also write PSEUDO_LABELED manifests (HIGH_CONFIDENCE_CANDIDATE, unverified) as separate files")
    parser.add_argument("--copy-audio", action="store_true", help="copy the needed chapter WAVs into the bundle")
    parser.add_argument("--out", type=Path, default=BUNDLE)
    args = parser.parse_args(argv)

    splits = split_rows(gold.read_gold(), args.test_chapters, args.val_chapters, args.include_unverified)
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    info = {"date": date.today().isoformat(), "base_model": MODEL_NAME, "lang": LANG,
            "test_chapters": args.test_chapters, "val_chapters": args.val_chapters,
            "include_unverified": args.include_unverified, "text_normalisation": "strip_punctuation only",
            "speakers": "unknown (VAHNT narrator identity not available)", "splits": {}}
    for name, rows in splits.items():
        if name.endswith("_pseudo_labeled") and not args.include_unverified:
            (args.out / "manifests" / f"{name}_manifest.json").unlink(missing_ok=True)  # never leave stale pseudo files
            continue
        lines = [manifest_line(row, "audio") for row in rows]
        with (args.out / "manifests" / f"{name}_manifest.json").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
        info["splits"][name] = {
            "segments": len(lines),
            "hours": round(sum(line["duration"] for line in lines) / 3600, 3),
            "chapters": len({row["source_chapter"] for row in rows}),
            "label_source": dict(Counter(line["label_source"] for line in lines)),
        }
    if args.copy_audio:
        (args.out / "audio").mkdir(exist_ok=True)
        for chapter in sorted({row["source_chapter"] for rows in splits.values() for row in rows}):
            shutil.copy2(ASR_AUDIO_DIR / f"{chapter}.wav", args.out / "audio" / f"{chapter}.wav")
    (args.out / "bundle_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info["splits"], indent=2))
    if not splits["test"]:
        print("WARNING: test manifest is EMPTY - no human-verified segments in the test chapters yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
