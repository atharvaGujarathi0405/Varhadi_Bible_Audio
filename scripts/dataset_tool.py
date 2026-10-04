#!/usr/bin/env python3
"""KisaanDost Varhadi dataset tooling (P1). See docs/dataset.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_baseline.config import AUDIO_DIR, CHUNKS_DIR
from asr_baseline.utils import write_csv
from varhadi_data import gold
from varhadi_data.prompts import SHEET_FIELDS
from varhadi_data.config import ALIGNMENTS, DATA_DIR, SUBDIRS, VAHNT_TEXT
from varhadi_data.manifest import (
    ALLOWED,
    assign_splits,
    read_manifest,
    speaker_leaks,
    upsert,
    vahnt_rows,
    validate_row,
    write_manifest,
    write_splits,
)
from varhadi_data.recordings import register_recording, withdraw_speaker


def cmd_init(_):
    for name in SUBDIRS:
        (DATA_DIR / name).mkdir(parents=True, exist_ok=True)
    print(f"Created {', '.join(SUBDIRS)} under {DATA_DIR}")


def cmd_register_vahnt(_):
    from asr_baseline.audio_utils import get_duration

    new = vahnt_rows(AUDIO_DIR, get_duration)
    write_manifest(upsert(read_manifest(), new))
    print(f"Registered {len(new)} VAHNT chapters (speaker_id=unknown, transcript NOT AVAILABLE)")


def cmd_register_recording(args):
    row = register_recording(
        args.audio,
        args.speaker_key,
        args.district,
        args.domain,
        args.consent,
        age_group=args.age_group,
        gender=args.gender,
        prompt_text=args.prompt_text,
        prompt_id=args.prompt_id,
    )
    print(f"Registered {row['utterance_id']} ({row['duration_sec']}s, quality={row['recording_quality']})")


def cmd_withdraw(args):
    removed = withdraw_speaker(args.speaker_id)
    print(f"Withdrew {args.speaker_id}: {removed} manifest rows and all stored audio deleted")


def cmd_split(args):
    rows = assign_splits(read_manifest(), tuple(args.ratios), args.seed)
    write_manifest(rows)
    print(json.dumps(write_splits(rows), indent=2))


def cmd_check(_):
    rows = read_manifest()
    errors = [error for row in rows for error in validate_row(row)]
    errors += gold.validate_gold(gold.read_gold())
    errors += [f"speaker {s} in splits {sorted(v)}" for s, v in speaker_leaks(rows).items()]
    for error in errors:
        print("ERROR", error)
    print(f"{len(rows)} manifest rows, {len(errors)} errors")
    return 1 if errors else 0


def cmd_stats(_):
    rows, gold_rows = read_manifest(), gold.read_gold()
    by = lambda field: {k: sum(r[field] == k for r in rows) for k in sorted({r[field] for r in rows})}
    print(json.dumps({
        "utterances": len(rows),
        "minutes": round(sum(float(r["duration_sec"] or 0) for r in rows) / 60, 1),
        "known_speakers": len({r["speaker_id"] for r in rows} - {"unknown"}),
        "by_domain": by("domain"),
        "by_transcript_status": by("transcript_status"),
        "by_split": by("split"),
        "gold_segments": {s: sum(r["review_status"] == s for r in gold_rows) for s in sorted(gold.STATUSES)},
    }, indent=2, ensure_ascii=False))


def cmd_session_sheet(args):
    from varhadi_data.prompts import crop_scope, read_prompts, session_sheet

    crops, always = crop_scope(DATA_DIR)
    sheet, skipped = session_sheet(read_prompts(DATA_DIR), args.crops or crops, always)
    write_csv(args.out, sheet, SHEET_FIELDS)
    print(f"{len(sheet)} prompts -> {args.out} (crops: {', '.join(args.crops or crops)} + {', '.join(always)})")
    if skipped:
        print(f"Skipped {len(skipped)} read prompts with no native Varhadi rewrite yet: {' '.join(skipped)}")


def cmd_gold_seed(_):
    rows = gold.read_gold()
    seeded = gold.seed_from_chunks(rows, CHUNKS_DIR, AUDIO_DIR)
    gold.write_gold(seeded)
    print(f"Added {len(seeded) - len(rows)} pending segments from {CHUNKS_DIR}")


def cmd_gold_import_alignments(_):
    rows = gold.read_gold()
    imported = gold.import_alignments(rows, DATA_DIR / ALIGNMENTS, DATA_DIR / VAHNT_TEXT)
    gold.write_gold(imported)
    print(f"Queued {len(imported) - len(rows)} aligned VAHNT segments for review (pending_review)")


def cmd_gold_review(args):
    rows = gold.review(gold.read_gold(), args.segment_id, args.decision, args.reviewer, args.corrected)
    gold.write_gold(rows)
    print(f"{args.segment_id} -> {args.decision} ({gold.STATUS_FOR_DECISION[args.decision]})")


def cmd_gold_export(args):
    count = gold.export_gold(chapters=args.chapters)
    scope = f" from {', '.join(args.chapters)}" if args.chapters else ""
    print(f"Exported {count} verified segments{scope}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create the data/ layout").set_defaults(fn=cmd_init)
    sub.add_parser("register-vahnt", help="add downloaded VAHNT chapters to the manifest").set_defaults(fn=cmd_register_vahnt)

    rec = sub.add_parser("register-recording", help="add one consented native recording")
    rec.add_argument("--audio", type=Path, required=True)
    rec.add_argument("--speaker-key", required=True, help="private identifier; hashed, never stored")
    rec.add_argument("--district", required=True)
    rec.add_argument("--domain", required=True, choices=sorted(ALLOWED["domain"]))
    rec.add_argument("--consent", required=True, choices=["given", "refused"])
    rec.add_argument("--age-group", default="", help="only if the speaker consented")
    rec.add_argument("--gender", default="", help="only if the speaker consented")
    rec.add_argument("--prompt-text", default="", help="Varhadi text the speaker was asked to read (goes to review)")
    rec.add_argument("--prompt-id", default="", help="prompt_id from the session sheet, e.g. R01 or E03")
    rec.set_defaults(fn=cmd_register_recording)

    wd = sub.add_parser("withdraw-speaker", help="delete all data of a speaker who withdrew consent")
    wd.add_argument("speaker_id")
    wd.set_defaults(fn=cmd_withdraw)

    sp = sub.add_parser("split", help="speaker-level train/validation/test split")
    sp.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    sp.add_argument("--seed", type=int, default=13)
    sp.set_defaults(fn=cmd_split)

    sub.add_parser("check", help="validate manifest, gold transcripts and speaker leakage").set_defaults(fn=cmd_check)
    sub.add_parser("stats", help="dataset summary").set_defaults(fn=cmd_stats)
    sub.add_parser("gold-seed", help="queue ASR chunks for human review").set_defaults(fn=cmd_gold_seed)

    ss = sub.add_parser("session-sheet", help="recording-session sheet for the crops in scope")
    ss.add_argument("--out", type=Path, required=True)
    ss.add_argument("--crops", nargs="+", help="override data/prompts/crop_scope.json")
    ss.set_defaults(fn=cmd_session_sheet)

    sub.add_parser(
        "gold-import-alignments", help="queue aligned VAHNT segments (published text) for review"
    ).set_defaults(fn=cmd_gold_import_alignments)

    rv = sub.add_parser(
        "gold-review", help="record a human review of one segment (prefer scripts/review_server.py, which plays the clip)"
    )
    rv.add_argument("--segment-id", required=True)
    rv.add_argument("--decision", required=True, choices=sorted(gold.STATUS_FOR_DECISION))
    rv.add_argument("--reviewer", required=True, help="reviewer pseudonym/initials")
    rv.add_argument("--corrected", default="", help="what the reviewer actually heard (needs_correction only)")
    rv.set_defaults(fn=cmd_gold_review)

    ex = sub.add_parser("gold-export", help="export verified segments as an ASR evaluation set")
    ex.add_argument("--chapters", nargs="+", help="only these source chapters, e.g. the held-out test chapters")
    ex.set_defaults(fn=cmd_gold_export)

    args = parser.parse_args()
    try:
        return args.fn(args) or 0
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
