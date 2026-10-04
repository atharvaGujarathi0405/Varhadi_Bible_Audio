"""Recording-session sheets for native Varhadi agricultural speech (P1 data gap).

Workflow (docs/data_collection.md):
    prompt draft (standard Marathi) -> native speaker rewrites read prompts into Varhadi
    -> session sheet -> recording -> register-recording (consent, pseudonym)
    -> exact transcription + listening review (review_server) -> verified gold

Elicitation prompts are questions; the speaker answers freely, so they have no expected
text and are always transcribed after recording. Read prompts are only usable once a
native speaker has filled in `varhadi_text`: the standard-Marathi draft is never read out
as if it were Varhadi.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROMPTS = Path("prompts/recording_prompts.csv")
CROP_SCOPE = Path("prompts/crop_scope.json")
SHEET_FIELDS = ["order", "prompt_id", "type", "topic", "crop", "show_text", "expected_read_text", "recording_file", "notes"]


def read_prompts(data_dir: Path) -> list[dict]:
    with (data_dir / PROMPTS).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def crop_scope(data_dir: Path) -> tuple[list[str], list[str]]:
    scope = json.loads((data_dir / CROP_SCOPE).read_text(encoding="utf-8"))
    return scope["crops"], scope.get("always_include", ["general"])


def session_sheet(prompts: list[dict], crops: list[str], always: list[str]) -> tuple[list[dict], list[str]]:
    """Return (sheet rows, prompt_ids skipped because their Varhadi rewrite is missing)."""
    sheet, skipped = [], []
    for prompt in prompts:
        if prompt["crop"] not in crops and prompt["crop"] not in always:
            continue
        if prompt["type"] == "read" and not prompt["varhadi_text"].strip():
            skipped.append(prompt["prompt_id"])
            continue
        is_read = prompt["type"] == "read"
        sheet.append({
            "order": len(sheet) + 1,
            "prompt_id": prompt["prompt_id"],
            "type": prompt["type"],
            "topic": prompt["topic"],
            "crop": prompt["crop"],
            "show_text": prompt["varhadi_text"] if is_read else prompt["text"],
            "expected_read_text": prompt["varhadi_text"] if is_read else "",
            "recording_file": "",
            "notes": "",
        })
    return sheet, skipped
