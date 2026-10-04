# Native Varhadi agricultural speech: collection workflow

**Why this is needed.** All speech data so far is the VAHNT audio Bible: general Varhadi
*scripture* read by unknown narrator(s). It is enough to build and test the ASR pipeline,
but it is **not** agricultural speech. It contains no crop, pest, fertiliser or scheme
vocabulary and no farmer speakers. The final KisaanDost ASR needs consented recordings of
native Varhadi speakers talking about farming. **None exist yet.**

## Pipeline

```
prompt draft (standard Marathi, data/prompts/recording_prompts.csv)
   │  native Varhadi speaker rewrites READ prompts → varhadi_text, rewritten_by
   ▼
session sheet       python scripts/dataset_tool.py session-sheet --out session_01.csv [--crops cotton soybean]
   │
   ▼
consent + recording (phone or recorder; quiet place; one prompt per file)
   │
   ▼
intake              python scripts/dataset_tool.py register-recording --audio rec.m4a --speaker-key "<private>"
   │                    --district Amravati --domain agriculture --consent given --prompt-id R02
   │                    [--prompt-text "<varhadi_text of R02>"] [--age-group ..] [--gender ..]
   │                → pseudonymous VH_S###, original kept byte-for-byte + 16 kHz copy, quality flags
   ▼
exact transcription + listening review
   │                python scripts/review_server.py   (play clip → correct / needs_correction / reject)
   ▼
verified gold       → speaker-level split → evaluation / training manifests
```

## Prompts

`data/prompts/recording_prompts.csv` has 27 prompts, each tagged with a `crop` (cotton,
soybean, orange, tur, or general):

| Type | Count | How it is used |
|---|---|---|
| elicitation (E01–E11) | 11 | A question; the speaker answers freely in their own Varhadi. There is no expected text, so the answer is transcribed after recording. Usable now. |
| read (R01–R16) | 16 | Short sentences covering crop, pest, disease, fertiliser, irrigation, scheme and number vocabulary. **Standard-Marathi drafts.** A native Varhadi speaker must write the natural Varhadi version in `varhadi_text` (and their pseudonym in `rewritten_by`) before a read prompt is used. `session-sheet` skips read prompts that have no rewrite. |

Elicited free speech is the more valuable data: it is natural, and it contains the words
farmers actually use. Read prompts guarantee coverage of specific terms.

**Crop scope** is configured in `data/prompts/crop_scope.json`:
`["cotton", "soybean", "orange", "tur"]`, plus `general`, which is always included. It is
marked *proposed*; confirm the final scope before collecting. `--crops` on `session-sheet`
overrides it for one session.

## Consent and privacy (enforced by the tools)
- A recording is rejected unless `--consent given`. Use a written or recorded consent form
  that states the purpose (research, model training), the retention period, and the right
  to withdraw.
- `--speaker-key` can be any private identifier, such as a name or phone number. Only an
  HMAC of it, keyed by `SPEAKER_ID_SALT` in `.env`, is stored. The model and every CSV only
  ever see `VH_S###`.
- Record `--age-group` and `--gender` only if the speaker agreed to share them.
- `withdraw-speaker VH_S###` deletes all of that speaker's audio, metadata and transcripts,
  and retires the ID.
- Recordings, transcripts and the speaker registry are gitignored, because the GitHub repo
  is public.

## Transcription rules
1. Write exactly what was said, in Devanagari, keeping Varhadi forms as spoken (काई, नाई,
   त्याले…). Do **not** convert them to standard Marathi.
2. For read prompts, the prompt text is only the candidate. People rarely read verbatim, so
   correct it to what was actually said.
3. Keep English or Hindi words as spoken (code-switching), in Devanagari.
4. Numbers in words, as spoken.
5. Mark unclear speech by rejecting the clip, not by guessing.

## Targets
- Several speakers from different districts (Amravati, Akola, Yavatmal, Washim, Buldhana,
  Wardha, Nagpur…). Splits are made **by speaker** (`dataset_tool.py split`), so the test
  speakers must never appear in training.
- A held-out agricultural test set from speakers who are not in training. This is the
  evaluation that decides whether the ASR works for KisaanDost. VAHNT results do not.
- Current counts: **0 speakers, 0 recordings.** Nothing has been collected or fabricated.
