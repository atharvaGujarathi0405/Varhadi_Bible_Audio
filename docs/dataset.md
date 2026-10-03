# Dataset (P1)

Tooling: `scripts/dataset_tool.py` (Windows `.venv` or WSL `.venv-asr-wsl`). Code: `varhadi_data/`.

## Layout

```
data/
  raw/recordings/<VH_S###>/        original uploads, byte-for-byte          (gitignored)
  processed/recordings/<VH_S###>/  16 kHz mono WAV used for ASR             (gitignored)
  private/speaker_registry.csv     pseudonym <-> keyed hash, no raw PII     (gitignored)
  metadata/manifest.csv            one row per utterance                    (tracked)
  transcripts/gold_transcripts.csv human review workflow                    (tracked)
  splits/{train,validation,test}.csv                                        (tracked)
  evaluation/gold_eval.csv         verified references, asr_metadata format (tracked)
  evaluation/audio/                exported verified clips                  (gitignored)
  prompts/recording_prompts.csv    elicitation + read prompts               (tracked)
```
VAHNT audio stays in `audio/`. The downloader owns it, and the manifest just points to it.

## Manifest columns

`utterance_id, audio_path, speaker_id, district, age_group, gender, duration_sec, language,
dialect, domain, transcript, transcript_status, recording_quality, consent_status, source,
license, split`

`validate_row` enforces these rules, and `dataset_tool.py check` runs them along with the leakage check:
- `transcript_status`: `not_available` (the transcript must be empty), `pending_review`, or `verified` (the transcript must be non-empty).
- `speaker_id` is `VH_S###` or `unknown`. It is never a name or phone number.
- `source=native_recording` requires `consent_status=given`.
- Allowed values: domain `general|scripture|agriculture`, quality `good|fair|poor|unchecked`, split `train|validation|test|unassigned`.

**VAHNT rows:** `speaker_id=unknown`, `transcript_status=not_available`, `domain=scripture`,
`license=unverified: local research use only`. No references are fabricated.

## Commands

```bash
python scripts/dataset_tool.py init
python scripts/dataset_tool.py register-vahnt
python scripts/dataset_tool.py register-recording --audio rec.m4a --speaker-key "<private id>" \
    --district Amravati --domain agriculture --consent given [--age-group 30-45] [--gender f] \
    [--prompt-text "text the speaker was asked to read"]
python scripts/dataset_tool.py withdraw-speaker VH_S004
python scripts/dataset_tool.py split [--ratios 0.8 0.1 0.1] [--seed 13]
python scripts/dataset_tool.py check
python scripts/dataset_tool.py stats
python scripts/dataset_tool.py gold-seed
python scripts/dataset_tool.py gold-review --segment-id JHN_002_c000 --corrected "..." --reviewer AG --status verified
python scripts/dataset_tool.py gold-export
# then score the baseline on the verified set:
python scripts/run_asr_baseline.py --metadata data/evaluation/gold_eval.csv --predictions asr_outputs/gold_predictions.csv
python -m asr_baseline.evaluate --predictions asr_outputs/gold_predictions.csv --metadata data/evaluation/gold_eval.csv
```

## Native recordings: privacy and quality
- **Consent gate:** a recording without `--consent given` is rejected before any ID is allocated or any file is stored.
- **Pseudonymous IDs:** `VH_S###` is derived from `HMAC-SHA256(SPEAKER_ID_SALT, normalised speaker key)`.
  - The raw key is never written anywhere, so the same person gets the same ID in later sessions.
  - `SPEAKER_ID_SALT` must be set in `.env`; the tool refuses to run without it.
- **Consented fields only:** `age_group` and `gender` are recorded only when the speaker agreed to share them.
- **Withdrawal:** `withdraw-speaker` deletes the speaker's raw and processed audio, manifest rows,
  gold rows and exported eval clips, then rebuilds `gold_eval.csv`. The pseudonym is retired and never reused.
- **Quality checks:** recordings are flagged, never auto-fixed. `poor` means more than 0.1% clipped samples or RMS below -40 dBFS. `fair` means RMS below -30 dBFS.
  Recordings that can't be decoded, or are shorter than 0.5 s or longer than 600 s, are rejected.

## Speaker-level splits
- Rows are grouped by `speaker_id`, so no speaker appears in two splits. `check` and `split` both fail on leakage.
- `unknown` speakers (all of VAHNT) always go to **train**. Their speakers can't be told apart, so leakage into test can't be ruled out.
- Known speakers are shuffled with a fixed seed and assigned greedily by duration: test first, then validation, then train.
- With few speakers, the realised ratios differ from the targets. The tool reports them rather than forcing them.

## Gold transcripts
`segment_id, audio_path, speaker_id, start_sec, end_sec, raw_asr, corrected, reviewer,
review_status, reviewed_at, provenance`
- **Original audio is untouched.** A segment is a time range of a source file.
- **Two text fields:** `raw_asr` keeps the untouched baseline output, and `corrected` is what a human heard.
- **Anchoring risk:** `gold-seed` pre-fills `raw_asr` from the ASR chunks, which can bias reviewers toward the model's errors.
  Reviewers must transcribe what they hear, not polish the model output.
- **Read prompts:** the text the speaker was asked to read is queued as `pending_review` and never auto-verified.
- **Verified only:** only `verified` rows with a reviewer and non-empty `corrected` text are exported, so WER/CER is only ever computed against human-verified references.

## Recording prompts
`data/prompts/recording_prompts.csv` has two kinds of prompt:
- **Elicitation prompts (E01–E11):** speakers answer freely in their own Varhadi, which gives natural speech.
- **Read prompts (R01–R16):** these cover crop, pest, disease, fertilizer, irrigation, scheme and number vocabulary.

All are **standard Marathi drafts** (`draft_needs_native_review`). A native Varhadi speaker must rewrite the
read prompts before use. They are not Varhadi text.

## Current status (2026-10-04)
- 89 VAHNT chapters registered, 705.9 min, speaker unknown, transcripts NOT AVAILABLE.
- Native recordings: 0. Verified gold segments: 0. Splits: not yet meaningful (no known speakers).

## VAHNT text alignment (option b)

Code: `varhadi_data/vahnt_text.py`, `varhadi_data/align.py`, `scripts/vahnt_align.py`.

```bash
# Windows .venv (Playwright; bible.com serves a JS challenge to plain HTTP)
python scripts/vahnt_align.py fetch-text JHN.1.VAHNT        # -> data/transcripts/vahnt_text/JHN_001.json
# WSL .venv-asr-wsl (NeMo)
python scripts/vahnt_align.py align JHN_001 [--max-sec 20]  # -> data/transcripts/alignments/JHN_001.csv
```

- **What text is kept:** the published chapter text is stored verbatim with its URL, fetch time and licence note.
  Only what a narrator reads is kept: section headings and verse text. Verse numbers,
  cross-references and footnotes are dropped.
- **How alignment works:** CTC Viterbi forced alignment (numpy, no new dependency) over the whole chapter,
  using the baseline model's Marathi CTC log-probs. These are computed chunk by chunk with the
  inference silence cuts, so long chapters fit in memory. See `align.py` for the method.
- **Speech that isn't in the text:** the start and end of the audio may stay unaligned, which absorbs the book/chapter
  announcement. Unscripted speech in the middle of a chapter is not modelled; it would lower that segment's `align_score`.
- **Segments:** verses and headings are grouped into segments of about 20 s or less, cut at the midpoint of the gap between them.
- **Output columns:** `segment_id, audio_path, start_sec, end_sec, units, n_headings, text, align_score,
  raw_asr, cer_vs_raw_asr`.

**These are candidates, not ground truth.** The published text may differ from what the
narrator actually said. Aligned segments must go through human review before use as test references.

**Bias warning:** `cer_vs_raw_asr` compares the published text with the *baseline* model's
own decode. Never filter the evaluation set on it, because that would keep only the segments
the baseline already handles and inflate its score. `align_score` comes from the same model,
so use it to prioritise review, not to silently drop segments.

### Pilot: JHN_001 (2026-10-04)
- **Text:** 51 verses and 4 headings parsed (51/51 verse refs).
- **Alignment:** 31 segments, 8.9–20.2 s (median 15.0 s), covering 6.32–471.56 s of the 473.57 s audio. It took 87 s on CPU.
- **Intro:** 0–6.32 s ("योहानाची सुवारता अध्याय एक", the spoken book/chapter announcement) was left unaligned, as intended.
- **Boundary check (automatic, not a listening test):** for all 31 segments, the baseline decode of the
  aligned span starts and ends on the same words as the aligned text.
- `align_score`: -1.21 to -0.29 (median -0.77).
- QUALITATIVE OBSERVATION: most text/ASR differences are systematic Varhadi-vs-standard-Marathi
  forms: काई→काही, नाई→नाही, त्याले→त्याला, पयले→पहिले, देला→दिला, म्हतलं→म्हटलं, मांग→मागे.
  This is the gap Varhadi adaptation targets. `cer_vs_raw_asr` (median 0.125) is against
  **unverified** published text, so it is NOT a baseline CER result.
- **Not yet done:** human listening spot-check, importing into the gold workflow, other chapters.
