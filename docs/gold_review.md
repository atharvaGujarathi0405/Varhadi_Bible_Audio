# Gold review: triage, human verification and gold evaluation

## 1. Why automatic transcripts are not ground truth

Every candidate transcript comes from forced alignment of the **published** VAHNT text to the
audio. Three things can make it wrong even when the alignment looks perfect:
- **The narrator didn't read exactly the published text.** For example, translator notes in
  parentheses ("(जवळपास तीन हजार लिटर)") are probably not read aloud. Words can also differ.
- **The alignment can drift.** If speech that isn't in the text is absorbed into a segment, its boundaries end up wrong.
- **Every automatic signal comes from the same pretrained Marathi model** that we want to evaluate.
  Agreement with it partly means "the baseline already handles this", which is circular.

So a candidate only becomes **gold** when a human has **listened** and accepted or corrected it.

## 2. Segment states

| State | Set by | Meaning | Gold evaluation? | Training? |
|---|---|---|---|---|
| `candidate` (review_status `pending_review`) | alignment import | automatic text, not reviewed | ✗ | ✗ |
| `HIGH_CONFIDENCE_CANDIDATE` | triage | strong automatic signals, **still unverified** | ✗ | only as `PSEUDO_LABELED`, only with `--include-unverified`, in a separate manifest |
| `NEEDS_REVIEW` | triage | uncertain: review first | ✗ | ✗ |
| `REJECT_CANDIDATE` | triage | probably misaligned or unusable (a suggestion only) | ✗ | ✗ |
| `verified` + decision `correct` (**VERIFIED**) | human | listened; candidate matches the speech | ✓ | ✓ gold |
| `verified` + decision `needs_correction` (**CORRECTED**) | human | listened; typed what was actually said | ✓ | ✓ gold |
| `rejected` (**REJECTED**) | human | unusable, unclear or misaligned | ✗ | ✗ |
| `needs_realignment` | human | text fine, time span wrong | ✗ | ✗ |

Triage columns (`triage_status`, `confidence_score`, …) and review columns (`decision`,
`review_status`, `reviewer`, …) are separate. Triage never writes review columns. The triage
script asserts this, and so does a test.

## 3. How confidence and triage work

Code: `varhadi_data/triage.py`. Config, with every weight and threshold: `varhadi_data/triage_config.json` (`triage-v1`).

```
confidence = Σ weight_i · component_i  −  Σ artifact penalties        (clipped to [0, 1])
```

Each component is in [0, 1]. Ramps map a raw value linearly from `bad` (0) to `good` (1):

| Component | Weight | Raw signal | Mapping |
|---|---|---|---|
| alignment | 0.35 | `align_score`, the mean CTC log-prob of the aligned tokens | −2.0 → 0, −0.5 → 1 |
| agreement | 0.30 | candidate vs baseline ASR: mean of CER (0.45 → 0, 0.12 → 1) and (insertions+deletions)/words (0.35 → 0, 0.05 → 1) | |
| audio_quality | 0.15 | mean of silence ratio (frames < −40 dBFS; 0.7 → 0, 0.3 → 1), clipping ratio (0.01 → 0, 0.0005 → 1), loudness (−40…−30 dBFS up, −6…−1 dBFS down) | |
| duration | 0.10 | segment seconds | 1 in [3, 25], linear to 0 at 1 and 45 |
| speaking_rate | 0.10 | candidate characters per second, a boundary-quality proxy | 1 in [6.5, 12.5], linear to 0 at 4.5 and 16 |

Artifact penalties are text problems, all of them correctable by a reviewer:

| Flag | Penalty | Why |
|---|---|---|
| `parentheses` | 0.25 | translator notes / footnotes that are usually not read aloud |
| `empty_asr` | 0.30 | the baseline heard nothing |
| `digits_or_latin` | 0.15 | numerals or Latin script in a Devanagari reading |
| `repeated_word` | 0.15 | the same word three or more times in a row (malformed text) |
| `contains_heading` | 0.05 | section heading included (usually read, but not always) |

Classification:
- **`REJECT_CANDIDATE`:** a hard failure (empty candidate, duration or speaking rate outside the
  hard limits, mostly silence), **or** a *base* confidence (before artifact penalties) below
  `thresholds.review` = **0.50**. Text artifacts never cause rejection on their own, and
  `parentheses` segments are never auto-rejected: the audio is usable once a reviewer deletes the note.
- **`HIGH_CONFIDENCE_CANDIDATE`:** final confidence ≥ `thresholds.high_confidence` = **0.90**, and no blocking flag (`parentheses`, `empty_asr`, …).
- **`NEEDS_REVIEW`:** everything else.

**How the defaults were chosen (2026-10-04, 1,296 candidates):**
- **Speaking rate:** the OK range is about the 1st–99th percentile (6.6–12.0 chars/s observed).
- **Agreement:** the CER ramp treats the systematic Varhadi-vs-Marathi difference (median CER 0.127) as full agreement, so dialect forms aren't punished.
- **High-confidence threshold:** 0.90 is about the median confidence (0.909). It's deliberately conservative, since there are no human labels to calibrate against yet.
- **Rejections:** the first version auto-rejected 9 translator-note segments with good audio, which led to the never-reject rule.
- **Status:** these values are **provisional.** Once JHN_001 has been reviewed, compare triage against the human decisions and recalibrate (bump `version`).

ASR decoding stability (agreement between CTC and RNNT, or across beams) is **NOT AVAILABLE**
without re-decoding, so it is not used.

**Known bias.** `agreement` uses the baseline model. High confidence therefore leans towards
segments the baseline already gets right. Triage may **order** a review queue, but it must
**never** decide which test segments get reviewed: every segment of a test chapter needs a human decision.

## 4. Human review process

```bash
python scripts/triage_gold_candidates.py --chapter JHN_001 --report   # order the queue
python scripts/review_server.py --chapter JHN_001                     # http://127.0.0.1:8765
```

- **Queue order:** `NEEDS_REVIEW`, then `REJECT_CANDIDATE`, then `HIGH_CONFIDENCE_CANDIDATE`, each with the lowest confidence first.
- **What the page shows:** the audio, the candidate, the baseline ASR, the alignment and confidence scores, warning flags,
  duration, the segment ID and source link, and progress counts (reviewed, verified, corrected, rejected, remaining,
  plus the triage counts).
- **Keys:** `V` verify · `E` edit (then `Ctrl+Enter` saves the correction) · `R` reject · `A` needs
  realignment · `Space` replay · `N`/`P` next/previous. After a decision the page moves to the next pending segment.
- **Listening is enforced.** `V` and the save-correction button only unlock after the clip has played to
  the end (at least 95%), and the server re-checks this.
- **Persistence:** every decision is written to `data/transcripts/gold_transcripts.csv` immediately.
  The server keeps no state in memory, so a refresh or crash loses nothing, and the page reopens at the first pending segment.
- **Reviewer ID:** initials or a pseudonym of 2–20 letters, digits, `_` or `-` (names with spaces and phone numbers are refused).

**Transcription rule.** Write what is **spoken**. Keep Varhadi forms (काई, नाई, त्याले, पयले…) as
heard, even where the candidate or the ASR uses standard Marathi. Unclear audio gets rejected with a note, never a guess.

## 5. How corrections are stored

`candidate_text` is the original automatic candidate and is **never modified**. A correction goes into
`corrected_text`, with `decision=needs_correction`, `reviewer`, `reviewed_at` (UTC) and optional
`review_notes`. The scoring reference is `corrected_text` for corrected rows and `candidate_text` for verified rows
(`gold.reference()`). Re-reviewing a segment replaces its decision on the same row, so there are no duplicates.
The prompt03 names `VERIFIED`/`CORRECTED`/`REJECTED` are accepted as aliases of the stored decisions `correct`/`needs_correction`/`rejected`.

## 6. How gold evaluation works

`scripts/run_gold_eval.py --test-chapters JHN_001` exports **only** `review_status=verified` rows of the
test chapters, decodes each clip with the unchanged baseline, and reports WER, CER,
substitutions, deletions, insertions and the common error pairs. Both sides get the same
punctuation stripping and no other normalisation. It **exits with an error if there are zero verified segments**,
and results are labelled **PILOT EVALUATION** unless `--final` is given.

## 7. How pseudo-labeled data is kept separate

`experiments/varhadi_adaptation/build_manifests.py` uses gold only by default. With `--include-unverified`,
pending `HIGH_CONFIDENCE_CANDIDATE` segments are written to **separate** files
(`train_pseudo_labeled_manifest.json`, `val_pseudo_labeled_manifest.json`) with
`label_source: PSEUDO_LABELED`. They never enter the test split or the gold manifests. Training on them
means passing those files explicitly and reporting it.

## 8. Research limitations
- The triage weights and thresholds are hand-set heuristics, not learned or calibrated yet.
- The confidence signals depend on the baseline model (circularity, see §3).
- VAHNT is scripture read by unknown narrator(s). It is not agricultural speech, and speaker-independent generalisation can't be measured.
- Gold quality depends on the reviewer's Varhadi knowledge. There is a single reviewer so far, so no inter-annotator agreement.

## 9. Reproduce

```bash
python scripts/vahnt_align.py align --all                      # WSL, resumes
python scripts/dataset_tool.py gold-import-alignments
python scripts/triage_gold_candidates.py --all --export-csv --report   # data/evaluation/triage_report.csv, triage_summary.json
python scripts/review_server.py --chapter JHN_001              # human review (all 31 for the test chapter)
wsl -d Ubuntu -- bash -lc "cd /mnt/e/KisaanDost && HF_HUB_OFFLINE=1 .venv-asr-wsl/bin/python scripts/run_gold_eval.py --test-chapters JHN_001"
```
