You are continuing development of the existing KisaanDost / Varhadi ASR project.

DO NOT jump to frontend, RAG, TTS, IVR, vision, or other P2+ application features yet.

Our immediate goal is:

    FINISH P1 DATA PREPARATION
        ↓
    CREATE VERIFIED GOLD TEST DATA
        ↓
    GET FIRST REAL WER/CER
        ↓
    PREPARE P2 ASR FINE-TUNING PIPELINE

Read the current progress log and repository before doing anything:

    project_progress.txt
    prompt.md
    docs/dataset.md
    docs/asr_baseline.md

==================================================
CURRENT VERIFIED STATUS
==================================================

P0 is COMPLETE.

- 89/89 downloaded VAHNT chapters
- ~705.9 minutes of audio
- 1,771 chunks
- 0 failed transcriptions
- chunked inference working
- raw baseline predictions available
- no WER/CER yet because no verified references exist

P1 dataset tooling is COMPLETE.

P1b forced alignment is PARTIALLY COMPLETE.

- Published VAHNT text has been fetched.
- 11/89 chapters have currently been aligned.
- 78 chapters remain.
- Alignment uses the pretrained Marathi CTC model's log probabilities.
- Candidate aligned text is NOT automatically considered ground truth.
- All imported alignments are pending human review.
- No verified gold segments exist yet.

The progress log explicitly says:

1. Finish alignment
2. Import alignments
3. Human review
4. Build held-out test set
5. Score baseline
6. Then start P2 fine-tuning

Follow this order.

==================================================
PHASE 1 — REPOSITORY SAFETY CHECK
==================================================

Before modifying anything:

1. Run git status.
2. Inspect the current branch and latest commits.
3. Verify that the existing downloader and ASR baseline are untouched.
4. Run the existing test suite.
5. Inspect the current alignment state.
6. Inspect data/metadata/manifest.csv status.

IMPORTANT:

The progress log says:

    data/metadata/manifest.csv is deleted in the working tree
    and the deletion was NOT made by Claude.

DO NOT silently restore, delete, overwrite, or commit this file.

First determine whether the current working tree intentionally removed it.

If restoration is necessary for the dataset workflow, explain why before doing it.

Do not destroy any existing dataset metadata.

==================================================
PHASE 2 — FINISH VAHNT ALIGNMENT
==================================================

Continue the existing forced-alignment implementation.

Existing tools:

    varhadi_data/vahnt_text.py
    varhadi_data/align.py
    scripts/vahnt_align.py

Current state:

    11/89 chapters aligned
    78 remaining

Resume from the existing checkpoint/state.

DO NOT restart from chapter 1 unnecessarily.

Use the existing resume mechanism.

Run:

    align --all

or the repository's actual equivalent command after inspecting the CLI.

IMPORTANT:

The machine is CPU-only and has limited RAM.

The previous full alignment run was stopped because of memory pressure, not because the alignment algorithm failed.

Therefore:

- process incrementally
- avoid loading all 89 chapters into memory
- do not launch multiple large alignment jobs simultaneously
- preserve completed chapters
- allow resume after interruption
- log memory/resource usage
- do not modify the downloader

If the complete alignment is too expensive, process it in safe batches.

After completion, report:

- chapters aligned
- chapters failed
- segments generated
- failures/retries
- total audio duration aligned
- peak memory if measurable

==================================================
PHASE 3 — ALIGNMENT QUALITY INSPECTION
==================================================

Do NOT assume that forced alignment equals ground truth.

The alignment output is a CANDIDATE REFERENCE.

Create/maintain a clear status:

    pending_review
    verified
    rejected
    needs_realignment

For each candidate segment preserve:

- audio path
- start time
- end time
- candidate transcript
- source chapter
- source URL
- alignment metadata
- review status

Do NOT modify the original audio.

==================================================
PHASE 4 — HUMAN REVIEW WORKFLOW
==================================================

Build or verify the human-review workflow.

We need a practical way to listen to aligned clips and verify the candidate transcript.

The reviewer must be able to inspect:

    audio clip
    candidate transcript
    source text
    review status

Possible decisions:

    VERIFIED
    CORRECT
    NEEDS_CORRECTION
    REJECTED

For corrected samples store:

    raw/candidate transcript
    corrected transcript
    reviewer
    review timestamp
    review status

Never overwrite the original candidate transcript.

IMPORTANT:

A segment can only become:

    verified

after a human has actually listened to it.

==================================================
PHASE 5 — CREATE A SMALL VERIFIED DATASET FIRST
==================================================

Do NOT wait for all 89 chapters to be manually reviewed before measuring anything.

Create a practical initial gold dataset.

Start with JHN_001 because it already has aligned segments.

Target:

    30–60 verified segments

or approximately:

    10–20 minutes of verified speech

if feasible.

The purpose is to establish the evaluation pipeline.

Then create:

    TRAIN
    VALIDATION
    TEST

with speaker-level separation wherever speaker identity is known.

IMPORTANT:

VAHNT is scripture speech from unknown narrator(s).

Therefore explicitly record this limitation.

Do not claim that this is agricultural speech.

==================================================
PHASE 6 — FIRST REAL WER/CER
==================================================

Once verified references exist:

Run the existing evaluator.

Use:

    baseline predictions
    +
verified reference transcripts

Calculate:

    WER
    CER
    substitutions
    deletions
    insertions

Do NOT calculate metrics against unverified alignment text.

Do NOT report the previously observed alignment CER of 0.125 as WER/CER baseline.

That was only a qualitative/diagnostic observation.

The first reported WER/CER must come from verified human-reviewed references.

Generate:

    baseline_metrics.json
    baseline_per_sample.csv
    baseline_report.md

Also include:

- number of evaluated samples
- duration evaluated
- number of speakers if known
- dataset split
- verification procedure
- limitations

If the dataset is too small for a meaningful final result, label it:

    PILOT EVALUATION

not final evaluation.

==================================================
PHASE 7 — ERROR ANALYSIS
==================================================

Use the verified references to analyze baseline errors.

Create categories such as:

1. Varhadi → standard Marathi substitution
2. pronunciation-related error
3. unseen/rare vocabulary
4. code-switching
5. grammatical/dialectal variation
6. proper noun
7. deletion
8. insertion
9. agricultural terminology if present

Do not automatically assign categories with an LLM unless clearly marked as assisted analysis.

Preserve raw prediction and reference.

The goal is to identify what ASR adaptation needs to learn.

==================================================
PHASE 8 — AGRICULTURAL DATA GAP
==================================================

Do NOT conclude that the current Bible dataset is sufficient for the final KisaanDost agricultural ASR.

The project still needs native Varhadi agricultural speech.

Design the data collection structure now, but do not fabricate data.

Use the existing recording prompts:

    data/prompts/recording_prompts.csv

Inspect the prompts.

They are currently standard-Marathi drafts that native Varhadi speakers must rewrite.

Prepare a clean workflow for:

    native speaker
        ↓
    Varhadi utterance
        ↓
    recording
        ↓
    exact transcription
        ↓
    human verification
        ↓
    metadata

Priority agricultural categories:

- cotton
- soybean
- orange
- tur
- crop disease
- pests
- irrigation
- fertilizer
- crop practices
- agricultural government schemes

Keep the final crop scope configurable.

Do NOT invent recordings or transcripts.

==================================================
PHASE 9 — PREPARE P2, BUT DO NOT TRAIN YET
==================================================

Only after the baseline evaluation pipeline is working:

Inspect the actual .nemo checkpoint configuration.

We need to understand:

- encoder configuration
- decoder
- tokenizer
- vocabulary
- sample rate
- feature configuration
- CTC/RNNT configuration
- optimizer
- scheduler
- augmentation
- training data format

Do NOT guess the fine-tuning configuration.

Read the actual checkpoint/model configuration and AI4Bharat NeMo training requirements.

Then create:

    experiments/
        baseline/
        varhadi_adaptation/

and prepare:

    train_manifest.json
    val_manifest.json
    test_manifest.json

using the project's verified dataset format.

Do not train yet.

==================================================
PHASE 10 — COLAB TRAINING PLAN
==================================================

The actual fine-tuning will run on a GPU in Google Colab.

Prepare a reproducible training package for Colab.

It should contain:

1. environment/setup instructions
2. model download/authentication instructions
3. dataset preparation
4. training manifest generation
5. fine-tuning script
6. validation
7. checkpoint saving
8. experiment configuration
9. evaluation script
10. model export/loading instructions

The local CPU machine should only be used for:

- data preparation
- preprocessing
- alignment
- small-scale inference
- testing
- evaluation

Do not attempt expensive full-model fine-tuning locally.

==================================================
PHASE 11 — DO NOT START FULL P2 TRAINING YET
==================================================

Before training, produce a report:

    P2_READY.md

Containing:

- verified dataset size
- training/validation/test counts
- total duration
- speaker count
- model checkpoint
- tokenizer information
- sample rate
- fine-tuning configuration
- GPU requirements
- estimated training cost/time
- known risks
- expected outputs

Then STOP and report to me.

==================================================
IMPORTANT RESEARCH RULES
==================================================

Never fabricate:

- transcripts
- speaker identities
- WER
- CER
- dataset size
- training results
- accuracy
- improvement
- agricultural speech data

Always distinguish:

    candidate alignment
    vs
    human-verified gold transcript

Always distinguish:

    general Varhadi Bible speech
    vs
    agricultural Varhadi speech

Never claim the ASR is Varhadi-adapted until actual fine-tuning has been performed and evaluated.

==================================================
EXPECTED END STATE OF THIS TASK
==================================================

When you finish this task, I expect:

1. Remaining VAHNT alignment completed or clearly documented
2. Alignment data safely stored
3. Human review workflow ready
4. Initial verified gold dataset created if possible
5. First REAL baseline WER/CER calculated if enough verified data exists
6. Baseline error analysis
7. Agricultural speech data collection pipeline ready
8. P2 fine-tuning configuration inspected
9. Colab training package prepared
10. No frontend/RAG/TTS/IVR work yet

At the end, give me:

A. Files changed
B. Commands executed
C. Tests passed
D. Alignment progress
E. Verified dataset statistics
F. First real WER/CER, if available
G. Current blockers
H. Exact next command I should run

Do not silently proceed into unrelated phases.

Start with the repository audit and current alignment status.