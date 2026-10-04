You are working on my KisaanDost research repository.

IMPORTANT: Before changing anything, inspect the repository and understand the existing implementation. Do NOT rewrite working components unnecessarily.

PROJECT CONTEXT
---------------
This is a research project on:

"KisaanDost: A Region-Adaptive Multimodal AI Assistant for Varhadi-Speaking Farmers"

Current research focus:
1. Establish a pretrained Marathi ASR baseline on Varhadi speech.
2. Build a verified Varhadi speech dataset.
3. Fine-tune/adapt the ASR model.
4. Compare baseline vs adapted model using WER/CER and downstream agricultural metrics.

Current ASR model:
ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large

Existing data:
- 89/89 VAHNT New Testament audio chapters downloaded.
- Approximately 706 minutes of audio.
- Audio is Varhadi/general speech, NOT agricultural speech.
- Forced alignment has been performed for approximately 40/89 chapters.
- Approximately 1,296 candidate aligned segments currently exist from the aligned chapters.
- Human-verified gold segments are currently 0.
- Existing alignment candidates must NOT automatically become gold truth.
- Do NOT fabricate WER/CER or verified data.

EXISTING IMPORTANT COMPONENTS
-----------------------------
Downloader:
- downloader.py
- browser.py
- chapters.py
- config.py
- progress.py
- audio/
- progress.json
- failed.json
- chapter_metadata.json

DO NOT modify the downloader unless absolutely necessary.

ASR:
- asr_baseline/
- config.py
- audio_utils.py
- model.py
- transcribe.py
- evaluate.py
- error_analysis.py
- utils.py
- chunking.py

Dataset:
- varhadi_data/
- data/
- scripts/dataset_tool.py

Alignment:
- varhadi_data/vahnt_text.py
- varhadi_data/align.py
- scripts/vahnt_align.py

Existing review/evaluation:
- scripts/review_server.py
- scripts/run_gold_eval.py
- varhadi_data/gold.py

Existing adaptation scaffolding:
- experiments/varhadi_adaptation/
- experiments/evaluation/

Current problem
---------------
Manual review of ~1,296 segments one-by-one is too slow.

I want you to implement a SAFE, research-valid, confidence-based review pipeline that dramatically reduces manual work without falsely claiming that model-generated transcripts are ground truth.

CORE RESEARCH RULE
------------------
NEVER automatically mark a segment as human-verified merely because:
- ASR agrees with the candidate transcript
- alignment score is high
- CER/WER against the candidate is low
- the model is confident

The system must distinguish:

1. AUTO-ACCEPTED CANDIDATE
   = high-quality candidate suitable for prioritization/training consideration, but NOT gold truth.

2. HUMAN VERIFIED GOLD
   = a human listened to the audio and explicitly accepted/corrected it.

3. REJECTED
   = unusable/misaligned/unclear.

4. NEEDS_REVIEW
   = uncertain and requires a human.

This distinction is mandatory.

OBJECTIVE
---------
Build an automatic triage system:

Audio
 ↓
Forced alignment / candidate transcript
 ↓
Independent quality signals
 ↓
Automatic triage
 ├── HIGH_CONFIDENCE_CANDIDATE
 ├── NEEDS_REVIEW
 └── REJECT_CANDIDATE
 ↓
Human reviews only the necessary subset
 ↓
Verified gold dataset

Do NOT use baseline WER/CER against the same candidate transcript as the only confidence signal.

CONFIDENCE SIGNALS
------------------
Use multiple available signals where possible:

1. Alignment score
2. Segment duration
3. Audio quality:
   - silence ratio
   - clipping
   - abnormal loudness
   - extremely short/long duration
4. ASR decoding stability if available
5. Agreement between:
   - alignment candidate
   - baseline ASR transcript
6. Number of substitutions / insertions / deletions between candidate and ASR
7. Suspicious punctuation/metadata artifacts
8. Parentheses/footnote/section-heading contamination
9. Repeated or obviously malformed text
10. Boundary quality if alignment metadata provides it

Do NOT invent a confidence score without explaining how it is calculated.

Implement a transparent scoring system, preferably something like:

confidence_score =
    weighted_alignment_score
    + audio_quality_score
    + ASR_candidate_agreement
    + duration_quality
    - artifact_penalties

Normalize scores to [0,1].

Put all thresholds and weights in configuration, NOT hard-coded throughout the code.

For example:

HIGH_CONFIDENCE_THRESHOLD=0.85
REVIEW_THRESHOLD=0.60

But choose sensible defaults based on the existing data distribution rather than blindly assuming these values.

IMPORTANT:
These thresholds determine TRIAGE ONLY.
They do not determine gold truth.

DATA SCHEMA
-----------
Extend the existing gold/review metadata without breaking compatibility.

Each segment should be able to record fields such as:

segment_id
chapter_id
audio_path
start_sec
end_sec
candidate_transcript
baseline_asr
alignment_score
audio_quality_score
agreement_score
artifact_score
confidence_score
triage_status

Possible triage_status:

HIGH_CONFIDENCE_CANDIDATE
NEEDS_REVIEW
REJECT_CANDIDATE

Human review fields:

review_status
reviewer_id
review_timestamp
reviewed_transcript
review_decision
review_notes

Possible review_decision:

VERIFIED
CORRECTED
REJECTED

VERY IMPORTANT:
Preserve the original candidate transcript permanently.

If a reviewer edits it:

candidate_transcript = original automatic candidate

reviewed_transcript = human-corrected transcript

Never overwrite the original candidate.

HUMAN REVIEW UI
---------------
Upgrade the existing:

scripts/review_server.py

Do NOT create a completely separate review system unless necessary.

Add a review mode that prioritizes:

1. NEEDS_REVIEW first
2. lowest-confidence candidates next
3. high-confidence candidates last if the reviewer wants to inspect them

UI should show:

- audio player
- candidate transcript
- baseline ASR transcript
- alignment score
- confidence score
- major warning flags
- segment duration
- chapter/segment ID
- review progress

Keyboard shortcuts:

V = verify
E = edit/correct
R = reject
Space = replay
N = next
P = previous

If the browser supports it, automatically move to the next segment after a decision.

Do NOT require the reviewer to manually navigate between files.

The UI must save every decision immediately.

If the page refreshes or the server crashes, previous decisions must remain.

Add a clear progress indicator:

Reviewed: X / Y
Verified: X
Corrected: X
Rejected: X
Remaining: X

Also show:

High-confidence candidates: X
Needs review: X
Rejected candidates: X

IMPORTANT HUMAN REVIEW BEHAVIOR
--------------------------------
Do NOT force the reviewer to standardize Varhadi into standard Marathi.

The reviewer must transcribe what is actually spoken.

For example, if the speaker clearly says:

"काई"

and the candidate says:

"काही"

the reviewer should preserve "काई" if that is what the speaker actually says.

Likewise:

"नाई" != automatically "नाही"
"त्याले" != automatically "त्याला"

The goal is speech ground truth, not literary Marathi correction.

If audio is unclear:
- do not guess
- mark NEEDS_REVIEW or REJECTED depending on the workflow

If candidate text is clearly wrong:
- correct it
- preserve original candidate
- mark CORRECTED

AUTOMATIC REVIEW PRIORITIZATION
--------------------------------
Add a script, preferably:

scripts/triage_gold_candidates.py

It should:

1. Load the current candidate/alignment dataset.
2. Calculate quality/confidence signals.
3. Assign triage status.
4. Write an auditable report.
5. Never modify raw audio.
6. Never delete candidates.
7. Never claim human verification.

Example CLI:

python scripts/triage_gold_candidates.py

Support:

--chapter JHN_001
--all
--threshold 0.85
--export-csv
--report

Produce something like:

data/evaluation/triage_report.csv

and:

data/evaluation/triage_summary.json

The report should contain counts and distributions.

Also print:

Total candidates: 1296
High-confidence candidates: ...
Needs review: ...
Rejected: ...

Do NOT assume these numbers in advance.

RESEARCH-SAFE TRAINING POLICY
-----------------------------
Implement clear dataset states:

candidate
triage_high_confidence
needs_review
verified
corrected
rejected

Only:

verified + corrected

may automatically enter the GOLD evaluation dataset.

For training, make it possible to explicitly choose:

--include-unverified

but default behavior must exclude unverified candidates.

For example:

python scripts/build_manifests.py

must only use verified/corrected segments by default.

If an option exists to use high-confidence unverified candidates for semi-supervised experimentation, clearly label them as:

PSEUDO_LABELED

and NEVER mix them silently with human gold data.

GOLD EVALUATION
---------------
Keep the existing:

scripts/run_gold_eval.py

compatible.

It must calculate WER/CER ONLY from:

review_status = VERIFIED/CORRECTED

Do not calculate "gold WER" from HIGH_CONFIDENCE_CANDIDATE.

The evaluator should fail clearly if there are zero verified segments.

Also generate:

- substitutions
- deletions
- insertions
- common error pairs

using the existing evaluation infrastructure.

PILOT WORKFLOW
--------------
After implementation, use JHN_001 as the pilot.

JHN_001 currently has approximately 31 aligned segments.

Run:

python scripts/triage_gold_candidates.py --chapter JHN_001 --report

Then launch:

python scripts/review_server.py --chapter JHN_001

The reviewer should be able to review the suspicious segments first instead of manually searching all 31.

After several human decisions, verify that:

- decisions persist
- original candidates remain unchanged
- corrected transcripts are stored separately
- verified count increases correctly
- rejected segments are excluded
- refresh does not lose decisions

Then run:

python scripts/run_gold_eval.py --test-chapters JHN_001

If no verified segments exist, it must fail safely.

TESTING
-------
Add tests for:

1. confidence calculation
2. triage classification
3. score normalization
4. malformed/missing scores
5. preserving original candidate transcript
6. corrected transcript storage
7. verified/rejected state transitions
8. review persistence
9. reviewer progress
10. evaluator excluding unverified data
11. PSEUDO_LABELED data never entering gold evaluation
12. duplicate review submission
13. page refresh persistence
14. invalid reviewer IDs
15. no accidental modification of downloader data

Run the full existing test suite.

Do NOT break the existing downloader or ASR baseline.

DATA INTEGRITY
--------------
Before modifying files:

- inspect git status
- inspect relevant schemas
- inspect current gold_transcripts.csv
- inspect review_server.py
- inspect alignment output format
- inspect existing tests

After changes:

- run tests
- run the JHN_001 triage pilot
- show exactly what files changed
- show git diff summary
- confirm downloader files were not modified
- confirm no raw audio was altered
- confirm no fabricated gold labels were created

IMPORTANT PROJECT CONSTRAINTS
-----------------------------
We have roughly 2 months left.

Do not over-engineer this.

Priority:

P0:
- reliable triage
- fast human review
- persistent review state
- research-safe gold separation

P1:
- reports
- evaluation integration
- tests

P2:
- UI polish

Do NOT spend time building a complicated ML classifier for confidence.
A transparent heuristic/rule-based triage system is acceptable for this phase.

Do NOT fine-tune the ASR model yet.

Do NOT build the RAG/LLM/TTS/frontend/IVR yet unless the current code already requires them.

The immediate goal is:

AUTOMATE DATA TRIAGE → MINIMIZE HUMAN REVIEW → CREATE REAL GOLD DATA → CALCULATE REAL BASELINE WER/CER.

DOCUMENTATION
-------------
Create/update:

docs/gold_review.md

Explain:

1. Why automatic transcripts are not ground truth.
2. How confidence/triage works.
3. Difference between candidate and verified gold.
4. Human review process.
5. How corrections are stored.
6. How gold evaluation works.
7. How pseudo-labeled data is kept separate.
8. Research limitations.
9. How to reproduce the pipeline.

Also update:

EXPERIMENT_LOG.md

with the implementation and pilot results.

FINAL OUTPUT
------------
At the end, report:

1. Files inspected
2. Files changed
3. New scripts/functions
4. Confidence signals used
5. Triage formula
6. Thresholds
7. Number of candidates classified in JHN_001
8. Number requiring review
9. Number automatically classified as high-confidence candidates
10. Number rejected
11. Number human-verified (should remain 0 unless I actually reviewed them)
12. Tests passed
13. Any issues
14. Exact commands to run the new workflow

CRITICAL:
Do not claim that any segment is "verified" unless the review workflow records an actual human verification decision.

Start by inspecting the repository. Do not make changes until you understand the current gold schema, alignment output, and review server implementation.