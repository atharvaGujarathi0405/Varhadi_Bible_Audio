# Experiment Log

Every entry records only measured results. Unmeasured = PENDING, unavailable = NOT AVAILABLE.

---

## Exp 0.1 — Pretrained Marathi ASR, single 30 s sample

- **Date:** 2026-09-23
- **Objective:** Prove the raw baseline pipeline end to end on Varhadi speech.
- **Dataset:** `asr_audio/JHN_001_sample30s.wav`, the first 30 s of VAHNT John 1. General Varhadi scripture speech, not agricultural.
- **Model:** `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`, CTC decoder, AI4Bharat NeMo 1.23.0rc0, torch 2.14.0+cpu.
- **Configuration:** no correction, no normalization, no fine-tuning.
- **Result:** transcribed in 3.75 s. The raw output is in `asr_outputs/predictions.csv`. WER/CER: NOT AVAILABLE (no reference).
- **Limitations:** a full 473 s chapter in one pass was OOM-killed at about 6 GB RSS.
- **Next step:** chunked inference.

## Exp 0.2 — Chunked inference on a full chapter

- **Date:** 2026-10-04
- **Objective:** Remove the memory ceiling so whole chapters can be transcribed. Same raw baseline, no correction.
- **Dataset:** `audio/JHN_001.mp3` (473.6 s), plus the 30 s sample for the comparison.
- **Model:** same as Exp 0.1.
- **Configuration:** chunks cut at the lowest-energy point, target 20 s, max 30 s, overlap 0 s (see `docs/asr_baseline.md`).
- **Result:** 19 chunks, 53.9 s inference (~8.8x realtime), peak RSS 1.74 GB. The stitched raw output is in `asr_outputs/predictions.csv` and the per-chunk outputs in `asr_outputs/chunks/JHN_001.csv`. WER/CER: NOT AVAILABLE.
- **QUALITATIVE OBSERVATION:** with 0 overlap at silence cuts, the 30 s sample's chunked output was nearly identical to the single pass (a few boundary spellings differ). Exact-match overlap stitching failed on a real seam, so it was replaced with a longest-common-run stitcher.
- **Limitations:** no references, so neither the effect of chunking nor the baseline accuracy can be scored.
- **Next step:** batch over all 89 downloaded chapters (approval needed), then P1 reference alignment.

## P1b pilot: VAHNT text forced alignment (JHN_001)

- **Date:** 2026-10-04
- **Objective:** Test whether the published VAHNT text can be aligned to the chapter audio to produce candidate paired (audio, text) segments for Varhadi adaptation (option b).
- **Dataset:** `audio/JHN_001.mp3` (473.57 s) and the published VAHNT text for John 1 (51 verses, 4 headings), fetched from bible.com.
- **Model:** the baseline CTC head (Marathi softmax), used only for alignment.
- **Configuration:** numpy CTC Viterbi with unaligned audio allowed at the start and end; segments of 20 s or less; log-probs computed with the inference chunk cuts (20/30/0).
- **Result:** 31 segments covering 6.32–471.56 s, computed in 87 s. In all 31 segments, the baseline decode of each span starts and ends on the same words as the aligned text (automatic check). `align_score` median -0.77.
- **QUALITATIVE OBSERVATION:** the text/ASR differences are mostly systematic Varhadi-vs-Marathi forms (काई/काही, नाई/नाही, त्याले/त्याला, पयले/पहिले).
- **Limitations:** no human listening check yet. The published text is unverified against the narration. The CER against the published text is NOT a verified baseline metric. Only one chapter has been tried.
- **Next step:** human spot-check a sample of segments, then scale to the downloaded chapters and import them as pending_review gold rows.

## Exp 0.3: Raw baseline over all downloaded VAHNT chapters

- **Date:** 2026-10-04 (started 2026-10-03, stopped at 41 files and resumed the next morning)
- **Objective:** Raw pretrained-Marathi transcriptions of every downloaded Varhadi chapter, for error analysis and as the baseline side of later comparisons. No correction, normalisation or fine-tuning.
- **Dataset:** 89 VAHNT chapters (705.9 min, 208–880 s each), speaker unknown, general scripture speech (not agricultural).
- **Model:** same as Exp 0.1. Chunked 20/30/0 (Exp 0.2).
- **Result:** 89/89 chapters transcribed, 0 failed, 0 empty outputs, 1,771 chunks. Throughput, measured on the 87 chapters that did not reuse earlier chunks: 689.0 min of audio in 72.6 min of inference (~9.5x realtime overall, per-chapter median 10.3x, range 4.4–11.5x). The slowest chapters overlapped with other jobs running on the same CPU. Outputs: `asr_outputs/predictions.csv`, `asr_outputs/chunks/`, `asr_outputs/baseline_report.md`.
- **WER/CER:** NOT AVAILABLE. No verified references yet; `metrics.json` records 0 evaluated samples.
- **Pipeline fix found here:** file-level resume compared raw path strings, so `JHN_001` had been recorded twice (once from a single-file run, once from the batch), which inflated the report to 91 files and 714.3 min. Paths are now stored repo-relative and normalised on resume. The duplicate row had identical text and was removed, keeping the original row with its real 53.9 s timing.
- **Limitations:** transcription only; scripture domain; a single unknown narrator (or narrators), so it says nothing about speaker variation.
- **Next step:** align the published text (P1b) to obtain candidate references, then human review.
