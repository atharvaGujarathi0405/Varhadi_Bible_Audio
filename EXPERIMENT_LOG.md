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
