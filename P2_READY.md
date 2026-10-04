# P2 readiness: Varhadi ASR adaptation

Date: 2026-10-04 · Status: **NOT READY TO TRAIN** (blockers below). Nothing has been trained.

## Verdict

The pipeline is prepared: manifests, config, Colab notebook, scoring and comparison all exist.
Training should not start yet, for three reasons:
1. **0 verified segments.** There is no test set, so Experiment A cannot be scored, and B would have
   nothing to be compared against. → A human must review JHN_001 in `scripts/review_server.py`.
2. **Alignment 40/89 chapters.** The background-job manager stopped it twice for low system memory;
   nothing failed. → Rerun `align --all`; it resumes at the remaining 49 chapters.
3. **CPU smoke test not yet run.** `experiments/varhadi_adaptation/cpu_smoke_test.sh` would
   check the config, manifests and the fork's fine-tune script together. It needs a few GB of
   free RAM, which this machine did not have during the session.

## Data (measured 2026-10-04)

| | Segments | Hours | Chapters | Labels |
|---|---|---|---|---|
| Candidate pool (aligned, in review queue) | 1,296 | 5.54 | 40 (JHN 21, LUK 19) | published text, **unverified** |
| Verified | **0** | 0 | 0 | — |
| Rejected / needs realignment | 0 | 0 | — | — |

Manifests from `build_manifests.py` (test chapter JHN_001, validation chapter JHN_002):

| Split | Strict default (verified only) | With `--allow-unverified-train --allow-unverified-val` |
|---|---|---|
| train | 0 | 1,249 segments, 5.35 h, 38 chapters, all unverified |
| val | 0 | 16 segments, 0.07 h (JHN_002 is short; use 2–3 val chapters) |
| test | 0 | 0 (the test split is **always** verified-only) |

- **Speakers:** unknown. VAHNT narrator identity is not available, so a speaker-level split is
  impossible. Chapters are held out instead (test and validation chapters never appear in training).
  This measures generalisation to unseen *text*, not to unseen *speakers*.
- **Domain:** general Varhadi scripture speech. **Not agricultural.** Native agricultural recordings:
  0 (workflow ready: `docs/data_collection.md`).
- **Segment lengths:** 2.1–40.7 s, median 15.6 s. 5 segments are over 30 s; the checkpoint's
  `max_duration: 30` drops them from training automatically.
- **Alignment quality signal:** `align_score` median −0.80; 54 segments are below −1.5. Inspection
  shows boundaries still matching, with the mismatch *inside* the segment. Example: the published
  text includes translator notes in parentheses that the narrator likely does not read. These need
  *needs_correction* in review. **Do not filter the test set on this score** (bias).

**Decision needed:** whether to train on unverified aligned candidates (weak labels). That is the
practical route with option (b), since 5 h cannot be hand-verified quickly, but it must be reported as
such. `bundle_info.json` records the choice. A sensible middle ground: verify the test chapter fully,
spot-check a random sample of training segments to estimate the label error rate, and report that rate.

## Model (read from the checkpoint, not assumed)

| Item | Value | Source |
|---|---|---|
| Checkpoint | `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`, file `indicconformer_stt_mr_hybrid_rnnt_large.nemo` (498 MB) | HF repo |
| Class | `EncDecHybridRNNTCTCBPEModel` (RNNT + aux CTC, CTC loss weight 0.3) | model_config.yaml |
| Encoder | Conformer, 17 layers, d_model 512, 8 heads, 4× striding subsampling, rel-pos attention | model_config.yaml |
| Tokenizer | multilingual, 22 langs × 256 SentencePiece BPE = 5632 tokens; separate softmax per language; Marathi = local ids 0–255 | model_config.yaml, verified on the loaded model |
| Features | 16 kHz, 80 mel, 25 ms window / 10 ms stride, per-feature normalisation | model_config.yaml |
| Data format | NeMo JSONL with `lang` (`return_language_id: true`); `offset` supported | fork source `manifest.py`, `collections.py` |
| Pretraining optimiser | AdamW, Noam (scale 0.3535, warmup 1500), batch 8, max 30 s | model_config.yaml |
| NeMo | AI4Bharat fork, branch nemo-v2, commit `8dce88cf8e94963e2033c3137f7b9993b51db88a`, 1.23.0rc0 | local env |

## Fine-tuning configuration (`experiments/varhadi_adaptation/finetune_varhadi.yaml`)

Uses the fork's own `examples/asr/speech_to_text_finetune.py`. Every value is tagged with its source.
- **From the fork's reference fine-tune config:** AdamW lr 1e-4, betas (0.9, 0.98),
  weight decay 1e-3, CosineAnnealing with warmup 5000 and min_lr 5e-6, max 50 epochs, best 5
  checkpoints by `val_wer`.
- **From the checkpoint:** batch 8, max/min duration 30/0.2 s, SpecAugment (2×27 freq, 10×0.05
  time), `return_language_id: true`, tokenizer unchanged.
- **Colab adaptations:** 1 device, `strategy: auto`, 2 workers, checkpoints on Drive,
  resume after disconnect.
- **Not tuned:** these are documented defaults, not an optimum. With about 1,250 training segments
  at batch 8 (about 157 steps/epoch), warmup 5000 is roughly 32 epochs, which is probably too long.
  Revisit warmup and epochs once the final train size is known, and record any change as a deviation.

## Compute

- **Where:** Google Colab GPU (`experiments/varhadi_adaptation/colab_finetune.ipynb`). The local
  machine is CPU-only (8 cores, ~7.7 GB available to WSL) and is used only for data prep, alignment,
  review and scoring.
- **GPU memory:** UNMEASURED. A 16 GB T4 with batch 8 of segments up to 30 s is expected to fit,
  based on the checkpoint's own batch size, but this is unverified. The notebook says to run
  `trainer.max_steps=20` first and record the memory and step time.
- **Training time and cost:** UNMEASURED. Steps per epoch ≈ train segments / 8 (≈157 for the current
  pool), so 50 epochs ≈ 7,800 steps. Wall time = steps × measured step time from the 20-step check.
  No estimate is given until that measurement exists.
- **CPU step time:** PENDING (smoke test not run). Not expected to be practical for real training.

## Risks
1. **Weak labels.** Published text ≠ narration in places (translator notes, wording). Training on
   unverified candidates teaches those errors; measure the label error rate on a sample.
2. **Single narrator, single domain.** Gains on VAHNT may not transfer to farmers' agricultural
   speech. The final claim needs a native agricultural test set from unseen speakers.
3. **Small, chapter-held-out test set.** The first results are PILOT only; there is no significance testing yet.
4. **Catastrophic forgetting.** Fine-tuning on 5 h of one speaker can degrade standard Marathi.
   Consider also scoring a Marathi set before/after (not yet available in the project).
5. **Colab environment drift.** Pinned fork commit, and the local pins (pytorch-lightning 2.2.1,
   datasets <3, numpy shim) are in the notebook; a newer Colab Python may still need fixes.
6. **Licence.** VAHNT audio and text licence is unverified; keep the bundle private (Drive) and do not
   publish an adapted model trained on it without checking bible.com's terms.

## Expected outputs (once run)
- `exp_dir/varhadi_adaptation/.../*.nemo`: best checkpoints by val_wer (on Drive).
- `experiments/varhadi_adapted/varhadi_adapted_{metrics.json,per_sample.csv,error_pairs.csv,report.md}`
  from `scripts/run_gold_eval.py --name varhadi_adapted --checkpoint best.nemo`.
- `experiments/evaluation/comparison.{csv,json,md}` from `compare.py`, which refuses to run unless
  A and B scored identical segment IDs.
- An `EXPERIMENT_LOG.md` entry with bundle_info.json, config deviations, GPU type, step time, epochs and the
  measured A vs B scores. No claim of improvement unless the measured scores show it.
