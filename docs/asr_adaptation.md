# Varhadi ASR adaptation (P2)

**Status: NOT STARTED.** Nothing has been trained, and no adapted model exists. Readiness, data
counts, configuration, cost and risks are in [`P2_READY.md`](../P2_READY.md).

## Experiment design
- **A (baseline):** `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`, pretrained, no changes.
- **B (Varhadi-adapted):** the same checkpoint fine-tuned on Varhadi data (transfer learning,
  not training from scratch), keeping its tokenizer.
- Both are scored by `scripts/run_gold_eval.py` on the **same** human-verified, held-out test
  chapters, with the same punctuation-only normalisation. `experiments/evaluation/compare.py`
  refuses to compare runs that scored different segment IDs.

## Files
| Path | Purpose |
|---|---|
| `experiments/varhadi_adaptation/checkpoint_model_config.yaml` | the checkpoint's own config, extracted from the `.nemo` (source of truth) |
| `experiments/varhadi_adaptation/checkpoint_config_summary.txt` | readable summary of the above |
| `experiments/varhadi_adaptation/build_manifests.py` | NeMo train/val/test manifests (chapter holdout, test = verified only) |
| `experiments/varhadi_adaptation/finetune_varhadi.yaml` | fine-tuning config for the fork's `speech_to_text_finetune.py`; every value is source-tagged |
| `experiments/varhadi_adaptation/colab_finetune.ipynb` | Colab GPU run: env at the pinned fork commit, data bundle from Drive, training, checkpoints to Drive |
| `scripts/run_gold_eval.py` | scoring on verified references (`--checkpoint` for B) |
| `experiments/evaluation/compare.py` | `comparison.{csv,json,md}` |

## Key facts read from the checkpoint (not assumed)
- Hybrid RNNT + CTC (`EncDecHybridRNNTCTCBPEModel`), CTC loss weight 0.3.
- Conformer encoder: 17 layers, d_model 512, 8 heads, 4× subsampling.
- Input: 16 kHz, 80 mel features (25 ms window, 10 ms stride).
- Multilingual tokenizer: 22 languages × 256 BPE = 5632 tokens, with a separate softmax per
  language. Training therefore needs `return_language_id: true`, and each manifest line needs
  `"lang": "mr"`, which the fork reads as `item['lang']`.
- Pretraining optimiser: AdamW, Noam schedule (lr scale 0.3535, warmup 1500, d_model 512),
  batch 8, max duration 30 s, SpecAugment (2 freq masks of width 27, 10 time masks of 0.05).
