#!/usr/bin/env python3
"""Generate asr_outputs/baseline_report.md from real pipeline outputs. No fabricated numbers."""
from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_baseline.config import DECODER, METRICS_PATH, MODEL_NAME, PREDICTIONS_PATH, REPORT_PATH


def gpu_status() -> str:
    try:
        import torch

        return f"CUDA available: {torch.cuda.is_available()}"
    except ImportError:
        return "torch not installed in this environment"


def main() -> int:
    if not PREDICTIONS_PATH.exists():
        print(f"No predictions found at {PREDICTIONS_PATH}; run scripts/run_asr_baseline.py first")
        return 1

    with PREDICTIONS_PATH.open(encoding="utf-8", newline="") as source:
        predictions = list(csv.DictReader(source))
    ok = [row for row in predictions if row.get("status") == "ok"]
    with_ref = [row for row in ok if row.get("reference", "").strip()]
    speakers = {row.get("speaker_id") for row in predictions if row.get("speaker_id")}
    total_duration_min = sum(float(row["duration_sec"]) for row in ok if row.get("duration_sec")) / 60

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else None

    lines = [
        "# KisaanDost Marathi ASR Baseline on Varhadi Speech",
        "",
        "## 1. Objective",
        "",
        "How well does an existing pretrained Marathi ASR model transcribe Varhadi speech "
        "without any Varhadi-specific adaptation? This is Experiment 0 of the KisaanDost "
        "research plan (a region-adaptive multimodal assistant for Varhadi-speaking farmers "
        "in Vidarbha). No fine-tuning, correction, dictionary, or normalization is applied here.",
        "",
        "## 2. Dataset",
        "",
        f"- Files attempted: {len(predictions)}",
        f"- Successful transcriptions: {len(ok)}",
        f"- Failed: {len(predictions) - len(ok)}",
        f"- Files with ground-truth reference (WER/CER-eligible): {len(with_ref)}",
        f"- Speakers identified: {len(speakers) if speakers else 'not recorded'}",
        f"- Total successfully transcribed duration: {total_duration_min:.1f} minutes",
        "",
        "Source: VAHNT (Varhadi-Nagpuri New Testament) audio Bible, downloaded by the "
        "existing `downloader.py` pipeline. See the README's Dataset Provenance section.",
        "",
        "## 3. Audio",
        "",
        "- Source format: MP3, as downloaded by `downloader.py`",
        "- ASR input format: 16 kHz mono WAV, cached once under `asr_audio/` (`audio/` is never modified)",
        "",
        "## 4. Model",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Decoder: {DECODER} (hybrid CTC/RNNT checkpoint)",
        "- Source: https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large",
        "- Architecture: Conformer-Large (~120M parameters), hybrid CTC-RNNT",
        "- Framework: NeMo (AI4Bharat fork, `nemo-v2` branch) — not plain nemo_toolkit or transformers",
        "",
        "## 5. Hardware",
        "",
        f"- Platform: {platform.platform()}",
        f"- {gpu_status()}",
        "",
        "## 6. Preprocessing",
        "",
        "Source MP3 is decoded, resampled to 16 kHz, converted to mono, and cached once as WAV "
        "under `asr_audio/`. No Varhadi correction, dictionary, translation, spelling correction, "
        "or language-model post-processing is applied to the prediction at any stage.",
        "",
        "## 7. Results",
        "",
    ]
    if metrics and metrics.get("evaluated_samples"):
        lines += [
            f"- Evaluated samples: {metrics['evaluated_samples']}",
            f"- WER: {metrics['wer']:.4f}",
            f"- CER: {metrics['cer']:.4f}",
        ]
    else:
        lines += [
            "No WER/CER available: no samples in `asr_metadata.csv` currently have a "
            "ground-truth `reference_text`. Predictions were still generated in "
            "transcription-only mode; see `asr_outputs/predictions.csv`.",
        ]
    lines += [
        "",
        "## 8. Error Analysis",
        "",
        "See `asr_outputs/error_analysis.csv` for per-sample substitution/deletion/insertion "
        "counts. `manual_category` is left blank for human annotation; an error must not be "
        'auto-labeled as a specific cause (e.g. "Varhadi vocabulary") without review.',
        "",
        "## 9. Limitations",
        "",
        "- The model is a Marathi baseline; it has received no Varhadi-specific adaptation.",
        "- The evaluated set (if any) may be small and not representative of all Varhadi speakers.",
        "- Ground-truth transcript availability/alignment is currently limited or absent for most files.",
        "- This audio is New Testament scripture, not agricultural speech; results do not "
        "generalize to farmer/agricultural vocabulary.",
        "",
        "## 10. Next Research Step",
        "",
        "Build a manually verified Varhadi speech-text evaluation corpus (ideally including "
        "agricultural speech), then perform Varhadi-specific adaptation (Experiment 1).",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
