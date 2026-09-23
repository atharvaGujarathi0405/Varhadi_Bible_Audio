from __future__ import annotations

from config import AUDIO_DIR, ROOT_DIR  # reuse the downloader's paths; do not duplicate

# Model card: https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large
# Requires the AI4Bharat NeMo fork (nemo-v2 branch), not plain nemo_toolkit or transformers.
MODEL_NAME = "ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large"
# The checkpoint file inside the repo is named without "_ctc" (verified against the actual
# repo listing). NeMo 1.23.0rc0's from_pretrained() guesses the filename as
# f"{model_name}.nemo" and gets it wrong for this repo, which sends it down a broken
# "download whole repo as an already-extracted directory" fallback. model.py downloads this
# exact filename and calls restore_from() instead, per NeMo's own from_pretrained() docstring
# ("Use restore_from() to instantiate from a local .nemo file").
CHECKPOINT_FILENAME = "indicconformer_stt_mr_hybrid_rnnt_large.nemo"
LANGUAGE_ID = "mr"
DECODER = "ctc"  # hybrid checkpoint also exposes "rnnt"; ctc is the faster CPU path
SAMPLE_RATE = 16000

ASR_AUDIO_DIR = ROOT_DIR / "asr_audio"  # cached 16kHz mono WAV copies; audio/ is never touched
ASR_METADATA_PATH = ROOT_DIR / "asr_metadata.csv"

ASR_OUTPUTS_DIR = ROOT_DIR / "asr_outputs"
PREDICTIONS_PATH = ASR_OUTPUTS_DIR / "predictions.csv"
METRICS_PATH = ASR_OUTPUTS_DIR / "metrics.json"
PER_SAMPLE_METRICS_PATH = ASR_OUTPUTS_DIR / "per_sample_metrics.csv"
ERROR_ANALYSIS_PATH = ASR_OUTPUTS_DIR / "error_analysis.csv"
REPORT_PATH = ASR_OUTPUTS_DIR / "baseline_report.md"
PLOTS_DIR = ASR_OUTPUTS_DIR / "plots"
