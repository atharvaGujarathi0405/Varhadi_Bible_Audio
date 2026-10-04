#!/usr/bin/env bash
# CPU smoke test of the Colab fine-tuning path (WSL). NOT training: 2 steps on 4 short
# segments, output written to /tmp and discarded. Checks that finetune_varhadi.yaml, the
# manifest format (offset + lang) and the fork's speech_to_text_finetune.py work together,
# and measures CPU step time / peak memory.
#
#   bash experiments/varhadi_adaptation/cpu_smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-asr-wsl/bin/python
NEMO_EXAMPLES=/home/atharva/ai4bharat-nemo/examples/asr
WORK=/tmp/kd_smoke
rm -rf "$WORK" && mkdir -p "$WORK/manifests"

# 4 shortest train + 1 val segments from an existing bundle (built with --allow-unverified-train
# --allow-unverified-val purely for this pipeline check), paths made absolute.
$PY - "$WORK" <<'EOF'
import json, sys
work = sys.argv[1]
root = "data/processed/p2_bundle"
def pick(split, n):
    lines = [json.loads(l) for l in open(f"{root}/manifests/{split}_manifest.json", encoding="utf-8")]
    lines = sorted(lines, key=lambda l: l["duration"])[:n]
    for l in lines:
        l["audio_filepath"] = f"/mnt/e/KisaanDost/asr_audio/{l['audio_filepath'].split('/')[-1]}"
    with open(f"{work}/manifests/{split}.json", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(l, ensure_ascii=False) + "\n" for l in lines)
    print(split, [round(l["duration"], 1) for l in lines])
pick("train", 4)
pick("val", 1)
EOF

BASE=$($PY -c "from huggingface_hub import try_to_load_from_cache as t; print(t('ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large','indicconformer_stt_mr_hybrid_rnnt_large.nemo'))")

# numpy.sctypes shim (same as asr_baseline/model.py) via sitecustomize for the fork's script
mkdir -p "$WORK/shim" && cat > "$WORK/shim/sitecustomize.py" <<'EOF'
import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {"int": [np.int8, np.int16, np.int32, np.int64], "float": [np.float16, np.float32, np.float64]}
EOF

HF_HUB_OFFLINE=1 PYTHONPATH="$WORK/shim" /usr/bin/time -v $PY "$NEMO_EXAMPLES/speech_to_text_finetune.py" \
  --config-path="$PWD/experiments/varhadi_adaptation" --config-name=finetune_varhadi \
  init_from_nemo_model="$BASE" \
  model.train_ds.manifest_filepath="$WORK/manifests/train.json" \
  model.validation_ds.manifest_filepath="$WORK/manifests/val.json" \
  model.train_ds.batch_size=1 model.train_ds.num_workers=0 model.train_ds.pin_memory=false \
  model.validation_ds.batch_size=1 model.validation_ds.num_workers=0 model.validation_ds.pin_memory=false \
  trainer.accelerator=cpu trainer.max_epochs=1 +trainer.limit_train_batches=2 +trainer.limit_val_batches=1 \
  exp_manager.exp_dir="$WORK/exp" exp_manager.resume_if_exists=false \
  2>&1 | tee "$WORK/smoke.log" | grep -E --line-buffered "Epoch 0|it/s|s/it|val_wer|Error|Traceback|Maximum resident|Elapsed|Exit status" || true
echo "full log: $WORK/smoke.log"
