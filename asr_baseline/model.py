from __future__ import annotations

import os
from dataclasses import dataclass

from .config import CHECKPOINT_FILENAME, DECODER, LANGUAGE_ID, MODEL_NAME


@dataclass
class LoadedModel:
    model: object
    device: str
    name: str = MODEL_NAME
    decoder: str = DECODER


def _shim_numpy_sctypes() -> None:
    """NeMo 1.23.0rc0's audio loader (segment.py) reads np.sctypes['int'/'float'],
    which NumPy 2.0 removed. Restore just those two keys rather than pinning numpy<2,
    which would fight pandas/scipy's own numpy>=2 requirement."""
    import numpy as np

    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "float": [np.float16, np.float32, np.float64],
        }


def load_model(checkpoint_path: str | None = None) -> LoadedModel:
    """Load the pretrained Marathi NeMo checkpoint. Requires the AI4Bharat NeMo fork
    (see requirements-asr.txt) to be installed; not compatible with plain transformers.

    Uses hf_hub_download + restore_from() rather than ASRModel.from_pretrained(MODEL_NAME):
    from_pretrained() guesses the checkpoint filename as f"{model_name}.nemo", which does not
    match this repo's actual file (CHECKPOINT_FILENAME, see config.py), and silently falls
    back to a broken code path. Downloading the known filename and calling restore_from()
    directly is what from_pretrained()'s own docstring recommends for a local .nemo file.

    `checkpoint_path`: a local .nemo (e.g. a Varhadi-adapted checkpoint from Colab) to
    load instead of the pretrained baseline. Default None = the baseline, unchanged.
    """
    import torch
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download

    _shim_numpy_sctypes()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = str(checkpoint_path) if checkpoint_path else MODEL_NAME
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            repo_id=MODEL_NAME, filename=CHECKPOINT_FILENAME, token=os.environ.get("HF_TOKEN")
        )
    model = nemo_asr.models.ASRModel.restore_from(restore_path=checkpoint_path, map_location=device)
    model.freeze()
    model = model.to(device)
    model.cur_decoder = DECODER
    return LoadedModel(model=model, device=device, name=name)


def transcribe_one(loaded: LoadedModel, wav_path: str) -> str:
    """Return the RAW ASR hypothesis for one 16kHz mono WAV file. No post-processing.

    In "ctc" mode this model's transcribe() returns a nested list (e.g. [[' text']])
    rather than a flat list of Hypothesis objects; unwrap to the string without touching
    its contents.
    """
    result = loaded.model.transcribe([wav_path], batch_size=1, language_id=LANGUAGE_ID)
    hypothesis = result[0]
    while isinstance(hypothesis, list):
        hypothesis = hypothesis[0]
    return hypothesis.text if hasattr(hypothesis, "text") else hypothesis
