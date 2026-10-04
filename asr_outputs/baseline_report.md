# KisaanDost Marathi ASR Baseline on Varhadi Speech

## 1. Objective

How well does an existing pretrained Marathi ASR model transcribe Varhadi speech without any Varhadi-specific adaptation? This is Experiment 0 of the KisaanDost research plan (a region-adaptive multimodal assistant for Varhadi-speaking farmers in Vidarbha). No fine-tuning, correction, dictionary, or normalization is applied here.

## 2. Dataset

- Files attempted: 90
- Successful transcriptions: 90
- Failed: 0
- Files with ground-truth reference (WER/CER-eligible): 0
- Speakers identified: not recorded
- Total successfully transcribed duration: 706.4 minutes

Source: VAHNT (Varhadi-Nagpuri New Testament) audio Bible, downloaded by the existing `downloader.py` pipeline. See the README's Dataset Provenance section.

## 3. Audio

- Source format: MP3, as downloaded by `downloader.py`
- ASR input format: 16 kHz mono WAV, cached once under `asr_audio/` (`audio/` is never modified)

## 4. Model

- Model: `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`
- Decoder: ctc (hybrid CTC/RNNT checkpoint)
- Source: https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large
- Architecture: Conformer-Large (~120M parameters), hybrid CTC-RNNT
- Framework: NeMo (AI4Bharat fork, `nemo-v2` branch) — not plain nemo_toolkit or transformers

## 5. Hardware

- Platform: Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39
- CUDA available: False

## 6. Preprocessing

Source MP3 is decoded, resampled to 16 kHz, converted to mono, and cached once as WAV under `asr_audio/`. No Varhadi correction, dictionary, translation, spelling correction, or language-model post-processing is applied to the prediction at any stage.

## 7. Results

No WER/CER available: no samples in `asr_metadata.csv` currently have a ground-truth `reference_text`. Predictions were still generated in transcription-only mode; see `asr_outputs/predictions.csv`.

## 8. Error Analysis

See `asr_outputs/error_analysis.csv` for per-sample substitution/deletion/insertion counts. `manual_category` is left blank for human annotation; an error must not be auto-labeled as a specific cause (e.g. "Varhadi vocabulary") without review.

## 9. Limitations

- The model is a Marathi baseline; it has received no Varhadi-specific adaptation.
- The evaluated set (if any) may be small and not representative of all Varhadi speakers.
- Ground-truth transcript availability/alignment is currently limited or absent for most files.
- This audio is New Testament scripture, not agricultural speech; results do not generalize to farmer/agricultural vocabulary.

## 10. Next Research Step

Build a manually verified Varhadi speech-text evaluation corpus (ideally including agricultural speech), then perform Varhadi-specific adaptation (Experiment 1).
