# VAHNT Audio Downloader

Downloads the 260 New Testament chapter MP3 files for the Varhadi-Nagpuri New Testament (`VAHNT`, Bible version `3451`). This milestone collects audio only; transcript extraction, TTS, alignment, speech-to-text, and dataset processing are intentionally not included.

## Installation

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Commands

```powershell
python downloader.py
python downloader.py --book MAT
python downloader.py --book MAT --headless
python downloader.py --chapter MAT.1.VAHNT
python downloader.py --retry-failed
python downloader.py --status
```

Chromium is visible by default. Add `--headless` for unattended runs. Downloads are sequential and wait two seconds between chapters by default; change that with `--delay SECONDS`.

## Output and resume behavior

Audio is saved as `audio/{BOOK}_{CHAPTER:03d}.mp3`. `progress.json` records successful chapter IDs, `failed.json` records chapters that exhausted three attempts, and `chapter_metadata.json` records the source page and local file for every success. A file is skipped only after it passes the minimum-size and MP3-header validation, even if it is already listed in progress.

Logs are written to `logs/downloader.log`. Downloads use Playwright to observe the page's actual audio response, then reuse the browser user agent and cookies for a direct `requests` download. No CDN URL is assumed or constructed.

## Tests

```powershell
python -m pytest
```

The tests are offline and do not download Bible.com content.

## Current limitations

- Only direct MP3 resources are supported. A streaming manifest is reported as a failure rather than being converted.
- Validation is intentionally basic: file size and MP3 header checks are used; no transcript or audio dataset processing is performed.
- The downloader intentionally uses one browser session per chapter and no parallelism.

## Dataset provenance

Audio in `audio/` is the VAHNT (Varhadi-Nagpuri) New Testament audio Bible, scraped from
`bible.com` (version 3451). It is downloaded for research use in a final-year academic
project (KisaanDost). Downloading this audio does not by itself grant a license to
redistribute it or to train and publish a model on it — check bible.com's terms before
using it beyond local research experiments. This is scripture audio, not agricultural or
farmer speech; treat it only as general Varhadi speech exposure, not as a proxy for the
agricultural domain the wider project targets.

## ASR baseline (Experiment 0)

`asr_baseline/` is a separate, isolated pipeline that transcribes the downloaded audio with
a pretrained Marathi ASR model (`ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`, via
the AI4Bharat NeMo fork) and, where a ground-truth transcript is available, scores WER/CER
with `jiwer`. It does not modify `downloader.py`, `audio/`, `progress.json`, or any other
downloader state, and has its own resume mechanism based on `asr_outputs/predictions.csv`.

**No Varhadi correction of any kind is applied** — predictions are the raw model output, so
that a later Varhadi-adapted model can be compared against a clean baseline.

### Environment

NeMo's `pynini` dependency has no Windows wheel, so install and run the ASR pipeline in
WSL2/Linux (see `requirements-asr.txt` for exact steps), not the Windows venv used for the
downloader:

```bash
wsl
cd /mnt/e/Varhadi_Bible_Audio
python3 -m venv .venv-asr && source .venv-asr/bin/activate
pip install -r requirements-asr.txt
# then install the AI4Bharat NeMo fork per requirements-asr.txt's comments
```

### Commands

```bash
python scripts/validate_asr_dataset.py                 # inspect audio + reference availability
python scripts/run_asr_baseline.py --audio audio/example.mp3   # single file
python scripts/run_asr_baseline.py                      # batch, resumable via predictions.csv
python scripts/run_asr_baseline.py --force               # re-run everything
python -m asr_baseline.evaluate --predictions asr_outputs/predictions.csv  # WER/CER (only for rows with a reference)
python -m asr_baseline.error_analysis
python scripts/generate_asr_report.py
```

### Ground truth

`asr_metadata.csv` (`audio_path,reference_text,speaker_id`) is currently a header-only
template: no sentence-level Varhadi transcripts exist for this audio yet, so WER/CER cannot
be computed against it. Add rows with a manually transcribed `reference_text` to evaluate a
sample; audio without a row is still transcribed (transcription-only mode), just never scored.

### Tests

ASR unit tests (`tests/test_asr_*.py`) run with `requirements-asr.txt` installed, use mocked
audio/CSV fixtures, and never download the model. Run them the same way as the downloader's
tests: `python -m pytest`.