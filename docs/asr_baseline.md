# ASR Baseline (Experiment 0)

Pretrained Marathi ASR (`ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`, CTC decoder)
run on Varhadi speech with **no** fine-tuning, correction, dictionary, translation, or
normalization. The output is stored exactly as the model produced it.

## Pipeline

```
audio/*.mp3 (never modified)
  -> validate_audio
  -> ensure_wav_copy: 16 kHz mono WAV cached in asr_audio/
  -> duration <= max_chunk_sec ? transcribe_one(whole file)
                               : transcribe_chunked (below)
  -> asr_outputs/predictions.csv (one row per file, file-level resume)
  -> asr_baseline.evaluate (WER/CER only for rows with a verified reference)
```

## Chunked inference

**Why:** one-pass inference on a 473 s chapter got OOM-killed at about 6 GB RSS. WSL2 gets about 7.7 GB on this 16 GB machine, and Conformer self-attention scales quadratically with utterance length.

**Settings** (CLI flag > environment variable > default):

| Setting | CLI | Env | Default |
|---|---|---|---|
| target chunk length | `--chunk-sec` | `ASR_CHUNK_SEC` | 20 s |
| max chunk length (audio longer than this is chunked) | `--max-chunk-sec` | `ASR_CHUNK_MAX_SEC` | 30 s |
| overlap | `--overlap-sec` | `ASR_CHUNK_OVERLAP_SEC` | 0 s |

**Cut points:** for each chunk, the cut goes at the lowest-energy 25 ms frame between
`target` and `max` seconds from the chunk start. That is usually a pause between words.

**Stitching strategy:**
- **overlap = 0 (default):** chunks are contiguous and disjoint and are joined with a space. No audio is transcribed twice, so no words get duplicated.
- **overlap > 0:** each chunk starts `overlap` s before the previous cut. Edge words are heard
  with truncated context, so the two chunks often transcribe the overlap differently. The
  stitcher finds the longest common run of at least 2 words (`difflib.SequenceMatcher`)
  between the last 30 words so far and the first 30 words of the next chunk. It keeps the
  previous text up to the end of that run and continues the next chunk after it. With no
  such run, the texts are joined unchanged and the seam is logged; no words are guessed away.
  A single shared word is not trusted because common words like "होता" repeat constantly.

**Resume and failure recovery:** each chunk appends a row (`chunk_index, start_sec, end_sec,
prediction, status, error_message`) to `asr_outputs/chunks/<stem>.csv`. On rerun, chunks
whose latest row is `ok` are reused and failed chunks are retried. If any chunk still fails,
the file is recorded as `failed` in `predictions.csv` and retried on the next batch run.
`--force` clears both resume layers. Temporary chunk WAVs live in `asr_audio/chunks/` and
are deleted after each chunk. The downloader's `progress.json` is never read or written.

**Logs:** `logs/asr_baseline.log`.

## Measurements (real, 2026-10-04, CPU, WSL2, torch 2.14.0+cpu)

| Input | Mode | Chunks | Inference | Peak RSS |
|---|---|---|---|---|
| `JHN_001_sample30s.wav` (30 s) | single pass | 1 | 3.7–6.4 s | not measured |
| `JHN_001_sample30s.wav` (30 s) | chunked 10/15/0 | 3 | 2.6 s | not measured |
| `JHN_001.mp3` (473.6 s) | single pass | 1 | OOM-killed (~6 GB) | — |
| `JHN_001.mp3` (473.6 s) | chunked 20/30/0 | 19 | 53.9 s (~8.8x realtime) | 1.74 GB (1.68 GB is the loaded model) |

QUALITATIVE OBSERVATION (no reference exists, so this is not a score): on the 30 s
sample, chunking at silence with 0 overlap gave text nearly identical to the single
pass. It differed only in a few boundary-word spellings (`बन्याच्या`→`बनच्या`, `होत`→`होतं`),
because the model gets less acoustic context near a cut. With 2 s overlap, the first seam's
edge words were heard differently in each chunk (`बनच्या` vs `ह वन्याच्या`). That case is what
motivated the longest-common-run stitcher over exact suffix/prefix matching.

## Limitations
- WER/CER: NOT AVAILABLE. There are no verified Varhadi reference transcripts yet.
- Chunk boundaries can slightly change words near a cut compared with one-pass decoding.
  This has not been quantified because there are no references to quantify it against.
- The energy-based cut is a heuristic, not a VAD. Fully continuous speech with no pause in
  the target..max window is cut at the quietest frame anyway.
