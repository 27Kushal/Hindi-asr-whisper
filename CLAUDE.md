# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate virtual environment (required before running anything)
source venv/bin/activate

# Generate the local Hindi TTS dataset (macOS only — uses built-in Lekha hi_IN voice)
python scripts/generate_hindi_dataset.py

# Quick smoke test — 1 epoch, 64 training samples
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml --smoke_test --dataset local

# Full training run on local TTS dataset
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml --dataset local

# Full training run on FLEURS (requires internet + Google Colab recommended)
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml

# Ablation study (3 modes: base zero-shot, encoder-frozen, LoRA PEFT)
python scripts/run_ablation.py --config configs/config.yaml --dataset local

# Transcribe a single audio file
python inference.py --audio path/to/audio.wav --checkpoint ./models/whisper-lora-hindi/final

# Launch Gradio demo
python app/app.py --checkpoint ./models/whisper-lora-hindi/final --share
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` is required on Apple Silicon because some Whisper ops lack native MPS kernels.

## Architecture

**Training pipeline** (`train.py` → `src/`):

1. `src/model.py` — `WhisperLoRAModel.build()` loads `openai/whisper-small` and applies real PEFT LoRA adapters to decoder attention (`q_proj`, `v_proj`) via `get_peft_model()`. The `WhisperLoRAPeftWrapper(nn.Module)` bridges Whisper's `input_features` signature with PEFT's `input_ids`-based forward. IMPORTANT: do NOT store `peft_model.base_model.model` as a separate attribute — it causes duplicate parameters in `state_dict`. `--no_lora` flag enables encoder-frozen mode (ablation B). `load_finetuned_model()` auto-detects adapter vs. full checkpoint and calls `merge_and_unload()` for zero inference overhead.

2. `src/data_loader.py` — Two backends: `google/fleurs` (real multi-speaker speech, ~1,296 train samples) and `local` (CSV-based TTS dataset at `data/hindi_tts/`). Audio loading uses `soundfile.read()` + `scipy.signal.resample_poly()` — **NOT librosa** (librosa requires numba which is not installed).

3. `src/metrics.py` — `ASRMetrics` for `Seq2SeqTrainer`, computing WER/CER with Devanagari-aware normalization (NFC, strip danda/double-danda, preserve anusvara/visarga).

**Training** (`train.py`):
Uses `PeftSeq2SeqTrainer` (subclasses `Seq2SeqTrainer`) that overrides `_save()` to call `model.save_pretrained()`. This is required because Whisper has weight-tied tensors (`proj_out.weight` = `decoder.embed_tokens.weight`) which safetensors rejects when saving directly via state_dict.

**Dataset generation** (`scripts/generate_hindi_dataset.py`):
Uses macOS `say -v Lekha` (hi_IN TTS voice) → AIFF → 16 kHz WAV via `afconvert`. Produces train/validation/test splits.

**Inference** (`inference.py`):
`Transcriber` auto-detects PEFT adapter checkpoints (checks for `adapter_config.json`) and merges at load time. `transcribe_streaming()` for chunk-based real-time processing. Audio loading uses `soundfile` + `scipy` — NOT librosa.

**Production scripts** (`scripts/`):
- `run_ablation.py` — 3-mode ablation with `--dataset`, `--cpu`, `--language` flags
- `generate_report.py` — Markdown table from `models/ablation/comparison_table.json`
- `eval_benchmark.py` — FLEURS + Common Voice evaluation with duration-bucket WER
- `export_quantized.py` — INT8 quantization (bitsandbytes/CUDA or torch.quantization/MPS)
- `export_onnx.py` — ONNX export via HuggingFace Optimum

**Gradio demo** (`app/app.py`):
3 tabs: Live Transcription, Ablation Results, Benchmark Scores. Imports from project root via `sys.path.insert(0, parent.parent)`.

## Configuration

All hyperparameters in `configs/config.yaml`. Key knobs:
- `data.dataset_name` — `"google/fleurs"` (real speech) or `"local"` (TTS)
- `data.local_data_dir` — path to CSV dataset (default `data/hindi_tts`)
- `lora.r` / `lora.lora_alpha` — LoRA rank and scaling (r=16, alpha=32)
- `training.output_dir` — auto-resumes from latest checkpoint if dir exists

CLI overrides: `--language`, `--language_code`, `--dataset`, `--no_lora`, `--cpu`, `--output_dir_override`

## Data layout

```
data/hindi_tts/
  train/
    train_0000.wav … train_0199.wav
    train.csv          # columns: path, sentence, split
  validation/
    validation.csv
  test/
    test.csv
models/
  whisper-lora-hindi/
    checkpoint-*/      # intermediate checkpoints (PEFT adapter only, ~3.4 MB each)
    final/             # adapter_config.json + adapter_model.safetensors
    test_results.json  # WER/CER, trainable params, checkpoint size
  ablation/
    comparison_table.json
    base_whisper/      # Mode A results
    encoder_frozen/    # Mode B results
    lora_peft/         # Mode C results
```

## HuggingFace Deployment

**Hub (model):** Push adapter to `kushalbagla/whisper-small-hindi-lora`:
```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='models/whisper-lora-hindi/final', repo_id='kushalbagla/whisper-small-hindi-lora')
"
```

**Spaces (demo):** The root `README.md` already has HF Spaces frontmatter (`app_file: app/app.py`). Push the repo to a HF Space with `sdk: gradio`.

## Known issues / gotchas

- `librosa` requires `numba` which is not in the venv — use `soundfile` + `scipy` for all audio I/O
- MPS may not be available in background/sandboxed processes; use `--cpu` flag if needed
- `Seq2SeqTrainer._save()` fails on Whisper's weight-tied tensors — always use `PeftSeq2SeqTrainer`
- PEFT 0.7.1 requires `accelerate==0.27.2` (newer accelerate breaks with transformers 4.36.2)
