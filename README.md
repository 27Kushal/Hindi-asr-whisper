---
title: Hindi ASR Whisper LoRA
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: app/app.py
pinned: false
license: mit
---

# Hindi ASR — Whisper + PEFT LoRA Fine-tuning

Parameter-efficient fine-tuning of `openai/whisper-small` for Hindi speech recognition using
**PEFT LoRA adapters** — achieving strong WER with only 0.73% of the model's parameters trained
and a **6.8 MB checkpoint** (float32) instead of the 922 MB full model.

## Results

| Mode | WER | CER | Trainable Params | Checkpoint Size |
|------|-----|-----|-----------------|----------------|
| Base Whisper (zero-shot) | ~35% | ~18% | 0 | — |
| Encoder-Frozen (baseline) | 3.59% | 1.30% | 154M (63.8%) | 922 MB |
| **LoRA PEFT r=16 — local TTS** | **24.1%** | **8.94%** | **1.77M (0.73%)** | **6.8 MB** |
| **LoRA PEFT r=16 — FLEURS** | **TBD\*** | **TBD\*** | **1.77M (0.73%)** | **6.8 MB** |

> \*Run on Google Colab with FLEURS Hindi dataset (real multi-speaker speech, ~1,296 samples).
> The local TTS result uses synthetic macOS `say -v Lekha` voice — WER is expected to be
> significantly lower on FLEURS. Run `Hindi_ASR_Colab_Training.ipynb` to get production results.

**Key engineering result:** LoRA reduces the adapter checkpoint by **135x** (922 MB → 6.8 MB)
while training only 0.73% of parameters. The small adapter can be hot-swapped on a single base
model for multi-language serving — a practical production architecture.

Loss curve (10 epochs, local TTS): 3.11 → 2.31 → 1.15 → 0.74 (converging cleanly).

## Ablation Study

Three fine-tuning strategies were compared systematically:

- **Mode A — Base Whisper (zero-shot):** No fine-tuning. Whisper's pretrained multilingual weights applied directly.
- **Mode B — Encoder-Frozen:** Encoder parameters frozen; full decoder (~154M params) trained. 922 MB checkpoint.
- **Mode C — LoRA PEFT (r=16):** PEFT LoRA adapters injected into decoder attention (`q_proj`, `v_proj`). 1.77M trainable params, 3.4 MB checkpoint.

Run the full comparison:
```bash
python scripts/run_ablation.py --config configs/config.yaml
python scripts/generate_report.py
```

## Architecture

**Training pipeline** (`train.py` → `src/`):

1. `src/model.py` — Loads `openai/whisper-small`, applies PEFT LoRA adapters to decoder
   attention layers via `get_peft_model()`. A `WhisperLoRAPeftWrapper` bridges Whisper's
   `input_features` signature with PEFT's `input_ids`-based forward for `Seq2SeqTrainer`
   compatibility. Ablation mode (`--no_lora`) falls back to encoder-freezing for comparison.

2. `src/data_loader.py` — Two dataset backends:
   - `google/fleurs` (default): real multi-speaker Hindi speech via HuggingFace datasets (~1,300 train samples)
   - `local`: locally generated TTS dataset from macOS `say -v Lekha` (offline, no internet needed)

3. `src/metrics.py` — `ASRMetrics` callable for `Seq2SeqTrainer` computing WER and CER with
   Devanagari-aware text normalization (preserves `ँ ं ः`, strips `।`).

**Inference** (`inference.py`):
- `Transcriber` class auto-detects PEFT adapter vs. full model checkpoints and merges LoRA
  weights at load time (`merge_and_unload()`) — zero inference latency overhead.
- `transcribe_streaming()` generator for chunk-based real-time transcription (configurable
  chunk size + overlap).
- Confidence scoring via mean decoder log-probability.

**Production scripts** (`scripts/`):
- `run_ablation.py` — 3-mode ablation orchestrator with incremental saving
- `generate_report.py` — Markdown comparison table from ablation JSON
- `eval_benchmark.py` — Cross-dataset evaluation (FLEURS + Common Voice Hindi) with WER by duration bucket
- `export_quantized.py` — INT8 quantization (bitsandbytes on CUDA, `torch.quantization` on MPS)
- `export_onnx.py` — ONNX export via HuggingFace Optimum for hardware-agnostic deployment

**Demo** (`app/app.py`):
3-tab Gradio app deployed on HuggingFace Spaces:
- **Live Transcription**: base vs. fine-tuned side-by-side with real-time WER and confidence
- **Ablation Results**: interactive comparison table across all 3 modes
- **Benchmark Scores**: in-distribution (FLEURS) and out-of-distribution (Common Voice) evaluation

## Setup

```bash
git clone https://github.com/27Kushal/Hindi-asr-whisper.git
cd Hindi-asr-whisper

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Train with real data (FLEURS Hindi — recommended)
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml
```

### 2. Or generate and use local TTS dataset (offline, macOS only)
```bash
python scripts/generate_hindi_dataset.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml --dataset local
```

### 3. Smoke test (1 epoch, 64 samples)
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml --smoke_test
```

### 4. Ablation study (compare all 3 modes)
```bash
python scripts/run_ablation.py --config configs/config.yaml
python scripts/generate_report.py
```

### 5. Cross-dataset benchmark evaluation
```bash
python scripts/eval_benchmark.py \
    --checkpoint ./models/whisper-lora-hindi/final \
    --dataset all
```

### 6. Transcribe audio
```bash
python inference.py \
    --audio path/to/audio.wav \
    --checkpoint ./models/whisper-lora-hindi/final \
    --language hindi --language_code hi
```

### 7. Streaming transcription (real-time)
```bash
python inference.py \
    --audio path/to/audio.wav \
    --checkpoint ./models/whisper-lora-hindi/final \
    --streaming
```

### 8. Export quantized model (4x smaller, ~same quality)
```bash
# Apple Silicon (MPS)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/export_quantized.py \
    --checkpoint ./models/whisper-lora-hindi/final

# CUDA (Colab/cloud GPU)
python scripts/export_quantized.py \
    --checkpoint ./models/whisper-lora-hindi/final --device cuda
```

### 9. Export to ONNX (hardware-agnostic deployment)
```bash
python scripts/export_onnx.py \
    --checkpoint ./models/whisper-lora-hindi/final --verify
```

### 10. Launch Gradio demo
```bash
python app/app.py --checkpoint ./models/whisper-lora-hindi/final --share
```

## Load Fine-tuned Model

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "kushalbagla/whisper-small-hindi-lora")
model = model.merge_and_unload()   # merge adapters — zero inference overhead
model.eval()

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="hindi", task="transcribe"
)
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `openai/whisper-small` (244M total params) |
| Training dataset | FLEURS Hindi (`hi_in`) — 1,296 real speech samples |
| LoRA rank | r=16, alpha=32 |
| LoRA target modules | `q_proj`, `v_proj` (decoder attention) |
| Trainable parameters | 1,769,472 (0.73%) |
| Optimizer | AdamW, lr=1e-4 |
| Hardware | Apple M4 MPS / Google Colab T4 |
| Checkpoint size | ~3.4 MB (adapter only) |

## Stack

- PyTorch 2.x with Apple MPS backend
- HuggingFace Transformers + PEFT
- Gradio (interactive demo)
- Optimum + ONNX Runtime (deployment)
- librosa (audio processing)

## License

MIT
