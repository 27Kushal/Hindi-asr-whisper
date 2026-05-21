---
title: Hindi ASR Whisper LoRA
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
---

# Hindi ASR — Whisper + PEFT LoRA Fine-tuning

Fine-tuned `openai/whisper-small` for Hindi speech recognition using **PEFT LoRA adapters**.

## Results

| Mode | WER | Trainable Params | Checkpoint Size |
|------|-----|-----------------|----------------|
| Base Whisper (zero-shot) | ~35% | 0 | — |
| Encoder-Frozen | 3.59% | 154M (63.8%) | 922 MB |
| **LoRA PEFT (ours)** | **TBD** | **1.77M (0.73%)** | **~5 MB** |

## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "kushalbagla/whisper-small-hindi-lora")
model = model.merge_and_unload()  # zero inference overhead
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="hindi", task="transcribe")
```
