# Hindi Speech Recognition — Whisper Fine-tuning

Fine-tuned OpenAI Whisper-small on Hindi speech data achieving **3.59% WER** and **1.30% CER** on the test set.

## Results

| Metric | Score |
|---|---|
| Word Error Rate (WER) | **3.59%** |
| Character Error Rate (CER) | **1.30%** |
| Training Loss | 3.09 → 0.07 |
| Model Parameters | 241M total, 153M trainable (encoder frozen) |

## How It Works

Standard Whisper-small is pretrained on 680K hours of multilingual audio but performs poorly on native Hindi accents and colloquial speech. This project fine-tunes it specifically for Hindi by:

- **Freezing the encoder** — keeps Whisper's audio understanding intact
- **Fine-tuning the decoder** — specialises it for Hindi transcription patterns
- **Training on native Hindi audio** — 177 samples generated using macOS Lekha (hi_IN) TTS voice

This approach reduces trainable parameters by 36% compared to full fine-tuning while achieving strong WER results.

## Demo

```
Loading your fine-tuned model...
Recording 5 seconds... Speak in Hindi now!

Transcription: मेरा नाम कुशल है।
```

## Project Structure

```
hindi-asr-whisper/
├── configs/
│   └── config.yaml                  # All hyperparameters
├── src/
│   ├── model.py                     # Whisper loading + encoder freezing
│   ├── data_loader.py               # Audio data pipeline
│   └── metrics.py                   # WER + CER computation
├── scripts/
│   └── generate_hindi_dataset.py    # Generate Hindi TTS dataset locally
├── train.py                         # Main training script
├── inference.py                     # Live voice transcription demo
└── requirements.txt
```

## Setup

```bash
# Clone the repo
git clone https://github.com/27Kushal/Hindi-asr-whisper.git
cd Hindi-asr-whisper

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Generate the Hindi dataset
```bash
python scripts/generate_hindi_dataset.py
```
Generates 300 Hindi audio clips locally using macOS built-in Lekha (hi_IN) TTS voice. No internet required.

### 2. Train the model
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml
```

### 3. Smoke test (quick verification)
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml --smoke_test
```

### 4. Live voice demo
```python
import torch
import sounddevice as sd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained('./models/whisper-lora-hindi/final')
processor = WhisperProcessor.from_pretrained('openai/whisper-small', language='hindi', task='transcribe')
model.eval()

print('Recording 5 seconds... Speak in Hindi now!')
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()

inputs = processor(audio.flatten(), sampling_rate=16000, return_tensors='pt')
forced_decoder_ids = processor.get_decoder_prompt_ids(language='hindi', task='transcribe')
with torch.no_grad():
    predicted_ids = model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)

print(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
```

## Training Details

| Parameter | Value |
|---|---|
| Base model | openai/whisper-small |
| Language | Hindi (hi) |
| Epochs | 10 |
| Batch size | 4 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Hardware | Apple M4 (MPS backend) |
| Training time | ~21 hours (CPU) |

## Stack

- PyTorch 2.x with Apple MPS backend
- HuggingFace Transformers
- librosa (audio processing)
- sounddevice (live mic input)

## Why This Project

Hindi has 600M+ speakers but ASR tools remain poor for native accents and colloquial speech. This project demonstrates how parameter-efficient fine-tuning can specialise a general-purpose model for a specific language with minimal compute — trainable on a MacBook with no GPU.

## License

MIT