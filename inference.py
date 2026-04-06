"""
inference.py
────────────
Transcribe audio files using a trained Whisper+LoRA model.
Also serves a Gradio demo for interactive testing.

Usage:
    # Transcribe a single file
    python inference.py --audio path/to/audio.wav --checkpoint ./models/whisper-lora-tamil/final

    # Launch Gradio web demo
    python inference.py --demo --checkpoint ./models/whisper-lora-tamil/final

    # Use the base Whisper model (no fine-tuning) for comparison
    python inference.py --audio path/to/audio.wav --use_base
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent))
from transformers import WhisperForConditionalGeneration
def load_finetuned_model(checkpoint_path, base_model_id):
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    model.eval()
    return model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SAMPLING_RATE = 16_000


# ──────────────────────────────────────────────────────────
#  Core transcription
# ──────────────────────────────────────────────────────────
class Transcriber:
    """
    Wraps Whisper (base or LoRA fine-tuned) for transcription.

    After training, LoRA weights are merged into the base model
    (merge_and_unload), so inference has identical speed.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        base_model_id: str = "openai/whisper-small",
        language: str = "tamil",
        language_code: str = "ta",
        device: str = None,
    ):
        self.base_model_id = base_model_id
        self.language = language
        self.language_code = language_code
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = self._load(checkpoint_path)

    def _load(self, checkpoint_path):
        logger.info(f"Loading model on {self.device}...")
        processor = WhisperProcessor.from_pretrained(
            self.base_model_id,
            language=self.language,
            task="transcribe",
        )
        if checkpoint_path and Path(checkpoint_path).exists():
            model = load_finetuned_model(checkpoint_path, self.base_model_id)
        else:
            logger.info("No checkpoint found — using base Whisper model")
            model = WhisperForConditionalGeneration.from_pretrained(self.base_model_id)

        model = model.to(self.device)
        model.eval()
        return model, processor

    def transcribe(
        self,
        audio_input,          # path (str) or numpy array (float32, 16kHz)
        beam_size: int = 5,
    ) -> dict:
        """
        Transcribe audio.

        Returns:
            {
                "text": "transcribed text",
                "language": "ta",
                "duration_sec": 4.2,
            }
        """
        waveform, duration = self._load_audio(audio_input)

        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language,
            task="transcribe",
        )

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=beam_size,
                max_length=225,
            )

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return {
            "text": transcription,
            "language": self.language_code,
            "duration_sec": round(duration, 2),
        }

    def _load_audio(self, audio_input) -> tuple:
        if isinstance(audio_input, (str, Path)):
            waveform, sr = librosa.load(str(audio_input), sr=SAMPLING_RATE, mono=True)
        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input.astype(np.float32)
        elif isinstance(audio_input, tuple):
            # Gradio returns (sample_rate, numpy_array)
            sr, waveform = audio_input
            waveform = waveform.astype(np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)  # stereo → mono
            if sr != SAMPLING_RATE:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=SAMPLING_RATE)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        duration = len(waveform) / SAMPLING_RATE
        return waveform, duration


# ──────────────────────────────────────────────────────────
#  Gradio demo
# ──────────────────────────────────────────────────────────
def launch_demo(transcriber: Transcriber, base_transcriber: Transcriber = None):
    """
    Gradio demo with side-by-side comparison:
      LEFT:  Base Whisper (no fine-tuning)
      RIGHT: Fine-tuned Whisper + LoRA

    This is the key thing to show in your project demo:
    the fine-tuned model handles accented / noisy audio much better.
    """
    import gradio as gr

    def transcribe_both(audio):
        if audio is None:
            return "No audio provided", "No audio provided"

        finetuned_result = transcriber.transcribe(audio)
        finetuned_text = finetuned_result["text"]

        if base_transcriber:
            base_result = base_transcriber.transcribe(audio)
            base_text = base_result["text"]
        else:
            base_text = "Base model not loaded"

        return base_text, finetuned_text

    language = transcriber.language.capitalize()

    with gr.Blocks(title=f"Whisper LoRA — {language} ASR") as demo:
        gr.Markdown(f"## Whisper + LoRA Fine-tuned ASR — {language}")
        gr.Markdown(
            "Record or upload audio and compare the **base Whisper** model "
            "against the **LoRA fine-tuned** model. "
            "The fine-tuned model should handle native accents and colloquial speech better."
        )

        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Input Audio",
            )

        with gr.Row():
            btn = gr.Button("Transcribe", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Whisper (no fine-tuning)")
                base_output = gr.Textbox(label="Transcription", lines=4)

            with gr.Column():
                gr.Markdown(f"### Fine-tuned Whisper + LoRA ({language})")
                finetuned_output = gr.Textbox(label="Transcription", lines=4)

        btn.click(
            fn=transcribe_both,
            inputs=[audio_input],
            outputs=[base_output, finetuned_output],
        )

        gr.Markdown("""
        ---
        **How this works:**
        - Base model: `openai/whisper-small` — trained on 680K hours of multilingual audio
        - Fine-tuned: same model with LoRA adapters trained on AI4Bharat IndicSUPERB data
        - LoRA trains only ~3% of parameters — but specialises the model for the target language and accent
        """)

    demo.launch(share=True)


# ──────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",       type=str, default=None)
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--base_model",  type=str, default="openai/whisper-small")
    parser.add_argument("--language",    type=str, default="tamil")
    parser.add_argument("--language_code", type=str, default="ta")
    parser.add_argument("--demo",        action="store_true", help="Launch Gradio demo")
    parser.add_argument("--use_base",    action="store_true", help="Use base model only")
    parser.add_argument("--beam_size",   type=int, default=5)
    args = parser.parse_args()

    checkpoint = None if args.use_base else args.checkpoint

    transcriber = Transcriber(
        checkpoint_path=checkpoint,
        base_model_id=args.base_model,
        language=args.language,
        language_code=args.language_code,
    )

    if args.demo:
        base_transcriber = Transcriber(
            checkpoint_path=None,  # always base for comparison
            base_model_id=args.base_model,
            language=args.language,
            language_code=args.language_code,
        )
        launch_demo(transcriber, base_transcriber)

    elif args.audio:
        result = transcriber.transcribe(args.audio, beam_size=args.beam_size)
        print(f"\nTranscription ({result['language']}, {result['duration_sec']}s):")
        print(f"  {result['text']}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
