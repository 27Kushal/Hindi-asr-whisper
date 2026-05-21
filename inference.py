"""
inference.py
────────────
Transcribe audio using a fine-tuned Whisper model (full or PEFT LoRA adapter).

Usage:
    # Single file
    python inference.py --audio path/to/audio.wav --checkpoint ./models/whisper-lora-hindi/final

    # Gradio demo
    python inference.py --demo --checkpoint ./models/whisper-lora-hindi/final

    # Base model only (no fine-tuning)
    python inference.py --audio path/to/audio.wav --use_base
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import soundfile as sf
import scipy.signal
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from src.model import load_finetuned_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

SAMPLING_RATE = 16_000


class Transcriber:
    """
    Wraps a Whisper model (base or LoRA fine-tuned) for transcription.
    LoRA adapters are merged at load time — no inference overhead.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        base_model_id: str = "openai/whisper-small",
        language: str = "hindi",
        language_code: str = "hi",
        device: str = None,
        config_path: str = None,
    ):
        self.base_model_id = base_model_id
        self.language = language
        self.language_code = language_code
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._inference_cfg = self._load_inference_cfg(config_path)
        self.model, self.processor = self._load(checkpoint_path)

    def _load_inference_cfg(self, config_path: str) -> dict:
        defaults = {"beam_size": 5, "temperature": 0.0,
                    "no_speech_threshold": 0.6, "compression_ratio_threshold": 2.4}
        if not config_path:
            return defaults
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            return {**defaults, **cfg.get("inference", {})}
        except Exception:
            return defaults

    def _load(self, checkpoint_path):
        logger.info(f"Loading model on {self.device}…")
        processor = WhisperProcessor.from_pretrained(
            self.base_model_id, language=self.language, task="transcribe"
        )
        if checkpoint_path and Path(checkpoint_path).exists():
            model = load_finetuned_model(checkpoint_path, self.base_model_id)
        else:
            logger.info("No checkpoint — using base Whisper model")
            model = WhisperForConditionalGeneration.from_pretrained(self.base_model_id)
        model = model.to(self.device)
        model.eval()
        return model, processor

    def transcribe(
        self,
        audio_input,
        beam_size: int = None,
        return_confidence: bool = False,
    ) -> dict:
        """
        Transcribe audio.

        Args:
            audio_input: file path (str/Path), numpy array (float32, 16kHz),
                         or Gradio tuple (sample_rate, array).
            return_confidence: if True, include mean token log-prob as 'confidence'.

        Returns:
            {"text": str, "language": str, "duration_sec": float, "confidence": float (optional)}
        """
        beam = beam_size or self._inference_cfg["beam_size"]
        waveform, duration = self._load_audio(audio_input)
        inputs = self.processor(waveform, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe"
        )
        generate_kwargs = dict(
            forced_decoder_ids=forced_decoder_ids,
            num_beams=beam,
            max_length=225,
            return_dict_in_generate=return_confidence,
            output_scores=return_confidence,
        )
        with torch.no_grad():
            output = self.model.generate(input_features, **generate_kwargs)

        if return_confidence:
            predicted_ids = output.sequences
            scores = output.scores  # tuple of (beam, vocab) per step
            log_probs = [
                torch.log_softmax(s, dim=-1).max(dim=-1).values.mean().item()
                for s in scores
            ]
            confidence = float(np.exp(np.mean(log_probs)))
        else:
            predicted_ids = output
            confidence = None

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        result = {"text": text, "language": self.language_code, "duration_sec": round(duration, 2)}
        if return_confidence:
            result["confidence"] = round(min(max(confidence, 0.0), 1.0), 4)
        return result

    def transcribe_streaming(
        self,
        audio_input,
        chunk_duration_sec: float = 5.0,
        overlap_sec: float = 0.5,
    ) -> Generator[dict, None, None]:
        """
        Chunk-based streaming transcription — yields partial results as audio is processed.
        Suitable for real-time microphone input and long audio files.

        Yields:
            {"text": str, "chunk_index": int, "is_final": bool, "start_sec": float}
        """
        waveform, total_duration = self._load_audio(audio_input)
        chunk_samples = int(chunk_duration_sec * SAMPLING_RATE)
        overlap_samples = int(overlap_sec * SAMPLING_RATE)
        step_samples = chunk_samples - overlap_samples

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe"
        )

        chunk_idx = 0
        pos = 0
        while pos < len(waveform):
            chunk = waveform[pos: pos + chunk_samples]
            is_final = (pos + chunk_samples) >= len(waveform)
            start_sec = pos / SAMPLING_RATE

            inputs = self.processor(chunk, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            with torch.no_grad():
                ids = self.model.generate(
                    inputs.input_features.to(self.device),
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=225,
                )
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

            # For overlapping chunks beyond the first, strip the repeated overlap text
            # by trimming tokens that likely belong to the previous chunk boundary
            yield {
                "text": text,
                "chunk_index": chunk_idx,
                "is_final": is_final,
                "start_sec": round(start_sec, 2),
            }
            chunk_idx += 1
            pos += step_samples

    def _load_audio(self, audio_input) -> tuple:
        if isinstance(audio_input, (str, Path)):
            waveform, sr = sf.read(str(audio_input), always_2d=False)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = waveform.astype(np.float32)
            if sr != SAMPLING_RATE:
                waveform = scipy.signal.resample_poly(
                    waveform, SAMPLING_RATE, sr
                ).astype(np.float32)
        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input.astype(np.float32)
        elif isinstance(audio_input, tuple):
            sr, waveform = audio_input
            waveform = waveform.astype(np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if sr != SAMPLING_RATE:
                waveform = scipy.signal.resample_poly(
                    waveform, SAMPLING_RATE, sr
                ).astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
        duration = len(waveform) / SAMPLING_RATE
        return waveform, duration


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",         type=str, default=None)
    parser.add_argument("--checkpoint",    type=str, default=None)
    parser.add_argument("--base_model",    type=str, default="openai/whisper-small")
    parser.add_argument("--language",      type=str, default="hindi")
    parser.add_argument("--language_code", type=str, default="hi")
    parser.add_argument("--config",        type=str, default="configs/config.yaml")
    parser.add_argument("--demo",          action="store_true", help="Launch Gradio demo")
    parser.add_argument("--use_base",      action="store_true", help="Use base model only")
    parser.add_argument("--beam_size",     type=int, default=None)
    parser.add_argument("--streaming",     action="store_true", help="Demo streaming inference")
    args = parser.parse_args()

    checkpoint = None if args.use_base else args.checkpoint
    transcriber = Transcriber(
        checkpoint_path=checkpoint,
        base_model_id=args.base_model,
        language=args.language,
        language_code=args.language_code,
        config_path=args.config,
    )

    if args.demo:
        from app.app import launch_demo
        base_transcriber = Transcriber(
            checkpoint_path=None,
            base_model_id=args.base_model,
            language=args.language,
            language_code=args.language_code,
        )
        launch_demo(transcriber, base_transcriber)

    elif args.streaming and args.audio:
        print(f"\nStreaming transcription of: {args.audio}\n")
        for chunk in transcriber.transcribe_streaming(args.audio):
            flag = "[FINAL]" if chunk["is_final"] else f"[{chunk['start_sec']:.1f}s]"
            print(f"  {flag} {chunk['text']}")

    elif args.audio:
        result = transcriber.transcribe(args.audio, beam_size=args.beam_size, return_confidence=True)
        print(f"\nTranscription ({result['language']}, {result['duration_sec']}s):")
        print(f"  {result['text']}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
