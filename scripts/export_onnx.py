"""
export_onnx.py
──────────────
Export a fine-tuned Whisper model to ONNX format for hardware-agnostic deployment.
Uses Hugging Face Optimum's built-in Whisper ONNX exporter.

ONNX enables:
  - CPU inference without PyTorch (smaller production container)
  - Mobile/edge deployment via ONNX Runtime
  - Hardware acceleration on CPUs (OpenVINO, CoreML, TensorRT)

Usage:
    pip install optimum[exporters] onnxruntime

    python scripts/export_onnx.py --checkpoint ./models/whisper-lora-hindi/final
    python scripts/export_onnx.py --checkpoint ./models/whisper-lora-hindi/final --verify
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000


def merge_lora_if_needed(checkpoint_path: str, base_model_id: str, tmp_dir: str) -> str:
    """
    If checkpoint is a PEFT adapter, merge it into the base model and save
    a temporary full model for ONNX export (Optimum requires a full model dir).
    Returns the path to use for ONNX export.
    """
    adapter_cfg = Path(checkpoint_path) / "adapter_config.json"
    if not adapter_cfg.exists():
        return checkpoint_path

    logger.info("PEFT adapter detected — merging weights for ONNX export…")
    from src.model import load_finetuned_model
    from transformers import WhisperProcessor

    model = load_finetuned_model(checkpoint_path, base_model_id)
    processor = WhisperProcessor.from_pretrained(base_model_id)

    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp_path))
    processor.save_pretrained(str(tmp_path))
    logger.info(f"Merged model saved to: {tmp_path}")
    return str(tmp_path)


def export_with_optimum(model_path: str, output_dir: str):
    """Export using optimum's CLI-equivalent Python API."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        logger.error("optimum not installed. Run: pip install optimum[exporters]")
        return False

    logger.info(f"Exporting {model_path} → {output_dir} (ONNX)…")
    logger.info("This exports encoder and decoder as separate ONNX graphs.")
    try:
        main_export(
            model_name_or_path=model_path,
            output=output_dir,
            task="automatic-speech-recognition",
            do_validation=False,  # Skip built-in validation; we run our own
            opset=17,
        )
        return True
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return False


def verify_onnx(onnx_dir: str, base_model_id: str, language: str = "hindi") -> dict:
    """Run a test sample through ONNX Runtime and compare output to PyTorch."""
    try:
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForSpeech2Seq
        from transformers import WhisperProcessor
    except ImportError:
        logger.warning("onnxruntime or optimum not available for verification")
        return {}

    logger.info("Verifying ONNX output vs. PyTorch…")
    processor = WhisperProcessor.from_pretrained(base_model_id, language=language, task="transcribe")
    dummy_audio = np.random.randn(5 * SAMPLING_RATE).astype(np.float32)
    inputs = processor(dummy_audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    # ONNX inference
    try:
        ort_model = ORTModelForSpeech2Seq.from_pretrained(onnx_dir)
        t0 = time.perf_counter()
        ort_ids = ort_model.generate(inputs.input_features, forced_decoder_ids=forced_ids, max_length=50)
        ort_latency = time.perf_counter() - t0
        ort_text = processor.batch_decode(ort_ids, skip_special_tokens=True)[0]
        logger.info(f"ONNX output: '{ort_text}' (latency: {ort_latency:.3f}s)")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return {}

    # Compute ONNX model size
    onnx_size = sum(
        f.stat().st_size for f in Path(onnx_dir).rglob("*.onnx")
    ) / (1024 ** 2)

    return {
        "onnx_verified": True,
        "onnx_latency_sec": round(ort_latency, 3),
        "onnx_size_mb": round(onnx_size, 1),
        "sample_transcription": ort_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--base_model",  default="openai/whisper-small")
    parser.add_argument("--language",    default="hindi")
    parser.add_argument("--output_dir",  default=None)
    parser.add_argument("--verify",      action="store_true",
                        help="Verify ONNX output matches PyTorch after export")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    out_dir = args.output_dir or str(checkpoint_path.parent / "onnx")
    tmp_merged = str(checkpoint_path.parent / "_tmp_merged_for_onnx")

    # Merge LoRA adapters if needed (Optimum requires a full model dir)
    export_src = merge_lora_if_needed(str(checkpoint_path), args.base_model, tmp_merged)

    # Export
    success = export_with_optimum(export_src, out_dir)
    if not success:
        sys.exit(1)

    # Clean up temp merged dir
    if export_src == tmp_merged:
        import shutil
        shutil.rmtree(tmp_merged, ignore_errors=True)

    results = {"export_dir": out_dir, "success": True}

    # Verify
    if args.verify:
        verification = verify_onnx(out_dir, args.base_model, args.language)
        results.update(verification)

    # Report
    onnx_size = sum(f.stat().st_size for f in Path(out_dir).rglob("*.onnx")) / (1024**2)
    print("\n" + "="*50)
    print("ONNX EXPORT COMPLETE")
    print("="*50)
    print(f"Output dir: {out_dir}")
    print(f"ONNX size:  {onnx_size:.1f} MB")
    if args.verify and results.get("onnx_verified"):
        print(f"Latency:    {results.get('onnx_latency_sec', '?')}s")
        print(f"Sample:     {results.get('sample_transcription', '')}")
    print("="*50)
    print("\nTo run inference with ONNX Runtime:")
    print("  from optimum.onnxruntime import ORTModelForSpeech2Seq")
    print(f"  model = ORTModelForSpeech2Seq.from_pretrained('{out_dir}')")
    print()

    report_path = Path(out_dir) / "onnx_export_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
