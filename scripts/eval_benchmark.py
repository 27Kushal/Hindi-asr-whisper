"""
eval_benchmark.py
─────────────────
Evaluate a fine-tuned checkpoint against external benchmarks.
Tests out-of-distribution generalization — the real measure of ASR quality.

Datasets:
  - fleurs       : google/fleurs (same split used for training evaluation)
  - common_voice : mozilla-foundation/common_voice_11_0 (independent benchmark)

Usage:
    python scripts/eval_benchmark.py \\
        --checkpoint ./models/whisper-lora-hindi/final \\
        --dataset common_voice \\
        --language hindi \\
        --language_code hi

    # Evaluate both datasets
    python scripts/eval_benchmark.py \\
        --checkpoint ./models/whisper-lora-hindi/final \\
        --dataset all
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import scipy.signal
from datasets import load_dataset
from transformers import WhisperProcessor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import load_finetuned_model
from src.metrics import normalise_text, ASRMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000

# Duration buckets for stratified WER analysis
DURATION_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 30)]


def load_audio(audio_dict: dict) -> np.ndarray:
    """Load and resample audio from a HuggingFace audio dict."""
    waveform = np.array(audio_dict["array"], dtype=np.float32)
    sr = audio_dict["sampling_rate"]
    if sr != SAMPLING_RATE:
        waveform = scipy.signal.resample_poly(waveform, SAMPLING_RATE, sr).astype(np.float32)
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


def get_transcript_field(item: dict, dataset: str) -> str:
    """Get the transcript field — varies by dataset."""
    if dataset == "common_voice":
        return item.get("sentence", "").strip()
    elif dataset == "fleurs":
        return (item.get("transcription") or item.get("raw_transcription", "")).strip()
    return ""


def evaluate_dataset(
    transcriber_model,
    processor: WhisperProcessor,
    dataset_name: str,
    language_code: str,
    language: str,
    hf_dataset_id: str,
    hf_language_code: str,
    device: str,
    max_samples: int = None,
) -> dict:
    logger.info(f"Loading {hf_dataset_id} ({hf_language_code})…")
    try:
        raw = load_dataset(hf_dataset_id, hf_language_code, split="test", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load {hf_dataset_id}: {e}")
        return {}

    if max_samples:
        raw = raw.select(range(min(max_samples, len(raw))))
    logger.info(f"  {len(raw)} test samples")

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    predictions, references, durations = [], [], []

    for item in tqdm(raw, desc=f"Evaluating {dataset_name}"):
        transcript = get_transcript_field(item, dataset_name)
        if not transcript:
            continue
        waveform = load_audio(item["audio"])
        duration = len(waveform) / SAMPLING_RATE
        if duration > 30:
            continue

        inputs = processor(waveform, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        with torch.no_grad():
            ids = transcriber_model.generate(
                inputs.input_features.to(device),
                forced_decoder_ids=forced_decoder_ids,
                max_length=225,
            )
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        predictions.append(normalise_text(pred, language_code))
        references.append(normalise_text(transcript, language_code))
        durations.append(duration)

    if not predictions:
        logger.warning("No valid samples processed.")
        return {}

    from evaluate import load as eval_load
    wer_metric = eval_load("wer")
    cer_metric = eval_load("cer")

    pairs = [(p, r, d) for p, r, d in zip(predictions, references, durations) if r.strip()]
    preds, refs, durs = zip(*pairs)

    overall_wer = wer_metric.compute(predictions=list(preds), references=list(refs))
    overall_cer = cer_metric.compute(predictions=list(preds), references=list(refs))

    # Stratified WER by audio duration bucket
    bucket_results = {}
    for lo, hi in DURATION_BUCKETS:
        bucket_preds = [p for p, r, d in zip(preds, refs, durs) if lo <= d < hi]
        bucket_refs  = [r for p, r, d in zip(preds, refs, durs) if lo <= d < hi]
        if bucket_preds:
            b_wer = wer_metric.compute(predictions=bucket_preds, references=bucket_refs)
            bucket_results[f"{lo}-{hi}s"] = {"wer": round(b_wer, 4), "n": len(bucket_preds)}

    return {
        "dataset": dataset_name,
        "language_code": language_code,
        "num_samples": len(preds),
        "wer": round(overall_wer, 4),
        "cer": round(overall_cer, 4),
        "wer_by_duration": bucket_results,
    }


DATASET_CONFIGS = {
    "fleurs": {
        "hf_id": "google/fleurs",
        "get_hf_code": lambda lc: lc if "_" in lc else f"{lc}_in",  # hi → hi_in
    },
    "common_voice": {
        "hf_id": "mozilla-foundation/common_voice_11_0",
        "get_hf_code": lambda lc: lc.split("_")[0],  # hi_in → hi
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    required=True, help="Path to fine-tuned checkpoint")
    parser.add_argument("--dataset",       default="common_voice",
                        choices=["fleurs", "common_voice", "all"])
    parser.add_argument("--language",      default="hindi")
    parser.add_argument("--language_code", default="hi_in",
                        help="FLEURS-style language code (e.g. hi_in, ta_in)")
    parser.add_argument("--base_model",    default="openai/whisper-small")
    parser.add_argument("--max_samples",   type=int, default=None)
    parser.add_argument("--output_dir",    default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ISO code (hi) vs FLEURS code (hi_in)
    iso_code = args.language_code.split("_")[0]

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = load_finetuned_model(args.checkpoint, args.base_model)
    model = model.to(device)
    model.eval()

    processor = WhisperProcessor.from_pretrained(
        args.base_model, language=args.language, task="transcribe"
    )

    datasets_to_eval = (
        ["fleurs", "common_voice"] if args.dataset == "all" else [args.dataset]
    )

    all_results = {}
    for ds_name in datasets_to_eval:
        ds_cfg = DATASET_CONFIGS[ds_name]
        hf_code = ds_cfg["get_hf_code"](args.language_code)
        result = evaluate_dataset(
            model, processor, ds_name, iso_code, args.language,
            ds_cfg["hf_id"], hf_code, device, args.max_samples,
        )
        if result:
            all_results[ds_name] = result

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK EVALUATION RESULTS")
    print("="*60)
    for ds_name, res in all_results.items():
        print(f"\n{ds_name.upper()} ({res['num_samples']} samples)")
        print(f"  WER: {res['wer']:.4f}  |  CER: {res['cer']:.4f}")
        if res.get("wer_by_duration"):
            print("  WER by duration:")
            for bucket, stats in res["wer_by_duration"].items():
                print(f"    {bucket}: {stats['wer']:.4f}  (n={stats['n']})")
    print("="*60 + "\n")

    # Save results
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds_name, res in all_results.items():
        out_path = out_dir / f"eval_{ds_name}.json"
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
