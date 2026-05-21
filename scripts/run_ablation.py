"""
run_ablation.py
───────────────
Runs a systematic 3-mode ablation study and saves a comparison table.

Modes:
  A — base_whisper    : zero-shot evaluation, no training at all
  B — encoder_frozen  : freeze encoder, train full decoder (legacy approach)
  C — lora_peft       : PEFT LoRA adapters on decoder attention (new approach)

Usage:
    python scripts/run_ablation.py --config configs/config.yaml
    python scripts/run_ablation.py --config configs/config.yaml --modes A C   # skip B
    python scripts/run_ablation.py --config configs/config.yaml --language tamil --language_code ta
"""

import argparse
import json
import logging
import os
import sys
import subprocess
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODE_CONFIGS = {
    "A": {
        "name": "base_whisper",
        "label": "Base Whisper (zero-shot)",
        "description": "Unmodified openai/whisper-small, no fine-tuning",
        "train": False,
        "output_dir": "models/ablation/base_whisper",
    },
    "B": {
        "name": "encoder_frozen",
        "label": "Encoder-Frozen Fine-tuning",
        "description": "Encoder frozen, full decoder trained (~154M trainable params)",
        "train": True,
        "extra_flags": ["--no_lora"],
        "output_dir": "models/ablation/encoder_frozen",
    },
    "C": {
        "name": "lora_peft",
        "label": "LoRA PEFT Fine-tuning",
        "description": "PEFT LoRA adapters on decoder attention (r=16, ~1.77M trainable params)",
        "train": True,
        "extra_flags": [],
        "output_dir": "models/ablation/lora_peft",
    },
}


def run_training(cfg_path: str, output_dir: str, extra_flags: list, language: str = None,
                 language_code: str = None, dataset: str = None, cpu: bool = False) -> dict:
    """Run a training mode and return the test results."""
    cmd = [
        sys.executable, "train.py",
        "--config", cfg_path,
        "--output_dir_override", output_dir,
    ] + extra_flags

    if language:
        cmd += ["--language", language, "--language_code", language_code]
    if dataset:
        cmd += ["--dataset", dataset]
    if cpu:
        cmd += ["--cpu"]

    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "TRANSFORMERS_OFFLINE": "1"}
    logger.info(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Training failed (exit code {result.returncode})")
        return {}

    results_path = Path(output_dir) / "test_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        data["training_time_sec"] = round(elapsed)
        return data
    return {"training_time_sec": round(elapsed)}


def evaluate_base(cfg_path: str, output_dir: str, language: str, language_code: str) -> dict:
    """Evaluate the unmodified base Whisper model (Mode A)."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch
    from src.data_loader import DataConfig, IndicDataLoader
    from src.metrics import ASRMetrics

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    lang = language or cfg["model"]["language"]
    lang_code = language_code or cfg["model"]["language_code"]

    logger.info("Evaluating base Whisper (zero-shot)…")
    processor = WhisperProcessor.from_pretrained(
        cfg["model"]["base"], language=lang, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model"]["base"])
    model.eval()

    data_cfg = DataConfig(
        dataset_name=cfg["data"]["dataset_name"],
        language_code=cfg["data"]["language_code"],
        local_data_dir=cfg["data"].get("local_data_dir", "data/hindi_tts"),
        max_eval_samples=cfg["data"]["max_eval_samples"],
        train_split=cfg["data"]["train_split"],
        eval_split=cfg["data"]["eval_split"],
        test_split=cfg["data"]["test_split"],
    )
    loader = IndicDataLoader(data_cfg, processor.feature_extractor, processor.tokenizer)
    dataset = loader.get_dataset()

    compute_metrics = ASRMetrics(tokenizer=processor.tokenizer, language_code=lang_code)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Run inference on test split
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
    all_pred_ids, all_label_ids = [], []

    from torch.utils.data import DataLoader
    import numpy as np
    from src.data_loader import DataCollatorSpeechSeq2SeqWithPadding

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
    )
    loader_obj = DataLoader(dataset["test"], batch_size=4, collate_fn=collator)

    with torch.no_grad():
        for batch in loader_obj:
            input_features = batch["input_features"].to(device)
            ids = model.generate(
                input_features, forced_decoder_ids=forced_decoder_ids, max_length=225
            )
            all_pred_ids.extend(ids.cpu().numpy().tolist())
            all_label_ids.extend(batch["labels"].numpy().tolist())

    metrics = compute_metrics((np.array(all_pred_ids), np.array(all_label_ids)))
    total = sum(p.numel() for p in model.parameters())

    result = {
        "test_wer": metrics["wer"],
        "test_cer": metrics["cer"],
        "trainable_params": 0,
        "total_params": total,
        "checkpoint_size_bytes": 0,
        "training_time_sec": 0,
        "mode": "base_whisper",
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "test_results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def build_comparison_table(results: dict) -> dict:
    table = {}
    for mode_key, data in results.items():
        cfg = MODE_CONFIGS[mode_key]
        checkpoint_mb = data.get("checkpoint_size_bytes", 0) / (1024 ** 2)
        train_hours = data.get("training_time_sec", 0) / 3600
        table[mode_key] = {
            "label": cfg["label"],
            "description": cfg["description"],
            "wer": data.get("test_wer", "—"),
            "cer": data.get("test_cer", "—"),
            "trainable_params": data.get("trainable_params", 0),
            "total_params": data.get("total_params", 0),
            "checkpoint_size_mb": round(checkpoint_mb, 1),
            "training_time_hours": round(train_hours, 2),
        }
    return table


def print_markdown_table(table: dict):
    print("\n" + "="*90)
    print("ABLATION STUDY RESULTS")
    print("="*90)
    header = f"{'Mode':<30} {'WER':>7} {'CER':>7} {'Trainable Params':>18} {'Ckpt (MB)':>10} {'Train (h)':>10}"
    print(header)
    print("─" * 90)
    for mode_key, row in table.items():
        wer = f"{row['wer']:.4f}" if isinstance(row['wer'], float) else row['wer']
        cer = f"{row['cer']:.4f}" if isinstance(row['cer'], float) else row['cer']
        params = f"{row['trainable_params']:,}" if row['trainable_params'] else "0"
        ckpt = f"{row['checkpoint_size_mb']:.1f}" if row['checkpoint_size_mb'] else "—"
        hours = f"{row['training_time_hours']:.2f}" if row['training_time_hours'] else "—"
        print(f"{row['label']:<30} {wer:>7} {cer:>7} {params:>18} {ckpt:>10} {hours:>10}")
    print("="*90 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--modes", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"],
                        help="Which ablation modes to run (default: all three)")
    parser.add_argument("--language", default=None)
    parser.add_argument("--language_code", default=None)
    parser.add_argument("--dataset", default=None, help="Override dataset (google/fleurs or local)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--output_dir", default="models/ablation")
    args = parser.parse_args()

    results = {}
    existing_results = Path(args.output_dir) / "comparison_table.json"
    if existing_results.exists():
        with open(existing_results) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing mode results")

    for mode_key in args.modes:
        cfg = MODE_CONFIGS[mode_key]
        output_dir = str(Path(args.output_dir) / cfg["name"])

        if mode_key in results:
            logger.info(f"Mode {mode_key} ({cfg['name']}) already done — skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Mode {mode_key}: {cfg['label']}")
        logger.info(f"{'='*60}")

        if not cfg["train"]:
            # Mode A: base model zero-shot
            data = evaluate_base(args.config, output_dir, args.language, args.language_code)
        else:
            data = run_training(
                args.config, output_dir,
                cfg.get("extra_flags", []),
                args.language, args.language_code,
                dataset=args.dataset,
                cpu=args.cpu,
            )

        if data:
            results[mode_key] = data
            # Save incrementally so a crash doesn't lose completed modes
            table = build_comparison_table(results)
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(existing_results, "w") as f:
                json.dump(table, f, indent=2)
            logger.info(f"Mode {mode_key} complete — results saved")

    table = build_comparison_table(results)
    with open(existing_results, "w") as f:
        json.dump(table, f, indent=2)

    print_markdown_table(table)
    logger.info(f"\nComparison table saved to: {existing_results}")


if __name__ == "__main__":
    main()
