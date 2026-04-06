"""
train.py
────────
Main training entry point.

Run:
    python train.py --config configs/config.yaml

For quick smoke test (CPU, tiny data):
    python train.py --config configs/config.yaml --smoke_test

For a different language (e.g. Telugu):
    python train.py --config configs/config.yaml --language telugu --language_code te
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from src.model import WhisperLoRAModel, LoRAConfig
from src.data_loader import IndicDataLoader, DataConfig, DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import ASRMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
#  Config loading
# ──────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Allow CLI flags to override YAML config values."""
    if args.language:
        cfg["model"]["language"] = args.language
    if args.language_code:
        cfg["model"]["language_code"] = args.language_code
        cfg["data"]["language_code"] = args.language_code
    if args.smoke_test:
        cfg["data"]["max_train_samples"] = 64
        cfg["data"]["max_eval_samples"] = 32
        cfg["training"]["num_train_epochs"] = 1
        cfg["training"]["eval_steps"] = 20
        cfg["training"]["save_steps"] = 20
        cfg["training"]["logging_steps"] = 5
        cfg["training"]["output_dir"] = "./models/smoke_test"
        logger.info("Smoke test mode: small data, 1 epoch")
    return cfg


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Whisper LoRA fine-tuning for Indian languages")
    parser.add_argument("--config",        type=str, default="configs/config.yaml")
    parser.add_argument("--language",      type=str, default=None, help="Override language name")
    parser.add_argument("--language_code", type=str, default=None, help="Override ISO language code")
    parser.add_argument("--smoke_test",    action="store_true",    help="Quick test run with minimal data")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    logger.info(f"Language: {cfg['model']['language']} ({cfg['model']['language_code']})")
    logger.info(f"Base model: {cfg['model']['base']}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # ── 1. Build model + processor with LoRA ──────────────
    lora_cfg = LoRAConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
    )
    builder = WhisperLoRAModel(
        base_model_id=cfg["model"]["base"],
        language=cfg["model"]["language"],
        language_code=cfg["model"]["language_code"],
        task=cfg["model"]["task"],
        lora_config=lora_cfg,
    )
    model, processor = builder.build()

    # ── 2. Load + preprocess dataset ──────────────────────
    data_cfg = DataConfig(
        dataset_name=cfg["data"]["dataset_name"],
        language_code=cfg["data"]["language_code"],
        max_audio_duration_sec=cfg["data"]["max_audio_duration_sec"],
        min_audio_duration_sec=cfg["data"]["min_audio_duration_sec"],
        train_split=cfg["data"]["train_split"],
        eval_split=cfg["data"]["eval_split"],
        test_split=cfg["data"]["test_split"],
        max_train_samples=cfg["data"]["max_train_samples"],
        max_eval_samples=cfg["data"]["max_eval_samples"],
    )
    loader = IndicDataLoader(data_cfg, processor.feature_extractor, processor.tokenizer)
    dataset = loader.get_dataset()

    logger.info(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} samples")

    # ── 3. Data collator ──────────────────────────────────
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── 4. Metrics ────────────────────────────────────────
    compute_metrics = ASRMetrics(
        tokenizer=processor.tokenizer,
        language_code=cfg["model"]["language_code"],
    )

    # ── 5. Training arguments ─────────────────────────────
    t = cfg["training"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=t["output_dir"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_steps=t["warmup_steps"],
        num_train_epochs=t["num_train_epochs"],
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        logging_steps=t["logging_steps"],
        fp16=t["fp16"] and torch.cuda.is_available(),
        bf16=t["bf16"] and torch.cuda.is_available(),
        dataloader_num_workers=t["dataloader_num_workers"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        gradient_checkpointing=t["gradient_checkpointing"],
        optim=t["optim"],
        weight_decay=t["weight_decay"],
        report_to=t["report_to"],
        push_to_hub=t["push_to_hub"],
        predict_with_generate=True,   # required for Seq2Seq eval
        generation_max_length=225,
        remove_unused_columns=False,
    )

    # ── 6. Resume from checkpoint if available ────────────
    output_dir = Path(t["output_dir"])
    last_checkpoint = None
    if output_dir.exists():
        last_checkpoint = get_last_checkpoint(str(output_dir))
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # ── 7. Trainer ────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # ── 8. Train ──────────────────────────────────────────
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ── 9. Save final model ───────────────────────────────
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    logger.info(f"Model saved to: {final_path}")

    # ── 10. Evaluate on test set ──────────────────────────
    if "test" in dataset:
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(
            eval_dataset=dataset["test"],
            metric_key_prefix="test",
        )
        logger.info(f"Test WER: {test_results.get('test_wer', 'N/A'):.4f}")
        logger.info(f"Test CER: {test_results.get('test_cer', 'N/A'):.4f}")

        # Save results
        import json
        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Test results saved to: {results_path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
