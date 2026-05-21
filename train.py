"""
train.py
────────
Main training entry point.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml

Smoke test (1 epoch, 64 samples):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml --smoke_test

Encoder-frozen ablation mode (no LoRA):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml --no_lora

Different language:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py --config configs/config.yaml --language telugu --language_code te
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
    set_seed,
)


class PeftSeq2SeqTrainer(Seq2SeqTrainer):
    """Overrides _save to use model.save_pretrained(), bypassing safetensors weight-tie check."""

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
from transformers.trainer_utils import get_last_checkpoint

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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.language:
        cfg["model"]["language"] = args.language
    if args.language_code:
        cfg["model"]["language_code"] = args.language_code
        cfg["data"]["language_code"] = args.language_code
    if args.dataset:
        cfg["data"]["dataset_name"] = args.dataset
    if args.output_dir_override:
        cfg["training"]["output_dir"] = args.output_dir_override
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


def main():
    parser = argparse.ArgumentParser(description="Whisper LoRA fine-tuning for Indian languages")
    parser.add_argument("--config",        type=str, default="configs/config.yaml")
    parser.add_argument("--language",      type=str, default=None)
    parser.add_argument("--language_code", type=str, default=None)
    parser.add_argument("--dataset",       type=str, default=None,
                        help="Override dataset (google/fleurs or local)")
    parser.add_argument("--no_lora",           action="store_true",
                        help="Ablation: encoder-frozen mode, no LoRA adapters")
    parser.add_argument("--output_dir_override", type=str, default=None,
                        help="Override output_dir from config")
    parser.add_argument("--smoke_test",    action="store_true")
    parser.add_argument("--cpu",           action="store_true",
                        help="Force CPU training (disable MPS/CUDA)")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    use_lora = not args.no_lora
    logger.info(f"Mode: {'LoRA PEFT' if use_lora else 'encoder-frozen (ablation)'}")
    logger.info(f"Language: {cfg['model']['language']} ({cfg['model']['language_code']})")
    logger.info(f"Dataset: {cfg['data']['dataset_name']}")
    logger.info(f"Base model: {cfg['model']['base']}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'MPS/CPU'}")

    # ── 1. Build model + processor ────────────────────────
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
        use_lora=use_lora,
    )
    model, processor = builder.build()

    # ── 2. Dataset ────────────────────────────────────────
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
        local_data_dir=cfg["data"].get("local_data_dir", "data/hindi_tts"),
    )
    loader = IndicDataLoader(data_cfg, processor.feature_extractor, processor.tokenizer)
    dataset = loader.get_dataset()
    logger.info(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])} samples")

    # ── 3. Collator + metrics ─────────────────────────────
    # Unwrap PEFT model to get base config for decoder_start_token_id
    base_model = model.base_model if hasattr(model, "base_model") else model
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=base_model.config.decoder_start_token_id,
    )
    compute_metrics = ASRMetrics(
        tokenizer=processor.tokenizer,
        language_code=cfg["model"]["language_code"],
    )

    # ── 4. Training arguments ─────────────────────────────
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
        fp16=t["fp16"] and torch.cuda.is_available() and not args.cpu,
        bf16=t["bf16"] and torch.cuda.is_available() and not args.cpu,
        no_cuda=args.cpu,
        dataloader_num_workers=t["dataloader_num_workers"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        gradient_checkpointing=t["gradient_checkpointing"],
        optim=t["optim"],
        weight_decay=t["weight_decay"],
        report_to=t["report_to"],
        push_to_hub=t["push_to_hub"],
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
    )

    # ── 5. Resume from checkpoint ─────────────────────────
    output_dir = Path(t["output_dir"])
    last_checkpoint = None
    if output_dir.exists():
        last_checkpoint = get_last_checkpoint(str(output_dir))
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # ── 6. Train ──────────────────────────────────────────
    trainer = PeftSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training…")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    training_time = time.time() - t0
    logger.info(f"Training completed in {training_time/3600:.2f} hours")

    # ── 7. Save ───────────────────────────────────────────
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    logger.info(f"Model saved to: {final_path}")

    # ── 8. Test evaluation ────────────────────────────────
    if "test" in dataset:
        logger.info("Evaluating on test set…")
        test_results = trainer.evaluate(
            eval_dataset=dataset["test"],
            metric_key_prefix="test",
        )
        logger.info(f"Test WER: {test_results.get('test_wer', 'N/A'):.4f}")
        logger.info(f"Test CER: {test_results.get('test_cer', 'N/A'):.4f}")

        # Compute trainable/total params for ablation records
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        checkpoint_size = sum(
            f.stat().st_size for f in final_path.rglob("*.bin")
        ) + sum(f.stat().st_size for f in final_path.rglob("*.safetensors"))

        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {
                    **test_results,
                    "trainable_params": trainable,
                    "total_params": total,
                    "checkpoint_size_bytes": checkpoint_size,
                    "training_time_sec": round(training_time),
                    "mode": "lora_peft" if use_lora else "encoder_frozen",
                    "language": cfg["model"]["language"],
                    "dataset": cfg["data"]["dataset_name"],
                },
                f, indent=2,
            )
        logger.info(f"Results saved to: {results_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
