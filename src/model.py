import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


class WhisperLoRAPeftWrapper(nn.Module):
    """
    Thin wrapper around a PEFT-applied Whisper model.

    PEFT's PeftModelForSeq2SeqLM.forward() passes `input_ids=None` to the base model,
    but Whisper expects `input_features`, not `input_ids`. This wrapper exposes Whisper's
    native forward signature so Seq2SeqTrainer works correctly while LoRA adapters (which
    are injected in-place on the attention layers) remain active.
    """

    def __init__(self, peft_model):
        super().__init__()
        self.peft_model = peft_model
        # NOTE: do NOT store peft_model.base_model.model as a separate nn.Module attribute —
        # that would double-register all parameters in state_dict.
        self.config = peft_model.config
        self.generation_config = peft_model.generation_config

    def forward(self, input_features, labels=None, **kwargs):
        # LoRA adapters are injected in-place on peft_model.base_model.model so they stay active.
        return self.peft_model.base_model.model(input_features=input_features, labels=labels, **kwargs)

    def generate(self, *args, **kwargs):
        return self.peft_model.generate(*args, **kwargs)

    def save_pretrained(self, path, **kwargs):
        self.peft_model.save_pretrained(path, **kwargs)

    def print_trainable_parameters(self):
        self.peft_model.print_trainable_parameters()


class WhisperLoRAModel:
    def __init__(
        self,
        base_model_id: str,
        language: str,
        language_code: str,
        task: str = "transcribe",
        lora_config: Optional[LoRAConfig] = None,
        use_lora: bool = True,
    ):
        self.base_model_id = base_model_id
        self.language = language
        self.language_code = language_code
        self.task = task
        self.lora_config = lora_config or LoRAConfig()
        self.use_lora = use_lora

    def build(self):
        logger.info(f"Loading processor: {self.base_model_id}")
        processor = WhisperProcessor.from_pretrained(
            self.base_model_id, language=self.language, task=self.task
        )
        logger.info(f"Loading base model: {self.base_model_id}")
        model = WhisperForConditionalGeneration.from_pretrained(self.base_model_id)

        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.generation_config.language = self.language_code
        model.generation_config.task = self.task
        model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=self.language, task=self.task
        )

        if self.use_lora:
            peft_config = LoraConfig(
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                bias=self.lora_config.bias,
                target_modules=self.lora_config.target_modules,
                task_type=TaskType.SEQ_2_SEQ_LM,
            )
            peft_model = get_peft_model(model, peft_config)
            # Wrap to fix Whisper's input_features vs. PEFT's input_ids mismatch
            model = WhisperLoRAPeftWrapper(peft_model)
            model.print_trainable_parameters()
            logger.info(f"LoRA applied — target modules: {self.lora_config.target_modules}")
        else:
            for param in model.model.encoder.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Encoder-frozen mode — {trainable:,} / {total:,} trainable "
                f"({100 * trainable / total:.2f}%)"
            )

        _print_param_summary(model)
        return model, processor


def _print_param_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"\n{'─'*50}")
    print(f"  Trainable:  {trainable:,} params ({pct:.3f}%)")
    print(f"  Frozen:     {total - trainable:,} params")
    print(f"  Total:      {total:,} params")
    print(f"{'─'*50}\n")


def load_finetuned_model(checkpoint_path: str, base_model_id: str = "openai/whisper-small"):
    """
    Load a fine-tuned checkpoint, auto-detecting PEFT adapter vs. full model.
    LoRA adapters are merged into the base model so inference has no overhead.
    """
    adapter_cfg = Path(checkpoint_path) / "adapter_config.json"
    if adapter_cfg.exists():
        logger.info(f"Loading PEFT adapter from {checkpoint_path}")
        base = WhisperForConditionalGeneration.from_pretrained(base_model_id)
        model = PeftModel.from_pretrained(base, checkpoint_path)
        model = model.merge_and_unload()
        logger.info("LoRA adapters merged — zero inference overhead")
    else:
        logger.info(f"Loading full model from {checkpoint_path}")
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

    model.eval()
    return model
