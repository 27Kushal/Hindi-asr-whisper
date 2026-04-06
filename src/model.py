import logging
from dataclasses import dataclass
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: list = None

class WhisperLoRAModel:
    def __init__(self, base_model_id, language, language_code, task="transcribe", lora_config=None):
        self.base_model_id = base_model_id
        self.language = language
        self.language_code = language_code
        self.task = task

    def build(self):
        logger.info(f"Loading processor: {self.base_model_id}")
        processor = WhisperProcessor.from_pretrained(
            self.base_model_id, language=self.language, task=self.task
        )
        logger.info(f"Loading base model: {self.base_model_id}")
        model = WhisperForConditionalGeneration.from_pretrained(self.base_model_id)

        # Freeze encoder — only train decoder
        for param in model.model.encoder.parameters():
            param.requires_grad = False

        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.generation_config.language = self.language_code
        model.generation_config.task = self.task
        model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=self.language, task=self.task
        )
        logger.info(f"  Forced decoder language: {self.language} ({self.language_code})")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = 100 * trainable / total
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%) — encoder frozen")
        print(f"\n{'─'*50}")
        print(f"  Trainable:  {trainable:,} params ({pct:.1f}%)")
        print(f"  Frozen:     {total - trainable:,} params")
        print(f"  Total:      {total:,} params")
        print(f"{'─'*50}\n")

        return model, processor

def load_finetuned_model(checkpoint_path, base_model_id):
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    model.eval()
    return model
