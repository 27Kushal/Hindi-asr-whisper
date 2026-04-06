"""
data_loader.py
──────────────
Loads the locally generated Hindi TTS dataset.
Audio files are in data/hindi_tts/{split}/
Transcriptions are in data/hindi_tts/{split}/{split}.csv
"""

import logging
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
from datasets import Dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000


@dataclass
class DataConfig:
    dataset_name: str = "local"
    language_code: str = "hi"
    max_audio_duration_sec: float = 30.0
    min_audio_duration_sec: float = 1.0
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 500
    sampling_rate: int = SAMPLING_RATE
    local_data_dir: str = "data/hindi_tts"


class IndicDataLoader:
    def __init__(
        self,
        config: DataConfig,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
    ):
        self.config = config
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def get_dataset(self) -> DatasetDict:
        logger.info("Loading local Hindi TTS dataset...")
        splits = {}

        for split_key, split_name in [
            ("train",      self.config.train_split),
            ("validation", self.config.eval_split),
            ("test",       self.config.test_split),
        ]:
            csv_path = Path(self.config.local_data_dir) / split_name / f"{split_name}.csv"
            if not csv_path.exists():
                logger.warning(f"  CSV not found: {csv_path} — skipping")
                continue

            rows = []
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)

            if self.config.max_train_samples and split_key == "train":
                rows = rows[:self.config.max_train_samples]
            if self.config.max_eval_samples and split_key == "validation":
                rows = rows[:self.config.max_eval_samples]

            logger.info(f"  {split_key}: {len(rows)} samples")

            features = []
            labels = []
            skipped = 0

            for row in rows:
                audio_path = row["path"]
                sentence = row["sentence"].strip()

                try:
                    waveform, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
                    duration = len(waveform) / SAMPLING_RATE

                    if duration < self.config.min_audio_duration_sec:
                        skipped += 1
                        continue
                    if duration > self.config.max_audio_duration_sec:
                        skipped += 1
                        continue

                    # Normalise
                    max_val = np.abs(waveform).max()
                    if max_val > 0:
                        waveform = waveform / max_val

                    input_feat = self.feature_extractor(
                        waveform,
                        sampling_rate=SAMPLING_RATE,
                        return_tensors="np",
                    ).input_features[0]

                    label_ids = self.tokenizer(sentence).input_ids

                    features.append(input_feat)
                    labels.append(label_ids)

                except Exception as e:
                    logger.warning(f"  Skipping {audio_path}: {e}")
                    skipped += 1

            if skipped:
                logger.info(f"  Skipped {skipped} samples")

            ds = Dataset.from_dict({
                "input_features": features,
                "labels": labels,
            })
            splits[split_key] = ds

        if not splits:
            raise RuntimeError(
                "No splits loaded. Run: python scripts/generate_hindi_dataset.py"
            )

        return DatasetDict(splits)


import torch
from dataclasses import field as dc_field

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features: list) -> dict:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch