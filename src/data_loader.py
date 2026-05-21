"""
data_loader.py
──────────────
Supports two dataset sources:
  - "google/fleurs"  : real multi-speaker speech via HuggingFace datasets (default)
  - "local"          : locally generated CSV-based TTS dataset (data/hindi_tts/)
"""

import logging
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import scipy.signal
import torch
from dataclasses import field as dc_field
from datasets import Dataset, DatasetDict, load_dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000


@dataclass
class DataConfig:
    dataset_name: str = "google/fleurs"
    language_code: str = "hi_in"          # FLEURS code, e.g. hi_in, ta_in, te_in
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
        if self.config.dataset_name == "google/fleurs":
            return self._load_fleurs()
        else:
            return self._load_local()

    # ── FLEURS (real multi-speaker speech) ────────────────────────────────
    def _load_fleurs(self) -> DatasetDict:
        logger.info(f"Loading google/fleurs ({self.config.language_code})…")
        splits = {}
        split_map = {
            "train":      (self.config.train_split,  self.config.max_train_samples),
            "validation": (self.config.eval_split,   self.config.max_eval_samples),
            "test":       (self.config.test_split,   None),
        }
        for key, (hf_split, max_samples) in split_map.items():
            try:
                raw = load_dataset(
                    "google/fleurs",
                    self.config.language_code,
                    split=hf_split,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"Could not load FLEURS split '{hf_split}': {e}")
                continue

            if max_samples:
                raw = raw.select(range(min(max_samples, len(raw))))

            logger.info(f"  {key}: {len(raw)} samples")
            features, labels = [], []
            skipped = 0

            for item in raw:
                audio = item["audio"]
                transcript = item.get("transcription") or item.get("raw_transcription", "")
                transcript = transcript.strip()
                if not transcript:
                    skipped += 1
                    continue

                waveform = np.array(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
                if sr != SAMPLING_RATE:
                    waveform = scipy.signal.resample_poly(
                        waveform, SAMPLING_RATE, sr
                    ).astype(np.float32)

                duration = len(waveform) / SAMPLING_RATE
                if not (self.config.min_audio_duration_sec <= duration <= self.config.max_audio_duration_sec):
                    skipped += 1
                    continue

                waveform = _normalise_waveform(waveform)
                feat = self.feature_extractor(
                    waveform, sampling_rate=SAMPLING_RATE, return_tensors="np"
                ).input_features[0]
                label_ids = self.tokenizer(transcript).input_ids
                features.append(feat)
                labels.append(label_ids)

            if skipped:
                logger.info(f"  Skipped {skipped} samples in {key}")

            if features:
                splits[key] = Dataset.from_dict({"input_features": features, "labels": labels})

        if not splits:
            raise RuntimeError(
                "No FLEURS splits loaded. Check your internet connection and language code."
            )
        return DatasetDict(splits)

    # ── Local CSV (macOS TTS dataset) ──────────────────────────────────────
    def _load_local(self) -> DatasetDict:
        logger.info("Loading local Hindi TTS dataset…")
        splits = {}
        split_map = {
            "train":      self.config.train_split,
            "validation": self.config.eval_split,
            "test":       self.config.test_split,
        }
        for key, split_name in split_map.items():
            csv_path = Path(self.config.local_data_dir) / split_name / f"{split_name}.csv"
            if not csv_path.exists():
                logger.warning(f"  CSV not found: {csv_path} — skipping")
                continue

            rows = []
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            if self.config.max_train_samples and key == "train":
                rows = rows[:self.config.max_train_samples]
            if self.config.max_eval_samples and key == "validation":
                rows = rows[:self.config.max_eval_samples]

            logger.info(f"  {key}: {len(rows)} samples")
            features, labels = [], []
            skipped = 0

            for row in rows:
                audio_path = row["path"]
                sentence = row["sentence"].strip()
                try:
                    waveform, sr = sf.read(audio_path, always_2d=False)
                    if waveform.ndim > 1:
                        waveform = waveform.mean(axis=1)
                    waveform = waveform.astype(np.float32)
                    if sr != SAMPLING_RATE:
                        waveform = scipy.signal.resample_poly(
                            waveform, SAMPLING_RATE, sr
                        ).astype(np.float32)
                    duration = len(waveform) / SAMPLING_RATE
                    if not (self.config.min_audio_duration_sec <= duration <= self.config.max_audio_duration_sec):
                        skipped += 1
                        continue
                    waveform = _normalise_waveform(waveform)
                    feat = self.feature_extractor(
                        waveform, sampling_rate=SAMPLING_RATE, return_tensors="np"
                    ).input_features[0]
                    label_ids = self.tokenizer(sentence).input_ids
                    features.append(feat)
                    labels.append(label_ids)
                except Exception as e:
                    logger.warning(f"  Skipping {audio_path}: {e}")
                    skipped += 1

            if skipped:
                logger.info(f"  Skipped {skipped} samples in {key}")

            if features:
                splits[key] = Dataset.from_dict({"input_features": features, "labels": labels})

        if not splits:
            raise RuntimeError(
                "No local splits loaded. Run: python scripts/generate_hindi_dataset.py"
            )
        return DatasetDict(splits)


def _normalise_waveform(waveform: np.ndarray) -> np.ndarray:
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features: list) -> dict:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch
