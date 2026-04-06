"""
curriculum.py
─────────────
Curriculum Learning for Low-Resource ASR.

Core idea:
  Instead of training on all data randomly, we order samples from
  EASY → HARD. This mimics how humans learn and has been shown to
  improve convergence speed and final WER in low-resource settings.

Difficulty metrics we use:
  1. Audio duration       — shorter clips are easier
  2. Transcript length    — fewer words = easier
  3. Signal-to-noise ratio (SNR) — cleaner audio = easier
  4. Character n-gram perplexity  — common words = easier

Usage:
    from scripts.curriculum import CurriculumSampler
    sampler = CurriculumSampler(dataset["train"])
    ordered_dataset = sampler.get_ordered_dataset()
"""

import logging
from typing import List

import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000


# ──────────────────────────────────────────────────────────
#  Difficulty scoring
# ──────────────────────────────────────────────────────────
def compute_snr(waveform: np.ndarray) -> float:
    """
    Estimate SNR using a simple energy ratio heuristic.
    Assumes the quietest 10% of frames is noise floor.
    """
    frame_energies = np.array([
        np.sum(waveform[i:i+160]**2)
        for i in range(0, len(waveform) - 160, 160)
    ])
    if len(frame_energies) == 0:
        return 0.0
    noise_floor = np.percentile(frame_energies, 10) + 1e-8
    signal_power = np.mean(frame_energies) + 1e-8
    snr_db = 10 * np.log10(signal_power / noise_floor)
    return float(snr_db)


def score_difficulty(sample: dict) -> float:
    """
    Lower score = easier.

    We normalise and combine three signals:
      - Duration (shorter = easier):  weight 0.3
      - Transcript length (fewer chars = easier): weight 0.3
      - SNR (higher = easier, so we invert): weight 0.4
    """
    audio = sample["audio"]
    waveform = np.array(audio["array"], dtype=np.float32)
    duration = len(waveform) / audio["sampling_rate"]

    transcript = sample.get("sentence", sample.get("text", ""))
    char_count = len(transcript.strip())

    snr = compute_snr(waveform)
    # Invert SNR: low SNR (noisy) → high difficulty score
    snr_difficulty = max(0, 40 - snr) / 40  # normalise to [0,1]

    # Normalise duration (0–30s → 0–1)
    dur_score = min(duration / 30.0, 1.0)

    # Normalise transcript length (0–200 chars → 0–1)
    len_score = min(char_count / 200.0, 1.0)

    difficulty = 0.3 * dur_score + 0.3 * len_score + 0.4 * snr_difficulty
    return difficulty


# ──────────────────────────────────────────────────────────
#  Curriculum sampler
# ──────────────────────────────────────────────────────────
class CurriculumSampler:
    """
    Reorders a dataset from easy → hard based on difficulty scores.

    Two modes:
      "full":  Score all samples, sort once, train in order.
      "paced": Split into N buckets. Gradually unlock harder buckets
               as training progresses (requires custom training loop).
    """

    def __init__(self, dataset: Dataset, mode: str = "full", n_buckets: int = 5):
        self.dataset = dataset
        self.mode = mode
        self.n_buckets = n_buckets

    def get_ordered_dataset(self) -> Dataset:
        logger.info("Computing difficulty scores for curriculum learning...")
        scores = []
        for i, sample in enumerate(self.dataset):
            try:
                score = score_difficulty(sample)
            except Exception:
                score = 0.5  # fallback: medium difficulty
            scores.append(score)
            if (i + 1) % 500 == 0:
                logger.info(f"  Scored {i+1}/{len(self.dataset)} samples")

        sorted_indices = np.argsort(scores)  # ascending: easy → hard
        ordered = self.dataset.select(sorted_indices.tolist())

        easy_threshold = np.percentile(scores, 33)
        hard_threshold = np.percentile(scores, 66)
        logger.info(
            f"Curriculum: easy < {easy_threshold:.2f} | "
            f"medium < {hard_threshold:.2f} | hard >= {hard_threshold:.2f}"
        )
        return ordered

    def get_buckets(self) -> List[Dataset]:
        """
        Returns N dataset buckets ordered easy → hard.
        Use with a custom training loop that progressively adds buckets.
        """
        ordered = self.get_ordered_dataset()
        n = len(ordered)
        bucket_size = n // self.n_buckets
        buckets = []
        for i in range(self.n_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < self.n_buckets - 1 else n
            buckets.append(ordered.select(range(start, end)))
        return buckets
