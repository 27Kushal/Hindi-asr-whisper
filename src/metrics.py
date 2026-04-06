"""
metrics.py
──────────
Evaluation metrics for ASR:

  WER  — Word Error Rate   (primary metric, lower is better)
  CER  — Character Error Rate (better for morphologically rich Indian languages)

  WER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions, N = total reference words

For agglutinative languages like Tamil or Telugu, CER is often more
informative than WER because word boundaries are less consistent.
"""

import re
import unicodedata
from typing import List

import numpy as np
import evaluate
from transformers import WhisperTokenizer

# Load HuggingFace evaluate metrics
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


# ──────────────────────────────────────────────────────────
#  Text normalisation
# ──────────────────────────────────────────────────────────
def normalise_text(text: str, language_code: str = "ta") -> str:
    """
    Light normalisation before computing WER/CER.
    Preserves Unicode characters (Indic scripts).
    """
    # Unicode NFC normalisation (handles composed vs decomposed forms)
    text = unicodedata.normalize("NFC", text)

    # Lowercase (relevant for romanised text, no-op for scripts)
    text = text.lower()

    # Remove punctuation but keep Indic characters and spaces
    # \w matches Unicode word chars including Indic scripts
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────────────────────────────────────────
#  Metric computation callback for HuggingFace Trainer
# ──────────────────────────────────────────────────────────
class ASRMetrics:
    """
    Passed to Seq2SeqTrainer as compute_metrics.

    Usage:
        metrics_fn = ASRMetrics(tokenizer)
        trainer = Seq2SeqTrainer(..., compute_metrics=metrics_fn)
    """

    def __init__(self, tokenizer: WhisperTokenizer, language_code: str = "ta"):
        self.tokenizer = tokenizer
        self.language_code = language_code

    def __call__(self, eval_preds) -> dict:
        pred_ids, label_ids = eval_preds

        # Replace -100 (padding) with the pad token id before decoding
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # Decode predictions and references
        predictions = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Normalise
        predictions = [normalise_text(p, self.language_code) for p in predictions]
        references = [normalise_text(r, self.language_code) for r in references]

        # Filter empty references (they break metrics)
        pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        predictions, references = zip(*pairs)

        wer = wer_metric.compute(predictions=list(predictions), references=list(references))
        cer = cer_metric.compute(predictions=list(predictions), references=list(references))

        return {
            "wer": round(wer, 4),
            "cer": round(cer, 4),
        }


# ──────────────────────────────────────────────────────────
#  Standalone evaluation on a list of predictions
# ──────────────────────────────────────────────────────────
def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    language_code: str = "ta",
) -> dict:
    """
    Compute WER and CER on a list of (prediction, reference) pairs.
    Useful for post-hoc evaluation on a test set.
    """
    assert len(predictions) == len(references), "Length mismatch"

    preds_norm = [normalise_text(p, language_code) for p in predictions]
    refs_norm  = [normalise_text(r, language_code) for r in references]

    pairs = [(p, r) for p, r in zip(preds_norm, refs_norm) if r.strip()]
    preds_norm, refs_norm = zip(*pairs)

    wer = wer_metric.compute(predictions=list(preds_norm), references=list(refs_norm))
    cer = cer_metric.compute(predictions=list(preds_norm), references=list(refs_norm))

    # Per-sample errors for analysis
    sample_errors = []
    for p, r in zip(preds_norm, refs_norm):
        sample_wer = wer_metric.compute(predictions=[p], references=[r])
        sample_errors.append({
            "prediction": p,
            "reference":  r,
            "wer": round(sample_wer, 4),
        })

    # Sort by worst WER for error analysis
    sample_errors.sort(key=lambda x: x["wer"], reverse=True)

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "worst_samples": sample_errors[:10],  # top 10 worst for analysis
    }
