"""
metrics.py
──────────
WER and CER evaluation for Hindi ASR.

  WER = (Substitutions + Deletions + Insertions) / N reference words
  CER = edit distance at character level / N reference chars

For Indic scripts, CER is often more informative than WER because
word segmentation is less consistent than in English.
"""

import re
import unicodedata
from typing import List

import numpy as np
import evaluate
from transformers import WhisperTokenizer

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# Devanagari punctuation to strip (sentence-final markers, not linguistic content)
_DEVANAGARI_STRIP = str.maketrans("", "", "।॥")


def normalise_text(text: str, language_code: str = "hi") -> str:
    """
    Language-aware normalisation before computing WER/CER.
    Preserves Indic Unicode characters; strips only punctuation.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    if language_code in ("hi", "mr"):
        # Devanagari — strip danda/double-danda, keep anusvara/visarga/chandrabindu
        text = text.translate(_DEVANAGARI_STRIP)
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    elif language_code in ("ta",):
        # Tamil — strip Tamil punctuation markers
        text = re.sub(r"[।॥।॥]", "", text)
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    elif language_code in ("te", "kn"):
        # Telugu / Kannada
        text = re.sub(r"[।॥।॥]", "", text)
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    else:
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


class ASRMetrics:
    """
    Callable passed to Seq2SeqTrainer as compute_metrics.
    Reports WER and CER after each evaluation step.
    """

    def __init__(self, tokenizer: WhisperTokenizer, language_code: str = "hi"):
        self.tokenizer = tokenizer
        self.language_code = language_code

    def __call__(self, eval_preds) -> dict:
        pred_ids, label_ids = eval_preds
        label_ids = label_ids.copy()
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        predictions = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references  = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        predictions = [normalise_text(p, self.language_code) for p in predictions]
        references  = [normalise_text(r, self.language_code) for r in references]

        pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        preds, refs = zip(*pairs)

        wer = wer_metric.compute(predictions=list(preds), references=list(refs))
        cer = cer_metric.compute(predictions=list(preds), references=list(refs))
        return {"wer": round(wer, 4), "cer": round(cer, 4)}


def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    language_code: str = "hi",
) -> dict:
    """
    Standalone evaluation on a list of (prediction, reference) string pairs.
    Returns overall WER/CER plus the 10 worst-performing samples for error analysis.
    """
    assert len(predictions) == len(references), "Length mismatch"

    preds_norm = [normalise_text(p, language_code) for p in predictions]
    refs_norm  = [normalise_text(r, language_code) for r in references]

    pairs = [(p, r) for p, r in zip(preds_norm, refs_norm) if r.strip()]
    if not pairs:
        return {"wer": 1.0, "cer": 1.0, "worst_samples": []}
    preds_norm, refs_norm = zip(*pairs)

    wer = wer_metric.compute(predictions=list(preds_norm), references=list(refs_norm))
    cer = cer_metric.compute(predictions=list(preds_norm), references=list(refs_norm))

    sample_errors = []
    for p, r in zip(preds_norm, refs_norm):
        s_wer = wer_metric.compute(predictions=[p], references=[r])
        sample_errors.append({"prediction": p, "reference": r, "wer": round(s_wer, 4)})

    sample_errors.sort(key=lambda x: x["wer"], reverse=True)

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "worst_samples": sample_errors[:10],
    }
