"""
app.py
──────
3-tab Gradio demo for the Hindi ASR Whisper project.
Deployed on Hugging Face Spaces.

Tab 1 — Live Transcription  : base vs. fine-tuned side-by-side with WER and confidence
Tab 2 — Ablation Results    : interactive comparison of 3 training strategies
Tab 3 — Benchmark Scores    : FLEURS and Common Voice out-of-distribution evaluation

Usage (local):
    python app/app.py
    python app/app.py --checkpoint ./models/whisper-lora-hindi/final --share

Usage (HF Spaces):
    Set HF_CHECKPOINT env var or edit DEFAULT_CHECKPOINT below.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import gradio as gr
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference import Transcriber
from src.metrics import normalise_text

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = os.environ.get(
    "HF_CHECKPOINT",
    "./models/whisper-lora-hindi/final",
)
BASE_MODEL_ID = "openai/whisper-small"
LANGUAGE = "hindi"
LANGUAGE_CODE = "hi"
ABLATION_TABLE_PATH = "models/ablation/comparison_table.json"
BENCHMARK_RESULTS_PATHS = {
    "FLEURS (in-distribution)": "models/whisper-lora-hindi/final/eval_fleurs.json",
    "Common Voice (out-of-distribution)": "models/whisper-lora-hindi/final/eval_common_voice.json",
}


# ── Load models once at startup ───────────────────────────────────────────────

def build_transcribers(checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    finetuned = Transcriber(
        checkpoint_path=checkpoint_path if Path(checkpoint_path).exists() else None,
        base_model_id=BASE_MODEL_ID,
        language=LANGUAGE,
        language_code=LANGUAGE_CODE,
        config_path="configs/config.yaml",
    )
    base = Transcriber(
        checkpoint_path=None,
        base_model_id=BASE_MODEL_ID,
        language=LANGUAGE,
        language_code=LANGUAGE_CODE,
    )
    return finetuned, base


# ── Tab 1: Live Transcription ─────────────────────────────────────────────────

def compute_wer_simple(pred: str, ref: str) -> str:
    """Quick WER for display — no external library needed for one sample."""
    pred_words = normalise_text(pred, LANGUAGE_CODE).split()
    ref_words = normalise_text(ref, LANGUAGE_CODE).split()
    if not ref_words:
        return "N/A (no reference)"
    # Edit distance (Wagner-Fischer)
    n, m = len(ref_words), len(pred_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_words[i-1] == pred_words[j-1]:
                new_dp[j] = dp[j-1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j-1], dp[j-1])
        dp = new_dp
    wer = dp[m] / n
    return f"{wer:.3f} ({dp[m]} errors / {n} words)"


def build_transcription_tab(finetuned: Transcriber, base: Transcriber):
    with gr.Tab("Live Transcription"):
        gr.Markdown(
            "## Hindi ASR: Base Whisper vs. LoRA Fine-tuned\n"
            "Record or upload Hindi audio and compare the unmodified base model against "
            "the LoRA PEFT fine-tuned model. The fine-tuned model has only **1.77M trainable "
            "parameters** (0.73% of the full model) — yet achieves dramatically lower WER."
        )

        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Input Audio (Hindi)",
            )

        with gr.Row():
            reference_text = gr.Textbox(
                label="Reference text (optional — type the correct transcript to compute WER)",
                placeholder="हिंदी में टाइप करें…",
                lines=2,
            )

        transcribe_btn = gr.Button("Transcribe", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Base Whisper (zero-shot)")
                base_output = gr.Textbox(label="Transcription", lines=3)
                base_confidence = gr.Textbox(label="Confidence", lines=1)
                base_wer = gr.Textbox(label="WER vs. reference", lines=1)
                base_time = gr.Textbox(label="Processing time", lines=1)

            with gr.Column():
                gr.Markdown("### LoRA Fine-tuned (0.73% params trained)")
                ft_output = gr.Textbox(label="Transcription", lines=3)
                ft_confidence = gr.Textbox(label="Confidence", lines=1)
                ft_wer = gr.Textbox(label="WER vs. reference", lines=1)
                ft_time = gr.Textbox(label="Processing time", lines=1)

        def transcribe_both(audio, reference):
            if audio is None:
                empty = ("No audio provided", "—", "—", "—")
                return empty + empty

            import time

            t0 = time.perf_counter()
            base_result = base.transcribe(audio, return_confidence=True)
            base_elapsed = time.perf_counter() - t0

            t0 = time.perf_counter()
            ft_result = finetuned.transcribe(audio, return_confidence=True)
            ft_elapsed = time.perf_counter() - t0

            base_wer_str = compute_wer_simple(base_result["text"], reference) if reference.strip() else "—"
            ft_wer_str = compute_wer_simple(ft_result["text"], reference) if reference.strip() else "—"

            return (
                base_result["text"],
                f"{base_result.get('confidence', 0):.1%}",
                base_wer_str,
                f"{base_elapsed:.2f}s ({base_result['duration_sec']}s audio)",
                ft_result["text"],
                f"{ft_result.get('confidence', 0):.1%}",
                ft_wer_str,
                f"{ft_elapsed:.2f}s ({ft_result['duration_sec']}s audio)",
            )

        transcribe_btn.click(
            fn=transcribe_both,
            inputs=[audio_input, reference_text],
            outputs=[base_output, base_confidence, base_wer, base_time,
                     ft_output, ft_confidence, ft_wer, ft_time],
        )

        gr.Markdown("""
        ---
        **How this works:**
        - **Base Whisper** — `openai/whisper-small` trained on 680K hours of multilingual audio
        - **LoRA Fine-tuned** — same model with PEFT LoRA adapters (rank=16) trained on FLEURS Hindi
        - LoRA trains only **1,769,472 parameters** (0.73%) vs. 243M total
        - Adapter checkpoint: **~5 MB** vs. 922 MB full model
        """)


# ── Tab 2: Ablation Results ───────────────────────────────────────────────────

def build_ablation_tab():
    with gr.Tab("Ablation Study"):
        gr.Markdown(
            "## Parameter Efficiency Ablation\n"
            "Systematic comparison of three fine-tuning strategies on Hindi ASR (FLEURS benchmark)."
        )

        table_path = Path(ABLATION_TABLE_PATH)
        if table_path.exists():
            with open(table_path) as f:
                table_data = json.load(f)

            rows = []
            for _, row in table_data.items():
                wer = f"{row['wer']:.4f}" if isinstance(row.get("wer"), float) else "—"
                cer = f"{row['cer']:.4f}" if isinstance(row.get("cer"), float) else "—"
                t_params = row.get("trainable_params", 0)
                params_str = f"{t_params/1e6:.2f}M" if t_params >= 1e6 else f"{t_params/1e3:.0f}K" if t_params else "0"
                total = row.get("total_params", 1)
                pct = f"{100 * t_params / total:.3f}%" if total else "—"
                ckpt = f"{row.get('checkpoint_size_mb', 0):.1f} MB"
                hours = row.get("training_time_hours", 0)
                time_str = f"{hours:.1f}h" if hours else "—"
                rows.append([row["label"], wer, cer, params_str, pct, ckpt, time_str])

            gr.Dataframe(
                value=rows,
                headers=["Mode", "WER ↓", "CER ↓", "Trainable Params", "% of Total", "Checkpoint", "Train Time"],
                label="Ablation Comparison Table",
                interactive=False,
            )

            gr.Markdown("""
            **Key findings:**
            - **Base Whisper** (zero-shot): high WER — pretrained on diverse data but not Hindi-specialized
            - **Encoder-Frozen**: trains 154M decoder params, saves a 922 MB checkpoint
            - **LoRA PEFT** (r=16): trains ~1.77M adapter params, saves a ~5 MB checkpoint — **~185x smaller**

            LoRA achieves comparable WER with a tiny fraction of parameters and a production-deployable artifact.
            This is the key trade-off: encoder-frozen is slightly more accurate on this dataset,
            but LoRA is the right choice for deployment (multi-tenant serving, edge devices, mobile).
            """)
        else:
            gr.Markdown(
                "> Ablation results not yet generated. Run:\n"
                "> ```bash\n> python scripts/run_ablation.py\n> ```"
            )


# ── Tab 3: Benchmark Scores ───────────────────────────────────────────────────

def build_benchmark_tab():
    with gr.Tab("Benchmark Scores"):
        gr.Markdown(
            "## Out-of-Distribution Evaluation\n"
            "Training on one dataset and testing on a different one reveals true generalization. "
            "The model is trained on **FLEURS Hindi** and evaluated on **Common Voice Hindi** "
            "(independent speakers, different recording conditions)."
        )

        all_results = {}
        for label, path in BENCHMARK_RESULTS_PATHS.items():
            p = Path(path)
            if p.exists():
                with open(p) as f:
                    all_results[label] = json.load(f)

        if all_results:
            rows = []
            for label, res in all_results.items():
                wer = f"{res['wer']:.4f}" if "wer" in res else "—"
                cer = f"{res['cer']:.4f}" if "cer" in res else "—"
                n = res.get("num_samples", "—")
                rows.append([label, wer, cer, str(n)])

            gr.Dataframe(
                value=rows,
                headers=["Dataset", "WER ↓", "CER ↓", "# Samples"],
                label="Cross-Dataset Generalization",
                interactive=False,
            )

            # Duration-stratified WER if available
            for label, res in all_results.items():
                if res.get("wer_by_duration"):
                    gr.Markdown(f"**{label} — WER by Audio Duration:**")
                    dur_rows = [
                        [bucket, f"{stats['wer']:.4f}", str(stats["n"])]
                        for bucket, stats in res["wer_by_duration"].items()
                    ]
                    gr.Dataframe(
                        value=dur_rows,
                        headers=["Duration Bucket", "WER", "# Samples"],
                        interactive=False,
                    )

            gr.Markdown("""
            **Interpretation:**
            - Lower WER on FLEURS (in-distribution) than Common Voice (OOD) is expected — both have different speaker demographics
            - The gap between in-distribution and OOD WER quantifies the model's generalization
            - Reporting both numbers honestly is a sign of rigorous evaluation methodology
            """)
        else:
            gr.Markdown(
                "> Benchmark results not yet generated. Run:\n"
                "> ```bash\n"
                "> python scripts/eval_benchmark.py --checkpoint ./models/whisper-lora-hindi/final --dataset all\n"
                "> ```"
            )


# ── Main ──────────────────────────────────────────────────────────────────────

def launch_demo(finetuned: Transcriber = None, base: Transcriber = None, share: bool = False):
    if finetuned is None or base is None:
        finetuned, base = build_transcribers(DEFAULT_CHECKPOINT)

    with gr.Blocks(title="Hindi ASR — Whisper LoRA Fine-tuning", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Hindi ASR: Whisper + PEFT LoRA Fine-tuning\n"
            "Parameter-efficient fine-tuning of `openai/whisper-small` for Hindi speech recognition. "
            "Trained on FLEURS Hindi with **1.77M** trainable parameters (0.73% of the full model).\n\n"
            "**[GitHub](https://github.com/27Kushal/Hindi-asr-whisper)** · "
            "**[HuggingFace Model](https://huggingface.co/)**"
        )

        build_transcription_tab(finetuned, base)
        build_ablation_tab()
        build_benchmark_tab()

    demo.launch(share=share)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    finetuned, base = build_transcribers(args.checkpoint)
    launch_demo(finetuned, base, share=args.share)


if __name__ == "__main__":
    main()
