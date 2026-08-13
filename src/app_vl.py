"""
MedQwen-VL Gradio app — multimodal medical image Q&A (English).

Upload a medical image (dermatology, endoscopy, fundus, X-ray, ultrasound),
ask a question, and the fine-tuned Qwen2.5-VL-3B model answers. Talks to an
OpenAI-compatible vLLM server that supports image_url content.

Usage:
    # 1. start the VL server (on the GPU host)
    python src/serve/vllm_serve_vl.py

    # 2. launch the UI (point at the server if remote)
    python src/app_vl.py
    INFERENCE_URL=http://<vm-ip>:8000 python src/app_vl.py
"""

import base64
import mimetypes
import os

import gradio as gr
from openai import OpenAI

# ── Server config ─────────────────────────────────────────────────────────────
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8000")
MODEL_NAME    = os.environ.get("MODEL_NAME", "medqwen-vl")  # lora-module name in vllm_serve_vl
# Trained without a system prompt — keep serving faithful (see cfg.vl_system_prompt).
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "") or None
MAX_TOKENS    = 128
TEMPERATURE   = 0.0   # deterministic — classification-style task

client = OpenAI(base_url=f"{INFERENCE_URL}/v1", api_key="not-needed")


def _data_url(image_path):
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def answer(image_path, question):
    if not image_path:
        return "⚠️ Please upload a medical image first."
    if not question or not question.strip():
        question = "What does this medical image show?"

    # single-turn multimodal message (matches how the model was trained)
    user_content = [
        {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
        {"type": "text",      "text": question},
    ]
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
    )
    return resp.choices[0].message.content


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MedQwen-VL — Medical Image Q&A") as demo:
    gr.Markdown(
        """
        # 🏥 MedQwen-VL — Medical Image Q&A
        **Fine-tuned Qwen2.5-VL-3B (LoRA) · dermatology · endoscopy · fundus · X-ray · ultrasound**

        > For research/demo only — not a substitute for professional medical diagnosis.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(label="Medical image", type="filepath", height=360)
            question = gr.Textbox(
                label="Question",
                value="What does this medical image show?",
                placeholder="e.g. What does this medical image show?",
            )
            with gr.Row():
                submit = gr.Button("Analyze", variant="primary")
                clear  = gr.ClearButton([image, question])
        with gr.Column(scale=1):
            output = gr.Textbox(label="Answer", lines=10)

    gr.Examples(
        examples=[
            "What does this medical image show?",
            "What imaging modality is this?",
            "What is the ICD-10 code for the condition shown in this image?",
        ],
        inputs=question,
        label="Example questions",
    )

    gr.Markdown(
        f"<small>Connected to: <code>{INFERENCE_URL}</code> &nbsp;|&nbsp; "
        f"model=<code>{MODEL_NAME}</code> &nbsp;|&nbsp; "
        f"max_tokens={MAX_TOKENS} &nbsp;|&nbsp; temperature={TEMPERATURE}</small>"
    )

    submit.click(answer, [image, question], [output])
    question.submit(answer, [image, question], [output])

if __name__ == "__main__":
    print(f"Connecting to inference server at: {INFERENCE_URL}")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
