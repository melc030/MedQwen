"""
MedQwen Gradio Chatbot

Talks to an OpenAI-compatible inference server:
  - Local Mac  : MLX-LM  → python serve/mlx_serve.py   (port 8080)
  - Cloud GPU  : vLLM    → python serve/vllm_serve.py  (port 8000)

Usage:
    # 1. start the inference server in another terminal
    python serve/mlx_serve.py

    # 2. launch the chat UI
    python app.py

    # point at a different server (e.g. vLLM on Azure)
    INFERENCE_URL=http://<vm-ip>:8000 python app.py
"""

import os
import gradio as gr
from openai import OpenAI

# ── Server config ─────────────────────────────────────────────────────────────
INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8080")
# MLX server uses "MedQwen", vLLM uses the lora-module name "medqwen"
MODEL_NAME    = os.environ.get("MODEL_NAME", "MedQwen")
SYSTEM_PROMPT = "你是一个专业的医疗问答助手，请根据用户的问题给出准确、简洁的医疗建议。"
MAX_TOKENS    = 512
TEMPERATURE   = 0.7

client = OpenAI(
    base_url=f"{INFERENCE_URL}/v1",
    api_key="not-needed",          # MLX / vLLM don't require auth locally
)


def chat(message, history):
    """
    history: list of {"role": ..., "content": ...} dicts (Gradio 6 messages format)
    Returns updated history with the new assistant reply appended.
    """
    def extract_text(content):
        # Gradio 6 may wrap content as [{"type": "text", "text": "..."}]
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""

    # build API payload from history (skip any existing system message)
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        if turn["role"] != "system":
            api_messages.append({"role": turn["role"], "content": extract_text(turn["content"])})
    api_messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=api_messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
    )
    reply = response.choices[0].message.content

    # return updated history
    updated = list(history)
    updated.append({"role": "user",      "content": message})
    updated.append({"role": "assistant", "content": reply})
    return updated


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MedQwen — 医疗问答助手") as demo:
    gr.Markdown(
        """
        # 🏥 MedQwen 医疗问答助手
        **Fine-tuned Qwen2.5-1.5B-Instruct on 30K Chinese medical Q&A pairs (LoRA)**

        > 本助手仅供参考，不能替代专业医生的诊断和建议。
        """
    )

    chatbot = gr.Chatbot(
        label="对话",
        height=500,
        buttons=["copy", "copy_all"],
        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=medqwen"),
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="请输入您的医疗问题，例如：糖尿病的早期症状有哪些？",
            label="",
            scale=9,
            autofocus=True,
        )
        send_btn = gr.Button("发送", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.ClearButton([msg, chatbot], value="清除对话")
        gr.Markdown(
            f"<small>连接至: `{INFERENCE_URL}` &nbsp;|&nbsp; "
            f"max_tokens={MAX_TOKENS} &nbsp;|&nbsp; temperature={TEMPERATURE}</small>"
        )

    gr.Examples(
        examples=[
            "无症状颈动脉粥样硬化的影像学检查有些什么？",
            "糖尿病人能喝不含蔗糖的中老年奶粉吗？",
            "头孢地尼胶囊能治理什么疾病？",
            "宝宝支气管炎要如何治疗？",
            "我最近头疼、发烧，可能是什么病？",
        ],
        inputs=msg,
        label="示例问题",
    )

    # wire up both Enter key and Send button
    msg.submit(chat, [msg, chatbot], [chatbot]).then(fn=lambda: "", outputs=msg)
    send_btn.click(chat, [msg, chatbot], [chatbot]).then(fn=lambda: "", outputs=msg)

if __name__ == "__main__":
    print(f"Connecting to inference server at: {INFERENCE_URL}")
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,          # set True for a public Gradio link
        inbrowser=True,       # auto-opens browser tab
        theme=gr.themes.Soft(),
    )
