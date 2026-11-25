#!/usr/bin/env python3
# ruff: noqa: BLE001, SIM115, SIM117
"""Professional three-model Gradio demo for Google Colab.

Discovers the three verified GGUF artifacts, keeps only one model resident at a
time, and streams llama-server's OpenAI-compatible chat endpoint.
"""

from __future__ import annotations

import html
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import requests

try:
    import gradio as gr
except ImportError:
    subprocess.run(["python3", "-m", "pip", "install", "-q", "gradio==5.49.1"], check=True)
    import gradio as gr


# Runtime and artifact registry ------------------------------------------------

if not os.path.exists("/content/drive/MyDrive"):
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except Exception as exc:
        print(f"Drive notice: {exc}")

SERVER = "/content/llama-bin/llama-server"
if not os.path.exists(SERVER):
    raise FileNotFoundError(
        "llama-server is missing. Run Cell 4 (Engine & Toolchain Setup) before the demo cell."
    )

os.environ["LD_LIBRARY_PATH"] = "/content/llama-bin:" + os.environ.get("LD_LIBRARY_PATH", "")

SEARCH_PATHS = [
    "/content",
    "/content/drive/MyDrive/Research Project/gguf",
    "/content/drive/MyDrive/Research project/gguf",
    "/content/drive/MyDrive/gguf",
]

MODEL_SPECS = {
    "Mistral 7B": {
        "filename": "Mistral-7B-Finetuned-Q4_K_M.gguf",
        "base": "mistralai/Mistral-7B-Instruct-v0.3",
        "adapter": "LoRA r=64, α=128",
        "description": "Compact baseline with direct instruction-style answers.",
        "chat_kwargs": None,
    },
    "Granite 3.3 8B": {
        "filename": "Granite-3.3-8B-Finetuned-Q4_K_M.gguf",
        "base": "ibm-granite/granite-3.3-8b-instruct",
        "adapter": "LoRA r=64, α=128",
        "description": "IBM Granite instruction model adapted for the research corpus.",
        "chat_kwargs": None,
    },
    "Qwen3 14B": {
        "filename": "Qwen3-14B-Finetuned-Q4_K_M.gguf",
        "base": "Qwen/Qwen3-14B",
        "adapter": "LoRA r=16, α=32",
        "description": "Largest experiment; thinking mode is disabled for concise final answers.",
        "chat_kwargs": {"enable_thinking": False},
    },
}


def discover_models() -> dict[str, dict[str, object]]:
    """Resolve all three exact filenames, preferring Colab's local SSD."""
    resolved: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for label, spec in MODEL_SPECS.items():
        matches: list[str] = []
        for directory in SEARCH_PATHS:
            candidate = os.path.join(directory, str(spec["filename"]))
            if os.path.isfile(candidate):
                matches.append(candidate)
        if not matches:
            missing.append(str(spec["filename"]))
            continue
        path = matches[0]
        resolved[label] = {
            **spec,
            "path": path,
            "size_gb": os.path.getsize(path) / 1e9,
            "location": "Colab SSD" if path.startswith("/content/") and "/drive/" not in path else "Google Drive",
        }
    if missing:
        searched = "\n".join(f"  • {path}" for path in SEARCH_PATHS)
        absent = "\n".join(f"  • {name}" for name in missing)
        raise FileNotFoundError(
            "The demo requires all three verified GGUF files.\n"
            f"Missing:\n{absent}\n\nSearched:\n{searched}\n"
            "Run Cell 3 to audit Drive paths, then Cell 5 if conversion is incomplete."
        )
    return resolved


AVAILABLE_MODELS = discover_models()
MODEL_OPTIONS = list(MODEL_SPECS)
print("\nVerified GGUF artifacts:")
for label, spec in AVAILABLE_MODELS.items():
    print(f"  • {label:<18} {spec['size_gb']:.2f} GB · {spec['location']} · {spec['path']}")


# llama-server lifecycle -------------------------------------------------------


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
    ).returncode == 0


SERVER_PORT = get_free_port()
BACKEND_MARKER = Path("/content/llama-bin/backend.txt")
USE_GPU = (
    has_nvidia_gpu()
    and BACKEND_MARKER.exists()
    and BACKEND_MARKER.read_text(encoding="utf-8").strip() == "cuda"
)
MODEL_LOCK = threading.RLock()
current_server_proc: subprocess.Popen | None = None
current_log_handle = None
current_loaded_model: str | None = None


def tail_log(limit: int = 3500) -> str:
    try:
        return Path("/tmp/llama_server.log").read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return "No llama-server log is available."


def stop_server() -> None:
    global current_server_proc, current_log_handle, current_loaded_model
    if current_server_proc is not None:
        current_server_proc.terminate()
        try:
            current_server_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            current_server_proc.kill()
            current_server_proc.wait(timeout=3)
    if current_log_handle is not None:
        current_log_handle.close()
    current_server_proc = None
    current_log_handle = None
    current_loaded_model = None


def is_healthy() -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def ensure_server(model_label: str, timeout_seconds: int = 300) -> tuple[bool, str, float]:
    """Load one selected model. Switching cleanly unloads the previous model."""
    global current_server_proc, current_log_handle, current_loaded_model
    with MODEL_LOCK:
        if model_label not in AVAILABLE_MODELS:
            return False, f"Unknown model: {model_label}", 0.0
        if current_loaded_model == model_label and current_server_proc and is_healthy():
            return True, "already loaded", 0.0

        stop_server()
        subprocess.run(
            ["fuser", "-k", f"{SERVER_PORT}/tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        spec = AVAILABLE_MODELS[model_label]
        command = [
            SERVER,
            "-m",
            str(spec["path"]),
            "--host",
            "127.0.0.1",
            "--port",
            str(SERVER_PORT),
            "-c",
            "4096",
            "-t",
            str(max(2, min(8, os.cpu_count() or 2))),
            "--jinja",
            "--cache-prompt",
        ]
        if USE_GPU:
            command.extend(["--n-gpu-layers", "99"])
        if spec["chat_kwargs"]:
            command.extend(["--chat-template-kwargs", json.dumps(spec["chat_kwargs"])])

        current_log_handle = open("/tmp/llama_server.log", "w", encoding="utf-8")
        started = time.monotonic()
        current_server_proc = subprocess.Popen(
            command,
            stdout=current_log_handle,
            stderr=current_log_handle,
            env=os.environ.copy(),
        )
        current_loaded_model = model_label

        while time.monotonic() - started < timeout_seconds:
            if current_server_proc.poll() is not None:
                error = tail_log()
                stop_server()
                return False, error, time.monotonic() - started
            if is_healthy():
                return True, "ready", time.monotonic() - started
            time.sleep(1)

        error = f"Model loading exceeded {timeout_seconds} seconds.\n{tail_log()}"
        stop_server()
        return False, error, time.monotonic() - started


# Inference and UI helpers -----------------------------------------------------


def status_html(state: str, message: str) -> str:
    safe_message = html.escape(message)
    return (
        f'<div class="runtime-status runtime-{state}">'
        f'<span class="status-dot" aria-hidden="true"></span><span>{safe_message}</span></div>'
    )


def model_metadata(model_label: str) -> str:
    spec = AVAILABLE_MODELS[model_label]
    runtime = "T4 GPU offload" if USE_GPU else "CPU inference"
    return (
        f"**Base model**  \n`{spec['base']}`\n\n"
        f"**Adapter**  \n{spec['adapter']}\n\n"
        f"**Artifact**  \n{spec['size_gb']:.2f} GB · {spec['location']}\n\n"
        f"**Runtime**  \n{runtime} · 4,096-token context\n\n"
        f"{spec['description']}"
    )


def load_model_ui(model_label: str):
    ok, detail, elapsed = ensure_server(model_label)
    if ok:
        suffix = "already resident" if elapsed == 0 else f"loaded in {elapsed:.1f}s"
        return status_html("ready", f"{model_label} ready · {suffix}"), model_metadata(model_label), []
    concise = detail.splitlines()[-1] if detail else "unknown server error"
    return status_html("error", f"Could not load {model_label}: {concise}"), model_metadata(model_label), []


def normalize_history(history) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content.strip()})
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_text, assistant_text = turn[0], turn[1]
            if user_text:
                messages.append({"role": "user", "content": str(user_text)})
            if assistant_text:
                messages.append({"role": "assistant", "content": str(assistant_text)})
    return messages


def chat_handler(user_message, history, selected_model, max_tokens, temperature, top_p):
    question = str(user_message or "").strip()
    if not question:
        yield "Enter a complete question before asking the model."
        return

    if current_loaded_model != selected_model or not is_healthy():
        yield f"Loading {selected_model}. The first switch can take one to three minutes from Drive…"
        ready, error, _ = ensure_server(selected_model)
        if not ready:
            yield f"Model loading failed. Open the cell output for the llama-server log.\n\n{error[-800:]}"
            return

    messages = normalize_history(history)
    messages.append({"role": "user", "content": question})
    payload = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repeat_penalty": 1.1,
        "seed": 42,
        "cache_prompt": True,
        "stream": True,
    }

    accumulated = ""
    try:
        with requests.post(
            f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, 600),
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data_string = line[6:].strip()
                if data_string == "[DONE]":
                    break
                try:
                    event = json.loads(data_string)
                    delta = event.get("choices", [{}])[0].get("delta", {}).get("content", "")
                except (json.JSONDecodeError, IndexError, TypeError):
                    continue
                if delta:
                    accumulated += delta
                    yield accumulated
    except requests.RequestException as exc:
        yield f"Inference request failed: {exc}\n\nServer log:\n{tail_log(800)}"
        return

    if not accumulated.strip():
        yield "The model returned an empty response. Try a complete question or raise max tokens."


CUSTOM_CSS = """
:root {
  --research-navy: #0b1f33;
  --research-blue: #2457d6;
  --research-paper: #f7f9fc;
  --research-line: #d7e0ea;
  --research-muted: #5e6f82;
  --research-red: #b84040;
}
.gradio-container { background: var(--research-paper) !important; color: #142231; }
.app-shell { max-width: 1380px; margin: 0 auto; padding: 10px 18px 28px; }
.research-header {
  padding: 24px 28px; color: white; background: var(--research-navy);
  border-left: 6px solid var(--research-blue); border-radius: 8px;
}
.research-header h1 { margin: 0 0 7px; color: #ffffff !important; font-size: 30px; line-height: 1.1; letter-spacing: -0.02em; }
.research-header p { margin: 0; color: #c8d5e2; font-size: 15px; }
.control-panel { background: white; border: 1px solid var(--research-line); border-radius: 8px; padding: 18px; }
.control-panel h3 { color: var(--research-navy); margin-top: 0; }
.runtime-status {
  display: flex; align-items: center; gap: 9px; margin: 8px 0 16px; padding: 10px 12px;
  border: 1px solid var(--research-line); border-radius: 6px; background: #f3f6fa; font-size: 14px;
}
.status-dot { width: 9px; height: 9px; border-radius: 50%; background: #74869a; flex: 0 0 auto; }
.runtime-ready .status-dot { background: #14845c; }
.runtime-error { color: #842f2f; background: #fff4f4; border-color: #e7baba; }
.runtime-error .status-dot { background: var(--research-red); }
.model-metadata { color: var(--research-muted); font-size: 14px; }
.model-metadata code { white-space: normal !important; overflow-wrap: anywhere; }
.chat-panel { border: 1px solid var(--research-line) !important; border-radius: 8px !important; background: white !important; }
.chat-panel .message.user { background: #e9efff !important; color: #10244d !important; }
.chat-panel .message.bot { background: #f3f6fa !important; color: #142231 !important; }
.notice {
  margin-top: 12px; padding: 10px 14px; border-left: 3px solid var(--research-red);
  background: #fff7f7; color: #6f3535; font-size: 13px;
}
footer { display: none !important; }
@media (max-width: 760px) {
  .app-shell { padding: 6px 4px 20px; }
  .research-header { padding: 20px; }
  .research-header h1 { font-size: 25px; }
}
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Source Sans 3"), "Arial", "sans-serif"],
).set(
    button_primary_background_fill="#2457d6",
    button_primary_background_fill_hover="#1d48b4",
    button_primary_border_color="#2457d6",
    block_border_width="0px",
    input_border_color="#b8c5d3",
    input_border_color_focus="#2457d6",
)

with gr.Blocks(title="Indian Emigration Q&A Model Lab", theme=theme, css=CUSTOM_CSS) as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.HTML(
            """
            <header class="research-header">
              <h1>Indian Emigration Q&amp;A Model Lab</h1>
              <p>Comparative inference interface · Mistral 7B · Granite 3.3 8B · Qwen3 14B</p>
            </header>
            """
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=290, elem_classes="control-panel"):
                gr.Markdown("### Model console")
                model_selector = gr.Dropdown(
                    choices=MODEL_OPTIONS,
                    value=MODEL_OPTIONS[0],
                    label="Fine-tuned model",
                    info="Changing models unloads the previous artifact to stay within Colab memory.",
                    interactive=True,
                )
                model_status = gr.HTML(status_html("idle", "Select a model; it loads on first use"))
                model_info = gr.Markdown(model_metadata(MODEL_OPTIONS[0]), elem_classes="model-metadata")
                with gr.Accordion("Generation settings", open=False):
                    max_tokens = gr.Slider(64, 768, value=320, step=32, label="Maximum new tokens")
                    temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                gr.Markdown(
                    "**Recommended:** use complete, specific questions. Keep temperature at 0.2–0.3 for the formal demo."
                )

            with gr.Column(scale=3, min_width=520):
                chatbot = gr.Chatbot(
                    type="messages",
                    height=460,
                    show_label=False,
                    elem_classes="chat-panel",
                    placeholder=(
                        "Select a model, then ask about visa interviews, ECR/ECNR, "
                        "financial documentation, or emigration procedures."
                    ),
                )
                chat = gr.ChatInterface(
                    fn=chat_handler,
                    type="messages",
                    chatbot=chatbot,
                    textbox=gr.Textbox(
                        placeholder="Ask a complete Indian emigration or visa question…",
                        container=False,
                        scale=8,
                    ),
                    additional_inputs=[model_selector, max_tokens, temperature, top_p],
                    examples=[
                        ["What is the difference between ECR and ECNR status?"],
                        ["How should a student explain their choice of university in a visa interview?"],
                        ["Which financial documents may be requested during a student visa application?"],
                    ],
                    cache_examples=False,
                    submit_btn="Ask model",
                    stop_btn="Stop",
                    concurrency_limit=1,
                )
                gr.HTML(
                    '<div class="notice"><strong>Research prototype.</strong> Outputs may be incomplete or outdated. '
                    "Verify legal, consular, and immigration guidance against current official sources.</div>"
                )

        model_selector.change(
            fn=load_model_ui,
            inputs=model_selector,
            outputs=[model_status, model_info, chatbot],
            concurrency_limit=1,
        )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=12).launch(
        share=True,
        debug=False,
        show_error=True,
    )

