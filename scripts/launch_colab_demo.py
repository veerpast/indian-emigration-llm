#!/usr/bin/env python3
# ruff: noqa: BLE001, S110, SIM115
# 💬 Complete Live Chat Interface with Streaming (Powered by llama-server)
# Paste this entire cell into Google Colab and run it!

import glob
import json
import os
import shutil
import socket
import subprocess
import tarfile
import time
from pathlib import Path

import requests

# 1. Mount Google Drive if available
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as e:
        print(f"Drive notice: {e}")

# 2. Ensure Gradio is installed
try:
    import gradio as gr
except ImportError:
    os.system('pip install -q gradio')
    import gradio as gr

# 3. Setup the current llama.cpp Ubuntu x64 release (~15s on fresh runtime)
SERVER = '/content/llama-bin/llama-server'
if not os.path.exists(SERVER):
    print("Setting up llama.cpp server engine (~15s)...")
    releases = requests.get(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15", timeout=30
    )
    releases.raise_for_status()
    asset = next(
        asset
        for release in releases.json()
        for asset in release.get("assets", [])
        if asset["name"].endswith("-bin-ubuntu-x64.tar.gz")
    )
    archive = Path('/content/llama-bin.tar.gz')
    extracted = Path('/content/llama-extract')
    binary_dir = Path('/content/llama-bin')
    binary_dir.mkdir(parents=True, exist_ok=True)
    with requests.get(asset["browser_download_url"], stream=True, timeout=180) as response:
        response.raise_for_status()
        with archive.open('wb') as handle:
            shutil.copyfileobj(response.raw, handle)
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as bundle:
        bundle.extractall(extracted, filter='data')
    for source in extracted.rglob('*'):
        if source.is_file():
            shutil.copy2(source, binary_dir / source.name)
    for executable in binary_dir.glob('llama-*'):
        executable.chmod(executable.stat().st_mode | 0o111)
    os.system('cp -a /content/llama-bin/*.so* /usr/lib/ 2>/dev/null || true')
    os.system('ldconfig')

os.environ['LD_LIBRARY_PATH'] = '/content/llama-bin:' + os.environ.get('LD_LIBRARY_PATH', '')

# 4. Search for GGUF models in /content (fast SSD) and Google Drive
search_paths = [
    '/content',
    '/content/drive/MyDrive/Research Project/gguf',
    '/content/drive/MyDrive/Research project/gguf',
    '/content/drive/MyDrive/gguf'
]

available_ggufs = {}
for path in search_paths:
    if os.path.exists(path):
        for f in sorted(glob.glob(f'{path}/*.gguf')):
            label = os.path.basename(f).replace('.gguf', '')
            if label not in available_ggufs:
                available_ggufs[label] = f

if not available_ggufs:
    raise FileNotFoundError("❌ No .gguf files found in /content or Google Drive!")

print("\n✅ Ready models found:")
for k, v in available_ggufs.items():
    print(f"  • {k} ({os.path.getsize(v)/1e9:.2f} GB) at {v}")

# 5. Dynamically assign a guaranteed free port (avoids Colab 8080 collision!)
def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

SERVER_PORT = get_free_port()
print(f"📡 Dedicated llama-server port: {SERVER_PORT}")

# 6. Background llama-server manager (Keeps model in memory & streams responses)
current_server_proc = None
current_loaded_model = None

def ensure_server(model_label):
    global current_server_proc, current_loaded_model
    gguf_path = available_ggufs[model_label]
    
    # Check if this model is already active and healthy
    if current_loaded_model == model_label and current_server_proc and current_server_proc.poll() is None:
        try:
            r = requests.get(f'http://127.0.0.1:{SERVER_PORT}/health', timeout=1)
            if r.status_code == 200:
                return True, ""
        except Exception:
            pass

    # Clean up any existing server instance
    if current_server_proc:
        try:
            current_server_proc.terminate()
            current_server_proc.wait(timeout=3)
        except Exception:
            try: current_server_proc.kill()
            except Exception: pass
        current_server_proc = None

    os.system(f'fuser -k {SERVER_PORT}/tcp 2>/dev/null || true')
    time.sleep(1)

    print(f"⚡ Starting llama-server for '{model_label}' on port {SERVER_PORT}...")
    log_file = open('/tmp/llama_server.log', 'w')
    
    cmd = [
        SERVER,
        '-m', gguf_path,
        '--host', '127.0.0.1',
        '--port', str(SERVER_PORT),
        '-c', '2048',       # 2k context window (lightweight on RAM & CPU)
        '-t', '2'           # 2 threads matching Colab vCPU
    ]
    current_server_proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    current_loaded_model = model_label

    # Wait for server readiness (up to 45 seconds for CPU mmap)
    for i in range(45):
        if current_server_proc.poll() is not None:
            log_file.close()
            with open('/tmp/llama_server.log', 'r') as f:
                err_log = f.read()
            print(f"❌ Server exited prematurely:\n{err_log}")
            return False, err_log
        try:
            r = requests.get(f'http://127.0.0.1:{SERVER_PORT}/health', timeout=1)
            if r.status_code == 200:
                print("✅ Model loaded into RAM and llama-server is ready!")
                return True, ""
        except Exception:
            pass
        time.sleep(1)

    return False, "Server took longer than 45s to load weights."

# Pre-load the first model into memory right now before launching UI
model_options = list(available_ggufs.keys())
print(f"\nInitializing model '{model_options[0]}' into memory...")
ok, err = ensure_server(model_options[0])
if not ok:
    print(f"⚠️ Warning during initial pre-load: {err}")

# 7. Real-time Streaming Chat Handler (Never freezes or times out!)
def chat_handler(user_msg, history, selected_model, max_tokens, temp, top_p):
    ready, err_msg = ensure_server(selected_model)
    if not ready:
        yield f"Error loading '{selected_model}':\n{err_msg[-400:] if err_msg else 'Timed out waiting for server.'}"
        return

    # Build standard messages payload (without generic system prompt so it matches training format)
    messages = []
    for turn in history:
        if isinstance(turn, (list, tuple)):
            u, b = turn[0], turn[1]
            if u: messages.append({"role": "user", "content": u})
            if b: messages.append({"role": "assistant", "content": b})
        elif isinstance(turn, dict):
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
    messages.append({"role": "user", "content": user_msg})

    payload = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temp),
        "top_p": float(top_p),
        "stream": True
    }

    try:
        response = requests.post(f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions", json=payload, stream=True, timeout=120)
        response.raise_for_status()
        accumulated = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:].strip()
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data['choices'][0]['delta'].get('content', '')
                        if delta:
                            accumulated += delta
                            yield accumulated
                    except Exception:
                        pass
        if not accumulated:
            yield "No response generated by model."
    except Exception as e:
        yield f"Inference error: {e}"

# 8. Gradio UI
with gr.Blocks(title="Model Evaluation & Demo", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🚀 Fine-Tuned LLM Live Demonstration")
    gr.Markdown("Real-time streaming inference powered by `llama.cpp` server.")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(choices=model_options, value=model_options[0], label="🤖 Select Model", interactive=True)
            max_tokens_slider = gr.Slider(32, 512, value=192, step=16, label="Max Tokens")
            temp_slider = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="Temperature (0.2 - 0.3 for factual answers)")
            top_p_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            gr.Markdown(
                "### 💡 Demo Tips\n"
                "- **Temperature 0.2 - 0.3**: Best for accurate, factual emigration answers.\n"
                "- **Ask complete questions**: e.g., *'What documents are required for an Indian student visa?'*\n"
                "- **Streaming Active**: Instant token streaming!"
            )
        with gr.Column(scale=3):
            chat = gr.ChatInterface(
                fn=chat_handler,
                additional_inputs=[model_selector, max_tokens_slider, temp_slider, top_p_slider],
                chatbot=gr.Chatbot(height=520)
            )

demo.launch(share=True, debug=False)
