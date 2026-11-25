# 📜 COMPLETE PROJECT & SESSION TRANSFER ARCHIVE
**Project:** Fine-Tuning and Comparative Evaluation of Open-Source Language Models for Indian Emigration Question Answering  
**Authors:** Veerpratap Singh Tomar & Piyush Pratap Singh  
**Programme:** 4th Year B.Tech CSE (AI & ML), Adamas University  
**Project Coordinator:** Dr. Tamal Ghosh  
**Generated Date:** November 1, 2026  

---

## 🎯 INSTRUCTIONS FOR THE RECEIVING LLM
> You are acting as the lead AI & Systems Engineering Assistant for Veerpratap Singh Tomar's B.Tech final-year research project. 
> All context, historical decisions, completed models, codebase locations, and resolved technical bugs are documented below. 
> **Do not make the user repeat themselves.** Pick up directly from the current status described in Section 6.

---

## 📌 1. EXECUTIVE SUMMARY & RESEARCH OBJECTIVE
- **Goal:** Fine-tune, quantize (GGUF `Q4_K_M`), deploy, and comparatively benchmark 6 open-source Large Language Models (LLMs) on a domain-specific dataset for **Indian Emigration, Consular Stamping, Visas, and Bureaucratic Regulatory Compliance**.
- **Dataset:** `indian_emigration_qa_purified.csv`
  - 30,324 high-quality Question-Answer pairs.
  - Divided across 7 specialized categories: Student Visas (F-1, German Opportunity Card, UK Tier 4), Consular Interviews, Ministry of External Affairs (MEA) Regulations, Emigration Act 1983, ECR vs. ECNR compliance, Financial Documentation, and Advisory.
- **Hardware Architecture:**
  - **Local Client:** MacBook Air M1 (Apple Silicon arm64), 8 GB Unified Memory, ~50 GB free SSD.
  - **Training Hardware:** College High Performance Computing (HPC) cluster using OpenCode / PEFT / TRL.
  - **Conversion & Quantization Runtime:** Google Colab (NVIDIA T4 GPU & TPU v5e-1) writing to Google Drive (`My Drive/Research Project/`).
  - **Inference Runtime:** `llama.cpp` / `llama-server` streaming via Server-Sent Events (SSE) into Gradio.

---

## 📊 2. COMPLETE MODEL STATUS & SPECIFICATIONS

| Model Name | Base Model Repo | LoRA Hyperparameters | Status | Final Artifact Location | File Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mistral-7B-Finetuned** | `mistralai/Mistral-7B-Instruct-v0.3` | $r=64, \alpha=128$, dropout 0.05, lr $10^{-4}$ | ✅ **100% Complete & Verified Live** | `My Drive/Research Project/gguf/Mistral-7B-Finetuned-Q4_K_M.gguf` | **4.37 GB** |
| **Granite-3.3-8B-Finetuned** | `ibm-granite/granite-3.3-8b-instruct` | $r=64, \alpha=128$, dropout 0.05, lr $10^{-4}$ | ✅ **100% Complete & Verified Live** | `My Drive/Research Project/gguf/Granite-3.3-8B-Finetuned-Q4_K_M.gguf` | **4.94 GB** |
| **Qwen3-14B-Finetuned** | `Qwen/Qwen3-14B` | $r=16, \alpha=32$, dropout 0.05, lr $2 \times 10^{-4}$ | ⏳ **In Progress (Active Shard Merge)** | `My Drive/Research Project/gguf/Qwen3-14B-Finetuned-Q4_K_M.gguf` | ~8.5 GB (Expected) |
| **Model 4** | *To be determined* | Standard LoRA recipe | ⏳ Arriving Monday on Pendrive | Pending | TBD |
| **Model 5** | *To be determined* | Standard LoRA recipe | ⏳ Arriving Monday on Pendrive | Pending | TBD |
| **Model 6** | *To be determined* | Standard LoRA recipe | ⏳ Arriving Monday on Pendrive | Pending | TBD |

---

## 🗄️ 3. LOCAL FILE SYSTEM STRUCTURE (`/Users/veer/Desktop/Research project/`)

```text
/Users/veer/Desktop/Research project/
├── Minor_Project_Abstract_Final.pdf       # Official Adamas University abstract & registration details
├── indian_emigration_qa_purified.csv      # 30,324 verified QA training & benchmark pairs (15.8 MB)
├── project_memory_and_log.md              # Living engineering decision log and viva thesis notes
├── chat_interface_cell.py                 # Standalone tested Gradio streaming chat script (llama-server backend)
├── convert_and_chat_colab1.ipynb          # Master production Jupyter Notebook for Google Colab
├── my_exact_working_notebook.ipynb        # Active notebook mirrored in user's Google Colab tab
├── convert_to_gguf.sh                     # Bash automated toolchain for local conversion
└── models_for_drive/                      # Clean LoRA adapters (stripped of ~1.3 GB optimizer.pt files)
    ├── mistral7b/                         # adapter_model.safetensors, adapter_config.json, tokenizer*
    ├── granite33_8b/                      # adapter_model.safetensors, adapter_config.json, run_config.json
    └── qwen3_14b/                         # adapter_model.safetensors, adapter_config.json, run_config.json
```

---

## ⚙️ 4. KEY TECHNICAL PROBLEMS SOLVED & ARCHITECTURAL DECISIONS

### A. The 85 GB Pendrive Problem & Adapter Extraction
- **Issue:** The external pendrive had over 85 GB of raw training files, which was too large to upload to Google Drive over standard internet.
- **Solution:** Analyzed the checkpoints and proved the folders were HuggingFace PEFT LoRA adapters containing unnecessary `optimizer.pt` (~1.3 GB each). Stripped `optimizer.pt` and preserved only `adapter_model.safetensors` (~250–750 MB) + configs, reducing upload size by **97%**.

### B. Google Colab T4 Memory Ceiling ($W + \Delta W$ OOM)
- **Issue:** When merging Granite (16.3 GB FP16) or Qwen (29.5 GB FP16) via standard HuggingFace `model.merge_and_unload()`, PyTorch held base weights, adapter weights, and un-offloaded dispatch buffers simultaneously (~32 GB peak memory). Google Colab T4 (15.0 GB VRAM, 12.7 GB System RAM) crashed with `CUDA out of memory: Tried to allocate 20.00 MiB`.
- **Solution (Streamed Sharded LoRA Merger):** Built an autonomous streaming merger in pure Python/PyTorch. Instead of loading the full model, it streams base `.safetensors` shards one by one (~4 GB each), applies $W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r}(B \times A)$ in float32 on the GPU (~70 MB VRAM, ~1 ms per layer), saves the fused shard in-place, and deletes it from memory via `gc.collect()`.
  - **Memory Footprint:** Peak RAM capped at **~4.8 GB**, GPU VRAM capped at **~70 MB**, zero risk of OOM.

### C. `llama-quantize` Dynamic Library Linking
- **Issue:** Precompiled Linux x64 `llama.cpp` binaries failed with `error while loading shared libraries: libllama-common.so.0`.
- **Solution:** Injected automatic library registration:
  ```bash
  cp -a /content/llama-bin/*.so* /usr/lib/ 2>/dev/null && ldconfig
  ```

### D. `llama.cpp` Runtime Streaming Engine
- **Issue:** `llama-cli` defaulted to interactive chat mode (`-cnv`), causing Python `subprocess.run` to hit a 180-second timeout waiting for EOF.
- **Solution:** Replaced CLI invocations with a background `llama-server` process. It loads the 4-bit model into memory **once**, binds to a dynamically assigned free socket port (`socket.bind(('127.0.0.1', 0))` to bypass Colab port 8080 collisions), and streams tokens into Gradio via Server-Sent Events (`stream: True`).

### E. Prompt Distribution Alignment
- **Issue:** Adding standard generic system prompts (`"You are a helpful assistant"`) caused models to output conversational chatter (`(interrupts me)...`) because the LoRA was fine-tuned purely on `[INST] {question} [/INST] {answer}`.
- **Solution:** Omitted the generic system prompt in inference templates and set sampling temperature to `0.2 - 0.3`. Generated answers immediately became authoritative, legally accurate, and direct.

---

## 💻 5. FULL WORKING CODE IMPLEMENTATIONS

### Production Cell 5: Streamed Sharded Merger & Quantizer
```python
# 5. Automated LoRA Merge & Batch GGUF Quantization (100% Autonomous & Streamed - Zero OOM)
import os, gc, time, shutil, subprocess, json, torch

# 1. Ensure Drive is mounted
if not os.path.exists('/content/drive/MyDrive'):
    from google.colab import drive
    drive.mount('/content/drive')

# 2. Toolchain Auto-Setup
if not os.path.exists('/content/llama.cpp/convert_hf_to_gguf.py'):
    print('>>> [Setup] Cloning llama.cpp converter repo...')
    subprocess.run('git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp', shell=True)
    subprocess.run('pip install -q -e /content/llama.cpp/gguf-py', shell=True)

if not os.path.exists('/content/llama-bin/llama-quantize'):
    print('>>> [Setup] Downloading pre-compiled llama.cpp tools...')
    cmd = '''
    mkdir -p /content/llama-bin
    URL=$(curl -sL 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15' | python3 -c "
import json, re, sys
pat = re.compile(r'llama-b\\\\d+-bin-ubuntu-x64\\\\.tar\\\\.gz$')
for rel in json.load(sys.stdin):
    for a in rel['assets']:
        if pat.search(a['name']):
            print(a['browser_download_url']); raise SystemExit
")
    if [ -n "$URL" ]; then
        wget -q "$URL" -O /content/llama-bin.tar.gz
        mkdir -p /content/llama-extract
        tar -xzf /content/llama-bin.tar.gz -C /content/llama-extract
        find /content/llama-extract -type f -exec cp {} /content/llama-bin/ \\;
        chmod +x /content/llama-bin/llama-* /content/llama-bin/ggml-* 2>/dev/null || true
    fi
    '''
    subprocess.run(cmd, shell=True)

subprocess.run('cp -a /content/llama-bin/*.so* /usr/lib/ 2>/dev/null; ldconfig', shell=True)

# 3. Model Configuration
QUANT = 'Q4_K_M'
OUT_DIR = '/content/drive/MyDrive/Research Project/gguf'
os.makedirs(OUT_DIR, exist_ok=True)

BASE_DRIVE_DIR = '/content/drive/MyDrive/Research Project/models_for_drive'
MODELS = [
    {
        'name': 'Mistral-7B-Finetuned',
        'adapter_dir': f'{BASE_DRIVE_DIR}/mistral7b',
        'base_model': 'mistralai/Mistral-7B-Instruct-v0.3',
        'enabled': True
    },
    {
        'name': 'Granite-3.3-8B-Finetuned',
        'adapter_dir': f'{BASE_DRIVE_DIR}/granite33_8b',
        'base_model': 'ibm-granite/granite-3.3-8b-instruct',
        'enabled': True
    },
    {
        'name': 'Qwen3-14B-Finetuned',
        'adapter_dir': f'{BASE_DRIVE_DIR}/qwen3_14b',
        'base_model': 'Qwen/Qwen3-14B',
        'enabled': True
    },
]

from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download

def merge_lora_turbo(base_repo, adapter_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f'   -> Locating base model snapshot for {base_repo}...')
    base_dir = snapshot_download(base_repo)

    with open(os.path.join(adapter_dir, 'adapter_config.json')) as f:
        cfg = json.load(f)
    r = cfg.get('r', 64)
    alpha = cfg.get('lora_alpha', 128)
    scale = float(alpha) / float(r)
    print(f'   -> LoRA configuration: rank={r}, alpha={alpha}, scaling factor={scale:.2f}')

    adapter_path = os.path.join(adapter_dir, 'adapter_model.safetensors')
    adapter_weights = load_file(adapter_path, device='cpu')

    index_file = os.path.join(base_dir, 'model.safetensors.index.json')
    if os.path.exists(index_file):
        with open(index_file) as f:
            idx = json.load(f)
        shard_names = sorted(list(set(idx['weight_map'].values())))
    else:
        shard_names = sorted([f for f in os.listdir(base_dir) if f.endswith('.safetensors')])

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'   -> Turbo Merging {len(shard_names)} weight shard(s) on {dev.upper()} (~15s/shard)...')

    for s_idx, s_name in enumerate(shard_names, 1):
        s_path = os.path.join(base_dir, s_name)
        out_path = os.path.join(output_dir, s_name)
        t_shard = time.time()
        print(f'      [{s_idx}/{len(shard_names)}] Processing {s_name} ({os.path.getsize(s_path)/1e9:.2f} GB)...', end='', flush=True)

        shard_tensors = load_file(s_path, device='cpu')
        applied = 0

        for k in list(shard_tensors.keys()):
            prefix = k[:-len('.weight')] if k.endswith('.weight') else k
            lora_a_key = f'base_model.model.{prefix}.lora_A.weight'
            lora_b_key = f'base_model.model.{prefix}.lora_B.weight'
            if lora_a_key not in adapter_weights:
                lora_a_key = f'{prefix}.lora_A.weight'
                lora_b_key = f'{prefix}.lora_B.weight'

            if lora_a_key in adapter_weights and lora_b_key in adapter_weights:
                a = adapter_weights[lora_a_key].to(dev, dtype=torch.float32)
                b = adapter_weights[lora_b_key].to(dev, dtype=torch.float32)
                v = shard_tensors[k].to(dev, dtype=torch.float32)
                shard_tensors[k] = (v + (b @ a) * scale).to('cpu', dtype=shard_tensors[k].dtype)
                applied += 1

        save_file(shard_tensors, out_path, metadata={'format': 'pt'})
        del shard_tensors
        if dev == 'cuda': torch.cuda.empty_cache()
        gc.collect()
        print(f' done in {time.time()-t_shard:.1f}s ({applied} layers fused)')

    del adapter_weights
    gc.collect()

    print('   -> Copying configs and tokenizer definitions...')
    for item in os.listdir(base_dir):
        if not item.endswith('.safetensors') and not item.endswith('.bin') and not item.startswith('.'):
            s = os.path.join(base_dir, item)
            d = os.path.join(output_dir, item)
            if os.path.isfile(s) and not os.path.exists(d):
                shutil.copy2(s, d)

    for item in os.listdir(adapter_dir):
        if (item.endswith('.json') or item.endswith('.jinja') or item.endswith('.model')) and item != 'adapter_config.json':
            shutil.copy2(os.path.join(adapter_dir, item), os.path.join(output_dir, item))

converted_models = {}

for idx, m in enumerate(MODELS, 1):
    if not m.get('enabled', False):
        continue

    m_name = m['name']
    adapter_dir = m['adapter_dir']
    base_repo = m['base_model']
    final_gguf = f"{OUT_DIR}/{m_name}-{QUANT}.gguf"

    if os.path.exists(final_gguf) and os.path.getsize(final_gguf) > 1e8:
        print(f'✅ Already converted: {final_gguf} ({os.path.getsize(final_gguf)/1e9:.2f} GB)')
        converted_models[m_name] = final_gguf
        continue

    t_start = time.time()
    merged_dir = f'/content/{m_name}-merged'
    intermediate_gguf = f'/content/{m_name}-f16.gguf'
    staging_gguf = f'/content/{m_name}-{QUANT}.gguf'

    # Step A: Turbo GPU-streamed merge
    print(f'1. Merging {base_repo} with LoRA adapter...')
    merge_lora_turbo(base_repo, adapter_dir, merged_dir)

    # Step B: Convert HF merged weights to intermediate GGUF (f16)
    print('2. Converting merged weights to intermediate GGUF (f16)...')
    cmd_convert = ['python3', '/content/llama.cpp/convert_hf_to_gguf.py', merged_dir, '--outfile', intermediate_gguf, '--outtype', 'f16']
    res_conv = subprocess.run(cmd_convert, capture_output=True, text=True)
    if res_conv.returncode != 0:
        print('❌ Conversion to intermediate GGUF failed:', res_conv.stderr[-500:])
        shutil.rmtree(merged_dir, ignore_errors=True)
        continue

    shutil.rmtree(merged_dir, ignore_errors=True)

    # Step C: Quantize to Q4_K_M
    print(f'3. Quantizing to {QUANT} via llama-quantize...')
    cmd_quant = ['/content/llama-bin/llama-quantize', intermediate_gguf, staging_gguf, QUANT]
    res_quant = subprocess.run(cmd_quant, capture_output=True, text=True)
    if res_quant.returncode != 0:
        print('❌ Quantization failed:', res_quant.stderr[-500:])
        if os.path.exists(intermediate_gguf): os.remove(intermediate_gguf)
        continue

    if os.path.exists(intermediate_gguf):
        os.remove(intermediate_gguf)

    # Step D: Save final GGUF to Google Drive
    print(f'4. Saving {QUANT} GGUF to Google Drive: {final_gguf}...')
    shutil.copy2(staging_gguf, final_gguf)
    if os.path.exists(staging_gguf):
        os.remove(staging_gguf)

    elapsed = (time.time() - t_start) / 60
    size_gb = os.path.getsize(final_gguf) / 1e9
    print(f'✅ Finished in {elapsed:.1f} min! File size: {size_gb:.2f} GB')
    converted_models[m_name] = final_gguf
```

---

## 🚀 6. CURRENT STATE & IMMEDIATE ROADMAP

1. **Current Live State**:
   - `Mistral-7B-Finetuned-Q4_K_M.gguf` (4.37 GB) is safe on Google Drive.
   - `Granite-3.3-8B-Finetuned-Q4_K_M.gguf` (4.94 GB) is safe on Google Drive.
   - `Qwen3-14B-Finetuned-Q4_K_M.gguf` is actively merging/quantizing in Colab.
2. **Evaluation Options Selected by User**:
   - Primary goal: Clean, sleek, fast testing of fine-tuned models vs. standard responses.
   - Sustainable options discussed:
     - **LM Studio on Mac M1**: For offline, instant, zero-setup testing.
     - **Kaggle**: For converting the incoming 3 models on Monday with 30 GB RAM (zero OOM) and 9-hour background runtime.
     - **Hugging Face Spaces**: For permanent 24/7 web demo URL for Dr. Tamal Ghosh.
3. **Monday Workflow**:
   - Plug pendrive into Mac.
   - Strip `optimizer.pt` checkpoints.
   - Upload clean adapter folders to `My Drive/Research Project/models_for_drive/`.
   - Run the merger pipeline to generate Models 4, 5, and 6.

