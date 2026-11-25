# 🧠 Project Memory & Session Log

This document preserves the complete context, decisions, file paths, configurations, and future workflow steps.

---

## 🏆 Current Progress & Completed Models

| Model | Status | GGUF File Location on Google Drive | Size |
|---|---|---|---|
| **Mistral-7B-Finetuned** | ✅ **VERIFIED WORKING & STREAMING** | `My Drive/Research Project/gguf/Mistral-7B-Finetuned-Q4_K_M.gguf` | ~4.3 GB |
| **Granite-3.3-8B-Finetuned** | ⏳ Queued for Phase 2 | `My Drive/Research Project/models_for_drive/granite33_8b` | (Ready) |
| **Qwen3-14B-Finetuned** | ⏳ Queued for Phase 2 | `My Drive/Research Project/models_for_drive/qwen3_14b` | (Ready) |
| **Model 4** | ⏳ Arriving on Pendrive | Pending | — |
| **Model 5** | ⏳ Arriving on Pendrive | Pending | — |
| **Model 6** | ⏳ Arriving on Pendrive | Pending | — |

---

## 💻 1. System & Environment Specifications

- **Local Machine:** MacBook Air M1, 8 GB Unified Memory, ~55 GB free SSD storage.
- **External Pendrive:** Mounted at `/Volumes/Untitled` (format: exFAT/FAT32).
- **Desktop Project Directory:** `/Users/veer/Desktop/Research project/`
- **Google Colab Environment:** 
  - Standard/TPU/T4 GPU runtime
  - Output target: `My Drive/Research Project/gguf/`

---

## 📦 2. Model Discovery & Configurations

### Critical Discovery:
The models on the pendrive are **LoRA adapters** (not 85 GB full weights). Training optimizer checkpoints (`optimizer.pt` ~1.3 GB) were stripped to keep upload size minimal (**~1.64 GB total for the first 3**).

### Staged Clean Adapters on Desktop:
Location: `/Users/veer/Desktop/Research project/models_for_drive/`
1. `mistral7b/` (640 MB) — Base: `mistralai/Mistral-7B-Instruct-v0.3` (✅ Converted)
2. `granite33_8b/` (755 MB) — Base: `ibm-granite/granite-3.3-8b-instruct`
3. `qwen3_14b/` (245 MB) — Base: `Qwen/Qwen3-14B`

### Workflow for the Remaining Models:
1. Plug in pendrive containing the 3 new models.
2. Extract clean adapter folders (exclude `optimizer.pt`) directly into:  
   📁 `/Users/veer/Desktop/Research project/models_for_drive/`
3. Upload those 3 new folders to Google Drive:  
   ☁️ `My Drive/Research Project/models_for_drive/`
4. In `convert_and_chat_colab.ipynb` (Cell 3), add their names and base repos to the `MODELS` list.
5. Re-run Cell 5: Existing models (`Mistral-7B-Finetuned`) are skipped automatically, and the remaining models will convert.
6. Run Cell 6: **All 6 models appear together in the Gradio chat demo dropdown!**

---

## 📂 3. Google Drive Structure

```text
My Drive/
└── Research Project/
    ├── models_for_drive/            <-- All adapter folders
    │   ├── mistral7b/
    │   ├── granite33_8b/
    │   ├── qwen3_14b/
    │   ├── [model_4]/
    │   ├── [model_5]/
    │   └── [model_6]/
    └── gguf/                        <-- Converted GGUF outputs (Q4_K_M)
        ├── Mistral-7B-Finetuned-Q4_K_M.gguf  ✅ DONE (~4.3 GB)
        ├── Granite-3.3-8B-Finetuned-Q4_K_M.gguf
        ├── Qwen3-14B-Finetuned-Q4_K_M.gguf
        └── ... (all 6 GGUFs)
```

---

## 🛠️ 4. Key Lessons & Verified Fixes

1. **NumPy 2.x & Scipy Conflict (`cannot import name '_slice'`):**
   - Fixed by pinning: `pip install -q "numpy<2.3" -U scipy` followed by a one-time session restart.
2. **TorchAO Conflict in PEFT (`incompatible version of torchao`):**
   - Fixed by: `pip uninstall -y torchao`. PEFT bypasses the torchao check when uninstalled.
3. **Colab Free RAM OOM Crash (12.7 GB limit on 14.5 GB model):**
   - Resolved by using **TPU v5e-1 with 47.0 GB System RAM**, giving over 30 GB of free headroom during merging.
4. **Intermediate Quantization Format:**
   - Must use `--outtype f16` intermediate from `convert_hf_to_gguf.py` so `llama-quantize` can compute `Q4_K_M` scales without errors.
5. **Shared Library Registration (`libllama-common.so.0`):**
   - Fixed by copying libraries to system path: `cp -a /content/llama-extract/*/* /usr/lib/ && ldconfig`.
6. **Gradio ChatInterface Compatibility:**
   - Removed `type='messages'` keyword argument which caused `TypeError` in Colab's Gradio version.
7. **`llama-cli` Hanging on Generation (`Generation timed out 180s`):**
   - **Root Cause**: Modern `llama-cli` defaults to an interactive chat REPL loop (`-cnv`). When called via Python `subprocess`, it finished generating in ~10 seconds but stayed open waiting for keyboard input, triggering the 180s timeout.
   - **Fix**: Added `--single-turn` (exits immediately after response), `--no-display-prompt` (cleans output), and `-c 2048` (reduces KV cache allocation from 32k to 2k, making CPU initialization instant).
8. **Permanent Solution: `llama-server` Streaming Engine:**
   - Switched from calling `llama-cli` per prompt to running **`llama-server`** in the background.
   - **`--log-disable` Constraint**: `llama-server` does not accept `--log-disable` (which is exclusive to `llama-cli` and caused an instant exit code 1). Removed the flag and routed stdout/stderr to `/tmp/llama_server.log`.
   - **Port 8080 Collision in Colab**: Colab's VM internal services occupy port 8080 (`couldn't bind HTTP server socket`). Replaced hardcoded 8080 with a dynamic OS-assigned free port (`socket.bind(('127.0.0.1', 0))`).
   - **Training Prompt Alignment**: LoRA fine-tuning on `indian_emigration_qa_purified.csv` was trained directly on `[INST] {question} [/INST]` without a generic system prompt. Omitting the generic system prompt avoids domain skew and ensures the model outputs direct answers. Recommended temperature: 0.2–0.3.
   - **Why this eliminates timeouts completely**:
     - The 4.37 GB model is loaded into RAM **once**; no disk reloading lag per turn.
     - Tokens are streamed via Server-Sent Events (`stream: true`) to Gradio.
     - Because words stream out in real time starting from second 1, the connection never stays idle and can never time out!

### Technical Deep Dive: T4 GPU Memory Ceiling vs. Sharded LoRA Merging
- **Problem**: When merging an 8B model (Granite-3.3-8B is 16.3 GB FP16) or 14B model (Qwen3-14B is 29.5 GB FP16) using standard HuggingFace/PEFT `model.merge_and_unload()`, PyTorch must keep both the base weights W and the un-offloaded adapter buffers in memory simultaneously (~32 GB peak memory). On Google Colab T4 GPU (15.0 GB VRAM, 12.7 GB System RAM), this triggers `CUDA out of memory` during `accelerate.hooks.onload_layer`.
- **Solution (Streamed Sharded Merger)**: Instead of instantiating the full causal LM graph in PyTorch, the streamed shard merger loads the base model .safetensors shards one by one (~4 GB each). For each shard, it computes W_merged = W_base + (alpha / r) * (B @ A) only on the adapted projection layers (q, k, v, o, gate, up, down), saves the fused shard to disk, and immediately releases memory via `del` and `gc.collect()`.
- **Resource Footprint**: Peak RAM is capped at ~4.8 GB (well within the 12.7 GB ceiling), requires 0 MB GPU VRAM, and runs in ~2 minutes per model.
