# 🧪 Convert & Chat — Step-by-Step Workflow

> **Who is this for?** Your research partner on the Windows laptop (1 TB storage, 16 GB RAM).  
> **What do they need?** A browser + internet. All heavy computation runs on Google's Colab servers.  
> **Time estimate:** ~30 min setup + ~15 min per model conversion + chatting.

---

## Phase 1: Sync Models to Google Drive (~20-60 min depending on internet)

### Step 1 — Install Google Drive for Desktop

1. Open a browser and go to **https://www.google.com/drive/download/**
2. Click **"Download Drive for Desktop"**
3. Run the installer (`GoogleDriveSetup.exe`)
4. Sign in with the **same Google account** you'll use for Colab
5. After setup, you'll see a **Google Drive (G:)** drive in File Explorer

### Step 2 — Create the models folder in Drive

1. Open **File Explorer** → navigate to `G:\My Drive\`
2. Create a new folder called `models`
3. Full path should be: `G:\My Drive\models\`

### Step 3 — Copy your 6 model folders into Drive

1. Locate your 6 fine-tuned model folders on the laptop  
   (each folder should contain `config.json`, `tokenizer.*`, and `*.safetensors` or `*.bin` files)
2. Copy all 6 folders into `G:\My Drive\models\`

   Your Drive should now look like:
   ```
   G:\My Drive\models\
   ├── finetune-1\
   │   ├── config.json
   │   ├── tokenizer.json
   │   ├── tokenizer_config.json
   │   ├── model.safetensors  (or model-00001-of-00002.safetensors, etc.)
   │   └── ...
   ├── finetune-2\
   │   └── ...
   ├── finetune-3\
   │   └── ...
   ├── finetune-4\
   │   └── ...
   ├── finetune-5\
   │   └── ...
   └── finetune-6\
       └── ...
   ```

### Step 4 — Wait for sync to complete

1. Look at the **Google Drive icon** in the Windows taskbar (bottom-right, near the clock)
2. Click it → it shows upload progress
3. **Wait until it says "Up to date"** or shows a ✓ checkmark
4. This can take a while for ~84 GB (6 × ~14 GB). On a 50 Mbps upload: ~4 hours.

> **💡 Faster option:** Instead of copying the raw folders, you can **zip each model folder** first:
> 1. Right-click each model folder → **"Compress to ZIP file"** (or use 7-Zip)
> 2. Copy the 6 `.zip` files into `G:\My Drive\models\`
> 3. Zips upload faster because it's one big file instead of thousands of small ones
> 4. The notebook has a cell (Cell 3b) that will unzip them for you in Colab

---

## Phase 2: Open the Notebook in Google Colab (~2 min)

### Step 5 — Upload the notebook to Colab

1. Open a browser → go to **https://colab.research.google.com/**
2. Sign in with the **same Google account** used for Drive
3. Click **File → Upload notebook**
4. Upload the file: **`convert_and_chat_colab.ipynb`**

### Step 6 — Set the runtime to GPU

1. In Colab, click **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **T4 GPU**
3. Click **Save**

---

## Phase 3: Configure (~2 min)

### Step 7 — Run Cell 1 (Sanity check)

1. Click the **▶ play button** on Cell 1
2. It prints RAM, disk space, and GPU info
3. You should see something like:
   ```
   RAM total : 12.7 GB   available: 11.2 GB
   Disk      : 78.2 GB free of 107.7 GB
   GPU       : Tesla T4, 15360 MiB
   ```

### Step 8 — Run Cell 2 (Mount Google Drive)

1. Click **▶** on Cell 2
2. A popup asks you to **allow Colab to access Google Drive** — click **Allow**
3. It prints: `Mounted at /content/drive`

### Step 9 — Edit Cell 3 (⚠️ This is the only cell you edit!)

1. **Replace the placeholder paths** with your actual model folder names:

   ```python
   MODELS = [
       '/content/drive/MyDrive/models/YOUR-MODEL-1-NAME',
       '/content/drive/MyDrive/models/YOUR-MODEL-2-NAME',
       '/content/drive/MyDrive/models/YOUR-MODEL-3-NAME',
       '/content/drive/MyDrive/models/YOUR-MODEL-4-NAME',
       '/content/drive/MyDrive/models/YOUR-MODEL-5-NAME',
       '/content/drive/MyDrive/models/YOUR-MODEL-6-NAME',
   ]
   ```

   > **How to find the exact folder names:** In Colab, run this in a new cell:
   > ```python
   > !ls '/content/drive/MyDrive/models/'
   > ```
   > It will list exactly what folders are in your Drive.

2. **Set the chat template** to match your fine-tune format:
   - If your models use **ChatML**: `'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'`
   - If your models use **Llama-2/3 style**: `'[INST] {prompt} [/INST]'`
   - If your models use **Alpaca**: `'### Instruction:\n{prompt}\n\n### Response:\n'`
   - If unsure, leave it as `'{prompt}'` and adjust later if chat output looks wrong

3. **Run Cell 3** → it should show ✅ next to each model that it found

### Step 10 — Run Cell 3b (only if you uploaded zips)

- If you uploaded `.zip` files instead of raw folders, run this cell to extract them
- If you synced the folders directly, **skip this cell**

---

## Phase 4: Convert All Models (~15 min per model)

### Step 11 — Run Cell 4 (Install dependencies)

1. Click **▶** on Cell 4
2. This takes ~2-3 minutes (downloads llama.cpp, installs Python packages)
3. Wait until you see: `torch X.X cuda True | transformers X.X | gguf ok`

### Step 12 — Run Cell 5 (Batch conversion)

1. Click **▶** on Cell 5
2. **This is the long step** — it converts all 6 models one by one
3. For each model, you'll see progress:
   ```
   ============================================================
   [1/6] Converting: finetune-1
   ============================================================
     copying to local disk (faster than reading from Drive)...
     weights: 14.21 GB in 2 file(s)
     $ python3 /content/llama.cpp/convert_hf_to_gguf.py ...
     intermediate: 7.68 GB
     quantized: 4.37 GB
     saving to Drive...
     ✅ done in 12.3 min → finetune-1-Q4_K_M.gguf (4.37 GB)
   ```
4. At the end, you'll see a summary:
   ```
   SUMMARY
   ✅ Converted: 6/6
      finetune-1: 4.37 GB
      finetune-2: 4.41 GB
      ...
   ```

> **⚠️ If a model fails or gets "Killed":**
> - The Colab free tier has ~13 GB RAM — enough for 7B models
> - If it fails, comment out completed models in `MODELS` and re-run Cell 5 (it skips already-converted ones)
> - Or try: `Runtime > Change runtime type` → **High-RAM** (if you have Colab Pro)

---

## Phase 5: Chat with Your Models! 🎉

### Step 13 — Run Cell 6 (Launch Gradio Chat UI)

1. Click **▶** on Cell 6
2. Wait ~30 seconds for Gradio to start
3. You'll see two links:
   ```
   Running on local URL: http://127.0.0.1:7860
   Running on public URL: https://xxxxx.gradio.live   ← USE THIS ONE
   ```
4. **Click the public URL** (the `gradio.live` one)

### Step 14 — Chat!

1. In the Gradio UI:
   - **🤖 Model dropdown** (left side) — pick which of the 6 models to chat with
   - **Max tokens** — how long the response can be (256 is a good default)
   - **Temperature** — 0.1-0.3 = focused/factual, 0.7-1.0 = creative
   - **Top-p** — leave at 0.9 unless you know what you're doing
2. Type a message in the chat box and hit **Submit**
3. **First message takes ~30-60 seconds** (model needs to load into memory)
4. Subsequent messages are faster (~5-15 seconds)
5. To **switch models**: pick a different one from the dropdown and send a new message
6. To **clear chat history**: click the 🗑️ button

---

## Phase 6: When You're Done

### Step 15 — Save your GGUFs

- All converted `.gguf` files are already saved on Google Drive at:
  `My Drive/gguf/`
- You can download them from Drive to the laptop anytime

### Step 16 — Disconnect Colab

1. Go back to the Colab tab
2. Click **Runtime → Disconnect and delete runtime**
3. This frees up the GPU for other users (and stops burning your free quota)

---

## ❓ Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Cell 3 shows ❌ NOT FOUND | Check folder names. Run `!ls '/content/drive/MyDrive/models/'` to see what's there. |
| Conversion killed / OOM | Free tier = ~13 GB RAM. Convert 2-3 models at a time. |
| Chat output is gibberish | Wrong `CHAT_TEMPLATE`. Check what format the models were fine-tuned with. |
| Gradio link expired | Re-run Cell 6. Public links last 72 hours. |
| Drive sync stuck | Check internet. Restart Google Drive from the taskbar icon. |
| "Unknown model architecture" | Run `!cd /content/llama.cpp && git pull` then re-run Cell 4. |

---

## 📐 Quick Reference: GGUF Quant Sizes (7B model)

| Quant | File Size | Quality |
|-------|-----------|---------|
| Q3_K_M | ~3.3 GB | Acceptable |
| **Q4_K_M** | **~4.5 GB** | **Good (default)** |
| Q5_K_M | ~5.1 GB | Very good |
| Q6_K | ~5.9 GB | Excellent |
| Q8_0 | ~7.7 GB | Near-lossless |
