#!/usr/bin/env bash
# Convert a HuggingFace model folder (full weights or LoRA adapter) to a quantized GGUF.
#
# Usage:
#   ./convert_to_gguf.sh <model_dir> [quant] [out_dir]
#
#   <model_dir>  folder with config.json + *.safetensors (or adapter_config.json for LoRA)
#   [quant]      Q4_K_M (default) | Q5_K_M | Q8_0 | F16 | Q6_K | Q3_K_M ...
#   [out_dir]    where the .gguf lands (default: next to the model folder)
#
# Examples:
#   ./convert_to_gguf.sh ~/models/qwen2.5-7b-sft
#   ./convert_to_gguf.sh ~/models/my-lora-adapter Q5_K_M ~/gguf --base Qwen/Qwen2.5-7B-Instruct

set -euo pipefail

MODEL_DIR="${1:?usage: $0 <model_dir> [quant] [out_dir] [--base HF_REPO]}"
QUANT="${2:-Q4_K_M}"
OUT_DIR="${3:-$(dirname "$MODEL_DIR")}"
BASE_REPO=""

# allow --base anywhere
ARGS=("$@")
for i in "${!ARGS[@]}"; do
  if [[ "${ARGS[$i]}" == "--base" ]]; then BASE_REPO="${ARGS[$((i+1))]:-}"; fi
done

MODEL_DIR="${MODEL_DIR%/}"
OUT_DIR="${OUT_DIR%/}"
NAME="$(basename "$MODEL_DIR")"
mkdir -p "$OUT_DIR"

LLAMA_DIR="${LLAMA_CPP_DIR:-$HOME/tools/llama.cpp}"

# ---------- 1. toolchain ----------
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required (for llama-quantize). Install it first." >&2
  exit 1
fi
if ! command -v llama-quantize >/dev/null 2>&1; then
  echo ">>> installing llama.cpp via brew (provides llama-quantize)"
  brew install llama.cpp
fi
if [[ ! -f "$LLAMA_DIR/convert_hf_to_gguf.py" ]]; then
  echo ">>> cloning llama.cpp into $LLAMA_DIR (for convert_hf_to_gguf.py)"
  mkdir -p "$(dirname "$LLAMA_DIR")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

# python env with torch/transformers/gguf
VENV="${VENV_DIR:-$HOME/tools/gguf-venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo ">>> creating python venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch transformers safetensors sentencepiece protobuf gguf numpy
fi
PY="$VENV/bin/python"

# ---------- 2. LoRA merge (only if it is an adapter) ----------
MERGE_SRC="$MODEL_DIR"
if [[ -f "$MODEL_DIR/adapter_config.json" ]]; then
  if [[ -z "$BASE_REPO" ]]; then
    echo "ERROR: $MODEL_DIR is a LoRA adapter. Re-run with --base <hf-repo-id> to merge it into the base model." >&2
    exit 1
  fi
  MERGED="$OUT_DIR/${NAME}-merged"
  echo ">>> merging adapter into $BASE_REPO -> $MERGED"
  "$PY" - "$BASE_REPO" "$MODEL_DIR" "$MERGED" <<'PYEOF'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base_id, adapter, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
PYEOF
  MERGE_SRC="$MERGED"
fi

# ---------- 3. HF -> GGUF (F16) ----------
F16="$OUT_DIR/${NAME}-f16.gguf"
if [[ ! -f "$F16" ]]; then
  echo ">>> converting to GGUF (F16): $F16"
  "$PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGE_SRC" --outfile "$F16" --outtype f16
fi

# ---------- 4. quantize ----------
if [[ "$QUANT" == "F16" ]]; then
  FINAL="$F16"
else
  FINAL="$OUT_DIR/${NAME}-${QUANT}.gguf"
  echo ">>> quantizing to $QUANT: $FINAL"
  llama-quantize "$F16" "$FINAL" "$QUANT"
fi

echo
echo "DONE: $FINAL"
ls -lh "$FINAL"
