# Run the three-model demo in Google Colab

This guide uses the three fine-tuned `Q4_K_M` GGUF files already stored in
Google Drive. The final notebook is inference-only: it does not download the
base models, merge adapters, or convert/quantize weights.

## Start the demo

1. Open the [final Colab notebook](https://colab.research.google.com/github/veerpast/indian-emigration-llm/blob/main/notebooks/Final_LLM_Pipeline.ipynb).
2. In **Runtime → Change runtime type**, choose **T4 GPU** if Colab offers a
   free GPU. If only CPU is available, the notebook hides Qwen3 14B to avoid
   likely memory exhaustion; Mistral 7B and Granite 8B may still run slowly.
3. Run the three code cells from top to bottom. Approve the Drive access prompt.
   The first cell checks these exact files in `My Drive/Research Project/gguf/`:
   - `Mistral-7B-Finetuned-Q4_K_M.gguf`
   - `Granite-3.3-8B-Finetuned-Q4_K_M.gguf`
   - `Qwen3-14B-Finetuned-Q4_K_M.gguf`
4. The next cell installs the inference runtime in the temporary Colab VM.
   It does not write to Drive or alter model files.
5. The final cell starts the Gradio chat and prints a temporary `gradio.live`
   URL. Select a model and send a question. Only one model is loaded at a time;
   the first response after a model switch takes longer while it loads.

## Free-runtime notes

Colab's free GPU availability and session duration vary and are not guaranteed.
Keep the notebook open and use the demo during the live presentation; the
`gradio.live` URL is temporary, not a permanent public service. If no GPU is
available, choose Mistral 7B for the most practical CPU fallback. Model outputs
are research demonstrations, not legal or immigration advice.

## Troubleshooting

- **GGUF file not found:** confirm the exact capitalization and names in
  `My Drive/Research Project/gguf/`, then rerun the first code cell.
- **No GPU assigned:** use the CPU fallback for a short Mistral demo, or retry
  later when Colab offers a free GPU. Qwen3 14B is hidden automatically on CPU.
- **`gradio.live` link unavailable:** check that the final cell completed and
  keep the Colab runtime connected. The URL expires when the runtime stops.
- **Model load fails:** inspect the final cell output for the `llama-server`
  log and verify that the assigned runtime has enough RAM/VRAM.
