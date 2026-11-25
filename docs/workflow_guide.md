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
5. The final cell starts Gradio in the background and prints a temporary
   `gradio.live` URL once ready. The cell then returns. Select a model and send
   a question to load it; only one model is loaded at a time. The first model
   load may take a minute or more when reading from Drive.

The final cell downloads the launcher from a fixed repository commit rather
than embedding a second copy of the interface code in the notebook. It keeps
startup logs in `/content/launch_colab_demo.log`; rerunning the cell checks the
same process instead of starting a duplicate server.

## Free-runtime notes

Colab's free GPU availability and session duration vary and are not guaranteed.
Keep the notebook open and use the demo during the live presentation; the
`gradio.live` URL is temporary, not a permanent public service. If no GPU is
available, choose Mistral 7B for the most practical CPU fallback. Model outputs
are research demonstrations, not legal or immigration advice.
Anyone with the temporary link may be able to access the running demo. Do not
enter personal or confidential information, and stop the runtime after use.

## Troubleshooting

- **GGUF file not found:** confirm the exact capitalization and names in
  `My Drive/Research Project/gguf/`, then rerun the first code cell.
- **No GPU assigned:** use the CPU fallback for a short Mistral demo, or retry
  later when Colab offers a free GPU. Qwen3 14B is hidden automatically on CPU.
- **`gradio.live` link unavailable:** check that the final cell completed and
  keep the Colab runtime connected. The URL expires when the runtime stops.
- **No link appears after about a minute:** the final cell returns with the
  latest launcher log. Rerun it to check startup again; it reuses the running
  process rather than starting a duplicate.
- **Model load fails:** inspect the final cell output for the `llama-server`
  log and verify that the assigned runtime has enough RAM/VRAM.

## Presentation run-through

Before presenting, open the notebook, confirm Drive access and the exact GGUF
filenames, run all three cells, and test the temporary link. Keep the notebook
tab open; the free GPU allocation and sharing link are not guaranteed.

Suggested short introduction:

> This research project studies domain adaptation for Indian emigration
> question answering. We cleaned and pre-processed a Reddit-derived corpus,
> fine-tuned three open instruction models with LoRA on college HPC, and
> prepared quantized GGUF artifacts for lower-resource inference. This Colab
> demo loads those existing artifacts from Drive. The current evidence confirms
> the pipeline and artifacts, not which model is best; a controlled evaluation
> is still required.

For a fair live comparison, ask each selected model the same question, such as
“What is the difference between ECR and ECNR status?”, and state that answers
are generated responses rather than verified legal advice. If Colab does not
provide a usable runtime, present the website's clearly labelled walkthrough
and explain that it is illustrative, not live model inference. Do not enter
personal or confidential information into the public temporary link.
