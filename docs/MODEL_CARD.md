# Model card: Indian Emigration Q&A adapters and GGUF models

## Overview

Three open-source instruction models were adapted to the local 30,324-row
Reddit-derived Indian emigration Q&A corpus using parameter-efficient LoRA
fine-tuning on the college HPC system. The data was scraped, cleaned, and
pre-processed before training. The adapters were merged with their base models and
quantized to GGUF `Q4_K_M` in Google Colab for resource-conscious inference via
`llama.cpp`.

## Artifacts verified on Google Drive (25 November 2026)

| Model | Base repository | LoRA | Training recipe | GGUF size |
|---|---|---|---|---:|
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | r=64, α=128, dropout=0.05 | lr=1e-4, 3 epochs (reported in project log) | 4,372,815,680 bytes |
| Granite 3.3 8B Instruct | `ibm-granite/granite-3.3-8b-instruct` | r=64, α=128, dropout=0.05 | lr=1e-4, 3 epochs, effective batch 32, max length 1024 | 4,942,873,440 bytes |
| Qwen3 14B | `Qwen/Qwen3-14B` | r=16, α=32, dropout=0.05 | lr=2e-4, 1 epoch, effective batch 8, max length 1024 | 9,001,753,568 bytes |

The Granite and Qwen recipes are backed by checked-in `run_config.json` files.
The repository does not currently contain an equivalent run file for Mistral;
its optimizer settings are therefore labelled as reported rather than
machine-verifiable.

All adapters target the attention and MLP projection layers: `q_proj`,
`k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

## Intended use

- Academic research and comparative model evaluation.
- Educational demonstrations of LoRA, GGUF conversion, and local inference.
- Drafting candidate answers that are independently verified against current
  official government sources.

## Not intended for

- Legal advice or guaranteed visa outcomes.
- Automated decisions about people, travel, eligibility, or compliance.
- Unsourced claims about current immigration regulations.
- Deployment without abuse testing, monitoring, and a clear escalation path.

## Runtime

The canonical notebook launches `llama-server` and talks to its OpenAI-compatible
`/v1/chat/completions` endpoint using Server-Sent Events. Models are loaded one
at a time. Recommended demonstration sampling is temperature 0.2–0.3 and
top-p 0.9.

## Evaluation status

The repository currently verifies artifact presence, configuration, and the
conversion/inference path. It does **not** yet contain a complete, independently
reproducible benchmark with held-out splits, baselines, confidence intervals,
or human-rating agreement. Do not present the models as outperforming a
baseline until those results are produced.

Recommended minimum evaluation:

- base model vs. fine-tuned model on the same frozen test set;
- exact/semantic answer similarity only as supporting metrics;
- factuality and citation checks against current official sources;
- blinded human ratings for correctness, relevance, completeness, and safety;
- latency, tokens/second, peak RAM, and artifact size;
- category-wise results and error analysis;
- bootstrap confidence intervals and inter-rater agreement.

## Limitations and risks

- Quantization can reduce quality relative to merged FP16 weights.
- The training distribution is dominated by student visa interview examples.
- Regulations change after training.
- Anecdotal source answers may not generalize.
- Fine-tuning does not eliminate hallucination.
- Model outputs may reproduce sensitive or copyrighted source material.

