# Fine-Tuning Open-Source LLMs for Indian Emigration Q&A

[![Project site](https://img.shields.io/badge/live-project_site-2f6bff)](https://indian-emigration-llm-research.silkenbay32.chatgpt.site)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/veerpast/indian-emigration-llm/blob/main/notebooks/Final_LLM_Pipeline.ipynb)
[![Dataset audit](https://github.com/veerpast/indian-emigration-llm/actions/workflows/quality.yml/badge.svg)](https://github.com/veerpast/indian-emigration-llm/actions/workflows/quality.yml)

An academic pipeline for adapting open-source instruction models to Indian
emigration, visa interview, and regulatory question answering with LoRA;
merging and quantizing them to GGUF `Q4_K_M`; and serving them through
`llama.cpp` with a streaming Gradio interface.

This is a final-year B.Tech CSE (AI & ML) project at Adamas University by
**Veerpratap Singh Tomar** and **Piyush Pratap Singh**, coordinated by
**Dr. Tamal Ghosh**.

> **Research status:** three of six planned model families are fully represented
> in the current repository/Drive artifacts. The project does not yet publish a
> complete comparative benchmark, so it makes no unsupported “best model” claim.
> Model outputs are not legal or immigration advice.

## What is verified

- A local **30,324-row Reddit-derived** Q&A corpus across seven post-processing
  category labels.
- Three LoRA adapter bundles with exact base-model and adapter configuration.
- Three final GGUF `Q4_K_M` artifacts verified on Google Drive on
  **25 November 2026**.
- A canonical Colab notebook for low-memory shard-wise LoRA merge,
  Hugging Face-to-GGUF conversion, quantization, and live streaming inference.
- A presentation site and a reproducible structural dataset audit.

## System at a glance

```mermaid
flowchart LR
    A[Authorized sources] --> B[Clean and de-identify]
    B --> C[30,324 QA pairs]
    C --> D[College HPC: LoRA SFT]
    D --> E[PEFT adapters]
    E --> F[Colab: shard-wise merge]
    F --> G[FP16 GGUF]
    G --> H[Q4_K_M quantization]
    H --> I[llama-server SSE]
    I --> J[Gradio evaluation UI]
```

The low-memory merger applies the LoRA update to one base-model shard at a
time:

$$W_{merged} = W_{base} + \frac{\alpha}{r}(B A)$$

It writes each fused shard before releasing it, avoiding the peak memory cost
of holding the entire base model, adapter, and merged model simultaneously.

## Model registry

| Fine-tuned model | Base model | LoRA configuration | Recorded training configuration | Final GGUF |
|---|---|---|---|---:|
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` | r=64, α=128, dropout=0.05 | lr=1e-4, 3 epochs (project log; run file absent) | 4.37 GB |
| Granite 3.3 8B | `ibm-granite/granite-3.3-8b-instruct` | r=64, α=128, dropout=0.05 | lr=1e-4, 3 epochs, effective batch 32, max length 1024 | 4.94 GB |
| Qwen3 14B | `Qwen/Qwen3-14B` | r=16, α=32, dropout=0.05 | lr=2e-4, 1 epoch, effective batch 8, max length 1024 | 9.00 GB |

All three adapters target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`. See [the model card](docs/MODEL_CARD.md) for intended
use, limitations, and the evidence boundary.

The three GGUF files are stored in the project’s
[Google Drive artifact folder](https://drive.google.com/drive/folders/1kypaQhJfw3-65aerFxr4GMpn0wzMKWtG).
Large model weights are intentionally excluded from Git history.

## Dataset

The local CSV has exactly three fields:

```text
question,answer,source
```

| Post-processing category label | Rows | Share |
|---|---:|---:|
| Indian Student Visa Consular Interview Experiences | 17,934 | 59.14% |
| Complete Consular Visa Interview Debriefs | 7,813 | 25.77% |
| Immigration Legal & Regulatory QA (USCIS/DOS) | 2,894 | 9.54% |
| Reddit Travel & Relocation Community Q&A | 742 | 2.45% |
| Study Abroad, Accommodation & Visa Advisory Corpus | 693 | 2.29% |
| F-1 Student Visa Interview Records | 164 | 0.54% |
| Consular Visa Interview Guidelines | 84 | 0.28% |

All 30,324 rows originated from Reddit and were subsequently scraped, cleaned,
pre-processed, and assigned one of the seven labels above. The label named
`Reddit Travel & Relocation Community Q&A` is one thematic bucket; it does not
mean that the other six buckets came from non-Reddit sources. The current CSV
retains the processed category in a column named `source`, so this README uses
“category label” to avoid confusing that field with acquisition provenance.

The full CSV remains local and is ignored by Git until the row-level provenance,
rightsholder permissions, privacy review, and redistribution terms are complete.
That choice protects the research and the people whose content may appear in
source material. See the [dataset card](docs/DATA_CARD.md).

Run the local structural audit:

```bash
python3 scripts/validate_dataset.py indian_emigration_qa_purified.csv
python3 scripts/validate_dataset.py indian_emigration_qa_purified.csv --json
```

The audit checks schema, empty fields, duplicates, source counts, and possible
contact-detail patterns for manual review. It does not certify factual accuracy
or permission to use the text.

## Collecting Reddit data responsibly

The repository includes an example collector for reproducibility, but it is
guarded because the policy environment matters as much as the code.

As of 25 November 2026:

- Reddit’s [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy)
  says API access requires explicit approval.
- Reddit’s [Data API Terms](https://redditinc.com/policies/data-api-terms)
  restrict using user content to train an AI model without express permission
  from the applicable rightsholders.
- Reddit’s [Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki)
  specifies OAuth and an identifying user-agent format.

Therefore, “publicly visible” does not mean “free to train on.” Before collection:

1. Define the research purpose, communities, fields, retention period, and
   deletion process.
2. Request and receive Reddit API approval for that purpose.
3. Obtain the permissions required for model-training use of user content.
4. Register an appropriate Reddit application and store credentials outside
   source control.
5. Collect only necessary fields through OAuth; do not bypass limits or scrape
   HTML as a workaround.
6. Remove deleted/removed content, direct identifiers, and irrelevant text.
7. Keep stable item IDs in a private provenance ledger so deletion requests can
   be honored; do not publish usernames.
8. Review the resulting dataset and license with the project supervisor before
   training or release.

### Authorized collector setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill the three values in `.env`, then run only after approvals are in place:

```bash
python3 scripts/reddit_collector.py \
  --subreddits india,Indians_StudyAbroad \
  --query 'student visa interview' \
  --limit 100 \
  --min-score 2 \
  --i-have-approval
```

The collector uses PRAW/OAuth, filters NSFW and deleted/removed material,
selects a high-scoring answer, omits usernames, and records stable item IDs for
private provenance and deletion handling. The approval flag is a safety stop;
it is not proof of permission.

## HPC fine-tuning

Training occurred on the college HPC environment using supervised fine-tuning
with PEFT/LoRA. The durable experimental record consists of:

- `adapter_config.json` for architecture/base-model provenance;
- `run_config.json` for Granite and Qwen optimizer/training settings;
- the resulting `adapter_model.safetensors` files (kept outside Git);
- tokenizers and chat templates;
- the project log and canonical conversion notebook.

A paper-grade rerun should additionally pin the base model revision, dataset
hash/split manifest, package lock, CUDA/PyTorch versions, HPC allocation,
training command, wall-clock time, and evaluation outputs. The complete protocol
is in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Colab conversion and quantization

Open the canonical notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/veerpast/indian-emigration-llm/blob/main/notebooks/Final_LLM_Pipeline.ipynb)

Expected Drive layout:

```text
My Drive/Research Project/
├── models_for_drive/
│   ├── mistral7b/
│   ├── granite33_8b/
│   └── qwen3_14b/
└── gguf/
    ├── Mistral-7B-Finetuned-Q4_K_M.gguf
    ├── Granite-3.3-8B-Finetuned-Q4_K_M.gguf
    └── Qwen3-14B-Finetuned-Q4_K_M.gguf
```

Then:

1. Choose a Colab runtime with sufficient system RAM and disk.
2. Run the hardware audit and mount Drive.
3. Confirm each adapter and base repository in the central model registry.
4. Install the pinned toolchain cell.
5. Run the shard-wise merge and batch conversion cell. Existing outputs are
   skipped after a size check.
6. Optionally copy one GGUF to `/content` for faster reads.
7. Launch the streaming Gradio demo.

The notebook creates an intermediate FP16 GGUF and quantizes it with
`llama-quantize Q4_K_M`. It dynamically allocates a free local port for
`llama-server`, avoiding Colab’s internal port collisions.

## Streaming inference

`scripts/launch_colab_demo.py` discovers GGUF files in common Drive locations,
loads one model at a time with `llama-server`, waits on `/health`, and streams
the OpenAI-compatible `/v1/chat/completions` response into Gradio.

Recommended demonstration settings:

```text
temperature: 0.2–0.3
top_p:       0.9
context:     2048 tokens in the lightweight demo
max_tokens:  192 (adjust as needed)
```

Do not add an arbitrary system prompt if it was absent from the fine-tuning
distribution. Match each model’s checked-in chat template.

## Live demo

The project site is designed for the final presentation. It explains the
pipeline, provides an interactive (clearly labelled synthetic) Q&A walkthrough,
and exposes the verified artifact/model facts. The actual model runtime stays
in Colab because the three GGUF files total more than 18 GB.

- **Presentation deployment:**
  [indian-emigration-llm-research.silkenbay32.chatgpt.site](https://indian-emigration-llm-research.silkenbay32.chatgpt.site)
- **GitHub Pages mirror:**
  [veerpast.github.io/indian-emigration-llm](https://veerpast.github.io/indian-emigration-llm/)

For the real model demo, run the Colab notebook and open the generated temporary
Gradio URL. That link exists only while the Colab runtime is connected.

## Evaluation plan for the paper

A defensible comparison should use the same frozen, leakage-controlled test set
for every base and fine-tuned model.

| Dimension | Suggested measurement |
|---|---|
| Correctness | blinded 1–5 expert rating against current official sources |
| Relevance | blinded 1–5 rating |
| Completeness | required-fact checklist per question |
| Safety | unsupported legal claim and harmful-guidance rate |
| Similarity | BERTScore or embedding similarity as a supporting metric only |
| Efficiency | TTFT, tokens/s, peak RAM, model bytes |
| Reliability | mean, median, bootstrap 95% CI, and category-wise results |
| Agreement | Cohen’s κ or Krippendorff’s α for multiple raters |

Always compare against the untouched base model and include qualitative error
analysis. Avoid treating lexical overlap as factual correctness.

## Repository structure

```text
.
├── dist/                         # presentation website
├── docs/
│   ├── DATA_CARD.md
│   ├── MODEL_CARD.md
│   ├── REPRODUCIBILITY.md
│   └── research/                 # historical engineering notes
├── models_for_drive/             # configs/tokenizers; weights ignored by Git
├── notebooks/
│   ├── Final_LLM_Pipeline.ipynb # canonical Colab workflow
│   └── archive/                  # earlier working notebooks
├── scripts/
│   ├── convert_to_gguf.sh
│   ├── launch_colab_demo.py
│   ├── reddit_collector.py
│   └── validate_dataset.py
├── tests/
├── CITATION.cff
├── LICENSE
└── README.md
```

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
python3 scripts/validate_dataset.py indian_emigration_qa_purified.csv
make serve
```

Then open `http://localhost:8000`.

## Research and safety limitations

- Immigration rules change; model weights are static.
- The category distribution is highly imbalanced toward student visa interviews.
- Some answers are anecdotal and may be wrong even when reproduced faithfully.
- LoRA adaptation and Q4 quantization do not prevent hallucinations.
- The full corpus requires a stronger provenance/permission record before
  redistribution.
- Current artifacts prove an engineering pipeline, not comparative superiority.

## Authors and acknowledgement

- **Veerpratap Singh Tomar** — B.Tech CSE (AI & ML), Adamas University
- **Piyush Pratap Singh** — B.Tech CSE (AI & ML), Adamas University
- **Dr. Tamal Ghosh** — Project Coordinator

Please cite the project using [CITATION.cff](CITATION.cff). Source code and
original documentation are MIT-licensed. Dataset/source-content rights are
separate; see [LICENSE](LICENSE) and the dataset card.

