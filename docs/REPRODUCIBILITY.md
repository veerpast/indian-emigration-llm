# Reproducibility protocol

## Artifact layers

The project deliberately separates four layers:

1. **Local corpus** — private until provenance and redistribution review is
   complete.
2. **LoRA adapters** — stored locally/Drive; weight binaries are excluded from
   Git.
3. **Canonical Colab notebook** — demo-only inference over the three existing
   Drive GGUF files; it does not merge adapters or convert/quantize models.
4. **Presentation site** — explains the verified pipeline and links to the
   runnable notebook/repository; it does not pretend to host 18+ GB of models.

## Environment capture

For every final experiment, record:

```text
git commit
dataset SHA-256
train/validation/test split hashes
base model repository and revision
adapter configuration and run configuration
Python, PyTorch, Transformers, PEFT, TRL, and CUDA versions
GPU/HPC allocation and random seed
llama.cpp commit or release
GGUF SHA-256 and byte size
generation parameters
```

## Canonical execution

1. Open `notebooks/Final_LLM_Pipeline.ipynb` in Colab and choose a free T4 GPU
   if one is available; CPU fallback is supported but slow.
2. Run the Drive verification cell and confirm the three expected GGUFs are
   present in `My Drive/Research Project/gguf/`.
3. Run the inference-runtime setup cell, then the Gradio/`llama-server` demo.
4. Record the Colab backend and generation settings for each demonstration.
5. For a reproducible model-conversion experiment, use the separate conversion
   procedure and record its tool versions, inputs, and output hashes.

## Split discipline

The local CSV currently has no checked-in split manifest. Before a paper claim:

- remove exact and near-duplicates before splitting;
- group rows that originate from the same thread/document;
- stratify across the seven post-processing categories;
- freeze the test manifest before hyperparameter selection;
- publish only row hashes if the raw text cannot be redistributed.

## Paper claim boundary

Claims supported today:

- three named adapters/configurations exist locally;
- three named Q4_K_M artifacts exist on Drive with verified byte sizes;
- the notebook verifies the existing Drive artifacts and streams inference;
- the local corpus has 30,324 structurally valid rows across seven labels.

Claims requiring additional evidence:

- comparative quality rankings;
- improvement over base models;
- legal/regulatory correctness;
- generalization beyond the corpus;
- memory or speed numbers not captured from a controlled benchmark run.

