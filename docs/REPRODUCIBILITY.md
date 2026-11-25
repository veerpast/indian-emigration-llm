# Reproducibility protocol

## Artifact layers

The project deliberately separates four layers:

1. **Local corpus** — private until provenance and redistribution review is
   complete.
2. **LoRA adapters** — stored locally/Drive; weight binaries are excluded from
   Git.
3. **Canonical Colab notebook** — merges each adapter into its exact base model,
   converts to FP16 GGUF, then quantizes to `Q4_K_M`.
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

1. Upload adapters to `My Drive/Research Project/models_for_drive/`.
2. Open `notebooks/Final_LLM_Pipeline.ipynb` in Colab.
3. Mount Drive and run the environment audit.
4. Run the model registry cell and verify paths/base repositories.
5. Run toolchain setup.
6. Run streamed shard merge and `Q4_K_M` quantization.
7. Verify output byte size and compute SHA-256.
8. Run the Gradio/`llama-server` interface.
9. Export machine-readable benchmark results before interpreting them.

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
- the notebook implements low-memory shard-wise merging and streaming inference;
- the local corpus has 30,324 structurally valid rows across seven labels.

Claims requiring additional evidence:

- comparative quality rankings;
- improvement over base models;
- legal/regulatory correctness;
- generalization beyond the corpus;
- memory or speed numbers not captured from a controlled benchmark run.

