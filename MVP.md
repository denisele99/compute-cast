# GPT-2 MVP: inputs, assumptions and changes

Source of truth for fields: `backend/data/mvp_feature_contract.json` (served by `GET /api/schema`).
Estimator: `backend/app/services/estimators/estimate_gpt2.py` (formulas in its `ESTIMATOR.md`).

## Scope

Standard GPT-2 causal decoder, full training of all parameters, FP32 parameters/gradients/AdamW states,
BF16 autocast, FP32 math attention, one GPU, fixed-length sequences. Analytical and **uncalibrated**:
the coefficients are illustrative heuristics, not measurements or confidence bounds.

## 1. Direct inputs and file extractions (by form section)

`*` = required. Editable: **yes** · **custom only** (locked when it comes from a checkpoint; editable after
turning on *Hypothetical custom configuration*, which labels results hypothetical) · **no**.

### Model
| Field (contract feature) | Config key | Required | Editable | Notes |
| --- | --- | --- | --- | --- |
| model_type | `model_type` | * | no | must be `gpt2`; anything else is an unsupported conflict |
| num_layers | `n_layer` | * memory | custom only | |
| hidden_size | `n_embd` | * memory | custom only | |
| num_attention_heads | `n_head` | * memory | custom only | must divide hidden_size |
| intermediate_size | `n_inner` | only if non-default | custom only | null → 4 × hidden_size |
| vocab_size | `vocab_size` | * memory | custom only | |
| max_position_embeddings | `n_positions` | * memory | custom only | sequence_length must not exceed it |
| attention / embedding / residual dropout | `attn_pdrop` / `embd_pdrop` / `resid_pdrop` | * memory | custom only (contract: fixed) | missing in a config → GPT2Config default 0.1, flagged uncertain |

### Training
| Field | Source keys | Required | Editable | Notes |
| --- | --- | --- | --- | --- |
| microbatch_size | `per_device_train_batch_size`, `microbatch_size`, DataLoader `batch_size` | * memory | yes | examples per GPU per forward/backward |
| gradient_accumulation_steps | same name | * memory | yes | prefilled 1 |
| duration mode | `max_steps` > 0 → updates, else epochs | — | yes | choose ONE |
| optimizer_updates | `max_steps`, `optimizer_updates` | * runtime (updates mode) | yes | |
| num_epochs | `num_train_epochs`, `num_epochs` | * runtime (epochs mode) | yes | whole epochs only |

### Data
| Field | Source keys | Required | Editable | Notes |
| --- | --- | --- | --- | --- |
| num_training_examples | `num_training_examples`, `num_examples`, dataset_info `splits.train.num_examples` | * runtime (epochs mode) | yes | post-processed sequences, NOT raw rows or the uploaded sample |
| sequence_length | `max_length`/`max_seq_length` with `padding="max_length"`, `sequence_length` | * memory | yes | a truncation limit without fixed padding is flagged uncertain |

### Hardware (not uploaded: Nebius catalogue, general catalogue or custom)
| Field | Source | Required | Editable |
| --- | --- | --- | --- |
| gpu_id / GPU name | catalogue selection or custom | — | yes |
| VRAM per GPU | catalogue | * memory | custom GPU only |
| Usable VRAM limit | default = 100% of VRAM | * memory | yes |
| BF16 support, peak BF16 dense, peak FP32, memory bandwidth | general catalogue (`hardware_catalog.json`) | * runtime | custom GPU only |
| Price, price unit | `nebius_spot_prices.json` (preemptible, per GPU-hour) or manual | * cost | yes |

## Uploads

One drop zone at the start accepts several files or a ZIP (flattened; ≤50 members, ≤30 MB expanded,
nested ZIPs and unknown types skipped visibly). Each file is classified by content:

| Section | Recognised files | What is read |
| --- | --- | --- |
| Model | Hugging Face `config.json`, model metadata JSON/YAML, `model.py` (GPT2Config(...) only) | GPT-2 dimensions, dropouts, unsupported flags |
| Training | exported `TrainingArguments` JSON/YAML, own training JSON/YAML, DeepSpeed JSON, `train.py` | batch, accumulation, duration; conflicts (fp16, checkpointing, LoRA, optimizers, compile, multi-GPU) |
| Data | preprocessing / collator / tokenizer config, `dataset_info.json`, optional JSONL/CSV sample | fixed sequence length, example count; sample stats (behind a toggle) |

Uploading everything at once is preferred (contract `upload_guidance.preferred_flow`); a file can feed
several sections. Python files are parsed with `ast` only and their values are always marked uncertain;
JSON/YAML use `json` / `yaml.safe_load`. Not needed: weights, a dataset, learning rate, weight decay, seed.
Model source is exclusive: a Hugging Face model **or** uploaded model files; only one Hugging Face model at a time.

## 2. Visible fixed assumptions (locked in v1)

Shown in the collapsible *Execution assumptions* panel as one-option dropdowns: standard GPT-2 template,
full causal MHA, `gelu_new`, learned absolute positions, GPT-2 LayerNorm, tied embeddings, full training,
BF16 autocast, FP32 parameters/gradients/AdamW states, unfused AdamW (no foreach/amsgrad), no extra master
weights, math attention with FP32 intermediates, no checkpointing/compile/offload/KV cache, full-position
FP32 cross-entropy, fixed padding, no packing, drop_last=false, accumulation flushed at epoch end, 1 GPU,
no distribution, and the estimate excludes data pipeline, evaluation/saving and startup time.

## 3. Internal calculations (never user input)

Parameter count, head dim, effective batch, microbatches/updates per epoch, processed positions, state,
cast-cache, activation, loss and scratch bytes, phase peaks, FLOPs, traffic and time — see `ESTIMATOR.md`.

## Estimator assumptions (illustrative, uncalibrated)

BF16 and FP32 efficiency 0.3, bandwidth efficiency 0.5, 10 µs kernel launch overhead, 40 other kernels per
layer, 8 loss kernels, AdamW 20 FLOPs and 40 bytes per parameter with 12 kernels per tensor, 14 loss bytes
per logit, 2 optimizer scratch copies, 0.5 GiB runtime baseline, 15% capacity margin.

**Fit rule:** `(allocated peak + 0.5 GiB) × 1.15 ≤ usable VRAM limit`, with the limit defaulting to 100%
of VRAM (the margin is already in the estimate). **Cost:** `training compute hours × 1 GPU × price per
GPU-hour`. **Benchmark cost:** 3 warm-up + 10 measured updates (the benchmark defaults) of compute time; provisioning excluded; an estimate, not a bill.

## Changes in this version

- New active path: `gpt2_inputs.py` (schema/validation from the contract), `gpt2_extract.py` (files, ZIP,
  Hugging Face, conflicts, examples), `estimate_gpt2.py` (prototype formulas unchanged, split into memory
  and runtime so memory works without GPU specs), `gpt2_predict.py` (fit, cost, comparison).
- Hugging Face restricted to two GPT-2 checkpoints; config.json fetched live with a bundled snapshot
  fallback (`backend/data/hf_config_snapshots.json`) when the Hub is unreachable.
- Example sets in `examples/gpt2_mvp/` (supported, and one with conflicts).
- UI: compact fields; errors/notes in hover/click popovers; one review notice instead of per-field
  buttons; sample statistics behind a toggle; removable files and Hugging Face model; Execution
  assumptions panel; results rewired to the GPT-2 estimate.
- Benchmark mode: the generated `run_benchmark.py` was replaced by the trusted `benchmark/` runner plus a validated `benchmark_config.json` (see BENCHMARK.md).
- Legacy (kept, marked "LEGACY - NOT USED BY THE GPT-2 MVP", still tested in `tests/test_legacy.py`):
  MLP/classic-transformer profiles, `gpu_memory.py`, `gpu_runtime.py`, `estimator.py`,
  `normalize_inputs.py`, `collect_workload.py`, legacy mappers in `static_extract.py`.

## Known gaps

- Coefficients are uncalibrated; validate against benchmarks before trusting absolute numbers.
- GPU specs for H100/H200/L40S are `pending_review`; B200, B300, GB300 and RTX PRO 6000 have no specs, so
  only memory is estimated for them.
- Nebius execution is not implemented; analysis sessions are in memory (single worker, lost on restart).
