# Metadata-only analytical prototype

Run with Python, without PyTorch, a GPU, model weights or dataset access:

    python estimate_gpt2.py gpt2_estimator_input.json --output result.json

The estimator takes the four JSON objects model, training, hardware and
estimator_policy. The last two are normally supplied by your application catalog
and versioned policy. Example hardware numbers describe a synthetic GPU, not a
real catalog entry. All efficiency/buffer/overhead coefficients are illustrative,
uncalibrated assumptions. Outputs are not measured results or confidence bounds.

This script is a GPT-2 analytical prototype, separate from model.py/train.py,
which implement a different custom architecture. Do not feed that model_config
directly into this estimator. This prototype does not implement HF downloading.

## Exact parameter accounting for supported GPT-2 template

Let L=layers, H=hidden width, I=MLP width, V=vocabulary, C=position capacity,
A=heads, B=microbatch size, S=executed length, G=accumulation steps.

    block_matrix_elements = 4H^2 + 2HI
    block_parameters = 4H^2 + 2HI + 9H + I
    P = VH + CH + L*block_parameters + 2H

Block bias/normalization term: 4H from attention biases, I+H from MLP biases,
4H from two LayerNorms. Final LayerNorm adds 2H. Shared output embedding adds
no independent parameter storage, but still performs a vocabulary projection.

## Memory envelope (bytes)

    state = 16P
    cast_cache = 2*[L*(4H^2+2HI) + VH]
    saved = L*[BS*(26H+4I+16) + (8+d_attn)*B*A*S^2] + dropout_masks
    loss_buffers = loss_bytes_per_logit * B*S*V
    scratch = max(4BAS^2, 4BSI, 4BSH)
    forward_backward_peak = state + cast_cache + saved + loss_buffers + scratch
    optimizer_peak = state + optimizer_scratch_copies*4*largest_parameter_tensor
    allocated_peak = max(forward_backward_peak, optimizer_peak)
    capacity = (allocated_peak + runtime_baseline_bytes)*(1+margin)

d_attn is 1 if attention dropout is nonzero, else 0. Extra dropout masks are
accounted for in the source. The 26H/4I, score, loss, scratch and optimizer
coefficients are a conservative-style heuristic buffer envelope, NOT a proven
upper bound or a formula extracted from a particular PyTorch implementation.
Math attention is modeled with FP32 score/probability storage. Cast-cache and
loss overlap are conservatively summed. Autograd lifetimes vary by version.
No published activation formula is claimed as an exact match to this profile.

The 16P baseline counts FP32 weights, gradients and two AdamW moments. Small
optimizer counters are omitted. Gradients need not all be allocated at the
instant all forward tensors are live, so summing their maxima is approximate.
No quadratic term may simply be reused for FlashAttention.

## Runtime model

Forward FLOPs per microbatch:

    block projections = 2BS*(4H^2+2HI), per layer
    attention matrix multiplies = 4BS^2H, per layer
    vocabulary projection = 2BSHV

Training matrix-multiply work is approximated as three times forward work
(forward, input gradient, weight gradient). Small attention operations have an
additional heuristic term. Normalization, activation and dropout execution are
approximated through traffic and overhead, not a complete operator simulator.

    group_time = max(FLOPs/effective_compute, bytes/effective_bandwidth)
    effective_compute = dtype_specific_peak*efficiency
    effective_bandwidth = peak_bandwidth*efficiency
    microbatch_time = sum(group_times) + modeled_kernel_overhead
    update_time = G*microbatch_time + AdamW_time

Attention uses FP32 ceilings (TF32 disabled); projections use BF16 ceilings.
Traffic and kernel counts are explicitly heuristic; inspect microbatch() in
estimate_gpt2.py. This avoids treating all training FLOPs as BF16 tensor-core work.

Epoch mode calculates full and partial microbatches separately and flushes the
partial accumulation group at each epoch end. Integer epochs only, no packing,
drop_last=false. Update-count mode assumes full microbatches and full groups.

## Accuracy and next work

These calculations can run using only metadata. Physical timing accuracy still
requires backend-specific validation. Replace heuristic coefficients with a
versioned implementation ruleset/calibration, using held-out models and shapes.
For sensitivity ranges, rerun with separately justified optimistic/pessimistic
policies. Do not present such ranges as statistical confidence intervals.

Useful background:
- https://arxiv.org/abs/2205.05198 (activation analysis under its own assumptions)
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention
  (math-backend precision)

The included generated gpt2_estimate_result.json is output of the analytical
script, not GPU measurement. Training code is never imported or executed.

## Integration in the app (GPT-2 MVP)

- `estimate_gpt2.py` keeps the prototype formulas and CLI unchanged. The code is split into
  `memory_estimate()` (no hardware needed) and `runtime_estimate()` (needs GPU ceilings), and
  `estimate()` returns the prototype's combined output. Parity with the prototype on
  `examples/gpt2_mvp/gpt2_estimator_input.json` is checked in `tests/test_api.py`.
- The fit check compares `capacity_with_runtime_and_margin` against the **usable VRAM limit**
  (`hardware.usable_vram_gib`, default 100% of VRAM) instead of total VRAM, so no second margin is applied.
- `gpt2_predict.py` builds the `model` / `training` / `hardware` / `estimator_policy` objects from the
  form, using `DEFAULT_POLICY` (the prototype's illustrative policy), and adds cost:
  `total_training_compute.seconds / 3600 × price per GPU-hour` (1 GPU).
- `GET /api/analysis/{id}/files/estimator_input.json` returns the exact objects used, runnable with the CLI.
