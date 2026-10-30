"""Metadata-only GPT-2 training estimate (active MVP estimator). Standard library only; no torch.

Memory lifetimes, traffic and efficiency are explicit uncalibrated heuristics.
This is not an exact simulator of a particular Transformers/PyTorch release.
Formulas: see ESTIMATOR.md next to this file. CLI (unchanged from the prototype):

    python -m app.services.estimators.estimate_gpt2 gpt2_estimator_input.json --output result.json

The formulas are those of the original prototype, split into memory_estimate() (no hardware
needed) and runtime_estimate() (needs GPU ceilings) so peak memory can be reported even when
compute specs are unknown. estimate() keeps the prototype's combined output.
"""
import argparse
import json
import math
from pathlib import Path

# Illustrative, uncalibrated policy (prototype gpt2_estimator_input.json "estimator_policy").
DEFAULT_POLICY = {
    'label': 'ILLUSTRATIVE uncalibrated assumptions; not measured confidence bounds',
    'bf16_efficiency': 0.3,
    'fp32_efficiency': 0.3,
    'bandwidth_efficiency': 0.5,
    'launch_overhead_us': 10,
    'other_kernels_per_layer': 40,
    'loss_kernels': 8,
    'adam_kernels_per_tensor': 12,
    'adam_flops_per_parameter': 20,
    'adam_bytes_per_parameter': 40,
    'loss_bytes_per_logit': 14,
    'optimizer_scratch_copies': 2,
    'runtime_baseline_gib': 0.5,
    'capacity_margin_fraction': 0.15,
}

FORMULAS = {
    'parameters': 'P = V·H + C·H + L·(4H² + 2HI + 9H + I) + 2H   (I = n_inner or 4H)',
    'training_state': 'state = 16·P   (FP32 weights + FP32 gradients + two FP32 AdamW moments)',
    'autocast_cache': 'cast_cache = 2·[L·(4H² + 2HI) + V·H]   (BF16 copies of linear weights)',
    'saved_activations': 'saved = L·[B·S·(26H + 4I + 16) + (8 + d_attn)·B·A·S²] + dropout_masks',
    'loss_buffers': 'loss = loss_bytes_per_logit · B·S·V',
    'scratch': 'scratch = max(4·B·A·S², 4·B·S·I, 4·B·S·H)',
    'peaks': 'fwd_bwd = state + cast_cache + saved + loss + scratch;  optimizer = state + copies·4·largest_tensor;  '
             'allocated = max(fwd_bwd, optimizer)',
    'capacity': 'capacity = (allocated + runtime_baseline)·(1 + margin);  fits if capacity ≤ usable VRAM limit',
    'flops': 'per layer fwd: 2BS(4H² + 2HI) projections + 4BS²H attention; head fwd 2BSHV; training ≈ 3 × forward '
             '(+ 15·B·A·S² small attention ops)',
    'time': 'group_time = max(FLOPs / (peak·efficiency), bytes / (bandwidth·efficiency)); projections and head use '
            'BF16 ceilings, attention FP32 (TF32 disabled); microbatch = Σ groups + kernel overhead',
    'update': 'update_time = G·microbatch_time + AdamW_time;  AdamW = max(flops·P/FP32, bytes·P/BW) + launches',
    'duration': 'updates mode: updates·update_time.  epochs mode: per epoch full microbatches + one partial '
                'microbatch + ceil(microbatches/G) optimizer steps (partial group flushed at epoch end)',
}


def _check_model(model):
    if model.get('model_type') != 'gpt2':
        raise ValueError('Only model_type=gpt2 is supported.')
    for flag in ('add_cross_attention', 'reorder_and_upcast_attn', 'scale_attn_by_inverse_layer_idx'):
        if model.get(flag, False):
            raise ValueError('Unsupported model option: ' + flag)
    if not model.get('tie_word_embeddings', True) or not model.get('scale_attn_weights', True):
        raise ValueError('Requires tied embeddings and standard scaled attention.')
    if model.get('activation_function', 'gelu_new') != 'gelu_new':
        raise ValueError('Requires gelu_new.')
    L, H, A, V, C = [model[k] for k in ('n_layer', 'n_embd', 'n_head', 'vocab_size', 'n_positions')]
    I = model.get('n_inner') or 4 * H
    for key in ('attn_pdrop', 'embd_pdrop', 'resid_pdrop'):
        if not 0 <= model.get(key, 0.1) < 1:
            raise ValueError('Invalid dropout: ' + key)
    return L, H, A, V, C, I


def _check_shapes(L, H, A, V, C, I, B, S, G):
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in (L, H, A, V, C, I, B, S, G)):
        raise ValueError('Dimensions, batch, sequence and accumulation must be positive integers.')
    if H % A or S > C:
        raise ValueError('Requires hidden_size divisible by heads and sequence_length <= n_positions.')


def memory_estimate(model, training, policy):
    L, H, A, V, C, I = _check_model(model)
    B, S, G = [training[k] for k in ('microbatch_size', 'sequence_length', 'gradient_accumulation_steps')]
    _check_shapes(L, H, A, V, C, I, B, S, G)
    # Two LayerNorms/block, qkv/output projections with biases, two MLP projections.
    block_matrix = 4 * H * H + 2 * H * I
    block_params = block_matrix + 9 * H + I
    P = V * H + C * H + L * block_params + 2 * H
    linear_weights = L * block_matrix + V * H
    largest_tensor = max(V * H, C * H, 3 * H * H, H * I)
    state = 16 * P  # FP32 weights + gradients + two moments; excludes scalar counters.
    cast_cache = 2 * linear_weights
    N = B * S
    # Heuristic saved-buffer envelope, NOT a measured backend allocation formula.
    dropout_masks = N * H * (L + 1) if model.get('resid_pdrop', .1) else 0
    dropout_masks += N * H if model.get('embd_pdrop', .1) else 0
    score_bytes = 8 + (1 if model.get('attn_pdrop', .1) else 0)
    saved = L * (N * (26 * H + 4 * I + 16) + score_bytes * B * A * S * S)
    saved += dropout_masks
    # BF16 logits plus several FP32 loss/gradient buffers: conservative overlap.
    loss_buffers = policy['loss_bytes_per_logit'] * N * V
    scratch = max(4 * B * A * S * S, 4 * N * I, 4 * N * H)
    forward_backward = state + cast_cache + saved + loss_buffers + scratch
    optimizer_peak = state + policy['optimizer_scratch_copies'] * 4 * largest_tensor
    allocated = max(forward_backward, optimizer_peak)
    baseline = policy['runtime_baseline_gib'] * 2**30
    recommended = (allocated + baseline) * (1 + policy['capacity_margin_fraction'])
    return {
        'parameters': P, 'head_dim': H // A, 'effective_batch_size': B * G,
        'resolved_intermediate_size': I, 'largest_parameter_tensor': largest_tensor,
        'memory_bytes': dict(training_state=state, autocast_cache=cast_cache,
                             saved_activations=saved, loss_buffers=loss_buffers,
                             backward_scratch=scratch, forward_backward_phase=forward_backward,
                             optimizer_phase=optimizer_peak, estimated_allocated_peak=allocated,
                             runtime_baseline=baseline,
                             capacity_with_runtime_and_margin=recommended),
    }


def runtime_estimate(model, training, hardware, policy):
    L, H, A, V, C, I = _check_model(model)
    B, S, G = [training[k] for k in ('microbatch_size', 'sequence_length', 'gradient_accumulation_steps')]
    _check_shapes(L, H, A, V, C, I, B, S, G)
    if not hardware['bf16_supported']:
        raise ValueError('The profile requires native BF16 support.')
    for key in ('bf16_efficiency', 'fp32_efficiency', 'bandwidth_efficiency'):
        if not 0 < policy[key] <= 1:
            raise ValueError('Efficiency must be in (0,1].')
    block_matrix = 4 * H * H + 2 * H * I
    P = V * H + C * H + L * (block_matrix + 9 * H + I) + 2 * H
    bf = hardware['bf16_dense_tflops'] * 1e12 * policy['bf16_efficiency']
    fp = hardware['fp32_tflops'] * 1e12 * policy['fp32_efficiency']
    bw = hardware['memory_bandwidth_gbps'] * 1e9 * policy['bandwidth_efficiency']
    if min(bf, fp, bw) <= 0:
        raise ValueError('Hardware ceilings must be positive.')

    def microbatch(batch):
        n = batch * S
        # Dense full S*S math attention, not triangular FlashAttention accounting.
        block_fwd = 2 * n * block_matrix
        attn_fwd = 4 * batch * S * S * H
        head_fwd = 2 * n * H * V
        # Trainable dense matrices: forward + dInput + dWeight ~ 3x forward.
        block_flops = 3 * block_fwd
        attn_flops = 3 * attn_fwd + 15 * batch * A * S * S
        head_flops = 3 * head_fwd
        # Simplified traffic, not summed storage allocation.
        block_traffic = 12 * block_matrix + 3 * n * (26 * H + 4 * I)
        attn_traffic = 6 * 4 * batch * A * S * S + 12 * 4 * n * H
        head_traffic = 12 * H * V + 20 * n * V + 6 * n * H
        overhead = policy['launch_overhead_us'] * 1e-6
        seconds = L * (
            max(block_flops / bf, block_traffic / bw)
            + max(attn_flops / fp, attn_traffic / bw)
            + policy['other_kernels_per_layer'] * overhead
        )
        seconds += max(head_flops / bf, head_traffic / bw)
        seconds += policy['loss_kernels'] * overhead
        return seconds, L * (block_flops + attn_flops) + head_flops

    # Unfused AdamW: approximate bytes moved/parameter and separate kernel launches.
    parameter_tensor_count = 4 + 12 * L
    optimizer_seconds = max(policy['adam_flops_per_parameter'] * P / fp,
                            policy['adam_bytes_per_parameter'] * P / bw)
    optimizer_seconds += (parameter_tensor_count * policy['adam_kernels_per_tensor']
                          * policy['launch_overhead_us'] * 1e-6)
    micro_seconds, micro_flops = microbatch(B)
    update_seconds = G * micro_seconds + optimizer_seconds
    duration = None
    if training.get('optimizer_updates') is not None:
        updates = training['optimizer_updates']
        if not isinstance(updates, int) or updates <= 0:
            raise ValueError('optimizer_updates must be a positive integer.')
        if training.get('num_epochs') is not None:
            raise ValueError('Supply optimizer_updates OR epochs/examples, not both.')
        duration = dict(mode='optimizer_updates', optimizer_updates=updates, seconds=updates * update_seconds,
                        processed_positions=updates * B * G * S)
    elif training.get('num_epochs') is not None:
        epochs, examples = training['num_epochs'], training['num_training_examples']
        if any(not isinstance(x, int) or x <= 0 for x in (epochs, examples)):
            raise ValueError('Epoch mode requires positive integer epochs and processed examples.')
        full, remainder = divmod(examples, B)
        batches = full + bool(remainder)
        updates_per_epoch = math.ceil(batches / G)
        seconds = full * micro_seconds + updates_per_epoch * optimizer_seconds
        if remainder:
            seconds += microbatch(remainder)[0]
        duration = dict(mode='epochs', optimizer_updates=epochs * updates_per_epoch, seconds=epochs * seconds,
                        microbatches_per_epoch=batches, optimizer_updates_per_epoch=updates_per_epoch,
                        processed_positions=epochs * examples * S)
    return {'training_flops_per_microbatch': micro_flops, 'seconds_per_microbatch': micro_seconds,
            'optimizer_seconds_per_update': optimizer_seconds,
            'seconds_per_full_optimizer_update': update_seconds, 'total_training_compute': duration}


def estimate(model, training, hardware, policy):
    """Prototype-compatible combined output. capacity_check uses hardware['usable_vram_gib'] if given."""
    mem = memory_estimate(model, training, policy)
    run = runtime_estimate(model, training, hardware, policy)
    limit = hardware.get('usable_vram_gib', hardware['vram_gib'])
    return {
        'status': 'uncalibrated_analytical_estimate',
        'scope': 'GPT-2 full training; FP32 state; BF16 autocast; FP32 math attention; one GPU',
        **mem, **run,
        'capacity_check': 'estimated_fit' if mem['memory_bytes']['capacity_with_runtime_and_margin'] <= limit * 2**30
        else 'exceeds_budget',
        'assumptions': policy,
        'limitations': ['No loading, evaluation, saving or data-pipeline time.',
                        'Timing is hypothetical if capacity is exceeded.',
                        'Memory coefficients and traffic are heuristics, not backend guarantees.',
                        'Epoch mode flushes partial accumulation at each epoch end; drop_last=false.'],
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('metadata', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    metadata = json.loads(a.metadata.read_text())
    result = estimate(metadata['model'], metadata['training'], metadata['hardware'], metadata['estimator_policy'])
    rendered = json.dumps(result, indent=2)
    if a.output:
        a.output.write_text(rendered)
    print(rendered)
