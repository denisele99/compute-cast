"""Approximate projection for multi-technique training strategies (GPT-2 MVP).

The calibrated-style estimator in ``estimate_gpt2`` models ONE GPU with full activations. Recommendations
may also propose activation checkpointing, memory-efficient attention, ZeRO/FSDP sharding and
data / tensor / pipeline parallelism. This module scales the single-GPU memory components and the
single-GPU training time with simple, documented factors so those plans get *rough* numbers. Every
result is labelled approximate: the factors are textbook rules of thumb, not measurements.
"""
from __future__ import annotations

import math

from ...parsers import gpt2_inputs as G
from . import estimate_gpt2 as E
from . import gpt2_predict

GIB = 2 ** 30
SHARDING = ('none', 'zero1', 'zero2', 'zero3', 'fsdp')
CHECKPOINTING = ('none', 'selective', 'full')
ATTENTION = ('math', 'flash')
# Rule-of-thumb factors (illustrative, uncalibrated).
POLICY = {
    'full_checkpoint_compute_overhead': 0.33,       # one extra forward pass
    'selective_checkpoint_compute_overhead': 0.05,  # recompute attention scores only
    'data_parallel_efficiency': 0.92,               # all-reduce overlapped with backward
    'zero3_efficiency': 0.85,                       # extra all-gathers per layer
    'tensor_parallel_loss_per_doubling': 0.10,      # intra-layer all-reduces (NVLink)
}
ASSUMPTIONS = [
    'Approximate projection from the single-GPU estimate with rule-of-thumb factors; not calibrated or measured.',
    'Training state 16·P = FP32 weights 4·P + gradients 4·P + AdamW moments 8·P; ZeRO-1 shards moments, ZeRO-2 also gradients, ZeRO-3/FSDP everything, over the data-parallel group.',
    'Tensor parallelism divides weights, activations and logits by its degree (Megatron-style with sequence parallelism).',
    'Pipeline parallelism divides model state by the number of stages; 1F1B keeps about one stage-worth of activations per in-flight microbatch, so activations stay roughly unchanged. Bubble = (stages − 1) / (microbatches + stages − 1).',
    'Full activation checkpointing keeps each block input plus one block of recomputed activations; selective checkpointing and flash attention drop the S² attention-score buffers.',
    'Throughput scales with GPU count times the efficiency factors; communication on slow interconnects can be much worse.',
]


def _int(v, default=1):
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 1 else default


def normalise(strategy):
    s = dict(strategy or {})
    out = {'data_parallel': _int(s.get('data_parallel')), 'tensor_parallel': _int(s.get('tensor_parallel')),
           'pipeline_parallel': _int(s.get('pipeline_parallel')),
           'sharding': s.get('sharding') if s.get('sharding') in SHARDING else 'none',
           'activation_checkpointing': s.get('activation_checkpointing') if s.get('activation_checkpointing') in CHECKPOINTING else 'none',
           'attention': s.get('attention') if s.get('attention') in ATTENTION else 'math'}
    out['gpu_count'] = out['data_parallel'] * out['tensor_parallel'] * out['pipeline_parallel']
    return out


def is_trivial(s):
    return s['gpu_count'] == 1 and s['sharding'] == 'none' and s['activation_checkpointing'] == 'none' and s['attention'] == 'math'


def check(s, values):
    """Structural problems with a strategy for this model (empty list when consistent)."""
    issues = []
    heads, hidden, layers = values.get('model.num_attention_heads'), values.get('model.hidden_size'), values.get('model.num_layers')
    T, P, D = s['tensor_parallel'], s['pipeline_parallel'], s['data_parallel']
    if T > 1 and ((heads and heads % T) or (hidden and hidden % T)):
        issues.append(f'tensor parallel degree {T} must divide the {heads} attention heads and hidden size {hidden}')
    if layers and P > layers:
        issues.append(f'{P} pipeline stages exceed the {layers} layers')
    if s['sharding'] != 'none' and D < 2:
        issues.append('ZeRO/FSDP sharding needs a data-parallel degree of at least 2 to save memory')
    if s['gpu_count'] > 1024:
        issues.append('more than 1024 GPUs')
    return issues


def project(values, strategy, extraction=None, acknowledged=None):
    """Rough per-GPU memory, time and cost for ``values`` trained with ``strategy``.

    ``values`` hold the per-GPU microbatch and the accumulation steps of each data-parallel replica.
    """
    s = normalise(strategy)
    D, T, Pp = s['data_parallel'], s['tensor_parallel'], s['pipeline_parallel']
    accumulation = values.get('training.gradient_accumulation_steps') or 1
    # One GPU processing the same global batch: same number of updates and token positions.
    single = gpt2_predict.predict({**values, 'training.gradient_accumulation_steps': accumulation * D}, None, extraction, acknowledged)
    if single.get('status') != 'ok':
        return {'status': 'blocked', 'errors': [e['message'] for e in single['validation']['errors'][:3]]}
    v = G.validate(values, None, extraction, acknowledged)['values']
    model, training = G.gpt2_model(v), G.gpt2_training(v)
    L, H, A = model['n_layer'], model['n_embd'], model['n_head']
    B, S = training['microbatch_size'], training['sequence_length']
    score_bytes = 8 + (1 if model.get('attn_pdrop', .1) else 0)
    m = single['memory']  # GiB
    P = single['parameters']

    weights, grads, moments = 4 * P, 4 * P, 8 * P
    if s['sharding'] == 'zero1':
        moments /= D
    elif s['sharding'] == 'zero2':
        grads, moments = grads / D, moments / D
    elif s['sharding'] in ('zero3', 'fsdp'):
        weights, grads, moments = weights / D, grads / D, moments / D
    model_split = T * Pp
    state = (weights + grads + moments) / model_split
    cast_cache = m['autocast_cache'] * GIB / model_split / (D if s['sharding'] in ('zero3', 'fsdp') else 1)

    saved = m['saved_activations'] * GIB
    attention_scores = L * score_bytes * B * A * S * S
    if s['activation_checkpointing'] == 'full':
        saved = saved / L + 2 * B * S * H * L  # one recomputed block + BF16 block inputs
    elif s['activation_checkpointing'] == 'selective' or s['attention'] == 'flash':
        saved = max(saved - attention_scores, 0)
    saved /= T
    loss = m['loss_buffers'] * GIB / T
    scratch = m['backward_scratch'] * GIB
    if s['attention'] == 'flash':  # no B·A·S² score scratch
        scratch = min(scratch, 4 * B * S * max(H, single['resolved_intermediate_size']))
    scratch /= T
    largest = max(model['vocab_size'] * H, 3 * H * H, H * single['resolved_intermediate_size']) / T
    forward_backward = state + cast_cache + saved + loss + scratch
    optimizer = state + E.DEFAULT_POLICY['optimizer_scratch_copies'] * 4 * largest
    allocated = max(forward_backward, optimizer)
    capacity = (allocated + E.DEFAULT_POLICY['runtime_baseline_gib'] * GIB) * (1 + E.DEFAULT_POLICY['capacity_margin_fraction'])
    usable = v['hardware.usable_vram_gib']

    hours = cost = None
    efficiency = 1.0
    if D > 1:
        efficiency *= POLICY['zero3_efficiency'] if s['sharding'] in ('zero3', 'fsdp') else POLICY['data_parallel_efficiency']
    if T > 1:
        efficiency *= max(0.3, 1 - POLICY['tensor_parallel_loss_per_doubling'] * math.log2(T))
    if Pp > 1:
        efficiency *= accumulation / (accumulation + Pp - 1)
    overhead = 1 + {'full': POLICY['full_checkpoint_compute_overhead'], 'selective': POLICY['selective_checkpoint_compute_overhead']}.get(s['activation_checkpointing'], 0)
    runtime = single.get('runtime') or {}
    total = runtime.get('total_training_compute')
    if total:
        hours = total['seconds'] / 3600 * overhead / (s['gpu_count'] * efficiency)
        price, unit = v.get('hardware.price_per_hour'), v.get('hardware.price_unit')
        if price is not None:
            cost = hours * price * (s['gpu_count'] if unit != 'instance_hour' else 1)
    return {'status': 'ok', 'approximate': True, 'strategy': s,
            'per_gpu_capacity_gib': capacity / GIB, 'allocated_peak_gib': allocated / GIB, 'usable_vram_gib': usable,
            'percent_of_usable': 100 * capacity / GIB / usable, 'fits': capacity <= usable * GIB,
            'memory_gib': {'training_state': state / GIB, 'autocast_cache': cast_cache / GIB, 'saved_activations': saved / GIB,
                           'loss_buffers': loss / GIB, 'backward_scratch': scratch / GIB},
            'training_hours': hours, 'cost_usd': cost, 'gpu_count': s['gpu_count'], 'efficiency': efficiency,
            'compute_overhead': overhead, 'global_effective_batch': B * accumulation * D,
            'optimizer_updates': total['optimizer_updates'] if total else None, 'assumptions': ASSUMPTIONS}
