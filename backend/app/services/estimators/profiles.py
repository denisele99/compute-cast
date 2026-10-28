"""Supported prediction profiles: validation, parameter layout and activation formulas.

PyTorch only. Two explicit profiles are supported:

* ``mlp_regression``  - ``nn.Sequential(nn.Linear(I, H), nn.ReLU(), nn.Linear(H, O))``
  fully trained with MSE regression, gradient checkpointing disabled.
* ``transformer_lm_classic`` - GPT-2-style decoder/encoder stack: standard multi-head
  attention (no GQA/MQA), feed-forward width ``4 * hidden_size`` with a non-gated
  GELU/ReLU MLP, LayerNorm, 16-bit activations, eager/standard attention, full
  training, gradient checkpointing disabled.

Nothing here imports or executes model code. Parameter counts and tensor shapes are
derived from explicit architecture settings using the formulas below, so they are only
as accurate as those settings. Bias and weight-tying choices are explicit inputs.
"""
from __future__ import annotations

import math

FRAMEWORK = 'pytorch'

PROFILES = {
    'mlp_regression': {
        'label': 'Two-layer MLP regression (Linear → ReLU → Linear)',
        'summary': 'nn.Sequential(Linear(I,H), ReLU(), Linear(H,O)); full training with MSE loss; '
                   'gradient checkpointing disabled.',
        'tasks': ['regression'],
        'precisions': ['fp32', 'fp16', 'bf16'],
    },
    'transformer_lm_classic': {
        'label': 'Classic transformer language model (GPT-2 style)',
        'summary': 'Standard multi-head attention, 4×hidden_size non-gated feed-forward, LayerNorm, '
                   '16-bit activations, eager/standard attention, full training, gradient '
                   'checkpointing disabled.',
        'tasks': ['causal_language_modeling', 'masked_language_modeling'],
        'precisions': ['fp16', 'bf16'],
    },
}

PARAMETER_DTYPES = ('fp32', 'fp16', 'bf16')
TORCH_DTYPE = {'fp32': 'torch.float32', 'fp16': 'torch.float16', 'bf16': 'torch.bfloat16'}

FORMULAS = {
    'mlp_parameters': {
        'formula': 'P = I*H + H*O + bias*(H + O)',
        'basis': 'nn.Linear(in, out) stores weight [out, in] and, when bias=True, bias [out].',
        'sources': ['https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html'],
    },
    'transformer_parameters': {
        'formula': 'P = V*H + learned_pos*P_max*H + L*(4*H*H + 2*H*4H + bias*(3H + H + 4H + H) '
                   '+ 2*ln*(H + ln_bias*H)) + ln*(H + ln_bias*H) + (1 - tied)*V*H + lm_head_bias*V',
        'basis': 'GPT-2 layout: token embedding, optional learned positions, per layer fused QKV '
                 '(3H×H), attention output (H×H), MLP up (4H×H) and down (H×4H) projections, two '
                 'LayerNorms, a final LayerNorm and an LM head that is tied to the token embedding '
                 'unless untied. Reproduces GPT-2 small: 124,439,808 parameters.',
        'sources': ['https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py'],
    },
    'mlp_activations': {
        'formula': 'saved = B*(I + H)*b_act',
        'basis': 'Linear saves its input; ReLU output is shared with the second Linear input. '
                 'Same rule as the memory estimator simple_mlp profile.',
        'sources': ['https://docs.pytorch.org/docs/stable/notes/autograd.html#saved-tensors'],
    },
    'transformer_activations': {
        'formula': 'saved = 34*L*B*S*H + 5*L*B*A*S*S bytes (16-bit activations)',
        'basis': 'Korthikanti et al. 2022, Section 4.1, no recomputation, no sequence parallelism. '
                 'Logits/loss buffers are counted separately by the memory estimator.',
        'sources': ['https://arxiv.org/html/2205.05198v1#S4.SS1'],
    },
}

# Hugging Face model types that are known not to match the classic profile.
UNSUPPORTED_MODEL_TYPES = {
    'llama': 'Llama uses a gated SwiGLU MLP, RMSNorm, rotary positions and often grouped-query attention.',
    'mistral': 'Mistral uses a gated MLP, grouped-query and sliding-window attention.',
    'mixtral': 'Mixtral is a mixture-of-experts model.',
    'qwen2': 'Qwen2 uses a gated MLP and grouped-query attention.',
    'qwen3': 'Qwen3 uses a gated MLP and grouped-query attention.',
    'gemma': 'Gemma uses a gated MLP and multi-query/grouped attention.',
    'gemma2': 'Gemma 2 uses a gated MLP and grouped-query attention.',
    'phi3': 'Phi-3 uses a gated MLP.',
    'falcon': 'Falcon uses multi-query/grouped attention and parallel attention blocks.',
    't5': 'T5 is an encoder-decoder model.',
    'bart': 'BART is an encoder-decoder model.',
    'mt5': 'mT5 is an encoder-decoder model.',
}


class ProfileError(ValueError):
    """Settings are valid JSON but outside a supported profile."""


def _int(cfg, key, minimum=1):
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProfileError(f'{key} must be a whole number >= {minimum}.')
    return value


def _bool(cfg, key):
    value = cfg.get(key)
    if not isinstance(value, bool):
        raise ProfileError(f'{key} must be explicitly true or false.')
    return value


def profile_issues(profile, model, training):
    """Return (errors, unsupported) lists of {field, message}; never raises for bad input."""
    errors, unsupported = [], []

    def bad(field, message, kind=errors):
        kind.append({'field': field, 'message': message})

    if profile not in PROFILES:
        bad('model.profile', 'Select a supported prediction profile.', unsupported)
        return errors, unsupported
    spec = PROFILES[profile]
    if training.get('gradient_checkpointing') is True:
        bad('training.gradient_checkpointing', 'Gradient checkpointing is not supported by either '
            'profile; the activation formulas assume no recomputation.', unsupported)
    if training.get('training_method') not in (None, 'full'):
        bad('training.training_method', 'Only full training is supported (no LoRA/frozen layers).',
            unsupported)
    if training.get('task') not in (None, *spec['tasks']):
        bad('training.task', f"{spec['label']} supports task(s): {', '.join(spec['tasks'])}.",
            unsupported)
    if training.get('precision') not in (None, *spec['precisions']):
        bad('training.precision', 'The classic transformer formula requires 16-bit activations '
            '(fp16 or bf16).' if profile == 'transformer_lm_classic'
            else 'Precision must be fp32, fp16 or bf16.', unsupported)
    if profile == 'transformer_lm_classic':
        if training.get('attention_backend') not in (None, 'eager', 'standard'):
            bad('training.attention_backend', 'Only eager/standard attention matches the formula; '
                'FlashAttention and SDPA change saved tensors and traffic.', unsupported)
        H, A = model.get('hidden_size'), model.get('num_attention_heads')
        if isinstance(H, int) and isinstance(A, int) and A > 0 and H % A:
            bad('model.num_attention_heads', 'hidden_size must be divisible by num_attention_heads.')
        kv = model.get('num_key_value_heads')
        if kv is not None and A is not None and kv != A:
            bad('model.num_key_value_heads', 'Grouped/multi-query attention is not supported; '
                'num_key_value_heads must equal num_attention_heads.', unsupported)
        inter = model.get('intermediate_size')
        if inter is not None and isinstance(H, int) and inter != 4 * H:
            bad('model.intermediate_size', f'Feed-forward width must be 4 × hidden_size '
                f'({4 * H}); got {inter}.', unsupported)
        if model.get('mlp_activation') not in (None, 'gelu', 'relu'):
            bad('model.mlp_activation', 'Gated MLPs (SwiGLU/GeGLU) are not supported; use a '
                'non-gated GELU or ReLU feed-forward.', unsupported)
        S, P = training.get('sequence_length'), model.get('max_position_embeddings')
        if (model.get('position_embedding') == 'learned' and isinstance(S, int)
                and isinstance(P, int) and S > P):
            bad('training.sequence_length', f'Sequence length {S} exceeds max_position_embeddings {P}.')
    return errors, unsupported


def mlp_tensors(model):
    I, H, O = (_int(model, k) for k in ('input_size', 'hidden_size', 'output_size'))
    bias = _bool(model, 'linear_bias')
    rows = [('0.weight', [H, I])] + ([('0.bias', [H])] if bias else [])
    rows += [('2.weight', [O, H])] + ([('2.bias', [O])] if bias else [])
    return rows


def transformer_tensors(model):
    V, H, L, A = (_int(model, k) for k in
                  ('vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads'))
    bias, ln_bias = _bool(model, 'linear_bias'), _bool(model, 'layer_norm_bias')
    tied, head_bias = _bool(model, 'tie_word_embeddings'), _bool(model, 'lm_head_bias')
    learned = model.get('position_embedding')
    if learned not in ('learned', 'none'):
        raise ProfileError('position_embedding must be "learned" or "none".')
    rows = [('transformer.wte.weight', [V, H])]
    if learned == 'learned':
        rows.append(('transformer.wpe.weight', [_int(model, 'max_position_embeddings'), H]))

    def norm(prefix):
        return [(prefix + '.weight', [H])] + ([(prefix + '.bias', [H])] if ln_bias else [])

    def linear(prefix, out, inp):
        return [(prefix + '.weight', [out, inp])] + ([(prefix + '.bias', [out])] if bias else [])

    for i in range(L):
        p = f'transformer.h.{i}'
        rows += norm(p + '.ln_1') + linear(p + '.attn.c_attn', 3 * H, H)
        rows += linear(p + '.attn.c_proj', H, H) + norm(p + '.ln_2')
        rows += linear(p + '.mlp.c_fc', 4 * H, H) + linear(p + '.mlp.c_proj', H, 4 * H)
    rows += norm('transformer.ln_f')
    if not tied:
        rows.append(('lm_head.weight', [V, H]))
    if head_bias:
        rows.append(('lm_head.bias', [V]))
    return rows


def parameter_layout(profile, model, parameter_dtype):
    """Return (tensor_inventory_rows, summary). Shared (tied) weights are counted once."""
    if parameter_dtype not in PARAMETER_DTYPES:
        raise ProfileError('Parameter storage dtype must be explicit: fp32, fp16 or bf16.')
    rows = mlp_tensors(model) if profile == 'mlp_regression' else transformer_tensors(model)
    dtype = TORCH_DTYPE[parameter_dtype]
    size = {'fp32': 4, 'fp16': 2, 'bf16': 2}[parameter_dtype]
    inventory = [{'name': name, 'shape': shape, 'numel': math.prod(shape), 'dtype': dtype,
                  'device': 'not_allocated', 'layout': 'torch.strided',
                  'logical_dense_bytes': math.prod(shape) * size, 'storage_bytes': None,
                  'derived_from': 'profile_formula'} for name, shape in rows]
    total = sum(t['numel'] for t in inventory)
    summary = {'total_params': total, 'trainable_params': total,
               'logical_weight_bytes': total * size, 'largest_tensor_params': max(t['numel'] for t in inventory)}
    if profile == 'transformer_lm_classic':
        H, L = model['hidden_size'], model['num_hidden_layers']
        per_layer = sum(t['numel'] for t in inventory if t['name'].startswith('transformer.h.0.'))
        summary['params_per_layer'] = per_layer
        summary['embedding_params'] = sum(t['numel'] for t in inventory if '.wte.' in t['name'] or '.wpe.' in t['name'])
        summary['tied_lm_head'] = model['tie_word_embeddings']
        summary['layers'] = L
        summary['hidden_size'] = H
    return inventory, summary


def activation_estimate(profile, model, batch, sequence_length, activation_bytes):
    """Saved-activation bytes per microbatch; same formulas as the memory estimator."""
    if profile == 'mlp_regression':
        value = batch * (model['input_size'] + model['hidden_size']) * activation_bytes
        return {'saved_activation_bytes': value, 'formula_id': 'mlp_activations'}
    L, H, A = model['num_hidden_layers'], model['hidden_size'], model['num_attention_heads']
    S = sequence_length
    value = 34 * L * batch * S * H + 5 * L * batch * A * S * S
    return {'saved_activation_bytes': value, 'formula_id': 'transformer_activations'}
