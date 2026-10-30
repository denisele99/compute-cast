"""Trusted model registry and benchmark_config.json validation (shared by backend and runner).

Pure Python (no torch import at module level) so the backend can validate configurations with
exactly the rules the runner enforces. Only architectures listed here can be benchmarked, and
only the Hugging Face IDs/revisions in HF_MODELS map onto them. Nothing here loads remote code.
"""
from __future__ import annotations

import hashlib
import json
import math

CONFIG_SCHEMA_VERSION = '1.0'
RESULTS_SCHEMA_VERSION = '1.0'
BENCHMARK_CODE_VERSION = '0.1.1'

# Conservative MVP limits. The backend may tighten (never loosen) them via policy.
LIMITS = {
    'num_layers': 48, 'hidden_size': 2048, 'num_attention_heads': 32, 'intermediate_size': 8192,
    'vocab_size': 65536, 'max_position_embeddings': 4096, 'sequence_length': 2048, 'microbatch_size': 64,
    'gradient_accumulation_steps': 32, 'warmup_updates': 10, 'measured_updates': 50,
    'max_total_microbatches': 400,        # (warmup + measured) * accumulation
    'max_tokens_per_microbatch': 65536,   # microbatch * sequence_length
    'max_parameters': 1_000_000_000,
}

# Hugging Face IDs whose architecture the trusted decoder reproduces exactly. Dimensions come from
# their config.json (bundled snapshot). Weights are NOT loaded in this MVP (random initialisation),
# so no revision is fetched; revision stays null until pretrained loading is added and pinned.
HF_MODELS = {
    'openai-community/gpt2': {'architecture': 'gpt2_decoder', 'revision': None, 'dimensions': {
        'num_layers': 12, 'hidden_size': 768, 'num_attention_heads': 12, 'intermediate_size': None,
        'vocab_size': 50257, 'max_position_embeddings': 1024, 'attention_dropout': 0.1,
        'embedding_dropout': 0.1, 'residual_dropout': 0.1}},
    'distilbert/distilgpt2': {'architecture': 'gpt2_decoder', 'revision': None, 'dimensions': {
        'num_layers': 6, 'hidden_size': 768, 'num_attention_heads': 12, 'intermediate_size': None,
        'vocab_size': 50257, 'max_position_embeddings': 1024, 'attention_dropout': 0.1,
        'embedding_dropout': 0.1, 'residual_dropout': 0.1}},
}

ARCHITECTURES = {
    'gpt2_decoder': {
        'label': 'Standard GPT-2 causal decoder (trusted implementation: models/decoder.py)',
        'module': 'models.decoder',
        'weights': {'random_init': 'supported', 'pretrained': 'not supported in this MVP'},
        'precision': ['bf16_autocast'],
        'optimizer': ['adamw'],
        'attention': ['math_fp32'],
        'training_mode': ['full'],
        'input_source': ['synthetic'],
        'loss': 'causal language-model cross-entropy over all positions (FP32)',
    },
}

OPTIMIZER_SETTINGS = {'lr': (float, 1e-6, 1.0, 1e-4), 'beta1': (float, 0.0, 0.9999, 0.9),
                      'beta2': (float, 0.0, 0.99999, 0.999), 'eps': (float, 1e-12, 1e-3, 1e-8),
                      'weight_decay': (float, 0.0, 1.0, 0.0)}

SCHEMA = {  # section -> field -> (type, required, extra)
    'model': {'source': ('enum', True, ('huggingface', 'custom')), 'hf_model_id': ('str_or_null', True, None),
              'hf_revision': ('str_or_null', True, None), 'weights': ('enum', True, ('random_init',)),
              'num_layers': ('int', True, None), 'hidden_size': ('int', True, None),
              'num_attention_heads': ('int', True, None), 'intermediate_size': ('int_or_null', True, None),
              'vocab_size': ('int', True, None), 'max_position_embeddings': ('int', True, None),
              'attention_dropout': ('prob', True, None), 'embedding_dropout': ('prob', True, None),
              'residual_dropout': ('prob', True, None)},
    'input': {'source': ('enum', True, ('synthetic',)), 'sequence_length': ('int', True, None)},
    'training': {'mode': ('enum', True, ('full',)), 'precision': ('enum', True, ('bf16_autocast',)),
                 'optimizer': ('enum', True, ('adamw',)), 'optimizer_settings': ('object', True, None),
                 'microbatch_size': ('int', True, None), 'gradient_accumulation_steps': ('int', True, None),
                 'attention': ('enum', True, ('math_fp32',))},
    'measurement': {'warmup_updates': ('int0', True, None), 'measured_updates': ('int', True, None),
                    'seed': ('int0', True, None)},
    'hardware': {'requested_gpu': ('str_or_null', True, None), 'gpu_count': ('enum', True, (1,))},
}
TOP_LEVEL = {'schema_version', 'architecture', 'model', 'input', 'training', 'measurement', 'hardware'}


class ConfigError(ValueError):
    """benchmark_config.json is invalid; .errors holds [{'field', 'message'}]."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__('; '.join(f"{e['field']}: {e['message']}" for e in errors))


def _type_ok(kind, value, extra):
    if kind == 'enum':
        return value in extra
    if kind in ('int', 'int0', 'int_or_null'):
        if value is None and kind == 'int_or_null':
            return True
        return isinstance(value, int) and not isinstance(value, bool) and value >= (0 if kind == 'int0' else 1)
    if kind == 'prob':
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value < 1
    if kind == 'str_or_null':
        return value is None or (isinstance(value, str) and 0 < len(value) <= 200)
    if kind == 'object':
        return isinstance(value, dict)
    return False


def parameter_count(m):
    """Exact unique parameters of the trusted GPT-2 decoder (tied embeddings)."""
    L, H, V, C = m['num_layers'], m['hidden_size'], m['vocab_size'], m['max_position_embeddings']
    I = m['intermediate_size'] or 4 * H
    return V * H + C * H + L * (4 * H * H + 2 * H * I + 9 * H + I) + 2 * H


def validate(config, limits=None):
    """Return the normalized config or raise ConfigError. Rejects unknown fields and bad types."""
    lim = {**LIMITS, **(limits or {})}
    errors = []

    def err(field, message):
        errors.append({'field': field, 'message': message})

    if not isinstance(config, dict):
        raise ConfigError([{'field': '$', 'message': 'Configuration must be a JSON object.'}])
    for key in sorted(set(config) - TOP_LEVEL):
        err(key, 'Unknown field.')
    for key in sorted(TOP_LEVEL - set(config)):
        err(key, 'Missing required field.')
    if config.get('schema_version') != CONFIG_SCHEMA_VERSION:
        err('schema_version', f'Must be "{CONFIG_SCHEMA_VERSION}".')
    if config.get('architecture') not in ARCHITECTURES:
        err('architecture', f"Unsupported architecture; supported: {', '.join(ARCHITECTURES)}.")
    for section, fields in SCHEMA.items():
        block = config.get(section)
        if not isinstance(block, dict):
            if section in config:
                err(section, 'Must be an object.')
            continue
        for key in sorted(set(block) - set(fields)):
            err(f'{section}.{key}', 'Unknown field.')
        for key, (kind, required, extra) in fields.items():
            if key not in block:
                if required:
                    err(f'{section}.{key}', 'Missing required field.')
                continue
            if not _type_ok(kind, block[key], extra):
                hint = f'one of {list(extra)}' if kind == 'enum' else {'int': 'a positive integer', 'int0': 'a non-negative integer',
                       'int_or_null': 'a positive integer or null', 'prob': 'a number in [0, 1)',
                       'str_or_null': 'a string or null', 'object': 'an object'}[kind]
                err(f'{section}.{key}', f'Must be {hint}.')
    if errors:
        raise ConfigError(errors)
    m, t, inp, meas = config['model'], config['training'], config['input'], config['measurement']
    opts = t['optimizer_settings']
    for key in sorted(set(opts) - set(OPTIMIZER_SETTINGS)):
        err(f'training.optimizer_settings.{key}', 'Unknown optimizer setting.')
    for key, (_, lo, hi, _) in OPTIMIZER_SETTINGS.items():
        value = opts.get(key)
        if value is None:
            err(f'training.optimizer_settings.{key}', 'Missing required field.')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            err(f'training.optimizer_settings.{key}', f'Must be a finite number in [{lo}, {hi}].')
    if m['source'] == 'huggingface':
        entry = HF_MODELS.get(m['hf_model_id'])
        if entry is None:
            err('model.hf_model_id', f"Unsupported model; supported: {', '.join(HF_MODELS)}.")
        else:
            if entry['architecture'] != config['architecture']:
                err('model.hf_model_id', 'Model is not compatible with the selected architecture.')
            if m['hf_revision'] != entry['revision']:
                err('model.hf_revision', f"Unsupported revision; allowed: {entry['revision']!r}.")
            for key, value in entry['dimensions'].items():
                if m[key] != value:
                    err(f'model.{key}', f'Must be {value} for {m["hf_model_id"]}; use source "custom" for other dimensions.')
    elif m['hf_model_id'] is not None or m['hf_revision'] is not None:
        err('model.hf_model_id', 'Custom models must not reference a Hugging Face ID or revision.')
    for key in ('num_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size', 'vocab_size', 'max_position_embeddings'):
        if m[key] is not None and m[key] > lim[key]:
            err(f'model.{key}', f'Exceeds the MVP limit of {lim[key]}.')
    if m['hidden_size'] % m['num_attention_heads']:
        err('model.num_attention_heads', 'Must divide hidden_size.')
    S, B, G = inp['sequence_length'], t['microbatch_size'], t['gradient_accumulation_steps']
    if S > m['max_position_embeddings']:
        err('input.sequence_length', f"Exceeds positional capacity ({m['max_position_embeddings']}).")
    for key, value, name in (('sequence_length', S, 'input.sequence_length'), ('microbatch_size', B, 'training.microbatch_size'),
                             ('gradient_accumulation_steps', G, 'training.gradient_accumulation_steps'),
                             ('warmup_updates', meas['warmup_updates'], 'measurement.warmup_updates'),
                             ('measured_updates', meas['measured_updates'], 'measurement.measured_updates')):
        if value > lim[key]:
            err(name, f'Exceeds the MVP limit of {lim[key]}.')
    if B * S > lim['max_tokens_per_microbatch']:
        err('training.microbatch_size', f"microbatch × sequence length exceeds {lim['max_tokens_per_microbatch']} tokens.")
    total = (meas['warmup_updates'] + meas['measured_updates']) * G
    if total > lim['max_total_microbatches']:
        err('measurement.measured_updates', f"(warmup + measured) × accumulation = {total} exceeds {lim['max_total_microbatches']} microbatches.")
    if meas['seed'] > 2 ** 31 - 1:
        err('measurement.seed', 'Must be below 2^31.')
    if not errors and parameter_count(m) > lim['max_parameters']:
        err('model', f"{parameter_count(m):,} parameters exceed the MVP limit of {lim['max_parameters']:,}.")
    if errors:
        raise ConfigError(errors)
    return json.loads(json.dumps(config))


def fingerprint(config):
    """Stable SHA-256 of the canonical configuration (requested_gpu excluded: it does not change the workload)."""
    workload = {k: v for k, v in config.items() if k != 'hardware'}
    return hashlib.sha256(json.dumps(workload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_model(config):
    """Construct the trusted model for a validated config (imports torch lazily)."""
    from models.decoder import GPT2Decoder
    return GPT2Decoder(config['model'])


def make_inputs(config, generator, device):
    """Synthetic token ids [B, S] generated once, outside the timed loop; labels = inputs (shifted in loss)."""
    import torch
    B, S, V = config['training']['microbatch_size'], config['input']['sequence_length'], config['model']['vocab_size']
    ids = torch.randint(0, V, (B, S), generator=generator, dtype=torch.long).to(device)
    return ids, ids


def loss_fn(logits, labels):
    """Causal LM cross-entropy in FP32 over all positions (next-token prediction)."""
    import torch.nn.functional as F
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
