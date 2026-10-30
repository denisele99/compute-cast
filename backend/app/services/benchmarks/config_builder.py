"""Workload form values -> validated benchmark_config.json, with provenance for the review screen.

Provenance per config field:
  * ``detected`` - extracted from an uploaded file / the selected Hugging Face config;
  * ``user``     - entered in the workload form, or edited on the benchmark review screen (override);
  * ``default``  - a benchmark default the user has not changed (e.g. warm-up/measured updates);
  * ``assumption`` - fixed by the trusted runner profile (synthetic inputs, random init, BF16 autocast...).

Overrides (dotted paths from the review screen) are re-applied on every regeneration; an override that
is no longer valid for the current workload is dropped and reported rather than silently kept.
Policy (credentials, allowances, runtime limits, image) is never part of the config.
"""
from __future__ import annotations

import copy

from ...parsers import gpt2_inputs as G
from ...parsers.detect_hardware_lookup import get_option, options
from ..estimators import estimate_gpt2 as E
from .registry import registry

GIB = 2 ** 30
EDITABLE = ('measurement.warmup_updates', 'measurement.measured_updates', 'measurement.seed',
            'training.microbatch_size', 'training.gradient_accumulation_steps', 'input.sequence_length',
            'training.optimizer_settings.lr', 'training.optimizer_settings.beta1', 'training.optimizer_settings.beta2',
            'training.optimizer_settings.eps', 'training.optimizer_settings.weight_decay')
DEFAULT_MEASUREMENT = {'warmup_updates': 3, 'measured_updates': 10, 'seed': 0}
FORM_KEYS = {'model.num_layers': 'num_layers', 'model.hidden_size': 'hidden_size',
             'model.num_attention_heads': 'num_attention_heads', 'model.intermediate_size': 'intermediate_size',
             'model.vocab_size': 'vocab_size', 'model.max_position_embeddings': 'max_position_embeddings',
             'model.attention_dropout': 'attention_dropout', 'model.embedding_dropout': 'embedding_dropout',
             'model.residual_dropout': 'residual_dropout'}
ASSUMPTION_NOTES = {
    'input.source': 'Synthetic random token ids: excludes real data loading/tokenisation and says nothing about model quality.',
    'model.weights': 'Random initialisation (pretrained weights are not loaded in this version); speed and memory only.',
    'model.hf_revision': 'Not pinned: no Hugging Face files are downloaded by the runner, the architecture comes from the trusted registry.',
    'training.precision': 'BF16 autocast with FP32 parameters, gradients and AdamW states (matches the estimator).',
    'training.attention': 'Math attention with FP32 scores/softmax (matches the estimator).',
    'training.optimizer': 'torch.optim.AdamW(foreach=False, fused=False).', 'training.mode': 'Full training of all parameters.',
    'hardware.gpu_count': 'One GPU per job; multi-GPU is rejected.',
}


def _get(d, path):
    for part in path.split('.'):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def _set(d, path, value):
    *head, last = path.split('.')
    for part in head:
        d = d[part]
    d[last] = value


def _form_provenance(provenance, key, extracted=None, value=None):
    """Form provenance entries are {source, status, ...}; 'manual' means typed by the user."""
    entry = (provenance or {}).get(key)
    source = entry.get('source') if isinstance(entry, dict) else entry
    if source is None:
        if extracted and key in extracted and extracted[key] == value:
            return 'detected'
        return 'user'
    if source in ('manual', 'user'):
        return 'user'
    return 'default' if source == 'default' else 'detected'


def base_config(values, provenance, hf_model_id, extracted=None):
    """Config from validated form values, plus per-field provenance and missing inputs."""
    R = registry()
    custom = bool(values.get('model.custom_config')) or not hf_model_id
    prov, missing = {}, []
    model = {'source': 'custom' if custom else 'huggingface', 'hf_model_id': None if custom else hf_model_id,
             'hf_revision': None, 'weights': 'random_init'}
    prov['model.source'] = 'user' if values.get('model.custom_config') else 'detected'
    prov['model.hf_model_id'] = 'detected' if not custom else 'assumption'
    for key, name in FORM_KEYS.items():
        model[name] = values.get(key)
        prov[f'model.{name}'] = _form_provenance(provenance, key, extracted, values.get(key))
        if name == 'intermediate_size' and values.get(key) is None:
            prov[f'model.{name}'] = 'default'  # null = 4 x hidden size (GPT-2 default)
        if values.get(key) is None and name != 'intermediate_size':
            missing.append({'field': f'model.{name}', 'message': f"{G.FIELD_INDEX[key]['label']} is required."})
    config = {
        'schema_version': R.CONFIG_SCHEMA_VERSION, 'architecture': 'gpt2_decoder', 'model': model,
        'input': {'source': 'synthetic', 'sequence_length': values.get('data.sequence_length')},
        'training': {'mode': 'full', 'precision': 'bf16_autocast', 'optimizer': 'adamw',
                     'optimizer_settings': {k: spec[3] for k, spec in R.OPTIMIZER_SETTINGS.items()},
                     'microbatch_size': values.get('training.microbatch_size'),
                     'gradient_accumulation_steps': values.get('training.gradient_accumulation_steps'),
                     'attention': 'math_fp32'},
        'measurement': dict(DEFAULT_MEASUREMENT),
        'hardware': {'requested_gpu': values.get('hardware.gpu_name'), 'gpu_count': 1},
    }
    for path, key in (('input.sequence_length', 'data.sequence_length'), ('training.microbatch_size', 'training.microbatch_size'),
                      ('training.gradient_accumulation_steps', 'training.gradient_accumulation_steps'),
                      ('hardware.requested_gpu', 'hardware.gpu_name')):
        prov[path] = _form_provenance(provenance, key, extracted, values.get(key))
        if _get(config, path) is None and path != 'hardware.requested_gpu':
            missing.append({'field': path, 'message': f"{G.FIELD_INDEX[key]['label']} is required."})
    for path in ASSUMPTION_NOTES:
        prov[path] = 'assumption'
    for path in ('measurement.warmup_updates', 'measurement.measured_updates', 'measurement.seed'):
        prov[path] = 'default'
    for k in R.OPTIMIZER_SETTINGS:
        prov[f'training.optimizer_settings.{k}'] = 'default'
    if config['hardware']['requested_gpu'] is None:
        prov['hardware.requested_gpu'] = 'default'
    return config, prov, missing


def _error_set(R, config):
    try:
        R.validate(config)
        return set()
    except R.ConfigError as exc:
        return {(e['field'], e['message']) for e in exc.errors}


def apply_overrides(config, prov, overrides):
    """Apply review-screen edits; keep each only while it introduces no new validation error."""
    R = registry()
    applied, dropped = {}, []
    for path, value in (overrides or {}).items():
        if path not in EDITABLE:
            dropped.append({'field': path, 'value': value, 'reason': 'Not editable on the benchmark review screen.'})
            continue
        trial = copy.deepcopy(config)
        _set(trial, path, value)
        new = _error_set(R, trial) - _error_set(R, config)
        if new:
            dropped.append({'field': path, 'value': value, 'reason': sorted(new)[0][1]})
            continue
        config = trial
        prov[path] = 'user'
        applied[path] = value
    return config, applied, dropped


def memory_check(config, vram_gib, safety_fraction):
    """estimate_gpt2 memory for the benchmark shapes against VRAM x (1 - safety margin)."""
    m, t = config['model'], config['training']
    model = {'model_type': 'gpt2', 'n_layer': m['num_layers'], 'n_embd': m['hidden_size'], 'n_head': m['num_attention_heads'],
             'n_inner': m['intermediate_size'], 'vocab_size': m['vocab_size'], 'n_positions': m['max_position_embeddings'],
             'attn_pdrop': m['attention_dropout'], 'embd_pdrop': m['embedding_dropout'], 'resid_pdrop': m['residual_dropout'],
             'activation_function': 'gelu_new', 'tie_word_embeddings': True, 'scale_attn_weights': True}
    training = {'microbatch_size': t['microbatch_size'], 'sequence_length': config['input']['sequence_length'],
                'gradient_accumulation_steps': t['gradient_accumulation_steps']}
    mb = E.memory_estimate(model, training, dict(E.DEFAULT_POLICY))['memory_bytes']
    out = {'estimated_allocated_peak_gib': mb['estimated_allocated_peak'] / GIB,
           'estimated_capacity_gib': mb['capacity_with_runtime_and_margin'] / GIB, 'safety_fraction': safety_fraction,
           'vram_gib': vram_gib, 'limit_gib': None, 'fits': None,
           'rule': 'estimate_gpt2 capacity (allocated peak + runtime baseline, with margin) <= VRAM x (1 - safety fraction)'}
    if vram_gib:
        out['limit_gib'] = vram_gib * (1 - safety_fraction)
        out['fits'] = out['estimated_capacity_gib'] <= out['limit_gib']
    return out, model, training


def predicted_update_seconds(model, training, values):
    hardware = G.gpt2_hardware(values)
    try:
        return E.runtime_estimate(model, {**training, 'optimizer_updates': 1}, hardware, dict(E.DEFAULT_POLICY))['seconds_per_full_optimizer_update']
    except (ValueError, KeyError, TypeError):
        return None


def predict_for_config(config, values, safety_fraction=0.10):
    """Uncalibrated estimate_gpt2 prediction for exactly this (possibly imported) configuration."""
    memory, model, training = memory_check(config, values.get('hardware.vram_gib'), safety_fraction)
    seconds = predicted_update_seconds(model, training, values)
    meas = config['measurement']
    return {'seconds_per_update': seconds,
            'benchmark_compute_seconds': None if seconds is None else seconds * (meas['warmup_updates'] + meas['measured_updates']),
            'estimated_allocated_peak_gib': memory['estimated_allocated_peak_gib'], 'memory': memory,
            'gpu_label': values.get('hardware.gpu_name'),
            'note': 'Uncalibrated estimate_gpt2 prediction for exactly this benchmark configuration.'}


def benchmark_cost(prediction, values):
    """Estimated compute cost of the default benchmark (warm-up + measured updates); start-up excluded."""
    from ..estimators.pricing import estimate_cost
    runtime = (prediction or {}).get('runtime')
    if not runtime:
        return {'status': 'unavailable', 'reason': 'Runtime is unavailable, so the benchmark duration cannot be estimated.'}
    updates = DEFAULT_MEASUREMENT['warmup_updates'] + DEFAULT_MEASUREMENT['measured_updates']
    hours = updates * runtime['seconds_per_full_optimizer_update'] / 3600
    return {**estimate_cost(hours, G.pricing(values)), 'steps': DEFAULT_MEASUREMENT['measured_updates'],
            'warmup_steps': DEFAULT_MEASUREMENT['warmup_updates'],
            'seconds_per_optimizer_step': runtime['seconds_per_full_optimizer_update'],
            'assumption': 'Estimated compute time only; provisioning, image pull and start-up are excluded. Not a bill.'}


def nebius_hardware(gpu_id, policy, capacity_gib=None):
    """Map the analysis GPU to an allowed 1-GPU Nebius platform/preset, or offer alternatives."""
    allowed = policy.allowed_platforms
    rows = {}
    for o in options():
        if o['provider'] == 'nebius' and o.get('platform') in allowed and o.get('vram_gib'):
            rows.setdefault(o['platform'], o)
    alternatives = [{'gpu_id': o['id'], 'platform': p, 'preset': allowed[p], 'label': o['label'], 'vram_gib': o['vram_gib'],
                     'fits': None if capacity_gib is None else capacity_gib <= o['vram_gib'] * (1 - policy.memory_safety_fraction)}
                    for p, o in sorted(rows.items(), key=lambda kv: kv[1]['vram_gib'])]
    option = get_option(gpu_id) if gpu_id else None
    if option is None or option['provider'] != 'nebius':
        return {'valid': False, 'reason': 'Select a Nebius GPU for cloud benchmarks (catalogue/custom GPUs cannot be submitted).',
                'alternatives': alternatives}
    platform = option.get('platform')
    preset = allowed.get(platform)
    presets = [p['name'] for p in option.get('presets', []) if p.get('gpu_count') == 1]
    if preset is None:
        return {'valid': False, 'reason': f'{option["label"]} ({platform}) is not enabled for benchmark jobs on this server.',
                'alternatives': alternatives}
    if presets and preset not in presets:
        return {'valid': False, 'reason': f'Configured preset {preset} is not a listed 1-GPU preset of {platform}.',
                'alternatives': alternatives}
    vram = option.get('vram_gib') or (rows.get(platform) or {}).get('vram_gib')
    return {'valid': True, 'gpu_id': option['id'], 'platform': platform, 'preset': preset, 'label': option['label'],
            'vram_gib': vram, 'price': option.get('price'), 'alternatives': alternatives}


def build(values, provenance, hf_model_id, overrides, policy, target='local', extracted=None):
    """Return the review payload: config (or None), provenance, missing fields, checks and estimates."""
    R = registry()
    check = G.validate(values, provenance)
    v = check['values']
    config, prov, missing = base_config(v, provenance, hf_model_id, extracted)
    result = {'target': target, 'config': None, 'draft': config, 'provenance': prov, 'missing': missing,
              'assumption_notes': ASSUMPTION_NOTES, 'editable': list(EDITABLE), 'applied_overrides': {}, 'dropped_overrides': [],
              'errors': [], 'memory': None, 'hardware': None, 'fingerprint': None, 'ready': False,
              'limits': R.LIMITS, 'notes': ['Synthetic inputs by default: the benchmark measures speed and memory of complete '
                                            'optimizer updates, not data loading or model quality.']}
    if missing:
        return result
    config, result['applied_overrides'], result['dropped_overrides'] = apply_overrides(config, prov, overrides)
    result['draft'] = config
    try:
        config = R.validate(config)
    except R.ConfigError as exc:
        result['errors'] = exc.errors
        return result
    hardware = None
    vram = v.get('hardware.vram_gib')
    if target == 'nebius':
        hardware = nebius_hardware(v.get('hardware.gpu_id'), policy)
        vram = hardware.get('vram_gib') if hardware['valid'] else None
    memory, model, training = memory_check(config, vram, policy.memory_safety_fraction)
    if hardware is not None:
        hardware = nebius_hardware(v.get('hardware.gpu_id'), policy, memory['estimated_capacity_gib'])
    result.update(config=config, memory=memory, hardware=hardware, fingerprint=R.fingerprint(config),
                  parameter_count=R.parameter_count(config['model']))
    result['prediction'] = predict_for_config(config, v, policy.memory_safety_fraction)
    blockers, warnings = [], []
    if memory['fits'] is False:
        (blockers if target == 'nebius' else warnings).append(f"Estimated {memory['estimated_capacity_gib']:.1f} GiB exceeds the {memory['limit_gib']:.1f} GiB limit "
                        f"({memory['vram_gib']:.0f} GiB VRAM minus {policy.memory_safety_fraction:.0%} safety margin).")
    if target == 'nebius' and not hardware['valid']:
        blockers.append(hardware['reason'])
    elif target == 'nebius' and memory['fits'] is None:
        blockers.append('GPU memory of the selected Nebius platform is unknown, so the memory check cannot pass.')
    if target == 'local':
        warnings.append('Your local GPU is detected only when the package runs; the memory check uses the workload form VRAM.')
    result['blockers'], result['warnings'] = blockers, warnings
    result['ready'] = not blockers
    return result
