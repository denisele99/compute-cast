"""LEGACY - NOT USED BY THE GPT-2 MVP. Retained for future profiles/extensions; not imported by
active routes. Active MVP path: app/parsers/gpt2_inputs.py, app/parsers/gpt2_extract.py,
app/services/estimators/estimate_gpt2.py and gpt2_predict.py. Still covered by tests.

Orchestrator for the legacy profiles (gpu_memory + gpu_runtime + pricing).

Prediction orchestration: validated form values -> normalized bundle -> estimators.

Flow: validate -> collect_from_metadata (workload.json) -> hardware_json (hardware.json)
-> adapters for estimate_gpu_memory / estimate_runtime -> pricing. Entirely
metadata-based; no uploaded file is read or executed here.
"""
from __future__ import annotations

import copy

from ...parsers import normalize_inputs as N
from ...parsers.collect_workload import collect_from_metadata
from ...parsers.detect_hardware_lookup import form_values, get_option, options
from . import gpu_memory, gpu_runtime, pricing, profiles

RUNTIME_SCENARIO_KEYS = (
    'compute_arithmetic_seconds_per_microbatch', 'hbm_transfer_seconds_per_microbatch_proxy',
    'forward_backward_core_seconds_per_microbatch', 'data_supply_seconds_per_microbatch',
    'communication_seconds_per_microbatch', 'communication_seconds_per_update',
    'offload_seconds_per_microbatch', 'offload_seconds_per_update', 'optimizer_seconds_per_update',
    'gpu_core_only_forward_backward_seconds', 'training_seconds', 'training_hours', 'job_seconds',
    'job_hours', 'training_gpu_hours', 'job_gpu_hours', 'extra_job_overhead', 'missing')


def build_bundle(values):
    """Normalized estimator input bundle (plus workload.json and hardware.json)."""
    profile = values['model.profile']
    model, training = N.model_settings(values), N.training_settings(values)
    workload = collect_from_metadata(profile, model, training)
    return {'profile': profile, 'workload': workload, 'hardware': N.hardware_json(values),
            'dataset': N.dataset(values), 'deployment': N.deployment(values),
            'assumptions': N.assumptions(values), 'pricing': N.pricing(values)}


def memory_adapter(bundle):
    a = dict(bundle['assumptions']['memory'])
    a['activation_model'] = ('transformer_classic16' if bundle['profile'] == N.TX else 'simple_mlp')
    try:
        return gpu_memory.estimate_gpu_memory(bundle['workload'], a, bundle['hardware'],
                                              deployment=bundle['deployment'])
    except (ValueError, KeyError, TypeError) as exc:
        return {'status': 'error', 'gpu': {'peak_vram_gib': None}, 'errors': [str(exc)]}


def runtime_adapter(bundle):
    a = dict(bundle['assumptions']['runtime'])
    if bundle['profile'] == N.TX:
        a['runtime_model'] = 'transformer_dense'
    try:
        report = gpu_runtime.estimate_runtime(bundle['workload'], bundle['dataset'], bundle['hardware'],
                                              a, bundle['deployment'])
    except (ValueError, KeyError, TypeError) as exc:
        return {'status': 'error', 'missing_information': [str(exc)], 'scenarios': {}}
    scenarios = {label: {k: s.get(k) for k in RUNTIME_SCENARIO_KEYS}
                 for label, s in report.get('scenarios', {}).items()}
    keep = ('status', 'missing_information', 'assumptions', 'schedule', 'computation',
            'optimizer_work', 'compute_peak_key', 'selected_gpu_id', 'workload_fingerprint',
            'runtime_settings', 'scope', 'scenario_interpretation')
    return {**{k: report.get(k) for k in keep}, 'scenarios': scenarios}


def cost_adapter(bundle, runtime):
    out = {}
    for label, s in runtime.get('scenarios', {}).items():
        out[label] = pricing.estimate_cost(s.get('job_hours'), bundle['pricing'])
    if not out:
        out['unavailable'] = pricing.estimate_cost(None, bundle['pricing'])
    accumulation = bundle['workload'].get('training', {}).get('gradient_accumulation_steps', 1)
    bench = pricing.estimate_benchmark_cost(runtime, bundle['pricing'],
                                            bundle['assumptions']['benchmark'], accumulation)
    return out, bench


def _fit(memory):
    gpu = memory.get('gpu', {})
    peak, usable, total = gpu.get('peak_vram_gib'), gpu.get('usable_vram_gib'), gpu.get('total_vram_gib')
    if peak is None or not usable:
        return None
    return {'peak_vram_gib': peak, 'usable_vram_gib': usable, 'total_vram_gib': total,
            'percent_of_usable': 100 * peak / usable,
            'percent_of_total': 100 * peak / total if total else None,
            'fits': gpu.get('fits'), 'denominator': 'usable VRAM limit per GPU'}


def explanation(bundle, memory, runtime, defaults_used):
    profile = bundle['profile']
    pf = ['mlp_parameters', 'mlp_activations'] if profile == N.MLP else ['transformer_parameters', 'transformer_activations']
    mem_formulas = ['model_states', 'optimizer_states', 'master_weights', 'autocast_cache',
                    'mlp_saved' if profile == N.MLP else 'transformer_saved', 'auxiliary',
                    'optimizer_workspace', 'kernel_workspace', 'planning_total', 'capacity', 'state_sharding']
    rt_formulas = ['schedule', 'gemm', 'mlp' if profile == N.MLP else 'transformer', 'traffic',
                   'roofline', 'optimizer', 'communication', 'io', 'overlap', 'total']
    return {
        'profile': {'id': profile, **profiles.PROFILES[profile]},
        'parameter_and_activation_formulas': {k: profiles.FORMULAS[k] for k in pf},
        'memory_formulas': {k: gpu_memory.FORMULAS[k] for k in mem_formulas},
        'runtime_formulas': {k: gpu_runtime.FORMULAS[k] for k in rt_formulas},
        'cost_formulas': pricing.FORMULAS,
        'memory_planning_assumptions': memory.get('planning_assumptions'),
        'memory_notes': memory.get('assumption_notes', []),
        'runtime_assumptions': runtime.get('assumptions') or [],
        'runtime_defaults': gpu_runtime.DEFAULTS,
        'defaults_used': defaults_used,
        'limitations': [
            'Planning estimates from metadata and documented formulas, not measurements.',
            'Workspace, runtime-overhead and fragmentation allowances and runtime efficiencies are '
            'unsourced, editable planning assumptions.',
            'Peak VRAM excludes evaluation, checkpoint saving and initialization peaks.',
            'Runtime "fast"/"slow" are scenarios, not confidence bounds.',
            'Costs exclude storage, egress, taxes and queue time; preemptible jobs may be interrupted.',
        ],
    }


def predict(values, provenance=None):
    """Validate and predict. Returns blocked/validation details instead of fabricated numbers."""
    check = N.validate(values, provenance)
    if not check['can_estimate']['memory']:
        return {'status': 'blocked', 'validation': check}
    v = check['values']
    bundle = build_bundle(v)
    if bundle['workload']['status'] not in ('complete', 'incomplete'):
        return {'status': 'blocked', 'validation': check, 'workload': bundle['workload']}
    memory = memory_adapter(bundle)
    runtime = runtime_adapter(bundle)
    for item in check['missing']:
        if 'runtime' in item['required_for']:
            msg = f"Missing input: {item['label']} ({item['field']})."
            if msg not in (runtime.get('missing_information') or []):
                runtime.setdefault('missing_information', []).append(msg)
    costs, bench = cost_adapter(bundle, runtime)
    defaults_used = sorted(k for k, meta in (provenance or {}).items()
                           if isinstance(meta, dict) and meta.get('source') == 'default' and v.get(k) is not None)
    return {
        'status': 'ok',
        'kind': 'prediction',
        'provenance': 'metadata-only prediction (no code executed, no benchmark data used)',
        'memory': memory, 'fit': _fit(memory), 'runtime': runtime, 'cost': costs,
        'benchmark_cost': bench,
        'parameters': bundle['workload']['model'].get('parameter_summary'),
        'profile_estimates': bundle['workload'].get('profile_estimates'),
        'explanation': explanation(bundle, memory, runtime, defaults_used),
        'validation': check,
        'bundle': {k: bundle[k] for k in ('hardware', 'dataset', 'deployment', 'assumptions', 'pricing')},
        'workload': bundle['workload'],
    }


def compare(values, provenance=None, option_ids=None, limit=10):
    """Same workload on other catalogue hardware: swap only hardware identity/specs/price."""
    check = N.validate(values, provenance)
    if not check['can_estimate']['memory']:
        return {'status': 'blocked', 'validation': check}
    v = check['values']
    ratio = (v['hardware.usable_vram_gib'] / v['hardware.total_vram_gib']) if v['hardware.total_vram_gib'] else 0.9
    rows = [get_option(i) for i in option_ids] if option_ids else [
        r for r in options() if r['provider'] == 'nebius' and r['vram_gib']]
    out = []
    for option in [r for r in rows if r][:limit]:
        swapped = copy.deepcopy(v)
        for key in ('hardware.catalog_gpu_id', 'hardware.peak_tflops_fp32', 'hardware.peak_tflops_bf16',
                    'hardware.peak_tflops_fp16', 'hardware.memory_bandwidth_gb_s', 'hardware.price_per_hour',
                    'hardware.price_unit', 'hardware.price_region', 'hardware.price_note', 'hardware.gpus_per_instance'):
            swapped[key] = None
        fill, _ = form_values(option)
        swapped.update(fill)
        if swapped.get('hardware.total_vram_gib'):
            swapped['hardware.usable_vram_gib'] = swapped['hardware.total_vram_gib'] * ratio
        result = predict(swapped)
        row = {'id': option['id'], 'label': option['label'], 'region': (option.get('price') or {}).get('region'),
               'issues': option['issues'], 'status': result['status']}
        if result['status'] == 'ok':
            slow, fast = (result['runtime'].get('scenarios') or {}).get('slow', {}), \
                (result['runtime'].get('scenarios') or {}).get('fast', {})
            row.update(fit=result['fit'], memory_status=result['memory'].get('status'),
                       runtime_status=result['runtime'].get('status'),
                       job_hours={'fast': fast.get('job_hours'), 'slow': slow.get('job_hours')},
                       cost={k: c.get('cost') for k, c in result['cost'].items()},
                       price=(option.get('price') or None),
                       runtime_missing=(result['runtime'].get('missing_information') or [])[:3])
        else:
            row['reason'] = [m['label'] for m in result['validation']['missing']][:5]
        out.append(row)
    return {'status': 'ok', 'rows': out, 'usable_fraction': ratio,
            'note': 'Same workload, GPU count, strategy and system bandwidths; only GPU specs, VRAM '
                    'and price change. The usable VRAM fraction from your settings is reused.'}
