"""GPT-2 MVP prediction: validated inputs -> estimate_gpt2 -> fit, runtime, cost (active).

Purely analytical: nothing is run, downloaded or executed. Memory is always estimated when the
memory inputs are valid; runtime and cost need catalogue (or custom) GPU ceilings and a price.
"""
from __future__ import annotations

import copy

from ...parsers import gpt2_inputs as G
from ...parsers.detect_hardware_lookup import get_option, options
from . import estimate_gpt2 as E
from . import pricing

GIB = 2 ** 30


def hardware_values(option):
    """Autofill values (MVP keys) for one catalogue/Nebius search result."""
    specs = option.get('specs') or {}
    values = {'hardware.source': 'nebius' if option['provider'] == 'nebius' else 'catalog',
              'hardware.gpu_id': option['id'], 'hardware.gpu_name': option['label'],
              'hardware.catalog_gpu_id': specs.get('catalog_gpu_id'),
              'hardware.bf16_dense_tflops': specs.get('peak_tflops_bf16'), 'hardware.fp32_tflops': specs.get('peak_tflops_fp32'),
              'hardware.memory_bandwidth_gbps': specs.get('memory_bandwidth_gb_s'),
              'hardware.bf16_supported': True if specs.get('peak_tflops_bf16') else None,
              'hardware.price_per_hour': None, 'hardware.price_unit': None, 'hardware.price_region': None,
              'hardware.price_note': None}
    if option.get('vram_gib') is not None:
        values['hardware.vram_gib'] = float(option['vram_gib'])
        values['hardware.usable_vram_gib'] = float(option['vram_gib'])
    price = option.get('price')
    if price and price['value'] is not None and price['unit']:
        one_gpu = [p for p in option.get('presets', []) if p.get('gpu_count') == 1]
        values.update({'hardware.price_per_hour': price['value'], 'hardware.price_unit': price['unit'],
                       'hardware.price_region': price['region'],
                       'hardware.price_note': ('Preemptible/spot ' if price['preemptible'] else '')
                       + f"{price['currency']} per {price['unit'].replace('_', '-')}, effective {price['effective_time']}"
                       + ('' if one_gpu or not option.get('presets') else '; no 1-GPU preset listed')})
    return values


def _gib(n):
    return n / GIB


def predict(values, provenance=None, extraction=None, acknowledged=None):
    check = G.validate(values, provenance, extraction, acknowledged)
    if not check['can_estimate']['memory']:
        return {'status': 'blocked', 'validation': check}
    v = check['values']
    model, training, hardware = G.gpt2_model(v), G.gpt2_training(v), G.gpt2_hardware(v)
    policy = dict(E.DEFAULT_POLICY)
    memory = E.memory_estimate(model, training, policy)
    mb = memory['memory_bytes']
    usable = v['hardware.usable_vram_gib']
    capacity = mb['capacity_with_runtime_and_margin']
    fit = {'capacity_gib': _gib(capacity), 'allocated_peak_gib': _gib(mb['estimated_allocated_peak']),
           'usable_vram_gib': usable, 'total_vram_gib': v['hardware.vram_gib'],
           'percent_of_usable': 100 * _gib(capacity) / usable, 'fits': capacity <= usable * GIB,
           'denominator': 'usable VRAM limit per GPU',
           'rule': '(allocated peak + runtime baseline) × (1 + margin) ≤ usable VRAM limit'}
    runtime, runtime_error = None, None
    if check['can_estimate']['runtime']:
        try:
            runtime = E.runtime_estimate(model, training, hardware, policy)
        except ValueError as exc:
            runtime_error = str(exc)
    missing_runtime = [m['label'] for m in check['missing'] if 'runtime' in m['required_for']]
    cost = None
    if runtime:
        hours = runtime['total_training_compute']['seconds'] / 3600
        cost = pricing.estimate_cost(hours, G.pricing(v))
    else:
        cost = {'status': 'unavailable', 'reason': 'Runtime is unavailable, so cost cannot be estimated.'}
    from ..benchmark_service import benchmark_cost
    partial = {'runtime': runtime}
    return {
        'benchmark_cost': benchmark_cost(partial, v),
        'status': 'ok', 'kind': 'prediction', 'estimator': 'estimate_gpt2 (uncalibrated analytical estimate)',
        'provenance': 'metadata-only analytical estimate; no code executed, nothing downloaded or benchmarked',
        'hypothetical': check['hypothetical'], 'validation': check,
        'parameters': memory['parameters'], 'head_dim': memory['head_dim'],
        'effective_batch_size': memory['effective_batch_size'],
        'resolved_intermediate_size': memory['resolved_intermediate_size'],
        'memory': {k: _gib(b) for k, b in mb.items()}, 'fit': fit,
        'runtime': runtime, 'runtime_error': runtime_error, 'runtime_missing': missing_runtime,
        'cost': cost,
        'explanation': {'formulas': E.FORMULAS, 'policy': policy, 'fixed_assumptions': G.fixed_assumptions(),
                        'scope': 'GPT-2 full training; FP32 state; BF16 autocast; FP32 math attention; one GPU',
                        'limitations': ['Uncalibrated analytical estimate: memory coefficients, traffic and '
                                        'efficiencies are heuristics, not measurements or confidence bounds.',
                                        'Excludes data loading, evaluation, checkpoint saving and startup time.',
                                        'Timing is hypothetical if the memory estimate exceeds the usable limit.',
                                        'Epoch mode flushes partial accumulation at each epoch end; drop_last=false.',
                                        'Cost excludes storage, egress, taxes and queue time; preemptible jobs may be interrupted.']},
        'inputs': {'model': model, 'training': training, 'hardware': hardware, 'estimator_policy': policy},
    }


def compare(values, provenance=None, extraction=None, acknowledged=None, limit=10):
    """Same workload on each Nebius GPU with known VRAM; only hardware identity/specs/price change."""
    check = G.validate(values, provenance, extraction, acknowledged)
    if not check['can_estimate']['memory']:
        return {'status': 'blocked', 'validation': check}
    ratio = check['values']['hardware.usable_vram_gib'] / check['values']['hardware.vram_gib']
    rows = []
    for option in [r for r in options() if r['provider'] == 'nebius' and r['vram_gib']][:limit]:
        swapped = copy.deepcopy(check['values'])
        swapped.update(hardware_values(option))
        swapped['hardware.usable_vram_gib'] = swapped['hardware.vram_gib'] * ratio
        result = predict(swapped, provenance, extraction, acknowledged)
        row = {'id': option['id'], 'label': option['label'], 'region': (option.get('price') or {}).get('region'),
               'issues': option['issues'], 'price': option.get('price'), 'status': result['status']}
        if result['status'] == 'ok':
            row.update(fit=result['fit'], hours=(result['runtime'] or {}).get('total_training_compute', {}).get('seconds', 0) / 3600
                       if result['runtime'] else None, cost=result['cost'].get('cost'),
                       runtime_missing=result['runtime_missing'][:2])
        rows.append(row)
    return {'status': 'ok', 'rows': rows, 'usable_fraction': ratio,
            'note': 'Same GPT-2 workload; only GPU VRAM, ceilings and price change. Your usable-VRAM fraction is reused.'}


__all__ = ['predict', 'compare', 'hardware_values', 'get_option']
