"""Job and benchmark cost from ELAPSED job hours and an explicit price unit.

Elapsed hours already include startup/setup (the runtime estimator adds it once), so no
setup time is added here. GPU count is applied exactly once, according to the price unit:

  gpu_hour:      cost = elapsed_hours * gpus_allocated * price
  instance_hour: cost = elapsed_hours * ceil(gpus_allocated / gpus_per_instance) * price

``gpus_allocated`` is the number of GPUs rented (billed for the whole job, idle or not);
``gpus_used`` is the number the training configuration uses. Storage, egress, taxes and
queue time are excluded.
"""
from __future__ import annotations

import math

FORMULAS = {
    'cost_gpu_hour': 'cost = elapsed_job_hours × GPUs_allocated × price_per_GPU_hour',
    'cost_instance_hour': 'cost = elapsed_job_hours × ceil(GPUs_allocated / GPUs_per_instance) × price_per_instance_hour',
    'benchmark': 'elapsed = setup_minutes/60 + (warmup + measured steps) × seconds_per_optimizer_step / 3600',
}


def estimate_cost(elapsed_hours, pricing):
    """Return a cost dict, or {'status': 'unavailable', 'reason': ...} when inputs are missing."""
    price, unit = pricing.get('price_per_hour'), pricing.get('unit')
    allocated = pricing.get('gpus_allocated') or 1
    used = pricing.get('gpus_used') or 1
    if elapsed_hours is None:
        return {'status': 'unavailable', 'reason': 'Elapsed job time is not available.'}
    if price is None or unit is None:
        return {'status': 'unavailable', 'reason': 'Enter a price and its unit (per GPU-hour or per instance-hour).'}
    result = {'status': 'estimated', 'currency': pricing.get('currency', 'USD'), 'unit': unit,
              'price_per_hour': price, 'elapsed_hours': elapsed_hours,
              'gpus_allocated': allocated, 'gpus_used': used,
              'gpu_hours_used': elapsed_hours * used, 'gpu_hours_billed': elapsed_hours * allocated,
              'region': pricing.get('region'), 'note': pricing.get('note')}
    if unit == 'gpu_hour':
        result['cost'] = elapsed_hours * allocated * price
        result['formula'] = FORMULAS['cost_gpu_hour']
    elif unit == 'instance_hour':
        per = pricing.get('gpus_per_instance')
        if not per:
            return {'status': 'unavailable', 'reason': 'Per-instance pricing needs GPUs per instance.'}
        instances = math.ceil(allocated / per)
        result.update(instances=instances, instance_hours=elapsed_hours * instances,
                      cost=elapsed_hours * instances * price, formula=FORMULAS['cost_instance_hour'])
    else:
        return {'status': 'unavailable', 'reason': f'Unknown price unit {unit!r}.'}
    if allocated > used:
        result['warning'] = (f'{allocated - used} allocated GPU(s) are idle but still billed.')
    return result


def benchmark_seconds_per_step(scenario, accumulation):  # LEGACY: not used by the GPT-2 MVP
    """Optimizer-step seconds for a benchmark run on synthetic/sample data held in memory.

    Uses forward/backward core time, communication and offload per microbatch and the
    optimizer update, but not dataset disk loading (benchmarks use a small sample).
    """
    keys = ('forward_backward_core_seconds_per_microbatch', 'communication_seconds_per_microbatch',
            'offload_seconds_per_microbatch', 'optimizer_seconds_per_update',
            'communication_seconds_per_update', 'offload_seconds_per_update')
    core, comm_m, off_m, update, comm_u, off_u = (scenario.get(k) for k in keys)
    if None in (core, comm_m, off_m, update, comm_u, off_u):
        return None
    return accumulation * (core + comm_m + off_m) + update + comm_u + off_u


def estimate_benchmark_cost(runtime, pricing, benchmark, accumulation):  # LEGACY: not used by the GPT-2 MVP
    scenario = (runtime or {}).get('scenarios', {}).get('slow')
    if not scenario:
        return {'status': 'unavailable', 'reason': 'Runtime prediction is unavailable, so benchmark '
                'duration cannot be estimated.'}
    step = benchmark_seconds_per_step(scenario, accumulation)
    if step is None:
        return {'status': 'unavailable', 'reason': 'Per-step time is unavailable (see runtime missing information).'}
    steps = benchmark['steps'] + benchmark['warmup_steps']
    elapsed = benchmark['setup_minutes'] / 60 + steps * step / 3600
    cost = estimate_cost(elapsed, pricing)
    return {**cost, 'steps': benchmark['steps'], 'warmup_steps': benchmark['warmup_steps'],
            'setup_minutes': benchmark['setup_minutes'], 'seconds_per_optimizer_step': step,
            'scenario': 'slow', 'formula_benchmark': FORMULAS['benchmark'],
            'assumption': 'Slow runtime scenario; benchmark data is a small in-memory sample, so '
                          'disk loading is excluded; setup time is an editable assumption.'}
