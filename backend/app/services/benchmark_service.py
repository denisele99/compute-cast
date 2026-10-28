"""Benchmark workflow: local script generation, result import, Nebius availability.

Local: the backend only GENERATES ``run_benchmark.py`` (config embedded as JSON); the user
runs it on their own GPU. Importing a result parses JSON only. Nebius: no provisioning
integration exists, so the remote path reports itself unavailable (nothing is launched).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..clients.nebius import launch_gpu_job
from ..parsers import normalize_inputs as N

TEMPLATE = Path(__file__).resolve().parents[2] / 'scripts' / 'run_benchmark.py'
MAX_RESULT_BYTES = 200_000
FINGERPRINT_PREFIXES = ('model.', 'training.')


def fingerprint(values):
    relevant = {k: v for k, v in sorted(values.items()) if k.startswith(FINGERPRINT_PREFIXES) and v is not None}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def _shell_quote(name):
    return name if all(c.isalnum() or c in '._-/' for c in name) else "'" + name.replace("'", "'\\''") + "'"


def plan(analysis_id, values, files):
    """Script config, requirements and the CLI command matching the generated script."""
    model_files = [f for f in files if f['kind'] == 'model' and f['name'].endswith('.py')]
    train_cfg = [f for f in files if f['kind'] == 'training' and f['name'].endswith(('.json', '.yaml', '.yml'))]
    data = [f for f in files if f['kind'] == 'dataset']
    command = ['python', 'run_benchmark.py']
    if model_files:
        command += ['--model', _shell_quote(model_files[0]['name'])]
    if train_cfg:
        command += ['--train', _shell_quote(train_cfg[0]['name'])]
    if data:
        command += ['--data-sample', _shell_quote(data[0]['name'])]
    command += ['--output', 'benchmark_result.json']
    template = ['python', 'run_benchmark.py', '--model', 'MODEL_PATH', '--train', 'TRAINING_PATH',
                '--data-sample', 'DATA_PATH', '--output', 'OUTPUT_PATH']
    requirements = {
        'environment': ['Python 3.10+', 'PyTorch 2.1+ built with CUDA and an NVIDIA GPU',
                        'PyYAML (only for YAML --train configs)', 'Same Python dependencies your '
                        'model file imports (only when --model is used)'],
        'files': [
            {'flag': '--model', 'required': False, 'provided': [f['name'] for f in model_files],
             'description': 'Python file exposing build_model(config) -> nn.Module. Executes your code; '
                            'omit to benchmark the reference profile implementation.'},
            {'flag': '--train', 'required': False, 'provided': [f['name'] for f in train_cfg],
             'description': 'JSON/YAML overrides for the embedded training settings (training scripts are not run).'},
            {'flag': '--data-sample', 'required': False, 'provided': [f['name'] for f in data],
             'description': 'CSV/TSV/JSONL sample for batches; random tensors of the configured shapes otherwise.'},
        ],
        'credentials': ['None for local runs.'],
        'warnings': ['Local benchmarking executes code on your machine (PyTorch, and your model '
                     'file when --model is passed). Review files before running.',
                     'Runs on one GPU; DDP/ZeRO/FSDP communication and sharding are not reproduced.'],
    }
    return {'command': ' '.join(command), 'command_template': ' '.join(template),
            'requirements': requirements, 'fingerprint': fingerprint(values)}


def generate_script(analysis_id, values, prediction):
    model, training = N.model_settings(values), N.training_settings(values)
    training.pop('input_specs', None)
    a = N.assumptions(values)['benchmark']
    config = {'analysis_id': analysis_id, 'config_fingerprint': fingerprint(values),
              'profile': values['model.profile'], 'model': model, 'training': training,
              'benchmark': {'steps': a['steps'], 'warmup_steps': a['warmup_steps']},
              'predicted': {'peak_vram_gib': (prediction or {}).get('memory', {}).get('gpu', {}).get('peak_vram_gib')}}
    text = TEMPLATE.read_text(encoding='utf-8')
    payload = json.dumps(config, indent=1, sort_keys=True)
    if "'''" in payload:
        raise ValueError('Configuration cannot be embedded safely.')
    return text.replace('__COMPUTE_CAST_CONFIG__', payload)


def _num(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be a finite number >= {minimum}.')
    return value


def import_result(raw, analysis_id, values, prediction):
    """Validate a benchmark JSON and compare it with the current prediction."""
    if len(raw) > MAX_RESULT_BYTES:
        raise ValueError('Benchmark result exceeds 200 KB.')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError('Benchmark result is not valid JSON.') from exc
    if not isinstance(data, dict) or data.get('schema') != 'compute-cast-benchmark/1':
        raise ValueError('Not a Compute Cast benchmark result (schema compute-cast-benchmark/1).')
    step = data.get('seconds_per_optimizer_step')
    if not isinstance(step, dict):
        raise ValueError('seconds_per_optimizer_step is missing.')
    median = _num(step.get('median'), 'seconds_per_optimizer_step.median', 1e-12)
    peak = data.get('peak_memory_allocated_gib')
    if peak is not None:
        peak = _num(peak, 'peak_memory_allocated_gib')
    reserved = data.get('peak_memory_reserved_gib')
    if reserved is not None:
        reserved = _num(reserved, 'peak_memory_reserved_gib')
    issues = []
    if data.get('analysis_id') != analysis_id:
        issues.append('Generated for a different analysis.')
    current = fingerprint(values)
    if data.get('config_fingerprint') != current:
        issues.append('Generated for a different model/training configuration than the current one.')
    if not data.get('representative', False):
        issues.append('Marked not representative (CPU dry run or multi-GPU config on one GPU).')
    comparison = {}
    if prediction and prediction.get('status') == 'ok':
        predicted_peak = prediction['memory']['gpu'].get('peak_vram_gib')
        if predicted_peak and reserved is not None:
            comparison['peak_vram'] = {'predicted_gib': predicted_peak, 'measured_reserved_gib': reserved,
                                       'measured_allocated_gib': peak,
                                       'ratio_measured_to_predicted': reserved / predicted_peak}
        slow = (prediction['runtime'].get('scenarios') or {}).get('slow', {})
        fast = (prediction['runtime'].get('scenarios') or {}).get('fast', {})
        from .estimators.pricing import benchmark_seconds_per_step
        A = values.get('training.gradient_accumulation_steps') or 1
        ps, pf = benchmark_seconds_per_step(slow, A), benchmark_seconds_per_step(fast, A)
        if ps:
            comparison['seconds_per_optimizer_step'] = {'predicted_fast': pf, 'predicted_slow': ps,
                                                        'measured_median': median}
    return {'kind': 'benchmark', 'provenance': 'local benchmark result imported by user (measured)',
            'device': data.get('device'), 'torch_version': data.get('torch_version'),
            'model_source': data.get('model_source'), 'user_code_executed': data.get('user_code_executed'),
            'data_source': data.get('data_source'), 'parameter_count': data.get('parameter_count'),
            'settings': data.get('settings'), 'measured_steps': data.get('measured_steps'),
            'seconds_per_optimizer_step': step, 'peak_memory_allocated_gib': peak,
            'peak_memory_reserved_gib': reserved, 'issues': issues, 'notes': data.get('notes', []),
            'comparison': comparison, 'matches_current_config': data.get('config_fingerprint') == current,
            'finished_at': data.get('finished_at')}


NEBIUS_REMAINING_WORK = [
    'Nebius AI Cloud credentials (service account / IAM token) configured server-side.',
    'Choice of execution service (Compute instance vs. serverless/managed jobs) and container image with PyTorch/CUDA.',
    'Upload of the generated script, model file and data sample to the job (object storage or image).',
    'Provisioning, job submission, status polling, log/result retrieval and guaranteed teardown.',
    'Spend controls: explicit confirmation, max-duration/auto-stop and cost cap enforcement.',
]


def submit_benchmark(request):
    """Nebius execution. Not implemented: raises before any network call or provisioning."""
    try:
        job = launch_gpu_job(request)
    except NotImplementedError as exc:
        raise NotImplementedError('Nebius benchmark execution is not available: no provisioning '
                                  'integration or credentials are configured. No resources were '
                                  'created and no charges were incurred.') from exc
    return {'job_id': job['id'], 'status': 'submitted'}
