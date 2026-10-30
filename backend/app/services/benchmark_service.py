"""Benchmark workflow: local script generation, result import, Nebius availability.

Local: the backend only GENERATES ``run_benchmark.py`` (config embedded as JSON); the user
runs it on their own GPU. Importing a result parses JSON only. Nebius: no provisioning
integration exists, so the remote path reports itself unavailable (nothing is launched).
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

from ..clients.nebius import launch_gpu_job

TEMPLATE = Path(__file__).resolve().parents[2] / 'scripts' / 'run_benchmark.py'
MAX_RESULT_BYTES = 200_000
FINGERPRINT_PREFIXES = ('model.', 'training.')


def fingerprint(values):
    relevant = {k: v for k, v in sorted(values.items()) if k.startswith(FINGERPRINT_PREFIXES) and v is not None}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def _shell_quote(name):
    return name if all(c.isalnum() or c in '._-/' for c in name) else "'" + name.replace("'", "'\\''") + "'"


BENCHMARK_WARMUP_STEPS = 10
BENCHMARK_MEASURED_STEPS = 50


def _defines_build_model(meta):
    if not meta['name'].endswith('.py') or not meta.get('path'):
        return False
    try:
        tree = ast.parse(Path(meta['path']).read_text(encoding='utf-8'))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(isinstance(n, ast.FunctionDef) and n.name == 'build_model' for n in tree.body)


def plan(analysis_id, values, files):
    """Requirements and the CLI command matching the generated script (GPT-2 MVP)."""
    model_files = [f for f in files if _defines_build_model(f)]
    samples = [f for f in files if (f.get('extraction') or {}).get('kind') == 'dataset_sample'
               and (f['extraction'].get('stats') or {}).get('token_lengths')]
    command = ['python', 'run_benchmark.py']
    if model_files:
        command += ['--model', _shell_quote(model_files[0]['name'])]
    if samples:
        command += ['--data-sample', _shell_quote(samples[0]['name'])]
    command += ['--output', 'benchmark_result.json']
    template = ['python', 'run_benchmark.py', '[--model MODEL_PATH]', '[--train TRAINING_PATH]',
                '[--data-sample DATA_PATH]', '--output', 'OUTPUT_PATH']
    requirements = {
        'environment': ['Python 3.10+', 'PyTorch 2.1+ built with CUDA and a BF16-capable NVIDIA GPU',
                        'PyYAML (only for YAML --train overrides)'],
        'files': [
            {'flag': '--model', 'required': False, 'provided': [f['name'] for f in model_files],
             'description': 'Optional Python file exposing build_model(config) -> nn.Module (executes your code). '
                            'Omitted: the built-in GPT-2 reference implementation of your settings is used.'},
            {'flag': '--train', 'required': False, 'provided': [],
             'description': 'Optional JSON/YAML overrides of the embedded settings; not needed, settings are embedded.'},
            {'flag': '--data-sample', 'required': False, 'provided': [f['name'] for f in samples],
             'description': 'Optional JSONL with integer input_ids; random token ids of the configured shape otherwise.'},
        ],
        'credentials': ['None for local runs.'],
        'warnings': ['Local benchmarking executes code on your machine (PyTorch, and your model file when --model '
                     'is passed). Review files before running.',
                     f'Runs {BENCHMARK_WARMUP_STEPS} warm-up + {BENCHMARK_MEASURED_STEPS} measured optimizer steps on one GPU.'],
    }
    return {'command': ' '.join(command), 'command_template': ' '.join(template),
            'requirements': requirements, 'fingerprint': fingerprint(values)}


def benchmark_config(analysis_id, v):
    """Map the GPT-2 MVP settings onto run_benchmark.py's reference transformer."""
    H = v['model.hidden_size']
    model = {'vocab_size': v['model.vocab_size'], 'hidden_size': H, 'num_hidden_layers': v['model.num_layers'],
             'num_attention_heads': v['model.num_attention_heads'], 'position_embedding': 'learned',
             'max_position_embeddings': v['model.max_position_embeddings'], 'tie_word_embeddings': True,
             'linear_bias': True, 'layer_norm_bias': True, 'lm_head_bias': False, 'mlp_activation': 'gelu_new',
             'intermediate_size': v.get('model.intermediate_size') or 4 * H,
             'attention_dropout': v['model.attention_dropout'], 'embedding_dropout': v['model.embedding_dropout'],
             'residual_dropout': v['model.residual_dropout']}
    training = {'task': 'causal_language_modeling', 'batch_size_per_gpu': v['training.microbatch_size'],
                'gradient_accumulation_steps': v['training.gradient_accumulation_steps'], 'precision': 'bf16',
                'parameter_dtype': 'fp32', 'optimizer': 'AdamW', 'optimizer_options': {'foreach': False},
                'input_dtype': 'int64', 'target_dtype': 'int64', 'sequence_length': v['data.sequence_length'],
                'gpu_count': 1, 'distributed_strategy': 'single'}
    return {'analysis_id': analysis_id, 'config_fingerprint': fingerprint(v), 'profile': 'transformer_lm_classic',
            'architecture': 'gpt2', 'model': model, 'training': training,
            'benchmark': {'steps': BENCHMARK_MEASURED_STEPS, 'warmup_steps': BENCHMARK_WARMUP_STEPS}}


def generate_script(analysis_id, values, prediction):
    config = benchmark_config(analysis_id, values)
    config['predicted'] = {'allocated_peak_gib': (prediction or {}).get('memory', {}).get('estimated_allocated_peak')}
    text = TEMPLATE.read_text(encoding='utf-8')
    payload = json.dumps(config, indent=1, sort_keys=True)
    if "'''" in payload:
        raise ValueError('Configuration cannot be embedded safely.')
    return text.replace('__COMPUTE_CAST_CONFIG__', payload)


def benchmark_cost(prediction, values):
    """Analytical cost of the benchmark run itself (warm-up + measured steps, startup excluded)."""
    from .estimators.pricing import estimate_cost
    from ..parsers.gpt2_inputs import pricing
    runtime = (prediction or {}).get('runtime')
    if not runtime:
        return {'status': 'unavailable', 'reason': 'Runtime is unavailable, so the benchmark duration cannot be estimated.'}
    steps = BENCHMARK_WARMUP_STEPS + BENCHMARK_MEASURED_STEPS
    hours = steps * runtime['seconds_per_full_optimizer_update'] / 3600
    return {**estimate_cost(hours, pricing(values)), 'steps': BENCHMARK_MEASURED_STEPS,
            'warmup_steps': BENCHMARK_WARMUP_STEPS, 'seconds_per_optimizer_step': runtime['seconds_per_full_optimizer_update'],
            'assumption': 'Compute time only; provisioning, image pull and model loading are excluded (v1 scope).'}


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
        predicted_peak = prediction['memory']['estimated_allocated_peak']
        if predicted_peak and reserved is not None:
            comparison['peak_vram'] = {'predicted_gib': predicted_peak, 'measured_reserved_gib': reserved,
                                       'measured_allocated_gib': peak,
                                       'ratio_measured_to_predicted': reserved / predicted_peak}
        if prediction.get('runtime'):
            comparison['seconds_per_optimizer_step'] = {
                'predicted': prediction['runtime']['seconds_per_full_optimizer_update'], 'measured_median': median}
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
