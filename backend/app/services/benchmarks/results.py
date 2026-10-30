"""Validate untrusted results.json (local upload/paste or cloud bucket) and compare with predictions.

Results are data, never code: size-limited, parsed with NaN/Infinity rejected, checked against the
runner's schema, and the embedded effective_config is re-validated and re-fingerprinted. The stored
prediction is never overwritten; measurements sit beside it.
"""
from __future__ import annotations

import json
import math

from .registry import registry

MAX_RESULT_BYTES = 256 * 1024
STATUSES = {'succeeded', 'failed'}
FAILURES = {'invalid_config', 'out_of_memory', 'timeout', 'hardware_unsupported', 'runtime_error'}
GIB = 2 ** 30


class ResultError(ValueError):
    pass


class ResultDecodeError(ResultError):
    """The bytes are not (yet) a JSON document: empty, truncated or not UTF-8. Cloud collection retries these briefly."""


def _reject_constant(name):
    raise ResultError(f'Non-finite number {name} is not allowed.')


def _num(data, path, minimum=0.0, allow_none=False, integer=False):
    value = data
    for part in path.split('.'):
        value = value.get(part) if isinstance(value, dict) else None
    if value is None and allow_none:
        return None
    ok = isinstance(value, int) if integer else isinstance(value, (int, float))
    if isinstance(value, bool) or not ok or not math.isfinite(value) or value < minimum:
        raise ResultError(f'{path} must be a finite {"integer" if integer else "number"} >= {minimum}.')
    return value


def _dict(data, key):
    value = data.get(key)
    if not isinstance(value, dict):
        raise ResultError(f'{key} must be an object.')
    return value


def _text(value, limit=500):
    return value[:limit] if isinstance(value, str) else None


def parse(raw):
    """Parse and structurally validate results.json bytes/str. Returns a normalised dict."""
    R = registry()
    if isinstance(raw, str):
        raw = raw.encode('utf-8')
    if len(raw) > MAX_RESULT_BYTES:
        raise ResultError(f'results.json exceeds {MAX_RESULT_BYTES // 1024} KiB.')
    raw = raw.removeprefix(b'\xef\xbb\xbf')  # editors on Windows add a UTF-8 BOM
    if not raw.strip():
        raise ResultDecodeError('results.json is empty.')
    try:
        data = json.loads(raw.decode('utf-8'), parse_constant=_reject_constant)
    except UnicodeDecodeError as exc:
        raise ResultDecodeError(f'results.json is not valid UTF-8 ({len(raw)} bytes, starts {raw[:16]!r}).') from exc
    except json.JSONDecodeError as exc:
        raise ResultDecodeError(f'results.json is not valid JSON ({len(raw)} bytes): {exc.msg} at line {exc.lineno}.') from exc
    if not isinstance(data, dict):
        raise ResultError('results.json must be a JSON object.')
    if data.get('schema_version') != R.RESULTS_SCHEMA_VERSION:
        raise ResultError(f'Unsupported results schema_version (expected {R.RESULTS_SCHEMA_VERSION}).')
    status = data.get('status')
    if status not in STATUSES:
        raise ResultError('status must be "succeeded" or "failed".')
    out = {'schema_version': data['schema_version'], 'benchmark_code_version': _text(data.get('benchmark_code_version'), 40),
           'status': status, 'smoke_test': data.get('smoke_test') is True, 'failure': None,
           'started_at': _text(data.get('started_at'), 40), 'finished_at': _text(data.get('finished_at'), 40)}
    if status == 'failed':
        failure = data.get('failure')
        if not isinstance(failure, dict) or failure.get('type') not in FAILURES:
            raise ResultError('A failed result needs failure.type from the runner.')
        out['failure'] = {'type': failure['type'], 'message': _text(failure.get('message')) or ''}
    config = data.get('effective_config')
    if config is not None:
        try:
            config = R.validate(config)
        except R.ConfigError as exc:
            raise ResultError(f'effective_config is not a valid benchmark configuration: {exc}') from exc
        if data.get('config_fingerprint') != R.fingerprint(config):
            raise ResultError('config_fingerprint does not match effective_config.')
    elif status == 'succeeded':
        raise ResultError('A succeeded result needs effective_config.')
    out['effective_config'], out['config_fingerprint'] = config, data.get('config_fingerprint') if config else None
    if status == 'succeeded':
        timing, counts = _dict(data, 'timing'), _dict(data, 'counts')
        for key in ('measured_training_seconds', 'mean_update_seconds', 'median_update_seconds', 'min_update_seconds', 'max_update_seconds'):
            _num(timing, key, 1e-9)
        samples = timing.get('update_seconds')
        if not isinstance(samples, list) or not 1 <= len(samples) <= R.LIMITS['measured_updates']:
            raise ResultError('timing.update_seconds must list the measured updates.')
        for sample in samples:
            _num({'update_seconds': sample}, 'update_seconds', 1e-9)
        for key in ('warmup_updates', 'measured_updates', 'microbatches_per_update', 'microbatch_size', 'sequence_length'):
            _num(counts, key, 0, integer=True)
        if counts['measured_updates'] != len(samples) or counts['measured_updates'] != config['measurement']['measured_updates']:
            raise ResultError('Measured update count does not match the configuration.')
        expected = {'microbatches_per_update': config['training']['gradient_accumulation_steps'],
                    'microbatch_size': config['training']['microbatch_size'], 'sequence_length': config['input']['sequence_length'],
                    'warmup_updates': config['measurement']['warmup_updates']}
        if any(counts[k] != v for k, v in expected.items()):
            raise ResultError('Recorded counts do not match effective_config.')
        throughput = _dict(data, 'throughput')
        memory = data.get('memory') if isinstance(data.get('memory'), dict) else {}
        durations = data.get('durations') if isinstance(data.get('durations'), dict) else {}
        out.update(
            timing={k: timing[k] for k in ('measured_training_seconds', 'mean_update_seconds', 'median_update_seconds',
                                           'min_update_seconds', 'max_update_seconds')} | {'update_seconds': samples},
            counts={k: counts[k] for k in ('warmup_updates', 'measured_updates', 'microbatches_per_update', 'microbatch_size', 'sequence_length')},
            throughput={'tokens_per_second': _num(throughput, 'tokens_per_second'),
                        'sequences_per_second': _num(throughput, 'sequences_per_second'),
                        'units': 'synthetic fixed-length token positions per measured training second'},
            final_loss=_num(data, 'final_loss', -1e30, allow_none=True))
    else:
        memory = data.get('memory') if isinstance(data.get('memory'), dict) else {}
        durations = data.get('durations') if isinstance(data.get('durations'), dict) else {}
    out['memory'] = {'peak_allocated_bytes': _num(memory, 'peak_allocated_bytes', allow_none=True, integer=True),
                     'peak_reserved_bytes': _num(memory, 'peak_reserved_bytes', allow_none=True, integer=True)}
    out['durations'] = {'startup_seconds': _num(durations, 'startup_seconds', allow_none=True),
                        'runner_wall_seconds': _num(durations, 'runner_wall_seconds', allow_none=True)}
    hw = data.get('hardware') if isinstance(data.get('hardware'), dict) else {}
    out['hardware'] = {'device_type': _text(hw.get('device_type'), 10), 'gpu_name': _text(hw.get('gpu_name'), 120),
                       'gpu_count_visible': _num(hw, 'gpu_count_visible', allow_none=True, integer=True),
                       'gpu_count_used': _num(hw, 'gpu_count_used', allow_none=True, integer=True),
                       'total_memory_bytes': _num(hw, 'total_memory_bytes', allow_none=True, integer=True),
                       'compute_capability': _text(hw.get('compute_capability'), 10),
                       'multiprocessor_count': _num(hw, 'multiprocessor_count', allow_none=True, integer=True),
                       'bf16_supported': hw.get('bf16_supported') if isinstance(hw.get('bf16_supported'), bool) else None}
    sw = data.get('software') if isinstance(data.get('software'), dict) else {}
    out['software'] = {k: _text(sw.get(k), 120) for k in ('python', 'torch', 'cuda', 'platform')}
    if status == 'succeeded' and not out['smoke_test'] and out['hardware']['device_type'] != 'cuda':
        raise ResultError('A non-smoke result must come from a CUDA device.')
    if (out['hardware']['gpu_count_used'] or 0) > 1:
        raise ResultError('Multi-GPU results are not supported (one GPU per benchmark).')
    return out


def _gpu_matches(requested, detected):
    norm = lambda s: ''.join(ch for ch in (s or '').lower().replace('nvidia', '').replace('®', '') if ch.isalnum())
    a, b = norm(requested), norm(detected)
    return bool(a and b) and (a in b or b in a or a.split('nvlink')[0] in b)


def compare(result, review, expected_fingerprint, gpu_label):
    """Side-by-side predicted vs measured for one validated result.

    ``review`` is the config_builder payload for the analysis' current workload. Calibration ratios are
    shown only when the result's configuration fingerprint equals the current one AND the measured GPU
    matches the predicted GPU; no scaling to other shapes or GPUs is attempted.
    """
    issues = []
    same_config = result['config_fingerprint'] == expected_fingerprint
    if not same_config:
        issues.append('Measured configuration differs from the current benchmark configuration: no calibration applied.')
    if result['smoke_test']:
        issues.append('CPU smoke test: functional check only; not comparable with GPU predictions.')
    same_gpu = _gpu_matches(gpu_label, result['hardware']['gpu_name'])
    if result['hardware']['gpu_name'] and not same_gpu:
        issues.append(f"Measured on {result['hardware']['gpu_name']}, predicted for {gpu_label or 'an unspecified GPU'}: no calibration applied.")
    compatible = same_config and same_gpu and not result['smoke_test'] and result['status'] == 'succeeded'
    prediction = (review or {}).get('prediction') or {}
    rows = []
    if result['status'] == 'succeeded':
        measured_step = result['timing']['median_update_seconds']
        rows.append({'metric': 'Seconds per optimizer update', 'predicted': prediction.get('seconds_per_update'),
                     'measured': measured_step, 'measured_label': 'median of measured updates',
                     'ratio': measured_step / prediction['seconds_per_update'] if compatible and prediction.get('seconds_per_update') else None})
    peak = result['memory']['peak_allocated_bytes']
    rows.append({'metric': 'Peak allocated memory (GiB)', 'predicted': prediction.get('estimated_allocated_peak_gib'),
                 'measured': None if peak is None else peak / GIB, 'measured_label': 'torch.cuda.max_memory_allocated',
                 'ratio': (peak / GIB) / prediction['estimated_allocated_peak_gib']
                 if compatible and peak and prediction.get('estimated_allocated_peak_gib') else None})
    reserved = result['memory']['peak_reserved_bytes']
    rows.append({'metric': 'Peak reserved memory (GiB)', 'predicted': None, 'measured': None if reserved is None else reserved / GIB,
                 'measured_label': 'torch.cuda.max_memory_reserved (allocator cache; not predicted)', 'ratio': None})
    return {'compatible_for_calibration': compatible, 'rows': rows, 'issues': issues,
            'labels': ['Measured memory is shown separately; predictions are not modified.',
                       'Start-up, provisioning and image pull are excluded from measured training time.',
                       'Synthetic inputs: data loading is not measured.',
                       'Ratios apply only to this exact configuration and GPU; they are not scaled to other shapes.']}


def full_run_projection(result, prediction):
    """Measured seconds/update x the analysis' planned optimizer updates (same config only), labelled as an estimate."""
    runtime = (prediction or {}).get('runtime') or {}
    total = runtime.get('total_training_compute') or {}
    updates = total.get('optimizer_updates')
    if result['status'] != 'succeeded' or not updates:
        return None
    seconds = updates * result['timing']['median_update_seconds']
    return {'optimizer_updates': updates, 'seconds': seconds, 'hours': seconds / 3600,
            'label': 'Estimate: measured median update time x planned updates (synthetic inputs; data loading, evaluation, '
                     'checkpointing and start-up excluded). Not a measured full run.'}
