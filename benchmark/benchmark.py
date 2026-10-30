#!/usr/bin/env python3
"""Compute Cast benchmark runner (same file locally and in the Nebius container).

    python benchmark.py --config benchmark_config.json [--output-dir .] [--max-seconds 300]
    python benchmark.py --config benchmark_config.json --cpu-smoke    # functional check only

Runs only the trusted models in models/ selected through model_registry.py; it never imports user
code, installs packages or downloads model code. Writes results.json (always, also on failure) and
benchmark.log to --output-dir. Exit codes: 0 succeeded, 1 runtime error, 2 invalid configuration,
3 out of memory, 4 runner time limit reached, 5 unsupported hardware.

Measurement scope: complete optimizer updates (all accumulation microbatches + AdamW step) on
synthetic token ids, timed between GPU synchronisations after warm-up updates. Excluded: process
start-up/imports, model construction and input generation (reported separately as startup), real
dataset loading/tokenisation, evaluation, checkpointing and any cloud provisioning/image pull time.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model_registry as R  # noqa: E402

EXIT = {'succeeded': 0, 'runtime_error': 1, 'invalid_config': 2, 'out_of_memory': 3, 'timeout': 4,
        'hardware_unsupported': 5}
SCOPE = {
    'measured': 'Complete optimizer updates (forward, FP32 loss, backward for every accumulation microbatch, '
                'then AdamW step) on synthetic token ids, after warm-up updates, between GPU synchronisations.',
    'excluded': ['process start-up, imports and model construction (see durations.startup_seconds)',
                 'real dataset loading, tokenisation and host-to-device input transfer',
                 'evaluation, checkpoint saving and logging overhead outside the timed loop',
                 'cloud provisioning, image pull and job scheduling time',
                 'model quality: random weights and synthetic inputs measure speed and memory only'],
    'throughput_units': 'tokens_per_second = measured updates × accumulation × microbatch × sequence length '
                        '÷ measured training seconds (synthetic, unpadded fixed-length token positions)',
}


class BenchmarkTimeout(Exception):
    pass


class HardwareUnsupported(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    """Write in place and fsync. No temp file + rename: on an S3 bucket mount a rename is a server-side copy
    of whatever has been uploaded so far, which left an empty results.json in cloud runs."""
    text = json.dumps(data, indent=2, allow_nan=False) + '\n'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())  # forces the upload on FUSE bucket mounts before the container exits


def detect_hardware(torch, cpu_smoke):
    if torch.cuda.is_available() and not cpu_smoke:
        props = torch.cuda.get_device_properties(0)
        info = {'device_type': 'cuda', 'gpu_name': props.name, 'gpu_count_visible': torch.cuda.device_count(),
                'gpu_count_used': 1, 'total_memory_bytes': props.total_memory,
                'compute_capability': f'{props.major}.{props.minor}', 'multiprocessor_count': props.multi_processor_count,
                'bf16_supported': bool(torch.cuda.is_bf16_supported())}
        if not info['bf16_supported']:
            raise HardwareUnsupported(f"{props.name} has no native BF16 support; the bf16_autocast profile cannot run "
                                      'and is not substituted.')
        return info
    if not cpu_smoke:
        raise HardwareUnsupported('No CUDA GPU detected. Use --cpu-smoke for a functional check only.')
    return {'device_type': 'cpu', 'gpu_name': None, 'gpu_count_visible': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_count_used': 0, 'cpu': platform.processor() or platform.machine(), 'total_memory_bytes': None,
            'compute_capability': None, 'multiprocessor_count': None, 'bf16_supported': None}


def run(args):
    t0 = time.perf_counter()
    started_at = now()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('benchmark')
    log.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / 'benchmark.log', mode='w')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        log.addHandler(handler)
    result = {'schema_version': R.RESULTS_SCHEMA_VERSION, 'benchmark_code_version': R.BENCHMARK_CODE_VERSION,
              'status': 'failed', 'failure': None, 'smoke_test': bool(args.cpu_smoke), 'started_at': started_at,
              'config_fingerprint': None, 'effective_config': None, 'scope': SCOPE, 'limits': {'max_seconds': args.max_seconds}}

    def fail(kind, message):
        result['failure'] = {'type': kind, 'message': message}
        result['finished_at'] = now()
        result.setdefault('durations', {})['runner_wall_seconds'] = time.perf_counter() - t0
        write_json(out / 'results.json', result)
        log.error('%s: %s', kind, message)
        return EXIT[kind]

    try:
        raw = json.loads(Path(args.config).read_text(encoding='utf-8'), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        config = R.validate(raw)
    except R.ConfigError as exc:
        result['failure_details'] = exc.errors
        return fail('invalid_config', str(exc))
    except (OSError, ValueError) as exc:
        return fail('invalid_config', f'Cannot read configuration: {exc}')
    result['effective_config'] = config
    result['config_fingerprint'] = R.fingerprint(config)
    if hasattr(signal, 'SIGALRM'):
        def on_alarm(*_):
            raise BenchmarkTimeout()
        signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(int(args.max_seconds))
    deadline = t0 + args.max_seconds
    try:
        import torch
        result['software'] = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                              'platform': platform.platform()}
        hw = detect_hardware(torch, args.cpu_smoke)
        result['hardware'] = hw
        requested = config['hardware']['requested_gpu']
        result['hardware_match'] = {'requested_gpu': requested, 'detected_gpu': hw['gpu_name'],
                                    'note': 'Requested GPU is informational; measurements describe the detected device.'}
        device = torch.device('cuda:0' if hw['device_type'] == 'cuda' else 'cpu')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(config['measurement']['seed'])
        t, meas = config['training'], config['measurement']
        G, B, S = t['gradient_accumulation_steps'], t['microbatch_size'], config['input']['sequence_length']
        model = R.build_model(config).to(device=device, dtype=torch.float32).train()
        params = {id(p): p for p in model.parameters()}
        o = t['optimizer_settings']
        optimizer = torch.optim.AdamW(params.values(), lr=o['lr'], betas=(o['beta1'], o['beta2']), eps=o['eps'],
                                      weight_decay=o['weight_decay'], amsgrad=False, foreach=False, fused=False)
        result['settings'] = {'precision': 'bf16_autocast (FP32 parameters, gradients and AdamW states)',
                              'optimizer': 'torch.optim.AdamW(foreach=False, fused=False, amsgrad=False)',
                              'attention': 'math attention with FP32 Q·Kᵀ/softmax/·V', 'tf32_enabled': False,
                              'parameter_count': sum(p.numel() for p in params.values()),
                              'expected_parameter_count': R.parameter_count(config['model'])}
        gen = torch.Generator().manual_seed(config['measurement']['seed'])
        batches = [R.make_inputs(config, gen, device) for _ in range(G)]  # generated outside the timed loop
        sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        def update():
            optimizer.zero_grad(set_to_none=True)
            for ids, labels in batches:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits = model(ids)
                loss = R.loss_fn(logits, labels) / G
                loss.backward()
            optimizer.step()
            return loss

        startup = time.perf_counter() - t0
        log.info('startup %.2fs; device %s; %s parameters', startup, hw['gpu_name'] or 'cpu', f"{result['settings']['parameter_count']:,}")
        for i in range(meas['warmup_updates']):
            update()
            log.info('warm-up update %d/%d', i + 1, meas['warmup_updates'])
        sync()
        times = []
        for i in range(meas['measured_updates']):
            if time.perf_counter() > deadline:
                raise BenchmarkTimeout()
            sync()
            begin = time.perf_counter()
            loss = update()
            sync()
            times.append(time.perf_counter() - begin)
            log.info('measured update %d/%d: %.4fs', i + 1, meas['measured_updates'], times[-1])
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        measured = sum(times)
        tokens = meas['measured_updates'] * G * B * S
        result.update(
            status='succeeded',
            counts={'warmup_updates': meas['warmup_updates'], 'measured_updates': meas['measured_updates'],
                    'microbatches_per_update': G, 'microbatch_size': B, 'sequence_length': S},
            timing={'measured_training_seconds': measured, 'mean_update_seconds': measured / len(times),
                    'median_update_seconds': statistics.median(times), 'min_update_seconds': min(times),
                    'max_update_seconds': max(times), 'update_seconds': times},
            throughput={'tokens_per_second': tokens / measured, 'sequences_per_second': meas['measured_updates'] * G * B / measured},
            memory={'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else None,
                    'note': 'PyTorch caching-allocator peaks on the measured device; excludes CUDA context and '
                            'other processes.' if device.type == 'cuda' else 'Not measured on CPU.'},
            final_loss=float(loss.detach()),
            durations={'startup_seconds': startup, 'runner_wall_seconds': time.perf_counter() - t0,
                       'note': 'Runner-side times only; total job and billable durations come from the platform.'},
            finished_at=now())
        if args.cpu_smoke:
            result['notes'] = ['CPU functional smoke test: NOT valid for CUDA performance or memory estimates.']
        write_json(out / 'results.json', result)
        log.info('succeeded: mean update %.4fs, %.1f tokens/s', result['timing']['mean_update_seconds'], result['throughput']['tokens_per_second'])
        return 0
    except BenchmarkTimeout:
        return fail('timeout', f'Runner time limit of {args.max_seconds}s reached (includes start-up and warm-up).')
    except HardwareUnsupported as exc:
        return fail('hardware_unsupported', str(exc))
    except Exception as exc:  # classify OOM without importing torch at module level
        if type(exc).__name__ == 'OutOfMemoryError' or 'out of memory' in str(exc).lower():
            try:
                import torch
                result['memory'] = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()} if torch.cuda.is_available() else None
            except Exception:
                pass
            return fail('out_of_memory', str(exc).splitlines()[0][:500])
        log.error(traceback.format_exc())
        return fail('runtime_error', f'{type(exc).__name__}: {str(exc)[:500]}')
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='benchmark_config.json')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--max-seconds', type=float, default=float(os.environ.get('BENCH_MAX_SECONDS', 300)),
                        help='Runner execution limit including start-up (default 300 s).')
    parser.add_argument('--cpu-smoke', action='store_true',
                        help='Run on CPU as a functional smoke test (not valid for CUDA performance or memory).')
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 3600:
        parser.error('--max-seconds must be between 1 and 3600.')
    code = run(args)
    logging.shutdown()  # flush and close benchmark.log before the bucket mount is torn down
    if hasattr(os, 'sync'):
        os.sync()
    sys.exit(code)


if __name__ == '__main__':
    main()
