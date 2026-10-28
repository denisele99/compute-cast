#!/usr/bin/env python3
"""Local, single-GPU PyTorch benchmark for trusted user code.

Training adapter contract: build_benchmark(model_path, data_path) returns a
zero-argument callable that performs ONE full training microbatch, including
forward, loss, backward and optimizer work. It must match analysis_config.json.
The adapter owns model construction, device placement and data loading.
This imports/executes your training script: inspect it before running.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--data_sample', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error('steps and warmup must be positive')
    if Path(args.output).resolve() in {Path(v).resolve() for v in (args.model, args.train, args.data_sample, args.config)}:
        parser.error('output must not overwrite an input')
    config = json.loads(Path(args.config).read_text())
    if config['training']['script_gpu_count'] != 1 or config['training']['strategy'] != 'single':
        parser.error('This runner supports single-GPU training only; distributed benchmarking needs a distributed adapter/launcher.')
    try:
        import torch
    except ImportError:
        parser.error('Install a CUDA-compatible PyTorch build and your training dependencies first')
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU and matching PyTorch installation are required')
    capacity = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if abs(capacity - config['hardware']['vram_gib']) / config['hardware']['vram_gib'] > .1:
        parser.error('Local CUDA capacity differs from selected hardware; select matching custom hardware first')
    spec = importlib.util.spec_from_file_location('user_benchmark_training', args.train)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'build_benchmark'):
        parser.error('Training file must implement build_benchmark(model_path, data_path) -> callable; see --help')
    step = module.build_benchmark(args.model, args.data_sample)
    if not callable(step):
        parser.error('build_benchmark must return a callable')
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / args.steps
    result = {'schema_version': 1, 'config': config, 'steps': args.steps,
              'seconds_per_batch': seconds,
              'measured_peak_vram_gib': torch.cuda.max_memory_reserved() / 1024**3,
              'gpu_name': torch.cuda.get_device_name(0), 'capacity_gib': capacity,
              'scope': 'whole_training_microbatch',
              'note': 'CUDA reserved peak for this process; excludes other processes, driver allocations and full-run initialization/checkpoint peaks.'}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Saved {args.output}. Review hardware and workload, then import in the planner.')


if __name__ == '__main__':
    main()
