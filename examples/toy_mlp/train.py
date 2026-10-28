"""CPU/CUDA training example and single-GPU local benchmark adapter.

Importing this file does not start training. Configs live beside the model file.
Synthetic data is for testing the application, not an accuracy benchmark.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load_settings(folder):
    architecture = json.loads((folder / 'model_config.json').read_text())
    settings = json.loads((folder / 'train_config.json').read_text())
    if settings['precision'] != 'fp32' or settings['optimizer'] != 'AdamW':
        raise ValueError('This adapter implements FP32 AdamW only.')
    if settings['gpu_count'] != 1 or settings['distributed_strategy'] != 'single':
        raise ValueError('This adapter supports one GPU without sharding.')
    if settings['gradient_accumulation_steps'] != 1 or settings['gradient_checkpointing']:
        raise ValueError('This adapter requires accumulation=1 and checkpointing=false.')
    if not 1 <= settings['batch_size_per_gpu'] <= settings['num_examples']:
        raise ValueError('Batch size must be between 1 and the number of examples.')
    return architecture, settings


def synthetic_data(architecture, settings):
    generator = torch.Generator().manual_seed(settings['seed'])
    features = torch.randn(settings['num_examples'], architecture['input_size'], generator=generator)
    teacher = torch.randn(architecture['input_size'], architecture['output_size'], generator=generator)
    targets = features @ teacher / architecture['input_size'] ** .5
    return {'features': features, 'targets': targets}


def make_step(model_path, data, device):
    model_path = Path(model_path).resolve()
    architecture, settings = load_settings(model_path.parent)
    torch.manual_seed(settings['seed'])
    spec = importlib.util.spec_from_file_location('toy_benchmark_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    network = module.build_model(architecture).to(device).train()
    count = sum(parameter.numel() for parameter in network.parameters())
    if count != architecture['total_params']:
        raise ValueError('Update total_params in model_config.json to match the model.')
    features, targets = data['features'], data['targets']
    if tuple(features.shape) != (settings['num_examples'], architecture['input_size']):
        raise ValueError('Sample feature shape does not match configs.')
    if tuple(targets.shape) != (settings['num_examples'], architecture['output_size']):
        raise ValueError('Sample target shape does not match configs.')
    # The sample remains in CPU memory; host-to-device copies are timed per step.
    opt = torch.optim.AdamW(network.parameters(), lr=settings['learning_rate'])
    microbatch = settings['batch_size_per_gpu']
    position = 0

    def step():
        nonlocal position
        end = min(position + microbatch, len(features))
        x = features[position:end].to(device)
        y = targets[position:end].to(device)
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(network(x), y)
        loss.backward()
        opt.step()
        position = end if end < len(features) else 0
        return loss.detach()

    return step, network


def build_benchmark(model_path, data_path):
    """Contract consumed by backend/scripts/run_benchmark.py; CUDA required."""
    if not torch.cuda.is_available():
        raise RuntimeError('The local GPU benchmark requires CUDA; use train.py --device cpu for a smoke test.')
    data = torch.load(data_path, map_location='cpu', weights_only=True)
    step, _ = make_step(model_path, data, 'cuda:0')
    return step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--steps', type=int, help='Optional smoke-test step count; otherwise run configured epochs.')
    parser.add_argument('--write-sample', type=Path, help='Write synthetic .pt data for the benchmark runner, then exit.')
    args = parser.parse_args()
    architecture, settings = load_settings(HERE)
    data = synthetic_data(architecture, settings)
    if args.write_sample:
        if args.write_sample.exists():
            parser.error('Refusing to overwrite an existing sample file.')
        torch.save(data, args.write_sample)
        print(f'Saved synthetic sample: {args.write_sample}')
        return
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    device = 'cuda:0' if args.device == 'cuda' or args.device == 'auto' and torch.cuda.is_available() else 'cpu'
    step, network = make_step(HERE / 'model.py', data, device)
    iterations = args.steps or ((settings['num_examples'] + settings['batch_size_per_gpu'] - 1) // settings['batch_size_per_gpu']) * settings['epochs']
    losses = [float(step().cpu()) for _ in range(iterations)]
    if not all(torch.isfinite(parameter).all() for parameter in network.parameters()):
        raise RuntimeError('Training produced non-finite parameters.')
    print(f'Device: {device}; parameters: {sum(p.numel() for p in network.parameters())}; steps: {iterations}')
    print(f'First batch loss: {losses[0]:.6f}; last batch loss: {losses[-1]:.6f}')


if __name__ == '__main__':
    main()
