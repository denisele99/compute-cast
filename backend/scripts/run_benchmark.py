#!/usr/bin/env python3
"""Compute Cast local benchmark (generated). Runs on YOUR machine with PyTorch + a GPU.

  python run_benchmark.py --model MODEL_PATH --train TRAINING_PATH \
      --data-sample DATA_PATH --output OUTPUT_PATH

All arguments except --output are optional:
  --model        Python file exposing build_model(config: dict) -> torch.nn.Module.
                 WARNING: this IMPORTS AND EXECUTES your code. Omit it to benchmark the
                 built-in reference implementation of the selected profile instead.
  --train        JSON/YAML training config whose keys override the embedded settings
                 (e.g. batch_size_per_gpu). The training LOOP is this script's reference
                 loop (configured optimizer, precision, accumulation); training scripts
                 are not run.
  --data-sample  CSV/TSV/JSONL sample used to build batches; random tensors with the
                 configured shapes/dtypes are used when omitted.
  --output       Where to write the JSON result to import into Compute Cast.
  --device       cuda (default) or cpu (dry run; no GPU memory statistics).

Measures a short run (warm-up + measured optimizer steps) on ONE GPU: peak
torch.cuda.max_memory_allocated/reserved and seconds per optimizer step. Multi-GPU
DDP/ZeRO/FSDP behaviour is not reproduced; results are flagged as not representative then.
Requirements: Python 3.10+, PyTorch 2.1+ built with CUDA, optional PyYAML for YAML.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CONFIG = json.loads(r'''__COMPUTE_CAST_CONFIG__''')

DTYPES = {'fp32': 'float32', 'fp16': 'float16', 'bf16': 'bfloat16'}


def load_overrides(path):
    text = Path(path).read_text(encoding='utf-8')
    if Path(path).suffix.lower() in ('.yaml', '.yml'):
        import yaml
        data = yaml.safe_load(text)
    elif Path(path).suffix.lower() == '.json':
        data = json.loads(text)
    else:
        raise SystemExit('--train must be a JSON or YAML training config; training scripts are not run.')
    if not isinstance(data, dict):
        raise SystemExit('--train config must be a mapping.')
    return data


def reference_model(torch, profile, m):
    nn = torch.nn
    if profile == 'mlp_regression':
        return nn.Sequential(nn.Linear(m['input_size'], m['hidden_size'], bias=m['linear_bias']), nn.ReLU(),
                             nn.Linear(m['hidden_size'], m['output_size'], bias=m['linear_bias']))
    H, A, L, V = m['hidden_size'], m['num_attention_heads'], m['num_hidden_layers'], m['vocab_size']
    bias, ln_bias = m['linear_bias'], m['layer_norm_bias']
    I = m.get('intermediate_size') or 4 * H
    p_attn, p_embd, p_resid = (m.get(k, 0.1) for k in ('attention_dropout', 'embedding_dropout', 'residual_dropout'))
    act_name = m.get('mlp_activation', 'gelu')
    act = nn.GELU(approximate='tanh') if act_name == 'gelu_new' else nn.GELU() if act_name == 'gelu' else nn.ReLU()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_1 = nn.LayerNorm(H, bias=ln_bias)
            self.c_attn = nn.Linear(H, 3 * H, bias=bias)
            self.c_proj = nn.Linear(H, H, bias=bias)
            self.ln_2 = nn.LayerNorm(H, bias=ln_bias)
            self.c_fc = nn.Linear(H, I, bias=bias)
            self.mlp_proj = nn.Linear(I, H, bias=bias)
            self.attn_drop = nn.Dropout(p_attn)
            self.drop = nn.Dropout(p_resid)

        def forward(self, x, causal):
            B, S, _ = x.shape
            q, k, v = self.c_attn(self.ln_1(x)).split(H, dim=2)
            q, k, v = (t.view(B, S, A, H // A).transpose(1, 2) for t in (q, k, v))
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(H // A)  # eager attention
            if causal:
                mask = torch.ones(S, S, dtype=torch.bool, device=x.device).triu(1)
                scores = scores.masked_fill(mask, float('-inf'))
            y = self.attn_drop(torch.softmax(scores, dim=-1)) @ v
            x = x + self.drop(self.c_proj(y.transpose(1, 2).reshape(B, S, H)))
            return x + self.drop(self.mlp_proj(act(self.c_fc(self.ln_2(x)))))

    class ClassicLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(V, H)
            self.wpe = nn.Embedding(m['max_position_embeddings'], H) if m['position_embedding'] == 'learned' else None
            self.h = nn.ModuleList(Block() for _ in range(L))
            self.embd_drop = nn.Dropout(p_embd)
            self.ln_f = nn.LayerNorm(H, bias=ln_bias)
            self.lm_head = nn.Linear(H, V, bias=m['lm_head_bias'])
            if m['tie_word_embeddings']:
                self.lm_head.weight = self.wte.weight
            self.causal = CONFIG['training']['task'] == 'causal_language_modeling'

        def forward(self, ids):
            x = self.wte(ids)
            if self.wpe is not None:
                x = x + self.wpe(torch.arange(ids.shape[1], device=ids.device))
            x = self.embd_drop(x)
            for block in self.h:
                x = block(x, self.causal)
            return self.lm_head(self.ln_f(x))

    return ClassicLM()


def user_model(path, config):
    print(f'WARNING: importing and executing {path} (your code).', file=sys.stderr)
    spec = importlib.util.spec_from_file_location('user_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'build_model', None)):
        raise SystemExit(f'{path} must define build_model(config) returning an nn.Module.')
    return module.build_model(dict(config))


def sample_batches(torch, path, profile, m, t, device):
    rows = []
    suffix = Path(path).suffix.lower()
    with open(path, encoding='utf-8', newline='') as handle:
        if suffix in ('.csv', '.tsv'):
            rows = list(csv.DictReader(handle, delimiter='\t' if suffix == '.tsv' else ','))
        else:
            rows = [json.loads(line) for line in handle if line.strip()] if suffix == '.jsonl' else json.load(handle)
    if not rows:
        raise SystemExit('Data sample is empty.')
    B = t['batch_size_per_gpu']
    if profile == 'mlp_regression':
        cols = list(rows[0])
        targets = [c for c in cols if c.lower() in ('target', 'targets', 'label', 'labels', 'y')] or cols[-m['output_size']:]
        inputs = [c for c in cols if c not in targets][:m['input_size']]
        if len(inputs) != m['input_size'] or len(targets) != m['output_size']:
            raise SystemExit(f'Sample has {len(inputs)} feature / {len(targets)} target columns; '
                             f"expected {m['input_size']} / {m['output_size']}.")
        x = torch.tensor([[float(r[c]) for c in inputs] for r in rows], dtype=getattr(torch, t['input_dtype']))
        y = torch.tensor([[float(r[c]) for c in targets] for r in rows], dtype=getattr(torch, t['target_dtype']))
    else:
        S = t['sequence_length']
        key = next(k for k in ('input_ids', 'tokens', 'token_ids') if k in rows[0])
        ids = [(list(r[key]) + [0] * S)[:S] for r in rows]
        x = torch.tensor(ids, dtype=getattr(torch, t['input_dtype']))
        y = x.clone().to(getattr(torch, t['target_dtype']))
    while x.shape[0] < B:
        x, y = torch.cat([x, x]), torch.cat([y, y])
    i = 0
    while True:
        if i + B > x.shape[0]:
            i = 0
        yield x[i:i + B].to(device), y[i:i + B].to(device)
        i += B


def synthetic_batches(torch, profile, m, t, device):
    B = t['batch_size_per_gpu']
    gen = torch.Generator(device='cpu').manual_seed(0)
    while True:
        if profile == 'mlp_regression':
            x = torch.randn(B, m['input_size'], generator=gen, dtype=getattr(torch, t['input_dtype']))
            y = torch.randn(B, m['output_size'], generator=gen, dtype=getattr(torch, t['target_dtype']))
        else:
            x = torch.randint(0, m['vocab_size'], (B, t['sequence_length']), generator=gen).to(getattr(torch, t['input_dtype']))
            y = x.clone().to(getattr(torch, t['target_dtype']))
        yield x.to(device), y.to(device)


def make_optimizer(torch, model, t):
    name = t['optimizer']
    o = t.get('optimizer_options') or {}
    params = [p for p in model.parameters() if p.requires_grad]
    kwargs = {'foreach': o['foreach']} if 'foreach' in o else {}
    if name in ('Adam', 'AdamW'):
        return getattr(torch.optim, name)(params, lr=1e-4, amsgrad=o.get('amsgrad', False), **kwargs)
    if name == 'SGD':
        return torch.optim.SGD(params, lr=1e-3, momentum=o.get('momentum', 0), **kwargs)
    if name == 'RMSprop':
        return torch.optim.RMSprop(params, lr=1e-4, momentum=o.get('momentum', 0),
                                   centered=o.get('centered', False), **kwargs)
    return torch.optim.Adagrad(params, lr=1e-3, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model')
    parser.add_argument('--train')
    parser.add_argument('--data-sample')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--steps', type=int, default=CONFIG['benchmark']['steps'])
    parser.add_argument('--warmup', type=int, default=CONFIG['benchmark']['warmup_steps'])
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        raise SystemExit('PyTorch is required: https://pytorch.org/get-started/locally/')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA GPU not available. Use --device cpu for a dry run (no memory statistics).')
    profile, m = CONFIG['profile'], dict(CONFIG['model'])
    t = dict(CONFIG['training'])
    overrides = load_overrides(args.train) if args.train else {}
    t.update({k: v for k, v in overrides.items() if k in t})
    device = torch.device(args.device)
    notes = []
    if t.get('gpu_count', 1) > 1 or t.get('distributed_strategy', 'single') != 'single':
        notes.append('Configured for multiple GPUs/sharding; this benchmark runs ONE GPU without '
                     'collectives or sharding, so memory/time are not representative.')
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = datetime.now(timezone.utc).isoformat()
    model = user_model(args.model, m) if args.model else reference_model(torch, profile, m)
    param_dtype = getattr(torch, DTYPES[t['parameter_dtype']])
    model = model.to(device=device, dtype=param_dtype).train()
    count = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    optimizer = make_optimizer(torch, model, t)
    batches = (sample_batches(torch, args.data_sample, profile, m, t, device) if args.data_sample
               else synthetic_batches(torch, profile, m, t, device))
    compute = DTYPES[t['precision']]
    autocast = compute != DTYPES[t['parameter_dtype']] and compute != 'float32'
    scaler = torch.amp.GradScaler(args.device, enabled=autocast and compute == 'float16')
    loss_fn = torch.nn.MSELoss() if profile == 'mlp_regression' else torch.nn.CrossEntropyLoss()
    A = t.get('gradient_accumulation_steps', 1)
    times = []

    def sync():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    for step in range(args.warmup + args.steps):
        sync()
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(A):
            x, y = next(batches)
            if not autocast and x.is_floating_point():
                x = x.to(param_dtype)  # low-precision parameters without autocast
            with torch.autocast(args.device, dtype=getattr(torch, compute), enabled=autocast):
                out = model(x)
                if profile == 'mlp_regression':
                    loss = loss_fn(out.float(), y.float())
                else:
                    loss = loss_fn(out.float().view(-1, out.shape[-1]), y.view(-1).long())
            scaler.scale(loss / A).backward()
        scaler.step(optimizer)
        scaler.update()
        sync()
        if step >= args.warmup:
            times.append(time.perf_counter() - begin)
    gib = 1024 ** 3
    result = {
        'schema': 'compute-cast-benchmark/1',
        'analysis_id': CONFIG['analysis_id'], 'config_fingerprint': CONFIG['config_fingerprint'],
        'profile': profile, 'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat(),
        'device': {'type': args.device,
                   'name': torch.cuda.get_device_name(0) if args.device == 'cuda' else platform.processor() or 'cpu',
                   'total_memory_gib': torch.cuda.get_device_properties(0).total_memory / gib if args.device == 'cuda' else None},
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'model_source': 'user_build_model' if args.model else 'reference_profile_implementation',
        'user_code_executed': bool(args.model),
        'data_source': 'sample' if args.data_sample else 'synthetic',
        'parameter_count': count,
        'settings': {k: t.get(k) for k in ('batch_size_per_gpu', 'gradient_accumulation_steps', 'precision',
                                           'parameter_dtype', 'optimizer', 'sequence_length')},
        'warmup_steps': args.warmup, 'measured_steps': args.steps,
        'seconds_per_optimizer_step': {'median': statistics.median(times), 'mean': statistics.fmean(times),
                                       'max': max(times)} if times else None,
        'peak_memory_allocated_gib': torch.cuda.max_memory_allocated() / gib if args.device == 'cuda' else None,
        'peak_memory_reserved_gib': torch.cuda.max_memory_reserved() / gib if args.device == 'cuda' else None,
        'representative': not notes and args.device == 'cuda', 'notes': notes,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
