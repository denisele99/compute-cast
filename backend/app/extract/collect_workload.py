"""Collect PyTorch estimator inputs, using meta initialization by default.

Python 3.10+, PyTorch 2+; optional PyYAML for YAML configs. No GPU required.

Usage:
  python collect_workload.py --model-script model.py --training-script train.py \
      --model-config model_config.json --training-config training_config.json \
      --output workload.json

Model script contract (or supply --factory another_function):
  import torch.nn as nn
  def build_model(config):
      return nn.Linear(config["input_size"], config["output_size"])

Model config JSON: {"input_size": 100, "output_size": 64}
Training config JSON (use actual values; omitted values remain unknown):
  {"batch_size_per_gpu": 16, "epochs": 10, "num_examples": 100000,
   "sequence_length": 512, "gradient_accumulation_steps": 1,
   "precision": "bf16", "optimizer": "AdamW", "optimizer_state_dtype": "fp32",
   "gradient_dtype": "fp32", "master_weights": "none",
   "gradient_checkpointing": false, "gpu_count": 1,
   "distributed_strategy": "single", "attention_backend": "eager"}

Optional --hardware-config describes TARGET hardware, not the local machine.
--parameter-dtype explicitly casts floating meta parameters/buffers; autocast
precision in training config does not imply the parameter storage dtype.

The training script is scanned with AST and NEVER intentionally executed.
The MODEL script IS imported/executed and its factory runs under a meta device
context: use trusted code only. Meta is not a sandbox; explicit device choices,
imports, checkpoint downloads and side effects in user code can still run.
Model files must not import training entrypoints with top-level training code.
Use configuration constructors rather than from_pretrained/checkpoint loading.
Required local modules/packages must be present in the Python environment.
No checkpoint or dataset is required, but input shapes/dataset counts must be
provided to downstream estimators. Lazy modules need explicit known shapes.

Static scan only (stdlib, backward-compatible):
  python collect_workload.py --script train.py --output script_report.json

Live inspection API retained:
  report = collect_workload(model, optimizer, training={...})
  save_report(report, "workload.json")

No forward/backward, optimizer step, or benchmark is performed by the collector.
It collects facts and assumptions, not a peak-memory/runtime prediction.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import importlib.util
import sys
import uuid
from pathlib import Path

SETTING_NAMES = {
    'batch_size', 'batch_size_per_gpu', 'per_device_train_batch_size',
    'epochs', 'num_train_epochs', 'max_steps', 'num_examples',
    'sequence_length', 'max_length', 'max_seq_length', 'gradient_accumulation_steps',
    'gradient_checkpointing', 'fp16', 'bf16', 'precision', 'dtype',
    'num_workers', 'pin_memory', 'drop_last', 'eval_steps', 'save_steps',
    'world_size', 'gpu_count', 'distributed_strategy', 'attention_backend',
    'optim', 'optimizer', 'lr', 'momentum', 'amsgrad', 'foreach', 'fused',
}
REQUIRED = (
    'num_examples', 'epochs', 'batch_size_per_gpu', 'gradient_accumulation_steps',
    'precision', 'gradient_checkpointing', 'gpu_count', 'distributed_strategy',
    'attention_backend', 'master_weights',
)


def scan_training_script(path):
    """AST-only hints; no scope/control-flow resolution or code execution."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    candidates = []

    def add(name, node, line):
        if name not in SETTING_NAMES:
            return
        try:
            value = ast.literal_eval(node)
            json.dumps(value, allow_nan=False)
            literal = True
        except (ValueError, TypeError, SyntaxError):
            value, literal = None, False
        candidates.append({'name': name, 'line': line, 'literal': literal,
                           'value': value, 'expression': ast.unparse(node)})

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, (ast.Name, ast.Attribute)):
                    add(target.id if isinstance(target, ast.Name) else target.attr,
                        node.value, node.lineno)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            if isinstance(target, (ast.Name, ast.Attribute)):
                add(target.id if isinstance(target, ast.Name) else target.attr,
                    node.value, node.lineno)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                add(kw.arg, kw.value, node.lineno)
    return {'candidates': candidates,
            'limitations': 'Candidates may be unused, overridden, or in inactive branches. '
                           'External configs, CLI overrides and dynamic values need explicit input.'}


def _tensor_info(name, t):
    import torch
    result = {'name': name, 'shape': list(t.shape), 'numel': t.numel(),
              'dtype': str(t.dtype), 'device': str(t.device),
              'layout': str(t.layout), 'logical_dense_bytes': t.numel() * t.element_size()}
    # Storage accounting needs special handling for compressed/quantized tensors.
    result['storage_bytes'] = (
        t.untyped_storage().nbytes()
        if t.device.type != 'meta' and t.layout == torch.strided else None
    )
    return result


def _walk_tensors(value, prefix='state'):
    import torch
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_tensors(child, f'{prefix}.{key}')
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from _walk_tensors(child, f'{prefix}.{i}')


def _storage_totals(tensors):
    """Deduplicate actual dense storage globally, including shared views."""
    import torch
    seen, totals, unsupported = set(), Counter(), 0
    for t in tensors:
        if t.device.type == 'meta' or t.layout != torch.strided:
            unsupported += 1
            continue
        storage = t.untyped_storage()
        key = (str(t.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            totals[str(t.device)] += storage.nbytes()
    return {'bytes_by_device': dict(totals), 'unsupported_or_meta_tensors': unsupported}


def collect_workload(model, optimizer=None, *, training=None,
                     training_script=None, target_hardware=None,
                     input_specs=None, master_weight_tensors=None):
    """Read-only live inspection. Pass explicit master tensors if exposed.

    input_specs example: [{'name':'input_ids', 'shape':[8,512], 'dtype':'int64'}].
    target_hardware describes the TARGET; no local GPU is queried automatically.
    Actual gradients/states can be absent before backward/step; absence does not
    mean future memory cost is zero. No arbitrary optimizer state inference.
    """
    import torch
    training = dict(training or {})
    params = list(model.named_parameters())
    buffers = list(model.named_buffers())
    gradients = [(n, p.grad) for n, p in params if p.grad is not None]
    states, groups = [], []
    if optimizer is not None:
        for i, state in enumerate(optimizer.state.values()):
            states.extend(_walk_tensors(state, f'parameter_state.{i}'))
        for group in optimizer.param_groups:
            options = {k: v for k, v in group.items()
                       if k != 'params' and isinstance(v, (str, int, float, bool, type(None)))}
            options['parameter_count'] = sum(p.numel() for p in group['params'])
            options['trainable_parameter_count'] = sum(
                p.numel() for p in group['params'] if p.requires_grad)
            groups.append(options)
    master = list(master_weight_tensors or [])
    categories = {'weights': params, 'buffers': buffers, 'gradients': gradients,
                  'optimizer_states': states,
                  'explicit_master_weights': [(f'master.{i}', t) for i, t in enumerate(master)]}
    inventory = {k: [_tensor_info(n, t) for n, t in items]
                 for k, items in categories.items()}
    # Count gradient estimates using each parameter's dtype. Label this assumption.
    estimated_grad = sum(p.numel() * p.element_size() for _, p in params if p.requires_grad)
    config = getattr(model, 'config', None)
    architecture = {}
    for name in ('model_type', 'num_hidden_layers', 'n_layer', 'hidden_size', 'n_embd',
                 'num_attention_heads', 'n_head', 'num_key_value_heads',
                 'intermediate_size', 'vocab_size', 'max_position_embeddings'):
        value = getattr(config, name, None)
        if isinstance(value, (str, int, float, bool)):
            architecture[name] = value
    all_tensors = [t for items in categories.values() for _, t in items]
    missing = [name for name in REQUIRED if training.get(name) is None]
    if training.get('max_steps') is not None:
        missing = [n for n in missing if n not in ('num_examples', 'epochs')]
    return {
        'schema_version': 2,
        'torch_version': torch.__version__,
        'model': {'class': type(model).__name__,
                  'total_params': sum(p.numel() for _, p in params),
                  'trainable_params': sum(p.numel() for _, p in params if p.requires_grad),
                  'architecture': architecture,
                  'module_types': dict(Counter(type(m).__name__ for m in model.modules()))},
        'tensor_inventory': inventory,
        'observed_storage': _storage_totals(all_tensors),
        'dense_gradient_estimate_bytes': estimated_grad,
        'gradient_estimate_assumption': 'All trainable parameters receive dense gradients '
                                        'stored in their parameter dtype.',
        'optimizer': {'class': type(optimizer).__name__ if optimizer is not None else None,
                      'parameter_groups': groups,
                      'observed_state_tensor_count': len(states),
                      'future_state_memory_bytes': None},
        'training': training,
        'input_specs': input_specs,
        'target_hardware': target_hardware,
        'script_scan': scan_training_script(training_script) if training_script else None,
        'missing_training_fields': missing,
        'still_required': [
            'Input shapes/sequence lengths, padding/packing policy and task/loss.',
            'Optimizer-specific future state shapes/dtypes if not initialized.',
            'Explicit master-copy policy; not reliably discoverable across frameworks.',
            'Activation/workspace estimator matching architecture and kernels.',
            'Target GPU capacity, precision-specific compute and bandwidth.',
            'Runtime model: operation counts, utilization assumptions, data loading, '
            'evaluation/checkpoint overhead and distributed communication.',
        ],
        'limitations': [
            'Collection runs no training or benchmark; observed memory is not peak memory.',
            'Before backward/step, missing gradients and optimizer states are not zero-cost.',
            'Observed storage is current-device storage, not target GPU memory.',
            'Meta tensors have logical sizes but no allocated storage.',
            'Quantization metadata, sparse storage, offload and distributed sharding '
            'require specialized handling; logical sizes are not sufficient.',
            'Parameter counts deduplicate shared Parameter objects, but distinct '
            'Parameter views can still overlap storage.',
            'Optimizer-exposed master tensors may already appear in optimizer_states; '
            'global observed storage is deduplicated.',
            'No runtime or peak-memory prediction is invented for unsupported setups.',
        ],
    }


def save_report(report, path):
    Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_config(path):
    """Load a JSON/YAML object, without executing Python configuration code."""
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as exc:
            raise ValueError('YAML requires PyYAML; alternatively use JSON.') from exc
        config = yaml.safe_load(text)
    else:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise ValueError(f'{path}: config must be an object/mapping.')
    # Restrict YAML to JSON-compatible configuration values.
    json.dumps(config, allow_nan=False)
    return config


def initialize_meta_model(model_script, config, *, factory='build_model',
                          parameter_dtype=None):
    """Import trusted model code and call factory(config) under torch.device(meta).

    No optimizer is constructed. Rejects returned non-meta tensors and lazy
    tensors with unresolved shapes. This post-check is not an allocation guard.
    """
    import torch
    path = Path(model_script).resolve(strict=True)
    if not path.is_file():
        raise ValueError('model_script must be a Python file.')
    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16,
                 'bf16': torch.bfloat16, 'fp64': torch.float64}
    if parameter_dtype is not None and parameter_dtype not in dtype_map:
        raise ValueError(f'Unsupported parameter dtype: {parameter_dtype}')
    module_name = '_workload_model_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot import model file: {path}')
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    sys.path.insert(0, str(path.parent))
    sys.modules[module_name] = module
    try:
        with torch.device('meta'):
            spec.loader.exec_module(module)
            builder = getattr(module, factory, None)
            if not callable(builder):
                raise ValueError(f'{path.name} must expose {factory}(config) returning nn.Module.')
            model = builder(dict(config))
            if not isinstance(model, torch.nn.Module):
                raise ValueError('Model factory must return a torch.nn.Module.')
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                if torch.nn.parameter.is_lazy(tensor):
                    raise ValueError(f'{name}: unresolved lazy shape; use explicit layer dimensions.')
                if tensor.device.type != 'meta':
                    raise ValueError(f'{name} is on {tensor.device}, not meta. '
                                     'Remove explicit CPU/CUDA allocation from the factory.')
            if parameter_dtype is not None:
                model = model.to(dtype=dtype_map[parameter_dtype])
        return model
    finally:
        sys.path[:] = old_path
        sys.modules.pop(module_name, None)


def collect_from_files(model_script, training_script, model_config, *,
                       training_config=None, hardware_config=None,
                       factory='build_model', parameter_dtype=None):
    """Configs are file paths. All training AST findings remain unverified hints."""
    # Parse files first, before executing any model code.
    scan = scan_training_script(training_script)
    config = load_config(model_config)
    training = load_config(training_config) if training_config else {}
    hardware = load_config(hardware_config) if hardware_config else None
    model = initialize_meta_model(model_script, config, factory=factory,
                                  parameter_dtype=parameter_dtype)
    report = collect_workload(model, training=training,
                              target_hardware=hardware,
                              input_specs=training.get('input_specs'))
    report['collection_mode'] = 'meta_initialization'
    report['script_scan'] = scan
    report['model']['provided_config'] = config
    report['model']['logical_weight_bytes'] = sum(
        t['logical_dense_bytes'] for t in report['tensor_inventory']['weights'])
    report['model']['logical_buffer_bytes'] = sum(
        t['logical_dense_bytes'] for t in report['tensor_inventory']['buffers'])
    dtype_bytes = {'fp64': 8, 'fp32': 4, 'fp16': 2, 'bf16': 2}
    gradient_dtype = training.get('gradient_dtype')
    if gradient_dtype is not None:
        if gradient_dtype not in dtype_bytes:
            raise ValueError('gradient_dtype must be fp64, fp32, fp16 or bf16.')
        report['dense_gradient_estimate_bytes'] = (
            report['model']['trainable_params'] * dtype_bytes[gradient_dtype])
        report['gradient_estimate_assumption'] = (
            f'All trainable parameters receive dense {gradient_dtype} gradients; '
            'explicit training configuration, not observed gradients.')
    report['configured_optimizer'] = {
        'name': training.get('optimizer'),
        'state_dtype': training.get('optimizer_state_dtype'),
        'options': training.get('optimizer_options'),
        'master_weights': training.get('master_weights'),
        'note': 'User configuration only. Future optimizer tensors are not initialized.'}
    for field in ('optimizer', 'optimizer_state_dtype', 'gradient_dtype'):
        if training.get(field) is None:
            report['missing_training_fields'].append(field)
    if training.get('input_specs') is None and training.get('sequence_length') is None:
        report['missing_training_fields'].append('input_specs_or_sequence_length')
    report['provenance'] = {
        'model_script': str(Path(model_script).resolve()),
        'training_script': str(Path(training_script).resolve()),
        'model_config': str(Path(model_config).resolve()),
        'training_config': str(Path(training_config).resolve()) if training_config else None,
        'hardware_config': str(Path(hardware_config).resolve()) if hardware_config else None,
        'factory': factory,
        'parameter_dtype_override': parameter_dtype,
        'settings_policy': 'Only explicit training config is confirmed. '
                           'AST candidates are not automatically promoted to settings.'}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--script', help='Legacy AST-only mode; mutually exclusive with model mode.')
    parser.add_argument('--model-script')
    parser.add_argument('--training-script')
    parser.add_argument('--model-config', help='JSON or YAML passed as one dict to the factory.')
    parser.add_argument('--training-config', help='Optional explicit training settings JSON/YAML.')
    parser.add_argument('--hardware-config', help='Optional target hardware JSON/YAML.')
    parser.add_argument('--factory', default='build_model')
    parser.add_argument('--parameter-dtype', choices=('fp32', 'fp16', 'bf16', 'fp64'))
    parser.add_argument('--output', default='workload.json')
    args = parser.parse_args()
    if args.script:
        if any((args.model_script, args.training_script, args.model_config,
                args.training_config, args.hardware_config, args.parameter_dtype)):
            parser.error('--script cannot be combined with model-mode arguments.')
    elif not all((args.model_script, args.training_script, args.model_config)):
        parser.error('Provide --model-script, --training-script and --model-config.')
    inputs = [p for p in (args.script, args.model_script, args.training_script,
                          args.model_config, args.training_config, args.hardware_config) if p]
    if Path(args.output).resolve() in {Path(p).resolve() for p in inputs}:
        parser.error('--output must not overwrite an input file.')
    try:
        if args.script:
            report = {'script_scan': scan_training_script(args.script),
                      'note': 'AST hints only; no model was initialized.'}
        else:
            report = collect_from_files(
                args.model_script, args.training_script, args.model_config,
                training_config=args.training_config, hardware_config=args.hardware_config,
                factory=args.factory, parameter_dtype=args.parameter_dtype)
        save_report(report, args.output)
    except Exception as exc:
        parser.exit(1, f'Collection failed ({type(exc).__name__}): {exc}\n')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
