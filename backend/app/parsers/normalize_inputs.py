"""Canonical analysis input schema, validation and normalization.

Uploaded files, Hugging Face metadata, catalogue lookups and manual entry all populate the
SAME flat field map (``values``: dotted key -> value). Each value carries provenance in a
parallel map. This module validates that map and turns it into the normalized estimator
input bundle (model/training metadata for the collector, dataset, hardware, deployment,
assumptions and pricing). It never executes uploaded code.
"""
from __future__ import annotations

import math

from ..services.estimators.profiles import PROFILES, profile_issues

SOURCES = ('upload', 'static_extraction', 'huggingface', 'dataset_sample', 'manual',
           'catalogue', 'default')

OPTIMIZERS = ('AdamW', 'Adam', 'SGD', 'Adagrad', 'RMSprop')
STRATEGIES = ('single', 'ddp', 'zero1', 'zero2', 'zero3', 'fsdp_full_shard', 'fsdp_shard_grad_op')
FLOAT_DTYPES = ('fp32', 'fp16', 'bf16')
TENSOR_DTYPES = ('float32', 'float16', 'bfloat16', 'int64', 'int32')

MLP, TX = 'mlp_regression', 'transformer_lm_classic'


def F(key, label, kind, section, *, options=None, profiles=None, advanced=False, unit=None,
      help=None, default=None, minimum=None, group=None):
    return {'key': key, 'label': label, 'type': kind, 'section': section, 'options': options,
            'profiles': profiles, 'advanced': advanced, 'unit': unit, 'help': help,
            'default': default, 'minimum': minimum, 'group': group}


FIELDS = [
    # 01 Model
    F('model.profile', 'Prediction profile', 'select', 'model', options=list(PROFILES),
      help='PyTorch only. Choose the profile that exactly matches your model.'),
    F('model.hf_model_id', 'Hugging Face model id', 'text', 'model', advanced=True,
      help='Informational; metadata comes from config.json only.'),
    F('model.model_type', 'Model type (from config)', 'text', 'model', advanced=True),
    F('model.input_size', 'Input features (I)', 'int', 'model', profiles=[MLP], minimum=1),
    F('model.hidden_size', 'Hidden size (H)', 'int', 'model', profiles=[MLP, TX], minimum=1),
    F('model.output_size', 'Output features (O)', 'int', 'model', profiles=[MLP], minimum=1),
    F('model.vocab_size', 'Vocabulary size (V)', 'int', 'model', profiles=[TX], minimum=1),
    F('model.num_hidden_layers', 'Layers (L)', 'int', 'model', profiles=[TX], minimum=1),
    F('model.num_attention_heads', 'Attention heads (A)', 'int', 'model', profiles=[TX], minimum=1),
    F('model.position_embedding', 'Position embedding', 'select', 'model', profiles=[TX],
      options=['learned', 'none'], help='"none" = no parameterized positions (e.g. rotary/ALiBi).'),
    F('model.max_position_embeddings', 'Max positions', 'int', 'model', profiles=[TX], minimum=1),
    F('model.linear_bias', 'Linear layers have bias', 'bool', 'model', profiles=[MLP, TX],
      default=True, help='PyTorch nn.Linear default is bias=True.'),
    F('model.tie_word_embeddings', 'LM head tied to token embedding', 'bool', 'model',
      profiles=[TX], default=True),
    F('model.layer_norm_bias', 'LayerNorm has bias', 'bool', 'model', profiles=[TX],
      advanced=True, default=True),
    F('model.lm_head_bias', 'LM head has bias', 'bool', 'model', profiles=[TX], advanced=True,
      default=False),
    F('model.num_key_value_heads', 'Key/value heads', 'int', 'model', profiles=[TX],
      advanced=True, minimum=1, help='Must equal attention heads (no GQA/MQA).'),
    F('model.intermediate_size', 'Feed-forward width', 'int', 'model', profiles=[TX],
      advanced=True, minimum=1, help='Must be 4 × hidden size; empty means 4 × hidden size.'),
    F('model.mlp_activation', 'Feed-forward activation', 'select', 'model', profiles=[TX],
      advanced=True, options=['gelu', 'relu'], default='gelu'),
    # 02 Training setup
    F('training.task', 'Task', 'select', 'training',
      options=['regression', 'causal_language_modeling', 'masked_language_modeling']),
    F('training.loss', 'Loss', 'select', 'training', options=['mse', 'cross_entropy']),
    F('training.batch_size_per_gpu', 'Batch size per GPU', 'int', 'training', minimum=1),
    F('training.schedule_mode', 'Training length defined by', 'select', 'training',
      options=['epochs', 'max_steps'], default='epochs'),
    F('training.epochs', 'Epochs', 'int', 'training', minimum=1),
    F('training.max_steps', 'Max optimizer steps', 'int', 'training', minimum=1),
    F('training.gradient_accumulation_steps', 'Gradient accumulation steps', 'int', 'training',
      minimum=1, default=1),
    F('training.precision', 'Compute precision', 'select', 'training', options=list(FLOAT_DTYPES),
      help='Autocast/compute dtype. Does not determine parameter storage.'),
    F('training.parameter_dtype', 'Parameter storage dtype', 'select', 'training',
      options=list(FLOAT_DTYPES), help='Required explicitly; never inferred from compute precision.'),
    F('training.gradient_dtype', 'Gradient dtype', 'select', 'training', options=list(FLOAT_DTYPES),
      help='PyTorch stores .grad in the parameter dtype unless your code changes it.'),
    F('training.optimizer', 'Optimizer', 'select', 'training', options=list(OPTIMIZERS)),
    F('training.optimizer_state_dtype', 'Optimizer state dtype', 'select', 'training',
      options=list(FLOAT_DTYPES), help='torch.optim creates states in the parameter dtype.'),
    F('training.master_weights', 'Additional FP32 master weights', 'select', 'training',
      options=['none', 'fp32'], default='none',
      help='"fp32" only if your setup keeps an EXTRA fp32 copy (e.g. fp16/bf16 params + fp32 master).'),
    F('training.momentum', 'Momentum', 'float', 'training', advanced=True, minimum=0, default=0),
    F('training.amsgrad', 'AMSGrad', 'bool', 'training', advanced=True, default=False),
    F('training.centered', 'Centered RMSprop', 'bool', 'training', advanced=True, default=False),
    F('training.foreach', 'foreach optimizer kernels', 'select', 'training', advanced=True,
      options=['unknown', 'true', 'false'], default='unknown'),
    F('training.sequence_length', 'Sequence length (S)', 'int', 'training', profiles=[TX], minimum=1),
    F('training.attention_backend', 'Attention backend', 'select', 'training', profiles=[TX],
      options=['eager', 'standard', 'sdpa', 'flash_attention_2']),
    F('training.gradient_checkpointing', 'Gradient checkpointing', 'bool', 'training', default=False),
    F('training.input_dtype', 'Input dtype', 'select', 'training', options=list(TENSOR_DTYPES)),
    F('training.target_dtype', 'Target dtype', 'select', 'training', options=list(TENSOR_DTYPES)),
    F('training.drop_last', 'DataLoader drop_last', 'bool', 'training', default=False),
    F('training.sampler_drop_last', 'DistributedSampler drop_last', 'bool', 'training',
      advanced=True, default=False),
    F('assumptions.startup_minutes', 'Job startup/setup time', 'float', 'training', advanced=True,
      unit='min', minimum=0, default=0, help='Added once to elapsed job time (and cost).'),
    F('assumptions.kernel_workspace_mib', 'Kernel workspace allowance', 'float', 'training',
      advanced=True, unit='MiB', minimum=0, default=256),
    F('assumptions.runtime_overhead_mib', 'CUDA context/runtime allowance', 'float', 'training',
      advanced=True, unit='MiB', minimum=0, default=1536),
    F('assumptions.fragmentation_fraction', 'Allocator fragmentation allowance', 'float', 'training',
      advanced=True, minimum=0, default=0.15),
    F('assumptions.arithmetic_efficiency_fast', 'Arithmetic efficiency (fast scenario)', 'float',
      'training', advanced=True, minimum=0, default=0.45),
    F('assumptions.arithmetic_efficiency_slow', 'Arithmetic efficiency (slow scenario)', 'float',
      'training', advanced=True, minimum=0, default=0.15),
    F('assumptions.benchmark_steps', 'Benchmark measured optimizer steps', 'int', 'training',
      advanced=True, minimum=1, default=50),
    F('assumptions.benchmark_warmup_steps', 'Benchmark warm-up steps', 'int', 'training',
      advanced=True, minimum=0, default=10),
    F('assumptions.benchmark_setup_minutes', 'Benchmark provisioning/setup time', 'float',
      'training', advanced=True, unit='min', minimum=0, default=15),
    # 03 Dataset
    F('dataset.num_train_examples', 'Full training example count', 'int', 'dataset', minimum=1,
      help='The FULL training set, not the uploaded sample.'),
    F('dataset.train_size_gib', 'Full training data size on disk', 'float', 'dataset', unit='GiB',
      minimum=0, help='Needed for data-loading time unless data is assumed ready.'),
    F('dataset.num_validation_examples', 'Validation examples', 'int', 'dataset', advanced=True,
      minimum=0),
    F('dataset.padding', 'Padding', 'select', 'dataset', profiles=[TX],
      options=['max_length', 'longest_in_batch', 'packed'], default='max_length'),
    F('dataset.effective_sequence_length', 'Effective (padded) sequence length', 'int', 'dataset',
      profiles=[TX], minimum=1, advanced=True,
      help='Average padded length per batch; empty uses the sequence length.'),
    F('dataset.assume_data_ready', 'Assume data loading is hidden', 'bool', 'dataset',
      default=False, help='Excludes disk/host-to-GPU time from runtime. Explicit assumption.'),
    # Hardware
    F('hardware.source', 'Hardware source', 'select', 'hardware', options=['nebius', 'catalog', 'custom']),
    F('hardware.selection_id', 'Selected catalogue entry', 'text', 'hardware'),
    F('hardware.gpu_name', 'GPU', 'text', 'hardware'),
    F('hardware.catalog_gpu_id', 'Hardware catalogue id', 'text', 'hardware', advanced=True),
    F('hardware.total_vram_gib', 'Total VRAM per GPU', 'float', 'hardware', unit='GiB', minimum=0),
    F('hardware.usable_vram_gib', 'Usable VRAM limit per GPU', 'float', 'hardware', unit='GiB',
      minimum=0, help='Defaults to 90% of total VRAM.'),
    F('hardware.gpu_count_available', 'GPUs available (identical)', 'int', 'hardware', minimum=1,
      default=1),
    F('training.gpu_count_used', 'GPUs used by this training config', 'int', 'hardware', minimum=1,
      default=1),
    F('training.distributed_strategy', 'Distribution / sharding strategy', 'select', 'hardware',
      options=list(STRATEGIES), default='single',
      help='Multiple GPUs do not imply sharding; choose explicitly.'),
    F('training.gpus_per_node', 'GPUs per node', 'int', 'hardware', advanced=True, minimum=1),
    F('training.largest_gather_unit_params', 'Largest ZeRO-3/FSDP gather unit', 'int', 'hardware',
      advanced=True, minimum=1, unit='params'),
    F('training.offload_optimizer', 'Optimizer state location', 'select', 'hardware',
      advanced=True, options=['gpu', 'cpu', 'nvme'], default='gpu'),
    F('training.offload_parameters', 'Parameter location', 'select', 'hardware', advanced=True,
      options=['gpu', 'cpu', 'nvme'], default='gpu'),
    F('hardware.peak_tflops_fp32', 'Peak FP32', 'float', 'hardware', advanced=True, unit='TFLOPS', minimum=0),
    F('hardware.peak_tflops_bf16', 'Peak BF16 tensor (dense)', 'float', 'hardware', advanced=True,
      unit='TFLOPS', minimum=0),
    F('hardware.peak_tflops_fp16', 'Peak FP16 tensor (dense)', 'float', 'hardware', advanced=True,
      unit='TFLOPS', minimum=0),
    F('hardware.memory_bandwidth_gb_s', 'GPU memory bandwidth', 'float', 'hardware', advanced=True,
      unit='GB/s', minimum=0),
    F('hardware.price_per_hour', 'Price', 'float', 'hardware', unit='USD', minimum=0),
    F('hardware.price_unit', 'Price unit', 'select', 'hardware', options=['gpu_hour', 'instance_hour']),
    F('hardware.gpus_per_instance', 'GPUs per billed instance', 'int', 'hardware', minimum=1,
      help='Only used when the price is per instance-hour.'),
    F('hardware.price_region', 'Price region', 'text', 'hardware', advanced=True),
    F('hardware.price_note', 'Price note', 'text', 'hardware', advanced=True),
    F('hardware.cpu_ram_gib', 'CPU RAM per node', 'float', 'hardware', advanced=True, unit='GiB', minimum=0),
    F('hardware.nvme_capacity_gib', 'NVMe/SSD capacity per node', 'float', 'hardware',
      advanced=True, unit='GiB', minimum=0),
    F('hardware.storage_read_gb_s', 'NVMe/SSD sequential read', 'float', 'hardware', advanced=True,
      unit='GB/s', minimum=0),
    F('hardware.cpu_gpu_gb_s', 'CPU–GPU effective bandwidth', 'float', 'hardware', advanced=True,
      unit='GB/s', minimum=0),
    F('hardware.gpu_gpu_gb_s', 'GPU–GPU/network effective bandwidth', 'float', 'hardware',
      advanced=True, unit='GB/s', minimum=0),
]
FIELD_INDEX = {f['key']: f for f in FIELDS}

STATE_COUNT_OPTIMIZERS = {'AdamW', 'Adam', 'Adagrad'}


def schema():
    return {'schema_version': 1, 'framework': 'pytorch', 'fields': FIELDS,
            'profiles': PROFILES, 'sources': list(SOURCES)}


def _present(v):
    return v is not None and v != ''


def _coerce(field, value):
    """Return (value, error). Accepts JSON-native values only; strings for numbers are parsed."""
    if not _present(value):
        return None, None
    kind = field['type']
    if kind == 'int':
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None, 'Enter a whole number.'
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or int(value) != value:
            return None, 'Enter a whole number.'
        value = int(value)
    elif kind == 'float':
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None, 'Enter a number.'
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None, 'Enter a number.'
    elif kind == 'bool':
        if not isinstance(value, bool):
            return None, 'Choose yes or no.'
    elif kind == 'select':
        if value not in field['options']:
            return None, f"Choose one of: {', '.join(map(str, field['options']))}."
    elif kind == 'text':
        if not isinstance(value, str) or len(value) > 300:
            return None, 'Enter text up to 300 characters.'
    if field.get('minimum') is not None and kind in ('int', 'float') and value < field['minimum']:
        return None, f"Must be at least {field['minimum']}."
    return value, None


def applies(field, profile):
    return field['profiles'] is None or profile in field['profiles']


def validate(values, provenance=None):
    """Validate a flat field map.

    Returns normalized values plus field-level errors/warnings, the fields required by each
    calculation, which calculations can run, and unsupported-profile explanations.
    """
    values = dict(values or {})
    provenance = provenance or {}
    unknown = sorted(set(values) - set(FIELD_INDEX))
    errors, warnings, unsupported = [], [], []
    for key in unknown:
        warnings.append({'field': key, 'message': 'Unknown field ignored.'})
    v = {}
    for key, field in FIELD_INDEX.items():
        value, err = _coerce(field, values.get(key))
        if err:
            errors.append({'field': key, 'message': err})
        v[key] = value
    profile = v['model.profile']
    if profile:
        for key, field in FIELD_INDEX.items():
            if not applies(field, profile):
                v[key] = None
    # Derived defaults that are documented framework behaviour, not guesses.
    if profile == TX and v['model.intermediate_size'] is None and v['model.hidden_size']:
        v['model.intermediate_size'] = 4 * v['model.hidden_size']
    if v['training.foreach'] is None:
        v['training.foreach'] = 'unknown'

    required = {}

    def need(key, calc):
        required.setdefault(key, [])
        if calc not in required[key]:
            required[key].append(calc)

    need('model.profile', 'memory')
    if profile == MLP:
        for k in ('model.input_size', 'model.hidden_size', 'model.output_size', 'model.linear_bias'):
            need(k, 'memory')
    elif profile == TX:
        for k in ('model.vocab_size', 'model.hidden_size', 'model.num_hidden_layers',
                  'model.num_attention_heads', 'model.position_embedding', 'model.tie_word_embeddings',
                  'model.linear_bias', 'model.layer_norm_bias', 'model.lm_head_bias',
                  'training.sequence_length', 'training.attention_backend'):
            need(k, 'memory')
        if v['model.position_embedding'] == 'learned':
            need('model.max_position_embeddings', 'memory')
    for k in ('training.task', 'training.loss', 'training.batch_size_per_gpu', 'training.precision',
              'training.parameter_dtype', 'training.gradient_dtype', 'training.optimizer',
              'training.master_weights', 'training.gradient_checkpointing', 'training.input_dtype',
              'training.target_dtype', 'training.gpu_count_used', 'training.distributed_strategy',
              'hardware.total_vram_gib', 'hardware.usable_vram_gib', 'training.gradient_accumulation_steps'):
        need(k, 'memory')
    opt = v['training.optimizer']
    if opt in STATE_COUNT_OPTIMIZERS or (opt in ('SGD', 'RMSprop')):
        if not (opt == 'SGD' and not v['training.momentum']):
            need('training.optimizer_state_dtype', 'memory')
    # Runtime requirements.
    mode = v['training.schedule_mode'] or 'epochs'
    if mode == 'epochs':
        need('training.epochs', 'runtime')
        need('dataset.num_train_examples', 'runtime')
    else:
        need('training.max_steps', 'runtime')
    precision = v['training.precision']
    need('hardware.memory_bandwidth_gb_s', 'runtime')
    need('hardware.peak_tflops_fp32', 'runtime')
    if precision in ('bf16', 'fp16'):
        need(f'hardware.peak_tflops_{precision}', 'runtime')
    if not v['dataset.assume_data_ready']:
        for k in ('dataset.train_size_gib', 'hardware.storage_read_gb_s', 'hardware.cpu_gpu_gb_s'):
            need(k, 'runtime')
        if mode == 'max_steps':
            need('dataset.num_train_examples', 'runtime')
    used = v['training.gpu_count_used'] or 1
    if used > 1:
        need('hardware.gpu_gpu_gb_s', 'runtime')
    need('hardware.price_per_hour', 'cost')
    need('hardware.price_unit', 'cost')
    if v['hardware.price_unit'] == 'instance_hour':
        need('hardware.gpus_per_instance', 'cost')

    missing = [{'field': k, 'required_for': calcs, 'label': FIELD_INDEX[k]['label']}
               for k, calcs in required.items() if v.get(k) is None]

    # Cross-field checks.
    model = {k.split('.', 1)[1]: v[k] for k in v if k.startswith('model.')}
    training = {k.split('.', 1)[1]: v[k] for k in v if k.startswith('training.')}
    if profile:
        perr, punsup = profile_issues(profile, model, training)
        errors += perr
        unsupported += punsup
    task, loss = v['training.task'], v['training.loss']
    if task and loss:
        expected = 'mse' if task == 'regression' else 'cross_entropy'
        if loss != expected:
            unsupported.append({'field': 'training.loss', 'message':
                                f'Task {task} is only modelled with {expected} loss.'})
    if profile == TX:
        for key in ('training.input_dtype', 'training.target_dtype'):
            if v[key] and v[key] not in ('int64', 'int32'):
                errors.append({'field': key, 'message': 'Token ids/labels must be an integer dtype.'})
    if profile == MLP:
        for key in ('training.input_dtype', 'training.target_dtype'):
            if v[key] and v[key] not in ('float32', 'float16', 'bfloat16'):
                errors.append({'field': key, 'message': 'MLP regression features/targets must be floating point.'})
    total, usable = v['hardware.total_vram_gib'], v['hardware.usable_vram_gib']
    if total is not None and total <= 0:
        errors.append({'field': 'hardware.total_vram_gib', 'message': 'Must be greater than 0.'})
    if total is not None and usable is not None and usable > total:
        errors.append({'field': 'hardware.usable_vram_gib', 'message': 'Cannot exceed total VRAM.'})
    if usable is not None and usable <= 0:
        errors.append({'field': 'hardware.usable_vram_gib', 'message': 'Must be greater than 0.'})
    avail = v['hardware.gpu_count_available']
    strategy = v['training.distributed_strategy']
    if avail is not None and used > avail:
        errors.append({'field': 'training.gpu_count_used', 'message':
                       f'Training uses {used} GPUs but only {avail} are available.'})
    if used > 1 and strategy == 'single':
        errors.append({'field': 'training.distributed_strategy', 'message':
                       'More than one GPU is used: choose DDP, ZeRO or FSDP explicitly.'})
    if used == 1 and strategy not in (None, 'single'):
        warnings.append({'field': 'training.distributed_strategy', 'message':
                         'A distributed strategy with one GPU adds no sharding benefit.'})
    gpn = v['training.gpus_per_node']
    if gpn is not None and used % gpn:
        errors.append({'field': 'training.gpus_per_node', 'message': 'Must divide the GPU count used.'})
    if strategy in ('zero3', 'fsdp_full_shard') and v['training.largest_gather_unit_params'] is None:
        warnings.append({'field': 'training.largest_gather_unit_params', 'message':
                         'Unknown gather unit: the whole model is budgeted as one gathered unit.'})
    if (v['training.offload_optimizer'] or 'gpu') != 'gpu' or (v['training.offload_parameters'] or 'gpu') != 'gpu':
        if strategy not in ('zero1', 'zero2', 'zero3') and strategy is not None:
            unsupported.append({'field': 'training.offload_optimizer', 'message':
                                'CPU/NVMe offload is only modelled with a ZeRO strategy.'})
        warnings.append({'field': 'training.offload_optimizer', 'message':
                         'Offload runtime needs explicit transfer volumes; runtime will be partial.'})
    if mode == 'epochs' and v['training.max_steps'] is not None:
        warnings.append({'field': 'training.max_steps', 'message': 'Ignored in epochs mode.'})
    B, N = v['training.batch_size_per_gpu'], v['dataset.num_train_examples']
    if B and N:
        per_rank = N // used if v['training.sampler_drop_last'] else math.ceil(N / used)
        if B > per_rank:
            if v['training.drop_last']:
                errors.append({'field': 'training.batch_size_per_gpu', 'message':
                               f'With drop_last=true, batch size {B} > {per_rank} examples per GPU '
                               'produces no training batches.'})
            else:
                warnings.append({'field': 'training.batch_size_per_gpu', 'message':
                                 f'Batch size {B} exceeds {per_rank} examples per GPU; each epoch is '
                                 'one partial batch (drop_last=false).'})
    if v['training.optimizer'] == 'SGD' and v['training.amsgrad']:
        warnings.append({'field': 'training.amsgrad', 'message': 'AMSGrad does not apply to SGD.'})
    if v['training.parameter_dtype'] and v['training.precision'] and \
            v['training.parameter_dtype'] != 'fp32' and v['training.parameter_dtype'] != v['training.precision']:
        unsupported.append({'field': 'training.parameter_dtype', 'message':
                            'Low-precision parameters must match compute precision (mixed fp16/bf16 '
                            'storage and compute is not modelled).'})
    if v['training.master_weights'] == 'fp32' and v['training.parameter_dtype'] == 'fp32':
        warnings.append({'field': 'training.master_weights', 'message':
                         'Parameters are already fp32; an extra fp32 master copy is unusual.'})
    S, eff = v['training.sequence_length'], v['dataset.effective_sequence_length']
    if S and eff and eff > S:
        errors.append({'field': 'dataset.effective_sequence_length', 'message':
                       'Effective length cannot exceed the sequence length.'})
    if profile == TX and v['dataset.padding'] == 'longest_in_batch' and eff is None:
        warnings.append({'field': 'dataset.effective_sequence_length', 'message':
                         'Padding to the longest in batch: runtime uses the full sequence length '
                         'unless you enter the average padded length. Memory uses the full length.'})
    for key, meta in provenance.items():
        if isinstance(meta, dict) and meta.get('status') in ('uncertain', 'unresolved') \
                and not meta.get('reviewed') and key in FIELD_INDEX:
            warnings.append({'field': key, 'message': meta.get('message') or
                             'Extracted value is uncertain; review it.'})
    suggestions = {}

    def suggest(key, value, reason):
        if value is not None and v.get(key) != value:
            suggestions[key] = {'value': value, 'reason': reason}

    pd = v['training.parameter_dtype']
    if pd:
        suggest('training.gradient_dtype', pd, 'PyTorch stores .grad in the parameter dtype.')
        suggest('training.optimizer_state_dtype', pd, 'torch.optim creates states like the parameters.')
    if total:
        suggest('hardware.usable_vram_gib', round(0.9 * total, 2), 'Default usable limit: 90% of total VRAM.')
    if profile == MLP:
        suggest('training.task', 'regression', 'The MLP profile supports regression only.')
        suggest('training.loss', 'mse', 'The MLP profile is modelled with MSE loss.')
        suggest('training.input_dtype', 'float32', 'Typical feature tensor dtype; confirm.')
        suggest('training.target_dtype', 'float32', 'Typical MSE target dtype; confirm.')
    elif profile == TX:
        suggest('training.loss', 'cross_entropy', 'Language modelling uses cross-entropy.')
        suggest('training.input_dtype', 'int64', 'PyTorch token ids are int64 by default.')
        suggest('training.target_dtype', 'int64', 'PyTorch labels are int64 by default.')
    if v['training.gpu_count_used'] and v['training.distributed_strategy'] is None:
        suggest('training.distributed_strategy', 'single', 'One GPU: no distribution.')
    blocking = {e['field'] for e in errors} | {u['field'] for u in unsupported}
    blocked_memory = bool(errors or unsupported) or any('memory' in m['required_for'] for m in missing)
    can = {
        'memory': not blocked_memory,
        'runtime': not blocked_memory and not any('runtime' in m['required_for'] for m in missing),
    }
    can['cost'] = can['runtime'] and not any('cost' in m['required_for'] for m in missing)
    return {'values': v, 'errors': errors, 'warnings': warnings, 'unsupported': unsupported,
            'missing': missing, 'required': required, 'can_estimate': can, 'suggestions': suggestions,
            'blocking_fields': sorted(blocking | {m['field'] for m in missing if 'memory' in m['required_for']})}


# ----------------------------------------------------------------------------- bundle

def _section(v, prefix):
    return {k.split('.', 1)[1]: val for k, val in v.items() if k.startswith(prefix + '.') and val is not None}


def model_settings(v):
    return _section(v, 'model')


def training_settings(v):
    t = _section(v, 'training')
    if t.get('schedule_mode', 'epochs') == 'epochs':
        t.pop('max_steps', None)
    else:
        t.pop('epochs', None)
    if v.get('dataset.num_train_examples') is not None:
        t['num_examples'] = v['dataset.num_train_examples']
    t['gpu_count'] = t.pop('gpu_count_used', 1)
    t['training_method'] = 'full'
    opts = {}
    for key in ('momentum', 'amsgrad', 'centered'):
        if key in t:
            opts[key] = t.pop(key)
    foreach = t.pop('foreach', 'unknown')
    if foreach != 'unknown':
        opts['foreach'] = foreach == 'true'
    t['optimizer_options'] = opts
    for key in ('offload_optimizer', 'offload_parameters', 'gpus_per_node',
                'largest_gather_unit_params', 'schedule_mode', 'loss'):
        t.pop(key, None)
    return t


def deployment(v):
    used = v.get('training.gpu_count_used') or 1
    d = {'strategy': v.get('training.distributed_strategy') or 'single', 'world_size': used,
         'gpus_per_node': v.get('training.gpus_per_node') or used}
    if v.get('training.largest_gather_unit_params'):
        d['largest_gather_unit_params'] = v['training.largest_gather_unit_params']
    off = {}
    if (v.get('training.offload_optimizer') or 'gpu') != 'gpu':
        off['optimizer'] = v['training.offload_optimizer']
    if (v.get('training.offload_parameters') or 'gpu') != 'gpu':
        off['parameters'] = v['training.offload_parameters']
    if off:
        d['offload'] = off
    return d


def dataset(v):
    ds = {}
    if v.get('dataset.num_train_examples') is not None:
        ds['num_train_examples'] = v['dataset.num_train_examples']
    if v.get('dataset.train_size_gib') is not None:
        ds['train_on_disk_size_bytes'] = int(v['dataset.train_size_gib'] * 1024 ** 3)
    if v.get('dataset.num_validation_examples') is not None:
        ds['num_validation_examples'] = v['dataset.num_validation_examples']
    batching = {'drop_last': bool(v.get('training.drop_last'))}
    if v.get('dataset.padding'):
        batching['padding'] = v['dataset.padding']
    if v.get('dataset.effective_sequence_length'):
        batching['effective_sequence_length'] = v['dataset.effective_sequence_length']
    return {'dataset': ds, 'batching': batching}


def hardware_json(v):
    """Machine-format hardware.json accepted by both estimators. Per GPU / per node."""
    total, usable = v.get('hardware.total_vram_gib'), v.get('hardware.usable_vram_gib')
    gpu = {'gpu_id': v.get('hardware.catalog_gpu_id') or v.get('hardware.selection_id') or 'custom',
           'name': v.get('hardware.gpu_name') or 'Custom GPU',
           'count_available': v.get('hardware.gpu_count_available') or 1,
           'count_used': v.get('training.gpu_count_used') or 1}
    if total is not None:
        gpu['vram_gib_per_gpu'] = total
        gpu['other_usage_gib_per_gpu'] = 0
        gpu['safety_reserve_gib_per_gpu'] = max(0.0, total - (usable if usable is not None else total))
    peaks = {}
    for src, dst in (('fp32', 'fp32'), ('bf16', 'bf16_tensor_dense'), ('fp16', 'fp16_tensor_dense')):
        if v.get(f'hardware.peak_tflops_{src}') is not None:
            peaks[dst] = v[f'hardware.peak_tflops_{src}']
    if peaks:
        gpu['peak_compute_tflops'] = peaks
    if v.get('hardware.memory_bandwidth_gb_s') is not None:
        gpu['memory_bandwidth_gb_per_second'] = v['hardware.memory_bandwidth_gb_s']
    hw = {'schema_version': 1, 'gpu': gpu,
          'notes': ['GPU fields are per GPU; CPU RAM and storage are per node.',
                    'Usable VRAM is expressed as safety_reserve = total - usable limit; other '
                    'GPU usage is set to 0 because the usable limit already reserves headroom.']}
    if v.get('hardware.cpu_ram_gib') is not None:
        hw['cpu'] = {'ram_gib': v['hardware.cpu_ram_gib']}
    storage = {}
    if v.get('hardware.nvme_capacity_gib') is not None:
        storage['capacity_gib'] = v['hardware.nvme_capacity_gib']
    if v.get('hardware.storage_read_gb_s') is not None:
        storage['sequential_read_gb_per_second'] = v['hardware.storage_read_gb_s']
    if storage:
        hw['storage'] = storage
    link = {}
    if v.get('hardware.cpu_gpu_gb_s') is not None:
        link['cpu_gpu_effective_gb_per_second'] = v['hardware.cpu_gpu_gb_s']
    if v.get('hardware.gpu_gpu_gb_s') is not None:
        link['gpu_gpu_effective_gb_per_second'] = v['hardware.gpu_gpu_gb_s']
    if link:
        hw['interconnect'] = link
    return hw


def pricing(v):
    return {'price_per_hour': v.get('hardware.price_per_hour'),
            'unit': v.get('hardware.price_unit'),
            'gpus_per_instance': v.get('hardware.gpus_per_instance'),
            'currency': 'USD', 'region': v.get('hardware.price_region'),
            'note': v.get('hardware.price_note'),
            'gpus_allocated': v.get('hardware.gpu_count_available') or 1,
            'gpus_used': v.get('training.gpu_count_used') or 1}


def assumptions(v):
    def get(key):
        value = v.get('assumptions.' + key)
        return FIELD_INDEX['assumptions.' + key]['default'] if value is None else value
    memory = {'kernel_workspace_mib': get('kernel_workspace_mib'),
              'runtime_overhead_mib': get('runtime_overhead_mib'),
              'fragmentation_fraction': get('fragmentation_fraction')}
    runtime = {'arithmetic_efficiency_fast': get('arithmetic_efficiency_fast'),
               'arithmetic_efficiency_slow': get('arithmetic_efficiency_slow'),
               'startup_seconds': get('startup_minutes') * 60,
               'assume_data_ready': bool(v.get('dataset.assume_data_ready'))}
    benchmark = {'steps': get('benchmark_steps'), 'warmup_steps': get('benchmark_warmup_steps'),
                 'setup_minutes': get('benchmark_setup_minutes')}
    return {'memory': memory, 'runtime': runtime, 'benchmark': benchmark}


def defaults():
    """Default values (provenance 'default') for fields with documented defaults."""
    return {f['key']: f['default'] for f in FIELDS if f['default'] is not None}
