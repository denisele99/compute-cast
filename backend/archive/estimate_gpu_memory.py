#!/usr/bin/env python3
"""Metadata GPU/CPU/NVMe memory planner (v2); Python 3.10+, stdlib only.

  python estimate_gpu_memory.py workload.json --output memory.json
  python estimate_gpu_memory.py workload.json --deployment deployment.json \
      --graph graph.json --hardware hardware.json --output memory.json
  python estimate_gpu_memory.py --write-examples examples
  python estimate_gpu_memory.py --formulas

Inputs are collect_workload JSON plus optional explicit deployment/graph JSON.
API: estimate_gpu_memory(workload, assumptions=None, hardware=None,
                         deployment=None, graph=None).
Deployment presets: single, ddp, zero1/2/3, fsdp_full_shard,
fsdp_shard_grad_op, fsdp_hybrid_shard. Uniform one-process-per-GPU ownership;
GPU/CPU/NVMe state placement; CPU offload staging/optimizer scratch, gathered
weights/gradients and communication buffers are budgeted separately.
This is planning, NOT an emulator of actual DeepSpeed/FSDP allocations.
Native framework configs require translation into the explicit deployment schema.
Tensor/pipeline parallelism still needs rank-specific model inventories and is
rejected rather than guessed. Meta inventories must describe GLOBAL pre-shard
parameters, not a local already-sharded model or an FSDP FlatParameter snapshot.

Different computation graphs are supported via COMPLETE tensor lifetime manifests:
scope=per_rank, complete_training_step=true, excludes_model_states=true.
Each tensor has id, shape+dtype (or bytes), born, last_use, device; optional
storage_id/storage_bytes expresses aliasing. Lifetimes are inclusive. Include
forward AND backward inputs/outputs/loss/activations/transient tensors. Exclude
model states, optimizer scratch and distributed buffers already modeled elsewhere.
Do not divide local-batch activation memory by world_size. For checkpointing,
encode recomputed tensors' real lifetimes in the manifest. Parameters alone cannot
reveal a graph. Arbitrary operation names need no architecture-specific formula
once COMPLETE lifetimes and full underlying storage sizes are supplied.
workspace_complete=true skips generic kernel workspace allowances; otherwise
explicit graph scratch plus generic allowances are budgeted conservatively.

Without graph metadata, analytic simple_mlp and transformer_classic16 profiles
remain available. See FORMULAS for source citations and assumptions. Custom
saved_activation_bytes+auxiliary_tensor_bytes overrides are also retained.
All original allowance keys remain supported: optimizer/master/cache bytes,
kernel_workspace_[low/high]_mib and _fraction_[low/high],
optimizer_workspace_[low/high]_bytes, runtime_overhead_[low/high]_mib,
fragmentation_fraction_[low/high], activation_multiplier_[low/high],
additional_persistent_bytes, activation_dtype and activation_model.

Defaults: kernel workspace 64/256 MiB plus 5/20% of transient tensors;
runtime 512/1536 MiB; fragmentation 5/15%. These are UNSOURCED, editable
engineering allowances, not statistical limits or measured hardware constants.
Optimizer and loss workspace are likewise explicit engineering assumptions.
Offload tile default <=1e8 parameters and communication bucket allowance <=25 MiB
are planning choices, NOT assertions about framework default allocation sizes.
--formulas and each output include the full formula/source catalog.
No code from the model/training script executes; no GPU/PyTorch is required.
CPU/NVMe capacity checks cover modeled job allocations, not the entire machine.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from app.helpers import load_json

GIB = 1024 ** 3
MIB = 1024 ** 2
DTYPE_BYTES = {'fp64': 8, 'float64': 8, 'double': 8,
               'fp32': 4, 'float32': 4, 'float': 4,
               'fp16': 2, 'float16': 2, 'half': 2,
               'bf16': 2, 'bfloat16': 2,
               'int64': 8, 'long': 8, 'int32': 4, 'int16': 2,
               'int8': 1, 'uint8': 1, 'bool': 1}
SOURCES = {
    'tensor_size': 'https://docs.pytorch.org/docs/stable/generated/torch.Tensor.element_size.html',
    'autograd': 'https://docs.pytorch.org/docs/stable/notes/autograd.html#saved-tensors',
    'derivatives': 'https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml',
    'adam': 'https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html',
    'adamw': 'https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html',
    'sgd': 'https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html',
    'adagrad': 'https://docs.pytorch.org/docs/stable/generated/torch.optim.Adagrad.html',
    'rmsprop': 'https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html',
    'transformer': 'https://arxiv.org/html/2205.05198v1#S4.SS1',
    'amp': 'https://docs.pytorch.org/docs/stable/amp.html',
    'cuda': 'https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management',
}
FORMULAS = {
    'tensor_bytes': {
        'formula': 'numel * bytes_per_element; numel = product(shape)',
        'basis': 'Dense tensor storage accounting, excluding aliasing, padding and metadata.',
        'sources': ['tensor_size']},
    'model_states': {
        'formula': 'weights + buffers + P_trainable*b_grad + optimizer_states + extra_master_weights',
        'basis': 'Sum of separately stored dense tensors; not an extra universal bytes/parameter constant.',
        'sources': ['tensor_size', 'adamw', 'amp']},
    'optimizer_states': {
        'formula': 'sum_over_groups(P_group * state_count * b_state)',
        'state_counts': {'sgd': '1 if momentum>0 else 0', 'adam_adamw': '2 + int(amsgrad)',
                         'adagrad': '1', 'rmsprop': '1 + int(momentum>0) + int(centered)'},
        'basis': 'Derived by counting algorithm state buffers. Assumes one shared state dtype; '
                 'all trainable parameters optimized and eventually receive gradients. '
                 'Tiny per-tensor step counters omitted unless an explicit override includes them.',
        'sources': ['sgd', 'adam', 'adamw', 'adagrad', 'rmsprop']},
    'master_weights': {
        'formula': 'P_trainable * 4 if an ADDITIONAL FP32 master copy exists; otherwise 0',
        'basis': 'Tensor accounting, conditioned on implementation. FP32 model parameters '
                 'are not counted a second time merely because autocast is enabled.',
        'sources': ['tensor_size', 'amp']},
    'autocast_cache': {
        'formula': 'eligible_FP32_parameters * lower_precision_bytes',
        'basis': 'Potential additional cast-weight storage, distinct from a master copy; '
                 'assumes all FP32 trainable parameters eligible when low-precision compute '
                 'is configured. Explicit override can correct cache policy and eligibility.',
        'sources': ['amp']},
    'mlp_saved': {
        'formula': 'B * (input_features + hidden_features) * activation_bytes',
        'basis': 'Original tensor-inventory derivation for Linear-ReLU-Linear. Save the '
                 'first linear input and ReLU output; shared saved storage counted once. '
                 'Weights excluded; original input is not added again if it aliases the saved input. '
                 'Assumes full training and ordinary PyTorch eager autograd.',
        'sources': ['autograd', 'derivatives', 'tensor_size']},
    'transformer_saved': {
        'formula': 'L*B*S*H*(34 + 5*A*S/H) = 34*L*B*S*H + 5*L*B*A*S*S bytes',
        'basis': 'Korthikanti et al., 2022, Section 4.1, Equation 1, multiplied by L. '
                 'Single-stack conventional MHA; 4H GeLU feed-forward, dropout, 16-bit '
                 'activations plus one-byte masks; no checkpointing. Excludes output/loss. '
                 'Not applied to FlashAttention, GQA, gated MLPs or FP32.',
        'sources': ['transformer']},
    'auxiliary': {
        'formula': 'extra_input_bytes + output_elements*b_output + loss_buffer_allowance',
        'basis': 'Tensor-size derivation. For MLP: prediction B*O elements; target and original '
                 'input from specs, excluding saved-input alias. For token LM: logits B*S*V. '
                 'Loss allowance is HEURISTIC: 1/2 extra FP32 output-sized buffers in the '
                 'low/high scenarios (at least FP32 for MSE); not a universal loss implementation formula.',
        'sources': ['tensor_size', 'autograd']},
    'optimizer_workspace': {
        'formula': 'low = largest_parameter_numel*max(b_state,b_weight); '
                   'high = low + total_trainable_weight_bytes if foreach enabled/unknown, '
                   'otherwise low (unless explicitly overridden)',
        'basis': 'Engineering allowance for scratch. PyTorch documents approximately '
                 'sizeof(params) ADDITIONAL peak memory for foreach vs for-loop. The largest '
                 'parameter allowance is UNSOURCED heuristic, not a bound on for-loop/fused.',
        'sources': ['adam']},
    'kernel_workspace': {
        'formula': 'fixed_workspace + fraction*(saved_activations + auxiliary_tensors)',
        'basis': 'UNSOURCED configurable heuristic. No universal metadata-only kernel '
                 'workspace formula. Defaults are example planning values, not measured limits.',
        'sources': []},
    'planning_total': {
        'formula': 'T = states + cast_cache + extra_state + activations + auxiliary + '
                   'kernel_workspace + optimizer_workspace; total = T*(1+fragmentation_fraction) '
                   '+ runtime_overhead',
        'basis': 'Original conservative co-residency planning model. Fragmentation and CUDA '
                 'overhead defaults are UNSOURCED assumptions. CUDA docs explain allocated vs '
                 'reserved memory, but do not prescribe these fractions. Scenario endpoints '
                 'are NOT bounds, probabilities, or a measured max over execution time.',
        'sources': ['cuda']},
    'capacity': {
        'formula': 'usable = total_vram - other_usage - user_safety_reserve',
        'basis': 'Capacity accounting. Compare planning scenarios to usable memory. '
                 'Uses explicit byte/GiB capacity, not manufacturer GB labels.',
        'sources': ['tensor_size']},
    'units': {'formula': 'GiB = bytes / 2**30; MiB = bytes / 2**20',
              'basis': 'Binary unit definitions; bandwidth GB/s is not used in this estimator.',
              'sources': []},
}


class Unsupported(ValueError):
    """Metadata is valid but insufficient for a supported memory model."""


def number(value, name, *, integer=False, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < minimum or (integer and int(value) != value):
        raise ValueError(f'{name} must be >= {minimum}' + (' and integral' if integer else ''))
    return int(value) if integer else value


def dtype_bytes(dtype):
    key = str(dtype).lower().removeprefix('torch.')
    if key not in DTYPE_BYTES:
        raise Unsupported(f'Unsupported storage dtype {dtype!r}; quantization needs explicit accounting.')
    return DTYPE_BYTES[key]


def count(value, name):
    return number(value, name, integer=True)


def component(n, method, formula, notes=None):
    return {'bytes': n, 'gib': n/GIB if n is not None else None,
            'method': method, 'formula_id': formula, 'notes': notes or []}


def inventory_bytes(workload, category, fallback):
    inventory = workload.get('tensor_inventory', {})
    if category in inventory:
        total = 0
        for tensor in inventory[category]:
            if tensor.get('layout', 'torch.strided') != 'torch.strided':
                raise Unsupported(f'{category} contains sparse/non-strided tensors.')
            if category == 'weights' and str(tensor['dtype']).removeprefix('torch.') not in (
                    'float16','float32','float64','bfloat16','fp16','fp32','fp64','bf16'):
                raise Unsupported('Non-floating/quantized weights need a specialized storage model.')
            total += count(tensor['numel'], 'numel') * dtype_bytes(tensor['dtype'])
        reported = workload['model'].get(fallback)
        if reported is not None and reported != total:
            raise ValueError(f'{category} inventory disagrees with model.{fallback}')
        return total
    value = workload.get('model', {}).get(fallback)
    if value is None:
        raise Unsupported(f'Missing tensor inventory and model.{fallback}')
    return count(value, fallback)


def optimizer_groups(workload, pt):
    groups = workload.get('optimizer', {}).get('parameter_groups') or []
    if groups:
        if any('trainable_parameter_count' not in g for g in groups):
            raise Unsupported('Optimizer group trainable counts are missing.')
        if sum(count(g['trainable_parameter_count'], 'group count') for g in groups) != pt:
            raise Unsupported('Optimizer groups do not cover exactly the trainable parameters.')
    else:
        groups = [{'trainable_parameter_count': pt}]
    return groups


def state_count(name, opts):
    momentum = number(opts.get('momentum', 0), 'momentum')
    for key in ('amsgrad', 'centered'):
        if key in opts and not isinstance(opts[key], bool):
            raise ValueError(f'{key} must be boolean')
    if name == 'sgd': return int(momentum > 0)
    if name in ('adam', 'adamw'): return 2 + int(opts.get('amsgrad', False))
    if name == 'adagrad': return 1
    if name == 'rmsprop': return 1 + int(momentum > 0) + int(opts.get('centered', False))
    raise Unsupported(f'Optimizer {name!r} needs assumptions.optimizer_state_bytes.')


def estimate_states(w, a, notes):
    m, t = w['model'], w.get('training', {})
    p, pt = count(m['total_params'], 'total_params'), count(m['trainable_params'], 'trainable_params')
    if pt > p: raise ValueError('trainable_params exceeds total_params')
    weights = inventory_bytes(w, 'weights', 'logical_weight_bytes')
    try: buffers = inventory_bytes(w, 'buffers', 'logical_buffer_bytes')
    except Unsupported:
        buffers = None
    comps = {'weights': component(weights, 'metadata', 'tensor_bytes'),
             'buffers': component(buffers, 'metadata', 'tensor_bytes')}
    if buffers is None: notes.append('Buffer inventory is missing; zero is not assumed.')
    if t.get('gradient_dtype') is not None:
        if str(t['gradient_dtype']).removeprefix('torch.') not in (
                'float16','float32','float64','bfloat16','fp16','fp32','fp64','bf16'):
            raise Unsupported('Non-floating gradients need explicit sparse/quantized accounting.')
        grad = pt*dtype_bytes(t['gradient_dtype'])
    elif w.get('dense_gradient_estimate_bytes') is not None:
        grad = count(w['dense_gradient_estimate_bytes'], 'dense_gradient_estimate_bytes')
        notes.append('Gradient bytes use collector fallback: ' + w.get('gradient_estimate_assumption', 'unknown dtype assumption'))
    else: grad = None
    comps['gradients'] = component(grad, 'dense gradients for all trainable parameters', 'model_states')
    cfg = w.get('configured_optimizer', {})
    opt_name = cfg.get('name') or t.get('optimizer') or w.get('optimizer', {}).get('class')
    name = str(opt_name).split('.')[-1].lower()
    opts = t.get('optimizer_options') or cfg.get('options') or {}
    groups = optimizer_groups(w, pt)
    state_dtype = t.get('optimizer_state_dtype') or cfg.get('state_dtype')
    opt = None
    try:
        if 'optimizer_state_bytes' in a:
            opt = count(a['optimizer_state_bytes'], 'optimizer_state_bytes')
        elif pt == 0:
            opt = 0
        else:
            # Group options override global settings.
            parts = [(g['trainable_parameter_count'], state_count(name, {**opts, **g})) for g in groups]
            if not any(n*k for n, k in parts): opt = 0
            elif state_dtype is None:
                raise Unsupported('optimizer_state_dtype must be explicit; not inferred from compute precision.')
            else:
                if str(state_dtype).removeprefix('torch.') not in (
                        'float16','float32','float64','bfloat16','fp16','fp32','fp64','bf16'):
                    raise Unsupported('Quantized optimizer states need optimizer_state_bytes including metadata.')
                opt = sum(n*k*dtype_bytes(state_dtype) for n,k in parts)
            notes.append('Optimizer estimate excludes tiny scalar step counters; all trainable '
                         'parameters are assumed optimized and receive gradients.')
    except Unsupported as exc: notes.append(str(exc))
    comps['optimizer_states'] = component(opt, 'configured state tensors; not observed initialized states', 'optimizer_states')
    policy = t.get('master_weights', cfg.get('master_weights'))
    master = None
    if 'master_weight_bytes' in a: master = count(a['master_weight_bytes'], 'master_weight_bytes')
    elif policy == 'none' or pt == 0: master = 0
    elif policy == 'fp32': master = pt*4
    else: notes.append('Set training.master_weights to none/fp32, or override master_weight_bytes.')
    comps['master_weights'] = component(master, 'explicit extra-copy policy', 'master_weights')
    for key in ('weights', 'gradients', 'optimizer_states', 'master_weights'):
        if comps[key]['bytes'] is None: notes.append(f'Unknown model-state component: {key}')
    if pt < p:
        notes.append('Partial training: total trainable count is known, but layer-level '
                     'trainability is absent. Built-in activation profiles cannot infer it.')
    if any(c['bytes'] is None for c in comps.values()): total = None
    else: total = sum(c['bytes'] for c in comps.values())
    return comps, total, groups, state_dtype


def activation_dtype(w, a, notes):
    dtype = a.get('activation_dtype') or w.get('training', {}).get('precision')
    if dtype is None: raise Unsupported('Set training.precision or assumptions.activation_dtype.')
    b = dtype_bytes(dtype)
    if b not in (2,4,8): raise Unsupported('Activation dtype must be floating-point.')
    notes.append('Activation storage uses one configured compute dtype; real mixed-precision '
                 'graphs can retain additional FP32 tensors.')
    return b


def input_specs(w):
    specs = w.get('input_specs') or w.get('training', {}).get('input_specs')
    if not specs: raise Unsupported('input_specs with numeric batch-inclusive shapes are required.')
    return specs


def spec_size(spec, batch):
    shape = spec.get('shape')
    if not isinstance(shape, list) or not shape:
        raise Unsupported('Each input spec requires a nonempty numeric shape.')
    dims = [number(x, 'input dimension', integer=True, minimum=1) for x in shape]
    if dims[0] != batch: raise ValueError('input_specs must include batch_size_per_gpu as dimension 0')
    return math.prod(dims)*dtype_bytes(spec['dtype'])


def mlp_dimensions(w):
    m = w['model']
    if m.get('class') != 'Sequential' or m.get('module_types') != {'Sequential':1,'Linear':2,'ReLU':1}:
        raise Unsupported('simple_mlp requires Sequential(Linear, ReLU, Linear), verified module metadata.')
    inv = {x['name']:x for x in w.get('tensor_inventory',{}).get('weights',[])}
    if not {'0.weight','2.weight'} <= inv.keys() or not set(inv) <= {'0.weight','0.bias','2.weight','2.bias'}:
        raise Unsupported('simple_mlp requires the standard 0.weight/2.weight inventory.')
    shape1, shape2 = inv['0.weight']['shape'], inv['2.weight']['shape']
    if len(shape1)!=2 or len(shape2)!=2 or shape1[0]!=shape2[1]:
        raise ValueError('Inconsistent MLP layer dimensions.')
    h,i = shape1; o,_ = shape2
    return i,h,o


def estimate_activations(w, a, notes):
    t,m = w.get('training',{}), w['model']
    B = number(t.get('batch_size_per_gpu'), 'batch_size_per_gpu', integer=True, minimum=1)
    b = activation_dtype(w,a,notes)
    if 'saved_activation_bytes' in a:
        saved = count(a['saved_activation_bytes'], 'saved_activation_bytes')
        if 'auxiliary_tensor_bytes' not in a:
            raise Unsupported('Custom saved_activation_bytes requires auxiliary_tensor_bytes '
                              'to specify extra input/output/loss storage without double counting.')
        aux = count(a['auxiliary_tensor_bytes'], 'auxiliary_tensor_bytes')
        notes.append('Saved activations and auxiliary storage supplied by user; provenance must be checked.')
        return saved,aux,aux,'user_override'
    if t.get('gradient_checkpointing') is not False:
        raise Unsupported('Built-in activation rules require explicit gradient_checkpointing=false.')
    if m['trainable_params'] != m['total_params']:
        raise Unsupported('Built-in activation rules require full training; use explicit activation/auxiliary overrides.')
    profile = a.get('activation_model','auto')
    if profile=='auto':
        mlp_dimensions(w)  # Fail rather than guess arbitrary graph topology.
        profile='simple_mlp'
    specs=input_specs(w)
    sizes=[spec_size(s,B) for s in specs]
    if profile=='simple_mlp':
        I,H,O=mlp_dimensions(w)
        if specs[0]['shape'] != [B,I]: raise ValueError('First input must be MLP features [B,input_size].')
        if t.get('task')!='regression': raise Unsupported('simple_mlp auxiliary rule currently supports regression/MSE only.')
        if len(specs)!=2 or specs[1]['shape'] != [B,O]:
            raise Unsupported('MLP requires exactly features and same-shaped regression targets.')
        saved=B*(I+H)*b
        # Original features alias the saved first-linear input only when dtypes match.
        input_extra=0 if dtype_bytes(specs[0]['dtype'])==b else sizes[0]
        output=B*O*b
        # Target storage is not already included in activations. FP32 MSE scratch/casts
        # are uncertain: use 1..2 FP32 output-sized buffers as a planning assumption.
        aux_base=input_extra+sizes[1]+output
        loss_buffer=B*O*max(4,b,dtype_bytes(specs[1]['dtype']))
        notes.extend(['MLP saved-storage rule assumes standard eager Linear-ReLU-Linear '
                      'and no custom forward hooks/operations. Shared ReLU output counted once.',
                      'Regression loss assumed ordinary MSE. Auxiliary low/high include '
                      'one/two extra output-sized loss buffers at >=FP32: an UNSOURCED allowance.'])
        return saved,aux_base+loss_buffer,aux_base+2*loss_buffer,profile
    if profile!='transformer_classic16':
        raise Unsupported(f'Unknown activation model {profile!r}.')
    if b!=2: raise Unsupported('Published classic transformer formula requires 16-bit activations.')
    if t.get('attention_backend') not in ('eager','standard'):
        raise Unsupported('Classic transformer rule does not support FlashAttention/SDPA; use overrides.')
    arch={**m.get('provided_config',{}), **m.get('architecture',{})}
    def dim(*keys):
        for key in keys:
            if arch.get(key) is not None:
                return number(arch[key],key,integer=True,minimum=1)
        raise Unsupported(f'Missing architecture field: {keys[0]}')
    L,H,A=dim('num_hidden_layers','n_layer'),dim('hidden_size','n_embd'),dim('num_attention_heads','n_head')
    S=number(t.get('sequence_length'),'sequence_length',integer=True,minimum=1)
    if arch.get('intermediate_size',4*H)!=4*H:
        raise Unsupported('Classic transformer profile requires a 4H feed-forward width.')
    if arch.get('num_key_value_heads',A)!=A:
        raise Unsupported('Classic transformer profile does not model grouped/multi-query attention.')
    if H%A: raise ValueError('hidden_size must be divisible by attention heads')
    if t.get('task') not in ('causal_language_modeling','masked_language_modeling'):
        raise Unsupported('Transformer auxiliary rule supports token-level language modeling only.')
    V=dim('vocab_size')
    saved=34*L*B*S*H+5*L*B*A*S*S
    logits=B*S*V*b
    # Validate max token dimension in the first input. Full explicit input inventory
    # is counted separately from the floating transformer-block activations.
    if specs[0]['shape'] != [B,S]: raise ValueError('First transformer input must be [B,S].')
    if not any(s.get('name') in ('labels','targets') for s in specs):
        raise Unsupported('Transformer input_specs must include labels/targets.')
    aux_base=sum(sizes)+logits
    loss_buffer=B*S*V*4
    notes.extend(['EXPLICIT transformer profile assumes classic MHA, GeLU 4H MLP, '
                  'dropout and 16-bit saved tensors. Gated MLPs and other architectures '
                  'are not inferred or validated solely from parameter metadata.',
                  'Full B*S*V logits are materialized. Cross-entropy low/high adds one/two '
                  'FP32 logits-sized buffers as an UNSOURCED allowance; fused/chunked '
                  'loss and masked-only projection need overrides.',
                  'Transformer embedding/output-side residual allocations and small normalization '
                  'buffers are only approximately covered by workspace allowances.'])
    return saved,aux_base+loss_buffer,aux_base+2*loss_buffer,profile


def estimate_cast_cache(w,a,notes):
    if 'autocast_weight_cache_bytes' in a:
        return count(a['autocast_weight_cache_bytes'],'autocast_weight_cache_bytes')
    precision=w.get('training',{}).get('precision')
    if precision is None: raise Unsupported('Compute precision unknown; set autocast_weight_cache_bytes explicitly.')
    b=dtype_bytes(precision)
    if str(precision).removeprefix('torch.') not in ('fp16','bf16','fp32','fp64','float16','bfloat16','float32','float64'):
        raise Unsupported('Unrecognized compute precision; specify autocast_weight_cache_bytes.')
    if b>=4: return 0
    inventory=w.get('tensor_inventory',{}).get('weights')
    if inventory is None: raise Unsupported('Weights inventory required to estimate autocast cast-weight storage.')
    if any(dtype_bytes(x['dtype']) != 4 and str(x['dtype']).removeprefix('torch.') not in
           ({'fp16','float16'} if str(precision).removeprefix('torch.') in ('fp16','float16') else {'bf16','bfloat16'})
           for x in inventory):
        raise Unsupported('Mixed parameter/compute dtypes need explicit autocast_weight_cache_bytes.')
    eligible=sum(x['numel'] for x in inventory if dtype_bytes(x['dtype'])==4)
    # Collector has no per-parameter requires_grad; full-training profiles only.
    if eligible and w['model']['trainable_params']!=w['model']['total_params']:
        raise Unsupported('Partial training requires explicit autocast_weight_cache_bytes.')
    notes.append('Low-precision compute with FP32 weights budgets additional lower-precision '
                 'weight casts; eligibility/cache policy requires confirmation. Set '
                 'autocast_weight_cache_bytes explicitly to override.')
    return eligible*b


def optimizer_workspace(w,a,groups,state_dtype,notes):
    if 'optimizer_workspace_low_bytes' in a and 'optimizer_workspace_high_bytes' in a:
        return (count(a['optimizer_workspace_low_bytes'],'optimizer_workspace_low_bytes'),
                count(a['optimizer_workspace_high_bytes'],'optimizer_workspace_high_bytes'))
    tensors=w.get('tensor_inventory',{}).get('weights')
    if not tensors: raise Unsupported('Optimizer workspace needs weights inventory or explicit low/high bytes.')
    if w['model']['trainable_params']!=w['model']['total_params']:
        raise Unsupported('Partial training needs explicit optimizer workspace low/high bytes.')
    max_element_bytes=max(dtype_bytes(x['dtype']) for x in tensors)
    sb=dtype_bytes(state_dtype) if state_dtype is not None else max_element_bytes
    low=max(x['numel'] for x in tensors)*max(sb,max_element_bytes)
    total=sum(x['numel']*dtype_bytes(x['dtype']) for x in tensors)
    cfg=w.get('configured_optimizer',{})
    opts=w.get('training',{}).get('optimizer_options') or cfg.get('options') or {}
    foreach=any(g.get('foreach',opts.get('foreach')) is not False for g in groups)
    high=low+total if foreach else low
    notes.append('Optimizer scratch is heuristic. Largest-parameter buffer is budgeted; '
                 'foreach enabled/unknown adds ~one parameter-set buffer to high scenario. '
                 'Fused/custom kernels may differ; explicit workspace overrides supported.')
    return low,high


def capacity_check(hardware,low,high):
    if hardware is None: return {'status':'not_provided'}
    gpu=hardware.get('gpu',hardware)
    if 'vram_bytes' in gpu: capacity=count(gpu['vram_bytes'],'vram_bytes')
    elif 'vram_gib_per_gpu' in gpu: capacity=number(gpu['vram_gib_per_gpu'],'vram_gib_per_gpu')*GIB
    else: return {'status':'unknown_capacity','note':'Provide vram_bytes or vram_gib_per_gpu; advertised GB not converted implicitly.'}
    if 'other_usage_gib_per_gpu' in gpu:
        other=number(gpu['other_usage_gib_per_gpu'],'other_usage_gib_per_gpu')*GIB
        assumption=None
    else:
        other=0
        assumption='Other-process usage unknown; assumed zero for this capacity comparison.'
    reserve=number(gpu.get('safety_reserve_gib_per_gpu',0),'safety_reserve_gib_per_gpu')*GIB
    budget=capacity-other-reserve
    if budget<0: raise ValueError('Other usage + reserve exceeds capacity')
    if low is None or high is None: status='incomplete_estimate'
    elif low>budget: status='exceeds_budget_in_both_scenarios'
    elif high>budget: status='scenario_dependent'
    else: status='below_budget_in_both_scenarios'
    return {'status':status,'usable_bytes':budget,'usable_gib':budget/GIB,
            'assumption':assumption,'formula_id':'capacity',
            'note':'Scenario comparison is not a guarantee against OOM.'}


def _estimate_single_gpu_legacy(workload, assumptions=None, hardware=None):
    """Return JSON-compatible component accounting and explicit scenario assumptions."""
    w,a=workload,dict(assumptions or {})
    allowed = {
        'activation_model', 'saved_activation_bytes', 'auxiliary_tensor_bytes',
        'optimizer_state_bytes', 'master_weight_bytes', 'autocast_weight_cache_bytes',
        'activation_dtype', 'additional_persistent_bytes',
        'kernel_workspace_low_mib', 'kernel_workspace_high_mib',
        'kernel_workspace_fraction_low', 'kernel_workspace_fraction_high',
        'optimizer_workspace_low_bytes', 'optimizer_workspace_high_bytes',
        'runtime_overhead_low_mib', 'runtime_overhead_high_mib',
        'fragmentation_fraction_low', 'fragmentation_fraction_high',
        'activation_multiplier_low', 'activation_multiplier_high',
    }
    if set(a) - allowed:
        raise ValueError('Unknown assumption keys: ' + ', '.join(sorted(set(a) - allowed)))
    if ('optimizer_workspace_low_bytes' in a) != ('optimizer_workspace_high_bytes' in a):
        raise ValueError('Supply both optimizer workspace low/high overrides.')
    if not isinstance(w,dict) or not isinstance(w.get('model'),dict):
        raise ValueError('Expected collect_workload JSON with model metadata')
    t=w.get('training',{})
    notes,missing=[],[]
    report={'schema_version':1,'scope':'single_gpu_dense_training',
            'scenario_interpretation':'Planning sensitivity scenarios, not statistical intervals '
                                      'or guaranteed bounds. No execution lifetimes are traced.',
            'formula_catalog':FORMULAS,'sources':SOURCES,'assumptions':notes,
            'missing_or_unsupported':missing,'user_overrides':a}
    unsupported=[]
    if t.get('gpu_count',1)!=1: unsupported.append('Only one GPU is supported; no division by GPU count.')
    if t.get('distributed_strategy','single') not in ('single','none'):
        unsupported.append('Sharding/distributed strategy unsupported.')
    for key in ('offload','offload_param','offload_optimizer','zero_stage','deepspeed','fsdp',
                'quantization','torch_compile','cuda_graphs','retain_graph','create_graph'):
        if t.get(key): unsupported.append(f'{key} needs a specialized memory model.')
    if unsupported:
        missing.extend(unsupported)
        report.update(status='unsupported_configuration',components={},model_state_bytes=None,
                      planning_scenarios=None,capacity_check={'status':'not_evaluated'})
        return report
    notes.extend(['All states assumed GPU-resident and unsharded; all trainable parameters '
                  'receive dense gradients. Distinct Parameter views may overlap storage; '
                  'collector lacks alias information for meta tensors.',
                  'Microbatch size per GPU is used. Sequential gradient accumulation does '
                  'not multiply saved activations; retained graphs are unsupported.',
                  'No CUDA execution occurs. Loading/checkpoint/evaluation peaks, dataloader '
                  'GPU prefetch, custom hooks and training-framework allocations are not '
                  'individually modeled. Additional persistent storage can be supplied.'])
    try:
        comps,total,groups,state_dtype=estimate_states(w,a,notes)
    except Unsupported as exc:
        missing.append(str(exc)); comps={};total=None;groups=[];state_dtype=None
    report['components']=comps
    report['model_state_bytes']=total
    report['model_state_gib']=total/GIB if total is not None else None
    try: cast=estimate_cast_cache(w,a,notes)
    except Unsupported as exc: cast=None; missing.append(str(exc))
    comps['autocast_weight_cache']=component(cast,'configured cast-storage allowance','autocast_cache')
    extra=count(a.get('additional_persistent_bytes',0),'additional_persistent_bytes')
    comps['additional_persistent']=component(extra,'user override, default zero','tensor_bytes')
    try:
        saved,aux_low,aux_high,profile=estimate_activations(w,a,notes)
        report['activation_profile']=profile
    except Unsupported as exc:
        saved=aux_low=aux_high=None;missing.append(str(exc))
    comps['saved_activations']=component(saved,'architecture formula or explicit override',
        'transformer_saved' if a.get('activation_model')=='transformer_classic16' else
        'mlp_saved' if 'saved_activation_bytes' not in a else 'tensor_bytes')
    comps['auxiliary_tensors']={'low_bytes':aux_low,'high_bytes':aux_high,'formula_id':'auxiliary'}
    try: ow_low,ow_high=optimizer_workspace(w,a,groups,state_dtype,notes)
    except Unsupported as exc: ow_low=ow_high=None;missing.append(str(exc))
    if ow_low is not None and ow_high<ow_low: raise ValueError('Optimizer high workspace must be >= low')
    comps['optimizer_workspace']={'low_bytes':ow_low,'high_bytes':ow_high,'formula_id':'optimizer_workspace'}
    config={
        'kernel_workspace_low_mib':64,'kernel_workspace_high_mib':256,
        'kernel_workspace_fraction_low':0.05,'kernel_workspace_fraction_high':0.20,
        'runtime_overhead_low_mib':512,'runtime_overhead_high_mib':1536,
        'fragmentation_fraction_low':0.05,'fragmentation_fraction_high':0.15,
        'activation_multiplier_low':1.0,'activation_multiplier_high':1.0}
    for key,default in list(config.items()): config[key]=number(a.get(key,default),key)
    for base in ('kernel_workspace','kernel_workspace_fraction','runtime_overhead',
                 'fragmentation_fraction','activation_multiplier'):
        suffix='_mib' if base in ('kernel_workspace','runtime_overhead') else ''
        if config[base+'_high'+suffix]<config[base+'_low'+suffix]:
            raise ValueError(f'{base}: high must be >= low')
    report['planning_allowances']=config
    notes.append('Workspace, CUDA runtime, fragmentation and loss-buffer allowances are '
                 'UNSOURCED engineering defaults, not measured or literature-prescribed percentages.')
    scenarios=None
    if None not in (total,cast,saved,aux_low,aux_high,ow_low,ow_high):
        scenarios={}
        for label,aux,ow in [('low',aux_low,ow_low),('high',aux_high,ow_high)]:
            act=saved*config['activation_multiplier_'+label]
            kw=config['kernel_workspace_'+label+'_mib']*MIB+config['kernel_workspace_fraction_'+label]*(act+aux)
            subtotal=total+cast+extra+act+aux+kw+ow
            frag=subtotal*config['fragmentation_fraction_'+label]
            runtime=config['runtime_overhead_'+label+'_mib']*MIB
            predicted=math.ceil(subtotal+frag+runtime)
            scenarios[label]={'planning_bytes':predicted,'planning_gib':predicted/GIB,
                'saved_activation_bytes_after_multiplier':act,'auxiliary_tensor_bytes':aux,
                'kernel_workspace_bytes':kw,'optimizer_workspace_bytes':ow,
                'allocator_allowance_bytes':frag,'runtime_overhead_bytes':runtime,
                'formula_id':'planning_total'}
    else:
        missing.append('One or more components unknown: no complete peak planning total produced.')
    report['planning_scenarios']=scenarios
    report['status']='estimated_with_assumptions' if scenarios else 'incomplete'
    report['capacity_check']=capacity_check(hardware or w.get('target_hardware'),
        scenarios['low']['planning_bytes'] if scenarios else None,
        scenarios['high']['planning_bytes'] if scenarios else None)
    return report

# ---- Distributed placement and graph-lifetime extensions (schema version 2) ----
SOURCES.update({
    'zero': 'https://deepspeed.readthedocs.io/en/stable/zero3.html',
    'zero_memory': 'https://deepspeed.readthedocs.io/en/stable/memory.html',
    'fsdp': 'https://docs.pytorch.org/docs/stable/fsdp.html',
    'ddp': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html',
})
FORMULAS.update({
    'state_sharding': {
        'formula': 'rank_state_bytes = ceil(global_state_bytes / shard_group_size) '
                   'for sharded categories; otherwise global_state_bytes',
        'basis': 'Ideal balanced byte partition; padding/flattening overhead not inferred. '
                 'ZeRO1 shards optimizer/master; ZeRO2 also gradients; ZeRO3 also weights. '
                 'FSDP FULL_SHARD uses ZeRO3-like persistent ownership. FSDP SHARD_GRAD_OP '
                 'budgets full weights during compute. Buffers remain replicated. '
                 'No tensor/pipeline parallel partition is guessed.',
        'sources': ['zero', 'fsdp']},
    'state_placement': {
        'formula': 'sum rank-owned state bytes by configured gpu/cpu/nvme location',
        'basis': 'Steady-state storage accounting, not automatic framework configuration. '
                 'Transfers also need live GPU working sets and CPU staging; these are '
                 'separately budgeted. Offload is not a multiplication by zero.',
        'sources': ['zero']},
    'gather_working_set': {
        'formula': 'GPU full-weight buffers = min(P, gather_unit_params * live_units)*b_weight; '
                   'full-gradient work buffer = min(Pt,gather_unit_params)*b_grad',
        'basis': 'Conservative allocated-buffer model. Keeps local shards AND gathered '
                 'buffers, so shared storage may be overcounted. Covers current/prefetched '
                 'units; does not claim to exactly reproduce DeepSpeed/FSDP allocator '
                 'lifetimes. Unknown wrapping falls back to whole-model unit (reported). '
                 'DeepSpeed documents largest-layer parameter+gradient working sets; '
                 'FSDP documents prefetch/all-gather concurrency.',
        'sources': ['zero_memory', 'fsdp']},
    'communication_buffers': {
        'formula': 'copies * (reduce_bucket_bytes + allgather_bucket_bytes)',
        'basis': 'User-configured buffer accounting. Default buckets are planning allowances '
                 'capped at 25 MiB and relevant model/gradient size, not asserted framework '
                 'defaults. Optional explicit bytes override. Gradient bucket views/aliasing '
                 'may reduce actual extra allocation.',
        'sources': ['ddp', 'zero']},
    'offload_staging': {
        'formula': 'CPU staging: tile_params * sum(bytes/parameter for offloaded states); '
                   'CPU optimizer scratch and GPU working sets budgeted separately',
        'basis': 'Original tiled-transfer allowance. Defaults one simultaneous tile of '
                 'min(local owned params, gather unit, 1e8 params), not actual ZeRO Infinity '
                 'buffer pool defaults. Explicit byte values override; async double buffering '
                 'requires correspondingly larger allowances. CPU optimizer computation '
                 'is assumed when optimizer states are on CPU/NVMe.',
        'sources': ['zero']},
    'graph_liveness': {
        'formula': 'graph_peak[device] = max_t(sum live unique storage bytes at t '
                   '+ explicit workspace bytes at t)',
        'basis': 'Original interval-sweep accounting; each tensor lives on inclusive '
                 '[born,last_use]. Shared storage_id is deduplicated with explicit '
                 'underlying storage_bytes. Model states must be excluded. Complete '
                 'forward+backward input/output/loss/activation coverage is asserted by '
                 'the user, not inferred from parameter counts. Kernel workspace remains '
                 'an allowance unless workspace_complete=true.',
        'sources': ['tensor_size', 'autograd']},
    'node_memory': {
        'formula': 'node_cpu = ranks_per_node * per_rank_cpu + shared_cpu; '
                   'node_nvme = ranks_per_node * per_rank_nvme + shared_nvme',
        'basis': 'Balanced identical-rank accounting. Shared datasets/checkpoints must be '
                 'supplied separately. Node CPU includes selected graph transients, staging '
                 'and optimizer workspace; not a complete OS memory forecast.',
        'sources': []},
})

STATE_CATEGORIES = ('weights','buffers','gradients','optimizer_states','master_weights')
STRATEGIES = {'single','ddp','zero1','zero2','zero3','fsdp_full_shard',
              'fsdp_shard_grad_op','fsdp_hybrid_shard'}


def _config_keys(config, allowed, label):
    if not isinstance(config,dict): raise ValueError(f'{label} must be a JSON object')
    unknown=set(config)-set(allowed)
    if unknown: raise ValueError(f'Unknown {label} keys: {sorted(unknown)}')


def resolve_deployment(workload, deployment):
    """Explicit planning policy. DeepSpeed JSON is NOT silently reinterpreted."""
    t=workload.get('training',{})
    embedded=t.get('memory_deployment',{})
    d=dict(embedded if deployment is None else deployment)
    allowed={'strategy','world_size','gpus_per_node','shard_group_size','placement',
        'offload','largest_gather_unit_params','prefetch_units','gradient_storage_unsharded',
        'communication_buffers','offload_buffers','offload_tile_params',
        'shared_cpu_bytes_per_node','shared_nvme_bytes_per_node',
        'tensor_parallel_size','pipeline_parallel_size'}
    _config_keys(d,allowed,'deployment')
    if deployment is None and not embedded:
        if any(t.get(key) for key in ('offload','offload_param','offload_optimizer',
                                      'deepspeed','fsdp','zero_stage')):
            raise Unsupported('Supply explicit --deployment placement/strategy; native framework '
                              'config is not automatically translated.')
    strategy=d.get('strategy',t.get('distributed_strategy','single'))
    aliases={'none':'single','fsdp':'fsdp_full_shard','full_shard':'fsdp_full_shard'}
    strategy=aliases.get(strategy,strategy)
    if strategy not in STRATEGIES: raise Unsupported(f'Unsupported distribution strategy {strategy!r}')
    world=number(d.get('world_size',t.get('gpu_count',1)),'world_size',integer=True,minimum=1)
    if strategy=='single' and world!=1:
        raise ValueError('world_size>1 requires an explicit distributed strategy, such as ddp or zero3')
    for key in ('tensor_parallel_size','pipeline_parallel_size'):
        if d.get(key,1)!=1:
            raise Unsupported(f'{key}>1 needs rank-specific model inventories; not inferred from global counts.')
    pernode=number(d.get('gpus_per_node',world),'gpus_per_node',integer=True,minimum=1)
    if world%pernode: raise ValueError('world_size must be divisible by gpus_per_node in this balanced-node model')
    group=number(d.get('shard_group_size',pernode if strategy=='fsdp_hybrid_shard' else world),
                 'shard_group_size',integer=True,minimum=1)
    if world%group: raise ValueError('shard_group_size must divide world_size')
    if strategy=='fsdp_hybrid_shard' and pernode%group:
        raise ValueError('Hybrid shard groups must divide gpus_per_node')
    if strategy in ('single','ddp') and group!=world:
        raise ValueError('shard_group_size only applies to sharded strategies')
    placement={key:'gpu' for key in STATE_CATEGORIES}
    offload=d.get('offload',{})
    _config_keys(offload,{'parameters','optimizer','gradients'},'offload')
    if 'parameters' in offload: placement['weights']=offload['parameters']
    if 'optimizer' in offload:
        placement['optimizer_states']=placement['master_weights']=offload['optimizer']
        if offload['optimizer']!='gpu': placement['gradients']='cpu'
    if 'gradients' in offload: placement['gradients']=offload['gradients']
    _config_keys(d.get('placement',{}),STATE_CATEGORIES,'placement')
    placement.update(d.get('placement',{}))
    if any(value not in ('gpu','cpu','nvme') for value in placement.values()):
        raise ValueError('Placement values must be gpu, cpu or nvme')
    if placement['buffers']!='gpu':
        raise Unsupported('Model buffers remain GPU-resident in this planner; explicit graph '
                          'buffer migration is not inferred.')
    if not isinstance(d.get('gradient_storage_unsharded',False),bool):
        raise ValueError('gradient_storage_unsharded must be boolean')
    return {**d,'strategy':strategy,'world_size':world,'gpus_per_node':pernode,
            'shard_group_size':group,'placement':placement}


def graph_memory(graph):
    """Sweep full forward/backward tensor lifetimes; no operator-specific guessing.

    Tensor row: id, shape+dtype OR bytes, born, last_use, device (gpu/cpu/nvme),
    optional storage_id and storage_bytes for aliases, saved_for_backward bool.
    Graph has scope='per_rank', complete_training_step=true, excludes_model_states=true.
    Workspaces: [{step, gpu_bytes, cpu_bytes, nvme_bytes}]. Multiple rows at one
    step are ADDITIVE. workspace_complete asserts all activation/kernel workspaces
    (not optimizer scratch) are already included, so generic kernel allowance is skipped.
    """
    _config_keys(graph,{'scope','complete_training_step','excludes_model_states',
                       'workspace_complete','tensors','workspaces','description'},'graph')
    if graph.get('scope')!='per_rank': raise ValueError('Graph scope must be per_rank (local microbatch)')
    if graph.get('complete_training_step') is not True or graph.get('excludes_model_states') is not True:
        raise ValueError('Graph must assert complete_training_step=true and excludes_model_states=true')
    if not isinstance(graph.get('workspace_complete',False),bool): raise ValueError('workspace_complete must be boolean')
    rows=graph.get('tensors')
    if not isinstance(rows,list) or not rows: raise ValueError('Graph requires a nonempty tensors list')
    ids=set();groups={};locations=('gpu','cpu','nvme')
    for row in rows:
        _config_keys(row,{'id','shape','dtype','bytes','born','last_use','device',
                         'storage_id','storage_bytes','saved_for_backward'},'graph tensor')
        tid=row.get('id')
        if not isinstance(tid,str) or not tid or tid in ids: raise ValueError('Tensor ids must be unique nonempty strings')
        ids.add(tid)
        device=row.get('device','gpu')
        if device not in locations: raise ValueError('Graph tensor device must be gpu/cpu/nvme')
        born=count(row.get('born'),'born');last=count(row.get('last_use'),'last_use')
        if last<born: raise ValueError('last_use must be >= born')
        if 'bytes' in row: size=count(row['bytes'],'tensor bytes')
        else:
            if not isinstance(row.get('shape'),list): raise ValueError('Tensor needs bytes or shape/dtype')
            dims=[count(x,'shape dimension') for x in row['shape']]
            size=math.prod(dims)*dtype_bytes(row['dtype'])
        storage=row.get('storage_id',tid)
        if not isinstance(storage,str): raise ValueError('storage_id must be a string')
        key=(device,storage)
        entry=groups.setdefault(key,{'rows':[],'capacity':None})
        capacity=count(row['storage_bytes'],'storage_bytes') if 'storage_bytes' in row else None
        if capacity is not None:
            if entry['capacity'] is not None and entry['capacity']!=capacity:
                raise ValueError('Alias rows disagree on storage_bytes')
            entry['capacity']=capacity
        if not isinstance(row.get('saved_for_backward',False),bool): raise ValueError('saved_for_backward must be boolean')
        entry['rows'].append((born,last,size,row.get('saved_for_backward',False)))
    events={device:{} for device in locations};saved_events={device:{} for device in locations}
    def add(table,time,delta): table[time]=table.get(time,0)+delta
    for (device,_),entry in groups.items():
        parts=entry['rows']
        if len(parts)>1 and entry['capacity'] is None:
            raise ValueError('Aliased storage requires explicit full storage_bytes on at least one row')
        size=entry['capacity'] if entry['capacity'] is not None else parts[0][2]
        if any(part[2]>size for part in parts): raise ValueError('Logical tensor bytes exceed its storage_bytes')
        begin=min(p[0] for p in parts);end=max(p[1] for p in parts)
        add(events[device],begin,size);add(events[device],end+1,-size)
        saved=[p for p in parts if p[3]]
        if saved:
            add(saved_events[device],min(p[0] for p in saved),size)
            add(saved_events[device],max(p[1] for p in saved)+1,-size)
    workspaces={device:{} for device in locations}
    for row in graph.get('workspaces',[]):
        _config_keys(row,{'step','gpu_bytes','cpu_bytes','nvme_bytes'},'graph workspace')
        step=count(row.get('step'),'workspace step')
        for device in locations:
            workspaces[device][step]=workspaces[device].get(step,0)+count(row.get(device+'_bytes',0),'workspace bytes')
    def sweep(changes,scratch):
        current=peak=peakstep=0
        for time in sorted(set(changes)|set(scratch)):
            current+=changes.get(time,0)
            value=current+scratch.get(time,0)
            if value>peak: peak,peakstep=value,time
        return peak,peakstep
    peaks={device:sweep(events[device],workspaces[device]) for device in locations}
    saved={device:sweep(saved_events[device],{})[0] for device in locations}
    return {'peak_bytes':{k:v[0] for k,v in peaks.items()},
            'peak_steps':{k:v[1] for k,v in peaks.items()},
            'saved_storage_peak_bytes':saved,'unique_storage_count':len(groups),
            'workspace_complete':graph.get('workspace_complete',False),
            'formula_id':'graph_liveness',
            'note':'Tensor-lifetime coverage is supplied, not proven. Different device peaks may '
                   'occur at different steps. Saved-storage peak is diagnostic and is NOT added again.'}


def partition_and_place(workload,base,d,notes):
    strategy,group=d['strategy'],d['shard_group_size']
    shard_weights=strategy in ('zero3','fsdp_full_shard','fsdp_hybrid_shard')
    shard_grad=strategy in ('zero2','zero3','fsdp_full_shard','fsdp_shard_grad_op','fsdp_hybrid_shard')
    shard_opt=strategy not in ('single','ddp')
    if d.get('gradient_storage_unsharded',False): shard_grad=False
    divisors={'weights':group if shard_weights else 1,'buffers':1,
              'gradients':group if shard_grad else 1,
              'optimizer_states':group if shard_opt else 1,'master_weights':group if shard_opt else 1}
    bydevice={x:0 for x in ('gpu','cpu','nvme')};placed={}
    complete=True
    for key in STATE_CATEGORIES:
        global_bytes=base.get('components',{}).get(key,{}).get('bytes')
        local=(global_bytes+divisors[key]-1)//divisors[key] if global_bytes is not None else None
        if local is None: complete=False
        else: bydevice[d['placement'][key]]+=local
        placed[key]={'global_bytes':global_bytes,'per_rank_bytes':local,
                     'partition_divisor':divisors[key],'location':d['placement'][key],
                     'formula_ids':['state_sharding','state_placement']}
    notes.append('Balanced byte sharding assumes one process per GPU, identical ranks and '
                 'no tensor/pipeline parallelism. Buffers are replicated. Padding and ignored '
                 'parameters may need explicit additional_persistent_bytes.')
    if strategy=='fsdp_shard_grad_op':
        notes.append('FSDP SHARD_GRAD_OP budgets full parameter residency during computation, '
                     'even if parameters are sharded outside computation.')
    if shard_grad and workload.get('training',{}).get('gradient_accumulation_steps',1)>1:
        notes.append('Sharded gradient accumulation assumed synchronized each microbatch. '
                     'Set gradient_storage_unsharded=true if no_sync retains full gradients.')
    return placed,bydevice if complete else None,divisors


def distributed_working_set(w,d,base,divisors,notes):
    """Explicit transient working-set allowances, independent of persistent placement."""
    p=count(w['model']['total_params'],'total_params');pt=count(w['model']['trainable_params'],'trainable_params')
    inv=w.get('tensor_inventory',{}).get('weights',[])
    if not inv: raise Unsupported('Distributed/offload working set needs tensor weight inventory')
    bw=max(dtype_bytes(t['dtype']) for t in inv)
    grad_bytes=base.get('components',{}).get('gradients',{}).get('bytes')
    if grad_bytes is None: raise Unsupported('Working-set gradient size unknown')
    bg=grad_bytes/pt if pt else 0
    requires_weights=(divisors['weights']>1 or d['placement']['weights']!='gpu')
    requires_grad=(divisors['gradients']>1 or d['placement']['gradients']!='gpu')
    if 'largest_gather_unit_params' in d:
        unit=number(d['largest_gather_unit_params'],'largest_gather_unit_params',integer=True,minimum=1)
        if unit>p: raise ValueError('largest_gather_unit_params exceeds total parameters')
    else:
        unit=p
        if requires_weights or requires_grad:
            notes.append('Gather/wrap unit unknown: whole-model parameter count used. '
                         'Supply largest_gather_unit_params for useful ZeRO3/FSDP/offload estimates.')
    prefetch=count(d.get('prefetch_units',1),'prefetch_units')
    weight_work=min(p,unit*(1+prefetch))*bw if requires_weights else 0
    grad_work=math.ceil(min(pt,unit)*bg) if requires_grad else 0
    cfg=base.get('configured_optimizer',{})
    # Auto-cast transient full-unit copies, beyond sharded persistent cast allowance.
    global_cache=base.get('components',{}).get('autocast_weight_cache',{}).get('bytes')
    cache_work=math.ceil(global_cache*min(p,unit*(1+prefetch))/p) if requires_weights and global_cache is not None and p else 0
    comm=d.get('communication_buffers',{})
    _config_keys(comm,{'reduce_bucket_bytes','allgather_bucket_bytes','copies','gpu_bytes'},'communication_buffers')
    if d['world_size']>1:
        if 'gpu_bytes' in comm: communication=count(comm['gpu_bytes'],'communication gpu_bytes')
        else:
            reduction=count(comm.get('reduce_bucket_bytes',min(25*MIB,grad_bytes)),'reduce_bucket_bytes')
            allgather=count(comm.get('allgather_bucket_bytes',min(25*MIB,p*bw) if divisors['weights']>1 else 0),'allgather_bucket_bytes')
            communication=(reduction+allgather)*count(comm.get('copies',2),'communication copies')
    else: communication=0
    offloaded=[key for key in STATE_CATEGORIES if d['placement'][key]!='gpu']
    buffers=d.get('offload_buffers',{})
    _config_keys(buffers,{'gpu_staging_bytes','cpu_staging_bytes','cpu_optimizer_workspace_bytes'},'offload_buffers')
    tile=number(d.get('offload_tile_params',min(math.ceil(p/d['shard_group_size']),unit,100_000_000)),
                'offload_tile_params',integer=True,minimum=0)
    if tile>p: raise ValueError('offload_tile_params exceeds total parameters')
    bytes_per_param=0
    for key in offloaded:
        value=base.get('components',{}).get(key,{}).get('bytes')
        denominator=p if key=='weights' else pt
        if value is None: raise Unsupported(f'Offload size unknown for {key}')
        if denominator: bytes_per_param+=value/denominator
    cpu_staging=count(buffers.get('cpu_staging_bytes',math.ceil(tile*bytes_per_param)),'cpu_staging_bytes') if offloaded else count(buffers.get('cpu_staging_bytes',0),'cpu_staging_bytes')
    gpu_staging=count(buffers.get('gpu_staging_bytes',0),'gpu_staging_bytes')
    # Existing optimizer workspace may be global; only heuristics are scaled to
    # local shard/tile, never user-supplied exact overrides.
    ow=base.get('components',{}).get('optimizer_workspace',{})
    ol,oh=ow.get('low_bytes'),ow.get('high_bytes')
    user=base.get('user_overrides',{})
    if ol is not None and 'optimizer_workspace_low_bytes' not in user:
        opt_location=d['placement']['optimizer_states']
        factor=1/divisors['optimizer_states']
        if opt_location!='gpu' and pt: factor=min(factor,tile/pt)
        ol,oh=math.ceil(ol*factor),math.ceil(oh*factor)
    opt_gpu=opt_cpu=None
    if ol is not None:
        if d['placement']['optimizer_states']=='gpu': opt_gpu=(ol,oh);opt_cpu=(0,0)
        else:
            explicit=buffers.get('cpu_optimizer_workspace_bytes')
            opt_cpu=(count(explicit,'cpu_optimizer_workspace_bytes'),)*2 if explicit is not None else (ol,oh)
            opt_gpu=(0,0)
    notes.append('Gathered parameters/gradients, communication and staging are conservative '
                 'separate allocations; actual buffer reuse can lower memory. Transfer buffers '
                 'and tiled CPU optimizer scratch are configurable planning allowances, not '
                 'an emulation of framework buffer pools. Prefetch changes simultaneous residency.')
    return {'gpu_weight_working_bytes':weight_work,'gpu_gradient_working_bytes':grad_work,
            'gpu_cast_working_bytes':cache_work,'gpu_communication_bytes':communication,
            'gpu_staging_bytes':gpu_staging,'cpu_staging_bytes':cpu_staging,
            'gpu_optimizer_workspace':opt_gpu,'cpu_optimizer_workspace':opt_cpu,
            'gather_unit_params':unit,'offload_tile_params':tile,
            'formula_ids':['gather_working_set','communication_buffers','offload_staging']}


def _host_capacity(hardware,key,needed):
    if needed is None: return {'status':'incomplete_estimate'}
    if not hardware: return {'status':'not_provided'}
    cfg=hardware.get(key,{})
    candidates=('available_ram_gib','ram_gib') if key=='cpu' else ('available_gib','capacity_gib')
    found=next((x for x in candidates if x in cfg),None)
    if found is None: return {'status':'not_provided'}
    available=number(cfg[found],f'{key}.{found}')*GIB
    return {'status':'exceeds_budget' if needed>available else 'below_budget',
            'available_bytes':available,'needed_bytes':needed,
            'note':'Includes modeled job allocations only; OS/dataset/checkpoint requirements '
                   'need shared per-node allowances. Storage must be appropriate for offload.'}


def estimate_gpu_memory(workload,assumptions=None,hardware=None,*,deployment=None,graph=None):
    """v2: per-rank GPU/CPU/NVMe placement plus analytic or graph transient memory.

    deployment or training.memory_deployment describes ownership/placement.
    graph or workload.computation_graph is complete local-batch tensor liveness.
    World size does NOT divide activations. Output legacy model_state_bytes is
    global unpartitioned logical states; per_rank_state_bytes is separately named.
    """
    import copy
    if not isinstance(workload,dict): raise ValueError('workload must be an object')
    a=dict(assumptions or {});notes=[]
    try: d=resolve_deployment(workload,deployment)
    except Unsupported as exc:
        return {'schema_version':2,'status':'unsupported_configuration',
                'missing_or_unsupported':[str(exc)],'planning_scenarios':None,
                'components':{},'model_state_bytes':None,'sources':SOURCES,'formula_catalog':FORMULAS}
    g=graph if graph is not None else workload.get('computation_graph')
    gm=graph_memory(g) if g is not None else None
    local=copy.deepcopy(workload)
    t=local.setdefault('training',{})
    if (t.get('torch_compile') or t.get('cuda_graphs') or t.get('retain_graph') or t.get('create_graph')) and gm is None:
        raise Unsupported('Compiled/retained graphs require complete explicit tensor lifetimes.')
    # Normalize for global tensor accounting ONLY. Distribution is applied below.
    for key in ('offload','offload_param','offload_optimizer','zero_stage','deepspeed','fsdp',
                'torch_compile','cuda_graphs','retain_graph','create_graph'):
        t.pop(key,None)
    t['gpu_count']=1;t['distributed_strategy']='single'
    if gm is not None:
        if any(key in a for key in ('saved_activation_bytes','auxiliary_tensor_bytes')):
            raise ValueError('Use either graph metadata or saved/auxiliary overrides, not both')
        # No generic activation guesses: use exact supplied graph peak in v2 below.
        a['saved_activation_bytes']=a['auxiliary_tensor_bytes']=0
        notes.append('Graph liveness replaces saved+auxiliary totals; its saved-storage '
                     'diagnostic is not added a second time. Manifest must cover a full '
                     'representative local training step, including input/output/loss lifetimes.')
    base=_estimate_single_gpu_legacy(local,a,hardware)
    notes.extend(base.get('assumptions',[]))
    # The original notes describe the global accounting pass, not final placement.
    notes=[n for n in notes if not n.startswith('All states assumed GPU-resident')]
    base['assumptions']=notes
    base['schema_version']=2;base['scope']='distributed_and_offloaded_dense_training'
    base['deployment']=d;base['graph_memory']=gm
    base['input_assumptions']=dict(assumptions or {})
    placed,persistent,divisors=partition_and_place(workload,base,d,notes)
    base['state_placement']=placed;base['per_rank_state_bytes']=persistent
    base['global_model_state_bytes']=base.get('model_state_bytes')
    base['model_state_scope']='global_unpartitioned_logical_storage'
    try: working=distributed_working_set(workload,d,base,divisors,notes)
    except Unsupported as exc:
        working=None;base['missing_or_unsupported'].append(str(exc))
    base['working_set']=working
    # The legacy missing list includes an incomplete-total note; final totals decide completeness.
    missing=[x for x in base.get('missing_or_unsupported',[]) if not x.startswith('One or more components unknown')]
    # Custom graph removes legacy activation errors by design. No peaks are fabricated.
    comps=base.get('components',{})
    cache_global=comps.get('autocast_weight_cache',{}).get('bytes')
    cache_local=math.ceil(cache_global/divisors['weights']) if cache_global is not None else None
    base['per_rank_autocast_cache_bytes']=cache_local
    if gm:
        graph_gpu=gm['peak_bytes']['gpu'];graph_cpu=gm['peak_bytes']['cpu'];graph_nvme=gm['peak_bytes']['nvme']
        transient=(graph_gpu,graph_gpu)
        kernel_complete=gm['workspace_complete']
    else:
        saved=comps.get('saved_activations',{}).get('bytes')
        aux=comps.get('auxiliary_tensors',{})
        lo,hi=aux.get('low_bytes'),aux.get('high_bytes')
        transient=(saved+lo,saved+hi) if None not in (saved,lo,hi) else (None,None)
        graph_cpu=graph_nvme=0;kernel_complete=False
    scenarios=None
    config=base.get('planning_allowances',{})
    extra=comps.get('additional_persistent',{}).get('bytes',0)
    if (persistent is not None and working is not None and cache_local is not None
            and None not in transient and working['gpu_optimizer_workspace'] is not None and config):
        scenarios={}
        for index,label in enumerate(('low','high')):
            # Sensitivity applies only to saved activations for analytic profiles;
            # a graph peak is already a liveness result, so no activation-only multiplier.
            if gm:
                active=transient[index]
                if config['activation_multiplier_'+label]!=1:
                    notes.append('Activation multiplier ignored for explicit graph lifetimes.')
            else:
                active=comps['saved_activations']['bytes']*config['activation_multiplier_'+label]+aux[label+'_bytes']
            kernel=0 if kernel_complete else (config['kernel_workspace_'+label+'_mib']*MIB+
                    config['kernel_workspace_fraction_'+label]*active)
            gpu_work=sum(working[k] for k in ('gpu_weight_working_bytes','gpu_gradient_working_bytes',
                                              'gpu_cast_working_bytes','gpu_communication_bytes','gpu_staging_bytes'))
            optgpu=working['gpu_optimizer_workspace'][index]
            subtotal=persistent['gpu']+cache_local+extra+active+kernel+gpu_work+optgpu
            fragmentation=subtotal*config['fragmentation_fraction_'+label]
            runtime=config['runtime_overhead_'+label+'_mib']*MIB
            gpu=math.ceil(subtotal+fragmentation+runtime)
            cpu=persistent['cpu']+working['cpu_staging_bytes']+working['cpu_optimizer_workspace'][index]+graph_cpu
            nvme=persistent['nvme']+graph_nvme
            sharedcpu=count(d.get('shared_cpu_bytes_per_node',0),'shared_cpu_bytes_per_node')
            shareddisk=count(d.get('shared_nvme_bytes_per_node',0),'shared_nvme_bytes_per_node')
            nodecpu=d['gpus_per_node']*cpu+sharedcpu
            nodedisk=d['gpus_per_node']*nvme+shareddisk
            scenarios[label]={
                'planning_bytes':gpu,'planning_gib':gpu/GIB,
                'gpu_bytes_per_rank':gpu,'gpu_gib_per_rank':gpu/GIB,
                'cpu_bytes_per_rank':cpu,'nvme_bytes_per_rank':nvme,
                'cpu_bytes_per_node':nodecpu,'cpu_gib_per_node':nodecpu/GIB,
                'nvme_bytes_per_node':nodedisk,'nvme_gib_per_node':nodedisk/GIB,
                'transient_graph_or_activation_bytes':active,'kernel_workspace_bytes':kernel,
                'optimizer_workspace_bytes':optgpu,'distributed_working_bytes':gpu_work,
                'allocator_allowance_bytes':fragmentation,'runtime_overhead_bytes':runtime,
                'formula_ids':['planning_total','state_sharding','state_placement','node_memory']}
    else: missing.append('Incomplete state, graph, cache or workspace metadata; full totals remain null.')
    base['planning_scenarios']=scenarios
    base['status']='estimated_with_assumptions' if scenarios else 'incomplete'
    base['missing_or_unsupported']=missing
    effectivehardware=hardware or workload.get('target_hardware')
    base['capacity_check']=capacity_check(effectivehardware,
        scenarios['low']['planning_bytes'] if scenarios else None,
        scenarios['high']['planning_bytes'] if scenarios else None)
    base['node_capacity_checks']={
        'cpu':_host_capacity(effectivehardware,'cpu',scenarios['high']['cpu_bytes_per_node'] if scenarios else None),
        'storage':_host_capacity(effectivehardware,'storage',scenarios['high']['nvme_bytes_per_node'] if scenarios else None)}
    base['scenario_interpretation']='Per-rank planning scenarios for balanced identical ranks. '
    base['scenario_interpretation']+='No statistical bounds, transfer-time forecast or actual framework execution.'
    notes.extend(['CPU/NVMe estimates exclude OS, dataset caches and checkpoint copies unless '
                  'shared per-node bytes include them. NVMe offload requires suitable mounted storage.',
                  'Full checkpoint-save, initialization and evaluation peaks are not automatically '
                  'covered by steady training-state planning. Offload does not predict runtime.',
                  'Legacy components/model_state_bytes describe GLOBAL unpartitioned states; '
                  'state_placement and per_rank_state_bytes describe actual modeled residency.'])
    return base


EXAMPLE_DEPLOYMENT={
    'strategy':'zero3','world_size':4,'gpus_per_node':4,
    'largest_gather_unit_params':6464,'prefetch_units':1,
    'offload':{'parameters':'cpu','optimizer':'cpu'},
    'communication_buffers':{'copies':2},
    'shared_cpu_bytes_per_node':0,'shared_nvme_bytes_per_node':0,
}
EXAMPLE_GRAPH={
    'scope':'per_rank','complete_training_step':True,'excludes_model_states':True,
    'workspace_complete':False,
    'description':'Illustrative branch/alias lifetimes, not a graph extracted from example_workload. '
                  'All byte/shape/lifetime values must be replaced for the real computation.',
    'tensors':[
        {'id':'input','shape':[16,100],'dtype':'float32','born':0,'last_use':8,'saved_for_backward':True},
        {'id':'branch_a','shape':[16,64],'dtype':'float32','born':1,'last_use':7,'saved_for_backward':True},
        {'id':'branch_a_view','shape':[16,64],'dtype':'float32','born':2,'last_use':5,
         'storage_id':'branch_a','storage_bytes':4096},
        {'id':'branch_b','shape':[16,64],'dtype':'float32','born':3,'last_use':6,'saved_for_backward':True},
        {'id':'output','shape':[16,1],'dtype':'float32','born':4,'last_use':6},
        {'id':'targets','shape':[16,1],'dtype':'float32','born':0,'last_use':6},
        {'id':'backward_scratch','shape':[16,64],'dtype':'float32','born':6,'last_use':6},
    ],
    'workspaces':[{'step':6,'gpu_bytes':1048576}],
}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('workload',nargs='?')
    parser.add_argument('--assumptions')
    parser.add_argument('--hardware')
    parser.add_argument('--deployment', help='Explicit sharding/offload policy JSON')
    parser.add_argument('--graph', help='Complete per-rank tensor lifetime manifest JSON')
    parser.add_argument('--write-examples', metavar='DIR', help='Write example deployment/graph JSON then exit')
    parser.add_argument('--output',default='memory_estimate.json')
    parser.add_argument('--formulas',action='store_true')
    args=parser.parse_args()
    if args.write_examples:
        folder=Path(args.write_examples);folder.mkdir(parents=True,exist_ok=True)
        for name,value in [('deployment.json',EXAMPLE_DEPLOYMENT),('graph.json',EXAMPLE_GRAPH)]:
            target=folder/name
            if target.exists(): parser.error(f'Refusing to overwrite example file {target}')
        for name,value in [('deployment.json',EXAMPLE_DEPLOYMENT),('graph.json',EXAMPLE_GRAPH)]:
            (folder/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        print(f'Wrote illustrative examples to {folder}');return
    if args.formulas:
        print(json.dumps({'formulas':FORMULAS,'sources':SOURCES},indent=2));return
    if not args.workload: parser.error('Supply collector JSON, or use --formulas')
    inputs=[p for p in (args.workload,args.assumptions,args.hardware,args.deployment,args.graph) if p]
    if Path(args.output).resolve() in {Path(p).resolve() for p in inputs}:
        parser.error('Output must not overwrite an input file')
    try:
        r=estimate_gpu_memory(load_json(args.workload),
            load_json(args.assumptions) if args.assumptions else None,
            load_json(args.hardware) if args.hardware else None,
            deployment=load_json(args.deployment) if args.deployment else None,
            graph=load_json(args.graph) if args.graph else None)
        Path(args.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except (ValueError,KeyError,TypeError,OSError) as exc:
        parser.exit(1,f'Estimation failed: {exc}\n')
    print(f"{r['status']}: saved {args.output}")
    if r.get('model_state_gib') is not None:
        print(f"Global model states before sharding: {r['model_state_gib']:.6f} GiB")
    if r.get('planning_scenarios'):
        low,high=(r['planning_scenarios'][k]['planning_gib'] for k in ('low','high'))
        print(f'GPU per-rank planning scenarios: {low:.3f} / {high:.3f} GiB (not guaranteed bounds)')
        if r['planning_scenarios']['high'].get('cpu_gib_per_node') is not None:
            high_scenario=r['planning_scenarios']['high']
            print(f"High-scenario CPU per node: {high_scenario['cpu_gib_per_node']:.3f} GiB; "
                  f"NVMe per node: {high_scenario['nvme_gib_per_node']:.3f} GiB")


if __name__=='__main__':
    main()
