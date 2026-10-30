#!/usr/bin/env python3
"""LEGACY - NOT USED BY THE GPT-2 MVP. Retained for future profiles/extensions; not imported by
active routes. Active MVP path: app/parsers/gpt2_inputs.py, app/parsers/gpt2_extract.py,
app/services/estimators/estimate_gpt2.py and gpt2_predict.py. Still covered by tests.

Roofline runtime estimator for the MLP / classic-transformer profiles.

GPU training-runtime planner for collect_workload JSON (Python 3.10+, stdlib).

  python estimate_gpu_runtime.py --workload workload.json --dataset dataset.json \
      --hardware hardware.json --gpu-id nvidia_a100_80gb_pcie --output runtime.json
  python estimate_gpu_runtime.py --workload workload.json --dataset dataset.json \
      --hardware hardware.json --assumptions runtime_assumptions.json \
      --deployment deployment.json --output runtime.json
  python estimate_gpu_runtime.py --write-examples examples
  python estimate_gpu_runtime.py --formulas

API: estimate_runtime(workload, dataset, hardware, assumptions=None,
                      deployment=None, gpu_id=None).
Hardware can be a GPU catalog {'gpus': {id: specs}, 'system': {...}}, a machine
{'gpu': {...}, 'storage': {...}, 'interconnect': {...}}, or a GPU-spec object.
When given a multi-GPU catalog, select --gpu-id; no first-entry guessing.
A standalone GPU catalog has no CPU/disk/interconnect allocation information.

No model execution and no automatic benchmarks. Supported FLOP profiles:
- auto/simple_mlp: Sequential(Linear, ReLU, Linear), fully trainable.
- transformer_dense: explicitly selected, dense MHA/GQA attention, dense 2- or
  3-projection MLP, full-token output projection. No MoE/encoder-decoder guessing.
- custom: explicit forward_backward_flops_per_microbatch AND
  hbm_bytes_per_microbatch. Arbitrary graphs need computation/traffic metadata;
  saved activation bytes and parameter counts are not FLOP or traffic counts.

An optional workload.runtime_profile holds overrides; --assumptions takes priority.
Training.max_steps is OPTIMIZER updates. Epoch mode flushes the final partial
accumulation group each epoch. Final partial microbatches are charged at full
nominal batch shape (reported). DataParallel/ZeRO/FSDP split DATA across ranks,
not the per-microbatch compute of a full-model rank. Tensor/pipeline parallel
runtime and hybrid topology require explicit rank/communication modeling.

Default arithmetic/HBM efficiencies, traffic proxies, launch/host latencies,
optimizer arithmetic counts and overlap fractions are UNSOURCED planning choices.
They are not benchmark results or statistical confidence bounds. A roofline on
whole-step totals can miss mixed kernel bottlenecks. Peak TFLOPS are NOT achieved
training throughput, and no sparsity-based peak is silently used for dense work.

Data I/O must be described, or assume_data_ready=true explicitly asserts data
supply/H2D is hidden/already available. Missing required disk/interconnect/CPU
optimizer/communication information yields partial GPU-core times and null total
training time rather than silently treating unknown costs as zero.

Optional calibration = {gpu_id, batch_size_per_gpu, precision,
 sequence_length (when relevant), workload_fingerprint, forward_backward_seconds}. This is a matched
forward+loss+backward time, INCLUDING host/launch overhead but EXCLUDING optimizer,
data loading, offload, collectives and evaluation. Hardware/config mismatches fail.
Do not supply a whole training-step time in this field.

Evaluation/checkpoint events: assumptions.evaluation/checkpoint may contain
 every_optimizer_updates, seconds_per_event, at_end. No event/startup overhead is
 modeled unless supplied; omissions are explicitly noted. Queue/provisioning time,
OOM/failure/retry and convergence/early stopping are outside scope.
Run the memory estimator separately to establish that the chosen job can fit.
"""
from __future__ import annotations
import argparse
import json
import math
import hashlib
from pathlib import Path

GIB=1024**3
BYTES={'fp32':4,'float32':4,'fp16':2,'float16':2,'bf16':2,'bfloat16':2,
       'fp64':8,'float64':8,'int64':8,'long':8,'int32':4,'int16':2,'int8':1,'uint8':1,'bool':1}
SOURCES={
 'gemm':'https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/',
 'roofline':'https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html',
 'transformer':'https://arxiv.org/html/2205.05198v1#A1',
 'sampler':'https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler',
 'collectives':'https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md',
 'zero':'https://deepspeed.readthedocs.io/en/stable/zero3.html',
 'timing':'https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution',
}
FORMULAS={
 'schedule':{'formula':'samples/rank=ceil(N/D) (or floor with sampler_drop_last); '
             'microbatches/epoch=ceil(samples/rank/B) (or floor with loader drop_last); '
             'updates/epoch=ceil(microbatches/epoch/accumulation)',
             'basis':'Balanced DistributedSampler-like sharding and flush-at-epoch-end policy. '
                     'In max_steps mode, microbatches=max_steps*accumulation. Nominal full '
                     'batch shapes are costed even for the final partial batch.', 'sources':['sampler']},
 'gemm':{'formula':'FLOPs=2*M*N*K', 'basis':'One multiply-add counted as two FLOPs.','sources':['gemm']},
 'mlp':{'formula':'forward=2*B*(I*H+H*O); forward+backward=4*B*I*H+6*B*H*O',
        'basis':'Original GEMM count for full training, no input gradient: first weight gradient '
                'and second weight/input gradients. Add 2*B*I*H when input_requires_grad=true. '
                'Bias, ReLU and loss arithmetic omitted (traffic/latency still budgeted).','sources':['gemm']},
 'transformer':{'formula':'forward=L*(2*B*S*H*(H+2*KV)+2*B*S*H*H+'
       '2*projections*B*S*H*I+4*B*S*S*H*attention_pair_fraction)+2*B*S*H*V; '
       'forward+backward~=3*forward; KV=H*kv_heads/heads',
       'basis':'GEMM-based extension of conventional transformer accounting to GQA and gated '
               'MLPs. Fully trainable, full-token vocabulary projection. Softmax/norm/embedding '
               'scatter arithmetic omitted. 2 projections=ordinary MLP, 3=gated. Causal mask '
               'does not automatically halve GEMM work. This is NOT universal 6*parameter_count '
               'for arbitrary architectures.','sources':['gemm','transformer']},
 'checkpoint':{'formula':'extra FLOPs = recompute_forward_fraction * forward_FLOPs',
       'basis':'Recomputed forward operations; full layer checkpointing defaults to fraction=1 '
               'when enabled without an explicit fraction. Selective checkpointing needs its '
               'own fraction. Optimizer FLOPs are not multiplied.','sources':['transformer']},
 'traffic':{'formula':'HBM proxy=3*weight_bytes+6*activation_proxy_elements*activation_bytes',
       'basis':'UNSOURCED rough traffic proxy for repeat reads/writes. Not resident VRAM and not '
               'a cache/kernel-level traffic simulation. Explicit hbm_bytes_per_microbatch overrides it.','sources':[]},
 'roofline':{'formula':'core_seconds=max(FLOPs/(peak_FLOPs*arithmetic_efficiency), '
       'HBM_bytes/(peak_HBM_bandwidth*bandwidth_efficiency))',
       'basis':'Whole-microbatch roofline approximation. Efficiencies are UNSOURCED scenario '
               'assumptions, not end-to-end measured MFU; kernel-by-kernel summation can differ.','sources':['roofline']},
 'optimizer':{'formula':'HBM bytes/update~=2*weights+gradients+2*optimizer_states+2*extra_master; '
       'update time=max(optimizer_FLOPs/effective_FP32_FLOPs,bytes/effective_HBM_bandwidth)',
       'basis':'Read/write tensor accounting plus UNSOURCED arithmetic allowance: '
               'SGD 4, SGD+momentum 6, Adam/AdamW 16, Adagrad 8, RMSprop 12 FLOPs/parameter. '
               'Sharded updates operate on local owned parameters. CPU offload needs explicit '
               'CPU update time; GPU specifications cannot predict CPU optimizer throughput.','sources':['zero']},
 'communication':{'formula':'ring allreduce bytes/rank=2*(D-1)/D*gradient_bytes; '
       'allgather/reduce-scatter bytes/rank=(D-1)/D*payload; time=bytes/effective_link_Bps+latency',
       'basis':'Ring-equivalent traffic. DDP sync defaults to every microbatch unless '
               'synchronize_each_microbatch=false. ZeRO1/2 also budget updated-weight gather; '
               'ZeRO3/full FSDP budget two weight gathers per microbatch. Exact collective '
               'schedule and effective topology require overrides/calibration.','sources':['collectives','zero']},
 'io':{'formula':'seconds=transferred_bytes/effective_bytes_per_second + preprocessing_seconds',
       'basis':'Disk compressed bytes and CPU-GPU decoded bytes treated separately. Sequential '
               'disk specs discounted by configurable efficiency. Per-rank storage share is '
               'assumed to be node bandwidth/gpus_per_node. No remote-storage speed is inferred.','sources':[]},
 'overlap':{'formula':'combined(a,b)=a+b-overlap_fraction*min(a,b)',
       'basis':'UNSOURCED pipeline approximation: 0=sum, 1=max. Data/communication/offload '
               'overlap settings are assumptions, not measured scheduling behavior.','sources':[]},
 'total':{'formula':'training=microbatch_count*microbatch_seconds+update_count*update_seconds; '
       'job=training+evaluation+checkpoint+startup; GPU-hours=world_size*training_seconds/3600',
       'basis':'Elapsed-time accounting for balanced synchronous ranks, not sum of GPU times. '
               'All GPUs assumed allocated for the entire job. Queue time excluded.','sources':[]},
}

class Missing(ValueError): pass

def num(x,name,minimum=0,integer=False):
    if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<minimum or (integer and int(x)!=x):
        raise ValueError(f'{name} must be finite, >= {minimum}'+(' and integral' if integer else ''))
    return int(x) if integer else float(x)

def size(dtype):
    key=str(dtype).removeprefix('torch.')
    if key not in BYTES: raise Missing(f'Unsupported dtype {dtype}')
    return BYTES[key]

def boolean(x,name):
    if not isinstance(x,bool): raise ValueError(f'{name} must be boolean')
    return x

def read_json(path):
    def invalid(x): raise ValueError(f'Nonfinite JSON value {x}')
    value=json.loads(Path(path).read_text(),parse_constant=invalid)
    if not isinstance(value,dict): raise ValueError(f'{path}: expected JSON object')
    return value

def weight_bytes(w):
    inv=w.get('tensor_inventory',{}).get('weights')
    if inv is not None: return sum(num(x['numel'],'numel',integer=True)*size(x['dtype']) for x in inv)
    return num(w['model'].get('logical_weight_bytes'),'logical_weight_bytes')

def deployment_config(w,d):
    t=w.get('training',{});d=dict(d if d is not None else t.get('memory_deployment',{}))
    strategy=d.get('strategy',t.get('distributed_strategy','single'))
    strategy={'none':'single','fsdp':'fsdp_full_shard'}.get(strategy,strategy)
    if strategy not in ('single','ddp','zero1','zero2','zero3','fsdp_full_shard','fsdp_shard_grad_op','fsdp_hybrid_shard'):
        raise Missing(f'Unsupported distributed strategy {strategy}')
    world=num(d.get('world_size',t.get('gpu_count',1)),'world_size',1,True)
    gpn=num(d.get('gpus_per_node',world),'gpus_per_node',1,True)
    if world%gpn: raise ValueError('gpus_per_node must divide world_size')
    if strategy=='single' and world!=1: raise ValueError('Multiple GPUs require an explicit distributed strategy')
    if any(d.get(x,1)!=1 for x in ('tensor_parallel_size','pipeline_parallel_size')):
        raise Missing('Tensor/pipeline parallelism needs rank-specific compute and timing; not inferred.')
    group=num(d.get('shard_group_size',gpn if strategy=='fsdp_hybrid_shard' else world),'shard_group_size',1,True)
    if world%group: raise ValueError('shard_group_size must divide world_size')
    if strategy in ('single','ddp') and group!=world:
        raise ValueError('DDP/single shard_group_size must equal world_size')
    if strategy=='fsdp_hybrid_shard' and gpn%group:
        raise ValueError('Hybrid shard_group_size must divide gpus_per_node')
    placement=d.get('placement',{});off=d.get('offload',{})
    opt_location=placement.get('optimizer_states',off.get('optimizer','gpu'))
    offloaded=any(x!='gpu' for x in list(placement.values())+list(off.values()))
    if any(x not in ('gpu','cpu','nvme') for x in list(placement.values())+list(off.values())):
        raise ValueError('Offload placement must use gpu/cpu/nvme')
    return {**d,'strategy':strategy,'world_size':world,'gpus_per_node':gpn,
            'shard_group_size':group,'offloaded':offloaded,'optimizer_location':opt_location}

def schedule(w,data,d,notes):
    t=w['training'];ds=data.get('dataset',data);batch=data.get('batching',{})
    B=num(t.get('batch_size_per_gpu'),'batch_size_per_gpu',1,True)
    A=num(t.get('gradient_accumulation_steps',1),'gradient_accumulation_steps',1,True)
    N=ds.get('num_train_examples',t.get('num_examples'))
    if ds.get('num_train_examples') is not None and t.get('num_examples') is not None and ds['num_train_examples']!=t['num_examples']:
        raise ValueError('Dataset num_train_examples disagrees with workload num_examples')
    maximum=t.get('max_steps')
    if maximum is not None and maximum>0:
        updates=num(maximum,'max_steps',1,True);micro=updates*A
        per_epoch=epochs=None
    else:
        N=num(N,'num_train_examples',1,True)
        epochs=num(t.get('epochs'),'epochs',1,True)
        sampler_drop=boolean(t.get('sampler_drop_last',False),'sampler_drop_last')
        loader_drop=boolean(batch.get('drop_last',t.get('drop_last',False)),'drop_last')
        rank_samples=N//d['world_size'] if sampler_drop else math.ceil(N/d['world_size'])
        per_epoch=rank_samples//B if loader_drop else math.ceil(rank_samples/B)
        if not per_epoch: raise ValueError('Batch/drop_last configuration produces no training batches')
        micro=per_epoch*epochs;updates=math.ceil(per_epoch/A)*epochs
        notes.append('Epoch mode flushes partial accumulation at each epoch end. '
                     'Final partial batches use nominal full-batch compute; sampler may pad/duplicate examples.')
    return {'batch_size_per_gpu':B,'gradient_accumulation_steps':A,
            'effective_global_batch_size':B*A*d['world_size'],
            'num_train_examples':N,'epochs':epochs,'microbatches_per_epoch_per_rank':per_epoch,
            'microbatches_per_rank':micro,'optimizer_updates':updates,
            'nominal_example_slots_all_ranks':micro*B*d['world_size']}

def resolve_hardware(hw,gpu_id,w,notes):
    if 'gpus' in hw:
        gpu_id=gpu_id or (w.get('target_hardware') or {}).get('gpu_id')
        if gpu_id is None:
            if len(hw['gpus'])!=1: raise Missing('Select a GPU catalog entry with --gpu-id')
            gpu_id=next(iter(hw['gpus']))
        if gpu_id not in hw['gpus']: raise ValueError(f'Unknown gpu_id: {gpu_id}')
        gpu=hw['gpus'][gpu_id];system=hw.get('system',{})
    else:
        gpu=hw.get('gpu',hw);system=hw
        if gpu_id is not None and gpu.get('gpu_id') is not None and gpu_id!=gpu['gpu_id']:
            raise ValueError('--gpu-id disagrees with the machine hardware gpu_id')
        gpu_id=gpu_id or gpu.get('gpu_id',gpu.get('name'))
    peak=gpu.get('peak_compute_tflops',{})
    if not peak: raise Missing('Hardware requires peak_compute_tflops for the selected GPU')
    memory=gpu.get('memory',{})
    bandwidth=memory.get('bandwidth_gb_per_second',gpu.get('memory_bandwidth_gb_per_second'))
    if bandwidth is None: raise Missing('GPU HBM/VRAM memory bandwidth is required')
    return gpu_id,gpu,system,num(bandwidth,'GPU memory bandwidth',1e-12),peak

def compute_peak(peak,precision,a,notes):
    key=str(precision).removeprefix('torch.')
    key={'float32':'fp32','float16':'fp16','bfloat16':'bf16','float64':'fp64'}.get(key,key)
    if key not in ('fp32','fp16','bf16','fp64'): raise Missing(f'Unsupported compute precision {precision}')
    if 'compute_peak_key' in a: chosen=a['compute_peak_key']
    else:
        chosen=key+'_tensor_dense' if key in ('fp16','bf16') and peak.get(key+'_tensor_dense') is not None else key
    if 'sparse' in str(chosen): raise ValueError('Sparse peak cannot be used for this dense runtime model')
    if chosen=='tf32_tensor_dense' and key!='fp32': raise ValueError('TF32 selection requires fp32 storage/compute configuration')
    if chosen!=key and chosen not in (key+'_tensor_dense','tf32_tensor_dense'):
        raise ValueError('Selected peak key does not match configured precision')
    if peak.get(chosen) is None: raise Missing(f'Hardware peak {chosen} is missing; no other precision is substituted')
    if chosen=='tf32_tensor_dense': notes.append('TF32 selected explicitly: assumes eligible FP32 GEMMs actually use TF32.')
    return chosen,num(peak[chosen],chosen,1e-12)*1e12

def compute_profile(w,data,sc,a,notes):
    t=w['training'];m=w['model'];B=sc['batch_size_per_gpu']
    kind=a.get('runtime_model','auto');S=None;forward=None;proxy=None
    if 'forward_backward_flops_per_microbatch' in a:
        fb=num(a['forward_backward_flops_per_microbatch'],'forward_backward_flops_per_microbatch')
        forward=a.get('forward_flops_per_microbatch')
        if forward is not None: forward=num(forward,'forward_flops_per_microbatch')
        if 'hbm_bytes_per_microbatch' not in a: raise Missing('Custom FLOPs require hbm_bytes_per_microbatch; resident VRAM is not traffic')
        traffic=num(a['hbm_bytes_per_microbatch'],'hbm_bytes_per_microbatch');kind='custom'
        launches=num(a.get('kernel_launches_per_microbatch',0),'kernel_launches_per_microbatch',integer=True)
        notes.append('Custom FLOPs and HBM traffic supplied explicitly. Launch count defaults to zero if omitted.')
    else:
        if m.get('trainable_params')!=m.get('total_params'):
            raise Missing('Partial fine-tuning/LoRA needs explicit FLOPs and HBM traffic; parameter fraction alone is insufficient')
        if kind=='auto':
            if m.get('class')=='Sequential' and m.get('module_types')=={'Sequential':1,'Linear':2,'ReLU':1}: kind='simple_mlp'
            else: raise Missing('Select runtime_model=transformer_dense or supply custom FLOPs/traffic for this graph')
        b=size(a.get('activation_dtype',t.get('precision')))
        cfg={**m.get('provided_config',{}),**m.get('architecture',{})}
        if kind=='simple_mlp':
            inv={x['name']:x for x in w.get('tensor_inventory',{}).get('weights',[])}
            if '0.weight' in inv and '2.weight' in inv:
                H,I=inv['0.weight']['shape'];O,H2=inv['2.weight']['shape']
                if H!=H2: raise ValueError('Inconsistent MLP shapes')
            else:
                I,H,O=(num(cfg.get(k),k,1,True) for k in ('input_size','hidden_size','output_size'))
            forward=2*B*(I*H+H*O)
            fb=4*B*I*H+6*B*H*O
            if boolean(t.get('input_requires_grad',False),'input_requires_grad'): fb+=2*B*I*H
            proxy=B*(I+2*H+O);launches=12
            notes.append('MLP profile is full Linear-ReLU-Linear training. Bias/ReLU/loss FLOPs omitted; '
                         'launch count 12 and traffic proxy are unsourced planning approximations.')
        elif kind=='transformer_dense':
            def dim(*keys):
                for key in keys:
                    if cfg.get(key) is not None: return num(cfg[key],key,1,True)
                raise Missing(f'Missing transformer dimension {keys[0]}')
            L,H,heads=dim('num_hidden_layers','n_layer'),dim('hidden_size','n_embd'),dim('num_attention_heads','n_head')
            KVheads=num(cfg.get('num_key_value_heads',heads),'num_key_value_heads',1,True)
            if H%heads or heads%KVheads: raise ValueError('Invalid attention-head dimensions')
            I=num(cfg.get('intermediate_size',4*H),'intermediate_size',1,True);V=dim('vocab_size')
            S=data.get('batching',{}).get('effective_sequence_length',a.get('effective_sequence_length',t.get('sequence_length')))
            S=num(S,'effective_sequence_length',1,True)
            if data.get('batching',{}).get('padding')=='longest_in_batch' and 'effective_sequence_length' not in data.get('batching',{}) and 'effective_sequence_length' not in a:
                notes.append('Variable padded batches use workload sequence_length, not the mean raw length; '
                             'provide expected padded effective_sequence_length for runtime accuracy.')
            projections=num(a.get('mlp_projection_count',2),'mlp_projection_count',2,True)
            if projections not in (2,3): raise ValueError('mlp_projection_count must be 2 or 3')
            pairs=num(a.get('attention_pair_fraction',1),'attention_pair_fraction',1e-12)
            if pairs>1: raise ValueError('attention_pair_fraction must be <=1')
            KV=H*KVheads/heads
            forward=L*(2*B*S*H*(H+2*KV)+2*B*S*H*H+2*projections*B*S*H*I+4*B*S*S*H*pairs)+2*B*S*H*V
            fb=3*forward
            proxy=L*B*S*(5*H+projections*I)+B*S*V
            if t.get('attention_backend') in ('eager','standard'): proxy+=L*B*heads*S*S
            launches=60*L+5
            notes.append('Transformer FLOPs assume fully trainable dense layers, full-token logits, '
                         'and backward~=2*forward. Default MLP has TWO projections; gated MLPs '
                         'require mlp_projection_count=3. Flash/SDPA traffic and fused kernels '
                         'need explicit traffic/launch calibration. Softmax/norm FLOPs omitted.')
        else: raise Missing(f'Unknown runtime_model {kind}')
        traffic=num(a.get('hbm_bytes_per_microbatch',3*weight_bytes(w)+6*proxy*b),'hbm_bytes_per_microbatch')
        launches=num(a.get('kernel_launches_per_microbatch',launches),'kernel_launches_per_microbatch',integer=True)
        notes.append('Default HBM traffic is 3*weight_bytes+6*activation-proxy bytes; '
                     'this UNSOURCED heuristic is not peak VRAM or measured memory traffic.')
    recompute=a.get('recompute_forward_fraction',1 if t.get('gradient_checkpointing') else 0)
    recompute=num(recompute,'recompute_forward_fraction')
    if recompute and forward is None: raise Missing('Checkpoint recomputation needs forward_flops_per_microbatch')
    fb+=recompute*(forward or 0)
    # Traffic for checkpoint recomputation is not determined by FLOPs. Default factor is explicit.
    traffic*=num(a.get('checkpoint_traffic_multiplier',1+recompute/3),'checkpoint_traffic_multiplier',1e-12)
    if recompute and 'kernel_launches_per_microbatch' not in a:
        launches=math.ceil(launches*(1+recompute/3))
    if recompute: notes.append('Checkpointing adds the specified fraction of one forward pass; '
                              'default traffic and launch multipliers 1+fraction/3 are UNSOURCED approximations.')
    return {'kind':kind,'forward_flops_per_microbatch':forward,
            'forward_backward_flops_per_microbatch':fb,'hbm_bytes_per_microbatch_proxy':traffic,
            'kernel_launches_per_microbatch':launches,'effective_sequence_length':S}

def optimizer_work(w,d,a,notes):
    t=w['training'];m=w['model'];cfg=w.get('configured_optimizer',{})
    name=str(t.get('optimizer') or cfg.get('name') or w.get('optimizer',{}).get('class')).split('.')[-1].lower()
    opts=t.get('optimizer_options') or cfg.get('options') or {}
    if 'optimizer_flops_per_update' in a and 'optimizer_hbm_bytes_per_update' in a:
        return {'flops':num(a['optimizer_flops_per_update'],'optimizer_flops_per_update'),
                'bytes':num(a['optimizer_hbm_bytes_per_update'],'optimizer_hbm_bytes_per_update'),
                'compute_dtype':a.get('optimizer_compute_dtype','fp32'),
                'launches':num(a.get('optimizer_launches_per_update',6),'optimizer_launches_per_update',integer=True)}
    group_snapshot=w.get('optimizer',{}).get('parameter_groups') or []
    if len(group_snapshot)==1: opts={**opts,**group_snapshot[0]}
    momentum=num(opts.get('momentum',0),'momentum')
    if name in ('adam','adamw'): states=2+int(boolean(opts.get('amsgrad',False),'amsgrad'));ops=16
    elif name=='sgd': states=int(momentum>0);ops=6 if momentum else 4
    elif name=='adagrad': states=1;ops=8
    elif name=='rmsprop': states=1+int(momentum>0)+int(boolean(opts.get('centered',False),'centered'));ops=12
    else: raise Missing(f'Optimizer {name} needs explicit per-rank optimizer FLOPs and traffic')
    groups=w.get('optimizer',{}).get('parameter_groups') or []
    if len(groups)>1: raise Missing('Multiple optimizer groups need explicit optimizer_flops_per_update and optimizer_hbm_bytes_per_update')
    pt=num(m.get('trainable_params'),'trainable_params',integer=True)
    divisor=d['shard_group_size'] if d['strategy'] not in ('single','ddp') else 1
    local=math.ceil(pt/divisor)
    gradient_dtype=t.get('gradient_dtype')
    if gradient_dtype is None: raise Missing('gradient_dtype is needed for optimizer and communication accounting')
    state_dtype=t.get('optimizer_state_dtype',cfg.get('state_dtype'))
    if states and state_dtype is None: raise Missing('optimizer_state_dtype is needed')
    floating={'fp16','bf16','fp32','fp64','float16','bfloat16','float32','float64'}
    if str(gradient_dtype).removeprefix('torch.') not in floating or (states and str(state_dtype).removeprefix('torch.') not in floating):
        raise Missing('Quantized gradients/optimizer states require explicit optimizer work and traffic overrides')
    master=t.get('master_weights',cfg.get('master_weights'))
    if master not in ('none','fp32'): raise Missing('master_weights must explicitly be none or fp32')
    bw=weight_bytes(w)/max(1,num(m['total_params'],'total_params',1,True))
    traffic=local*(2*bw+size(gradient_dtype)+2*states*(size(state_dtype) if states else 0)+2*(4 if master=='fp32' else 0))
    notes.append('Optimizer arithmetic/launch counts are heuristic. Traffic counts read/write '
                 'weights, gradients, moments and extra master copies. Frozen/trainable mixed '
                 'weight dtypes need explicit per-rank overrides.')
    return {'flops':num(a.get('optimizer_flops_per_update',local*ops),'optimizer_flops_per_update'),
            'bytes':num(a.get('optimizer_hbm_bytes_per_update',traffic),'optimizer_hbm_bytes_per_update'),
            'compute_dtype':a.get('optimizer_compute_dtype','fp64' if max(bw,size(gradient_dtype),size(state_dtype) if states else 0)>4 else 'fp32'),
            'launches':num(a.get('optimizer_launches_per_update',6),'optimizer_launches_per_update',integer=True)}

def input_batch_bytes(w,data,sc,profile):
    sample=data.get('sample',{});B=sc['batch_size_per_gpu'];S=profile['effective_sequence_length']
    if sample.get('inputs'):
        specs=list(sample['inputs'])+([sample['target']] if sample.get('target') else [])
        total=0
        for spec in specs:
            dims=[S if x=='sequence_length' else num(x,'sample dimension',1,True) for x in spec.get('shape',[])]
            if any(x is None for x in dims): raise Missing('Sequence dimension needs effective_sequence_length')
            total+=B*math.prod(dims)*size(spec['dtype'])
        return total
    specs=w.get('input_specs') or w['training'].get('input_specs')
    if not specs: raise Missing('Input specs required for decoded host-to-GPU bytes')
    total=0
    for spec in specs:
        shape=list(spec['shape'])
        if not shape or shape[0]!=B: raise ValueError('Workload input specs must include the nominal batch dimension')
        if S is not None and len(shape)==2 and shape[1]==w['training'].get('sequence_length'): shape[1]=S
        total+=math.prod(num(x,'input dimension',1,True) for x in shape)*size(spec['dtype'])
    return total

def overlap(first,second,fraction): return first+second-fraction*min(first,second)

def event_seconds(config,updates,name):
    if not config: return 0,0
    every=config.get('every_optimizer_updates')
    if every is None:
        count_events=0
    else:
        every=num(every,name+'.every_optimizer_updates',1,True);count_events=updates//every
    at_end=boolean(config.get('at_end',False),name+'.at_end')
    if at_end and (every is None or updates%every): count_events+=1
    if count_events and config.get('seconds_per_event') is None: raise Missing(f'{name}.seconds_per_event required')
    return count_events,count_events*num(config.get('seconds_per_event',0),name+'.seconds_per_event')

DEFAULTS={
 'arithmetic_efficiency_fast':0.45,'arithmetic_efficiency_slow':0.15,
 'hbm_efficiency_fast':0.80,'hbm_efficiency_slow':0.40,
 'hbm_traffic_multiplier_fast':1.0,'hbm_traffic_multiplier_slow':2.0,
 'launch_latency_us_fast':3,'launch_latency_us_slow':10,
 'host_overhead_us_fast':20,'host_overhead_us_slow':100,
 'storage_efficiency_fast':0.80,'storage_efficiency_slow':0.40,
 'data_overlap_fast':1.0,'data_overlap_slow':0.5,
 'communication_overlap_fast':0.5,'communication_overlap_slow':0,
 'offload_overlap_fast':0.5,'offload_overlap_slow':0,
 'collective_latency_us':20,
}
EXTRA_KEYS={
 'runtime_model','compute_peak_key','forward_backward_flops_per_microbatch','forward_flops_per_microbatch',
 'hbm_bytes_per_microbatch','kernel_launches_per_microbatch','activation_dtype','effective_sequence_length',
 'mlp_projection_count','attention_pair_fraction','recompute_forward_fraction','checkpoint_traffic_multiplier',
 'optimizer_flops_per_update','optimizer_hbm_bytes_per_update','optimizer_launches_per_update','optimizer_compute_dtype',
 'assume_data_ready','preprocessing_seconds_per_microbatch','cpu_optimizer_seconds_per_update',
 'communication_bytes_per_microbatch','communication_bytes_per_update',
 'collective_calls_per_microbatch','collective_calls_per_update',
 'offload_cpu_gpu_bytes_per_microbatch','offload_cpu_gpu_bytes_per_update',
 'offload_nvme_bytes_per_microbatch','offload_nvme_bytes_per_update',
 'nvme_effective_gb_per_second','calibration','evaluation','checkpoint','startup_seconds',
}

def estimate_runtime(workload,dataset,hardware,assumptions=None,deployment=None,gpu_id=None):
    w,data=workload,dataset
    a={**w.get('runtime_profile',{}),**(assumptions or {})}
    unknown=set(a)-set(DEFAULTS)-EXTRA_KEYS
    if unknown: raise ValueError(f'Unknown runtime assumption keys: {sorted(unknown)}')
    options={**DEFAULTS,**a};notes=[];missing=[]
    for key in DEFAULTS:
        options[key]=num(options[key],key)
        if ('efficiency' in key or 'overlap' in key) and not 0<=options[key]<=1:
            raise ValueError(f'{key} must be between 0 and 1')
        if 'efficiency' in key and options[key]==0: raise ValueError(f'{key} must be positive')
    for key in ('arithmetic_efficiency','hbm_efficiency','storage_efficiency','data_overlap','communication_overlap','offload_overlap'):
        if options[key+'_fast']<options[key+'_slow']: raise ValueError(f'{key}_fast must be >= slow')
    for key in ('hbm_traffic_multiplier','launch_latency_us','host_overhead_us'):
        if options[key+'_fast']>options[key+'_slow']: raise ValueError(f'{key}_fast must be <= slow')
    report={'schema_version':1,'status':'incomplete','sources':SOURCES,'formula_catalog':FORMULAS,
        'assumptions':notes,'missing_information':missing,
        'scope':'training execution time; no queue/provisioning or memory-fit guarantee',
        'scenario_interpretation':'fast/slow are editable planning scenarios, not confidence bounds',
        'runtime_settings':options}
    try:
        d=deployment_config(w,deployment);sc=schedule(w,data,d,notes)
        selected,gpu,system,bandwidth,peaks=resolve_hardware(hardware,gpu_id,w,notes)
        precision=w['training'].get('precision')
        peak_key,peak=compute_peak(peaks,precision,a,notes)
        profile=compute_profile(w,data,sc,a,notes)
        opt=optimizer_work(w,d,a,notes)
    except Missing as exc:
        missing.append(str(exc));return report
    if opt['compute_dtype'] not in ('fp32','fp64'):
        raise ValueError('optimizer_compute_dtype must be fp32 or fp64')
    fingerprint_payload={
        'model':w['model'],
        'weight_shapes_dtypes':[{k:x.get(k) for k in ('name','shape','numel','dtype')}
            for x in w.get('tensor_inventory',{}).get('weights',[])],
        'training_execution':{k:w['training'].get(k) for k in
            ('precision','gradient_dtype','gradient_checkpointing','attention_backend',
             'training_method','input_requires_grad')},
        'profile':profile,
    }
    fingerprint=hashlib.sha256(json.dumps(fingerprint_payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    report.update(deployment=d,schedule=sc,selected_gpu_id=selected,compute_peak_key=peak_key,
                  computation=profile,optimizer_work=opt,workload_fingerprint=fingerprint)
    ds=data.get('dataset',data);N=sc['num_train_examples'];B=sc['batch_size_per_gpu']
    storage=system.get('storage',{});links=system.get('interconnect',{})
    data_ready=boolean(a.get('assume_data_ready',False),'assume_data_ready')
    data_disk_bytes=data_decoded=None;diskbw=h2dbw=None
    if not data_ready:
        try:
            data_decoded=input_batch_bytes(w,data,sc,profile)
            trainbytes=ds.get('train_on_disk_size_bytes')
            if trainbytes is None and ds.get('on_disk_size_gib') is not None:
                trainbytes=num(ds['on_disk_size_gib'],'on_disk_size_gib')*GIB
                validation=ds.get('num_validation_examples',0)
                if validation and N:
                    trainbytes*=N/(N+num(validation,'num_validation_examples'))
                    notes.append('Dataset on_disk_size_gib apportioned to train split by example count; '
                                 'supply train_on_disk_size_bytes if split sample sizes differ.')
            if trainbytes is None or not N: raise Missing('Data loading needs train_on_disk_size_bytes and num_train_examples')
            data_disk_bytes=num(trainbytes,'train_on_disk_size_bytes')/N*B
            read=storage.get('sequential_read_gb_per_second')
            h2d=links.get('cpu_gpu_effective_gb_per_second')
            if read is None: raise Missing('Storage read bandwidth missing; or explicitly set assume_data_ready=true')
            if h2d is None: raise Missing('CPU-GPU effective bandwidth missing; or explicitly set assume_data_ready=true')
            diskbw=num(read,'storage read bandwidth',1e-12)*1e9/d['gpus_per_node']
            h2dbw=num(h2d,'CPU-GPU bandwidth',1e-12)*1e9
        except Missing as exc: missing.append(str(exc))
    else: notes.append('assume_data_ready=true: disk, preprocessing and H2D time excluded by explicit user assumption.')
    notes.extend(['Default efficiencies, HBM traffic multipliers, kernel/host latencies and overlap '
                  'fractions are UNSOURCED scenario assumptions, not measured performance.',
                  'No automatic evaluation, checkpoint, model-load/compilation/startup cost is '
                  'included unless provided. Pipeline ramp-up and per-epoch restarts are not separately modeled.',
                  'Roofline uses full-microbatch aggregate work; different kernels can have '
                  'different bottlenecks. Tensor Core alignment, fusion and software versions affect efficiency.'])
    calibration=a.get('calibration')
    if calibration:
        expected={'gpu_id':selected,'batch_size_per_gpu':B,'precision':precision,
                  'workload_fingerprint':fingerprint}
        if profile['effective_sequence_length'] is not None: expected['sequence_length']=profile['effective_sequence_length']
        for key,value in expected.items():
            if calibration.get(key)!=value: raise ValueError(f'Calibration mismatch/missing field: {key}')
        measured=num(calibration.get('forward_backward_seconds'),'forward_backward_seconds',1e-12)
        notes.append('Matched forward+loss+backward calibration replaces roofline and launch/host costs; '
                     'collectives, offload, data and optimizer must be excluded from that measured scope.')
    scenarios={}
    for label in ('fast','slow'):
        ce=options['arithmetic_efficiency_'+label];be=options['hbm_efficiency_'+label]
        compute=profile['forward_backward_flops_per_microbatch']/(peak*ce)
        memory=profile['hbm_bytes_per_microbatch_proxy']*options['hbm_traffic_multiplier_'+label]/(bandwidth*1e9*be)
        launches=profile['kernel_launches_per_microbatch']*options['launch_latency_us_'+label]*1e-6
        host=options['host_overhead_us_'+label]*1e-6
        core=measured if calibration else max(compute,memory)+launches+host
        issues=[]
        if data_ready: data_time=0
        elif None not in (data_disk_bytes,data_decoded,diskbw,h2dbw):
            data_time=data_disk_bytes/(diskbw*options['storage_efficiency_'+label])+data_decoded/h2dbw+num(a.get('preprocessing_seconds_per_microbatch',0),'preprocessing_seconds_per_microbatch')
            notes.append('CPU preprocessing defaults to zero: assumed preprocessed data. Storage is '
                         'shared evenly per node; decoded H2D size differs from compressed disk size.')
        else: data_time=None;issues.append('Incomplete data-pipeline inputs')
        # Optimizer uses scalar FP32 throughput rather than tensor BF16/FP16 throughput.
        if d['optimizer_location']=='gpu':
            scalar_peak=peaks.get(opt['compute_dtype'])
            if scalar_peak is None: optimizer_time=None;issues.append(f"{opt['compute_dtype']} throughput missing for GPU optimizer")
            else:
                optimizer_time=max(opt['flops']/(num(scalar_peak,'optimizer scalar peak',1e-12)*1e12*ce),
                    opt['bytes']/(bandwidth*1e9*be))+opt['launches']*options['launch_latency_us_'+label]*1e-6
        elif 'cpu_optimizer_seconds_per_update' in a:
            optimizer_time=num(a['cpu_optimizer_seconds_per_update'],'cpu_optimizer_seconds_per_update')
        else: optimizer_time=None;issues.append('CPU/NVMe optimizer requires cpu_optimizer_seconds_per_update')
        comm_micro=comm_update=0
        if d['world_size']>1:
            try:
                if 'communication_bytes_per_microbatch' in a and 'communication_bytes_per_update' in a:
                    cm=num(a['communication_bytes_per_microbatch'],'communication_bytes_per_microbatch')
                    cu=num(a['communication_bytes_per_update'],'communication_bytes_per_update')
                else:
                    if d['strategy']=='fsdp_hybrid_shard' or d['shard_group_size']!=d['world_size']:
                        raise Missing('Replicated/hybrid topology needs explicit communication bytes per microbatch/update')
                    grad=num(w['model']['trainable_params'],'trainable_params',integer=True)*size(w['training']['gradient_dtype'])
                    W=weight_bytes(w);factor=(d['shard_group_size']-1)/d['shard_group_size']
                    sync=boolean(w['training'].get('synchronize_each_microbatch',True),'synchronize_each_microbatch')
                    strategy=d['strategy']
                    if strategy in ('ddp','zero1'):
                        cm=2*factor*grad if sync else 0;cu=0 if sync else 2*factor*grad
                    else: cm=factor*grad;cu=0
                    if strategy in ('zero1','zero2','fsdp_shard_grad_op'): cu+=factor*W
                    elif strategy in ('zero3','fsdp_full_shard'): cm+=2*factor*W
                    notes.append('Collective traffic uses ring-equivalent approximations and full gradient dtype. '
                                 'ZeRO/FSDP gather schedules need calibration; sync defaults to every microbatch.')
                link=links.get('gpu_gpu_effective_gb_per_second')
                if link is None: raise Missing('Distributed runtime requires effective GPU-GPU/network bandwidth')
                link=num(link,'GPU-GPU effective bandwidth',1e-12)*1e9
                calls_micro=num(a.get('collective_calls_per_microbatch',1 if cm else 0),'collective_calls_per_microbatch',integer=True)
                calls_update=num(a.get('collective_calls_per_update',1 if cu else 0),'collective_calls_per_update',integer=True)
                comm_micro=cm/link+calls_micro*options['collective_latency_us']*1e-6
                comm_update=cu/link+calls_update*options['collective_latency_us']*1e-6
                notes.append('Collective call count defaults to one per nonzero traffic category. '
                             'Layer/bucket-level collectives require explicit call counts for latency accuracy.')
            except Missing as exc:
                comm_micro=comm_update=None;issues.append(str(exc))
        off_micro=off_update=0
        if d['offloaded']:
            try:
                values={key:num(a[key],key) if key in a else None for key in (
                    'offload_cpu_gpu_bytes_per_microbatch','offload_cpu_gpu_bytes_per_update',
                    'offload_nvme_bytes_per_microbatch','offload_nvme_bytes_per_update')}
                if any(value is None for value in values.values()):
                    raise Missing('Offload needs explicit CPU-GPU and NVMe bytes per microbatch/update (zero allowed)')
                cb=links.get('cpu_gpu_effective_gb_per_second')
                if cb is None: raise Missing('Offload needs CPU-GPU effective bandwidth')
                cb=num(cb,'CPU-GPU effective bandwidth',1e-12)*1e9
                nvme_bytes=sum(value for key,value in values.items() if 'nvme' in key)
                nb=a.get('nvme_effective_gb_per_second')
                if nvme_bytes and nb is None: raise Missing('NVMe offload needs nvme_effective_gb_per_second per rank')
                nb=num(nb,'NVMe effective bandwidth',1e-12)*1e9 if nb is not None else 1
                off_micro=values['offload_cpu_gpu_bytes_per_microbatch']/cb+values['offload_nvme_bytes_per_microbatch']/nb
                off_update=values['offload_cpu_gpu_bytes_per_update']/cb+values['offload_nvme_bytes_per_update']/nb
            except Missing as exc:
                off_micro=off_update=None;issues.append(str(exc))
        training_seconds=None;job_seconds=None;extras={}
        if None not in (data_time,optimizer_time,comm_micro,comm_update,off_micro,off_update):
            micro_time=overlap(core,data_time,options['data_overlap_'+label])
            micro_time=overlap(micro_time,comm_micro,options['communication_overlap_'+label])
            micro_time=overlap(micro_time,off_micro,options['offload_overlap_'+label])
            update_time=optimizer_time+comm_update+off_update
            training_seconds=sc['microbatches_per_rank']*micro_time+sc['optimizer_updates']*update_time
            try:
                for event in ('evaluation','checkpoint'):
                    count_events,seconds=event_seconds(a.get(event),sc['optimizer_updates'],event)
                    extras[event]={'events':count_events,'seconds':seconds}
                extras['startup_seconds']=num(a.get('startup_seconds',0),'startup_seconds')
                job_seconds=training_seconds+sum(extras[x]['seconds'] for x in ('evaluation','checkpoint'))+extras['startup_seconds']
            except Missing as exc: issues.append(str(exc))
        scenarios[label]={
            'compute_arithmetic_seconds_per_microbatch':compute,
            'hbm_transfer_seconds_per_microbatch_proxy':memory,
            'forward_backward_core_seconds_per_microbatch':core,
            'data_supply_seconds_per_microbatch':data_time,
            'communication_seconds_per_microbatch':comm_micro,'communication_seconds_per_update':comm_update,
            'offload_seconds_per_microbatch':off_micro,'offload_seconds_per_update':off_update,
            'optimizer_seconds_per_update':optimizer_time,
            'gpu_core_only_forward_backward_seconds':core*sc['microbatches_per_rank'],
            'training_seconds':training_seconds,'training_hours':training_seconds/3600 if training_seconds is not None else None,
            'job_seconds':job_seconds,'job_hours':job_seconds/3600 if job_seconds is not None else None,
            'training_gpu_hours':training_seconds/3600*d['world_size'] if training_seconds is not None else None,
            'job_gpu_hours':job_seconds/3600*d['world_size'] if job_seconds is not None else None,
            'extra_job_overhead':extras,'missing':issues}
        missing.extend(issues)
    report['scenarios']=scenarios
    report['status']='estimated_with_assumptions' if all(x['training_seconds'] is not None for x in scenarios.values()) else 'partial_compute_only'
    report['assumptions']=list(dict.fromkeys(notes));report['missing_information']=list(dict.fromkeys(missing))
    return report

EXAMPLE_DATASET={
 'dataset':{'num_train_examples':100000,'num_validation_examples':0,
            'train_on_disk_size_bytes':40400000,'location_type':'local_ssd'},
 'sample':{'inputs':[{'name':'features','shape':[100],'dtype':'float32'}],
           'target':{'name':'targets','shape':[1],'dtype':'float32'}},
 'batching':{'drop_last':False},
}
EXAMPLE_HARDWARE={
 'gpu':{'gpu_id':'illustrative_gpu','name':'Illustrative GPU, not a real SKU',
        'peak_compute_tflops':{'fp32':30,'fp16_tensor_dense':120,'bf16_tensor_dense':120},
        'memory_bandwidth_gb_per_second':900},
 'storage':{'sequential_read_gb_per_second':3},
 'interconnect':{'cpu_gpu_effective_gb_per_second':24},
}
EXAMPLE_ASSUMPTIONS={'runtime_model':'auto','assume_data_ready':False,
                     'preprocessing_seconds_per_microbatch':0,'startup_seconds':0}

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('workload','dataset','hardware','assumptions','deployment','gpu-id'): p.add_argument('--'+name)
    p.add_argument('--output',default='runtime_estimate.json')
    p.add_argument('--formulas',action='store_true');p.add_argument('--write-examples',metavar='DIR')
    args=p.parse_args()
    if args.formulas:
        print(json.dumps({'sources':SOURCES,'formulas':FORMULAS,'scenario_defaults':DEFAULTS},indent=2));return
    if args.write_examples:
        folder=Path(args.write_examples);folder.mkdir(parents=True,exist_ok=True)
        files={'dataset.json':EXAMPLE_DATASET,'hardware.json':EXAMPLE_HARDWARE,'runtime_assumptions.json':EXAMPLE_ASSUMPTIONS}
        for name in files:
            if (folder/name).exists(): p.error(f'Refusing to overwrite {folder/name}')
        for name,value in files.items(): (folder/name).write_text(json.dumps(value,indent=2)+'\n')
        print(f'Wrote illustrative configs to {folder}');return
    if not all((args.workload,args.dataset,args.hardware)): p.error('--workload, --dataset and --hardware are required')
    if Path(args.output).resolve() in {Path(x).resolve() for x in (args.workload,args.dataset,args.hardware,args.assumptions,args.deployment) if x}:
        p.error('Output must not overwrite an input file')
    try:
        r=estimate_runtime(read_json(args.workload),read_json(args.dataset),read_json(args.hardware),
            read_json(args.assumptions) if args.assumptions else None,
            read_json(args.deployment) if args.deployment else None,args.gpu_id)
        Path(args.output).write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    except (ValueError,TypeError,KeyError,OSError) as exc: p.exit(1,f'Runtime estimation failed: {exc}\n')
    print(f"{r['status']}: saved {args.output}")
    for label,result in r.get('scenarios',{}).items():
        print(f"{label}: training hours={result['training_hours']}; job GPU-hours={result['job_gpu_hours']}")
    if r.get('missing_information'): print('Missing: '+'; '.join(r['missing_information']))

if __name__=='__main__': main()
