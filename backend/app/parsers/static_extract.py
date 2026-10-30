"""PARTLY LEGACY. Active in the GPT-2 MVP: Resolver, _call_name, _kw, _parse, load_config and
merge_findings (used by gpt2_extract.py). Legacy, NOT USED BY THE MVP: scan_training_script,
extract_model_script, extract_training_script, map_model_config, map_training_config, extract_file.

Static extraction from uploaded model/training files. Nothing is imported or executed.

Python files are parsed with ``ast`` only; JSON uses ``json`` and YAML uses
``yaml.safe_load``. Hugging Face configs are read as plain JSON; ``auto_map`` /
``trust_remote_code`` entries are reported and never executed.

Every finding is ``{field, value, status, source, file, line, message}`` where status is
``extracted`` (explicit literal/config value), ``uncertain`` (plausible but needs review,
e.g. conflicting or indirect) or ``unresolved`` (present but not statically resolvable;
value is None). Static analysis cannot resolve arbitrary Python, so anything not provably
literal is flagged rather than guessed.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from ..services.estimators.profiles import UNSUPPORTED_MODEL_TYPES

MAX_SOURCE_BYTES = 1_000_000

SETTING_NAMES = {
    'batch_size', 'batch_size_per_gpu', 'per_device_train_batch_size', 'train_batch_size',
    'epochs', 'num_train_epochs', 'num_epochs', 'max_steps', 'num_examples',
    'sequence_length', 'max_length', 'max_seq_length', 'seq_len', 'block_size',
    'gradient_accumulation_steps', 'gradient_checkpointing', 'fp16', 'bf16', 'precision',
    'num_workers', 'pin_memory', 'drop_last', 'world_size', 'gpu_count', 'nproc_per_node',
    'distributed_strategy', 'attention_backend', 'attn_implementation', 'optim', 'optimizer',
    'lr', 'momentum', 'amsgrad', 'centered', 'foreach', 'fused',
}

TRAINING_KEY_MAP = {
    'batch_size': 'training.batch_size_per_gpu', 'batch_size_per_gpu': 'training.batch_size_per_gpu',
    'per_device_train_batch_size': 'training.batch_size_per_gpu',
    'train_micro_batch_size_per_gpu': 'training.batch_size_per_gpu',
    'train_batch_size': 'training.batch_size_per_gpu',
    'epochs': 'training.epochs', 'num_train_epochs': 'training.epochs', 'num_epochs': 'training.epochs',
    'max_steps': 'training.max_steps',
    'gradient_accumulation_steps': 'training.gradient_accumulation_steps',
    'sequence_length': 'training.sequence_length', 'max_length': 'training.sequence_length',
    'max_seq_length': 'training.sequence_length', 'seq_len': 'training.sequence_length',
    'block_size': 'training.sequence_length',
    'gradient_checkpointing': 'training.gradient_checkpointing',
    'drop_last': 'training.drop_last', 'momentum': 'training.momentum',
    'amsgrad': 'training.amsgrad', 'centered': 'training.centered',
    'world_size': 'training.gpu_count_used', 'gpu_count': 'training.gpu_count_used',
    'nproc_per_node': 'training.gpu_count_used',
    'num_examples': 'dataset.num_train_examples', 'num_train_examples': 'dataset.num_train_examples',
    'optimizer_state_dtype': 'training.optimizer_state_dtype',
    'gradient_dtype': 'training.gradient_dtype', 'parameter_dtype': 'training.parameter_dtype',
    'master_weights': 'training.master_weights', 'task': 'training.task',
}
DTYPE_ALIASES = {
    'fp32': 'fp32', 'float32': 'fp32', 'float': 'fp32', 'torch.float32': 'fp32', 'torch.float': 'fp32',
    'fp16': 'fp16', 'float16': 'fp16', 'half': 'fp16', 'torch.float16': 'fp16', 'torch.half': 'fp16',
    'bf16': 'bf16', 'bfloat16': 'bf16', 'torch.bfloat16': 'bf16',
}
OPTIMIZER_NAMES = {'adamw': 'AdamW', 'adam': 'Adam', 'sgd': 'SGD', 'adagrad': 'Adagrad',
                   'rmsprop': 'RMSprop', 'adamw_torch': 'AdamW', 'adamw_torch_fused': 'AdamW',
                   'adamw_hf': 'AdamW'}
STRATEGY_NAMES = {'single': 'single', 'none': 'single', 'ddp': 'ddp', 'zero1': 'zero1',
                  'zero2': 'zero2', 'zero3': 'zero3', 'fsdp': 'fsdp_full_shard',
                  'fsdp_full_shard': 'fsdp_full_shard', 'full_shard': 'fsdp_full_shard',
                  'fsdp_shard_grad_op': 'fsdp_shard_grad_op', 'shard_grad_op': 'fsdp_shard_grad_op'}
NON_ARCH_CONFIGS = {'BitsAndBytesConfig', 'GenerationConfig', 'TrainingConfig', 'DataConfig',
                    'DeepSpeedConfig', 'SFTConfig', 'GRPOConfig', 'DPOConfig', 'AcceleratorConfig'}
ATTENTION_NAMES = {'eager': 'eager', 'standard': 'standard', 'sdpa': 'sdpa',
                   'flash_attention_2': 'flash_attention_2', 'flash': 'flash_attention_2'}


def finding(field, value, status, file, line=None, message=None, source='static_extraction'):
    return {'field': field, 'value': value, 'status': status, 'source': source,
            'file': file, 'line': line, 'message': message}


def _parse(path):
    data = Path(path).read_bytes()
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError('Python file exceeds 1 MB; static analysis skipped.')
    return ast.parse(data.decode('utf-8'), filename=str(path))


def _call_name(node):
    """Dotted name of a call target, e.g. 'torch.optim.AdamW'."""
    parts, cur = [], node.func if isinstance(node, ast.Call) else node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return '.'.join(reversed(parts))


def scan_training_script(path):
    """AST-only literal candidates for known settings (kept for collect_workload CLI)."""
    tree = _parse(path)
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
                if kw.arg:
                    add(kw.arg, kw.value, node.lineno)
    return {'candidates': candidates,
            'limitations': 'Candidates may be unused, overridden, or in inactive branches. '
                           'External configs, CLI overrides and dynamic values need explicit input.'}


class Resolver:
    """Resolve names to literals only when a single literal assignment exists."""

    def __init__(self, tree):
        self.assignments = {}
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else [])
            for target in targets:
                if isinstance(target, ast.Name):
                    self.assignments.setdefault(target.id, []).append(node.value)

    def resolve(self, node, depth=0):
        """Return (value, status, note). status: extracted | uncertain | unresolved."""
        if depth > 5:
            return None, 'unresolved', 'expression too indirect'
        try:
            return ast.literal_eval(node), 'extracted', None
        except (ValueError, TypeError, SyntaxError):
            pass
        if isinstance(node, ast.Name):
            values = self.assignments.get(node.id, [])
            if not values:
                return None, 'unresolved', f'{node.id} is not assigned a literal in this file'
            resolved = [self.resolve(v, depth + 1) for v in values]
            literal = {json.dumps(r[0]) for r in resolved if r[1] != 'unresolved'}
            if len(values) == 1 and resolved[0][1] != 'unresolved':
                return resolved[0][0], resolved[0][1], f'resolved from {node.id}'
            if len(literal) == 1 and all(r[1] != 'unresolved' for r in resolved):
                return resolved[0][0], 'uncertain', f'{node.id} is assigned several times'
            return None, 'unresolved', f'{node.id} has multiple or non-literal assignments'
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            container, status, _ = self.resolve(node.value, depth + 1)
            key = node.slice
            try:
                key = ast.literal_eval(key)
            except (ValueError, TypeError, SyntaxError):
                return None, 'unresolved', 'non-literal subscript'
            if isinstance(container, dict) and key in container:
                return container[key], 'uncertain' if status == 'uncertain' else 'extracted', \
                    f'resolved from {node.value.id}[{key!r}]'
            return None, 'unresolved', f'{ast.unparse(node)} not resolvable'
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub, ast.FloorDiv)):
            left, ls, _ = self.resolve(node.left, depth + 1)
            right, rs, _ = self.resolve(node.right, depth + 1)
            if ls != 'unresolved' and rs != 'unresolved' and all(
                    isinstance(x, int) and not isinstance(x, bool) for x in (left, right)):
                ops = {ast.Mult: lambda a, b: a * b, ast.Add: lambda a, b: a + b,
                       ast.Sub: lambda a, b: a - b, ast.FloorDiv: lambda a, b: a // b if b else None}
                value = ops[type(node.op)](left, right)
                if value is not None:
                    return value, 'uncertain' if 'uncertain' in (ls, rs) else 'extracted', \
                        f'computed from {ast.unparse(node)}'
        if isinstance(node, ast.Attribute):
            text = ast.unparse(node)
            if text in DTYPE_ALIASES:
                return text, 'extracted', None
        return None, 'unresolved', f'{ast.unparse(node)} is not statically resolvable'


def _kw(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _dtype_from(node, resolver):
    value, status, note = resolver.resolve(node)
    text = value if isinstance(value, str) else ast.unparse(node)
    return DTYPE_ALIASES.get(str(text).lower()), status, note


def _linear_dims(call, resolver, file):
    args = list(call.args)
    dims = []
    for i, key in enumerate(('in_features', 'out_features')):
        node = args[i] if i < len(args) else _kw(call, key)
        if node is None:
            dims.append((None, 'unresolved', f'{key} missing'))
        else:
            dims.append(resolver.resolve(node))
    bias_node = args[2] if len(args) > 2 else _kw(call, 'bias')
    bias = (True, 'extracted', 'PyTorch default bias=True') if bias_node is None else resolver.resolve(bias_node)
    return dims, bias


def _short(name):
    return name.split('.')[-1]


def extract_model_script(path, display_name=None):
    """Findings for a model .py file. Recognizes the two supported profiles only."""
    file = display_name or Path(path).name
    tree = _parse(path)
    res = Resolver(tree)
    findings, notes, unsupported = [], [], []
    sequentials = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _short(_call_name(n)) == 'Sequential']
    linears = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _short(_call_name(n)) == 'Linear']
    relus = [n for n in ast.walk(tree) if (isinstance(n, ast.Call) and _short(_call_name(n)) in ('ReLU', 'relu'))]
    profile_hint = None

    def add_mlp(first, second, status, why):
        (i, h), b1 = _linear_dims(first, res, file)
        (h2, o), b2 = _linear_dims(second, res, file)
        for field, (value, st, note) in (('model.input_size', i), ('model.hidden_size', h),
                                         ('model.output_size', o)):
            st = 'uncertain' if status == 'uncertain' and st == 'extracted' else st
            findings.append(finding(field, value, st, file, first.lineno,
                                    '; '.join(x for x in (why, note) if x) or None))
        if h[1] != 'unresolved' and h2[1] != 'unresolved' and h[0] != h2[0]:
            findings.append(finding('model.hidden_size', None, 'unresolved', file, second.lineno,
                                    f'Linear layer widths disagree: first out={h[0]}, second in={h2[0]}.'))
        bias_values = {json.dumps(b1[0]), json.dumps(b2[0])}
        if len(bias_values) > 1:
            findings.append(finding('model.linear_bias', None, 'unresolved', file, first.lineno,
                                    'Layers use different bias settings; the profile needs one setting.'))
        else:
            st = 'unresolved' if 'unresolved' in (b1[1], b2[1]) else (
                'uncertain' if status == 'uncertain' else b1[1])
            findings.append(finding('model.linear_bias', b1[0] if st != 'unresolved' else None, st,
                                    file, first.lineno, b1[2]))

    for seq in sequentials:
        kinds = [_short(_call_name(a)) if isinstance(a, ast.Call) else None for a in seq.args]
        if kinds == ['Linear', 'ReLU', 'Linear']:
            profile_hint = 'mlp_regression'
            add_mlp(seq.args[0], seq.args[2], 'extracted', 'nn.Sequential(Linear, ReLU, Linear)')
            break
        if 'Linear' in kinds:
            notes.append(f'{file}:{seq.lineno}: Sequential({", ".join(k or "?" for k in kinds)}) '
                         'does not match Linear → ReLU → Linear.')
    if profile_hint is None and len(linears) == 2 and relus:
        profile_hint = 'mlp_regression'
        add_mlp(linears[0], linears[1], 'uncertain',
                'two Linear layers and a ReLU found in a custom module; forward() order not verified')
    elif profile_hint is None and linears and not sequentials:
        notes.append(f'{file}: {len(linears)} nn.Linear layer(s) found; the layer graph could not be '
                     'matched to a supported profile statically.')

    hf_keys = {'n_embd': 'model.hidden_size', 'hidden_size': 'model.hidden_size',
               'd_model': 'model.hidden_size', 'n_layer': 'model.num_hidden_layers',
               'num_hidden_layers': 'model.num_hidden_layers', 'num_layers': 'model.num_hidden_layers',
               'n_head': 'model.num_attention_heads', 'num_attention_heads': 'model.num_attention_heads',
               'nhead': 'model.num_attention_heads', 'num_heads': 'model.num_attention_heads',
               'vocab_size': 'model.vocab_size', 'n_positions': 'model.max_position_embeddings',
               'max_position_embeddings': 'model.max_position_embeddings',
               'n_inner': 'model.intermediate_size', 'intermediate_size': 'model.intermediate_size',
               'dim_feedforward': 'model.intermediate_size',
               'num_key_value_heads': 'model.num_key_value_heads',
               'tie_word_embeddings': 'model.tie_word_embeddings'}
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        name = _call_name(call)
        short = _short(name)
        if short in ('LoraConfig', 'PeftConfig', 'PrefixTuningConfig', 'IA3Config'):
            unsupported.append({'field': 'training.training_method', 'message':
                                f'{file}:{call.lineno}: {short} — PEFT/LoRA partial training is not supported.'})
            continue
        if short in NON_ARCH_CONFIGS:
            if short == 'BitsAndBytesConfig':
                unsupported.append({'field': 'training.parameter_dtype', 'message':
                                    f'{file}:{call.lineno}: quantized weights are not supported.'})
            continue
        if short.endswith('Config') and short != 'Config' or short in (
                'TransformerEncoderLayer', 'TransformerDecoderLayer', 'MultiheadAttention'):
            model_type = short[:-6].lower() if short.endswith('Config') else None
            if model_type and model_type in UNSUPPORTED_MODEL_TYPES:
                unsupported.append({'field': 'model.profile', 'message':
                                    f'{file}:{call.lineno}: {short} — {UNSUPPORTED_MODEL_TYPES[model_type]}'})
                continue
            profile_hint = profile_hint or 'transformer_lm_classic'
            torch_layer = not short.endswith('Config')
            why = (f'{short}(...) — PyTorch built-in layer; parameter layout differs from the '
                   'GPT-2 profile' if torch_layer else f'{short}(...) keyword')
            for kw in call.keywords:
                if kw.arg in hf_keys:
                    value, st, note = res.resolve(kw.value)
                    st = 'uncertain' if torch_layer and st == 'extracted' else st
                    findings.append(finding(hf_keys[kw.arg], value, st, file, call.lineno,
                                            '; '.join(x for x in (why, note) if x)))
                elif kw.arg == 'attn_implementation':
                    value, st, note = res.resolve(kw.value)
                    findings.append(finding('training.attention_backend', ATTENTION_NAMES.get(str(value)),
                                            st, file, call.lineno, note))
            if model_type:
                findings.append(finding('model.model_type', model_type, 'uncertain', file, call.lineno,
                                        f'from {short}'))
        if short == 'from_pretrained' or short == 'from_config':
            repo = call.args[0] if call.args else _kw(call, 'pretrained_model_name_or_path')
            if repo is not None:
                value, st, note = res.resolve(repo)
                if isinstance(value, str):
                    findings.append(finding('model.hf_model_id', value, 'uncertain', file, call.lineno,
                                            'from_pretrained target; fetch its config.json to fill '
                                            'architecture fields (weights are not downloaded)'))
                    profile_hint = profile_hint or 'transformer_lm_classic'
            if _kw(call, 'trust_remote_code') is not None:
                notes.append(f'{file}:{call.lineno}: trust_remote_code is set; remote code is never '
                             'executed here, so custom architectures must be described manually.')
            for key in ('torch_dtype', 'dtype'):
                if _kw(call, key) is not None:
                    dtype, st, note = _dtype_from(_kw(call, key), res)
                    findings.append(finding('training.parameter_dtype', dtype,
                                            'uncertain' if dtype else 'unresolved', file, call.lineno,
                                            'load dtype; confirm parameters stay in this dtype during training'))
            attn = _kw(call, 'attn_implementation')
            if attn is not None:
                value, st, note = res.resolve(attn)
                findings.append(finding('training.attention_backend', ATTENTION_NAMES.get(str(value)),
                                        st, file, call.lineno, note))
        findings.extend(_dtype_call_findings(call, res, file))
    return {'kind': 'model_script', 'file': file, 'findings': findings, 'notes': notes,
            'unsupported': unsupported, 'profile_hint': profile_hint}


def _dtype_call_findings(call, res, file):
    name = _call_name(call)
    short = _short(name)
    out = []
    if short in ('half', 'bfloat16') and not call.args and isinstance(call.func, ast.Attribute):
        out.append(finding('training.parameter_dtype', 'fp16' if short == 'half' else 'bf16',
                           'uncertain', file, call.lineno,
                           f'.{short}() call; confirm it applies to the model parameters'))
    if short == 'to' and isinstance(call.func, ast.Attribute):
        node = _kw(call, 'dtype') or next((a for a in call.args if ast.unparse(a).startswith('torch.')
                                           and ast.unparse(a) in DTYPE_ALIASES), None)
        if node is not None:
            dtype, st, note = _dtype_from(node, res)
            if dtype:
                out.append(finding('training.parameter_dtype', dtype, 'uncertain', file, call.lineno,
                                   '.to(dtype) call; confirm it applies to the model parameters'))
    if name in ('torch.set_default_dtype',) and call.args:
        dtype, st, note = _dtype_from(call.args[0], res)
        out.append(finding('training.parameter_dtype', dtype, 'uncertain' if dtype else 'unresolved',
                           file, call.lineno, 'torch.set_default_dtype'))
    return out


def extract_training_script(path, display_name=None):
    """Findings for a training .py file."""
    file = display_name or Path(path).name
    tree = _parse(path)
    res = Resolver(tree)
    findings, notes, unsupported = [], [], []
    scan = scan_training_script(path)
    for cand in scan['candidates']:
        field = TRAINING_KEY_MAP.get(cand['name'])
        if cand['name'] in ('bf16', 'fp16') and cand['literal'] and cand['value'] is True:
            findings.append(finding('training.precision', cand['name'], 'extracted', file, cand['line'],
                                    f"{cand['name']}=True"))
            continue
        if cand['name'] in ('optim', 'optimizer') and cand['literal'] and isinstance(cand['value'], str):
            opt = OPTIMIZER_NAMES.get(cand['value'].lower())
            findings.append(finding('training.optimizer', opt, 'extracted' if opt else 'unresolved',
                                    file, cand['line'], None if opt else f"unsupported optimizer {cand['value']!r}"))
            continue
        if cand['name'] in ('attn_implementation', 'attention_backend'):
            value = ATTENTION_NAMES.get(str(cand['value'])) if cand['literal'] else None
            findings.append(finding('training.attention_backend', value,
                                    'extracted' if value else 'unresolved', file, cand['line'],
                                    None if value else cand['expression']))
            continue
        if cand['name'] == 'distributed_strategy' and cand['literal']:
            findings.append(finding('training.distributed_strategy', STRATEGY_NAMES.get(str(cand['value'])),
                                    'uncertain', file, cand['line'], None))
            continue
        if field is None:
            continue
        if cand['literal']:
            value = cand['value']
            if field in ('training.optimizer_state_dtype', 'training.gradient_dtype', 'training.parameter_dtype'):
                value = DTYPE_ALIASES.get(str(value).lower())
            findings.append(finding(field, value, 'extracted', file, cand['line'], None))
        else:
            value, status, note = res.resolve(ast.parse(cand['expression'], mode='eval').body)
            findings.append(finding(field, value if status != 'unresolved' else None, status, file,
                                    cand['line'], note or f"{cand['expression']} is not a literal"))
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        name = _call_name(call)
        short = _short(name)
        lower = short.lower()
        if lower in OPTIMIZER_NAMES and (name.startswith(('torch.optim', 'optim')) or short in (
                'AdamW', 'Adam', 'SGD', 'Adagrad', 'RMSprop')):
            findings.append(finding('training.optimizer', OPTIMIZER_NAMES[lower], 'extracted', file,
                                    call.lineno, f'{name}(...)'))
            for key in ('momentum', 'amsgrad', 'centered', 'foreach'):
                node = _kw(call, key)
                if node is not None:
                    value, st, note = res.resolve(node)
                    field = 'training.' + key
                    if key == 'foreach':
                        value = {True: 'true', False: 'false'}.get(value) if st != 'unresolved' else None
                    findings.append(finding(field, value, st, file, call.lineno, note))
        elif short in ('MSELoss', 'mse_loss'):
            findings.append(finding('training.loss', 'mse', 'extracted', file, call.lineno, f'{short}'))
            findings.append(finding('training.task', 'regression', 'uncertain', file, call.lineno,
                                    'inferred from MSE loss'))
        elif short in ('CrossEntropyLoss', 'cross_entropy'):
            findings.append(finding('training.loss', 'cross_entropy', 'extracted', file, call.lineno, short))
        elif short in ('autocast',):
            node = _kw(call, 'dtype')
            if node is not None:
                dtype, st, note = _dtype_from(node, res)
                findings.append(finding('training.precision', dtype, 'extracted' if dtype else 'unresolved',
                                        file, call.lineno, 'autocast dtype'))
            else:
                findings.append(finding('training.precision', 'fp16', 'uncertain', file, call.lineno,
                                        'autocast without dtype uses fp16 on CUDA by default; confirm'))
        elif short == 'GradScaler':
            findings.append(finding('training.precision', 'fp16', 'uncertain', file, call.lineno,
                                    'GradScaler usually indicates fp16 autocast'))
        elif short == 'gradient_checkpointing_enable':
            findings.append(finding('training.gradient_checkpointing', True, 'extracted', file, call.lineno,
                                    'gradient_checkpointing_enable()'))
        elif name in ('torch.utils.checkpoint.checkpoint', 'checkpoint.checkpoint') or short == 'checkpoint_sequential':
            findings.append(finding('training.gradient_checkpointing', True, 'uncertain', file, call.lineno,
                                    f'{name}(...) call'))
        elif short in ('DistributedDataParallel', 'DDP'):
            findings.append(finding('training.distributed_strategy', 'ddp', 'uncertain', file, call.lineno,
                                    f'{short}(...) wrapper'))
        elif short in ('FullyShardedDataParallel', 'FSDP', 'fully_shard'):
            strategy = 'fsdp_full_shard'
            node = _kw(call, 'sharding_strategy')
            if node is not None and 'SHARD_GRAD_OP' in ast.unparse(node):
                strategy = 'fsdp_shard_grad_op'
            elif node is not None and 'HYBRID' in ast.unparse(node):
                unsupported.append({'field': 'training.distributed_strategy', 'message':
                                    f'{file}:{call.lineno}: FSDP hybrid sharding is not modelled.'})
            findings.append(finding('training.distributed_strategy', strategy, 'uncertain', file,
                                    call.lineno, f'{short}(...) wrapper'))
        elif name in ('deepspeed.initialize',):
            findings.append(finding('training.distributed_strategy', None, 'unresolved', file, call.lineno,
                                    'DeepSpeed: upload the DeepSpeed JSON config to read the ZeRO stage'))
        elif short == 'DataLoader':
            for key, field in (('batch_size', 'training.batch_size_per_gpu'), ('drop_last', 'training.drop_last')):
                node = _kw(call, key)
                if node is not None:
                    value, st, note = res.resolve(node)
                    findings.append(finding(field, value if st != 'unresolved' else None, st, file,
                                            call.lineno, note or 'DataLoader argument'))
        elif short == 'scaled_dot_product_attention':
            findings.append(finding('training.attention_backend', 'sdpa', 'uncertain', file, call.lineno,
                                    'F.scaled_dot_product_attention call'))
        elif short == 'compile' and name.startswith('torch'):
            notes.append(f'{file}:{call.lineno}: torch.compile changes kernels and memory; the '
                         'profiles model eager execution.')
        findings.extend(_dtype_call_findings(call, res, file))
    return {'kind': 'training_script', 'file': file, 'findings': findings, 'notes': notes,
            'unsupported': unsupported, 'profile_hint': None}


def load_config(path):
    """Load a JSON/YAML object safely (no Python object construction)."""
    path = Path(path)
    data = path.read_bytes()
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError('Config exceeds 1 MB.')
    text = data.decode('utf-8')
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        config = yaml.safe_load(text)
    else:
        def reject(value):
            raise ValueError(f'Non-finite JSON constant: {value}')
        config = json.loads(text, parse_constant=reject)
    if not isinstance(config, dict):
        raise ValueError(f'{path.name}: config must be an object/mapping.')
    json.dumps(config, allow_nan=False, default=str)
    return config


def map_model_config(config, file, source='upload'):
    """Map a Hugging Face style config.json (or a plain model config) to canonical fields."""
    findings, notes, unsupported = [], [], []

    def add(field, value, status='extracted', message=None):
        findings.append(finding(field, value, status, file, None, message, source=source))

    model_type = config.get('model_type')
    if 'auto_map' in config:
        notes.append(f'{file}: auto_map present — this model ships custom code. Remote code is never '
                     'executed; architecture fields are read as plain JSON and must be verified.')
    if isinstance(model_type, str):
        add('model.model_type', model_type)
        if model_type in UNSUPPORTED_MODEL_TYPES:
            unsupported.append({'field': 'model.profile', 'message':
                                f'{model_type}: {UNSUPPORTED_MODEL_TYPES[model_type]}'})
    if any(k in config for k in ('num_experts', 'num_local_experts', 'moe_num_experts', 'n_routed_experts')):
        unsupported.append({'field': 'model.profile', 'message': 'Mixture-of-experts layers are not supported.'})
    if config.get('is_encoder_decoder'):
        unsupported.append({'field': 'model.profile', 'message': 'Encoder-decoder models are not supported.'})
    mlp_keys = ('input_size', 'output_size')
    if all(k in config for k in mlp_keys) and 'vocab_size' not in config:
        for key in ('input_size', 'hidden_size', 'output_size'):
            if key in config:
                add('model.' + key, config[key])
        if 'bias' in config:
            add('model.linear_bias', config['bias'])
        add('model.profile', 'mlp_regression', 'uncertain', 'input_size/output_size config keys')
        return {'kind': 'model_config', 'file': file, 'findings': findings, 'notes': notes,
                'unsupported': unsupported, 'profile_hint': 'mlp_regression'}
    keys = {'hidden_size': ('n_embd', 'hidden_size', 'd_model'),
            'num_hidden_layers': ('n_layer', 'num_hidden_layers', 'num_layers'),
            'num_attention_heads': ('n_head', 'num_attention_heads'),
            'vocab_size': ('vocab_size',),
            'max_position_embeddings': ('n_positions', 'max_position_embeddings', 'n_ctx'),
            'intermediate_size': ('n_inner', 'intermediate_size', 'ffn_dim'),
            'num_key_value_heads': ('num_key_value_heads', 'n_head_kv')}
    found_tx = False
    for field, names in keys.items():
        for name in names:
            if config.get(name) is not None:
                add('model.' + field, config[name], message=f'config key {name}')
                found_tx = True
                break
    if config.get('n_inner', 0) is None and 'n_inner' in config:
        notes.append(f'{file}: n_inner is null — Transformers uses 4 × n_embd.')
    if not found_tx:
        return {'kind': 'model_config', 'file': file, 'findings': findings, 'notes': notes +
                [f'{file}: no recognized architecture keys.'], 'unsupported': unsupported, 'profile_hint': None}
    add('model.profile', 'transformer_lm_classic', 'uncertain', 'transformer architecture keys')
    if 'tie_word_embeddings' in config:
        add('model.tie_word_embeddings', bool(config['tie_word_embeddings']))
    elif model_type:
        add('model.tie_word_embeddings', True, 'uncertain',
            'not in config.json; Transformers PretrainedConfig default is true')
    act = config.get('activation_function', config.get('hidden_act'))
    if isinstance(act, str):
        low = act.lower()
        if low.startswith('gelu'):
            add('model.mlp_activation', 'gelu', message=f'{act}')
        elif low == 'relu':
            add('model.mlp_activation', 'relu')
        elif low in ('silu', 'swiglu', 'swish', 'geglu'):
            unsupported.append({'field': 'model.mlp_activation', 'message':
                                f'{act} activation usually means a gated MLP, which is not supported.'})
    rope = config.get('position_embedding_type') == 'rotary' or 'rope_theta' in config or \
        'rotary_dim' in config or 'rotary_pct' in config
    if rope:
        add('model.position_embedding', 'none', 'uncertain', 'rotary positions have no parameters')
    elif any(config.get(n) is not None for n in ('n_positions', 'max_position_embeddings')):
        add('model.position_embedding', 'learned', 'uncertain',
            'max positions present; confirm positions are a learned embedding table')
    for key in ('bias', 'use_bias', 'attention_bias'):
        if isinstance(config.get(key), bool):
            add('model.linear_bias', config[key], message=f'config key {key}')
            break
    else:
        if model_type in ('gpt2', 'openai-gpt', 'bert', 'opt', 'gpt_neox'):
            add('model.linear_bias', True, 'uncertain', f'{model_type} uses biased Linear layers')
    if model_type in ('gpt2',):
        add('model.layer_norm_bias', True, 'uncertain', 'GPT-2 LayerNorm has bias')
        add('model.lm_head_bias', False, 'uncertain', 'GPT-2 LM head has no bias')
    if model_type == 'bert':
        notes.append(f'{file}: BERT adds token-type embeddings, a pooler and an MLM transform head '
                     'that the classic layout does not count; parameter count will be slightly low.')
    elif model_type and model_type not in ('gpt2', 'openai-gpt') and model_type not in UNSUPPORTED_MODEL_TYPES:
        notes.append(f'{file}: model_type {model_type!r} has not been verified against the GPT-2 '
                     'parameter layout; review the parameter count.')
    if config.get('torch_dtype') or config.get('dtype'):
        dtype = DTYPE_ALIASES.get(str(config.get('torch_dtype') or config.get('dtype')).lower())
        add('training.parameter_dtype', dtype, 'uncertain',
            'checkpoint torch_dtype — confirm the dtype parameters are stored in during training')
    attn = config.get('_attn_implementation') or config.get('attn_implementation')
    if attn:
        add('training.attention_backend', ATTENTION_NAMES.get(attn), 'uncertain', 'config attention setting')
    return {'kind': 'model_config', 'file': file, 'findings': findings, 'notes': notes,
            'unsupported': unsupported, 'profile_hint': 'transformer_lm_classic'}


def map_training_config(config, file, source='upload'):
    """Map training configs (this repo's format, HF TrainingArguments, DeepSpeed JSON)."""
    findings, notes, unsupported = [], [], []

    def add(field, value, status='extracted', message=None):
        findings.append(finding(field, value, status, file, None, message, source=source))

    for key, value in config.items():
        if isinstance(value, dict):
            continue
        field = TRAINING_KEY_MAP.get(key)
        if key in ('bf16', 'fp16') and value is True:
            add('training.precision', key, message=f'{key}=true')
        elif key == 'precision':
            add('training.precision', DTYPE_ALIASES.get(str(value).lower()))
        elif key in ('optim', 'optimizer') and isinstance(value, str):
            opt = OPTIMIZER_NAMES.get(value.lower())
            add('training.optimizer', opt, 'extracted' if opt else 'unresolved',
                None if opt else f'unsupported optimizer {value!r}')
        elif key in ('attention_backend', 'attn_implementation'):
            add('training.attention_backend', ATTENTION_NAMES.get(str(value)))
        elif key == 'distributed_strategy':
            add('training.distributed_strategy', STRATEGY_NAMES.get(str(value).lower()))
        elif key == 'gradient_checkpointing':
            add('training.gradient_checkpointing', bool(value))
        elif key == 'max_steps' and isinstance(value, int) and value > 0:
            add('training.max_steps', value)
            add('training.schedule_mode', 'max_steps', 'uncertain', 'max_steps > 0 overrides epochs in HF Trainer')
        elif key == 'max_steps':
            continue
        elif field in ('training.optimizer_state_dtype', 'training.gradient_dtype', 'training.parameter_dtype'):
            add(field, DTYPE_ALIASES.get(str(value).lower()))
        elif field:
            add(field, value)
    opt_opts = config.get('optimizer_options')
    if isinstance(opt_opts, dict):
        for key in ('momentum', 'amsgrad', 'centered'):
            if key in opt_opts:
                add('training.' + key, opt_opts[key])
    zero = config.get('zero_optimization')
    if isinstance(zero, dict):
        stage = zero.get('stage')
        if stage in (0, 1, 2, 3):
            add('training.distributed_strategy', {0: 'ddp', 1: 'zero1', 2: 'zero2', 3: 'zero3'}[stage],
                'uncertain', 'DeepSpeed zero_optimization.stage; confirm the GPU count')
        for key, field in (('offload_optimizer', 'training.offload_optimizer'),
                           ('offload_param', 'training.offload_parameters')):
            if isinstance(zero.get(key), dict) and zero[key].get('device') in ('cpu', 'nvme'):
                add(field, zero[key]['device'], 'uncertain', f'DeepSpeed {key}.device')
    for key in ('bf16', 'fp16'):
        if isinstance(config.get(key), dict) and config[key].get('enabled') is True:
            add('training.precision', key, message=f'DeepSpeed {key}.enabled')
    if isinstance(config.get('optimizer'), dict):
        opt = OPTIMIZER_NAMES.get(str(config['optimizer'].get('type', '')).lower())
        add('training.optimizer', opt, 'extracted' if opt else 'unresolved', 'DeepSpeed optimizer.type')
    if config.get('torch_compile'):
        notes.append(f'{file}: torch_compile is enabled; the profiles model eager execution.')
    if 'model_type' in config or 'n_embd' in config or 'num_hidden_layers' in config:
        sub = map_model_config(config, file, source)
        findings += sub['findings']
        notes += sub['notes']
        unsupported += sub['unsupported']
    return {'kind': 'training_config', 'file': file, 'findings': findings, 'notes': notes,
            'unsupported': unsupported, 'profile_hint': None}


def extract_file(path, kind, display_name=None):
    """Dispatch by upload kind ('model' or 'training') and extension."""
    suffix = Path(display_name or path).suffix.lower()
    file = display_name or Path(path).name
    if suffix == '.py':
        return extract_model_script(path, file) if kind == 'model' else extract_training_script(path, file)
    config = load_config(path if suffix in ('.json', '.yaml', '.yml') else path)
    return map_model_config(config, file) if kind == 'model' else map_training_config(config, file)


def merge_findings(results):
    """Combine findings across files: one entry per field, conflicts flagged for review."""
    by_field = {}
    for result in results:
        for f in result['findings']:
            by_field.setdefault(f['field'], []).append(f)
    merged = {}
    for field, items in by_field.items():
        resolved = [i for i in items if i['status'] != 'unresolved' and i['value'] is not None]
        explicit = [i for i in resolved if i['status'] == 'extracted']
        pool = explicit or resolved
        values = {json.dumps(i['value']) for i in pool}
        locations = [f"{i['file']}" + (f":{i['line']}" if i['line'] else '') for i in items]
        if not pool:
            first = items[0]
            merged[field] = {**first, 'value': None, 'status': 'unresolved',
                             'message': first['message'] or 'Found but not statically resolvable.',
                             'locations': locations}
        elif len(values) == 1:
            best = pool[0]
            status = best['status']
            message = best['message']
            if any(i['status'] == 'unresolved' for i in items):
                status = 'uncertain'
                message = (message + '; ' if message else '') + 'other occurrences were not resolvable'
            merged[field] = {**best, 'status': status, 'message': message, 'locations': locations}
        else:
            options = sorted(values)
            merged[field] = {**pool[0], 'status': 'uncertain',
                             'message': 'Conflicting values found: ' + ', '.join(
                                 f"{json.loads(v)!r}" for v in options) + '. Review which one applies.',
                             'candidates': [json.loads(v) for v in options], 'locations': locations}
    return merged
