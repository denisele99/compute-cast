"""GPT-2 MVP extraction from uploaded files, ZIP archives and the Hugging Face picker (active).

Nothing is imported or executed: Python is parsed with ``ast``, JSON with ``json``, YAML with
``yaml.safe_load``, CSV with ``csv``. Each file is classified by CONTENT into the Model,
Training and Data sections (one file may feed several). Settings that the v1 contract fixes
(e.g. gradient checkpointing, FP16, LoRA, dynamic padding) are reported as conflicts with the
requested value; they are never silently replaced. A conflict that has a supported
hypothetical scenario can be explicitly accepted by the user.

Values found in Python source are always "uncertain": static inspection is advisory and cannot
reliably resolve dynamic code (contract upload_guidance.source_code_policy).
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path

from . import dataset_sample
from .static_extract import Resolver, _call_name, _kw, _parse, load_config, merge_findings

ALLOWED_HF_MODELS = ('openai-community/gpt2', 'distilbert/distilgpt2')
SNAPSHOTS = Path(__file__).resolve().parents[2] / 'data' / 'hf_config_snapshots.json'
EXAMPLES = Path(__file__).resolve().parents[3] / 'examples' / 'gpt2_mvp'
EXAMPLE_SETS = {
    'gpt2_supported': {'label': 'GPT-2 small, supported settings',
                       'files': ['config.json', 'training_args.json', 'data_config.json', 'train.py']},
    'gpt2_conflicts': {'label': 'GPT-2 with unsupported settings (shows conflicts)',
                       'files': ['config.json', 'training_args_unsupported.json', 'data_config.json']},
}
MODEL_KEYS = {'n_layer': 'model.num_layers', 'n_embd': 'model.hidden_size', 'n_head': 'model.num_attention_heads',
              'n_inner': 'model.intermediate_size', 'vocab_size': 'model.vocab_size',
              'n_positions': 'model.max_position_embeddings', 'attn_pdrop': 'model.attention_dropout',
              'embd_pdrop': 'model.embedding_dropout', 'resid_pdrop': 'model.residual_dropout'}
TRAINING_HINT_KEYS = {'per_device_train_batch_size', 'num_train_epochs', 'max_steps', 'gradient_accumulation_steps',
                      'learning_rate', 'output_dir', 'microbatch_size', 'optimizer_updates', 'optim', 'bf16', 'fp16',
                      'train_micro_batch_size_per_gpu', 'zero_optimization', 'num_epochs', 'gradient_checkpointing'}
DATA_HINT_KEYS = {'max_length', 'max_seq_length', 'sequence_length', 'padding', 'block_size', 'packing',
                  'num_training_examples', 'num_examples', 'num_train_examples', 'splits', 'model_max_length',
                  'tokenizer_class', 'drop_last', 'pad_to_multiple_of'}
MAX_ZIP_MEMBERS = 50
MAX_ZIP_TOTAL = 30_000_000


class Collector:
    def __init__(self, file, source, status):
        self.file, self.source, self.status = file, source, status
        self.findings, self.conflicts, self.notes, self.sections = [], [], [], set()

    def add(self, field, value, line=None, message=None, status=None):
        self.sections.add(field.split('.')[0])
        self.findings.append({'field': field, 'value': value, 'status': status or self.status, 'source': self.source,
                              'file': self.file, 'line': line, 'message': message})

    def conflict(self, section, setting, requested, supported, hypothetical, line=None, field=None, why=''):
        cid = hashlib.sha1(f'{self.file}|{setting}|{requested}'.encode()).hexdigest()[:12]
        where = f'{self.file}' + (f':{line}' if line else '')
        message = (f'{where} requests {setting} = {requested!r}, which v1 does not support'
                   + (f' (v1: {supported}).' if supported is not None else '.') + (f' {why}' if why else '')
                   + (' Choose the supported hypothetical scenario to estimate anyway.' if hypothetical else ''))
        self.sections.add(section)
        self.conflicts.append({'id': cid, 'section': section, 'field': field, 'setting': setting,
                               'requested': requested, 'supported': supported, 'hypothetical': hypothetical,
                               'file': self.file, 'line': line, 'message': message})

    def result(self, kind):
        return {'kind': kind, 'file': self.file, 'sections': sorted(self.sections), 'findings': self.findings,
                'conflicts': self.conflicts, 'notes': self.notes}


# ------------------------------------------------------------------ mapping rules (configs & scripts)

def model_settings(c, items, line=None):
    """items: iterable of (key, value). Shared by config.json and GPT2Config(...) calls."""
    items = dict(items)
    mt = items.get('model_type')
    if mt is not None:
        if mt == 'gpt2':
            c.add('model.model_type', 'gpt2', line)
        else:
            c.conflict('model', 'model_type', mt, 'gpt2', None, line, 'model.model_type',
                       'Only the standard GPT-2 architecture is supported in the MVP.')
    for key, field in MODEL_KEYS.items():
        if key not in items:
            continue
        value = items[key]
        if key == 'n_inner' and value is None:
            c.notes.append(f'{c.file}: n_inner is null, so the MLP width resolves to 4 × n_embd.')
            continue
        c.add(field, value, line)
    rules = [('tie_word_embeddings', False, True, 'tied input/output embeddings'),
             ('scale_attn_weights', False, True, 'standard scaled attention'),
             ('add_cross_attention', True, False, 'no cross-attention'),
             ('reorder_and_upcast_attn', True, False, 'standard attention order'),
             ('scale_attn_by_inverse_layer_idx', True, False, 'standard attention scaling')]
    for key, bad, good, label in rules:
        if items.get(key) is bad:
            c.conflict('model', key, bad, good, f'Estimate as {label}', line)
    act = items.get('activation_function')
    if act is not None and act != 'gelu_new':
        c.conflict('model', 'activation_function', act, 'gelu_new', 'Estimate with gelu_new', line)
    attn = items.get('attn_implementation') or items.get('_attn_implementation')
    if attn not in (None, 'eager'):
        c.conflict('training', 'attention_backend', attn, 'math attention (FP32 intermediates)',
                   'Estimate with math attention', line)


def model_defaults(c, items):
    if not any(k in items for k in MODEL_KEYS) and 'model_type' not in items:
        return
    for key in ('attn_pdrop', 'embd_pdrop', 'resid_pdrop'):
        if key not in items:
            c.add(MODEL_KEYS[key], 0.1, message=f'{key} not set; GPT2Config default 0.1', status='uncertain')


def training_settings(c, items, line_of=lambda k: None):
    items = dict(items)
    for key in ('per_device_train_batch_size', 'microbatch_size', 'train_micro_batch_size_per_gpu', 'batch_size'):
        if items.get(key) is not None:
            c.add('training.microbatch_size', items[key], line_of(key), f'{key}')
            break
    if items.get('gradient_accumulation_steps') is not None:
        c.add('training.gradient_accumulation_steps', items['gradient_accumulation_steps'], line_of('gradient_accumulation_steps'))
    steps = items.get('optimizer_updates', items.get('max_steps'))
    epochs = items.get('num_train_epochs', items.get('num_epochs'))
    if isinstance(steps, int) and steps > 0:
        c.add('training.optimizer_updates', steps, line_of('max_steps'))
        c.add('training.duration_mode', 'optimizer_updates', line_of('max_steps'),
              'max_steps > 0 overrides epochs (Hugging Face Trainer semantics)')
    elif epochs is not None:
        if isinstance(epochs, (int, float)) and not isinstance(epochs, bool) and float(epochs).is_integer() and epochs > 0:
            c.add('training.num_epochs', int(epochs), line_of('num_train_epochs'))
            c.add('training.duration_mode', 'epochs', line_of('num_train_epochs'))
        else:
            c.conflict('training', 'num_epochs', epochs, 'whole epochs', None, line_of('num_train_epochs'),
                       'training.num_epochs', 'Fractional epochs are unsupported in v1; enter whole epochs or use optimizer updates.')
    fp16, bf16 = items.get('fp16'), items.get('bf16')
    if fp16 is True:
        c.conflict('training', 'precision_mode', 'fp16', 'bf16_autocast', 'Estimate with BF16 autocast', line_of('fp16'))
    elif bf16 is False and fp16 is False:
        c.conflict('training', 'precision_mode', 'fp32 (bf16=false, fp16=false)', 'bf16_autocast',
                   'Estimate with BF16 autocast', line_of('bf16'))
    if items.get('gradient_checkpointing') is True:
        c.conflict('training', 'activation_checkpointing', True, False, 'Estimate without activation checkpointing',
                   line_of('gradient_checkpointing'))
    optim = items.get('optim')
    if isinstance(optim, str) and optim not in ('adamw_torch', 'adamw_hf'):
        c.conflict('training', 'optimizer', optim, 'AdamW (unfused, no foreach)', 'Estimate with plain AdamW', line_of('optim'))
    if isinstance(items.get('optimizer'), dict):  # DeepSpeed
        kind = str(items['optimizer'].get('type', ''))
        if kind.lower() != 'adamw':
            c.conflict('training', 'optimizer', kind, 'AdamW', 'Estimate with plain AdamW')
    for key, setting, good, label in (('torch_compile', 'torch_compile', False, 'eager execution'),
                                      ('dataloader_drop_last', 'dataloader_drop_last', False, 'drop_last=false'),
                                      ('group_by_length', 'padding_policy', 'fixed_sequence_length', 'fixed-length batches'),
                                      ('tf32', 'tf32', False, 'TF32 disabled')):
        if items.get(key) is True:
            c.conflict('training' if key not in ('dataloader_drop_last', 'group_by_length') else 'data', setting, True,
                       good, f'Estimate with {label}', line_of(key))
    if items.get('deepspeed') or items.get('fsdp') or isinstance(items.get('zero_optimization'), dict):
        c.conflict('hardware', 'distributed_strategy', 'deepspeed/fsdp', 'none (one GPU)', 'Estimate on one GPU', line_of('deepspeed'))
    for key in ('world_size', 'nproc_per_node', 'n_gpu', 'gpu_count'):
        if isinstance(items.get(key), int) and items[key] > 1:
            c.conflict('hardware', 'gpu_count', items[key], 1, 'Estimate on one GPU', line_of(key))
    if any(k in items for k in ('lora_r', 'lora_alpha', 'peft_config', 'use_peft', 'lora_target_modules')):
        c.conflict('training', 'training_mode', 'LoRA/PEFT', 'full_all_parameters_trainable',
                   'Estimate full training of all parameters')
    if items.get('offload_optimizer') or items.get('offload_param'):
        c.conflict('training', 'offloading', True, False, 'Estimate without offload')


def data_settings(c, items, line=None):
    items = dict(items)
    length = next(((k, items[k]) for k in ('sequence_length', 'max_seq_length', 'max_length') if isinstance(items.get(k), int)), None)
    padding = items.get('padding')
    if length:
        key, value = length
        if padding == 'max_length' or key == 'sequence_length':
            c.add('data.sequence_length', value, line, f'{key} with fixed padding' if padding else key)
        elif padding in (True, 'longest', 'do_not_pad', False):
            c.add('data.sequence_length', value, line, f'{key}; padding={padding!r} is dynamic', status='uncertain')
            c.conflict('data', 'padding_policy', f'padding={padding!r}', f'pad to max_length={value}',
                       f'Estimate as fixed padding to {value} tokens', line)
        else:
            c.add('data.sequence_length', value, line, f'{key} found but padding not set: a truncation limit alone '
                  'does not fix the executed length; confirm padding="max_length"', status='uncertain')
    if isinstance(items.get('block_size'), int):
        c.add('data.sequence_length', items['block_size'], line, 'block_size (concatenated/grouped text)', status='uncertain')
        c.conflict('data', 'sequence_packing', 'grouped texts (block_size)', False,
                   f"Estimate as unpacked fixed-length sequences of {items['block_size']} tokens", line)
    if items.get('packing') is True:
        c.conflict('data', 'sequence_packing', True, False, 'Estimate without packing', line)
    if items.get('drop_last') is True:
        c.conflict('data', 'dataloader_drop_last', True, False, 'Estimate with drop_last=false', line)
    for key in ('num_training_examples', 'num_train_examples', 'num_examples'):
        if isinstance(items.get(key), int):
            c.add('data.num_training_examples', items[key], line, f'{key}: confirm this is the post-processed count')
            break
    splits = items.get('splits')
    if isinstance(splits, dict) and isinstance(splits.get('train'), dict) and isinstance(splits['train'].get('num_examples'), int):
        c.add('data.num_training_examples', splits['train']['num_examples'], line,
              'dataset_info train split: raw rows; post-processing may change the count', status='uncertain')
    if isinstance(items.get('model_max_length'), int):
        c.notes.append(f"{c.file}: tokenizer model_max_length={items['model_max_length']} is an upper limit, "
                       'not the executed sequence length.')


# ------------------------------------------------------------------ per file type

def from_config(path, name):
    cfg = load_config(path)
    c = Collector(name, 'upload', 'extracted')
    kinds = []
    if isinstance(cfg.get('model'), dict) and isinstance(cfg.get('training'), dict):  # estimator input format
        model_settings(c, cfg['model'].items())
        model_defaults(c, cfg['model'])
        training_settings(c, cfg['training'].items())
        data_settings(c, cfg['training'].items())
        if 'hardware' in cfg:
            c.notes.append(f'{name}: hardware section ignored; hardware comes from the GPU catalogue or custom entry.')
        return c.result('estimator_input')
    if 'model_type' in cfg or any(k in cfg for k in ('n_layer', 'n_embd', 'n_head')):
        model_settings(c, cfg.items())
        model_defaults(c, cfg)
        if 'auto_map' in cfg:
            c.notes.append(f'{name}: auto_map present (custom code); never executed.')
        kinds.append('model_config')
    if TRAINING_HINT_KEYS & set(cfg):
        training_settings(c, cfg.items())
        kinds.append('training_config')
    if DATA_HINT_KEYS & set(cfg):
        data_settings(c, cfg.items())
        kinds.append('data_config')
    if not kinds:
        c.notes.append(f'{name}: no recognised model, training or data settings.')
    return c.result('+'.join(kinds) or 'unrecognised')


def from_python(path, name):
    tree = _parse(path)
    res = Resolver(tree)
    c = Collector(name, 'static_extraction', 'uncertain')

    def literal_kwargs(call):
        out, lines = {}, {}
        for kw in call.keywords:
            if kw.arg:
                value, status, _ = res.resolve(kw.value)
                if status != 'unresolved':
                    out[kw.arg] = value
                else:
                    c.notes.append(f'{name}:{call.lineno}: {kw.arg}={ast.unparse(kw.value)} is not statically resolvable.')
                lines[kw.arg] = call.lineno
        return out, lines

    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        full = _call_name(call)
        short = full.split('.')[-1]
        line = call.lineno
        if short == 'GPT2Config':
            items, _ = literal_kwargs(call)
            model_settings(c, {'model_type': 'gpt2', **items}.items(), line)
        elif short.endswith('Config') and short not in ('TrainingArguments', 'BitsAndBytesConfig', 'GenerationConfig') \
                and short[:-6].lower() not in ('', 'auto') and short not in ('LoraConfig', 'PeftConfig'):
            c.conflict('model', 'model_type', short, 'GPT2Config', None, line, 'model.model_type',
                       'Only the standard GPT-2 architecture is supported in the MVP.')
        elif short in ('LoraConfig', 'PeftConfig', 'get_peft_model', 'prepare_model_for_kbit_training'):
            c.conflict('training', 'training_mode', short, 'full_all_parameters_trainable',
                       'Estimate full training of all parameters', line)
        elif short in ('from_pretrained',) and call.args:
            value, status, _ = res.resolve(call.args[0])
            owner = full.split('.')[0]
            if isinstance(value, str) and ('Tokenizer' not in owner):
                if value in ALLOWED_HF_MODELS or value in ('gpt2', 'distilgpt2'):
                    c.notes.append(f'{name}:{line}: loads {value!r}; select it in the Hugging Face picker to use its config.')
                elif 'Tokenizer' not in full:
                    c.conflict('model', 'pretrained_model', value, ' or '.join(ALLOWED_HF_MODELS), None, line,
                               'model.model_type', 'Only the listed GPT-2 checkpoints are supported in the MVP.')
            for kw in ('attn_implementation', 'torch_dtype'):
                node = _kw(call, kw)
                if node is not None:
                    v2, st, _ = res.resolve(node)
                    text = v2 if isinstance(v2, str) else ast.unparse(node)
                    if kw == 'attn_implementation' and text != 'eager':
                        c.conflict('training', 'attention_backend', text, 'math attention (FP32 intermediates)',
                                   'Estimate with math attention', line)
                    if kw == 'torch_dtype' and any(t in str(text) for t in ('float16', 'bfloat16', 'half')):
                        c.conflict('training', 'parameter_dtype', text, 'float32', 'Estimate with FP32 parameters', line)
        elif short in ('TrainingArguments', 'Seq2SeqTrainingArguments', 'SFTConfig'):
            items, lines = literal_kwargs(call)
            training_settings(c, items.items(), lambda k, ln=line: ln)
            data_settings(c, items.items(), line)
        elif short == 'DataLoader':
            items, _ = literal_kwargs(call)
            if 'batch_size' in items:
                c.add('training.microbatch_size', items['batch_size'], line, 'DataLoader batch_size')
            if items.get('drop_last') is True:
                c.conflict('data', 'dataloader_drop_last', True, False, 'Estimate with drop_last=false', line)
        elif 'max_length' in {k.arg for k in call.keywords} or 'padding' in {k.arg for k in call.keywords}:
            items, _ = literal_kwargs(call)
            data_settings(c, items.items(), line)
        elif short in ('AdamW', 'Adam', 'SGD', 'Adagrad', 'RMSprop', 'Adafactor', 'Lion') and (
                full.startswith(('torch.optim', 'optim')) or short == full):
            if short != 'AdamW':
                c.conflict('training', 'optimizer', short, 'AdamW', 'Estimate with plain AdamW', line)
            for flag in ('fused', 'foreach', 'amsgrad'):
                node = _kw(call, flag)
                if node is not None and res.resolve(node)[0] is True:
                    c.conflict('training', f'optimizer_{flag}', True, False, f'Estimate with {flag}=False', line)
        elif short == 'gradient_checkpointing_enable' or full in ('torch.utils.checkpoint.checkpoint', 'checkpoint.checkpoint'):
            c.conflict('training', 'activation_checkpointing', True, False, 'Estimate without activation checkpointing', line)
        elif short == 'compile' and full.startswith('torch'):
            c.conflict('training', 'torch_compile', True, False, 'Estimate with eager execution', line)
        elif short in ('DistributedDataParallel', 'FullyShardedDataParallel', 'FSDP', 'DDP') or full == 'deepspeed.initialize':
            c.conflict('hardware', 'distributed_strategy', short, 'none (one GPU)', 'Estimate on one GPU', line)
        elif short in ('autocast',):
            node = _kw(call, 'dtype')
            text = ast.unparse(node) if node is not None else 'float16 (CUDA default)'
            if 'bfloat16' not in text:
                c.conflict('training', 'precision_mode', text, 'bf16_autocast', 'Estimate with BF16 autocast', line)
        elif short == 'GradScaler':
            c.conflict('training', 'precision_mode', 'fp16 (GradScaler)', 'bf16_autocast', 'Estimate with BF16 autocast', line)
    for cand in _assignments(tree, res):
        key, value, line = cand
        if key in ('epochs', 'num_epochs', 'num_train_epochs', 'max_steps', 'batch_size', 'per_device_train_batch_size',
                   'gradient_accumulation_steps', 'max_length', 'max_seq_length', 'sequence_length', 'block_size'):
            if key in ('max_length', 'max_seq_length', 'sequence_length', 'block_size'):
                data_settings(c, {key: value}.items(), line)
            else:
                training_settings(c, {key: value}.items(), lambda k, ln=line: ln)
    if c.findings or c.conflicts:
        c.notes.append(f'{name}: values from Python source are advisory; review them.')
    return c.result('python_script')


def _assignments(tree, res):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value, status, _ = res.resolve(node.value)
            if status != 'unresolved' and isinstance(value, (int, float, bool, str)):
                yield node.targets[0].id, value, node.lineno


def extract(path, name):
    """Classify and extract one stored file. Returns a result dict or {'error': ...}."""
    suffix = Path(name).suffix.lower()
    if suffix == '.py':
        return from_python(path, name)
    if suffix in ('.csv', '.tsv', '.jsonl'):
        result = dataset_sample.analyze(path, name)
        return {'kind': 'dataset_sample', 'file': name, 'sections': ['data'], 'findings': [], 'conflicts': [],
                'notes': result['notes'], 'stats': result['stats']}
    if suffix == '.json':
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        if isinstance(raw, list):
            result = dataset_sample.analyze(path, name)
            return {'kind': 'dataset_sample', 'file': name, 'sections': ['data'], 'findings': [], 'conflicts': [],
                    'notes': result['notes'], 'stats': result['stats']}
    return from_config(path, name)


def huggingface_result(model_id, config, origin):
    c = Collector(f'{model_id}/config.json', 'huggingface', 'extracted')
    model_settings(c, config.items())
    model_defaults(c, config)
    c.notes.append(f'{model_id}: config.json from {origin}. Weights are never downloaded; no remote code runs.')
    out = c.result('huggingface_config')
    out['model_id'] = model_id
    return out


def snapshot(model_id):
    return json.loads(SNAPSHOTS.read_text(encoding='utf-8'))['models'][model_id]


def expand_zip(data, max_member_bytes):
    """Safe in-memory ZIP expansion: flat basenames only, size/count caps, no nested archives."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError('Not a valid ZIP archive.') from exc
    members, skipped, total = [], [], 0
    for info in zf.infolist():
        name = info.filename.replace('\\', '/').split('/')[-1]
        if info.is_dir() or not name or name.startswith('.') or '__MACOSX' in info.filename:
            continue
        if len(members) >= MAX_ZIP_MEMBERS:
            raise ValueError(f'ZIP has more than {MAX_ZIP_MEMBERS} files.')
        if name.lower().endswith('.zip'):
            skipped.append(f'{info.filename} (nested ZIP not allowed)')
            continue
        limit = max_member_bytes(name)
        if limit is None:
            skipped.append(f'{info.filename} (unsupported type)')
            continue
        if info.file_size > limit:
            skipped.append(f'{info.filename} (larger than {limit // 1_000_000} MB)')
            continue
        with zf.open(info) as handle:
            content = handle.read(limit + 1)
        if len(content) > limit:
            skipped.append(f'{info.filename} (larger than declared)')
            continue
        total += len(content)
        if total > MAX_ZIP_TOTAL:
            raise ValueError('ZIP expands to more than 30 MB.')
        if any(m[0] == name for m in members):
            skipped.append(f'{info.filename} (duplicate file name)')
            continue
        members.append((name, content))
    return members, skipped


def merge(results):
    """One field map, de-duplicated conflicts, model sources and notes for the whole analysis."""
    usable = [r for r in results if 'findings' in r]
    hf = next((r for r in usable if r['kind'] == 'huggingface_config'), None)
    if hf is not None:
        # A Hugging Face model plus your own training script: the checkpoint defines the model, so
        # model settings found in scripts are ignored. Uploaded model *config files* stay a second
        # source and are reported (one model source only).
        trimmed = []
        for r in usable:
            dropped = [f for f in r['findings'] if r['kind'] == 'python_script' and f['field'].startswith('model.')]
            if dropped:
                r = {**r, 'findings': [f for f in r['findings'] if f not in dropped],
                     'notes': r['notes'] + [f"{r['file']}: model settings ignored; the Hugging Face model "
                                            f"{hf.get('model_id')} defines the model."]}
            trimmed.append(r)
        usable = trimmed
    fields = merge_findings(usable)
    conflicts, seen = [], set()
    for r in usable:
        for c in r['conflicts']:
            if c['id'] not in seen:
                seen.add(c['id'])
                conflicts.append(c)
    sources = []
    for r in usable:
        if any(f['field'].startswith('model.') for f in r['findings']):
            sources.append({'kind': 'huggingface' if r['kind'] == 'huggingface_config' else 'upload',
                            'name': r.get('model_id') or r['file']})
    return {'fields': fields, 'conflicts': conflicts, 'model_sources': sources,
            'notes': [n for r in usable for n in r['notes']],
            'dataset_samples': [r['stats'] for r in usable if r.get('kind') == 'dataset_sample']}


__all__ = ['extract', 'merge', 'expand_zip', 'huggingface_result', 'snapshot', 'ALLOWED_HF_MODELS', 'EXAMPLE_SETS',
           'EXAMPLES']
