"""GPT-2 MVP input schema, validation and estimator-object construction (active).

Field definitions follow backend/data/mvp_feature_contract.json:
  1. direct inputs / file extractions -> FIELDS (editable: True, False or 'custom_only')
  2. visible fixed assumptions        -> fixed_assumptions() (locked, one option each)
  3. internal calculations/lookups     -> computed by estimate_gpt2.py, never user input
Uploaded files, the Hugging Face picker, the hardware catalogue, defaults and manual entry
all fill the same flat value map; provenance is kept alongside it by the client.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from ..services.estimators.estimate_gpt2 import DEFAULT_POLICY

CONTRACT = Path(__file__).resolve().parents[2] / 'data' / 'mvp_feature_contract.json'
SECTIONS = ('model', 'training', 'data', 'hardware')


def F(key, label, kind, section, *, editable=True, contract=None, unit=None, help=None, default=None,
      options=None, minimum=None, maximum=None, advanced=False):
    return {'key': key, 'label': label, 'type': kind, 'section': section, 'editable': editable,
            'contract_feature': contract, 'unit': unit, 'help': help, 'default': default,
            'options': options, 'minimum': minimum, 'maximum': maximum, 'advanced': advanced}


FIELDS = [
    F('model.model_type', 'Model type', 'select', 'model', editable=False, contract='model_type',
      options=['gpt2'], default='gpt2', help='MVP supports standard GPT-2 only (config field model_type).'),
    F('model.custom_config', 'Hypothetical custom configuration', 'bool', 'model', default=False,
      help='Turn on to edit dimensions of a selected/uploaded checkpoint. Results are then labelled hypothetical.'),
    F('model.num_layers', 'Layers (n_layer)', 'int', 'model', editable='custom_only', contract='num_layers', minimum=1),
    F('model.hidden_size', 'Hidden size (n_embd)', 'int', 'model', editable='custom_only', contract='hidden_size', minimum=1),
    F('model.num_attention_heads', 'Attention heads (n_head)', 'int', 'model', editable='custom_only',
      contract='num_attention_heads', minimum=1, help='Must divide the hidden size.'),
    F('model.intermediate_size', 'MLP width (n_inner)', 'int', 'model', editable='custom_only',
      contract='intermediate_size', minimum=1, help='Empty means 4 × hidden size (GPT-2 default).'),
    F('model.vocab_size', 'Vocabulary (vocab_size)', 'int', 'model', editable='custom_only', contract='vocab_size', minimum=1),
    F('model.max_position_embeddings', 'Max positions (n_positions)', 'int', 'model', editable='custom_only',
      contract='max_position_embeddings', minimum=1),
    F('model.attention_dropout', 'Attention dropout (attn_pdrop)', 'float', 'model', editable='custom_only',
      contract='attention_dropout', minimum=0, maximum=0.999999, default=0.1, help='GPT2Config default 0.1.'),
    F('model.embedding_dropout', 'Embedding dropout (embd_pdrop)', 'float', 'model', editable='custom_only',
      contract='embedding_dropout', minimum=0, maximum=0.999999, default=0.1, help='GPT2Config default 0.1.'),
    F('model.residual_dropout', 'Residual dropout (resid_pdrop)', 'float', 'model', editable='custom_only',
      contract='residual_dropout', minimum=0, maximum=0.999999, default=0.1, help='GPT2Config default 0.1.'),
    F('training.microbatch_size', 'Microbatch size per GPU', 'int', 'training', contract='microbatch_size', minimum=1,
      help='Examples per forward/backward pass (per_device_train_batch_size).'),
    F('training.gradient_accumulation_steps', 'Gradient accumulation steps', 'int', 'training',
      contract='gradient_accumulation_steps', minimum=1, default=1),
    F('training.duration_mode', 'Training length defined by', 'select', 'training', default='epochs',
      options=['epochs', 'optimizer_updates'], help='Choose optimizer updates OR epochs + training examples.'),
    F('training.num_epochs', 'Epochs', 'int', 'training', contract='num_epochs', minimum=1,
      help='Whole epochs only in v1.'),
    F('training.optimizer_updates', 'Optimizer updates (max_steps)', 'int', 'training', contract='optimizer_updates',
      minimum=1),
    F('data.num_training_examples', 'Training examples (post-processed)', 'int', 'data',
      contract='num_training_examples', minimum=1, help='Final sequences per epoch, not raw documents.'),
    F('data.sequence_length', 'Sequence length (padded)', 'int', 'data', contract='sequence_length', minimum=1,
      help='Fixed executed length (padding to max_length); must not exceed max positions.'),
    F('hardware.source', 'Hardware source', 'select', 'hardware', options=['nebius', 'catalog', 'custom']),
    F('hardware.gpu_name', 'GPU name', 'text', 'hardware', help='Filled when you pick a GPU above; type a name for a custom GPU.'),
    F('hardware.gpu_id', 'GPU id', 'text', 'hardware', contract='gpu_id', advanced=True,
      help='Internal id of the selected catalogue/Nebius entry, or "custom".'),
    F('hardware.catalog_gpu_id', 'Spec catalogue id', 'text', 'hardware', advanced=True),
    F('hardware.vram_gib', 'VRAM per GPU', 'float', 'hardware', unit='GiB', minimum=0),
    F('hardware.usable_vram_gib', 'Usable VRAM limit', 'float', 'hardware', unit='GiB', minimum=0,
      help='Defaults to 100% of VRAM: the estimate already adds a runtime baseline and margin.'),
    F('hardware.bf16_supported', 'Native BF16 support', 'bool', 'hardware', advanced=True,
      help='Optional. Filled from the GPU catalogue; needed only for the runtime estimate.'),
    F('hardware.bf16_dense_tflops', 'Peak BF16 dense', 'float', 'hardware', unit='TFLOPS', minimum=0, advanced=True,
      help='Optional. Filled from the GPU catalogue; needed only for the runtime estimate.'),
    F('hardware.fp32_tflops', 'Peak FP32', 'float', 'hardware', unit='TFLOPS', minimum=0, advanced=True,
      help='Optional. Filled from the GPU catalogue; needed only for the runtime estimate.'),
    F('hardware.memory_bandwidth_gbps', 'Memory bandwidth', 'float', 'hardware', unit='GB/s', minimum=0, advanced=True,
      help='Optional. Filled from the GPU catalogue; needed only for the runtime estimate.'),
    F('hardware.price_per_hour', 'Price', 'float', 'hardware', unit='USD', minimum=0, default=0,
      help='Price per GPU-hour. 0 until you pick a priced Nebius GPU or type your own price.'),
    F('hardware.price_unit', 'Price unit', 'select', 'hardware', options=['gpu_hour', 'instance_hour'], default='gpu_hour',
      help='One GPU in v1, so a 1-GPU instance-hour equals a GPU-hour.'),
    F('hardware.price_region', 'Price region', 'text', 'hardware', advanced=True),
    F('hardware.price_note', 'Price note', 'text', 'hardware', advanced=True),
]
FIELD_INDEX = {f['key']: f for f in FIELDS}
MODEL_DIMS = [f['key'] for f in FIELDS if f['editable'] == 'custom_only']
TO_GPT2_CONFIG = {'model.num_layers': 'n_layer', 'model.hidden_size': 'n_embd', 'model.num_attention_heads': 'n_head',
                  'model.intermediate_size': 'n_inner', 'model.vocab_size': 'vocab_size',
                  'model.max_position_embeddings': 'n_positions', 'model.attention_dropout': 'attn_pdrop',
                  'model.embedding_dropout': 'embd_pdrop', 'model.residual_dropout': 'resid_pdrop'}


@lru_cache
def contract():
    return json.loads(CONTRACT.read_text(encoding='utf-8'))


def fixed_assumptions():
    out = []
    for item in contract()['2_visible_fixed_assumptions']:
        value = item['value']
        out.append({'feature': item['feature'], 'section': item['section'],
                    'value': value if isinstance(value, (str, bool, int, float)) else json.dumps(value),
                    'raw_value': value, 'later_extension': item.get('later_extension'),
                    'note': item.get('implementation_status')})
    return out


def schema():
    return {'schema_version': 'mvp-1', 'framework': 'pytorch', 'architecture': 'gpt2', 'fields': FIELDS,
            'sections': list(SECTIONS), 'fixed_assumptions': fixed_assumptions(),
            'internal_calculations': contract()['3_internal_calculations_and_lookups'],
            'estimator_policy': DEFAULT_POLICY, 'upload_guidance': contract()['upload_guidance']}


def _coerce(field, value):
    if value is None or value == '':
        return None, None
    kind = field['type']
    if kind in ('int', 'float'):
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None, 'Enter a number.'
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None, 'Enter a number.'
        if kind == 'int':
            if int(value) != value:
                return None, 'Enter a whole number.'
            value = int(value)
        if field['minimum'] is not None and value < field['minimum']:
            return None, f"Must be at least {field['minimum']}."
        if field['maximum'] is not None and value > field['maximum']:
            return None, 'Must be below 1.'
    elif kind == 'bool' and not isinstance(value, bool):
        return None, 'Choose yes or no.'
    elif kind == 'select' and value not in field['options']:
        return None, f"Choose one of: {', '.join(field['options'])}."
    elif kind == 'text' and (not isinstance(value, str) or len(value) > 300):
        return None, 'Enter text up to 300 characters.'
    return value, None


def validate(values, provenance=None, extraction=None, acknowledged=None):
    """Validate the flat value map against the MVP contract.

    extraction: merged file/Hugging Face extraction for this analysis ({fields, conflicts, model_sources}).
    acknowledged: conflict ids for which the user explicitly chose the supported hypothetical scenario.
    """
    values, provenance, extraction = dict(values or {}), provenance or {}, extraction or {}
    acknowledged = set(acknowledged or [])
    errors, warnings, v = [], [], {}
    for key in sorted(set(values) - set(FIELD_INDEX)):
        warnings.append({'field': key, 'message': 'Unknown field ignored.'})
    for key, field in FIELD_INDEX.items():
        value, err = _coerce(field, values.get(key))
        if err:
            errors.append({'field': key, 'message': err})
        v[key] = value
    if v['model.model_type'] is None:
        v['model.model_type'] = 'gpt2'
    mode = v['training.duration_mode'] or 'epochs'
    required = {}

    def need(key, calc):
        required.setdefault(key, [])
        if calc not in required[key]:
            required[key].append(calc)

    for key in ('model.num_layers', 'model.hidden_size', 'model.num_attention_heads', 'model.vocab_size',
                'model.max_position_embeddings', 'model.attention_dropout', 'model.embedding_dropout',
                'model.residual_dropout', 'training.microbatch_size', 'training.gradient_accumulation_steps',
                'data.sequence_length', 'hardware.vram_gib', 'hardware.usable_vram_gib'):
        need(key, 'memory')
    if mode == 'epochs':
        need('training.num_epochs', 'runtime')
        need('data.num_training_examples', 'runtime')
    else:
        need('training.optimizer_updates', 'runtime')
    for key in ('hardware.bf16_supported', 'hardware.bf16_dense_tflops', 'hardware.fp32_tflops',
                'hardware.memory_bandwidth_gbps'):
        need(key, 'runtime')
    need('hardware.price_per_hour', 'cost')
    need('hardware.price_unit', 'cost')
    missing = [{'field': k, 'label': FIELD_INDEX[k]['label'], 'required_for': c}
               for k, c in required.items() if v.get(k) is None]

    # Contract constraints.
    if v['model.model_type'] != 'gpt2':
        errors.append({'field': 'model.model_type', 'message': 'Only standard GPT-2 is supported.'})
    H, A = v['model.hidden_size'], v['model.num_attention_heads']
    if H and A and H % A:
        errors.append({'field': 'model.num_attention_heads', 'message': f'{A} heads do not divide hidden size {H}.'})
    S, C = v['data.sequence_length'], v['model.max_position_embeddings']
    if S and C and S > C:
        errors.append({'field': 'data.sequence_length', 'message': f'Sequence length {S} exceeds max positions {C}.'})
    total, usable = v['hardware.vram_gib'], v['hardware.usable_vram_gib']
    if total is not None and total <= 0:
        errors.append({'field': 'hardware.vram_gib', 'message': 'Must be greater than 0.'})
    if total and usable is not None and usable > total:
        errors.append({'field': 'hardware.usable_vram_gib', 'message': 'Cannot exceed VRAM per GPU.'})
    if v['hardware.bf16_supported'] is False:
        errors.append({'field': 'hardware.bf16_supported', 'message':
                       'The v1 profile is BF16 autocast; this GPU cannot run it. Choose a BF16-capable GPU.'})
    if mode == 'epochs' and v['training.optimizer_updates'] is not None:
        warnings.append({'field': 'training.optimizer_updates', 'message': 'Ignored: duration is defined by epochs.'})
    if mode == 'optimizer_updates' and v['training.num_epochs'] is not None:
        warnings.append({'field': 'training.num_epochs', 'message': 'Ignored: duration is defined by optimizer updates.'})
    B, N = v['training.microbatch_size'], v['data.num_training_examples']
    if mode == 'epochs' and B and N and B > N:
        warnings.append({'field': 'training.microbatch_size', 'message':
                         f'Microbatch {B} exceeds {N} examples: each epoch is one partial batch (drop_last=false).'})

    # Editability: checkpoint dimensions change only in an explicit hypothetical custom config.
    extracted = extraction.get('fields', {})
    custom = bool(v['model.custom_config'])
    hypothetical = []
    for key in MODEL_DIMS:
        found = extracted.get(key)
        if found is None or found.get('value') is None:
            continue
        if v[key] != found['value']:
            if custom:
                hypothetical.append(f"{FIELD_INDEX[key]['label']}: {v[key]} instead of {found['value']} "
                                    f"({found.get('file')})")
            else:
                errors.append({'field': key, 'message': f"Differs from {found.get('file')} ({found['value']}). "
                               'Turn on "Hypothetical custom configuration" to change checkpoint dimensions.'})
    sources = extraction.get('model_sources', [])
    if len({s['kind'] for s in sources}) > 1:
        errors.append({'field': 'model.model_type', 'message': 'Use a Hugging Face model OR uploaded model files, '
                       'not both. Remove one of: ' + ', '.join(s['name'] for s in sources) + '.'})
    conflicts = []
    for c in extraction.get('conflicts', []):
        c = {**c, 'acknowledged': c['id'] in acknowledged and c.get('hypothetical') is not None}
        conflicts.append(c)
        if c['acknowledged']:
            hypothetical.append(f"{c['setting']}: estimated as {c['supported']} instead of {c['requested']} ({c['file']})")
        else:
            errors.append({'field': c.get('field') or 'conflicts', 'message': c['message']})
    for key, meta in provenance.items():
        if isinstance(meta, dict) and meta.get('status') in ('uncertain', 'unresolved') and not meta.get('reviewed') \
                and key in FIELD_INDEX:
            warnings.append({'field': key, 'message': meta.get('message') or 'Extracted value is uncertain; review it.'})

    suggestions = {}
    if total:
        suggestions['hardware.usable_vram_gib'] = {'value': total, 'reason': 'Default: 100% of VRAM (margin is in the estimate).'}
    blocked = bool(errors) or any('memory' in m['required_for'] for m in missing)
    can = {'memory': not blocked}
    can['runtime'] = can['memory'] and not any('runtime' in m['required_for'] for m in missing)
    can['cost'] = can['runtime'] and not any('cost' in m['required_for'] for m in missing)
    return {'values': v, 'errors': errors, 'warnings': warnings, 'missing': missing, 'required': required,
            'can_estimate': can, 'suggestions': suggestions, 'conflicts': conflicts,
            'hypothetical': hypothetical, 'unsupported': [],
            'blocking_fields': sorted({e['field'] for e in errors}
                                      | {m['field'] for m in missing if 'memory' in m['required_for']})}


def gpt2_model(v):
    model = {'model_type': 'gpt2', 'activation_function': 'gelu_new', 'tie_word_embeddings': True,
             'scale_attn_weights': True}
    for key, name in TO_GPT2_CONFIG.items():
        model[name] = v.get(key)
    return model


def gpt2_training(v):
    t = {'microbatch_size': v['training.microbatch_size'], 'sequence_length': v['data.sequence_length'],
         'gradient_accumulation_steps': v['training.gradient_accumulation_steps']}
    if (v.get('training.duration_mode') or 'epochs') == 'epochs':
        t.update(num_epochs=v.get('training.num_epochs'), num_training_examples=v.get('data.num_training_examples'))
    else:
        t['optimizer_updates'] = v.get('training.optimizer_updates')
    return t


def gpt2_hardware(v):
    return {'gpu_id': v.get('hardware.gpu_id'), 'label': v.get('hardware.gpu_name'),
            'bf16_supported': v.get('hardware.bf16_supported'), 'vram_gib': v.get('hardware.vram_gib'),
            'usable_vram_gib': v.get('hardware.usable_vram_gib'),
            'bf16_dense_tflops': v.get('hardware.bf16_dense_tflops'), 'fp32_tflops': v.get('hardware.fp32_tflops'),
            'memory_bandwidth_gbps': v.get('hardware.memory_bandwidth_gbps'), 'gpu_count': 1}


def pricing(v):
    return {'price_per_hour': v.get('hardware.price_per_hour'), 'unit': v.get('hardware.price_unit'),
            'gpus_per_instance': 1, 'currency': 'USD', 'region': v.get('hardware.price_region'),
            'note': v.get('hardware.price_note'), 'gpus_allocated': 1, 'gpus_used': 1}
