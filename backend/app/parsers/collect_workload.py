"""LEGACY - NOT USED BY THE GPT-2 MVP. Retained for future profiles/extensions; not imported by
active routes. Active MVP path: app/parsers/gpt2_inputs.py, app/parsers/gpt2_extract.py,
app/services/estimators/estimate_gpt2.py and gpt2_predict.py. Still covered by tests.

Metadata-only workload.json collector for the legacy profiles (CLI still works).

Metadata-only PyTorch workload collector: explicit settings -> workload.json.

Python 3.10+, stdlib (+ optional PyYAML). No GPU, no PyTorch, no model code required.

Usage (run from backend/):
  python -m app.parsers.collect_workload --profile mlp_regression \
      --model-config model.json --training-config train.json --output workload.json
  python -m app.parsers.collect_workload --profile transformer_lm_classic \
      --model-config config.json --training-config train.json \
      --model-script model.py --training-script train.py --output workload.json

Supported profiles (see app/services/estimators/profiles.py):
  mlp_regression          Linear -> ReLU -> Linear, MSE regression, full training.
  transformer_lm_classic  GPT-2-style MHA, 4H non-gated MLP, 16-bit activations,
                          eager/standard attention, full training.
Gradient checkpointing must be disabled for both.

Uploaded/model/training code is NEVER imported, executed or instantiated. Optional
--model-script/--training-script files are scanned with ``ast`` only and their findings
are reported as unverified hints; they are not promoted to settings. Configs are read
with json / yaml.safe_load. Hugging Face remote code is never executed.

Parameter counts, tensor shapes and saved-activation sizes are DERIVED from the explicit
architecture settings using the documented profile formulas, including explicit bias and
weight-tying settings. Parameter storage dtype must be given explicitly; it is never
inferred from compute precision. Missing values stay missing and are listed in
``missing_training_fields``; no value is invented.

The collector produces facts and derived metadata, not a peak-memory/runtime prediction;
the estimators in app/services/estimators consume its output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from ..services.estimators import profiles as P
    from .static_extract import extract_model_script, extract_training_script, load_config, scan_training_script
except ImportError:  # pragma: no cover - executed as a plain script
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.services.estimators import profiles as P
    from app.parsers.static_extract import (extract_model_script, extract_training_script,
                                            load_config, scan_training_script)

__all__ = ['collect_from_metadata', 'input_specs', 'scan_training_script', 'load_config', 'save_report']

SCHEMA_VERSION = 3
DTYPE_BYTES = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'float32': 4, 'float16': 2, 'bfloat16': 2,
               'int64': 8, 'int32': 4}
REQUIRED = {
    'common': ('batch_size_per_gpu', 'gradient_accumulation_steps', 'precision', 'parameter_dtype',
               'gradient_dtype', 'optimizer', 'master_weights', 'gradient_checkpointing',
               'gpu_count', 'distributed_strategy', 'task', 'input_dtype', 'target_dtype'),
    'transformer_lm_classic': ('sequence_length', 'attention_backend'),
    'mlp_regression': (),
}
MODEL_REQUIRED = {
    'mlp_regression': ('input_size', 'hidden_size', 'output_size', 'linear_bias'),
    'transformer_lm_classic': ('vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
                               'position_embedding', 'tie_word_embeddings', 'linear_bias',
                               'layer_norm_bias', 'lm_head_bias'),
}


def input_specs(profile, model, training):
    """Batch-inclusive input/target tensor specs implied by the profile."""
    B = training.get('batch_size_per_gpu')
    if not isinstance(B, int):
        return None
    if profile == 'mlp_regression':
        if not all(isinstance(model.get(k), int) for k in ('input_size', 'output_size')):
            return None
        return [{'name': 'features', 'shape': [B, model['input_size']],
                 'dtype': training.get('input_dtype')},
                {'name': 'targets', 'shape': [B, model['output_size']],
                 'dtype': training.get('target_dtype')}]
    S = training.get('sequence_length')
    if not isinstance(S, int):
        return None
    return [{'name': 'input_ids', 'shape': [B, S], 'dtype': training.get('input_dtype')},
            {'name': 'labels', 'shape': [B, S], 'dtype': training.get('target_dtype')}]


def collect_from_metadata(profile, model, training, *, script_findings=None, provenance=None):
    """Build workload.json from explicit, validated settings. Never executes code."""
    if profile not in P.PROFILES:
        raise ValueError(f'Unsupported profile {profile!r}; choose one of {sorted(P.PROFILES)}.')
    model, training = dict(model), dict(training)
    errors, unsupported = P.profile_issues(profile, model, training)
    missing = [k for k in REQUIRED['common'] + REQUIRED[profile] if training.get(k) is None]
    if training.get('max_steps') is None and (training.get('epochs') is None or training.get('num_examples') is None):
        missing.append('epochs_and_num_examples_or_max_steps')
    if training.get('optimizer') in ('Adam', 'AdamW', 'Adagrad', 'RMSprop') or (
            training.get('optimizer') == 'SGD' and (training.get('optimizer_options') or {}).get('momentum')):
        if training.get('optimizer_state_dtype') is None:
            missing.append('optimizer_state_dtype')
    model_missing = [k for k in MODEL_REQUIRED[profile] if model.get(k) is None]
    if profile == 'transformer_lm_classic' and model.get('position_embedding') == 'learned' \
            and model.get('max_position_embeddings') is None:
        model_missing.append('max_position_embeddings')
    report = {
        'schema_version': SCHEMA_VERSION,
        'collection_mode': 'metadata_only',
        'framework': P.FRAMEWORK,
        'profile': profile,
        'profile_summary': P.PROFILES[profile]['summary'],
        'status': 'complete',
        'missing_training_fields': missing,
        'missing_model_fields': model_missing,
        'profile_errors': errors,
        'unsupported': unsupported,
        'training': training,
        'formulas': {k: P.FORMULAS[k] for k in (
            ('mlp_parameters', 'mlp_activations') if profile == 'mlp_regression'
            else ('transformer_parameters', 'transformer_activations'))},
        'script_scan': script_findings,
        'provenance': provenance or {},
        'limitations': [
            'Derived from explicit settings and profile formulas; no model code was executed.',
            'Static extraction cannot resolve arbitrary Python; unresolved values must be entered.',
            'Parameter storage dtype is explicit input; it is not inferred from compute precision.',
            'All parameters are assumed trainable (full training) and receive dense gradients.',
            'No buffers are assumed for the profile layouts (GPT-2 attention masks are computed, '
            'not stored); custom buffers need a different workflow.',
        ],
    }
    if model_missing or errors or unsupported or training.get('parameter_dtype') is None:
        report['status'] = 'unsupported' if unsupported else 'incomplete'
        report['model'] = {'class': None, 'provided_config': model}
        return report
    inventory, summary = P.parameter_layout(profile, model, training['parameter_dtype'])
    specs = input_specs(profile, model, training)
    training['input_specs'] = specs
    report['input_specs'] = specs
    if profile == 'mlp_regression':
        report['model'] = {'class': 'Sequential', 'module_types': {'Sequential': 1, 'Linear': 2, 'ReLU': 1},
                           'architecture': {}}
    else:
        arch = {k: model[k] for k in ('num_hidden_layers', 'hidden_size', 'num_attention_heads',
                                      'vocab_size', 'max_position_embeddings') if model.get(k) is not None}
        arch['intermediate_size'] = model.get('intermediate_size') or 4 * model['hidden_size']
        arch['num_key_value_heads'] = model.get('num_key_value_heads') or model['num_attention_heads']
        report['model'] = {'class': 'ClassicTransformerLM', 'module_types': None, 'architecture': arch}
        report['runtime_profile'] = {'runtime_model': 'transformer_dense', 'mlp_projection_count': 2}
    report['model'].update({
        'total_params': summary['total_params'], 'trainable_params': summary['trainable_params'],
        'provided_config': model, 'logical_weight_bytes': summary['logical_weight_bytes'],
        'logical_buffer_bytes': 0, 'parameter_summary': summary,
    })
    report['tensor_inventory'] = {'weights': inventory, 'buffers': [], 'gradients': [],
                                  'optimizer_states': [], 'explicit_master_weights': []}
    grad = training.get('gradient_dtype')
    if grad in DTYPE_BYTES:
        report['dense_gradient_estimate_bytes'] = summary['trainable_params'] * DTYPE_BYTES[grad]
        report['gradient_estimate_assumption'] = (f'All trainable parameters receive dense {grad} '
                                                  'gradients (explicit configuration).')
    report['optimizer'] = {'class': training.get('optimizer'), 'parameter_groups': [],
                           'observed_state_tensor_count': 0, 'future_state_memory_bytes': None}
    report['configured_optimizer'] = {'name': training.get('optimizer'),
                                      'state_dtype': training.get('optimizer_state_dtype'),
                                      'options': training.get('optimizer_options') or {},
                                      'master_weights': training.get('master_weights'),
                                      'note': 'User configuration; one parameter group covering all parameters.'}
    precision = training.get('precision')
    if specs and precision in DTYPE_BYTES and (profile == 'mlp_regression' or training.get('sequence_length')):
        report['profile_estimates'] = P.activation_estimate(
            profile, model, training['batch_size_per_gpu'], training.get('sequence_length'),
            DTYPE_BYTES[precision])
    if missing:
        report['status'] = 'incomplete'
    return report


def save_report(report, path):
    Path(path).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', required=True, choices=sorted(P.PROFILES))
    parser.add_argument('--model-config', required=True, help='JSON/YAML architecture settings.')
    parser.add_argument('--training-config', required=True, help='JSON/YAML training settings.')
    parser.add_argument('--model-script', help='Optional; scanned with ast only, never executed.')
    parser.add_argument('--training-script', help='Optional; scanned with ast only, never executed.')
    parser.add_argument('--output', default='workload.json')
    args = parser.parse_args()
    inputs = [p for p in (args.model_config, args.training_config, args.model_script,
                          args.training_script) if p]
    if Path(args.output).resolve() in {Path(p).resolve() for p in inputs}:
        parser.error('--output must not overwrite an input file.')
    try:
        scans = []
        if args.model_script:
            scans.append(extract_model_script(args.model_script))
        if args.training_script:
            scans.append(extract_training_script(args.training_script))
        report = collect_from_metadata(args.profile, load_config(args.model_config),
                                       load_config(args.training_config),
                                       script_findings=scans or None)
        save_report(report, args.output)
    except (ValueError, OSError, SyntaxError) as exc:
        parser.exit(1, f'Collection failed ({type(exc).__name__}): {exc}\n')
    print(f"{report['status']}: saved {args.output}")
    for key in ('missing_model_fields', 'missing_training_fields'):
        if report.get(key):
            print(f"{key}: {', '.join(report[key])}")
    for item in report['profile_errors'] + report['unsupported']:
        print(f"{item['field']}: {item['message']}")


if __name__ == '__main__':
    main()
