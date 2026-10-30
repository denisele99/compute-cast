"""LEGACY module unit tests (MLP / classic-transformer profiles, kept so the retained code stays
working). MVP flow tests live in test_api.py."""
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path


from app.clients import huggingface as hf
from app.parsers import normalize_inputs as N
from app.parsers.collect_workload import collect_from_metadata
from app.parsers.detect_hardware_lookup import search
from app.parsers.static_extract import extract_model_script, extract_training_script, merge_findings
from app.services.estimators import pricing, profiles

ROOT = Path(__file__).resolve().parents[1]

GPT2 = {'model.profile': 'transformer_lm_classic', 'model.vocab_size': 50257, 'model.hidden_size': 768,
        'model.num_hidden_layers': 12, 'model.num_attention_heads': 12, 'model.position_embedding': 'learned',
        'model.max_position_embeddings': 1024, 'model.tie_word_embeddings': True, 'model.linear_bias': True,
        'model.layer_norm_bias': True, 'model.lm_head_bias': False}
TX_TRAIN = {'training.task': 'causal_language_modeling', 'training.loss': 'cross_entropy',
            'training.batch_size_per_gpu': 8, 'training.schedule_mode': 'epochs', 'training.epochs': 1,
            'training.gradient_accumulation_steps': 1, 'training.precision': 'bf16',
            'training.parameter_dtype': 'fp32', 'training.gradient_dtype': 'fp32', 'training.optimizer': 'AdamW',
            'training.optimizer_state_dtype': 'fp32', 'training.master_weights': 'none',
            'training.sequence_length': 1024, 'training.attention_backend': 'eager',
            'training.gradient_checkpointing': False, 'training.input_dtype': 'int64',
            'training.target_dtype': 'int64', 'training.gpu_count_used': 1,
            'training.distributed_strategy': 'single', 'dataset.num_train_examples': 10000}
H100 = {'hardware.total_vram_gib': 80.0, 'hardware.usable_vram_gib': 72.0, 'hardware.gpu_count_available': 1,
        'hardware.peak_tflops_fp32': 67, 'hardware.peak_tflops_bf16': 989.4, 'hardware.memory_bandwidth_gb_s': 3350,
        'hardware.price_per_hour': 2.0, 'hardware.price_unit': 'gpu_hour', 'dataset.assume_data_ready': True}
MLP = {'model.profile': 'mlp_regression', 'model.input_size': 100, 'model.hidden_size': 64,
       'model.output_size': 1, 'model.linear_bias': True, 'training.task': 'regression', 'training.loss': 'mse',
       'training.batch_size_per_gpu': 16, 'training.epochs': 10, 'training.gradient_accumulation_steps': 1,
       'training.precision': 'fp32', 'training.parameter_dtype': 'fp32', 'training.gradient_dtype': 'fp32',
       'training.optimizer': 'AdamW', 'training.optimizer_state_dtype': 'fp32', 'training.master_weights': 'none',
       'training.gradient_checkpointing': False, 'training.input_dtype': 'float32',
       'training.target_dtype': 'float32', 'training.gpu_count_used': 1, 'training.distributed_strategy': 'single',
       'dataset.num_train_examples': 100000}

MLP_SCRIPT = '''
import torch.nn as nn
import os
open(os.environ.get("CC_SENTINEL", "/nonexistent/sentinel"), "w").write("executed")
import module_that_does_not_exist
HIDDEN = 64
def build_model(config):
    return nn.Sequential(nn.Linear(100, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
raise SystemExit("importing this file would exit")
'''
TRAIN_SCRIPT = '''
import torch
from torch.utils.data import DataLoader
BATCH = 16
loader = DataLoader(ds, batch_size=BATCH, drop_last=False)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=False)
loss_fn = torch.nn.MSELoss()
epochs = 10
gradient_accumulation_steps = get_steps()
with torch.autocast("cuda", dtype=torch.bfloat16):
    pass
'''


def apply(fields):
    """Apply merged extraction findings the way the frontend autofill does."""
    return {k: f['value'] for k, f in fields.items() if f['value'] is not None}


class ProfileTests(unittest.TestCase):
    def test_gpt2_small_parameter_count(self):
        model = N.model_settings(N.validate(GPT2)['values'])
        inventory, summary = profiles.parameter_layout('transformer_lm_classic', model, 'fp32')
        self.assertEqual(summary['total_params'], 124_439_808)
        untied = profiles.parameter_layout('transformer_lm_classic', {**model, 'tie_word_embeddings': False}, 'fp32')[1]
        self.assertEqual(untied['total_params'] - summary['total_params'], 50257 * 768)

    def test_mlp_bias_setting(self):
        m = {'input_size': 100, 'hidden_size': 64, 'output_size': 1}
        self.assertEqual(profiles.parameter_layout('mlp_regression', {**m, 'linear_bias': True}, 'fp32')[1]['total_params'], 6529)
        self.assertEqual(profiles.parameter_layout('mlp_regression', {**m, 'linear_bias': False}, 'fp32')[1]['total_params'], 6464)

    def test_unsupported_configurations_explained(self):
        cases = {'model.num_key_value_heads': 4, 'model.intermediate_size': 2048,
                 'training.attention_backend': 'sdpa', 'training.gradient_checkpointing': True,
                 'training.precision': 'fp32'}
        for key, value in cases.items():
            result = N.validate({**GPT2, **TX_TRAIN, **H100, key: value})
            fields = [u['field'] for u in result['unsupported']]
            self.assertIn(key, fields, key)
            self.assertFalse(result['can_estimate']['memory'])
            self.assertTrue(all(u['message'] for u in result['unsupported']))

    def test_parameter_dtype_required_and_not_inferred(self):
        values = {**MLP, **H100}
        values.pop('training.parameter_dtype')
        result = N.validate(values)
        self.assertIn('training.parameter_dtype', [m['field'] for m in result['missing']])
        self.assertFalse(result['can_estimate']['memory'])
        self.assertNotIn('training.parameter_dtype', result['suggestions'])
        workload = collect_from_metadata('mlp_regression', N.model_settings(result['values']),
                                         N.training_settings(result['values']))
        self.assertEqual(workload['status'], 'incomplete')
        self.assertNotIn('tensor_inventory', workload)


class ExtractionTests(unittest.TestCase):
    def test_static_extraction_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / 'sentinel'
            script = Path(tmp) / 'model.py'
            script.write_text(MLP_SCRIPT.replace('/nonexistent/sentinel', str(sentinel)))
            os.environ['CC_SENTINEL'] = str(sentinel)
            result = extract_model_script(script)
            self.assertFalse(sentinel.exists())
            self.assertNotIn('module_that_does_not_exist', sys.modules)
        fields = merge_findings([result])
        self.assertEqual(result['profile_hint'], 'mlp_regression')
        self.assertEqual((fields['model.input_size']['value'], fields['model.input_size']['status']), (100, 'extracted'))
        self.assertEqual(fields['model.hidden_size']['value'], 64)
        self.assertIn('resolved from HIDDEN', fields['model.hidden_size']['message'])
        self.assertTrue(fields['model.linear_bias']['value'])

    def test_training_script_literals_and_unresolved(self):
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as handle:
            handle.write(TRAIN_SCRIPT)
        fields = merge_findings([extract_training_script(handle.name)])
        os.unlink(handle.name)
        self.assertEqual(fields['training.batch_size_per_gpu']['value'], 16)
        self.assertEqual(fields['training.optimizer']['value'], 'AdamW')
        self.assertEqual(fields['training.precision']['value'], 'bf16')
        self.assertEqual(fields['training.loss']['value'], 'mse')
        self.assertEqual(fields['training.gradient_accumulation_steps']['status'], 'unresolved')
        self.assertIsNone(fields['training.gradient_accumulation_steps']['value'])

    def test_conflicting_values_are_uncertain(self):
        a = {'findings': [{'field': 'training.epochs', 'value': 3, 'status': 'extracted', 'file': 'a', 'line': 1, 'message': None}]}
        b = {'findings': [{'field': 'training.epochs', 'value': 5, 'status': 'extracted', 'file': 'b', 'line': 2, 'message': None}]}
        merged = merge_findings([a, b])['training.epochs']
        self.assertEqual(merged['status'], 'uncertain')
        self.assertEqual(merged['candidates'], [3, 5])


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class HubTests(unittest.TestCase):
    def test_config_and_search_with_stubbed_http(self):
        gpt2 = {'model_type': 'gpt2', 'n_embd': 768, 'n_layer': 12, 'n_head': 12, 'vocab_size': 50257,
                'n_positions': 1024, 'n_inner': None, 'activation_function': 'gelu_new'}
        seen = []

        def opener(request, timeout):
            seen.append(request.full_url)
            body = [{'id': 'openai-community/gpt2', 'tags': ['pytorch']}] if '/api/models' in request.full_url else gpt2
            return FakeResponse(json.dumps(body).encode())
        self.assertEqual(hf.search_models('gpt2', opener=opener)[0]['id'], 'openai-community/gpt2')
        config = hf.fetch_config('openai-community/gpt2', opener=opener)
        self.assertTrue(seen[-1].endswith('/openai-community/gpt2/resolve/main/config.json'))
        from app.parsers.static_extract import map_model_config
        fields = merge_findings([map_model_config(config, 'gpt2/config.json', 'huggingface')])
        values = {**apply(fields), **TX_TRAIN, **H100, 'model.layer_norm_bias': True, 'model.lm_head_bias': False}
        check = N.validate(values)
        self.assertTrue(check['can_estimate']['memory'], check['missing'] + check['errors'] + check['unsupported'])

    def test_unsupported_architecture_and_network_errors(self):
        from app.parsers.static_extract import map_model_config
        llama = map_model_config({'model_type': 'llama', 'hidden_size': 4096, 'num_hidden_layers': 32,
                                  'num_attention_heads': 32, 'num_key_value_heads': 8, 'vocab_size': 32000,
                                  'intermediate_size': 11008, 'hidden_act': 'silu'}, 'cfg')
        self.assertTrue(any('gated' in u['message'].lower() for u in llama['unsupported']))

        def down(request, timeout):
            raise urllib.error.URLError('blocked')
        with self.assertRaises(hf.HubError):
            hf.fetch_config('gpt2', opener=down)
        with self.assertRaises(ValueError):
            hf.fetch_config('../etc/passwd')


class PricingAndCatalogTests(unittest.TestCase):
    def test_catalogue_search_limits_and_units(self):
        result = search('')
        self.assertLessEqual(len(result['items']), 10)
        h100 = search('h100 eu-north1')['items'][0]
        self.assertEqual(h100['price']['unit'], 'gpu_hour')
        self.assertTrue(h100['price']['preemptible'])
        self.assertEqual(h100['specs']['catalog_gpu_id'], 'nvidia_h100_sxm_80gb')
        # B200 is priced but absent from nebius_platforms.json: memory comes from the general catalogue,
        # peak specs stay empty (memory-only catalogue entry).
        b200 = search('b200')['items'][0]
        self.assertEqual(b200['vram_gib'], 180)
        self.assertIn('hardware_catalog.json', b200['vram_source'])
        self.assertTrue(any('GPU memory only' in i for i in b200['issues']))

    def test_gpu_count_applied_once(self):
        gpu = pricing.estimate_cost(2.0, {'price_per_hour': 1.0, 'unit': 'gpu_hour', 'gpus_allocated': 8, 'gpus_used': 8})
        self.assertEqual(gpu['cost'], 16.0)
        self.assertEqual(gpu['gpu_hours_billed'], 16.0)
        inst = pricing.estimate_cost(2.0, {'price_per_hour': 10.0, 'unit': 'instance_hour', 'gpus_per_instance': 8,
                                           'gpus_allocated': 8, 'gpus_used': 8})
        self.assertEqual(inst['cost'], 20.0)
        idle = pricing.estimate_cost(1.0, {'price_per_hour': 1.0, 'unit': 'gpu_hour', 'gpus_allocated': 2, 'gpus_used': 1})
        self.assertEqual(idle['cost'], 2.0)
        self.assertIn('idle', idle['warning'])
        self.assertEqual(pricing.estimate_cost(1.0, {'price_per_hour': None, 'unit': None})['status'], 'unavailable')

if __name__ == '__main__':
    unittest.main()
