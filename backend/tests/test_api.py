import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path

from fastapi.testclient import TestClient

from app.clients import huggingface as hf
from app.main import create_app
from app.parsers import normalize_inputs as N
from app.parsers.collect_workload import collect_from_metadata
from app.parsers.detect_hardware_lookup import search
from app.parsers.static_extract import extract_model_script, extract_training_script, merge_findings
from app.services.analysis_store import AnalysisStore
from app.services.estimators import pricing, profiles
from app.services.upload_cache import UploadCache, UploadError, validate_filename

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


def client(tmp):
    return TestClient(create_app(AnalysisStore(UploadCache(tmp))))


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
        b200 = search('b200')['items'][0]
        self.assertIsNone(b200['vram_gib'])
        self.assertTrue(any('missing from nebius_platforms' in i for i in b200['issues']))

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


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.c = client(self.tmp.name)
        self.aid = self.c.post('/api/analysis').json()['analysis_id']

    def tearDown(self):
        self.tmp.cleanup()

    def predict(self, values, record=True):
        r = self.c.post(f'/api/analysis/{self.aid}/predict', json={'values': values, 'record': record})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_transformer_prediction_fit_runtime_cost(self):
        result = self.predict({**GPT2, **TX_TRAIN, **H100, 'assumptions.startup_minutes': 6})
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['parameters']['total_params'], 124_439_808)
        fit = result['fit']
        self.assertEqual(fit['usable_vram_gib'], 72.0)
        self.assertEqual(fit['fits'], fit['peak_vram_gib'] <= 72.0)
        self.assertAlmostEqual(fit['percent_of_usable'], 100 * fit['peak_vram_gib'] / 72.0)
        slow = result['runtime']['scenarios']['slow']
        self.assertAlmostEqual(slow['job_seconds'], slow['training_seconds'] + 360)
        self.assertAlmostEqual(result['cost']['slow']['cost'], slow['job_hours'] * 1 * 2.0)
        self.assertEqual(result['benchmark_cost']['status'], 'estimated')
        self.assertIn('transformer_activations', result['explanation']['parameter_and_activation_formulas'])
        saved = result['profile_estimates']['saved_activation_bytes']
        self.assertEqual(saved, 34 * 12 * 8 * 1024 * 768 + 5 * 12 * 8 * 12 * 1024 * 1024)

    def test_memory_fit_false_on_small_gpu(self):
        result = self.predict({**GPT2, **TX_TRAIN, **H100, 'hardware.total_vram_gib': 16.0,
                               'hardware.usable_vram_gib': 14.4})
        self.assertFalse(result['fit']['fits'])
        self.assertGreater(result['fit']['percent_of_usable'], 100)

    def test_required_fields_block_and_runtime_missing_does_not(self):
        values = {**MLP, **H100}
        values.pop('training.batch_size_per_gpu')
        blocked = self.predict(values)
        self.assertEqual(blocked['status'], 'blocked')
        self.assertIn('training.batch_size_per_gpu', [m['field'] for m in blocked['validation']['missing']])
        partial = {**MLP, **H100, 'dataset.assume_data_ready': False}
        result = self.predict(partial)
        self.assertEqual(result['status'], 'ok')
        self.assertIsNotNone(result['memory']['gpu']['peak_vram_gib'])
        self.assertIsNone(result['runtime']['scenarios']['slow']['training_seconds'])
        self.assertTrue(any('disk' in m.lower() or 'storage' in m.lower() for m in result['runtime']['missing_information']))
        self.assertEqual(result['cost']['slow']['status'], 'unavailable')

    def test_batch_larger_than_dataset_is_warning_unless_drop_last(self):
        values = {**MLP, **H100, 'dataset.num_train_examples': 4}
        check = self.c.post(f'/api/analysis/{self.aid}/validate', json={'values': values}).json()
        self.assertTrue(check['can_estimate']['memory'])
        self.assertIn('training.batch_size_per_gpu', [w['field'] for w in check['warnings']])
        check = self.c.post(f'/api/analysis/{self.aid}/validate',
                            json={'values': {**values, 'training.drop_last': True}}).json()
        self.assertIn('training.batch_size_per_gpu', [e['field'] for e in check['errors']])

    def test_upload_and_manual_entry_normalize_identically(self):
        files = {'model': ('model.py', MLP_SCRIPT), 'training': ('train.py', TRAIN_SCRIPT)}
        extraction = None
        for kind, (name, text) in files.items():
            r = self.c.post(f'/api/analysis/{self.aid}/uploads', data={'kind': kind}, files={'file': (name, text.encode())})
            self.assertEqual(r.status_code, 200, r.text)
            extraction = r.json()['extraction']
        self.assertFalse(extraction['executed_code'])
        uploaded = {**apply(extraction['fields'])}
        self.assertEqual(uploaded['model.profile'], 'mlp_regression')
        # User completes the values static analysis could not resolve or that need confirmation.
        completed = {**MLP, **H100, **uploaded, 'training.gradient_accumulation_steps': 1}
        # The same settings typed by hand (train.py uses bf16 autocast, amsgrad=False, drop_last=False).
        manual = {**MLP, **H100, 'training.precision': 'bf16', 'training.amsgrad': False, 'training.drop_last': False}
        a, b = self.predict(completed, record=False), self.predict(manual, record=False)
        self.assertEqual(a['workload']['tensor_inventory'], b['workload']['tensor_inventory'])
        self.assertEqual(a['workload']['training'], b['workload']['training'])
        self.assertEqual(a['memory']['gpu'], b['memory']['gpu'])

    def test_original_snapshot_and_changes(self):
        self.predict({**MLP, **H100, 'training.batch_size_per_gpu': 8})
        second = self.predict({**MLP, **H100, 'training.batch_size_per_gpu': 16})
        change = next(c for c in second['snapshots']['changes'] if c['field'] == 'training.batch_size_per_gpu')
        self.assertEqual((change['original'], change['current']), (8, 16))
        third = self.predict({**MLP, **H100, 'training.batch_size_per_gpu': 32})
        self.assertEqual(third['snapshots']['original']['values']['training.batch_size_per_gpu'], 8)
        files = self.c.get(f'/api/analysis/{self.aid}/files/workload.json')
        self.assertEqual(files.json()['training']['batch_size_per_gpu'], 32)

    def test_hardware_comparison_same_workload(self):
        result = self.c.post(f'/api/analysis/{self.aid}/compare', json={'values': {**GPT2, **TX_TRAIN, **H100}}).json()
        rows = result['rows']
        self.assertTrue(rows)
        params = {r['fit']['peak_vram_gib'] for r in rows if r.get('fit')}
        self.assertEqual(len(params), 1)  # same workload => same peak on every GPU
        self.assertLessEqual(len(rows), 10)
        no_specs = [r for r in rows if r['status'] == 'ok' and any('No general-catalogue specs' in i for i in r['issues'])]
        self.assertTrue(no_specs)
        self.assertTrue(all(r['job_hours']['slow'] is None for r in no_specs))

    def test_cache_isolation_validation_and_cleanup(self):
        r = self.c.post(f'/api/analysis/{self.aid}/uploads', data={'kind': 'model'},
                        files={'file': ('model.py', b'x = 1\n')})
        folder = Path(self.tmp.name) / self.aid
        self.assertEqual(len(list(folder.iterdir())), 1)
        other = self.c.post('/api/analysis').json()['analysis_id']
        self.assertEqual(self.c.get(f'/api/analysis/{other}').json()['files'], [])
        for bad in ('../model.py', '.env', 'run.exe', 'a/b.py'):
            resp = self.c.post(f'/api/analysis/{self.aid}/uploads', data={'kind': 'model'}, files={'file': (bad, b'x')})
            self.assertEqual(resp.status_code, 422, bad)
        big = self.c.post(f'/api/analysis/{self.aid}/uploads', data={'kind': 'model'},
                          files={'file': ('big.py', b'#' * 1_000_001)})
        self.assertEqual(big.status_code, 422)
        self.assertEqual(len(list(folder.iterdir())), 1)
        self.assertEqual(self.c.delete(f'/api/analysis/{self.aid}').json()['cache_cleared'], True)
        self.assertFalse(folder.exists())
        self.assertEqual(self.c.get(f'/api/analysis/{self.aid}').status_code, 404)
        self.c.post(f'/api/analysis/{other}/close')
        self.assertFalse((Path(self.tmp.name) / other).exists())

    def test_expiry_sweep(self):
        store = AnalysisStore(UploadCache(self.tmp.name), ttl_seconds=60)
        aid = store.create()
        (Path(self.tmp.name) / aid / 'x.py').write_text('x')
        self.assertEqual(store.sweep(now=time.time() + 120), [aid])
        self.assertFalse((Path(self.tmp.name) / aid).exists())
        with self.assertRaises(UploadError):
            validate_filename('notes.txt', 'model')

    def test_dataset_sample_is_not_full_dataset(self):
        csv = 'f1,f2,f3,target\n' + '\n'.join(f'{i},{i},{i},{i}' for i in range(50))
        r = self.c.post(f'/api/analysis/{self.aid}/uploads', data={'kind': 'dataset'},
                        files={'file': ('sample.csv', csv.encode())}).json()['extraction']
        self.assertEqual(r['dataset_samples'][0]['sample_rows'], 50)
        self.assertNotIn('dataset.num_train_examples', r['fields'])
        self.assertEqual(r['fields']['model.input_size']['value'], 3)
        self.assertEqual(r['fields']['model.input_size']['status'], 'uncertain')

    def test_benchmark_script_import_and_nebius_unavailable(self):
        values = {**MLP, **H100}
        self.predict(values)
        plan = self.c.post(f'/api/analysis/{self.aid}/benchmark/plan', json={'values': values}).json()
        self.assertTrue(plan['command'].startswith('python run_benchmark.py'))
        self.assertIn('--output', plan['command'])
        self.assertFalse(plan['nebius']['available'])
        script = self.c.post(f'/api/analysis/{self.aid}/benchmark/script', json={'values': values}).text
        ast.parse(script)
        self.assertNotIn('__COMPUTE_CAST_CONFIG__', script)
        result = {'schema': 'compute-cast-benchmark/1', 'analysis_id': self.aid,
                  'config_fingerprint': plan['fingerprint'], 'representative': True,
                  'seconds_per_optimizer_step': {'median': 0.01, 'mean': 0.01, 'max': 0.02},
                  'peak_memory_allocated_gib': 0.5, 'peak_memory_reserved_gib': 0.6}
        imported = self.c.post(f'/api/analysis/{self.aid}/benchmark/import',
                               files={'file': ('r.json', json.dumps(result).encode())}).json()
        self.assertTrue(imported['matches_current_config'])
        self.assertEqual(imported['kind'], 'benchmark')
        self.assertIn('peak_vram', imported['comparison'])
        bad = self.c.post(f'/api/analysis/{self.aid}/benchmark/import', files={'file': ('r.json', b'{"a":1}')})
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(self.c.post(f'/api/analysis/{self.aid}/benchmark/nebius', json={}).status_code, 400)
        self.assertEqual(self.c.post(f'/api/analysis/{self.aid}/benchmark/nebius',
                                     json={'confirm_charges': True}).status_code, 501)

    def test_health_schema_and_docs(self):
        self.assertEqual(self.c.get('/api/health').json(), {'status': 'ok'})
        self.assertEqual(self.c.get('/api/schema').json()['framework'], 'pytorch')
        self.assertEqual(self.c.get('/openapi.json').status_code, 200)


@unittest.skipUnless(os.environ.get('BENCH_PYTHON'), 'set BENCH_PYTHON to a Python with torch installed')
class GeneratedScriptRunsTests(unittest.TestCase):
    def test_reference_models_match_predicted_parameter_counts(self):
        for values, expected in (({**MLP, **H100}, 6529),
                                 ({**GPT2, **TX_TRAIN, **H100, 'model.num_hidden_layers': 2, 'model.hidden_size': 64,
                                   'model.num_attention_heads': 4, 'model.vocab_size': 500,
                                   'model.max_position_embeddings': 64, 'training.sequence_length': 32,
                                   'training.batch_size_per_gpu': 2}, None)):
            with tempfile.TemporaryDirectory() as tmp:
                c = client(tmp)
                aid = c.post('/api/analysis').json()['analysis_id']
                pred = c.post(f'/api/analysis/{aid}/predict', json={'values': values}).json()
                script = Path(tmp) / 'run_benchmark.py'
                script.write_text(c.post(f'/api/analysis/{aid}/benchmark/script', json={'values': values}).text)
                out = Path(tmp) / 'out.json'
                subprocess.run([os.environ['BENCH_PYTHON'], str(script), '--device', 'cpu', '--steps', '2',
                                '--warmup', '1', '--output', str(out)], check=True, cwd=tmp)
                data = json.loads(out.read_text())
                self.assertEqual(data['parameter_count'], expected or pred['parameters']['total_params'])
                imported = c.post(f'/api/analysis/{aid}/benchmark/import',
                                  files={'file': ('out.json', out.read_bytes())}).json()
                self.assertTrue(imported['matches_current_config'])
                self.assertIn('not representative', ' '.join(imported['issues']))


if __name__ == '__main__':
    unittest.main()
