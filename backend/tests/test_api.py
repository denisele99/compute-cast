"""GPT-2 MVP API and estimator tests."""
import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app.clients import huggingface as hf
from app.main import create_app
from app.services.analysis_store import AnalysisStore
from app.services.estimators import estimate_gpt2 as E
from app.services.upload_cache import UploadCache

ROOT = Path(__file__).resolve().parents[2]
PROTOTYPE_INPUT = json.loads((ROOT / 'examples' / 'gpt2_mvp' / 'gpt2_estimator_input.json').read_text())
H100 = 'nebius:gpu-h100-sxm:eu-north1'


def client(tmp):
    return TestClient(create_app(AnalysisStore(UploadCache(tmp))))


class EstimatorTests(unittest.TestCase):
    def test_matches_prototype_outputs(self):
        d = PROTOTYPE_INPUT
        r = E.estimate(d['model'], d['training'], d['hardware'], d['estimator_policy'])
        self.assertEqual(r['parameters'], 124_439_808)  # GPT-2 small
        self.assertEqual(r['memory_bytes']['estimated_allocated_peak'], 4_070_796_800)
        self.assertAlmostEqual(r['seconds_per_full_optimizer_update'], 0.19447909539840003, places=12)
        self.assertEqual(r['total_training_compute']['optimizer_updates'], 1000)

    def test_epoch_mode_partial_batches_and_groups(self):
        d = PROTOTYPE_INPUT
        t = {'microbatch_size': 2, 'sequence_length': 512, 'gradient_accumulation_steps': 4,
             'num_epochs': 2, 'num_training_examples': 1001}
        run = E.runtime_estimate(d['model'], t, d['hardware'], d['estimator_policy'])
        self.assertEqual(run['total_training_compute']['microbatches_per_epoch'], 501)
        self.assertEqual(run['total_training_compute']['optimizer_updates'], 2 * 126)  # ceil(501/4) per epoch

    def test_memory_needs_no_hardware(self):
        d = PROTOTYPE_INPUT
        mem = E.memory_estimate(d['model'], d['training'], d['estimator_policy'])
        mb = mem['memory_bytes']
        self.assertEqual(mb['training_state'], 16 * mem['parameters'])
        self.assertEqual(mb['capacity_with_runtime_and_margin'],
                         (mb['estimated_allocated_peak'] + 0.5 * 2 ** 30) * 1.15)


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.c = client(self.tmp.name)
        self.aid = self.c.post('/api/analysis').json()['analysis_id']

    def tearDown(self):
        self.tmp.cleanup()

    def url(self, path):
        return f'/api/analysis/{self.aid}{path}'

    def load(self, example='gpt2_supported'):
        ext = self.c.post(self.url(f'/examples/{example}')).json()['extraction']
        return ext, {k: f['value'] for k, f in ext['fields'].items() if f['value'] is not None}

    def hardware(self, option_id=H100):
        return self.c.get(f'/api/hardware/options/{option_id}').json()['values']

    def predict(self, values, metadata=None, record=True):
        r = self.c.post(self.url('/predict'), json={'values': values, 'metadata': metadata or {}, 'record': record})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_schema_follows_contract(self):
        schema = self.c.get('/api/schema').json()
        contract = json.loads((ROOT / 'backend' / 'data' / 'mvp_feature_contract.json').read_text())
        self.assertEqual(len(schema['fixed_assumptions']), len(contract['2_visible_fixed_assumptions']))
        features = {f['contract_feature'] for f in schema['fields'] if f['contract_feature']}
        self.assertEqual(features, {i['feature'] for i in contract['1_direct_inputs_and_file_extractions']})
        self.assertEqual(schema['huggingface_models'], ['openai-community/gpt2', 'distilbert/distilgpt2'])

    def test_example_files_populate_sections_and_predict(self):
        ext, values = self.load()
        self.assertEqual({s for f in ext['files'] for s in f['sections']}, {'model', 'training', 'data'})
        self.assertFalse(ext['executed_code'])
        hw = self.hardware()
        self.assertEqual(hw['hardware.usable_vram_gib'], 0.9 * hw['hardware.vram_gib'])  # 90% default
        result = self.predict({**values, **hw})
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['parameters'], 124_439_808)
        fit = result['fit']
        self.assertEqual(fit['fits'], fit['capacity_gib'] <= fit['usable_vram_gib'])
        seconds = result['runtime']['total_training_compute']['seconds']
        self.assertAlmostEqual(result['cost']['cost'], seconds / 3600 * hw['hardware.price_per_hour'])
        self.assertEqual(result['cost']['unit'], 'gpu_hour')
        self.assertEqual(result['inputs']['training']['microbatch_size'], 8)
        self.assertEqual(result['inputs']['training']['sequence_length'], 512)

    def test_gpu_dropdown_is_merged_and_runtime_complete(self):
        groups = self.c.get('/api/hardware/options').json()
        self.assertTrue(groups['nebius'] and groups['catalog'])
        for row in groups['nebius'] + groups['catalog']:
            values = self.c.get(f"/api/hardware/options/{row['id']}").json()['values']
            for key in ('hardware.vram_gib', 'hardware.bf16_dense_tflops', 'hardware.fp32_tflops', 'hardware.memory_bandwidth_gbps'):
                self.assertTrue(values[key], (row['id'], key))
        catalog_ids = {r['id'] for r in groups['catalog']}
        self.assertNotIn('catalog:nvidia_h100_sxm_80gb', catalog_ids)        # offered on Nebius: listed once there
        self.assertIn('catalog:nvidia_a100_80gb_pcie', catalog_ids)          # not on Nebius
        self.assertFalse(any('b300' in r['id'] or 'rtx' in r['id'] for r in groups['nebius'] + groups['catalog']))  # incomplete specs

    def test_usable_vram_defaults_to_90_percent(self):
        _, values = self.load()
        values.update({'hardware.vram_gib': 40.0, 'hardware.usable_vram_gib': None})
        check = self.c.post(self.url('/validate'), json={'values': values}).json()
        self.assertEqual(check['values']['hardware.usable_vram_gib'], 36.0)
        self.assertEqual(check['suggestions']['hardware.usable_vram_gib']['value'], 36.0)

    def test_memory_only_on_custom_gpu_without_specs(self):
        _, values = self.load()
        result = self.predict({**values, 'hardware.source': 'custom', 'hardware.gpu_name': 'Mine',
                               'hardware.vram_gib': 24.0, 'hardware.usable_vram_gib': 24.0})
        self.assertEqual(result['status'], 'ok')
        self.assertIsNotNone(result['fit'])
        self.assertIsNone(result['runtime'])
        self.assertIn('Peak BF16 dense', result['runtime_missing'])
        self.assertEqual(result['cost']['status'], 'unavailable')

    def test_required_fields_block(self):
        _, values = self.load()
        values.pop('data.sequence_length')
        result = self.predict({**values, **self.hardware()})
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('data.sequence_length', [m['field'] for m in result['validation']['missing']])

    def test_checkpoint_dimensions_need_custom_mode(self):
        _, values = self.load()
        values.update(self.hardware(), **{'model.num_layers': 24})
        blocked = self.predict(values)
        self.assertEqual(blocked['status'], 'blocked')
        self.assertTrue(any('Hypothetical custom configuration' in e['message'] for e in blocked['validation']['errors']))
        custom = self.predict({**values, 'model.custom_config': True})
        self.assertEqual(custom['status'], 'ok')
        self.assertTrue(any('Layers' in h for h in custom['hypothetical']))

    def test_conflicts_block_until_hypothetical_is_chosen(self):
        ext, values = self.load('gpt2_conflicts')
        settings = {c['setting'] for c in ext['conflicts']}
        self.assertEqual(settings, {'precision_mode', 'activation_checkpointing', 'training_mode'})
        values.update(self.hardware())
        self.assertEqual(self.predict(values)['status'], 'blocked')
        ok = self.predict(values, {'acknowledged_conflicts': [c['id'] for c in ext['conflicts']]})
        self.assertEqual(ok['status'], 'ok')
        self.assertEqual(len(ok['hypothetical']), 3)

    def test_unsupported_architecture_has_no_hypothetical(self):
        cfg = json.dumps({'model_type': 'llama', 'hidden_size': 4096}).encode()
        ext = self.c.post(self.url('/uploads'), files=[('files', ('config.json', cfg))]).json()['extraction']
        conflict = ext['conflicts'][0]
        self.assertIsNone(conflict['hypothetical'])
        check = self.c.post(self.url('/validate'), json={'values': {}, 'metadata': {'acknowledged_conflicts': [conflict['id']]}}).json()
        self.assertTrue(any('llama' in e['message'] for e in check['errors']))

    def test_single_huggingface_model_and_exclusive_sources(self):
        with mock.patch.object(hf, 'fetch_config', side_effect=hf.HubError('offline')):
            r = self.c.post(self.url('/huggingface'), json={'model_id': 'distilbert/distilgpt2'})
        self.assertEqual(r.status_code, 200)
        self.assertIn('bundled snapshot', r.json()['origin'])
        self.assertEqual(r.json()['extraction']['fields']['model.num_layers']['value'], 6)
        self.assertEqual(self.c.post(self.url('/huggingface'), json={'model_id': 'openai-community/gpt2'}).status_code, 409)
        self.assertEqual(self.c.post(self.url('/huggingface'), json={'model_id': 'EleutherAI/gpt-neo-125m'}).status_code, 422)
        _, values = self.load()  # adds an uploaded config.json too
        check = self.c.post(self.url('/validate'), json={'values': values}).json()
        self.assertTrue(any('OR uploaded model files' in e['message'] for e in check['errors']))
        self.assertEqual(self.c.delete(self.url('/uploads/hf_distilbert__distilgpt2')).status_code, 200)
        self.assertEqual(self.c.post(self.url('/huggingface'), json={'model_id': 'openai-community/gpt2'}).status_code, 200)

    def test_zip_upload_is_flattened_and_filtered(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            z.write(ROOT / 'examples' / 'gpt2_mvp' / 'config.json', 'proj/config.json')
            z.write(ROOT / 'examples' / 'gpt2_mvp' / 'training_args.json', 'proj/training_args.json')
            z.writestr('../escape.py', 'x = 1\n')
            z.writestr('tool.exe', 'MZ')
            z.writestr('inner.zip', 'PK')
        r = self.c.post(self.url('/uploads'), files=[('files', ('bundle.zip', buf.getvalue()))]).json()
        self.assertEqual(sorted(f['name'] for f in r['added']), ['config.json', 'escape.py', 'training_args.json'])
        self.assertEqual(len(r['skipped']), 2)
        folder = Path(self.tmp.name) / self.aid
        self.assertTrue(all(p.parent == folder for p in folder.iterdir()))

    def test_python_upload_is_never_executed(self):
        with tempfile.TemporaryDirectory() as other:
            sentinel = Path(other) / 'ran'
            code = f'open({str(sentinel)!r}, "w").write("x")\nimport not_a_module\nGPT2Config(n_layer=6)\n'
            r = self.c.post(self.url('/uploads'), files=[('files', ('model.py', code.encode()))]).json()
            self.assertFalse(sentinel.exists())
        f = r['extraction']['fields']['model.num_layers']
        self.assertEqual((f['value'], f['status']), (6, 'uncertain'))

    def test_dataset_sample_is_not_the_dataset_size(self):
        rows = '\n'.join(json.dumps({'input_ids': list(range(20))}) for _ in range(30))
        r = self.c.post(self.url('/uploads'), files=[('files', ('sample.jsonl', rows.encode()))]).json()
        self.assertEqual(r['extraction']['dataset_samples'][0]['sample_rows'], 30)
        self.assertNotIn('data.num_training_examples', r['extraction']['fields'])

    def test_snapshot_changes_and_compare(self):
        _, values = self.load()
        values.update(self.hardware())
        self.predict(values)
        second = self.predict({**values, 'training.microbatch_size': 16})
        change = next(c for c in second['snapshots']['changes'] if c['field'] == 'training.microbatch_size')
        self.assertEqual((change['original'], change['current']), (8, 16))
        rows = self.c.post(self.url('/compare'), json={'values': values}).json()['rows']
        self.assertTrue(rows)
        self.assertEqual(len({round(r['fit']['capacity_gib'], 6) for r in rows if r.get('fit')}), 1)
        self.assertTrue(any(r['hours'] is None for r in rows if r['status'] == 'ok'))  # GPUs without specs
        self.assertEqual(self.c.get(self.url('/files/estimator_input.json')).json()['training']['microbatch_size'], 16)

    def test_cache_cleanup_and_isolation(self):
        self.load()
        folder = Path(self.tmp.name) / self.aid
        self.assertEqual(len(list(folder.iterdir())), 4)
        other = self.c.post('/api/analysis').json()['analysis_id']
        self.assertEqual(self.c.get(f'/api/analysis/{other}').json()['files'], [])
        bad = self.c.post(self.url('/uploads'), files=[('files', ('../x.py', b'x'))])
        self.assertEqual(bad.status_code, 422)
        self.assertTrue(self.c.delete(self.url('')).json()['cache_cleared'])
        self.assertFalse(folder.exists())
        self.c.post(f'/api/analysis/{other}/close')
        self.assertFalse((Path(self.tmp.name) / other).exists())
        store = AnalysisStore(UploadCache(self.tmp.name), ttl_seconds=60)
        aid = store.create()
        self.assertEqual(store.sweep(now=time.time() + 120), [aid])


if __name__ == '__main__':
    unittest.main()
