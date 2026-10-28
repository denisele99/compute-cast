import copy
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import create_app


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app())
        self.config = self.client.post('/api/analysis/normalize', json={}).json()['config']

    def estimate(self, config=None):
        response = self.client.post('/api/analysis/estimate', json=config or self.config)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_unknown_inputs_stay_unknown_and_default_limit_is_90_percent(self):
        r = self.estimate()
        for key in ('training_hours', 'training_cost_usd'):
            self.assertIsNone(r[key])
        self.assertIsNone(r['gpu']['fits'])
        self.assertIsNone(r['cpu']['required_gib'])
        self.assertAlmostEqual(r['gpu']['limit_gib'], 21.6)

    def test_distributed_schedule_accumulation_and_active_gpu_cost(self):
        self.config['training'].update(examples=17, batch_size=4, epochs=2, script_gpu_count=2,
                                       strategy='ddp', gradient_accumulation_steps=2, seconds_per_batch=.5)
        self.config['hardware'].update(available_gpus=8, price_per_gpu_hour=3)
        r = self.estimate()
        self.assertEqual(r['training_steps'], 6)
        self.assertEqual(r['optimizer_updates'], 4)
        self.assertAlmostEqual(r['training_hours'], 3 / 3600)
        self.assertAlmostEqual(r['gpu_hours'], 6 / 3600)
        self.assertEqual(r['training_cost_usd'], .005)

    def test_available_gpus_do_not_imply_speedup(self):
        baseline = self.estimate()
        self.config['hardware']['available_gpus'] = 8
        self.assertEqual(self.estimate()['training_steps'], baseline['training_steps'])

    def test_cross_field_and_scalar_validation(self):
        for group, key, value in [('training', 'batch_size', 100001), ('hardware', 'memory_limit_gib', 25),
                                  ('training', 'script_gpu_count', 2), ('training', 'epochs', 1.5),
                                  ('hardware', 'available_gpus', 0), ('training', 'strategy', 'fsdp_invented')]:
            config = copy.deepcopy(self.config)
            config[group][key] = value
            self.assertEqual(self.client.post('/api/analysis/estimate', json=config).status_code, 422)

    def test_measured_peak_compares_to_limit_not_capacity(self):
        self.config['training']['measured_peak_vram_gib'] = 22
        r = self.estimate()
        self.assertFalse(r['gpu']['fits'])
        self.assertAlmostEqual(r['gpu']['utilization_percent'], 22 / 24 * 100)

    def test_memory_planner_and_incomplete_metadata(self):
        self.config['model']['total_params'] = 1000000
        self.assertIsNone(self.estimate()['gpu']['peak_vram_gib'])
        self.config['model'].update(activation_gib=1, auxiliary_gib=.1)
        r = self.estimate()
        self.assertEqual(r['memory_details']['status'], 'estimated_with_assumptions')
        self.assertGreater(r['gpu']['peak_vram_gib'], 1.1)
        self.assertTrue(r['gpu']['fits'])

    def test_explicit_sharding_does_not_divide_all_memory(self):
        self.config['model'].update(total_params=100000000, activation_gib=1, auxiliary_gib=.1)
        self.config['hardware']['available_gpus'] = 4
        self.config['training'].update(script_gpu_count=4, strategy='ddp')
        ddp = self.estimate()['gpu']['peak_vram_gib']
        self.config['training']['strategy'] = 'zero3'
        zero3 = self.estimate()['gpu']['peak_vram_gib']
        self.assertIsNotNone(zero3)
        self.assertLess(zero3, ddp)
        self.assertGreater(zero3, ddp / 4)

    def test_catalog_joins_prices_and_marks_missing_specs(self):
        r = self.client.get('/api/hardware').json()
        self.assertEqual(len(r['items']), 10)
        self.assertEqual(r['total'], 12)
        first = r['items'][0]
        self.assertEqual(first['platform'], 'gpu-h200-sxm')
        self.assertAlmostEqual(first['vram_gib'], 141 * 10**9 / 1024**3)
        self.assertEqual(first['price_per_gpu_hour'], 2.2)
        self.assertIsNone(r['items'][2]['vram_gib'])
        found = self.client.get('/api/hardware?q=h100').json()['items']
        self.assertTrue(found)
        self.assertTrue(all('h100' in item['platform'] for item in found))

    def parse(self, files):
        r = self.client.post('/api/analysis/parse', json={'files': files})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_static_python_never_executes_and_dynamic_values_are_flagged(self):
        marker = Path('/tmp/compute-cast-parser-must-not-execute')
        if marker.exists():
            marker.unlink()
        r = self.parse([{'name': 'train.py', 'content': f'open({str(marker)!r}, "w").write("bad")\nbatch_size = 8\nepochs = get_epochs()'}])
        self.assertFalse(marker.exists())
        self.assertEqual(r['values']['training']['batch_size'], 8)
        self.assertIn('training.epochs', r['uncertain'])
        self.assertTrue(r['review_required'])

    def test_yaml_and_manual_values_use_same_schema(self):
        r = self.parse([{'name': 'train.yaml', 'content': 'training:\n  num_examples: 100\n  batch_size_per_gpu: 8\n  epochs: 2\n'}])
        config = copy.deepcopy(self.config)
        config['training'].update(r['values']['training'])
        normalized = self.client.post('/api/analysis/normalize', json=config).json()
        self.assertEqual(normalized['config']['training']['batch_size'], 8)
        self.assertEqual(self.estimate(config)['training_steps'], 26)

    def test_conflicts_are_not_silently_overwritten(self):
        r = self.parse([{'name': 'train.py', 'content': 'batch_size = 8'}, {'name': 'config.json', 'content': '{"batch_size":16}'}])
        self.assertIn('training.batch_size', r['uncertain'])
        self.assertNotIn('batch_size', r['values'].get('training', {}))

    def test_bad_configs_recursive_aliases_and_nonfinite_values(self):
        for content in ('[1, 2]', 'self: &self\n  again: *self', 'batch_size: .nan'):
            self.assertIn('config.yaml', self.parse([{'name': 'config.yaml', 'content': content}])['errors'])
        self.assertEqual(self.client.post('/api/analysis/parse', json={'files': [{'name': 'x.py', 'content': 'x' * 250001}]}).status_code, 422)
        self.assertIn('bad.exe', self.parse([{'name': 'bad.exe', 'content': 'x'}])['errors'])

    def test_benchmark_script_download_and_no_cloud_execution(self):
        r = self.client.get('/api/benchmark/script')
        self.assertEqual(r.status_code, 200)
        self.assertIn('torch.cuda.synchronize()', r.text)
        self.assertIn('build_benchmark', r.text)
        self.assertEqual(self.client.post('/api/benchmark/cloud').status_code, 501)

    def report(self):
        return {'schema_version': 1, 'config': copy.deepcopy(self.config), 'steps': 20,
                'seconds_per_batch': .25, 'measured_peak_vram_gib': 4, 'gpu_name': 'Test GPU',
                'capacity_gib': 24, 'scope': 'whole_training_microbatch', 'note': 'User supplied'}

    def test_matching_benchmark_import_updates_same_schema(self):
        r = self.client.post('/api/benchmark/import', json={'config': self.config, 'report': self.report()})
        self.assertEqual(r.status_code, 200, r.text)
        result = self.estimate(r.json()['config'])
        self.assertEqual(result['gpu']['source'], 'measured')
        self.assertEqual(result['gpu']['peak_vram_gib'], 4)
        self.assertAlmostEqual(result['training_hours'], 62500 * .25 / 3600)

    def test_mismatched_benchmark_rejected(self):
        report = self.report()
        report['config']['training']['batch_size'] = 8
        self.assertEqual(self.client.post('/api/benchmark/import', json={'config': self.config, 'report': report}).status_code, 422)
        report = self.report()
        report['measured_peak_vram_gib'] = 25
        self.assertEqual(self.client.post('/api/benchmark/import', json={'config': self.config, 'report': report}).status_code, 422)

    def test_metadata_only_runtime_uses_existing_planner_and_explicit_assumptions(self):
        self.config['model'].update(total_params=1000000, flops_per_microbatch=1e10, hbm_bytes_per_microbatch=1e9)
        self.config['hardware'].update(compute_tflops=30, fp32_tflops=30, memory_bandwidth_gbps=900, price_per_gpu_hour=2)
        # Incomplete data pipeline must not be silently treated as free.
        incomplete = self.estimate()
        self.assertIsNone(incomplete['training_hours'])
        self.config['training']['assume_data_ready'] = True
        result = self.estimate()
        self.assertEqual(result['runtime_details']['status'], 'estimated_with_assumptions')
        self.assertGreater(result['training_hours'], 0)
        self.assertIn('metadata-only', result['timing_source'])
        self.assertEqual(result['training_hours'], result['runtime_details']['scenarios']['slow']['training_hours'])
        self.assertGreater(result['training_cost_usd'], 0)

    def test_python_non_json_literals_become_uncertain(self):
        for expression in ('{1, 2}', 'b"hello"', '1j'):
            parsed = self.parse([{'name': 'train.py', 'content': 'batch_size = ' + expression}])
            self.assertIn('training.batch_size', parsed['uncertain'])
