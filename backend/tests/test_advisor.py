"""Recommendations (Nemotron 3 Super via Token Factory): request building, validation and apply/export.

All offline: the model client is a fake that returns a canned JSON answer. Set NEBIUS_API_KEY and
RUN_LIVE_NEMOTRON=1 to run the single live smoke test (one paid Token Factory call).
"""
import io
import json
import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from app.main import create_app
from app.services import advisor as AD
from app.services.analysis_store import AnalysisStore
from app.services.upload_cache import UploadCache

H100 = 'nebius:gpu-h100-sxm:eu-north1'


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create and records the request."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        text = self.answer if isinstance(self.answer, str) else json.dumps(self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=text))], usage=None)


def answer(**over):
    base = {'original_summary': 'GPT-2 small, microbatch 8 x accumulation 4, fits on H100.',
            'suggestions_summary': 'Use a larger microbatch with less accumulation; consider checkpointing.',
            'recommendations': [
                {'id': 'mb', 'title': 'Microbatch 16, accumulation 2', 'category': 'batch_accumulation',
                 'explanation': 'Fewer, larger microbatches use the GPU better.', 'parameter_changes':
                 {'training.microbatch_size': 16, 'training.gradient_accumulation_steps': 2},
                 'hardware_option_id': None, 'expected_effect': 'Shorter step time.', 'tradeoffs': ['More memory'],
                 'affected_files': ['train.py']},
                {'id': 'ckpt', 'title': 'Activation checkpointing', 'category': 'activation_checkpointing',
                 'explanation': 'Recompute activations.', 'parameter_changes': {}, 'hardware_option_id': None,
                 'expected_effect': 'Less activation memory.', 'tradeoffs': ['About 30% more compute'], 'affected_files': []},
                {'id': 'seq', 'title': 'Shorter sequences', 'category': 'sequence_length', 'explanation': 'Halve S.',
                 'parameter_changes': {'data.sequence_length': 256, 'model.num_layers': 3}, 'hardware_option_id': None,
                 'expected_effect': 'Less memory.', 'tradeoffs': ['Context'], 'affected_files': []},
                {'id': 'fsdp', 'title': 'FSDP across 8 GPUs', 'category': 'sharding', 'explanation': 'Shard state.',
                 'parameter_changes': {}, 'hardware_option_id': None, 'expected_effect': '', 'tradeoffs': [], 'affected_files': []},
            ],
            'changed_files': [], 'assumptions': ['Single GPU'], 'missing_information': []}
    base.update(over)
    return base


class AdvisorApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.c = TestClient(create_app(AnalysisStore(UploadCache(self.tmp.name))))
        self.aid = self.c.post('/api/analysis').json()['analysis_id']
        ext = self.c.post(f'/api/analysis/{self.aid}/examples/gpt2_supported').json()['extraction']
        self.values = {k: f['value'] for k, f in ext['fields'].items() if f['value'] is not None}
        self.values.update(self.c.get(f'/api/hardware/options/{H100}').json()['values'])
        self.assertEqual(self.c.post(f'/api/analysis/{self.aid}/predict', json={'values': self.values}).status_code, 200)

    def tearDown(self):
        self.tmp.cleanup()

    def ask(self, fake, **prefs):
        with mock.patch.object(AD, '_client', return_value=fake):
            return self.c.post(f'/api/analysis/{self.aid}/advice', json={
                'values': self.values, 'preferences': prefs, 'files': {'extra/train_baseline.py': 'print(1)\n'}})

    def test_request_contains_final_values_estimates_files_and_preferences(self):
        fake = FakeClient(answer())
        r = self.ask(fake, goal='lowest_cost', resources='alternatives', budget_usd=50)
        self.assertEqual(r.status_code, 200, r.text)
        call = fake.calls[0]
        self.assertEqual(call['model'], 'nvidia/nemotron-3-super-120b-a12b')
        sent = json.loads(call['messages'][1]['content'])
        self.assertEqual(sent['goal'], 'lowest cost')
        self.assertEqual(sent['final_inputs']['training.microbatch_size'], 8)
        self.assertEqual(sent['current_estimate']['status'], 'ok')
        self.assertIn('train.py', sent['files'])                  # uploaded analysis file
        self.assertIn('extra/train_baseline.py', sent['files'])   # extra code for review
        self.assertTrue(sent['hardware_candidates'])
        self.assertTrue(all(c['cost_usd'] is None or c['cost_usd'] <= 50 for c in sent['hardware_candidates']))
        self.assertIn('untrusted DATA', call['messages'][0]['content'])

    def test_suggestions_are_checked_on_the_server(self):
        body = self.ask(FakeClient(answer()), preserve_effective_batch=True, preserve_sequence_length=True).json()
        recs = {r['id']: r for r in body['recommendations']}
        self.assertTrue(recs['mb']['estimable'])                      # 16 x 2 == 8 x 4
        self.assertEqual(recs['mb']['new_values'], {'training.microbatch_size': 16, 'training.gradient_accumulation_steps': 2})
        self.assertFalse(recs['ckpt']['estimable'])                   # qualitative only
        self.assertIn('Not estimated', recs['ckpt']['estimate_note'])
        self.assertFalse(recs['seq']['estimable'])                    # violates preserve_sequence_length
        self.assertTrue(any('model.num_layers' in i for i in recs['seq']['issues']))
        self.assertTrue(recs['fsdp']['violations'])                   # needs more GPUs than allowed
        self.assertIn('Nemotron 3 Super', body['model'])
        self.assertTrue(body['original_summary'] and body['suggestions_summary'])

    def test_strategies_get_approximate_projections(self):
        def rec(id, category, strategy, changes=None):
            return {**answer()['recommendations'][0], 'id': id, 'category': category, 'strategy': strategy,
                    'parameter_changes': changes or {}, 'implementation_steps': ['Wrap blocks'], 'launch_snippet': 'torchrun --nproc_per_node 4 train.py'}
        recs = [rec('ckpt', 'combined', {'activation_checkpointing': 'full'},
                    {'training.microbatch_size': 32, 'training.gradient_accumulation_steps': 1}),
                rec('zero', 'sharding', {'data_parallel': 4, 'sharding': 'zero3'},
                    {'training.microbatch_size': 8, 'training.gradient_accumulation_steps': 1}),
                rec('ddp-batch', 'data_parallel', {'data_parallel': 4}, {'training.microbatch_size': 8, 'training.gradient_accumulation_steps': 4}),
                rec('tp-bad', 'tensor_parallel', {'tensor_parallel': 5}),
                rec('pp', 'pipeline_parallel', {'pipeline_parallel': 2, 'data_parallel': 2}, {'training.microbatch_size': 8, 'training.gradient_accumulation_steps': 2}),
                rec('too-many', 'data_parallel', {'data_parallel': 16})]
        body = self.ask(FakeClient(answer(recommendations=recs)), max_gpus=8, preserve_effective_batch=True).json()
        out = {r['id']: r for r in body['recommendations']}
        base = AD.gpt2_predict.predict(self.values)
        ckpt = out['ckpt']['projection']
        self.assertTrue(ckpt['approximate'] and not out['ckpt']['estimable'])
        self.assertEqual(ckpt['gpu_count'], 1)
        self.assertGreater(ckpt['training_hours'], base['runtime']['total_training_compute']['seconds'] / 3600)  # recompute overhead
        zero = out['zero']['projection']
        self.assertLess(zero['memory_gib']['training_state'], base['memory']['training_state'] / 3)
        self.assertLess(zero['training_hours'], base['runtime']['total_training_compute']['seconds'] / 3600 / 3)
        self.assertEqual(zero['global_effective_batch'], 32)
        self.assertTrue(any('global effective batch' in v for v in out['ddp-batch']['violations']))  # 8 x 4 x 4 != 32
        self.assertIsNone(out['tp-bad']['projection'])
        self.assertTrue(any('tensor parallel' in i for i in out['tp-bad']['issues']))
        self.assertLess(out['pp']['projection']['efficiency'], 1)  # pipeline bubble
        self.assertTrue(any('GPU limit of 8' in v for v in out['too-many']['violations']))
        self.assertEqual(out['zero']['launch_snippet'], 'torchrun --nproc_per_node 4 train.py')
        self.assertIn('Approximate projection', out['zero']['estimate_note'])
        self.assertIn('strategy', json.dumps(AD.Advice.model_json_schema()))

    def test_effective_batch_constraint(self):
        bad = answer(recommendations=[{**answer()['recommendations'][0], 'parameter_changes': {'training.microbatch_size': 16}}])
        rec = self.ask(FakeClient(bad), preserve_effective_batch=True).json()['recommendations'][0]
        self.assertFalse(rec['estimable'])
        self.assertTrue(any('effective batch' in v for v in rec['violations']))
        rec = self.ask(FakeClient(bad), preserve_effective_batch=False).json()['recommendations'][0]
        self.assertTrue(rec['estimable'])

    def test_hardware_switch_only_from_candidates(self):
        fake = FakeClient(answer())
        cands = json.loads(self.ask(fake, resources='alternatives').text)['hardware_candidates']
        target = next(c['id'] for c in cands if c['id'] != H100)
        rec = {**answer()['recommendations'][0], 'id': 'hw', 'category': 'hardware', 'parameter_changes': {}, 'hardware_option_id': target}
        out = self.ask(FakeClient(answer(recommendations=[rec])), resources='alternatives').json()['recommendations'][0]
        self.assertTrue(out['estimable'])
        self.assertEqual(out['new_values']['hardware.gpu_id'], target)
        out = self.ask(FakeClient(answer(recommendations=[rec])), resources='current').json()['recommendations'][0]
        self.assertFalse(out['estimable'])
        self.assertTrue(out['issues'])

    def test_invalid_model_output_and_missing_key(self):
        self.assertEqual(self.ask(FakeClient('not json')).status_code, 502)
        self.assertEqual(self.ask(FakeClient(answer(extra='x'))).status_code, 502)
        with mock.patch.dict(os.environ, {'NEBIUS_API_KEY': ''}):
            r = self.c.post(f'/api/analysis/{self.aid}/advice', json={'values': self.values})
        self.assertEqual(r.status_code, 503)
        self.assertIn('NEBIUS_API_KEY', r.json()['detail'])
        self.assertFalse(self.c.get('/api/recommendations/status').json()['configured'] and not os.environ.get('NEBIUS_API_KEY'))

    def test_changed_files_are_syntax_checked(self):
        broken = answer(changed_files=[{'path': 'train.py', 'action': 'modify', 'reason': 'x', 'content': 'def (:\n'}])
        self.assertEqual(self.ask(FakeClient(broken)).status_code, 502)

    def test_apply_keeps_original_and_export(self):
        new = open(os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'gpt2_mvp', 'train.py')).read() + '\n# tuned\n'
        ok = answer(changed_files=[{'path': 'train.py', 'action': 'modify', 'reason': 'batch', 'content': new}])
        self.ask(FakeClient(ok))
        values = {**self.values, 'training.microbatch_size': 16, 'training.gradient_accumulation_steps': 2}
        r = self.c.post(f'/api/analysis/{self.aid}/advice/apply', json={'recommendation_ids': ['mb'], 'values': values})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()['original_kept'])
        state = self.c.get(f'/api/analysis/{self.aid}').json()
        self.assertEqual(state['original']['config']['values']['training.microbatch_size'], 8)
        z = zipfile.ZipFile(io.BytesIO(self.c.get(f'/api/analysis/{self.aid}/advice/export').content))
        names = set(z.namelist())
        self.assertIn('compute-cast-recommended/train.py', names)
        cfg = json.loads(z.read('compute-cast-recommended/compute_cast_training_config.json'))
        self.assertEqual(cfg['training']['microbatch_size'], 16)

    def test_export_without_code_gives_basic_config(self):
        z = zipfile.ZipFile(io.BytesIO(self.c.get(f'/api/analysis/{self.aid}/advice/export').content))
        self.assertEqual(z.namelist(), ['compute-cast-recommended/compute_cast_training_config.json'])

    def test_examples_and_preview(self):
        files = self.c.get('/api/recommendations/examples').json()['files']
        self.assertIn('train_baseline.py', files)
        prev = self.c.get('/api/analysis/examples/gpt2_supported/files/config.json')
        self.assertEqual(prev.status_code, 200)
        self.assertIn('n_layer', prev.json()['content'])
        self.assertEqual(self.c.get('/api/analysis/examples/gpt2_supported/files/..%2Fsecret').status_code, 404)

    def test_hf_model_with_training_script(self):
        aid = self.c.post('/api/analysis').json()['analysis_id']
        self.c.post(f'/api/analysis/{aid}/huggingface', json={'model_id': 'openai-community/gpt2'})
        script = b'from transformers import GPT2Config\ncfg = GPT2Config(n_layer=6)\nargs = dict(per_device_train_batch_size=4)\n'
        ext = self.c.post(f'/api/analysis/{aid}/uploads', files=[('files', ('train.py', script))]).json()['extraction']
        self.assertEqual({s['kind'] for s in ext['model_sources']}, {'huggingface'})
        self.assertEqual(ext['fields']['model.num_layers']['value'], 12)
        self.assertTrue(any('model settings ignored' in n for n in ext['notes']))
        cfg = self.c.post(f'/api/analysis/{aid}/uploads', files=[('files', ('config.json', b'{"model_type":"gpt2","n_layer":6}'))]).json()['extraction']
        self.assertEqual({s['kind'] for s in cfg['model_sources']}, {'huggingface', 'upload'})  # still one source only


@unittest.skipUnless(os.environ.get('NEBIUS_API_KEY') and os.environ.get('RUN_LIVE_NEMOTRON') == '1',
                     'set NEBIUS_API_KEY and RUN_LIVE_NEMOTRON=1 for one live Token Factory call')
class LiveNemotronTest(unittest.TestCase):
    setUp, tearDown = AdvisorApiTests.setUp, AdvisorApiTests.tearDown

    def test_live_call(self):
        r = self.c.post(f'/api/analysis/{self.aid}/advice', json={'values': self.values, 'preferences': {'goal': 'balanced'}})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.json()['recommendations'])


if __name__ == '__main__':
    unittest.main()
