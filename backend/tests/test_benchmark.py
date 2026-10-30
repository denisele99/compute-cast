"""Benchmark mode tests: registry/config validation, package, results, auth, allowances, mocked Nebius.

All offline. No Nebius request is ever sent: the jobs tests use FakeJobsClient, and the SDK test only
constructs a CreateJobRequest message. Optional: BENCH_PYTHON=<python with torch> runs the real runner
on CPU (smoke) and, with BENCH_CUDA=1 on a GPU machine, on CUDA.
"""
import copy
import gzip
import io
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app.main import create_app
from app.services.analysis_store import AnalysisStore
from app.services.benchmarks import jobs as JOBS
from app.services.benchmarks import results as RS
from app.services.benchmarks.jobs import JobService
from app.services.benchmarks.nebius_client import FakeJobsClient, FakeResultStore, JobRequest, build_create_request, unwrap_object
from app.services.benchmarks.policy import load_policy
from app.services.benchmarks.registry import package_dir, registry
from app.services.benchmarks.store import Store
from app.services.upload_cache import UploadCache

R = registry()
EXAMPLE = json.loads((package_dir() / 'benchmark_config.example.json').read_text())
H100 = 'nebius:gpu-h100-sxm:eu-north1'
PASSWORD = 'test-demo-password'
SECRETS = ('S3-SECRET-VALUE', 'SESSION-SECRET-0123456789abcdef', PASSWORD)


def make_policy(db, **kw):
    base = replace(load_policy(), cloud_enabled=True, backend='fake', demo_password=PASSWORD,
                   session_secret=SECRETS[1], database_url=db, image='cr.example/compute-cast-bench@sha256:' + 'a' * 64,
                   project_id='project-test', credentials_file='/nonexistent/credentials.json', results_bucket_id='bucket-test',
                   results_bucket_name='results-test', s3_endpoint='https://storage.example', s3_access_key_id='AKTEST',
                   s3_secret_access_key=SECRETS[0], poll_min_interval_seconds=0, cancel_enabled=True,
                   session_allowance_usd=5.0, global_allowance_usd=20.0, reservation_usd_per_gpu_hour=6.0,
                   runner_max_seconds=300, startup_grace_seconds=900)
    return replace(base, **kw)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def fake_results(config, status='succeeded', failure=None, gpu='NVIDIA H100 80GB HBM3', smoke=False, **over):
    t, m = config['training'], config['measurement']
    n = m['measured_updates']
    data = {'schema_version': R.RESULTS_SCHEMA_VERSION, 'benchmark_code_version': R.BENCHMARK_CODE_VERSION, 'status': status,
            'failure': failure, 'smoke_test': smoke, 'config_fingerprint': R.fingerprint(config), 'effective_config': config,
            'software': {'python': '3.11', 'torch': '2.5.1', 'cuda': '12.4', 'platform': 'Linux'},
            'hardware': {'device_type': 'cpu' if smoke else 'cuda', 'gpu_name': None if smoke else gpu, 'gpu_count_visible': 1,
                         'gpu_count_used': 0 if smoke else 1, 'total_memory_bytes': 80 * 2 ** 30, 'compute_capability': '9.0',
                         'multiprocessor_count': 132, 'bf16_supported': True},
            'memory': {'peak_allocated_bytes': 9 * 2 ** 30, 'peak_reserved_bytes': 10 * 2 ** 30},
            'durations': {'startup_seconds': 12.0, 'runner_wall_seconds': 20.0}}
    if status == 'succeeded':
        data.update(counts={'warmup_updates': m['warmup_updates'], 'measured_updates': n, 'microbatches_per_update':
                            t['gradient_accumulation_steps'], 'microbatch_size': t['microbatch_size'],
                            'sequence_length': config['input']['sequence_length']},
                    timing={'measured_training_seconds': 0.2 * n, 'mean_update_seconds': 0.2, 'median_update_seconds': 0.2,
                            'min_update_seconds': 0.19, 'max_update_seconds': 0.21, 'update_seconds': [0.2] * n},
                    throughput={'tokens_per_second': 1000.0, 'sequences_per_second': 2.0}, final_loss=10.9)
    data.update(over)
    return data


# -- registry / config ---------------------------------------------------------------------------
class RegistryTests(unittest.TestCase):
    def bad(self, mutate, field=None):
        config = copy.deepcopy(EXAMPLE)
        mutate(config)
        with self.assertRaises(R.ConfigError) as ctx:
            R.validate(config)
        if field:
            self.assertIn(field, [e['field'] for e in ctx.exception.errors], ctx.exception.errors)
        return ctx.exception

    def test_example_is_valid(self):
        config = R.validate(EXAMPLE)
        self.assertEqual(R.parameter_count(config['model']), 81_912_576)
        self.assertEqual(R.fingerprint(config), R.fingerprint({**config, 'hardware': {'requested_gpu': 'x', 'gpu_count': 1}}))

    def test_rejects_unknown_and_user_controlled_fields(self):
        for key in ('command', 'image', 'pip', 'script_path'):
            self.bad(lambda c, k=key: c.__setitem__(k, 'x'))
        self.bad(lambda c: c['training'].__setitem__('dependencies', ['numpy']))
        self.bad(lambda c: c['model'].__setitem__('trust_remote_code', True))

    def test_rejects_bad_types_and_non_finite(self):
        self.bad(lambda c: c['training'].__setitem__('microbatch_size', '4'), 'training.microbatch_size')
        self.bad(lambda c: c['training'].__setitem__('microbatch_size', True), 'training.microbatch_size')
        self.bad(lambda c: c['training']['optimizer_settings'].__setitem__('lr', float('nan')))
        self.bad(lambda c: c['training']['optimizer_settings'].__setitem__('lr', float('inf')))
        self.bad(lambda c: c['model'].__setitem__('attention_dropout', 1.5))

    def test_rejects_unsupported_choices(self):
        self.bad(lambda c: c.__setitem__('architecture', 'llama'))
        self.bad(lambda c: c['model'].__setitem__('hf_model_id', 'meta-llama/Llama-3-8B'))
        self.bad(lambda c: c['model'].__setitem__('hf_revision', 'main'))
        self.bad(lambda c: c['model'].__setitem__('weights', 'pretrained'))
        self.bad(lambda c: c['training'].__setitem__('precision', 'fp16'))
        self.bad(lambda c: c['training'].__setitem__('mode', 'lora'))
        self.bad(lambda c: c['input'].__setitem__('source', 'dataset'))
        self.bad(lambda c: c['hardware'].__setitem__('gpu_count', 2))

    def test_shape_rules_and_limits(self):
        def custom(c, **kw):
            c['model'].update(source='custom', hf_model_id=None, **kw)
        self.bad(lambda c: custom(c, num_attention_heads=7))
        self.bad(lambda c: c['input'].__setitem__('sequence_length', 2048))  # > 1024 positions
        self.bad(lambda c: c['model'].__setitem__('num_layers', 7))  # HF dims must match the allowlist
        self.bad(lambda c: custom(c, num_layers=49))
        self.bad(lambda c: c['training'].__setitem__('microbatch_size', 65))
        self.bad(lambda c: (custom(c, max_position_embeddings=2048), c['training'].__setitem__('microbatch_size', 33),
                            c['input'].__setitem__('sequence_length', 2048)))  # 67,584 tokens > 65,536 per microbatch
        self.bad(lambda c: (c['training'].__setitem__('gradient_accumulation_steps', 32),
                            c['measurement'].__setitem__('measured_updates', 50)))  # total work cap
        self.bad(lambda c: custom(c, num_layers=48, hidden_size=2048, intermediate_size=8192, num_attention_heads=32))  # >1e9 params

    def test_configurable_limits(self):
        with self.assertRaises(R.ConfigError):
            R.validate(EXAMPLE, limits={**R.LIMITS, 'measured_updates': 5})


# -- API: config review, package, results ----------------------------------------------------------
class BenchApiBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = f'sqlite:///{self.tmp.name}/bench.sqlite3'
        self.clock = Clock()
        self.policy = make_policy(self.db)
        self.client_fake = FakeJobsClient()
        self.results_fake = FakeResultStore()
        self.svc = JobService(Store(self.db), self.policy, self.client_fake, self.results_fake, clock=self.clock)
        self.env = mock.patch.dict(os.environ, {'BENCH_DATABASE_URL': self.db})
        self.env.start()
        self.c = TestClient(create_app(AnalysisStore(UploadCache(self.tmp.name + '/cache')), bench=self.svc))
        self.aid = self.c.post('/api/analysis').json()['analysis_id']
        ext = self.c.post(f'/api/analysis/{self.aid}/examples/gpt2_supported').json()['extraction']
        self.values = {k: f['value'] for k, f in ext['fields'].items() if f['value'] is not None}
        self.values.update(self.c.get(f'/api/hardware/options/{H100}').json()['values'])

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def url(self, path):
        return f'/api/analysis/{self.aid}/benchmark{path}'

    def review(self, target='local', overrides=None, values=None):
        r = self.c.post(self.url('/config'), json={'values': values or self.values, 'target': target, 'overrides': overrides or {}})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()


class ConfigApiTests(BenchApiBase):
    def test_review_has_provenance_defaults_and_assumptions(self):
        r = self.review()
        self.assertTrue(r['ready'])
        self.assertEqual(r['provenance']['model.num_layers'], 'detected')
        self.assertEqual(r['provenance']['measurement.measured_updates'], 'default')
        self.assertEqual(r['provenance']['input.source'], 'assumption')
        self.assertEqual(r['config']['input']['source'], 'synthetic')
        self.assertIn('data loading', r['assumption_notes']['input.source'])
        self.assertEqual(r['fingerprint'], R.fingerprint(r['config']))
        self.assertNotIn('image', json.dumps(r['config']))

    def test_overrides_kept_when_valid_and_dropped_when_not(self):
        r = self.review(overrides={'measurement.measured_updates': 5, 'input.sequence_length': 99999, 'model.num_layers': 2})
        self.assertEqual(r['config']['measurement']['measured_updates'], 5)
        self.assertEqual(r['provenance']['measurement.measured_updates'], 'user')
        self.assertEqual({d['field'] for d in r['dropped_overrides']}, {'input.sequence_length', 'model.num_layers'})
        # Regenerating after a workload change keeps the still-valid choice.
        r2 = self.review(overrides={'measurement.measured_updates': 5}, values={**self.values, 'training.microbatch_size': 4})
        self.assertEqual(r2['config']['measurement']['measured_updates'], 5)
        self.assertEqual(r2['config']['training']['microbatch_size'], 4)

    def test_missing_fields_are_listed(self):
        values = {k: v for k, v in self.values.items() if k != 'data.sequence_length'}
        r = self.review(values=values)
        self.assertFalse(r['ready'])
        self.assertIsNone(r['config'])
        self.assertIn('input.sequence_length', [m['field'] for m in r['missing']])

    def test_nebius_hardware_must_be_allowed_and_fit(self):
        self.assertTrue(self.review('nebius')['hardware']['valid'])
        catalog = {**self.values, 'hardware.gpu_id': 'catalog:a100-80gb', 'hardware.source': 'catalog'}
        r = self.review('nebius', values=catalog)
        self.assertFalse(r['ready'])
        self.assertFalse(r['hardware']['valid'])
        self.assertTrue(r['hardware']['alternatives'])
        big = {**self.values, 'training.microbatch_size': 64, 'data.sequence_length': 1024}
        r = self.review('nebius', values=big, overrides={})
        self.assertFalse(r['ready'])  # B*S cap or memory: either way it must not be submittable
        policy = replace(self.policy, allowed_platforms={'gpu-l40s-a': '1gpu-8vcpu-32gb'})
        self.svc.policy = policy
        r = self.review('nebius')
        self.assertFalse(r['hardware']['valid'])
        self.assertEqual([a['platform'] for a in r['hardware']['alternatives']], ['gpu-l40s-a'])

    def test_memory_safety_margin_blocks_cloud(self):
        self.svc.policy = replace(self.policy, memory_safety_fraction=0.9)  # 8 GiB usable of 80
        r = self.review('nebius')
        self.assertFalse(r['memory']['fits'])
        self.assertFalse(r['ready'])
        self.assertTrue(r['blockers'])

    def test_package_contains_trusted_runner_and_config_only(self):
        r = self.c.post(self.url('/package'), json={'values': self.values, 'overrides': {'measurement.seed': 7}})
        self.assertEqual(r.status_code, 200, r.text)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = set(z.namelist())
        root = 'compute-cast-benchmark/'
        self.assertEqual(names, {root + n for n in ('benchmark.py', 'model_registry.py', 'models/__init__.py', 'models/decoder.py',
                                                    'requirements.txt', 'README.md', 'benchmark_config.json')})
        self.assertEqual(z.read(root + 'benchmark.py'), (package_dir() / 'benchmark.py').read_bytes())
        config = json.loads(z.read(root + 'benchmark_config.json'))
        self.assertEqual(config['measurement']['seed'], 7)
        R.validate(config)
        blob = b''.join(z.read(n) for n in names)
        for secret in SECRETS:
            self.assertNotIn(secret.encode(), blob)
        self.assertIn(b'torch==', z.read(root + 'requirements.txt'))

    def test_incomplete_package_is_refused(self):
        values = {k: v for k, v in self.values.items() if k != 'training.microbatch_size'}
        self.assertEqual(self.c.post(self.url('/package'), json={'values': values}).status_code, 422)


class ResultsTests(BenchApiBase):
    def post(self, data, **body):
        text = data if isinstance(data, str) else json.dumps(data)
        return self.c.post(self.url('/results'), json={'values': self.values, 'text': text, **body})

    def test_matching_result_is_compared_and_calibrated(self):
        config = self.review()['config']
        r = self.post(fake_results(config))
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertTrue(d['matches_current_config'])
        self.assertTrue(d['comparison']['compatible_for_calibration'])
        step = d['comparison']['rows'][0]
        self.assertAlmostEqual(step['ratio'], 0.2 / step['predicted'])
        self.assertIsNotNone(d['full_run_projection'])
        self.assertIn('Estimate', d['full_run_projection']['label'])
        self.assertEqual(d['result']['hardware']['gpu_name'], 'NVIDIA H100 80GB HBM3')
        state = self.c.get(f'/api/analysis/{self.aid}').json()
        self.assertEqual(len(state['benchmarks']), 1)

    def test_different_config_or_gpu_is_not_calibrated(self):
        config = self.review(overrides={'measurement.seed': 3})['config']
        d = self.post(fake_results(config)).json()
        self.assertFalse(d['matches_current_config'])
        self.assertFalse(d['comparison']['compatible_for_calibration'])
        self.assertTrue(all(row['ratio'] is None for row in d['comparison']['rows']))
        config = self.review()['config']
        d = self.post(fake_results(config, gpu='NVIDIA L40S')).json()
        self.assertFalse(d['comparison']['compatible_for_calibration'])
        self.assertIsNone(d['full_run_projection'])

    def test_smoke_result_is_labelled(self):
        config = self.review()['config']
        d = self.post(fake_results(config, smoke=True, memory={'peak_allocated_bytes': None, 'peak_reserved_bytes': None})).json()
        self.assertTrue(d['result']['smoke_test'])
        self.assertFalse(d['comparison']['compatible_for_calibration'])

    def test_untrusted_results_are_validated(self):
        config = self.review()['config']
        good = fake_results(config)
        cases = [
            'not json', '[]', json.dumps({**good, 'schema_version': '9'}),
            json.dumps(good).replace('"final_loss": 10.9', '"final_loss": NaN'),
            json.dumps({**good, 'config_fingerprint': '0' * 64}),
            json.dumps({**good, 'counts': {**good['counts'], 'measured_updates': 3}}),
            json.dumps({**good, 'timing': {**good['timing'], 'median_update_seconds': -1}}),
            json.dumps({**good, 'hardware': {**good['hardware'], 'device_type': 'cpu'}}),  # non-smoke must be CUDA
            json.dumps({**good, 'hardware': {**good['hardware'], 'gpu_count_used': 2}}),
            json.dumps({**good, 'effective_config': {**config, 'image': 'evil'}}),
            json.dumps({**good, 'status': 'failed', 'failure': {'type': 'whatever'}}),
        ]
        for text in cases:
            self.assertEqual(self.post(text).status_code, 422, text[:80])
        self.assertEqual(self.post('x' * (RS.MAX_RESULT_BYTES + 1)).status_code, 422)

    def test_failed_runner_result_is_accepted_as_failure(self):
        config = self.review()['config']
        bad = fake_results(config, status='failed', failure={'type': 'out_of_memory', 'message': 'CUDA out of memory'})
        d = self.post(bad).json()
        self.assertEqual(d['result']['failure']['type'], 'out_of_memory')


# -- auth -----------------------------------------------------------------------------------------
class AuthTests(BenchApiBase):
    def login(self, password=PASSWORD, name='Ada', ip='1.2.3.4'):
        return self.c.post('/api/bench/session', json={'display_name': name, 'password': password},
                           headers={'X-Forwarded-For': ip})

    def test_login_session_and_token_checks(self):
        r = self.login()
        self.assertEqual(r.status_code, 200, r.text)
        token = r.json()['token']
        self.assertNotIn(PASSWORD, r.text)
        ok = self.c.get('/api/bench/session', headers={'Authorization': f'Bearer {token}'})
        self.assertEqual(ok.json()['display_name'], 'Ada')
        payload, sig = token.split('.')
        for bad in (f'{payload}.{sig[:-2]}xx', 'garbage', ''):
            self.assertEqual(self.c.get('/api/bench/session', headers={'Authorization': f'Bearer {bad}'}).status_code, 401)
        self.assertEqual(self.c.get('/api/bench/session').status_code, 401)

    def test_wrong_password_and_rate_limit(self):
        for _ in range(self.policy.login_max_attempts):
            self.assertEqual(self.login('wrong').status_code, 401)
        self.assertEqual(self.login().status_code, 429)  # even the right password while limited
        self.assertEqual(self.login(ip='5.6.7.8').status_code, 200)

    def test_expired_session(self):
        token = self.login().json()['token']
        with mock.patch('time.time', return_value=10 ** 11):
            self.assertEqual(self.c.get('/api/bench/session', headers={'Authorization': f'Bearer {token}'}).status_code, 401)

    def test_unconfigured_server_refuses_login(self):
        self.svc.policy = replace(self.policy, demo_password=None)
        self.assertEqual(self.login().status_code, 503)

    def test_display_name_required(self):
        self.assertEqual(self.login(name='  ').status_code, 422)

    def test_status_never_exposes_secrets(self):
        text = self.c.get('/api/bench/status').text
        for secret in SECRETS:
            self.assertNotIn(secret, text)
        self.assertNotIn('credentials', text.lower().replace('missing_configuration', ''))
        self.assertNotIn(SECRETS[0], repr(self.policy))


# -- jobs (mocked Nebius) ---------------------------------------------------------------------------
class JobTests(BenchApiBase):
    def setUp(self):
        super().setUp()
        self.token = self.c.post('/api/bench/session', json={'display_name': 'Ada', 'password': PASSWORD}).json()['token']
        self.h = {'Authorization': f'Bearer {self.token}'}

    def submit(self, key='key-00000001', headers=None, **kw):
        body = {'analysis_id': self.aid, 'values': self.values, 'idempotency_key': key, 'confirm_charges': True, **kw}
        return self.c.post('/api/bench/jobs', json=body, headers=headers or self.h)

    def poll(self, job_id, headers=None):
        r = self.c.get(f'/api/bench/jobs/{job_id}', headers=headers or self.h)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_full_lifecycle_succeeds_and_releases_reservation(self):
        r = self.submit()
        self.assertEqual(r.status_code, 202, r.text)
        job = r.json()
        self.assertEqual(len(self.client_fake.requests), 1)  # dispatched by the background task
        req = self.client_fake.requests[0]
        self.assertEqual((req.platform, req.preset), ('gpu-h100-sxm', '1gpu-16vcpu-200gb'))
        self.assertEqual(req.results_source_path, f'compute-cast-benchmarks/{job["id"]}')
        self.assertEqual(req.platform_timeout_hours, 1)
        for secret in SECRETS:
            self.assertNotIn(secret.encode(), req.config_bytes)
        config = json.loads(req.config_bytes)
        states = [self.poll(job['id'])['state'] for _ in range(3)]
        self.assertEqual(states, ['provisioning', 'provisioning', 'running'])
        self.results_fake.put(req.results_source_path, fake_results(config))
        done = self.poll(job['id'])
        self.assertEqual(done['state'], 'succeeded', done)
        self.assertEqual(done['results']['status'], 'succeeded')
        self.assertFalse(done['resumable'])
        self.assertIn('not a Nebius bill', done['cost_label'])
        allowance = self.svc.store.allowance('global')
        self.assertEqual(allowance['reserved_usd'], 0)
        self.assertLessEqual(allowance['used_usd'], job['reserved_usd'])
        compared = self.c.post(self.url('/results'), json={'values': self.values, 'job_id': job['id']}, headers=self.h).json()
        self.assertTrue(compared['comparison']['compatible_for_calibration'])
        self.assertTrue(compared['source'].startswith('Nebius job'))

    def test_idempotent_resubmit_and_single_active_job(self):
        first = self.submit().json()
        again = self.submit()
        self.assertEqual(again.json()['id'], first['id'])
        self.assertEqual(len(self.client_fake.requests), 1)
        other = self.submit('key-00000002')
        self.assertEqual(other.status_code, 409)
        other_session = self.c.post('/api/bench/session', json={'display_name': 'Bob', 'password': PASSWORD}).json()['token']
        self.assertEqual(self.submit('key-00000003', headers={'Authorization': f'Bearer {other_session}'}).status_code, 409)

    def test_ownership(self):
        job = self.submit().json()
        other = self.c.post('/api/bench/session', json={'display_name': 'Bob', 'password': PASSWORD}).json()['token']
        self.assertEqual(self.c.get(f'/api/bench/jobs/{job["id"]}', headers={'Authorization': f'Bearer {other}'}).status_code, 404)
        self.assertEqual(self.c.get('/api/bench/jobs', headers={'Authorization': f'Bearer {other}'}).json()['jobs'], [])
        self.assertEqual(self.c.get(f'/api/bench/jobs/{job["id"]}').status_code, 401)

    def test_allowances(self):
        self.svc.policy = replace(self.policy, session_allowance_usd=0.5)
        token = self.c.post('/api/bench/session', json={'display_name': 'Cy', 'password': PASSWORD}).json()['token']
        r = self.submit(headers={'Authorization': f'Bearer {token}'})
        self.assertEqual(r.status_code, 402)
        self.assertIn('session', r.json()['detail']['message'])
        self.svc.policy = self.policy
        self.svc.store.ensure_allowance('global', 0.1)
        self.assertEqual(self.submit('key-00000009').status_code, 402)
        self.assertEqual(self.svc.store.allowance('global')['reserved_usd'], 0)

    def test_kill_switch_confirmation_and_configuration(self):
        self.assertEqual(self.submit(confirm_charges=False).status_code, 400)
        self.svc.policy = replace(self.policy, cloud_enabled=False)
        self.assertEqual(self.submit().status_code, 503)
        self.svc.policy = replace(self.policy, backend='sdk', image=None)
        r = self.submit()
        self.assertEqual(r.status_code, 503)
        self.assertIn('BENCH_IMAGE', r.json()['detail']['missing'])
        self.assertEqual(self.client_fake.requests, [])

    def test_default_image_and_digest_warning(self):
        from app.services.benchmarks.policy import DEFAULT_IMAGE
        with mock.patch.dict(os.environ, {'BENCH_IMAGE': ''}):
            self.assertEqual(load_policy().image, 'cr.eu-north1.nebius.cloud/e00yyx5y6aa4dfp7m4/benchmark@sha256:1de6db7c41473542e4e74997f3f068bb4b2d2f4c6a50f8c947f22386943becc9')
        self.svc.policy = replace(self.policy, image=DEFAULT_IMAGE)
        status = self.c.get('/api/bench/status').json()
        self.assertEqual((status['image'], status['image_warnings']), (DEFAULT_IMAGE, []))
        self.svc.policy = replace(self.policy, image='cr.eu-north1.nebius.cloud/e00yyx5y6aa4dfp7m4/benchmark:v1')
        self.assertTrue(any('sha256' in w for w in self.c.get('/api/bench/status').json()['image_warnings']))
        pinned = 'cr.eu-north1.nebius.cloud/e00yyx5y6aa4dfp7m4/benchmark@sha256:' + 'c' * 64
        self.svc.policy = replace(self.policy, image=pinned)
        self.assertEqual(self.c.get('/api/bench/status').json()['image_warnings'], [])
        self.submit()
        self.assertEqual(self.client_fake.requests[0].image, pinned)

    def test_unready_config_is_not_submitted(self):
        values = {**self.values, 'hardware.gpu_id': 'catalog:a100-80gb'}
        r = self.submit(values=values)
        self.assertEqual(r.status_code, 422)
        self.assertTrue(r.json()['detail']['blockers'])

    def test_platform_failure_without_results(self):
        self.client_fake.script = ['PROVISIONING', 'FAILED']
        job = self.submit().json()
        self.poll(job['id'])
        done = self.poll(job['id'])
        self.assertEqual(done['state'], 'failed')
        self.assertEqual(done['error']['type'], 'results_missing')
        self.assertEqual(len(self.client_fake.requests), 1)  # never retried

    def test_runner_oom_and_runner_timeout_are_terminal(self):
        for failure, state in (('out_of_memory', 'failed'), ('timeout', 'timed_out')):
            self.client_fake.script = ['RUNNING', 'FAILED']
            job = self.submit(key=f'key-{failure}').json()
            req = self.client_fake.requests[-1]
            self.results_fake.put(req.results_source_path, fake_results(json.loads(req.config_bytes), status='failed',
                                                                        failure={'type': failure, 'message': 'x'}))
            self.poll(job['id'])
            done = self.poll(job['id'])
            self.assertEqual(done['state'], state)
            self.assertEqual(done['error']['type'], failure)
        self.assertEqual(len(self.client_fake.requests), 2)

    def test_backend_deadline_cancels_and_times_out(self):
        self.client_fake.script = ['PROVISIONING']
        job = self.submit().json()
        self.assertEqual(self.poll(job['id'])['state'], 'provisioning')
        self.clock.t += self.policy.startup_grace_seconds + self.policy.runner_max_seconds + 1
        done = self.poll(job['id'])
        self.assertEqual(done['state'], 'timed_out')
        self.assertEqual(self.client_fake.cancelled, [self.client_fake.requests[0] and 'aijob-fake-1'])
        self.assertIsNone(self.svc.store.get_job(job['id'])['active_slot'])
        self.assertEqual(self.submit('key-00000002').status_code, 202)  # slot released

    def test_unreadable_results_are_retried_then_fail(self):
        self.client_fake.script = ['RUNNING', 'COMPLETED']
        job = self.submit().json()
        req = self.client_fake.requests[0]
        self.poll(job['id'])
        self.results_fake.put(req.results_source_path, b'')  # mount still uploading
        waiting = self.poll(job['id'])
        self.assertEqual(waiting['state'], 'collecting', waiting)
        self.assertIn('Waiting for results.json', waiting['provider_message'])
        self.results_fake.put(req.results_source_path, b'{"schema_version": ')  # truncated
        self.clock.t += JOBS.RESULTS_SETTLE_SECONDS / 2
        self.assertEqual(self.poll(job['id'])['state'], 'collecting')
        self.results_fake.put(req.results_source_path, fake_results(json.loads(req.config_bytes)))
        self.clock.t += self.policy.poll_min_interval_seconds + 1
        self.assertEqual(self.poll(job['id'])['state'], 'succeeded')

        job = self.submit(key='key-00000002').json()
        req = self.client_fake.requests[1]
        self.poll(job['id'])
        self.results_fake.put(req.results_source_path, b'\xff\xfe not json')
        self.results_fake.logs[req.results_source_path] = b'... INFO building model\n... ERROR runtime_error: boom\n'
        self.assertEqual(self.poll(job['id'])['state'], 'collecting')
        self.clock.t += JOBS.RESULTS_SETTLE_SECONDS + 1
        done = self.poll(job['id'])
        self.assertEqual((done['state'], done['error']['type']), ('failed', 'invalid_results'))
        self.assertIn('UTF-8', done['error']['message'])
        self.assertIn(f'{req.results_source_path}/', done['error']['message'])
        self.assertIn('runtime_error: boom', done['error']['message'])

    def test_bucket_framing_is_unwrapped(self):
        body = json.dumps({'a': 1}).encode()
        chunked = b'%x;chunk-signature=%s\r\n%s\r\n0;chunk-signature=%s\r\n\r\n' % (len(body), b'0' * 64, body, b'0' * 64)
        unsigned = b'%x\r\n%s\r\n0\r\nx-amz-checksum-crc32:AAAAAA==\r\n\r\n' % (len(body), body)
        for raw in (body, chunked, unsigned, gzip.compress(body)):
            self.assertEqual(unwrap_object(raw), body, raw[:20])
        self.assertEqual(unwrap_object(b'12'), b'12')  # a bare JSON number is not mistaken for a chunk header
        with self.assertRaises(RS.ResultDecodeError):
            RS.parse(b'  \n')

    def test_provider_state_codes_without_results(self):
        for code, state, kind in (('TimeoutExceeded', 'timed_out', 'timeout'), ('NotEnoughResources', 'failed', 'not_enough_resources'),
                                  ('ContainerFailed', 'failed', 'results_missing')):
            self.client_fake.script = [f'{"ERROR" if code == "NotEnoughResources" else "FAILED"}:{code}']
            job = self.submit(key=f'key-{code}').json()
            done = self.poll(job['id'])
            self.assertEqual((done['state'], done['error']['type']), (state, kind))
            self.assertIn(code, done['provider_message'] or '')

    def test_template_placeholders_count_as_unset(self):
        env = {'NEBIUS_SERVICE_ACCOUNT_ID': 'serviceaccount-<id>', 'NEBIUS_PUBLIC_KEY_ID': 'publickey-<id>',
               'NEBIUS_PRIVATE_KEY_FILE': '/etc/secrets/nebius-private.pem', 'NEBIUS_CREDENTIALS_FILE': '/home/me/credentials.json',
               'NEBIUS_PROJECT_ID': 'project-<id>', 'BENCH_NEBIUS_BACKEND': ''}
        with mock.patch.dict(os.environ, env):
            p = load_policy()
        self.assertIsNone(p.service_account_id)
        self.assertIsNone(p.project_id)
        self.assertEqual((p.credentials_file, p.backend), ('/home/me/credentials.json', 'sdk'))

    def test_sdk_backend_needs_service_account_credentials(self):
        self.svc.policy = replace(self.policy, backend='sdk', credentials_file=None)
        missing = self.c.get('/api/bench/status').json()['missing_configuration']
        self.assertTrue(any('NEBIUS_SERVICE_ACCOUNT_ID' in m for m in missing))
        self.svc.policy = replace(self.policy, backend='sdk', credentials_file=None, service_account_id='serviceaccount-x',
                                  public_key_id='publickey-x', private_key_file='/secrets/private.pem')
        self.assertEqual(self.c.get('/api/bench/status').json()['missing_configuration'], [])

    def test_cancel(self):
        self.client_fake.script = ['RUNNING']
        job = self.submit().json()
        r = self.c.post(f'/api/bench/jobs/{job["id"]}/cancel', headers=self.h)
        self.assertEqual(r.json()['state'], 'cancelling')
        self.assertEqual(self.poll(job['id'])['state'], 'cancelled')
        self.svc.policy = replace(self.policy, cancel_enabled=False)
        job2 = self.submit('key-00000002').json()
        self.assertEqual(self.c.post(f'/api/bench/jobs/{job2["id"]}/cancel', headers=self.h).status_code, 403)

    def test_submission_error_is_reconciled_by_name(self):
        original = self.client_fake.submit

        def flaky(request):
            original(request)  # the create landed, but the response was lost
            raise TimeoutError('deadline exceeded')
        self.client_fake.submit = flaky
        job = self.submit().json()
        row = self.svc.store.get_job(job['id'])
        self.assertEqual(row['provider_job_id'], 'aijob-fake-1')
        self.assertEqual(row['state'], 'queued')

    def test_submission_failure_releases_reservation(self):
        self.client_fake.submit = mock.Mock(side_effect=RuntimeError('PERMISSION_DENIED'))
        job = self.submit().json()
        row = self.svc.store.get_job(job['id'])
        self.assertEqual(row['state'], 'failed')
        self.assertEqual(self.svc.store.allowance('global')['reserved_usd'], 0)
        self.assertEqual(self.svc.store.allowance('global')['used_usd'], 0)

    def test_polling_is_throttled(self):
        self.svc.policy = replace(self.policy, poll_min_interval_seconds=30)
        job = self.submit().json()
        self.poll(job['id'])
        polls = self.client_fake.jobs['aijob-fake-1']['polls']
        self.poll(job['id'])
        self.assertEqual(self.client_fake.jobs['aijob-fake-1']['polls'], polls)


class StoreConcurrencyTests(unittest.TestCase):
    def test_reservations_are_atomic_under_threads(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(f'sqlite:///{tmp}/s.sqlite3')
            store.ensure_allowance('global', 1.0)
            results = []

            def worker(i):
                job = {'id': f'j{i}', 'session_id': f's{i}', 'display_name': 'x', 'idempotency_key': f'k{i}', 'analysis_id': None,
                       'config_json': '{}', 'fingerprint': 'f', 'platform': 'p', 'preset': 'q', 'image': 'i', 'state': 'submitting',
                       'reserved_usd': 0.4, 'created_at': 0, 'deadline_at': 1, 'active_slot': None}
                try:
                    store.reserve_and_create(job, ['global'])
                    results.append('ok')
                except Exception as exc:
                    results.append(type(exc).__name__)
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            self.assertEqual(results.count('ok'), 2)  # 2 x 0.4 <= 1.0 < 3 x 0.4
            self.assertAlmostEqual(store.allowance('global')['reserved_usd'], 0.8)


# -- Nebius SDK request construction (no network) ----------------------------------------------------
try:
    import nebius  # noqa: F401
    HAVE_NEBIUS = True
except ImportError:
    HAVE_NEBIUS = False


@unittest.skipUnless(HAVE_NEBIUS, 'pip install nebius to check request construction against the official SDK')
class NebiusRequestTests(unittest.TestCase):
    def test_create_request_fields(self):
        from nebius.api.nebius.ai.v1 import JobSpec
        config = json.dumps(EXAMPLE).encode()
        req = JobRequest(job_id='abc', name='cc-bench-abc', image='cr.example/x@sha256:' + 'b' * 64, platform='gpu-h100-sxm',
                         preset='1gpu-16vcpu-200gb', config_bytes=config, runner_max_seconds=300, platform_timeout_hours=1,
                         preemptible=False, results_bucket_id='bucket-1', results_source_path='prefix/abc',
                         labels={'app': 'compute-cast'}, registry_secret_version='secretver-1', subnet_id='vpcsubnet-1', disk_gib=100)
        msg = build_create_request('project-1', req)
        spec = msg.spec
        self.assertEqual((msg.metadata.parent_id, msg.metadata.name), ('project-1', 'cc-bench-abc'))
        self.assertEqual((spec.platform, spec.preset, spec.restart_attempts), ('gpu-h100-sxm', '1gpu-16vcpu-200gb', 0))
        self.assertEqual(spec.container_command, 'python')
        self.assertIn('--max-seconds 300', spec.args)
        self.assertEqual(spec.injected_files[0].container_path, '/config/benchmark_config.json')
        self.assertEqual(spec.injected_files[0].content, config)
        self.assertEqual(spec.volumes[0].mode, JobSpec.VolumeMount.Mode.READ_WRITE)
        self.assertEqual((spec.volumes[0].source, spec.volumes[0].source_path), ('bucket-1', 'prefix/abc'))
        self.assertEqual(spec.registry_credentials.mysterybox_secret_version, 'secretver-1')
        self.assertEqual(spec.timeout.total_seconds(), 3600)
        self.assertEqual((spec.subnet_id, spec.disk.size_bytes), ('vpcsubnet-1', 100 * 2 ** 30))
        self.assertIsNone(build_create_request('p', replace(req, subnet_id=None)).spec.subnet_id or None)
        self.assertFalse(spec.environment_variables)  # nothing secret passed to the container
        with self.assertRaises(ValueError):
            build_create_request('p', replace(req, config_bytes=b'x' * (64 * 1024 + 1)))


# -- real runner (optional) ------------------------------------------------------------------------
TINY = {**EXAMPLE, 'model': {**EXAMPLE['model'], 'source': 'custom', 'hf_model_id': None, 'num_layers': 2, 'hidden_size': 64,
                             'num_attention_heads': 4, 'vocab_size': 500, 'max_position_embeddings': 64},
        'input': {'source': 'synthetic', 'sequence_length': 32},
        'training': {**EXAMPLE['training'], 'microbatch_size': 2, 'gradient_accumulation_steps': 2},
        'measurement': {'warmup_updates': 1, 'measured_updates': 2, 'seed': 0}}


def run_runner(tmp, config, *extra):
    cfg = Path(tmp) / 'benchmark_config.json'
    cfg.write_text(json.dumps(config))
    proc = subprocess.run([os.environ['BENCH_PYTHON'], str(package_dir() / 'benchmark.py'), '--config', str(cfg),
                           '--output-dir', tmp, *extra], capture_output=True, text=True, timeout=600)
    return proc, json.loads((Path(tmp) / 'results.json').read_text())


@unittest.skipUnless(os.environ.get('BENCH_PYTHON'), 'set BENCH_PYTHON to a Python with torch installed')
class RunnerCpuSmokeTests(unittest.TestCase):
    def test_cpu_smoke_run_produces_valid_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, data = run_runner(tmp, TINY, '--cpu-smoke')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            parsed = RS.parse(json.dumps(data))
            self.assertTrue(parsed['smoke_test'])
            self.assertEqual(data['settings']['parameter_count'], R.parameter_count(R.validate(TINY)['model']))
            self.assertTrue((Path(tmp) / 'benchmark.log').exists())

    def test_invalid_config_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, data = run_runner(tmp, {**TINY, 'command': 'rm -rf /'}, '--cpu-smoke')
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(data['failure']['type'], 'invalid_config')

    def test_timeout_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            big = {**TINY, 'measurement': {'warmup_updates': 0, 'measured_updates': 50, 'seed': 0},
                   'model': {**TINY['model'], 'num_layers': 12, 'hidden_size': 512, 'num_attention_heads': 8,
                             'vocab_size': 32000, 'max_position_embeddings': 512},
                   'input': {'source': 'synthetic', 'sequence_length': 512},
                   'training': {**TINY['training'], 'microbatch_size': 8, 'gradient_accumulation_steps': 8}}
            proc, data = run_runner(tmp, big, '--cpu-smoke', '--max-seconds', '3')
            self.assertEqual(proc.returncode, 4, proc.stderr[-500:])
            self.assertEqual(data['failure']['type'], 'timeout')

    def test_no_gpu_without_smoke_flag(self):
        probe = subprocess.run([os.environ['BENCH_PYTHON'], '-c', 'import torch;print(torch.cuda.is_available())'],
                               capture_output=True, text=True)
        if probe.stdout.strip() == 'True':
            self.skipTest('GPU present')
        with tempfile.TemporaryDirectory() as tmp:
            proc, data = run_runner(tmp, TINY)
            self.assertEqual(proc.returncode, 5)
            self.assertEqual(data['failure']['type'], 'hardware_unsupported')


@unittest.skipUnless(os.environ.get('BENCH_PYTHON') and os.environ.get('BENCH_CUDA') == '1',
                     'set BENCH_PYTHON and BENCH_CUDA=1 on a machine with a BF16-capable NVIDIA GPU')
class RunnerCudaTests(unittest.TestCase):
    def test_cuda_run_measures_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, data = run_runner(tmp, TINY)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            parsed = RS.parse(json.dumps(data))
            self.assertEqual(parsed['hardware']['device_type'], 'cuda')
            self.assertGreater(parsed['memory']['peak_allocated_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
