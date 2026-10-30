"""Nebius AI Cloud Serverless Jobs integration behind a small, testable interface.

Built against the official Python SDK ``nebius`` 0.6.31 (``nebius.api.nebius.ai.v1.JobService``):
``create(CreateJobRequest)``, ``get(GetJobRequest)``, ``get_by_name(GetJobByNameRequest)`` and
``cancel(CancelJobRequest)``. Field names below are those of that SDK's generated messages:
JobSpec.image / container_command / args / platform / preset / timeout / restart_attempts /
preemptible / pricing_model / injected_files (FileInjection: container_path, content <= 64 KiB) /
volumes (VolumeMount: source bucket id, source_path, container_path, mode READ_WRITE) /
registry_credentials.mysterybox_secret_version.

Credentials: a dedicated service account authenticated with its credentials JSON file
(``SDK(credentials_file_name=NEBIUS_CREDENTIALS_FILE)``). Nothing credential-bearing is placed in the
job spec except an optional MysteryBox secret *reference* for private-registry pulls; the container
receives no environment secrets.

Live verification still required (no account was used while building this): operation/resource-id
behaviour of create(), args tokenisation, bucket mount permissions and object paths, and state timing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

CONFIG_PATH = '/config/benchmark_config.json'
RESULTS_DIR = '/results'

# Provider job state -> Compute Cast state (results decide succeeded vs failed/timed_out afterwards).
STATE_MAP = {
    'STATE_UNSPECIFIED': 'queued', 'PROVISIONING': 'provisioning', 'STARTING': 'provisioning',
    'IMAGE_PULLING': 'provisioning', 'RUNNING': 'running', 'COMPLETED': 'collecting', 'FAILED': 'collecting',
    'ERROR': 'collecting', 'CANCELLING': 'cancelling', 'CANCELLED': 'cancelled', 'DELETING': 'collecting',
}
PROVIDER_TERMINAL = {'COMPLETED', 'FAILED', 'ERROR', 'CANCELLED'}


@dataclass
class JobRequest:
    job_id: str
    name: str
    image: str
    platform: str
    preset: str
    config_bytes: bytes
    runner_max_seconds: float
    platform_timeout_hours: int
    preemptible: bool
    results_bucket_id: str
    results_source_path: str
    labels: dict
    registry_secret_version: str | None = None


@dataclass
class ProviderStatus:
    job_id: str
    state: str
    message: str
    started_at: float | None
    finished_at: float | None


class JobsClient(Protocol):
    label: str

    def submit(self, request: JobRequest) -> str: ...
    def get(self, provider_job_id: str) -> ProviderStatus: ...
    def find_by_name(self, name: str) -> str | None: ...
    def cancel(self, provider_job_id: str) -> None: ...


class ResultStore(Protocol):
    def fetch_results(self, source_path: str) -> bytes | None: ...


def runner_args(max_seconds):
    return f'/app/benchmark.py --config {CONFIG_PATH} --output-dir {RESULTS_DIR} --max-seconds {int(max_seconds)}'


def build_create_request(project_id, r: JobRequest):
    """Pure construction of CreateJobRequest from the SDK's generated messages (unit-tested offline)."""
    from nebius.api.nebius.ai.v1 import CreateJobRequest, FollowsSpotPriceSpec, JobSpec, OnDemandSpec, PricingModelSpec
    from nebius.api.nebius.common.v1 import ResourceMetadata
    if len(r.config_bytes) > 64 * 1024:
        raise ValueError('benchmark_config.json exceeds the 64 KiB FileInjection limit.')
    spec = dict(
        image=r.image, container_command='python', args=runner_args(r.runner_max_seconds),
        platform=r.platform, preset=r.preset, timeout=timedelta(hours=r.platform_timeout_hours),
        restart_attempts=0, preemptible=r.preemptible,
        pricing_model=PricingModelSpec(follows_spot_price=FollowsSpotPriceSpec()) if r.preemptible
        else PricingModelSpec(on_demand=OnDemandSpec()),
        injected_files=[JobSpec.FileInjection(container_path=CONFIG_PATH, content=r.config_bytes)],
        volumes=[JobSpec.VolumeMount(source=r.results_bucket_id, source_path=r.results_source_path,
                                     container_path=RESULTS_DIR, mode=JobSpec.VolumeMount.Mode.READ_WRITE)],
    )
    if r.registry_secret_version:
        spec['registry_credentials'] = JobSpec.RegistryCredentials(mysterybox_secret_version=r.registry_secret_version)
    return CreateJobRequest(metadata=ResourceMetadata(parent_id=project_id, name=r.name, labels=r.labels), spec=JobSpec(**spec))


def _ts(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).timestamp()
    return None


class NebiusSDKClient:
    label = 'Nebius AI Cloud Serverless Jobs'

    def __init__(self, policy):
        self.policy = policy
        self._sdk = None

    def _service(self):
        from nebius.api.nebius.ai.v1 import JobServiceClient
        from nebius.sdk import SDK
        if self._sdk is None:
            self._sdk = SDK(credentials_file_name=self.policy.credentials_file, user_agent_prefix='compute-cast-benchmark/0.1')
        return JobServiceClient(self._sdk)

    def submit(self, request):
        operation = self._service().create(build_create_request(self.policy.project_id, request)).wait()
        return operation.resource_id

    def get(self, provider_job_id):
        from nebius.api.nebius.ai.v1 import GetJobRequest
        job = self._service().get(GetJobRequest(id=provider_job_id)).wait()
        details = job.status.state_details
        return ProviderStatus(provider_job_id, job.status.state.name, f'{details.code} {details.message}'.strip(),
                              _ts(job.status.started_at), _ts(job.status.finished_at))

    def find_by_name(self, name):
        from nebius.aio.service_error import RequestError
        from nebius.api.nebius.ai.v1 import GetJobByNameRequest
        try:
            return self._service().get_by_name(GetJobByNameRequest(parent_id=self.policy.project_id, name=name)).wait().metadata.id
        except RequestError as exc:
            if 'NOT_FOUND' in str(exc.status):
                return None
            raise

    def cancel(self, provider_job_id):
        from nebius.api.nebius.ai.v1 import CancelJobRequest
        self._service().cancel(CancelJobRequest(id=provider_job_id)).wait()


class S3ResultStore:
    """Reads <prefix>/<job id>/results.json from the results bucket with a read-only S3 key (boto3)."""

    def __init__(self, policy):
        self.policy = policy

    def fetch_results(self, source_path):
        import boto3
        from botocore.exceptions import ClientError
        s3 = boto3.client('s3', endpoint_url=self.policy.s3_endpoint, region_name=self.policy.s3_region,
                          aws_access_key_id=self.policy.s3_access_key_id, aws_secret_access_key=self.policy.s3_secret_access_key)
        try:
            body = s3.get_object(Bucket=self.policy.results_bucket_name, Key=f'{source_path}/results.json')['Body']
            return body.read(256 * 1024 + 1)
        except ClientError as exc:
            if exc.response.get('Error', {}).get('Code') in ('NoSuchKey', '404'):
                return None
            raise


class FakeJobsClient:
    """Deterministic in-process provider for tests and offline demos. Never contacts Nebius.

    ``script`` maps poll counts to provider states; the result written on completion is supplied by
    the test via ``results``. The UI labels jobs from this client as simulated.
    """
    label = 'SIMULATED provider (BENCH_NEBIUS_BACKEND=fake) — not a real Nebius job'

    def __init__(self, script=('PROVISIONING', 'IMAGE_PULLING', 'RUNNING', 'COMPLETED'), results=None):
        self.script, self.results = list(script), results
        self.jobs, self.requests, self.cancelled = {}, [], []

    def submit(self, request):
        if request.name in {j['name'] for j in self.jobs.values()}:
            raise RuntimeError('ALREADY_EXISTS')
        pid = f'aijob-fake-{len(self.jobs) + 1}'
        self.jobs[pid] = {'name': request.name, 'polls': 0, 'request': request, 'started': None}
        self.requests.append(request)
        return pid

    def get(self, provider_job_id):
        job = self.jobs[provider_job_id]
        if provider_job_id in self.cancelled:
            return ProviderStatus(provider_job_id, 'CANCELLED', 'cancelled', job['started'], job['started'])
        state = self.script[min(job['polls'], len(self.script) - 1)]
        job['polls'] += 1
        now = datetime.now(timezone.utc).timestamp()
        if state == 'RUNNING' and job['started'] is None:
            job['started'] = now - 1
        finished = now if state in PROVIDER_TERMINAL else None
        return ProviderStatus(provider_job_id, state, '', job['started'] or (now - 2 if finished else None), finished)

    def find_by_name(self, name):
        return next((pid for pid, j in self.jobs.items() if j['name'] == name), None)

    def cancel(self, provider_job_id):
        self.cancelled.append(provider_job_id)


class FakeResultStore:
    def __init__(self):
        self.objects = {}

    def put(self, source_path, data):
        self.objects[source_path] = data if isinstance(data, bytes) else json.dumps(data).encode()

    def fetch_results(self, source_path):
        return self.objects.get(source_path)
