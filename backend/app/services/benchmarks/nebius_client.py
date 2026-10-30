"""Nebius AI Cloud Serverless Jobs integration behind a small, testable interface.

Built against the official Python SDK ``nebius`` 0.6.31 (``nebius.api.nebius.ai.v1.JobService``):
``create(CreateJobRequest)``, ``get(GetJobRequest)``, ``get_by_name(GetJobByNameRequest)`` and
``cancel(CancelJobRequest)``. Field names below are those of that SDK's generated messages:
JobSpec.image / container_command / args / platform / preset / timeout / restart_attempts /
preemptible / pricing_model / injected_files (FileInjection: container_path, content <= 64 KiB) /
volumes (VolumeMount: source bucket id, source_path, container_path, mode READ_WRITE) /
registry_credentials.mysterybox_secret_version.

Checked against docs.nebius.com (Serverless AI: "Managing jobs", "Lifecycle and statuses", "Automatic
job restarts", "Pricing and quotas"): container_command is the entrypoint and args its arguments;
injected files are read-only, <= 64 KiB, at an absolute path; timeout 1-168 h (default 24 h); exceeded
timeouts end FAILED with state_details.code TimeoutExceeded; restart_attempts 0 runs once; no capacity
within 30 min ends ERROR/NotEnoughResources; cancel immediately stops the VM; jobs are billed only while
running; subnet_id is required only when the project has several subnets.

Credentials: a dedicated service account, authenticated as the docs describe for server-to-server use
(private key + public key id + service account id), or with a credentials JSON file. Nothing
credential-bearing is placed in the job spec except an optional SecretStash (MysteryBox) secret
*reference* for private-registry pulls; the container receives no environment secrets.

Still needs a live run: create().wait().resource_id, bucket source_path -> object key layout and write
timing, and what started_at/finished_at cover relative to billing.
"""
from __future__ import annotations

import gzip
import json
import re
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
CALL_LIMIT = {'timeout': 60, 'auth_timeout': 60}  # per SDK call, so a network problem fails instead of hanging


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
    subnet_id: str | None = None
    disk_gib: int = 100


@dataclass
class ProviderStatus:
    job_id: str
    state: str
    message: str
    started_at: float | None
    finished_at: float | None
    code: str = ''          # state_details.code: StartFailed, ContainerFailed, TimeoutExceeded, NotEnoughResources


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
    from nebius.api.nebius.compute.v1 import DiskSpec
    if len(r.config_bytes) > 64 * 1024:
        raise ValueError('benchmark_config.json exceeds the 64 KiB FileInjection limit.')
    spec = dict(
        image=r.image, container_command='python', args=runner_args(r.runner_max_seconds),
        platform=r.platform, preset=r.preset, timeout=timedelta(hours=r.platform_timeout_hours),
        restart_attempts=0, preemptible=r.preemptible,
        disk=JobSpec.DiskSpec(type=DiskSpec.DiskType.NETWORK_SSD, size_bytes=r.disk_gib * 2 ** 30),
        pricing_model=PricingModelSpec(follows_spot_price=FollowsSpotPriceSpec()) if r.preemptible
        else PricingModelSpec(on_demand=OnDemandSpec()),
        injected_files=[JobSpec.FileInjection(container_path=CONFIG_PATH, content=r.config_bytes)],
        volumes=[JobSpec.VolumeMount(source=r.results_bucket_id, source_path=r.results_source_path,
                                     container_path=RESULTS_DIR, mode=JobSpec.VolumeMount.Mode.READ_WRITE)],
    )
    if r.subnet_id:
        spec['subnet_id'] = r.subnet_id
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
            p = self.policy
            if p.credentials_file:  # one-file credentials take priority when set
                self._sdk = SDK(credentials_file_name=p.credentials_file, parent_id=p.project_id,
                                user_agent_prefix='compute-cast-benchmark/0.1')
            else:
                self._sdk = SDK(service_account_id=p.service_account_id, service_account_public_key_id=p.public_key_id,
                                service_account_private_key_file_name=p.private_key_file, parent_id=p.project_id,
                                user_agent_prefix='compute-cast-benchmark/0.1')
        return JobServiceClient(self._sdk)

    def submit(self, request):
        operation = self._service().create(build_create_request(self.policy.project_id, request), **CALL_LIMIT).wait()
        return operation.resource_id

    def get(self, provider_job_id):
        from nebius.api.nebius.ai.v1 import GetJobRequest
        job = self._service().get(GetJobRequest(id=provider_job_id), **CALL_LIMIT).wait()
        details = job.status.state_details
        return ProviderStatus(provider_job_id, job.status.state.name, (details.message or '').strip(),
                              _ts(job.status.started_at), _ts(job.status.finished_at), code=details.code or '')

    def find_by_name(self, name):
        from nebius.aio.service_error import RequestError
        from nebius.api.nebius.ai.v1 import GetJobByNameRequest
        try:
            return self._service().get_by_name(GetJobByNameRequest(parent_id=self.policy.project_id, name=name), **CALL_LIMIT).wait().metadata.id
        except RequestError as exc:
            if 'NOT_FOUND' in str(exc.status):
                return None
            raise

    def cancel(self, provider_job_id):
        from nebius.api.nebius.ai.v1 import CancelJobRequest
        self._service().cancel(CancelJobRequest(id=provider_job_id), **CALL_LIMIT).wait()


_CHUNK_HEADER = re.compile(rb'^([0-9a-fA-F]+)(;[^\r\n]*)?\r\n')


def unwrap_object(data: bytes, content_encoding: str = '') -> bytes:
    """Undo transport framing an S3-compatible store may leave in the object body.

    Recent boto3/AWS SDKs upload with ``aws-chunked`` encoding (``<hex>;chunk-signature=...\\r\\n<data>\\r\\n``) and
    some non-AWS stores keep that framing in the stored object; objects may also be stored gzip-compressed.
    Plain JSON is returned unchanged.
    """
    if data[:2] == b'\x1f\x8b' or 'gzip' in content_encoding:
        try:
            data = gzip.decompress(data)[:256 * 1024 + 1]
        except (OSError, EOFError):
            return data
    if not _CHUNK_HEADER.match(data):
        return data
    out, rest = bytearray(), data
    while (m := _CHUNK_HEADER.match(rest)):
        size = int(m.group(1), 16)
        if size == 0:
            return bytes(out)
        start = m.end()
        out += rest[start:start + size]
        rest = rest[start + size:].removeprefix(b'\r\n')
    return bytes(out) if not rest.strip() else data


class S3ResultStore:
    """Reads <prefix>/<job id>/results.json from the results bucket with a read-only S3 key (boto3)."""

    def __init__(self, policy):
        self.policy = policy

    def fetch_results(self, source_path):
        data = self._get(f'{source_path}/results.json', 256 * 1024 + 1)
        return None if data is None else unwrap_object(*data)

    def fetch_log_tail(self, source_path, limit=600):
        """Last bytes of benchmark.log (diagnostics only; shown as text, never parsed or executed)."""
        data = self._get(f'{source_path}/benchmark.log', 64 * 1024)
        return None if data is None else unwrap_object(*data)[-limit:]

    def _get(self, key, limit):
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError
        # Non-AWS S3 endpoints: only send/validate checksums when the API requires them.
        config = Config(request_checksum_calculation='when_required', response_checksum_validation='when_required',
                        retries={'max_attempts': 3, 'mode': 'standard'})
        s3 = boto3.client('s3', endpoint_url=self.policy.s3_endpoint, region_name=self.policy.s3_region, config=config,
                          aws_access_key_id=self.policy.s3_access_key_id, aws_secret_access_key=self.policy.s3_secret_access_key)
        try:
            obj = s3.get_object(Bucket=self.policy.results_bucket_name, Key=key)
            return obj['Body'].read(limit), obj.get('ContentEncoding') or ''
        except ClientError as exc:
            if exc.response.get('Error', {}).get('Code') in ('NoSuchKey', '404'):
                return None
            raise


class FakeJobsClient:
    """Deterministic in-process provider for tests and offline demos. Never contacts Nebius.

    ``script`` maps poll counts to provider states (optionally ``STATE:code``); the result written on completion is supplied by
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
        state, _, code = self.script[min(job['polls'], len(self.script) - 1)].partition(':')
        job['polls'] += 1
        now = datetime.now(timezone.utc).timestamp()
        if state == 'RUNNING' and job['started'] is None:
            job['started'] = now - 1
        finished = now if state in PROVIDER_TERMINAL else None
        return ProviderStatus(provider_job_id, state, '', job['started'] or (now - 2 if finished else None), finished, code=code)

    def find_by_name(self, name):
        return next((pid for pid, j in self.jobs.items() if j['name'] == name), None)

    def cancel(self, provider_job_id):
        self.cancelled.append(provider_job_id)


class FakeResultStore:
    def __init__(self):
        self.objects, self.logs = {}, {}

    def put(self, source_path, data):
        self.objects[source_path] = data if isinstance(data, bytes) else json.dumps(data).encode()

    def fetch_results(self, source_path):
        return self.objects.get(source_path)

    def fetch_log_tail(self, source_path, limit=600):
        log = self.logs.get(source_path)
        return None if log is None else log[-limit:]
