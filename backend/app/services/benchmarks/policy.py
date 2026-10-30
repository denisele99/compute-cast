"""Backend-only benchmark policy from environment variables (never part of benchmark_config.json).

Secrets (BENCH_DEMO_PASSWORD, BENCH_SESSION_SECRET, Nebius service-account credentials, S3 read key)
are read here and never returned by any API, written to configs, packages or logs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def _b(name, default=False):
    return os.environ.get(name, str(default)).strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass(frozen=True)
class Policy:
    cloud_enabled: bool
    backend: str                       # "sdk" (real Nebius) or "fake" (tests/dev only, labelled in the UI)
    demo_password: str | None = field(repr=False)
    session_secret: str | None = field(repr=False)
    session_ttl_seconds: float
    login_max_attempts: int
    login_window_seconds: float
    database_url: str
    session_allowance_usd: float
    global_allowance_usd: float
    reservation_usd_per_gpu_hour: float
    runner_max_seconds: float
    startup_grace_seconds: float
    platform_timeout_hours: int
    poll_min_interval_seconds: float
    memory_safety_fraction: float
    image: str | None
    allowed_platforms: dict            # platform -> 1-GPU preset
    preemptible: bool
    project_id: str | None
    credentials_file: str | None = field(repr=False)
    results_bucket_id: str | None
    results_bucket_name: str | None
    results_prefix: str
    s3_endpoint: str | None
    s3_region: str | None
    s3_access_key_id: str | None = field(repr=False)
    s3_secret_access_key: str | None = field(repr=False)
    registry_secret_version: str | None = field(repr=False)
    cors_origins: tuple
    cancel_enabled: bool = False       # user-facing cancel; enable after live verification

    def missing_for_cloud(self):
        """Names (never values) of settings required before cloud jobs can be submitted."""
        need = {'BENCH_DEMO_PASSWORD': self.demo_password, 'BENCH_SESSION_SECRET': self.session_secret,
                'BENCH_DATABASE_URL': os.environ.get('BENCH_DATABASE_URL')}
        if self.backend == 'sdk':
            need.update({'BENCH_IMAGE': self.image, 'NEBIUS_PROJECT_ID': self.project_id,
                         'NEBIUS_CREDENTIALS_FILE': self.credentials_file, 'NEBIUS_RESULTS_BUCKET_ID': self.results_bucket_id,
                         'NEBIUS_RESULTS_BUCKET_NAME': self.results_bucket_name, 'NEBIUS_S3_ENDPOINT': self.s3_endpoint,
                         'NEBIUS_S3_ACCESS_KEY_ID': self.s3_access_key_id, 'NEBIUS_S3_SECRET_ACCESS_KEY': self.s3_secret_access_key})
        return [k for k, v in need.items() if not v]


def parse_platforms(text):
    out = {}
    for item in filter(None, (s.strip() for s in text.split(','))):
        platform, _, preset = item.partition(':')
        if platform and preset:
            out[platform] = preset
    return out


def load_policy() -> Policy:
    env = os.environ.get
    return Policy(
        cloud_enabled=_b('BENCH_CLOUD_ENABLED'), backend=env('BENCH_NEBIUS_BACKEND', 'sdk'),
        demo_password=env('BENCH_DEMO_PASSWORD') or None, session_secret=env('BENCH_SESSION_SECRET') or None,
        session_ttl_seconds=_f('BENCH_SESSION_TTL_MINUTES', 60) * 60,
        login_max_attempts=int(_f('BENCH_LOGIN_MAX_ATTEMPTS', 5)), login_window_seconds=_f('BENCH_LOGIN_WINDOW_SECONDS', 900),
        database_url=env('BENCH_DATABASE_URL') or 'sqlite:///./benchmark_dev.sqlite3',
        session_allowance_usd=_f('BENCH_SESSION_ALLOWANCE_USD', 5.0), global_allowance_usd=_f('BENCH_GLOBAL_ALLOWANCE_USD', 20.0),
        reservation_usd_per_gpu_hour=_f('BENCH_RESERVATION_USD_PER_GPU_HOUR', 6.0),
        runner_max_seconds=_f('BENCH_RUNNER_MAX_SECONDS', 300), startup_grace_seconds=_f('BENCH_STARTUP_GRACE_SECONDS', 900),
        platform_timeout_hours=max(1, int(_f('BENCH_PLATFORM_TIMEOUT_HOURS', 1))),
        poll_min_interval_seconds=_f('BENCH_POLL_MIN_INTERVAL_SECONDS', 10), memory_safety_fraction=_f('BENCH_MEMORY_SAFETY_FRACTION', 0.10),
        image=env('BENCH_IMAGE') or None,
        allowed_platforms=parse_platforms(env('BENCH_ALLOWED_PLATFORMS', 'gpu-h100-sxm:1gpu-16vcpu-200gb,'
                                              'gpu-h200-sxm:1gpu-16vcpu-200gb,gpu-l40s-a:1gpu-8vcpu-32gb,gpu-l40s-d:1gpu-16vcpu-96gb')),
        preemptible=_b('BENCH_PREEMPTIBLE'), project_id=env('NEBIUS_PROJECT_ID') or None,
        credentials_file=env('NEBIUS_CREDENTIALS_FILE') or None, results_bucket_id=env('NEBIUS_RESULTS_BUCKET_ID') or None,
        results_bucket_name=env('NEBIUS_RESULTS_BUCKET_NAME') or None, results_prefix=env('NEBIUS_RESULTS_PREFIX', 'compute-cast-benchmarks'),
        s3_endpoint=env('NEBIUS_S3_ENDPOINT') or None, s3_region=env('NEBIUS_S3_REGION') or None,
        s3_access_key_id=env('NEBIUS_S3_ACCESS_KEY_ID') or None, s3_secret_access_key=env('NEBIUS_S3_SECRET_ACCESS_KEY') or None,
        registry_secret_version=env('NEBIUS_REGISTRY_SECRET_VERSION') or None,
        cancel_enabled=_b('BENCH_CANCEL_ENABLED'),
        cors_origins=tuple(o.strip() for o in env('CORS_ALLOWED_ORIGINS', '').split(',') if o.strip()),
    )
