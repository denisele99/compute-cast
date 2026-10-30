"""Cloud benchmark job lifecycle: reserve -> submit -> monitor -> collect -> release.

States shown to users: submitting, queued, provisioning, running, cancelling, collecting (internal,
brief), succeeded, failed, timed_out, cancelled. Jobs are never retried automatically and are not
resumable. Allowances are budget *reservations* inside this app, not Nebius billing limits.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from . import results as RS
from .nebius_client import STATE_MAP, JobRequest
from .store import TERMINAL, AllowanceExceeded, ConcurrencyLimit, dumps

log = logging.getLogger('compute_cast.bench')
GLOBAL_KEY = 'global'
RECONCILE_AFTER_SECONDS = 60


class JobError(Exception):
    def __init__(self, status, message, detail=None):
        self.status, self.message, self.detail = status, message, detail
        super().__init__(message)


def provider_name(job_id):
    return f'cc-bench-{job_id}'


class JobService:
    def __init__(self, store, policy, client, result_store, clock=time.time):
        self.store, self.policy, self.client, self.result_store, self.clock = store, policy, client, result_store, clock
        store.ensure_allowance(GLOBAL_KEY, policy.global_allowance_usd)

    # -- helpers ----------------------------------------------------------------------------
    def reservation_usd(self):
        hours = (self.policy.runner_max_seconds + self.policy.startup_grace_seconds) / 3600
        return round(hours * self.policy.reservation_usd_per_gpu_hour, 4)

    def keys(self, job):
        return [f"session:{job['session_id']}", GLOBAL_KEY]

    def source_path(self, job_id):
        return f'{self.policy.results_prefix}/{job_id}'

    def status(self, session=None):
        p = self.policy
        out = {'cloud_enabled': p.cloud_enabled, 'missing_configuration': p.missing_for_cloud(),
               'provider': self.client.label, 'simulated': p.backend == 'fake', 'cancel_enabled': p.cancel_enabled,
               'runner_max_seconds': p.runner_max_seconds, 'startup_grace_seconds': p.startup_grace_seconds,
               'platform_timeout_hours': p.platform_timeout_hours, 'reservation_per_job_usd': self.reservation_usd(),
               'active_jobs': len(self.store.active_jobs()), 'max_concurrent_jobs': 1,
               'allowed_platforms': p.allowed_platforms, 'global': self._allowance(GLOBAL_KEY),
               'notes': ['Allowances are reservations inside Compute Cast, not a Nebius spending cap.',
                         'Nebius budget alerts notify; they do not stop spending. No maximum bill is guaranteed.']}
        if session:
            out['session'] = self._allowance(f"session:{session['id']}")
        return out

    def _allowance(self, key):
        row = self.store.allowance(key)
        if not row:
            return None
        return {'limit_usd': row['limit_usd'], 'reserved_usd': row['reserved_usd'], 'used_estimated_usd': row['used_usd'],
                'available_usd': max(row['limit_usd'] - row['reserved_usd'] - row['used_usd'], 0.0)}

    # -- submission -------------------------------------------------------------------------
    def submit(self, session, idempotency_key, review, analysis_id=None):
        p = self.policy
        existing = self.store.find_job(session['id'], idempotency_key)
        if existing:
            return existing, False
        if not p.cloud_enabled:
            raise JobError(503, 'Cloud benchmarks are switched off on this server (BENCH_CLOUD_ENABLED).')
        missing = p.missing_for_cloud()
        if missing:
            raise JobError(503, 'Cloud benchmarks are not fully configured on this server.', {'missing': missing})
        if not review.get('ready') or not review.get('config'):
            raise JobError(422, 'The benchmark configuration is not ready.',
                           {'missing': review.get('missing'), 'errors': review.get('errors'), 'blockers': review.get('blockers')})
        hardware = review.get('hardware') or {}
        if not hardware.get('valid'):
            raise JobError(422, hardware.get('reason') or 'Select a supported Nebius GPU.', {'alternatives': hardware.get('alternatives')})
        config_bytes = json.dumps(review['config'], sort_keys=True, allow_nan=False).encode()
        now = self.clock()
        job = {'id': uuid.uuid4().hex[:20], 'session_id': session['id'], 'display_name': session['display_name'],
               'idempotency_key': idempotency_key, 'analysis_id': analysis_id, 'config_json': config_bytes.decode(),
               'fingerprint': review['fingerprint'], 'platform': hardware['platform'], 'preset': hardware['preset'],
               'image': p.image or 'unset', 'state': 'submitting', 'reserved_usd': self.reservation_usd(), 'created_at': now,
               'deadline_at': now + p.startup_grace_seconds + p.runner_max_seconds, 'active_slot': 1}
        try:
            row, created = self.store.reserve_and_create(job, self.keys(job))
        except AllowanceExceeded as exc:
            which = 'session' if str(exc).startswith('session:') else 'global'
            raise JobError(402, f'The {which} benchmark allowance cannot cover this job '
                                f'(reservation ${job["reserved_usd"]:.2f}).') from None
        except ConcurrencyLimit:
            raise JobError(409, 'Another cloud benchmark is already running; only one GPU job runs at a time.') from None
        log.info('benchmark job %s reserved %.2f USD (session %s)', row['id'], row['reserved_usd'], session['id'][:8])
        return row, created

    def dispatch(self, job_id):
        """Send the create request once. Called from a background task after the 202 response."""
        job = self.store.get_job(job_id)
        if not job or job['state'] != 'submitting' or job['provider_job_id']:
            return
        p = self.policy
        request = JobRequest(job_id=job['id'], name=provider_name(job['id']), image=p.image, platform=job['platform'],
                             preset=job['preset'], config_bytes=job['config_json'].encode(), runner_max_seconds=p.runner_max_seconds,
                             platform_timeout_hours=p.platform_timeout_hours, preemptible=p.preemptible,
                             results_bucket_id=p.results_bucket_id or 'unset', results_source_path=self.source_path(job['id']),
                             labels={'app': 'compute-cast', 'submission': job['id']}, registry_secret_version=p.registry_secret_version)
        try:
            provider_id = self.client.submit(request)
        except Exception as exc:  # no automatic retry; check whether the create landed anyway
            log.warning('submit of %s raised %s', job_id, type(exc).__name__)
            provider_id = self._lookup(job_id)
            if not provider_id:
                self._finish(job, 'failed', 0.0, error={'type': 'submission_failed', 'message': f'{type(exc).__name__}: {str(exc)[:300]}'})
                return
        self.store.update_job(job_id, provider_job_id=provider_id, state='queued', submitted_at=self.clock())

    def _lookup(self, job_id):
        try:
            return self.client.find_by_name(provider_name(job_id))
        except Exception:
            return None

    # -- monitoring -------------------------------------------------------------------------
    def refresh(self, job_id, force=False):
        job = self.store.get_job(job_id)
        if job is None or job['state'] in TERMINAL:
            return job
        now = self.clock()
        if not force and job['last_checked_at'] and now - job['last_checked_at'] < self.policy.poll_min_interval_seconds:
            return job
        self.store.update_job(job_id, last_checked_at=now)
        if not job['provider_job_id']:
            if now - job['created_at'] > RECONCILE_AFTER_SECONDS:
                provider_id = self._lookup(job_id)
                if provider_id:
                    return self.store.update_job(job_id, provider_job_id=provider_id, state='queued', submitted_at=now)
                return self._finish(job, 'failed', 0.0, error={'type': 'submission_lost', 'message': 'No provider job was created.'})
            return self.store.get_job(job_id)
        try:
            status = self.client.get(job['provider_job_id'])
        except Exception as exc:
            log.warning('status of %s failed: %s', job_id, type(exc).__name__)
            return self.store.update_job(job_id, provider_message=f'Status check failed: {type(exc).__name__}')
        mapped = STATE_MAP.get(status.state, 'running')
        fields = {'provider_state': status.state, 'provider_message': status.message[:300] or None,
                  'provider_started_at': status.started_at, 'provider_finished_at': status.finished_at}
        if mapped == 'cancelled':
            return self._finish(job, 'cancelled', self._charge(status, now), **fields)
        if mapped == 'collecting':
            return self._collect(job, status, now, fields)
        if now > job['deadline_at']:
            self._cancel_quietly(job)
            return self._finish(job, 'timed_out', self._charge(status, now), error={
                'type': 'timeout', 'message': 'Exceeded start-up grace + runner limit; cancellation requested.'}, **fields)
        if job['state'] == 'cancelling':
            mapped = 'cancelling'
        return self.store.update_job(job_id, state=mapped, **fields)

    def _collect(self, job, status, now, fields):
        try:
            raw = self.result_store.fetch_results(self.source_path(job['id']))
        except Exception as exc:
            return self.store.update_job(job['id'], provider_message=f'Results not readable yet: {type(exc).__name__}', **fields)
        charge = self._charge(status, now)
        if raw is None:
            return self._finish(job, 'failed', charge, error={'type': 'results_missing', 'message':
                                f'Provider state {status.state}; no results.json was written. {status.message}'.strip()}, **fields)
        try:
            parsed = RS.parse(raw)
        except RS.ResultError as exc:
            return self._finish(job, 'failed', charge, error={'type': 'invalid_results', 'message': str(exc)}, **fields)
        if parsed['config_fingerprint'] and parsed['config_fingerprint'] != job['fingerprint']:
            return self._finish(job, 'failed', charge, error={'type': 'invalid_results', 'message': 'Results are for a different configuration.'}, **fields)
        if parsed['status'] == 'succeeded':
            state, error = 'succeeded', None
        else:
            state = 'timed_out' if parsed['failure']['type'] == 'timeout' else 'failed'
            error = parsed['failure']
        return self._finish(job, state, charge, error=error, results=parsed, **fields)

    def _charge(self, status, now):
        """Estimated (not billed) cost from provider start/finish timestamps at the reservation rate."""
        if not status.started_at:
            return 0.0
        seconds = max((status.finished_at or now) - status.started_at, 0.0)
        return seconds / 3600 * self.policy.reservation_usd_per_gpu_hour

    def _finish(self, job, state, charged, error=None, results=None, **fields):
        extra = dict(fields)
        if error is not None:
            extra['error_json'] = dumps(error)
        if results is not None:
            extra['results_json'] = dumps(results)
        log.info('benchmark job %s -> %s', job['id'], state)
        return self.store.finish_job(job['id'], state, charged, self.keys(job), **extra)

    def _cancel_quietly(self, job):
        try:
            self.client.cancel(job['provider_job_id'])
        except Exception as exc:
            log.warning('cancel of %s failed: %s', job['id'], type(exc).__name__)

    def cancel(self, job):
        if not self.policy.cancel_enabled:
            raise JobError(403, 'Cancellation is not enabled on this server.')
        if job['state'] in TERMINAL:
            return job
        if not job['provider_job_id']:
            raise JobError(409, 'Submission is still in progress; try again in a few seconds.')
        self.client.cancel(job['provider_job_id'])
        return self.store.update_job(job['id'], state='cancelling')

    # -- views ------------------------------------------------------------------------------
    def public(self, job):
        if job is None:
            return None
        now = self.clock()
        started = job['provider_started_at']
        finished = job['provider_finished_at']
        return {'id': job['id'], 'state': job['state'], 'display_name': job['display_name'], 'analysis_id': job['analysis_id'],
                'platform': job['platform'], 'preset': job['preset'], 'fingerprint': job['fingerprint'],
                'provider': self.client.label, 'simulated': self.policy.backend == 'fake',
                'provider_state': job['provider_state'], 'provider_message': job['provider_message'],
                'created_at': job['created_at'], 'submitted_at': job['submitted_at'], 'deadline_at': job['deadline_at'],
                'elapsed_seconds': (finished or now) - job['created_at'] if job['state'] not in TERMINAL or finished else None,
                'provider_started_at': started, 'provider_finished_at': finished,
                'provider_run_seconds': (finished - started) if started and finished else None,
                'reserved_usd': job['reserved_usd'], 'estimated_charge_usd': job['charged_usd'],
                'cost_label': 'Estimated from provider timestamps at the configured rate; not a Nebius bill.',
                'error': json.loads(job['error_json']) if job['error_json'] else None,
                'results': json.loads(job['results_json']) if job['results_json'] else None,
                'config': json.loads(job['config_json']), 'terminal': job['state'] in TERMINAL, 'resumable': False}


def create_service(store, policy):
    from .nebius_client import FakeJobsClient, FakeResultStore, NebiusSDKClient, S3ResultStore
    if policy.backend == 'fake':
        return JobService(store, policy, FakeJobsClient(), FakeResultStore())
    return JobService(store, policy, NebiusSDKClient(policy), S3ResultStore(policy))
