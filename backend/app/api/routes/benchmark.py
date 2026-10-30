"""Benchmark mode: validated benchmark_config.json, local package, results import, Nebius jobs.

Only trusted runner code (benchmark/) ever executes; the backend generates a JSON config, never code.
Cloud routes need a demo-password session (bearer token); policy and credentials stay server-side.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...parsers import gpt2_inputs as G
from ...services.benchmarks import auth as A
from ...services.benchmarks import config_builder as CB
from ...services.benchmarks import results as RS
from ...services.benchmarks.jobs import JobError, create_service
from ...services.benchmarks.package import build_zip
from ...services.benchmarks.policy import load_policy
from ...services.benchmarks.store import Store
from ...services.estimators import gpt2_predict
from .analysis import get_analysis, merged_extraction

router = APIRouter(tags=["Benchmarks"])
MAX_OVERRIDES = 32


class BenchConfigBody(BaseModel):
    values: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    overrides: dict = Field(default_factory=dict)  # dotted path -> value, from the review screen
    target: str = Field(default='local', pattern='^(local|nebius)$')


class ResultsBody(BenchConfigBody):
    text: str | None = Field(default=None, max_length=RS.MAX_RESULT_BYTES)
    job_id: str | None = Field(default=None, max_length=40)


class JobBody(BenchConfigBody):
    analysis_id: str = Field(min_length=1, max_length=64)
    idempotency_key: str = Field(min_length=8, max_length=64, pattern='^[A-Za-z0-9_-]+$')
    confirm_charges: bool = False


class LoginBody(BaseModel):
    display_name: str = Field(max_length=200)
    password: str = Field(max_length=200)


# -- context ------------------------------------------------------------------------------------
def bench(request: Request):
    """Lazily create the durable store and job service (tests inject app.state.bench)."""
    state = request.app.state
    if getattr(state, 'bench', None) is None:
        policy = load_policy()
        state.bench = create_service(Store(policy.database_url), policy)
    return state.bench


def _session(request, authorization):
    svc = bench(request)
    try:
        return A.authenticate(svc.store, svc.policy, authorization)
    except A.AuthError as exc:
        raise HTTPException(exc.status, exc.message) from exc


def _client_ip(request):
    # Render's proxy appends the connecting address as the LAST X-Forwarded-For entry.
    forwarded = request.headers.get('x-forwarded-for')
    if forwarded:
        return forwarded.split(',')[-1].strip()
    return request.client.host if request.client else 'unknown'


def _review(request, item, body, target=None):
    if len(body.overrides) > MAX_OVERRIDES:
        raise HTTPException(413, 'Too many overrides.')
    extraction = merged_extraction(item)
    ack = body.metadata.get('acknowledged_conflicts') or []
    values = G.validate(body.values, body.provenance, extraction, ack)['values']
    extracted = {k: f.get('value') for k, f in (extraction.get('fields') or {}).items()}
    return CB.build(values, body.provenance, extraction.get('huggingface_model'), body.overrides,
                    bench(request).policy, target or body.target, extracted), values


def _error(exc: JobError):
    detail = {'message': exc.message, **(exc.detail or {})}
    return HTTPException(exc.status, detail)


# -- analysis-scoped: config review, local package, results ------------------------------------
@router.post("/analysis/{analysis_id}/benchmark/config")
async def config(request: Request, analysis_id: str, body: BenchConfigBody):
    item = get_analysis(request, analysis_id)
    review, _ = await run_in_threadpool(_review, request, item, body)
    return review


@router.post("/analysis/{analysis_id}/benchmark/package")
async def package(request: Request, analysis_id: str, body: BenchConfigBody):
    item = get_analysis(request, analysis_id)
    review, _ = await run_in_threadpool(_review, request, item, body, 'local')
    if not review['config']:
        raise HTTPException(422, {'message': 'The benchmark configuration is incomplete or invalid.',
                                  'missing': review['missing'], 'errors': review['errors']})
    data = build_zip(review['config'])
    return Response(data, media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="compute-cast-benchmark.zip"'})


@router.post("/analysis/{analysis_id}/benchmark/results")
async def import_results(request: Request, analysis_id: str, body: ResultsBody,
                         authorization: str | None = Header(default=None)):
    """Validate a pasted/uploaded results.json (untrusted) or a finished cloud job's results; compare."""
    item = get_analysis(request, analysis_id)
    if body.job_id:
        session = _session(request, authorization)
        job = bench(request).store.get_job(body.job_id)
        if not job or job['session_id'] != session['id']:
            raise HTTPException(404, 'Benchmark job not found.')
        if not job['results_json']:
            raise HTTPException(409, 'This job has no results yet.')
        parsed, source = bench(request).public(job)['results'], f'Nebius job {job["id"]}'
    else:
        if not body.text:
            raise HTTPException(422, 'Paste or upload results.json.')
        try:
            parsed = RS.parse(body.text)
        except RS.ResultError as exc:
            raise HTTPException(422, str(exc)) from exc
        source = 'local run (uploaded by user; validated as untrusted data)'
    review, values = await run_in_threadpool(_review, request, item, body)
    prediction = CB.predict_for_config(parsed['effective_config'], values) if parsed['effective_config'] else {}
    comparison = RS.compare(parsed, {'prediction': prediction}, review.get('fingerprint'), values.get('hardware.gpu_name'))
    same_shapes = not any(k in review.get('applied_overrides', {}) for k in
                          ('training.microbatch_size', 'training.gradient_accumulation_steps', 'input.sequence_length'))
    projection = None
    if comparison['compatible_for_calibration'] and same_shapes:
        current = await run_in_threadpool(gpt2_predict.predict, body.values, body.provenance, merged_extraction(item),
                                          body.metadata.get('acknowledged_conflicts') or [])
        if current.get('status') == 'ok':
            projection = RS.full_run_projection(parsed, current)
    record = {'kind': 'benchmark', 'source': source, 'result': parsed, 'comparison': comparison,
              'prediction_for_config': prediction, 'full_run_projection': projection,
              'matches_current_config': parsed['config_fingerprint'] == review.get('fingerprint')}
    item['benchmarks'] = (item['benchmarks'] + [record])[-10:]
    return record


# -- cloud: status, session, jobs --------------------------------------------------------------
@router.get("/bench/status")
def status(request: Request, authorization: str | None = Header(default=None)):
    svc = bench(request)
    session = None
    if authorization:
        try:
            session = A.authenticate(svc.store, svc.policy, authorization)
        except A.AuthError:
            session = None
    return svc.status(session)


@router.post("/bench/session")
def login(request: Request, body: LoginBody):
    svc = bench(request)
    try:
        return A.login(svc.store, svc.policy, body.display_name, body.password, _client_ip(request))
    except A.AuthError as exc:
        raise HTTPException(exc.status, exc.message) from exc


@router.get("/bench/session")
def session_info(request: Request, authorization: str | None = Header(default=None)):
    s = _session(request, authorization)
    return {'display_name': s['display_name'], 'expires_at': s['expires_at'],
            'allowance': bench(request).status(s)['session']}


@router.post("/bench/jobs", status_code=202)
async def submit(request: Request, body: JobBody, background: BackgroundTasks,
                 authorization: str | None = Header(default=None)):
    session = _session(request, authorization)
    if not body.confirm_charges:
        raise HTTPException(400, 'Confirm that this job may incur Nebius charges.')
    svc = bench(request)
    existing = svc.store.find_job(session['id'], body.idempotency_key)
    if existing:
        return svc.public(existing)
    item = get_analysis(request, body.analysis_id)
    review, _ = await run_in_threadpool(_review, request, item, body, 'nebius')
    try:
        job, created = svc.submit(session, body.idempotency_key, review, body.analysis_id)
    except JobError as exc:
        raise _error(exc) from exc
    if created:
        background.add_task(svc.dispatch, job['id'])
    return svc.public(job)


@router.get("/bench/jobs")
def list_jobs(request: Request, authorization: str | None = Header(default=None)):
    session = _session(request, authorization)
    svc = bench(request)
    return {'jobs': [svc.public(j) for j in svc.store.session_jobs(session['id'])]}


def _owned(request, authorization, job_id):
    session = _session(request, authorization)
    job = bench(request).store.get_job(job_id)
    if not job or job['session_id'] != session['id']:
        raise HTTPException(404, 'Benchmark job not found.')
    return job


@router.get("/bench/jobs/{job_id}")
async def get_job(request: Request, job_id: str, authorization: str | None = Header(default=None)):
    job = _owned(request, authorization, job_id)
    svc = bench(request)
    return svc.public(await run_in_threadpool(svc.refresh, job['id']))


@router.post("/bench/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str, authorization: str | None = Header(default=None)):
    job = _owned(request, authorization, job_id)
    svc = bench(request)
    try:
        return svc.public(await run_in_threadpool(svc.cancel, job))
    except JobError as exc:
        raise _error(exc) from exc
