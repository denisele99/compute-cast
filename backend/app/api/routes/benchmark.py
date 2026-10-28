"""Benchmark mode: local script download + result import; Nebius run reported unavailable."""
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ...parsers.normalize_inputs import validate
from ...services import benchmark_service as bench
from ...services.estimators import estimator
from .analysis import get_analysis
from .estimate import ConfigBody

router = APIRouter(prefix="/analysis/{analysis_id}/benchmark", tags=["Benchmarks"])


def _prediction(body):
    result = estimator.predict(body.values, body.provenance)
    if result['status'] != 'ok':
        raise HTTPException(422, 'Complete the required inputs (prediction must run) before benchmarking.')
    return result


@router.post("/plan")
async def plan(request: Request, analysis_id: str, body: ConfigBody):
    item = get_analysis(request, analysis_id)
    result = await run_in_threadpool(_prediction, body)
    v = result['validation']['values']
    return {**bench.plan(analysis_id, v, list(item['files'].values())),
            'predicted_cost': result['benchmark_cost'],
            'predicted_peak_vram_gib': result['memory']['gpu'].get('peak_vram_gib'),
            'nebius': {'available': False, 'reason': 'No Nebius provisioning integration or credentials '
                       'are configured in this backend.', 'remaining_work': bench.NEBIUS_REMAINING_WORK}}


@router.post("/script")
async def script(request: Request, analysis_id: str, body: ConfigBody):
    get_analysis(request, analysis_id)
    result = await run_in_threadpool(_prediction, body)
    text = bench.generate_script(analysis_id, result['validation']['values'], result)
    return Response(text, media_type='text/x-python',
                    headers={'Content-Disposition': 'attachment; filename="run_benchmark.py"'})


@router.post("/import")
async def import_result(request: Request, analysis_id: str, file: UploadFile = File(...)):
    item = get_analysis(request, analysis_id)
    raw = await file.read(bench.MAX_RESULT_BYTES + 1)
    config = item.get('config') or {}
    values = config.get('values') or {}
    prediction = await run_in_threadpool(estimator.predict, values, config.get('provenance')) if values else None
    try:
        result = bench.import_result(raw.decode('utf-8', errors='replace'), analysis_id,
                                     validate(values)['values'] if values else {}, prediction)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    item['benchmarks'] = (item['benchmarks'] + [result])[-10:]
    return result


class NebiusRun(BaseModel):
    confirm_charges: bool = False


@router.post("/nebius")
def nebius(request: Request, analysis_id: str, body: NebiusRun):
    get_analysis(request, analysis_id)
    if not body.confirm_charges:
        raise HTTPException(400, 'Explicit confirmation of predicted charges is required.')
    try:
        return bench.submit_benchmark(body)
    except NotImplementedError as exc:
        raise HTTPException(501, {'message': str(exc), 'remaining_work': bench.NEBIUS_REMAINING_WORK}) from exc
