"""Benchmark mode: local script download + result import; Nebius run reported unavailable."""
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ...parsers.gpt2_inputs import validate
from ...services import benchmark_service as bench
from ...services.estimators import gpt2_predict
from .analysis import get_analysis, merged_extraction
from .estimate import ConfigBody

router = APIRouter(prefix="/analysis/{analysis_id}/benchmark", tags=["Benchmarks"])


def _prediction(item, body):
    ack = body.metadata.get('acknowledged_conflicts') or []
    result = gpt2_predict.predict(body.values, body.provenance, merged_extraction(item), ack)
    if result['status'] != 'ok':
        raise HTTPException(422, 'Complete the required inputs (prediction must run) before benchmarking.')
    return result


@router.post("/plan")
async def plan(request: Request, analysis_id: str, body: ConfigBody):
    item = get_analysis(request, analysis_id)
    result = await run_in_threadpool(_prediction, item, body)
    v = result['validation']['values']
    return {**bench.plan(analysis_id, v, list(item['files'].values())),
            'predicted_cost': bench.benchmark_cost(result, v),
            'predicted_allocated_peak_gib': result['memory']['estimated_allocated_peak'],
            'nebius': {'available': False, 'reason': 'No Nebius provisioning integration or credentials '
                       'are configured in this backend.', 'remaining_work': bench.NEBIUS_REMAINING_WORK}}


@router.post("/script")
async def script(request: Request, analysis_id: str, body: ConfigBody):
    item = get_analysis(request, analysis_id)
    result = await run_in_threadpool(_prediction, item, body)
    text = bench.generate_script(analysis_id, result['validation']['values'], result)
    return Response(text, media_type='text/x-python',
                    headers={'Content-Disposition': 'attachment; filename="run_benchmark.py"'})


@router.post("/import")
async def import_result(request: Request, analysis_id: str, file: UploadFile = File(...)):
    item = get_analysis(request, analysis_id)
    raw = await file.read(bench.MAX_RESULT_BYTES + 1)
    config = item.get('config') or {}
    values = config.get('values') or {}
    ack = (config.get('metadata') or {}).get('acknowledged_conflicts') or []
    prediction = (await run_in_threadpool(gpt2_predict.predict, values, config.get('provenance'),
                                          merged_extraction(item), ack)) if values else None
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
