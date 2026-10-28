"""Validation, prediction, hardware comparison and original/current snapshots."""
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...parsers import normalize_inputs as N
from ...services.analysis_store import diff
from ...services.estimators import estimator
from .analysis import get_analysis, store

router = APIRouter(prefix="/analysis/{analysis_id}", tags=["Estimation"])
LABELS = {f['key']: f['label'] for f in N.FIELDS}


class ConfigBody(BaseModel):
    values: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class PredictBody(ConfigBody):
    record: bool = True


class CompareBody(ConfigBody):
    option_ids: list[str] | None = None


def _size_ok(body):
    if len(json.dumps(body.model_dump(), default=str)) > 200_000:
        raise HTTPException(413, 'Configuration exceeds 200 KB.')


@router.post("/validate")
def validate(request: Request, analysis_id: str, body: ConfigBody):
    get_analysis(request, analysis_id)
    _size_ok(body)
    return N.validate(body.values, body.provenance)


@router.post("/config")
def save_config(request: Request, analysis_id: str, body: ConfigBody):
    _size_ok(body)
    store(request).save_config(analysis_id, body.model_dump())
    return {'saved': True}


def snapshot_view(item, current_values):
    original = item['original']
    if original is None:
        return {'original': None, 'changes': []}
    before = N.validate(original['config']['values'])['values']
    return {'original': {'captured_at': original['captured_at'], 'result_summary': original['result_summary'],
                         'values': before},
            'changes': diff(before, current_values, LABELS)}


@router.post("/predict")
async def predict(request: Request, analysis_id: str, body: PredictBody):
    item = get_analysis(request, analysis_id)
    _size_ok(body)
    result = await run_in_threadpool(estimator.predict, body.values, body.provenance)
    first_run = item['original'] is None
    if result['status'] == 'ok' and body.record:
        store(request).record_prediction(analysis_id, body.model_dump(exclude={'record'}), result)
        item['latest'] = {'workload': result['workload'], 'hardware': result['bundle']['hardware']}
    result['snapshots'] = snapshot_view(item, result['validation']['values'])
    result['snapshots']['is_original_run'] = first_run and result['status'] == 'ok' and body.record
    return result


@router.post("/compare")
async def compare(request: Request, analysis_id: str, body: CompareBody):
    get_analysis(request, analysis_id)
    _size_ok(body)
    return await run_in_threadpool(estimator.compare, body.values, body.provenance, body.option_ids)


@router.get("/files/{name}")
def download(request: Request, analysis_id: str, name: str):
    """workload.json / hardware.json from the latest recorded prediction."""
    item = get_analysis(request, analysis_id)
    key = {'workload.json': 'workload', 'hardware.json': 'hardware'}.get(name)
    if key is None:
        raise HTTPException(404, 'Unknown file.')
    if not item.get('latest'):
        raise HTTPException(409, 'Run a prediction first.')
    return Response(json.dumps(item['latest'][key], indent=2), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename="{name}"'})
