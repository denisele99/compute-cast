"""Validation, prediction, hardware comparison and original/current snapshots (GPT-2 MVP)."""
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...parsers import gpt2_inputs as G
from ...services.analysis_store import diff
from ...services.estimators import gpt2_predict
from .analysis import get_analysis, merged_extraction, store

router = APIRouter(prefix="/analysis/{analysis_id}", tags=["Estimation"])
LABELS = {f['key']: f['label'] for f in G.FIELDS}


class ConfigBody(BaseModel):
    values: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)  # acknowledged_conflicts: [conflict ids]


class PredictBody(ConfigBody):
    record: bool = True


def _size_ok(body):
    if len(json.dumps(body.model_dump(), default=str)) > 200_000:
        raise HTTPException(413, 'Configuration exceeds 200 KB.')


def context(item, body):
    return merged_extraction(item), body.metadata.get('acknowledged_conflicts') or []


@router.post("/validate")
def validate(request: Request, analysis_id: str, body: ConfigBody):
    item = get_analysis(request, analysis_id)
    _size_ok(body)
    extraction, ack = context(item, body)
    return G.validate(body.values, body.provenance, extraction, ack)


def snapshot_view(item, current_values):
    original = item['original']
    if original is None:
        return {'original': None, 'changes': []}
    before = G.validate(original['config']['values'])['values']
    return {'original': {'captured_at': original['captured_at'], 'result_summary': original['result_summary'],
                         'values': before},
            'changes': diff(before, current_values, LABELS)}


@router.post("/predict")
async def predict(request: Request, analysis_id: str, body: PredictBody):
    item = get_analysis(request, analysis_id)
    _size_ok(body)
    extraction, ack = context(item, body)
    result = await run_in_threadpool(gpt2_predict.predict, body.values, body.provenance, extraction, ack)
    first_run = item['original'] is None
    if result['status'] == 'ok' and body.record:
        store(request).record_prediction(analysis_id, body.model_dump(exclude={'record'}), result)
        item['latest'] = {'inputs': result['inputs']}
    result['snapshots'] = snapshot_view(item, result['validation']['values'])
    result['snapshots']['is_original_run'] = first_run and result['status'] == 'ok' and body.record
    return result


@router.post("/compare")
async def compare(request: Request, analysis_id: str, body: ConfigBody):
    item = get_analysis(request, analysis_id)
    _size_ok(body)
    extraction, ack = context(item, body)
    return await run_in_threadpool(gpt2_predict.compare, body.values, body.provenance, extraction, ack)


@router.get("/files/{name}")
def download(request: Request, analysis_id: str, name: str):
    """estimator_input.json (model/training/hardware/estimator_policy) from the latest recorded prediction."""
    item = get_analysis(request, analysis_id)
    if name != 'estimator_input.json':
        raise HTTPException(404, 'Unknown file.')
    if not item.get('latest'):
        raise HTTPException(409, 'Run a prediction first.')
    return Response(json.dumps(item['latest']['inputs'], indent=2), media_type='application/json',
                    headers={'Content-Disposition': f'attachment; filename="{name}"'})
