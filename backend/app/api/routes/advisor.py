"""Plan recommendations powered by NVIDIA Nemotron 3 Super (Nebius Token Factory, NEBIUS_API_KEY).

POST /analysis/{id}/advice            ask the model (goal, resources, constraints, budget, extra code)
POST /analysis/{id}/advice/apply      record which suggestions were applied (form values are set client-side
                                      and saved by the next estimate; the original snapshot is never changed)
GET  /analysis/{id}/advice/export     ZIP: edited code from applied advice + a basic training/model config
GET  /recommendations/status          whether the model is configured (never returns the key)
GET  /recommendations/examples        example code to try the recommender on
"""
from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from ...services import advisor as AD
from ...services.recommender import safe_path
from .analysis import get_analysis, merged_extraction

try:
    from openai import APIError
except ImportError:  # openai is needed only when recommendations are requested
    APIError = ()  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)
router = APIRouter(tags=["Recommendations"])
EXAMPLES = Path(__file__).resolve().parents[4] / 'examples' / 'recommender'
EXAMPLE_FILES = ('train_baseline.py', 'launch.sh')
CODE_KINDS = ['Training script (train.py): loop, optimizer, mixed precision, gradient accumulation',
              'Model definition or loading code (model.py)',
              'Launch command or script (launch.sh, torchrun/accelerate command)',
              'Distributed config (accelerate/DeepSpeed/FSDP .yaml or .json)',
              'Data loading / collator code (DataLoader settings, padding, packing)']


class AdviceBody(BaseModel):
    values: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    preferences: AD.Preferences = Field(default_factory=AD.Preferences)
    files: dict[str, str] = Field(default_factory=dict)  # extra code for review: relative name -> text


class ApplyBody(BaseModel):
    recommendation_ids: list[str] = Field(min_length=1, max_length=8)
    values: dict = Field(default_factory=dict)  # the form values after applying (saved as the current config)


@router.get("/recommendations/status")
def status():
    return {'configured': bool(os.environ.get('NEBIUS_API_KEY')), 'model': AD.MODEL_LABEL,
            'code_kinds': CODE_KINDS, 'goals': AD.GOALS,
            'note': 'Needs NEBIUS_API_KEY (a Nebius Token Factory key) on the backend. Each request is a paid model call.'}


@router.get("/recommendations/examples")
def examples():
    return {'files': {name: (EXAMPLES / name).read_text(encoding='utf-8') for name in EXAMPLE_FILES}}


@router.post("/analysis/{analysis_id}/advice")
async def advice(request: Request, analysis_id: str, body: AdviceBody):
    item = get_analysis(request, analysis_id)
    try:
        for name in body.files:
            safe_path(name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    extraction = merged_extraction(item)
    ack = body.metadata.get('acknowledged_conflicts') or []
    try:
        result = await run_in_threadpool(AD.advise, item, body.values, body.preferences, extraction, ack, body.files)
    except PermissionError as exc:
        raise HTTPException(503, str(exc)) from exc
    except APIError as exc:
        log.warning('Token Factory request failed: %s', type(exc).__name__)
        raise HTTPException(502, 'The Nemotron request to Nebius Token Factory failed. Check the API key and try again.') from exc
    except (ValidationError, SyntaxError) as exc:
        log.warning('Invalid advisor output: %s', type(exc).__name__)
        raise HTTPException(502, 'The model returned an answer that did not pass validation. Try again.') from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    item['advice'] = {'at': time.time(), **result}
    return result


@router.post("/analysis/{analysis_id}/advice/apply")
def apply(request: Request, analysis_id: str, body: ApplyBody):
    item = get_analysis(request, analysis_id)
    last = item.get('advice')
    if not last:
        raise HTTPException(409, 'Ask for recommendations first.')
    chosen = [r for r in last['recommendations'] if r['id'] in body.recommendation_ids]
    if not chosen:
        raise HTTPException(404, 'Unknown recommendation.')
    files = {f['path']: f for f in last.get('changed_files', []) if any(f['path'] in r['affected_files'] for r in chosen)}
    record = {'at': time.time(), 'recommendation_ids': [r['id'] for r in chosen],
              'new_values': {k: v for r in chosen if r['applicable'] for k, v in r['new_values'].items()},
              'changed_files': list(files.values()) or last.get('changed_files', [])}
    if body.values:
        item['config'] = {**(item.get('config') or {}), 'values': body.values}  # original snapshot is untouched
    item['applied_advice'] = record
    item.setdefault('advice_history', []).append(record)
    return {'applied': record, 'original_kept': item['original'] is not None}


@router.get("/analysis/{analysis_id}/advice/export")
def export(request: Request, analysis_id: str):
    item = get_analysis(request, analysis_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in AD.export_files(item).items():
            z.writestr(f'compute-cast-recommended/{safe_path(name)}', content)
    return Response(buf.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="compute-cast-recommended.zip"'})
