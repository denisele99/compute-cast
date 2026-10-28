"""Schema and Hugging Face metadata lookup (config.json only, no weights, no remote code)."""
from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from ...clients import huggingface as hf
from ...parsers import normalize_inputs as N
from ...parsers.dataset_sample import SUPPORTED_FORMATS
from ...parsers.static_extract import map_model_config
from ...services.upload_cache import ALLOWED, MAX_FILE_BYTES

router = APIRouter(tags=["Parsing"])


@router.get("/schema")
def schema():
    return {**N.schema(), 'uploads': {'allowed_extensions': {k: sorted(v) for k, v in ALLOWED.items()},
                                      'max_bytes': MAX_FILE_BYTES,
                                      'dataset_formats': SUPPORTED_FORMATS}}


@router.get("/hf/search")
async def hf_search(q: str = Query(min_length=1, max_length=100)):
    try:
        return {'items': await run_in_threadpool(hf.search_models, q)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except hf.HubError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/hf/config")
async def hf_config(model_id: str = Query(min_length=1, max_length=200), revision: str = 'main'):
    try:
        config = await run_in_threadpool(hf.fetch_config, model_id, revision)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except hf.HubError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {'model_id': model_id, 'config': config,
            'extraction': map_model_config(config, f'{model_id}/config.json', source='huggingface')}
