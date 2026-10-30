"""MVP input schema and the restricted Hugging Face model list."""
from fastapi import APIRouter, Query

from ...parsers import gpt2_inputs as G
from ...parsers.dataset_sample import SUPPORTED_FORMATS
from ...parsers.gpt2_extract import ALLOWED_HF_MODELS, EXAMPLE_SETS, snapshot
from ...services.upload_cache import ALLOWED, SUFFIX_BYTES, MAX_FILE_BYTES

router = APIRouter(tags=["Parsing"])


@router.get("/schema")
def schema():
    return {**G.schema(), 'uploads': {'allowed_extensions': sorted(ALLOWED['auto']),
                                      'max_bytes_default': MAX_FILE_BYTES['auto'], 'max_bytes_by_type': SUFFIX_BYTES,
                                      'dataset_formats': SUPPORTED_FORMATS},
            'huggingface_models': list(ALLOWED_HF_MODELS),
            'examples': [{'id': k, 'label': v['label'], 'files': v['files']} for k, v in EXAMPLE_SETS.items()]}


@router.get("/hf/search")
def hf_search(q: str = Query('', max_length=100)):
    """Searches only the MVP's supported GPT-2 checkpoints (no open Hub search)."""
    term = q.strip().lower()
    items = []
    for model_id in ALLOWED_HF_MODELS:
        if term and term not in model_id.lower():
            continue
        cfg = snapshot(model_id)
        items.append({'id': model_id, 'summary': f"GPT-2, {cfg['n_layer']} layers, hidden {cfg['n_embd']}, "
                                                 f"{cfg['n_head']} heads, context {cfg['n_positions']}"})
    return {'items': items, 'note': 'MVP: only standard GPT-2 checkpoints are selectable.'}
