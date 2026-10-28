"""Hardware catalogue search (general specs + Nebius availability/pricing kept separate)."""
from fastapi import APIRouter, HTTPException, Query

from ...parsers import detect_hardware_lookup as lookup
from ...parsers.detect_hardware_local_empty import LOCAL_HARDWARE_COMMANDS

router = APIRouter(prefix="/hardware", tags=["Hardware"])


@router.get("/search")
def search(q: str = Query('', max_length=100), limit: int = Query(10, ge=1, le=10),
           offset: int = Query(0, ge=0)):
    return lookup.search(q, limit, offset)


@router.get("/options/{option_id}")
def option(option_id: str, preset_gpu_count: int | None = None):
    row = lookup.get_option(option_id)
    if row is None:
        raise HTTPException(404, 'No catalogue entry with that id.')
    values, notes = lookup.form_values(row, preset_gpu_count)
    return {'option': row, 'values': values, 'notes': notes}


@router.get("/catalog")
def catalog():
    return {'gpus': lookup.catalog(), 'source': 'data/hardware_catalog.json'}


@router.get("/local-command")
def local_command():
    return LOCAL_HARDWARE_COMMANDS
