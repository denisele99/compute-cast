"""Hardware catalogue search (general specs + Nebius availability/pricing kept separate)."""
from fastapi import APIRouter, HTTPException, Query

from ...parsers import detect_hardware_lookup as lookup
from ...parsers.detect_hardware_local_empty import LOCAL_HARDWARE_COMMANDS
from ...services.estimators.gpt2_predict import hardware_values

router = APIRouter(prefix="/hardware", tags=["Hardware"])


@router.get("/search")
def search(q: str = Query('', max_length=100), limit: int = Query(10, ge=1, le=10),
           offset: int = Query(0, ge=0)):
    return lookup.search(q, limit, offset)


@router.get("/options")
def all_options():
    """Every pre-loaded GPU (Nebius platforms/regions and general catalogue), for the dropdown list."""
    rows = sorted(lookup.options(), key=lambda r: (r['provider'] != 'nebius', r['label'], r['id']))
    return {'items': [{k: r[k] for k in ('id', 'provider', 'platform', 'label', 'vram_gib', 'price', 'issues')} | {'has_specs': bool(r['specs'])}
                      for r in rows], 'price_snapshot': lookup.prices()[1]}


@router.get("/options/{option_id}")
def option(option_id: str):
    """Autofill values (GPT-2 MVP keys) for one result. Specs come from the general catalogue,
    availability and price from the separate Nebius files."""
    row = lookup.get_option(option_id)
    if row is None:
        raise HTTPException(404, 'No catalogue entry with that id.')
    notes = list(row['issues'])
    if row.get('presets'):
        notes.append('Nebius presets: ' + ', '.join(f"{p['name']} ({p['gpu_count']} GPU)" for p in row['presets'])
                     + '. v1 estimates one GPU.')
    return {'option': row, 'values': hardware_values(row), 'notes': notes}


@router.get("/catalog")
def catalog():
    return {'gpus': lookup.catalog(), 'source': 'data/hardware_catalog.json'}


@router.get("/local-command")
def local_command():
    return LOCAL_HARDWARE_COMMANDS
