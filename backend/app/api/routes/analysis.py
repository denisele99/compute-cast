"""Analysis sessions, upload cache and static extraction.

Upload -> static extraction (AST/JSON/YAML/CSV only) -> merged findings returned to the
client, which populates the form. Uploaded files are never executed.
"""
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...clients import huggingface as hf
from ...parsers import dataset_sample
from ...parsers.static_extract import extract_file, map_model_config, merge_findings
from ...services.upload_cache import MAX_FILES, MAX_TOTAL_BYTES, UploadError

router = APIRouter(prefix="/analysis", tags=["Analysis"])


def store(request: Request):
    return request.app.state.store


def get_analysis(request, analysis_id):
    try:
        return store(request).get(analysis_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def public_file(meta):
    return {k: meta[k] for k in ('file_id', 'kind', 'name', 'size', 'sha256', 'uploaded_at')} | {
        'extraction_error': meta.get('extraction', {}).get('error')}


def merged_extraction(item):
    results = [f['extraction'] for f in item['files'].values() if 'findings' in f.get('extraction', {})]
    fields = merge_findings(results)
    hints = sorted({r['profile_hint'] for r in results if r.get('profile_hint')})
    if 'model.profile' not in fields and len(hints) == 1:
        src = next(r for r in results if r.get('profile_hint') == hints[0])
        fields['model.profile'] = {'field': 'model.profile', 'value': hints[0], 'status': 'uncertain',
                                   'source': 'static_extraction', 'file': src['file'], 'line': None,
                                   'message': 'Profile inferred from file structure; confirm it matches exactly.'}
    elif len(hints) > 1:
        fields['model.profile'] = {'field': 'model.profile', 'value': None, 'status': 'unresolved',
                                   'source': 'static_extraction', 'file': None, 'line': None,
                                   'message': f'Files suggest different profiles: {", ".join(hints)}.'}
    return {'fields': fields,
            'notes': [n for r in results for n in r.get('notes', [])],
            'unsupported': [u for r in results for u in r.get('unsupported', [])],
            'dataset_samples': [r['stats'] for r in results if r.get('kind') == 'dataset_sample'],
            'errors': [{'file': f['name'], 'message': f['extraction']['error']}
                       for f in item['files'].values() if f.get('extraction', {}).get('error')],
            'files': [public_file(f) for f in item['files'].values()],
            'executed_code': False}


@router.post("")
def create(request: Request):
    s = store(request)
    analysis_id = s.create()
    return {'analysis_id': analysis_id, 'ttl_seconds': s.ttl,
            'cache': {'location': 'server temporary directory, one folder per analysis',
                      'max_files': MAX_FILES, 'max_total_bytes': MAX_TOTAL_BYTES}}


@router.get("/{analysis_id}")
def state(request: Request, analysis_id: str):
    item = get_analysis(request, analysis_id)
    return {'analysis_id': analysis_id, 'created_at': item['created_at'],
            'files': [public_file(f) for f in item['files'].values()],
            'config': item['config'], 'original': item['original'],
            'predictions': item['predictions'], 'benchmarks': item['benchmarks']}


@router.delete("/{analysis_id}")
def reset(request: Request, analysis_id: str):
    try:
        return store(request).delete(analysis_id)
    except UploadError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/{analysis_id}/close")
async def close(request: Request, analysis_id: str):
    """navigator.sendBeacon target on page close; best effort, expiry sweeping is the backstop."""
    try:
        return store(request).delete(analysis_id)
    except UploadError as exc:
        raise HTTPException(422, str(exc)) from exc


def _extract(meta):
    try:
        if meta['kind'] == 'dataset':
            return dataset_sample.analyze(meta['path'], meta['name'])
        return extract_file(meta['path'], meta['kind'], meta['name'])
    except (ValueError, SyntaxError, UnicodeDecodeError, OSError) as exc:
        line = f' (line {exc.lineno})' if isinstance(exc, SyntaxError) and exc.lineno else ''
        return {'error': f'{type(exc).__name__}{line}: {exc}'}
    except Exception as exc:  # YAML and csv errors subclass several bases
        return {'error': f'Could not parse: {type(exc).__name__}: {exc}'}


@router.post("/{analysis_id}/uploads")
async def upload(request: Request, analysis_id: str, kind: str = Form(...),
                 file: UploadFile = File(...)):
    item = get_analysis(request, analysis_id)
    try:
        meta = await run_in_threadpool(store(request).cache.save, analysis_id, kind, file.filename,
                                       file.file, list(item['files'].values()))
    except UploadError as exc:
        raise HTTPException(422, str(exc)) from exc
    meta['extraction'] = await run_in_threadpool(_extract, meta)
    item['files'][meta['file_id']] = meta
    return {'file': public_file(meta), 'file_extraction': meta['extraction'],
            'extraction': merged_extraction(item)}


@router.delete("/{analysis_id}/uploads/{file_id}")
def remove_upload(request: Request, analysis_id: str, file_id: str):
    item = get_analysis(request, analysis_id)
    meta = item['files'].pop(file_id, None)
    if meta is None:
        raise HTTPException(404, 'File not found in this analysis.')
    store(request).cache.remove(analysis_id, meta)
    return {'extraction': merged_extraction(item)}


@router.get("/{analysis_id}/extraction")
def extraction(request: Request, analysis_id: str):
    return merged_extraction(get_analysis(request, analysis_id))


class HFImport(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    revision: str = 'main'


@router.post("/{analysis_id}/huggingface")
async def import_hf(request: Request, analysis_id: str, body: HFImport):
    """Fetch config.json (metadata only) and add it to this analysis as an extraction source."""
    item = get_analysis(request, analysis_id)
    try:
        config = await run_in_threadpool(hf.fetch_config, body.model_id, body.revision)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except hf.HubError as exc:
        raise HTTPException(502, str(exc)) from exc
    result = map_model_config(config, f'{body.model_id}/config.json', source='huggingface')
    result['findings'].append({'field': 'model.hf_model_id', 'value': body.model_id, 'status': 'extracted',
                               'source': 'huggingface', 'file': result['file'], 'line': None, 'message': None})
    key = 'hf:' + body.model_id
    item['files'][key] = {'file_id': key, 'kind': 'model', 'name': result['file'], 'size': 0,
                          'sha256': None, 'uploaded_at': None, 'path': '', 'extraction': result}
    return {'config': config, 'extraction': merged_extraction(item)}
