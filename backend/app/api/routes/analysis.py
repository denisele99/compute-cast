"""Analysis sessions, the single upload drop zone (files or ZIP), examples and the Hugging Face picker.

Upload -> safe store -> static extraction (AST/JSON/YAML/CSV only) -> merged findings and conflicts
returned to the client, which fills the Model/Training/Data sections. Nothing is executed.
"""
import io
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...clients import huggingface as hf
from ...parsers import gpt2_extract as X
from ...services.upload_cache import ALLOWED, MAX_FILES, MAX_TOTAL_BYTES, UploadError, limit_for

router = APIRouter(prefix="/analysis", tags=["Analysis"])


def store(request: Request):
    return request.app.state.store


def get_analysis(request, analysis_id):
    try:
        return store(request).get(analysis_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def public_file(meta):
    ext = meta.get('extraction', {})
    return {'file_id': meta['file_id'], 'name': meta['name'], 'size': meta['size'], 'origin': meta.get('origin', 'upload'),
            'from_zip': meta.get('from_zip'), 'kind': ext.get('kind'), 'sections': ext.get('sections', []),
            'conflict_count': len(ext.get('conflicts', [])), 'extraction_error': ext.get('error')}


def merged_extraction(item):
    results = [f['extraction'] for f in item['files'].values() if 'findings' in f.get('extraction', {})]
    merged = X.merge(results)
    merged['errors'] = [{'file': f['name'], 'message': f['extraction']['error']}
                        for f in item['files'].values() if f.get('extraction', {}).get('error')]
    merged['files'] = [public_file(f) for f in item['files'].values()]
    merged['huggingface_model'] = next((f['model_id'] for f in item['files'].values() if f.get('model_id')), None)
    merged['executed_code'] = False
    return merged


def _extract(meta):
    try:
        return X.extract(meta['path'], meta['name'])
    except (ValueError, SyntaxError, UnicodeDecodeError, OSError) as exc:
        line = f' (line {exc.lineno})' if isinstance(exc, SyntaxError) and exc.lineno else ''
        return {'error': f'{type(exc).__name__}{line}: {exc}'}
    except Exception as exc:  # YAML/CSV parser errors have varied types
        return {'error': f'Could not parse: {type(exc).__name__}: {exc}'}


def _save(request, analysis_id, item, name, stream, origin, from_zip=None):
    meta = store(request).cache.save(analysis_id, 'auto', name, stream, list(item['files'].values()))
    meta.update(origin=origin, from_zip=from_zip)
    meta['extraction'] = _extract(meta)
    item['files'][meta['file_id']] = meta
    return meta


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
    return {'analysis_id': analysis_id, 'created_at': item['created_at'], 'files': [public_file(f) for f in item['files'].values()],
            'config': item['config'], 'original': item['original'], 'predictions': item['predictions'],
            'benchmarks': item['benchmarks']}


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


@router.post("/{analysis_id}/uploads")
async def upload(request: Request, analysis_id: str, files: list[UploadFile] = File(...)):
    """One drop zone: several files and/or ZIP archives; content decides the form section."""
    item = get_analysis(request, analysis_id)
    added, skipped = [], []
    for upload_file in files:
        name = upload_file.filename or ''
        try:
            if name.lower().endswith('.zip'):
                data = await upload_file.read(limit_for('auto', '.zip') + 1)
                if len(data) > limit_for('auto', '.zip'):
                    raise UploadError(f'{name} exceeds 30 MB.')

                def member_limit(member):
                    suffix = Path(member).suffix.lower()
                    return limit_for('auto', suffix) if suffix in ALLOWED['auto'] - {'.zip'} else None
                members, bad = await run_in_threadpool(X.expand_zip, data, member_limit)
                skipped += [f'{name}: {b}' for b in bad]
                for member, content in members:
                    try:
                        meta = await run_in_threadpool(_save, request, analysis_id, item, member, io.BytesIO(content),
                                                       'upload', name)
                        added.append(public_file(meta))
                    except UploadError as exc:
                        skipped.append(f'{name}/{member}: {exc}')
            else:
                meta = await run_in_threadpool(_save, request, analysis_id, item, name, upload_file.file, 'upload')
                added.append(public_file(meta))
        except (UploadError, ValueError) as exc:
            skipped.append(f'{name}: {exc}')
    if not added and skipped:
        raise HTTPException(422, '; '.join(skipped))
    return {'added': added, 'skipped': skipped, 'extraction': merged_extraction(item)}


@router.delete("/{analysis_id}/uploads/{file_id}")
def remove_upload(request: Request, analysis_id: str, file_id: str):
    item = get_analysis(request, analysis_id)
    meta = item['files'].pop(file_id, None)
    if meta is None:
        raise HTTPException(404, 'File not found in this analysis.')
    if meta.get('path'):
        store(request).cache.remove(analysis_id, meta)
    return {'extraction': merged_extraction(item)}


@router.get("/{analysis_id}/extraction")
def extraction(request: Request, analysis_id: str):
    return merged_extraction(get_analysis(request, analysis_id))


@router.get("/examples/list")
def examples():
    return {'examples': [{'id': k, 'label': v['label'], 'files': v['files']} for k, v in X.EXAMPLE_SETS.items()]}


@router.post("/{analysis_id}/examples/{example_id}")
async def load_example(request: Request, analysis_id: str, example_id: str):
    item = get_analysis(request, analysis_id)
    example = X.EXAMPLE_SETS.get(example_id)
    if example is None:
        raise HTTPException(404, 'Unknown example.')
    added = []
    for name in example['files']:
        with open(X.EXAMPLES / name, 'rb') as handle:
            meta = await run_in_threadpool(_save, request, analysis_id, item, name, handle, 'example', example['label'])
        added.append(public_file(meta))
    return {'added': added, 'skipped': [], 'extraction': merged_extraction(item)}


class HFSelect(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)


@router.post("/{analysis_id}/huggingface")
async def select_hf(request: Request, analysis_id: str, body: HFSelect):
    """Use ONE allowed Hugging Face model's config.json (live, or the bundled snapshot when offline)."""
    item = get_analysis(request, analysis_id)
    if body.model_id not in X.ALLOWED_HF_MODELS:
        raise HTTPException(422, 'MVP supports only: ' + ', '.join(X.ALLOWED_HF_MODELS))
    current = next((f for f in item['files'].values() if f.get('model_id')), None)
    if current is not None:
        raise HTTPException(409, f"{current['model_id']} is already selected. Remove it before choosing another model.")
    try:
        config = await run_in_threadpool(hf.fetch_config, body.model_id)
        origin = 'the Hugging Face Hub (live)'
    except (hf.HubError, ValueError):
        config = X.snapshot(body.model_id)
        origin = 'the bundled snapshot (Hub unreachable)'
    result = X.huggingface_result(body.model_id, config, origin)
    key = 'hf_' + body.model_id.replace('/', '__')  # URL-safe file id
    item['files'][key] = {'file_id': key, 'name': result['file'], 'size': 0, 'path': '', 'origin': 'huggingface',
                          'model_id': body.model_id, 'extraction': result}
    return {'origin': origin, 'extraction': merged_extraction(item)}
