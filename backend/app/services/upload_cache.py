"""Per-analysis temporary upload cache.

Location: ``$UPLOAD_CACHE_DIR`` or ``<system temp>/compute_cast_uploads`` — outside the
repository and outside any served/static directory. Each analysis gets its own
``<root>/<analysis_id>/`` directory (mode 0700); files are stored under random ids
(mode 0600) with the validated original name kept only as metadata.

Uploads are only ever parsed statically (AST / JSON / YAML / CSV). They are never imported,
executed or passed to an interpreter by the backend, including during prediction.

Cleanup: explicit reset/refresh/page-close (best effort) and expiry-based sweeping on
every request and at startup, because browser close notifications are unreliable.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

ALLOWED = {
    'model': {'.py', '.json', '.yaml', '.yml'},
    'training': {'.py', '.json', '.yaml', '.yml'},
    'dataset': {'.csv', '.tsv', '.jsonl', '.json'},
}
MAX_FILE_BYTES = {'model': 1_000_000, 'training': 1_000_000, 'dataset': 25_000_000}
MAX_FILES = 20
MAX_TOTAL_BYTES = 60_000_000
DEFAULT_TTL_SECONDS = 2 * 60 * 60
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._ -]{0,127}$')
ANALYSIS_ID = re.compile(r'^[0-9a-f]{32}$')


class UploadError(ValueError):
    pass


def validate_filename(name, kind):
    if kind not in ALLOWED:
        raise UploadError(f'Unknown upload kind {kind!r}.')
    base = os.path.basename(name or '')
    if base != name or not NAME.match(base) or '..' in base:
        raise UploadError('Invalid filename: use letters, digits, ".", "_", "-" or spaces, '
                          'no paths, up to 128 characters.')
    suffix = Path(base).suffix.lower()
    if suffix not in ALLOWED[kind]:
        raise UploadError(f'{base}: {kind} uploads accept {", ".join(sorted(ALLOWED[kind]))}.')
    return base, suffix


class UploadCache:
    def __init__(self, root=None, ttl_seconds=DEFAULT_TTL_SECONDS):
        self.root = Path(root or os.getenv('UPLOAD_CACHE_DIR')
                         or Path(tempfile.gettempdir()) / 'compute_cast_uploads')
        self.ttl = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def directory(self, analysis_id, create=False):
        if not ANALYSIS_ID.match(str(analysis_id)):
            raise UploadError('Invalid analysis id.')
        path = self.root / analysis_id
        if create:
            path.mkdir(exist_ok=True, mode=0o700)
        return path

    def save(self, analysis_id, kind, filename, stream, existing):
        base, suffix = validate_filename(filename, kind)
        if len(existing) >= MAX_FILES:
            raise UploadError(f'At most {MAX_FILES} files per analysis.')
        used = sum(f['size'] for f in existing)
        folder = self.directory(analysis_id, create=True)
        file_id = uuid.uuid4().hex
        target = folder / f'{file_id}{suffix}'
        digest, size, limit = hashlib.sha256(), 0, MAX_FILE_BYTES[kind]
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, 'wb') as handle:
                while chunk := stream.read(1 << 16):
                    size += len(chunk)
                    if size > limit:
                        raise UploadError(f'{base} exceeds {limit // 1_000_000} MB for {kind} uploads.')
                    if used + size > MAX_TOTAL_BYTES:
                        raise UploadError('Analysis upload cache limit (60 MB) reached.')
                    digest.update(chunk)
                    handle.write(chunk)
            if size == 0:
                raise UploadError(f'{base} is empty.')
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        self.touch(analysis_id)
        return {'file_id': file_id, 'kind': kind, 'name': base, 'size': size,
                'sha256': digest.hexdigest(), 'path': str(target), 'uploaded_at': time.time()}

    def remove(self, analysis_id, meta):
        path = Path(meta['path'])
        if path.parent == self.directory(analysis_id):
            path.unlink(missing_ok=True)

    def touch(self, analysis_id):
        folder = self.directory(analysis_id)
        if folder.exists():
            os.utime(folder)

    def clear(self, analysis_id):
        folder = self.directory(analysis_id)
        shutil.rmtree(folder, ignore_errors=True)
        return not folder.exists()

    def sweep(self, now=None, keep=()):
        """Delete analysis folders untouched for longer than the TTL. Returns removed ids."""
        now = now or time.time()
        removed = []
        for folder in self.root.iterdir() if self.root.exists() else []:
            if not folder.is_dir() or not ANALYSIS_ID.match(folder.name) or folder.name in keep:
                continue
            if now - folder.stat().st_mtime > self.ttl:
                shutil.rmtree(folder, ignore_errors=True)
                removed.append(folder.name)
        return removed
