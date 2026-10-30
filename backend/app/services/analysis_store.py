"""In-memory analysis sessions: configuration, provenance, snapshots and results.

Each analysis holds:
  * ``config``    — the complete analysis configuration (form values, provenance,
                    metadata) as last saved by the client;
  * ``original``  — an immutable snapshot captured at the FIRST estimation run;
  * ``predictions`` / ``benchmarks`` — kept separately, each with provenance;
  * ``files``     — upload-cache metadata and their extraction results.

State lives in process memory: it is lost on server restart and is not shared between
worker processes (run uvicorn with one worker). A page refresh starts a new analysis.
"""
from __future__ import annotations

import copy
import secrets
import threading
import time

from .upload_cache import DEFAULT_TTL_SECONDS, UploadCache


class AnalysisStore:
    def __init__(self, cache=None, ttl_seconds=DEFAULT_TTL_SECONDS):
        self.cache = cache or UploadCache(ttl_seconds=ttl_seconds)
        self.ttl = ttl_seconds
        self._items = {}
        self._lock = threading.Lock()

    def create(self):
        self.sweep()
        analysis_id = secrets.token_hex(16)
        now = time.time()
        with self._lock:
            self._items[analysis_id] = {'id': analysis_id, 'created_at': now, 'last_access': now,
                                        'files': {}, 'config': None, 'original': None,
                                        'predictions': [], 'benchmarks': []}
        self.cache.directory(analysis_id, create=True)
        return analysis_id

    def get(self, analysis_id):
        self.sweep()
        with self._lock:
            item = self._items.get(analysis_id)
            if item is None:
                raise KeyError('Analysis not found or expired; start a new analysis.')
            item['last_access'] = time.time()
        self.cache.touch(analysis_id)
        return item

    def delete(self, analysis_id):
        with self._lock:
            existed = self._items.pop(analysis_id, None) is not None
        cleared = self.cache.clear(analysis_id)
        return {'existed': existed, 'cache_cleared': cleared}

    def sweep(self, now=None):
        now = now or time.time()
        with self._lock:
            expired = [k for k, v in self._items.items() if now - v['last_access'] > self.ttl]
            for key in expired:
                self._items.pop(key, None)
            active = set(self._items)
        for key in expired:
            self.cache.clear(key)
        self.cache.sweep(now, keep=active)
        return expired

    def save_config(self, analysis_id, config):
        item = self.get(analysis_id)
        item['config'] = copy.deepcopy(config)
        return item

    def record_prediction(self, analysis_id, config, result):
        """Store a prediction; capture the original snapshot on the first run only."""
        item = self.get(analysis_id)
        item['config'] = copy.deepcopy(config)
        summary = summarize(result)
        if item['original'] is None:
            item['original'] = {'captured_at': time.time(), 'config': copy.deepcopy(config),
                                'result_summary': summary}
        item['predictions'] = (item['predictions'] + [{'at': time.time(), 'summary': summary}])[-20:]
        return item


def summarize(result):
    """Compact GPT-2 MVP result summary kept with each run (and the original snapshot)."""
    if result.get('status') != 'ok':
        return {'status': result.get('status')}
    runtime = result.get('runtime') or {}
    return {'status': 'ok', 'capacity_gib': result['fit']['capacity_gib'],
            'allocated_peak_gib': result['memory']['estimated_allocated_peak'], 'fits': result['fit']['fits'],
            'training_hours': (runtime.get('total_training_compute') or {}).get('seconds', 0) / 3600 if runtime else None,
            'cost': (result.get('cost') or {}).get('cost')}


def diff(original_values, current_values, labels):
    rows = []
    for key in sorted(set(original_values) | set(current_values)):
        a, b = original_values.get(key), current_values.get(key)
        if a != b and not (a in (None, '') and b in (None, '')):
            rows.append({'field': key, 'label': labels.get(key, key), 'original': a, 'current': b})
    return rows
