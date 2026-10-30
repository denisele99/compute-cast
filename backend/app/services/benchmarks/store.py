"""Durable storage for benchmark sessions, login attempts, allowances and jobs.

BENCH_DATABASE_URL: ``sqlite:////var/data/bench.sqlite3`` (Render persistent disk) or a
``postgresql://`` URL (Render Postgres; needs psycopg). Render's default filesystem and process
memory are ephemeral, so production must point this at durable storage.

Concurrency guarantees come from the database itself:
  * allowance reservation is a conditional UPDATE (``used + reserved + amount <= limit``) per row;
  * ``jobs.active_slot`` is UNIQUE and set to 1 only for non-terminal jobs, so at most one GPU job is
    active globally (NULLs do not collide);
  * ``jobs(session_id, idempotency_key)`` is UNIQUE, so retried clicks return the existing job.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

TERMINAL = ('succeeded', 'failed', 'timed_out', 'cancelled')
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS bench_sessions (id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
       created_at REAL NOT NULL, expires_at REAL NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS bench_login_attempts (client TEXT NOT NULL, at REAL NOT NULL)",
    """CREATE TABLE IF NOT EXISTS bench_allowances (key TEXT PRIMARY KEY, limit_usd REAL NOT NULL,
       reserved_usd REAL NOT NULL DEFAULT 0, used_usd REAL NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS bench_jobs (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, display_name TEXT,
       idempotency_key TEXT NOT NULL, analysis_id TEXT, config_json TEXT NOT NULL, fingerprint TEXT NOT NULL,
       platform TEXT NOT NULL, preset TEXT NOT NULL, image TEXT NOT NULL, state TEXT NOT NULL,
       provider_job_id TEXT, provider_state TEXT, provider_message TEXT, reserved_usd REAL NOT NULL,
       charged_usd REAL, created_at REAL NOT NULL, submitted_at REAL, provider_started_at REAL,
       provider_finished_at REAL, last_checked_at REAL, deadline_at REAL NOT NULL, error_json TEXT,
       results_json TEXT, active_slot INTEGER UNIQUE, UNIQUE (session_id, idempotency_key))""",
]


class AllowanceExceeded(Exception):
    pass


class ConcurrencyLimit(Exception):
    pass


class Store:
    def __init__(self, url):
        self.url = url
        self.pg = url.startswith(('postgres://', 'postgresql://'))
        self._lock = threading.RLock()
        if self.pg:
            import psycopg  # optional dependency, only for Postgres
            self._conn = psycopg.connect(url, autocommit=False)
        else:
            path = url.removeprefix('sqlite:///') if url.startswith('sqlite:///') else url
            self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
            self._conn.execute('PRAGMA journal_mode=WAL')
        with self.tx() as cur:
            for statement in SCHEMA:
                cur.execute(statement)

    # -- low level -------------------------------------------------------------------------
    def _q(self, sql):
        return sql.replace('?', '%s') if self.pg else sql

    class _Tx:
        def __init__(self, store):
            self.store = store

        def __enter__(self):
            self.store._lock.acquire()
            self.cur = self.store._conn.cursor()
            if not self.store.pg:
                self.cur.execute('BEGIN IMMEDIATE')
            return self

        def execute(self, sql, params=()):
            self.cur.execute(self.store._q(sql), params)
            return self.cur

        def __exit__(self, exc_type, exc, tb):
            try:
                if exc_type is None:
                    self.store._conn.commit() if self.store.pg else self.cur.execute('COMMIT')
                else:
                    self.store._conn.rollback() if self.store.pg else self.cur.execute('ROLLBACK')
            finally:
                self.store._lock.release()
            return False

    def tx(self):
        return Store._Tx(self)

    def _rows(self, sql, params=()):
        with self.tx() as t:
            cur = t.execute(sql, params)
            cols = [c[0] for c in cur.description] if cur.description else []
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def _integrity_errors(self):
        if self.pg:
            import psycopg
            return (psycopg.errors.UniqueViolation,)
        return (sqlite3.IntegrityError,)

    # -- sessions & login ------------------------------------------------------------------
    def create_session(self, display_name, ttl):
        sid, now = uuid.uuid4().hex, time.time()
        with self.tx() as t:
            t.execute('INSERT INTO bench_sessions VALUES (?, ?, ?, ?)', (sid, display_name, now, now + ttl))
        return {'id': sid, 'display_name': display_name, 'expires_at': now + ttl}

    def get_session(self, sid):
        rows = self._rows('SELECT * FROM bench_sessions WHERE id = ? AND expires_at > ?', (sid, time.time()))
        return rows[0] if rows else None

    def login_attempts(self, client, window):
        now = time.time()
        with self.tx() as t:
            t.execute('DELETE FROM bench_login_attempts WHERE at < ?', (now - window,))
            return t.execute('SELECT COUNT(*) FROM bench_login_attempts WHERE client = ?', (client,)).fetchone()[0]

    def record_failed_login(self, client):
        with self.tx() as t:
            t.execute('INSERT INTO bench_login_attempts VALUES (?, ?)', (client, time.time()))

    # -- allowances ------------------------------------------------------------------------
    def ensure_allowance(self, key, limit_usd):
        with self.tx() as t:
            if not t.execute('SELECT 1 FROM bench_allowances WHERE key = ?', (key,)).fetchone():
                t.execute('INSERT INTO bench_allowances (key, limit_usd) VALUES (?, ?)', (key, limit_usd))
            else:
                t.execute('UPDATE bench_allowances SET limit_usd = ? WHERE key = ?', (limit_usd, key))

    def allowance(self, key):
        rows = self._rows('SELECT * FROM bench_allowances WHERE key = ?', (key,))
        return rows[0] if rows else None

    # -- jobs ------------------------------------------------------------------------------
    def find_job(self, session_id, idempotency_key):
        rows = self._rows('SELECT * FROM bench_jobs WHERE session_id = ? AND idempotency_key = ?', (session_id, idempotency_key))
        return rows[0] if rows else None

    def reserve_and_create(self, job, allowance_keys):
        """Atomically reserve job['reserved_usd'] on every allowance key and insert the active job."""
        amount = job['reserved_usd']
        try:
            with self.tx() as t:
                for key in allowance_keys:
                    cur = t.execute('UPDATE bench_allowances SET reserved_usd = reserved_usd + ? '
                                    'WHERE key = ? AND used_usd + reserved_usd + ? <= limit_usd', (amount, key, amount))
                    if cur.rowcount != 1:
                        raise AllowanceExceeded(key)
                cols = list(job)
                t.execute(f"INSERT INTO bench_jobs ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
                          [job[c] for c in cols])
        except self._integrity_errors() as exc:
            existing = self.find_job(job['session_id'], job['idempotency_key'])
            if existing:
                return existing, False
            raise ConcurrencyLimit() from exc
        return self.get_job(job['id']), True

    def get_job(self, job_id):
        rows = self._rows('SELECT * FROM bench_jobs WHERE id = ?', (job_id,))
        return rows[0] if rows else None

    def session_jobs(self, session_id, limit=20):
        return self._rows('SELECT * FROM bench_jobs WHERE session_id = ? ORDER BY created_at DESC LIMIT ?', (session_id, limit))

    def active_jobs(self):
        return self._rows('SELECT * FROM bench_jobs WHERE active_slot IS NOT NULL')

    def update_job(self, job_id, **fields):
        if not fields:
            return self.get_job(job_id)
        with self.tx() as t:
            t.execute(f"UPDATE bench_jobs SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?", [*fields.values(), job_id])
        return self.get_job(job_id)

    def finish_job(self, job_id, state, charged_usd, allowance_keys, **fields):
        """Terminal transition: free the active slot and convert the reservation into estimated usage (once)."""
        with self.tx() as t:
            row = t.execute('SELECT reserved_usd, active_slot FROM bench_jobs WHERE id = ?', (job_id,)).fetchone()
            if row is None or row[1] is None:
                return self.get_job(job_id) if row else None
            reserved = row[0]
            charged = min(max(charged_usd or 0.0, 0.0), reserved)
            for key in allowance_keys:
                t.execute('UPDATE bench_allowances SET reserved_usd = MAX(reserved_usd - ?, 0), used_usd = used_usd + ? WHERE key = ?'
                          if not self.pg else
                          'UPDATE bench_allowances SET reserved_usd = GREATEST(reserved_usd - %s, 0), used_usd = used_usd + %s WHERE key = %s',
                          (reserved, charged, key))
            sets = {'state': state, 'charged_usd': charged, 'active_slot': None, **fields}
            t.execute(f"UPDATE bench_jobs SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?", [*sets.values(), job_id])
        return self.get_job(job_id)


def dumps(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)
