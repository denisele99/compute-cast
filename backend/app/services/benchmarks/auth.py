"""Demo-password sessions for cloud benchmarks.

* The password comes only from BENCH_DEMO_PASSWORD (a Render secret); it is never in source or the frontend.
* Successful login creates a server-side session row and returns a bearer token
  ``base64url(payload).base64url(HMAC-SHA256(BENCH_SESSION_SECRET, payload))`` with an expiry.
  A bearer header (not a cookie) works across the GitHub Pages -> Render origins without
  third-party-cookie/SameSite issues. The frontend keeps it in sessionStorage only.
* Failed attempts are rate-limited per client (hashed IP) in durable storage.
* The display name is a label only; allowances are keyed by the random server session id.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


class AuthError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode()


def _unb64(text):
    return base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))


def client_key(ip):
    return hashlib.sha256(f'login:{ip}'.encode()).hexdigest()[:32]


def login(store, policy, display_name, password, ip):
    if not policy.demo_password or not policy.session_secret:
        raise AuthError(503, 'Cloud benchmarks are not configured on this server.')
    name = (display_name or '').strip()
    if not 1 <= len(name) <= 40:
        raise AuthError(422, 'Enter a display name (1–40 characters).')
    key = client_key(ip)
    if store.login_attempts(key, policy.login_window_seconds) >= policy.login_max_attempts:
        raise AuthError(429, 'Too many failed attempts. Try again later.')
    ok = hmac.compare_digest(hashlib.sha256(password.encode()).digest(), hashlib.sha256(policy.demo_password.encode()).digest())
    if not ok:
        store.record_failed_login(key)
        raise AuthError(401, 'Incorrect password.')
    session = store.create_session(name, policy.session_ttl_seconds)
    store.ensure_allowance(f"session:{session['id']}", policy.session_allowance_usd)
    payload = _b64(json.dumps({'sid': session['id'], 'exp': session['expires_at']}).encode())
    sig = _b64(hmac.new(policy.session_secret.encode(), payload.encode(), hashlib.sha256).digest())
    return {'token': f'{payload}.{sig}', 'expires_at': session['expires_at'], 'display_name': name,
            'note': 'The display name is a label, not a verified identity.'}


def authenticate(store, policy, header):
    if not policy.session_secret:
        raise AuthError(503, 'Cloud benchmarks are not configured on this server.')
    if not header or not header.startswith('Bearer '):
        raise AuthError(401, 'Sign in to run cloud benchmarks.')
    token = header[7:].strip()
    try:
        payload, sig = token.split('.')
        expected = _b64(hmac.new(policy.session_secret.encode(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            raise ValueError
        data = json.loads(_unb64(payload))
    except (ValueError, json.JSONDecodeError):
        raise AuthError(401, 'Invalid session.') from None
    if data.get('exp', 0) < time.time():
        raise AuthError(401, 'Session expired; sign in again.')
    session = store.get_session(data.get('sid', ''))
    if session is None:
        raise AuthError(401, 'Session expired; sign in again.')
    return session
