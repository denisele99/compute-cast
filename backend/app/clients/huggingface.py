"""Hugging Face Hub metadata client: model search and config.json only.

Never downloads weights, tokenizers or Python files and never executes remote code
(``trust_remote_code`` / ``auto_map`` entries are only reported). Uses the public HTTP API
anonymously with short timeouts and a response-size cap.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

HUB = 'https://huggingface.co'
MAX_BYTES = 1_000_000
REPO_ID = re.compile(r'^[A-Za-z0-9][\w.-]{0,95}(/[A-Za-z0-9][\w.-]{0,95})?$')
REVISION = re.compile(r'^[\w.-]{1,64}$')


class HubError(RuntimeError):
    """Hub unavailable, model missing/gated, or response invalid."""


def _get_json(url, opener=None, timeout=10):
    request = urllib.request.Request(url, headers={'Accept': 'application/json',
                                                   'User-Agent': 'compute-cast/0.4'})
    open_ = opener or urllib.request.urlopen
    try:
        with open_(request, timeout=timeout) as response:
            data = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HubError('The model is gated/private or the Hub refused the request; enter '
                           'architecture values manually or upload config.json.') from exc
        if exc.code == 404:
            raise HubError('Model or config.json not found on the Hugging Face Hub.') from exc
        raise HubError(f'Hugging Face Hub returned HTTP {exc.code}.') from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HubError('Could not reach the Hugging Face Hub from the backend (network blocked '
                       'or offline). Upload config.json or enter values manually.') from exc
    if len(data) > MAX_BYTES:
        raise HubError('Hub response exceeds 1 MB; refusing to parse it.')
    try:
        return json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HubError('Hub returned invalid JSON.') from exc


def search_models(query, limit=10, opener=None):
    query = str(query).strip()
    if not query or len(query) > 100:
        raise ValueError('Enter a search term up to 100 characters.')
    url = f'{HUB}/api/models?' + urllib.parse.urlencode(
        {'search': query, 'limit': max(1, min(int(limit), 10)), 'sort': 'downloads', 'direction': -1})
    items = _get_json(url, opener)
    if not isinstance(items, list):
        raise HubError('Unexpected search response.')
    results = []
    for item in items[:10]:
        if not isinstance(item, dict) or not isinstance(item.get('id', item.get('modelId')), str):
            continue
        tags = [t for t in item.get('tags', []) if isinstance(t, str)]
        results.append({'id': item.get('id', item.get('modelId')), 'downloads': item.get('downloads'),
                        'pipeline_tag': item.get('pipeline_tag'), 'library_name': item.get('library_name'),
                        'pytorch': 'pytorch' in tags or item.get('library_name') == 'transformers',
                        'custom_code': 'custom_code' in tags})
    return results


def fetch_config(repo_id, revision='main', opener=None):
    if not REPO_ID.match(str(repo_id)) or '..' in repo_id:
        raise ValueError('Enter a model id such as "openai-community/gpt2".')
    if not REVISION.match(str(revision)):
        raise ValueError('Invalid revision.')
    url = f'{HUB}/{urllib.parse.quote(repo_id, safe="/")}/resolve/{urllib.parse.quote(revision)}/config.json'
    config = _get_json(url, opener)
    if not isinstance(config, dict):
        raise HubError('config.json is not a JSON object.')
    return config
