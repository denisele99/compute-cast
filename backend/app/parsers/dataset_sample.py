"""Statistics from an uploaded dataset SAMPLE (CSV/TSV, JSONL, JSON array of records).

Sample statistics describe the uploaded file only. Its row count and file size are never
used as the full training-set size; the full example count and size are separate inputs.
No tokenizer is run: token statistics come only from integer ``input_ids`` lists.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

MAX_ROWS = 10_000
SUPPORTED_FORMATS = {
    '.csv': 'Comma-separated values with a header row (numeric columns → feature/target counts).',
    '.tsv': 'Tab-separated values with a header row.',
    '.jsonl': 'One JSON object per line; integer "input_ids"/"labels" lists give sequence lengths.',
    '.json': 'A JSON array of objects (same rules as JSONL).',
}
TARGET_NAMES = {'target', 'targets', 'label', 'labels', 'y'}
TOKEN_KEYS = ('input_ids', 'tokens', 'token_ids')


def _num(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _records(path, suffix):
    if suffix in ('.csv', '.tsv'):
        with open(path, newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle, delimiter='\t' if suffix == '.tsv' else ',')
            for i, row in enumerate(reader):
                if i >= MAX_ROWS:
                    return True
                yield row
        return False
    if suffix == '.jsonl':
        with open(path, encoding='utf-8') as handle:
            for i, line in enumerate(handle):
                if i >= MAX_ROWS:
                    return True
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f'Line {i + 1} is not a JSON object.')
                    yield value
        return False
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('JSON dataset sample must be an array of objects.')
    for i, value in enumerate(data[:MAX_ROWS]):
        if not isinstance(value, dict):
            raise ValueError(f'Item {i} is not a JSON object.')
        yield value
    return len(data) > MAX_ROWS


def analyze(path, display_name=None):
    path = Path(path)
    name = display_name or path.name
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f'Unsupported dataset sample format {suffix!r}. Supported: '
                         + ', '.join(SUPPORTED_FORMATS))
    rows, columns, numeric, ints, token_lengths, text_lengths = 0, [], {}, {}, {}, {}
    gen = _records(path, suffix)
    truncated = False
    while True:
        try:
            row = next(gen)
        except StopIteration as stop:
            truncated = bool(stop.value)
            break
        rows += 1
        for key, value in row.items():
            if key not in columns:
                columns.append(key)
            if isinstance(value, list):
                if value and all(isinstance(x, int) and not isinstance(x, bool) for x in value):
                    token_lengths.setdefault(key, []).append(len(value))
                continue
            n = _num(value)
            if n is not None and not (isinstance(n, float) and not math.isfinite(n)):
                numeric[key] = numeric.get(key, 0) + 1
                is_int = (isinstance(value, int) and not isinstance(value, bool)) or (
                    isinstance(value, str) and value.strip().lstrip('-').isdigit())
                ints[key] = ints.get(key, 0) + int(is_int)
            elif isinstance(value, str):
                text_lengths.setdefault(key, []).append(len(value.split()))
    if rows == 0:
        raise ValueError('The sample contains no records.')
    numeric_cols = [c for c in columns if numeric.get(c) == rows]
    stats = {'file': name, 'format': suffix, 'sample_rows': rows, 'sample_truncated_at': MAX_ROWS if truncated else None,
             'sample_file_bytes': path.stat().st_size, 'columns': columns,
             'numeric_columns': numeric_cols,
             'integer_columns': [c for c in numeric_cols if ints.get(c) == rows],
             'note': 'Sample statistics only: the sample row count and file size are NOT the full '
                     'training set. Enter the full example count and size separately.'}
    findings, notes = [], []

    def add(field, value, status, message):
        findings.append({'field': field, 'value': value, 'status': status, 'source': 'dataset_sample',
                         'file': name, 'line': None, 'message': message})

    if token_lengths:
        key = next((k for k in TOKEN_KEYS if k in token_lengths), next(iter(token_lengths)))
        lengths = sorted(token_lengths[key])
        p95 = lengths[min(len(lengths) - 1, math.ceil(0.95 * len(lengths)) - 1)]
        stats['token_lengths'] = {'column': key, 'min': lengths[0], 'max': lengths[-1],
                                  'mean': sum(lengths) / len(lengths), 'p95': p95}
        add('training.sequence_length', lengths[-1], 'uncertain',
            f'Longest {key} in the sample is {lengths[-1]} tokens; the full dataset or truncation '
            'setting may differ.')
        add('training.input_dtype', 'int64', 'uncertain', 'Integer token ids; PyTorch default int64.')
        if any(k in token_lengths for k in ('labels', 'targets')) or 'labels' in columns:
            add('training.target_dtype', 'int64', 'uncertain', 'Integer labels; PyTorch default int64.')
        if len(set(lengths)) > 1:
            notes.append(f'{name}: sample lengths vary ({lengths[0]}–{lengths[-1]}); with '
                         f'longest-in-batch padding the effective length depends on batching '
                         f'(sample mean {stats["token_lengths"]["mean"]:.1f}).')
    elif numeric_cols:
        targets = [c for c in numeric_cols if c.lower() in TARGET_NAMES or c.lower().startswith('target')]
        stats['target_like_columns'] = targets
        if targets:
            add('model.input_size', len(numeric_cols) - len(targets), 'uncertain',
                f'{len(numeric_cols)} numeric columns minus target-like columns {targets}.')
            add('model.output_size', len(targets), 'uncertain', f'Target-like columns: {targets}.')
        else:
            notes.append(f'{name}: {len(numeric_cols)} numeric columns; no column is named like a '
                         'target, so input/output sizes were not inferred.')
        if [c for c in numeric_cols if c not in stats['integer_columns']]:
            add('training.input_dtype', 'float32', 'uncertain',
                'Text formats carry no dtype; float32 is the usual tensor dtype for these features.')
            add('training.target_dtype', 'float32', 'uncertain',
                'Text formats carry no dtype; float32 is the usual MSE target dtype.')
    if text_lengths:
        stats['text_word_counts'] = {k: {'max': max(v), 'mean': sum(v) / len(v)} for k, v in text_lengths.items()}
        notes.append(f'{name}: text columns found; word counts are not tokenizer tokens, so no '
                     'sequence length was inferred from them.')
    return {'kind': 'dataset_sample', 'file': name, 'stats': stats, 'findings': findings,
            'notes': notes, 'unsupported': [], 'profile_hint': None}


def parse_text_records(text, suffix):
    """Helper for tests: analyze in-memory content."""
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix=suffix, delete=False, encoding='utf-8') as handle:
        handle.write(text)
    try:
        return analyze(handle.name)
    finally:
        Path(handle.name).unlink(missing_ok=True)

