#!/usr/bin/env python3
"""Download GPU specifications and convert to Compute Cast's catalogue schema.

Standard library only; Python 3.10+. No pricing, measurements or GPU required.
Imported community specifications require review against the exact GPU datasheet.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
COMMIT = '578a5d7c9d148158822069ce2b43bab0133ff306'
UPSTREAM = f'https://raw.githubusercontent.com/gmasse/gpu-specs/{COMMIT}'
DATA_URL = UPSTREAM + '/data/specs.json'
LICENSE_URL = UPSTREAM + '/LICENSE'
DATA_SHA256 = 'f5bf9008815bd36505492e8ca739cc730a0d370f80e00f00aaa9753090402191'
LICENSE_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
# These exact variants already have curated entries in this repository.
IDS = {'rtx4090': 'nvidia_geforce_rtx_4090', 'a100_pcie_80gb': 'nvidia_a100_80gb_pcie'}
COMPUTE_FIELDS = {'fp32': 'fp32', 'fp64': 'fp64', 'fp16': 'fp16',
                  'fp16_tensor_dense': 'fp16_tensor_core',
                  'bf16_tensor_dense': 'bf16_tensor_core',
                  'tf32_tensor_dense': 'tf32_tensor_core'}
MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024


def download(url, expected_sha256):
    request = Request(url, headers={'User-Agent': 'ComputeCast-HardwareCatalog/1.0'})
    # urllib keeps normal system TLS certificate verification enabled.
    with urlopen(request, timeout=30) as response:
        if not response.geturl().startswith('https://'):
            raise ValueError('Refusing a non-HTTPS redirect')
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError('Source exceeds the 4 MiB download limit')
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Downloaded checksum does not match the pinned source; no output written')
    return content


def positive(value, label):
    if value is None or value in ('', '?', 'N/A', 'NA'):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label}: expected a finite nonnegative number or null')
    # Zero throughput represents unsupported/unspecified, not usable performance.
    return value if value > 0 else None


def text(value, label):
    if value is not None and not isinstance(value, str):
        raise ValueError(f'{label}: expected text or null')
    return value


def convert(data, retrieved_on):
    if not isinstance(data, dict) or not isinstance(data.get('_header'), dict):
        raise ValueError('Expected the gpu-specs JSON object with _header metadata')
    expected = {'gpu_memory': 'GB', 'memory_bandwidth': 'GB/s',
                **{field: 'TFLOPS' for field in COMPUTE_FIELDS.values()}}
    for field, unit in expected.items():
        if data['_header'].get(field, {}).get('unit') != unit:
            raise ValueError(f'Unexpected or missing source unit for {field}; expected {unit}')
    gpus = {}
    for key, row in data.items():
        if key.startswith('_'):
            continue
        if not isinstance(row, dict) or not isinstance(row.get('name'), str) or not row['name'].strip():
            raise ValueError(f'{key}: missing GPU name')
        manufacturer = text(row.get('manufacturer'), f'{key}.manufacturer')
        if not manufacturer:
            raise ValueError(f'{key}: manufacturer is required for stable GPU identity')
        identifier = IDS.get(key, re.sub(r'[^a-z0-9]+', '_', f'{manufacturer}_{key}'.lower()).strip('_'))
        if identifier in gpus:
            raise ValueError(f'Duplicate normalized GPU identity: {identifier}')
        capacity = row.get('gpu_memory')
        capacity_options = None
        if isinstance(capacity, str) and re.fullmatch(r'\d+(?:/\d+)+', capacity):
            capacity_options = [int(value) for value in capacity.split('/')]
            memory = None  # Do not choose between, for example, 16 GB and 32 GB V100s.
        else:
            memory = positive(capacity, f'{key}.gpu_memory')
        bandwidth = positive(row.get('memory_bandwidth'), f'{key}.memory_bandwidth')
        peaks = {target: positive(row.get(source), f'{key}.{source}') for target, source in COMPUTE_FIELDS.items()}
        urls = row.get('sources', [])
        if not isinstance(urls, list) or any(not isinstance(url, str) or not url.startswith('https://') for url in urls):
            raise ValueError(f'{key}: invalid source URLs')
        name = row['name']
        form = 'SXM' if 'sxm' in name.lower() else 'PCIe' if 'pcie' in name.lower() else None
        gpus[identifier] = {
            'name': f'{manufacturer} {name}', 'manufacturer': manufacturer,
            'architecture': text(row.get('architecture'), f'{key}.architecture'),
            'form_factor': form,
            'memory': {'advertised_capacity_gb': memory,
                       'advertised_capacity_options_gb': capacity_options,
                       'type': text(row.get('memory_type'), f'{key}.memory_type'),
                       'bandwidth_gb_per_second': bandwidth},
            'peak_compute_tflops': peaks,
            'source': {'url': DATA_URL, 'upstream_key': key, 'references': urls,
                       'retrieved_on': retrieved_on, 'last_verified': None,
                       'verification_status': 'community_data_requires_datasheet_review'},
            'notes': ['Imported from community data; exact GPU variant and values require review.',
                      'Advertised memory GB is retained as labelled, not asserted to be usable GiB.',
                      'Sparse throughput is excluded; missing dense throughput remains null.'],
        }
    if not gpus:
        raise ValueError('Source contains no GPU records')
    return {'schema_version': 1, 'gpus': gpus,
            'provenance': {'dataset': 'gmasse/gpu-specs', 'commit': COMMIT,
                           'url': DATA_URL, 'sha256': DATA_SHA256,
                           'license': 'Apache-2.0', 'license_url': LICENSE_URL,
                           'retrieved_on': retrieved_on,
                           'conversion': 'Compute Cast schema conversion; source fields selected/renamed; pricing excluded.'}}


def merge_curated(imported, curated):
    if not isinstance(curated, dict) or curated.get('schema_version') != 1 or not isinstance(curated.get('gpus'), dict):
        raise ValueError('Curated catalogue must have schema_version 1 and a gpus object')
    # Preserve whole curated records, including verification dates and unknowns.
    imported['gpus'].update(curated['gpus'])
    return imported


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'backend/data/hardware_catalog.downloaded.json')
    parser.add_argument('--merge', type=Path, default=ROOT / 'backend/data/hardware_catalog.json', help='Existing curated catalogue to preserve (default: project catalogue)')
    parser.add_argument('--no-merge', action='store_true', help='Generate only imported records')
    parser.add_argument('--force', action='store_true', help='Explicitly allow replacing existing output and its licence sidecar')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    license_path = output.with_name(output.stem + '.LICENSE.txt')
    if not args.no_merge and output == args.merge.resolve():
        parser.error('Choose an output separate from the curated merge input. Review the generated file before replacing the original.')
    if not args.force and (output.exists() or license_path.exists()):
        parser.error('Output or licence sidecar already exists. Choose another path or use --force.')
    try:
        # Validate the merge input before network work. Never modify it.
        curated = None if args.no_merge else json.loads(args.merge.read_text(encoding='utf-8'))
        raw = download(DATA_URL, DATA_SHA256)
        license_content = download(LICENSE_URL, LICENSE_SHA256).decode('utf-8')
        result = convert(json.loads(raw), datetime.now(timezone.utc).date().isoformat())
        if curated is not None:
            result = merge_curated(result, curated)
        serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
        output.parent.mkdir(parents=True, exist_ok=True)
        mode = 'w' if args.force else 'x'
        # Include upstream licence and identify the conversion in JSON provenance.
        with license_path.open(mode, encoding='utf-8') as handle:
            handle.write(license_content)
        with output.open(mode, encoding='utf-8') as handle:
            handle.write(serialized)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f'Catalogue generation failed: {exc}\n')
    print(f'Wrote {len(result["gpus"])} GPU records: {output}')
    print(f'Upstream licence: {license_path}')
    print('No prices included. Imported entries require datasheet/variant review; missing dense throughput stays null.')


if __name__ == '__main__':
    main()
