"""Hardware lookup across three separate data sources.

* ``data/hardware_catalog.json``  — general GPU specifications (VRAM, bandwidth, peak TFLOPS).
* ``data/nebius_platforms.json``  — Nebius platform availability (GPU memory, presets).
* ``data/nebius_spot_prices.json`` — Nebius preemptible prices (per region, with unit).

The sources are kept separate; this module only joins them for search results. Nebius
platforms are linked to catalogue specs through the explicit mapping below; an unmapped
platform is shown with "no catalogue specs" rather than guessed specs.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / 'data'

# Explicit platform -> general-catalogue mapping. Missing entries mean "no verified specs".
NEBIUS_PLATFORM_TO_GPU_ID = {
    'gpu-h100-sxm': 'nvidia_h100_sxm_80gb',
    'gpu-h200-sxm': 'nvidia_h200_sxm_141gb',
    'gpu-l40s-a': 'nvidia_l40s_48gb',
    'gpu-l40s-d': 'nvidia_l40s_48gb',
    'gpu-b200-sxm': 'nvidia_b200_sxm_180gb',
    'gpu-b200-sxm-a': 'nvidia_b200_sxm_180gb',
    'gpu-b300-sxm': 'nvidia_b300_sxm_288gb',
    'gpu-rtx6000': 'nvidia_rtx_pro_6000_server_96gb',
    'gpu-rtx6000-a': 'nvidia_rtx_pro_6000_server_96gb',
}
PRICE_UNITS = {'gpu hour': 'gpu_hour', 'gpu-hour': 'gpu_hour', 'instance hour': 'instance_hour',
               'hour': 'instance_hour'}
MAX_RESULTS = 10


def _load(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


@lru_cache
def catalog():
    return _load('hardware_catalog.json').get('gpus', {})


@lru_cache
def platforms():
    return [p for p in _load('nebius_platforms.json').get('items', [])
            if p.get('spec', {}).get('gpu_memory_gigabytes')]


@lru_cache
def prices():
    data = _load('nebius_spot_prices.json')
    return data.get('items', []), data.get('snapshotTime')


def _specs(gpu_id):
    gpu = catalog().get(gpu_id) if gpu_id else None
    if gpu is None:
        return None
    peak = gpu.get('peak_compute_tflops', {})
    return {'catalog_gpu_id': gpu_id, 'name': gpu['name'],
            'vram_gib': gpu.get('memory', {}).get('advertised_capacity_gb'),
            'memory_bandwidth_gb_s': gpu.get('memory', {}).get('bandwidth_gb_per_second'),
            'peak_tflops_fp32': peak.get('fp32'), 'peak_tflops_bf16': peak.get('bf16_tensor_dense'),
            'peak_tflops_fp16': peak.get('fp16_tensor_dense'),
            'source_url': gpu.get('source', {}).get('url'),
            'last_verified': gpu.get('source', {}).get('last_verified'),
            'review_status': gpu.get('source', {}).get('review_status', 'verified')}


def _price(item, snapshot):
    unit_title = item.get('price', {}).get('unitTitle', '')
    unit = PRICE_UNITS.get(unit_title.lower())
    try:
        value = float(item['price']['value'])
    except (KeyError, TypeError, ValueError):
        value = None
    return {'value': value, 'currency': str(item.get('price', {}).get('currency', '')).upper(),
            'unit': unit, 'unit_title': unit_title, 'region': item.get('region'),
            'preemptible': str(item.get('productTitle', '')).lower().startswith('preemptible'),
            'effective_time': item.get('effectiveTime'), 'snapshot_time': snapshot,
            'sku_id': item.get('skuId'), 'product_title': item.get('productTitle'),
            'source': 'data/nebius_spot_prices.json'}


@lru_cache
def options():
    """All selectable rows: Nebius platform×region prices, unpriced platforms, catalogue GPUs."""
    rows = []
    price_items, snapshot = prices()
    by_platform = {}
    for item in price_items:
        by_platform.setdefault(item.get('computeInstance', {}).get('platform'), []).append(item)
    known = set()
    for platform in platforms():
        name = platform['metadata']['name']
        known.add(name)
        spec = platform['spec']
        presets = [{'name': p['name'], 'gpu_count': p['resources'].get('gpu_count'),
                    'vcpu_count': p['resources'].get('vcpu_count'),
                    'ram_gib': p['resources'].get('memory_gibibytes')} for p in spec.get('presets', [])]
        base = {'provider': 'nebius', 'platform': name,
                'label': spec.get('short_human_readable_name') or spec.get('human_readable_name') or name,
                'vram_gib': spec.get('gpu_memory_gigabytes'),
                'vram_source': 'nebius_platforms.json gpu_memory_gigabytes (treated as GiB per GPU)',
                'presets': presets, 'specs': _specs(NEBIUS_PLATFORM_TO_GPU_ID.get(name))}
        items = by_platform.get(name, [])
        if not items:
            rows.append({**base, 'id': f'nebius:{name}', 'price': None,
                         'issues': ['No price for this platform in nebius_spot_prices.json.']})
        for item in items:
            rows.append({**base, 'id': f"nebius:{name}:{item.get('region')}", 'price': _price(item, snapshot),
                         'issues': []})
    for name, items in by_platform.items():
        if name in known:
            continue
        for item in items:
            title = str(item.get('productTitle', name)).replace('Preemptible ', '')
            specs = _specs(NEBIUS_PLATFORM_TO_GPU_ID.get(name))
            # Platform missing from nebius_platforms.json: fill GPU memory from the general catalogue.
            vram = specs['vram_gib'] if specs else None
            rows.append({'provider': 'nebius', 'platform': name, 'id': f"nebius:{name}:{item.get('region')}",
                         'label': title, 'vram_gib': vram,
                         'vram_source': 'hardware_catalog.json advertised_capacity_gb (platform not in nebius_platforms.json)'
                         if vram else None, 'presets': [], 'specs': specs, 'price': _price(item, snapshot),
                         'issues': [] if vram else ['Platform is priced but missing from nebius_platforms.json: '
                                                     'GPU memory unknown, enter it manually.']})
    for gpu_id in catalog():
        spec = _specs(gpu_id)
        rows.append({'provider': 'catalog', 'platform': None, 'id': f'catalog:{gpu_id}',
                     'label': spec['name'], 'vram_gib': spec['vram_gib'],
                     'vram_source': 'hardware_catalog.json advertised_capacity_gb (treated as GiB)',
                     'presets': [], 'specs': spec, 'price': None,
                     'issues': ['General catalogue entry: no provider price. Enter a price for cost.']})
    for row in rows:
        if row['specs'] is None:
            row['issues'].append('No general-catalogue specs (peak TFLOPS, bandwidth): runtime needs '
                                 'them entered manually.')
        elif row['specs']['peak_tflops_bf16'] is None:
            row['issues'].append('Catalogue has GPU memory only: enter peak TFLOPS and bandwidth for runtime.')
        elif row['specs']['review_status'] != 'verified':
            row['issues'].append('Catalogue specs are pending review (not live-verified).')
        row['search_text'] = ' '.join(str(x) for x in (row['label'], row['platform'], row.get('price') and
                                                       row['price']['region'], row['specs'] and row['specs']['name'])
                                      if x).lower()
    return rows


def runtime_ready(row):
    """True when the row has everything the runtime estimate needs (VRAM, BF16, FP32, bandwidth)."""
    spec = row.get('specs') or {}
    return bool(row.get('vram_gib') and spec.get('peak_tflops_bf16') and spec.get('peak_tflops_fp32')
                and spec.get('memory_bandwidth_gb_s'))


def choices():
    """GPU dropdown: Nebius AI Cloud entries (catalogue specs merged in) and the catalogue GPUs that Nebius
    does not offer. Only GPUs with every runtime parameter are listed; each GPU appears once per source."""
    rows = [r for r in options() if runtime_ready(r)]
    on_nebius = {r['specs']['catalog_gpu_id'] for r in rows if r['provider'] == 'nebius'}
    nebius = sorted((r for r in rows if r['provider'] == 'nebius'),
                    key=lambda r: (r['label'], r['price'] is None, (r['price'] or {}).get('region') or ''))
    catalog = sorted((r for r in rows if r['provider'] == 'catalog' and r['specs']['catalog_gpu_id'] not in on_nebius),
                     key=lambda r: r['label'])
    return {'nebius': nebius, 'catalog': catalog}


def search(query='', limit=MAX_RESULTS, offset=0):
    terms = [t for t in str(query).lower().replace('®', '').split() if t]
    matches = [r for r in options() if all(t in r['search_text'].replace('®', '') for t in terms)]
    matches.sort(key=lambda r: (r['provider'] != 'nebius', r['price'] is None,
                                r['price']['value'] if r['price'] and r['price']['value'] is not None else 0))
    limit = max(1, min(int(limit), MAX_RESULTS))
    return {'total': len(matches), 'offset': offset, 'limit': limit,
            'items': matches[offset:offset + limit],
            'price_snapshot': prices()[1],
            'notes': ['Nebius prices are preemptible (spot) snapshots from nebius_spot_prices.json, '
                      'per GPU-hour unless stated otherwise; on-demand prices differ.',
                      'GPU memory from Nebius/catalogue is treated as GiB per GPU; nvidia-smi '
                      'usually reports slightly less, which the 90% usable default covers.']}


def get_option(option_id):
    return next((r for r in options() if r['id'] == option_id), None)


def form_values(option, preset_gpu_count=None):  # LEGACY form keys; MVP uses gpt2_predict.hardware_values
    """Field values (and provenance) to autofill the hardware form from one option."""
    values, notes = {}, []
    specs = option.get('specs') or {}
    values['hardware.source'] = 'nebius' if option['provider'] == 'nebius' else 'catalog'
    values['hardware.selection_id'] = option['id']
    values['hardware.gpu_name'] = option['label']
    if option.get('vram_gib') is not None:
        values['hardware.total_vram_gib'] = float(option['vram_gib'])
        values['hardware.usable_vram_gib'] = round(0.9 * float(option['vram_gib']), 2)
    if specs:
        values['hardware.catalog_gpu_id'] = specs['catalog_gpu_id']
        for key in ('peak_tflops_fp32', 'peak_tflops_bf16', 'peak_tflops_fp16', 'memory_bandwidth_gb_s'):
            if specs.get(key) is not None:
                values['hardware.' + key] = specs[key]
    price = option.get('price')
    if price and price['value'] is not None and price['unit']:
        values['hardware.price_per_hour'] = price['value']
        values['hardware.price_unit'] = price['unit']
        values['hardware.price_region'] = price['region']
        values['hardware.price_note'] = (('Preemptible/spot ' if price['preemptible'] else '')
                                         + f"{price['currency']} per {price['unit'].replace('_', '-')}, "
                                         f"effective {price['effective_time']}")
    gpu_presets = [p['gpu_count'] for p in option.get('presets', []) if p.get('gpu_count')]
    if gpu_presets:
        values['hardware.gpus_per_instance'] = preset_gpu_count if preset_gpu_count in gpu_presets else min(gpu_presets)
        notes.append('Nebius presets: ' + ', '.join(f"{p['name']} ({p['gpu_count']} GPU, "
                                                    f"{p['ram_gib']} GiB RAM)" for p in option['presets']))
    return values, notes
