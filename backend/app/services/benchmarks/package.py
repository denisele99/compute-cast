"""Local benchmark package: the SAME trusted runner, registry, model code, pinned requirements and
README as the cloud image, plus the validated benchmark_config.json. No credentials, no generated code."""
from __future__ import annotations

import io
import json
import zipfile

from .registry import package_dir

FILES = ('benchmark.py', 'model_registry.py', 'models/__init__.py', 'models/decoder.py', 'requirements.txt', 'README.md')
ROOT = 'compute-cast-benchmark'


def build_zip(config):
    base = package_dir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.writestr(f'{ROOT}/{name}', (base / name).read_bytes())
        z.writestr(f'{ROOT}/benchmark_config.json', json.dumps(config, indent=2, sort_keys=False, allow_nan=False) + '\n')
    return buf.getvalue()
