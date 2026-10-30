"""Load the shared trusted model registry from the repository's benchmark/ package.

The backend validates configurations with exactly the module the runner uses, so rules cannot drift.
Override the location with BENCH_PACKAGE_DIR (for example when deploying the backend separately).
"""
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[4] / 'benchmark'


def package_dir() -> Path:
    return Path(os.environ.get('BENCH_PACKAGE_DIR') or DEFAULT_DIR)


@lru_cache
def registry():
    path = package_dir() / 'model_registry.py'
    spec = importlib.util.spec_from_file_location('compute_cast_benchmark_registry', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Benchmark registry not found at {path}; set BENCH_PACKAGE_DIR.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
