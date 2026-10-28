# Backend

Start using the root README commands. Python 3.10+.
Optionally copy `.env.example` to `.env`; existing environment variables take precedence.

| File | Purpose |
| --- | --- |
| app/main.py | Create the FastAPI application and the in-memory analysis store |
| app/api/routes/ | HTTP routes: parse (schema, Hugging Face), hardware, analysis (sessions, uploads), estimate, benchmark, recommender |
| app/parsers/static_extract.py | AST / JSON / YAML extraction from uploaded model and training files (never executes them) |
| app/parsers/dataset_sample.py | Statistics from uploaded dataset samples (CSV/TSV/JSONL/JSON) |
| app/parsers/normalize_inputs.py | Canonical field schema, validation, normalized estimator bundle |
| app/parsers/collect_workload.py | Metadata-only collector: settings → workload.json (also a CLI) |
| app/parsers/detect_hardware_lookup.py | Joins hardware catalogue, Nebius platforms and Nebius prices for search |
| app/services/estimators/ | profiles.py (profile formulas), gpu_memory.py, gpu_runtime.py, pricing.py, estimator.py (adapters) |
| app/services/upload_cache.py, analysis_store.py | Per-analysis upload cache and session state/snapshots |
| app/services/benchmark_service.py, scripts/run_benchmark.py | Local benchmark script generation and result import |
| app/clients/huggingface.py | Hub search and config.json fetch (metadata only) |
| app/clients/nebius.py | Nebius job launch placeholder; raises NotImplementedError |

Main endpoints (all under /api): `GET /health`, `GET /schema`, `GET /hf/search`, `GET /hardware/search`,
`GET /hardware/options/{id}`, `GET /hardware/local-command`, `POST /analysis`, `DELETE /analysis/{id}`,
`POST /analysis/{id}/close`, `POST /analysis/{id}/uploads`, `POST /analysis/{id}/huggingface`,
`POST /analysis/{id}/validate`, `POST /analysis/{id}/predict`, `POST /analysis/{id}/compare`,
`GET /analysis/{id}/files/{workload.json|hardware.json}`, `POST /analysis/{id}/benchmark/{plan|script|import}`.
`POST /analysis/{id}/benchmark/nebius` returns 501: no cloud resources are ever provisioned.

Upload cache: `$UPLOAD_CACHE_DIR` or `<system temp>/compute_cast_uploads/<analysis_id>/` (outside the repo, not served).
Limits: 1 MB per script/config, 25 MB per dataset sample, 20 files / 60 MB per analysis. Cleared on reset, on page close
(best effort beacon) and by a 2-hour inactivity expiry sweep. Uploads are only parsed statically, never executed.

Collector CLI: `python -m app.parsers.collect_workload --profile mlp_regression --model-config m.json --training-config t.json --output workload.json`

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
# optional: run the generated benchmark script on CPU with a torch-enabled Python
BENCH_PYTHON=/path/to/python-with-torch python -m unittest discover -s tests -v
```
