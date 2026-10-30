# Backend

Start using the root README commands. Python 3.10+.
Optionally copy `.env.example` to `.env`; existing environment variables take precedence.

Status: **active** = used by the GPT-2 MVP; **legacy** = retained, marked "NOT USED BY THE GPT-2 MVP", still tested.

| File | Status | Purpose |
| --- | --- | --- |
| app/main.py | active | FastAPI app and in-memory analysis store |
| app/api/routes/ | active | parse (schema, HF list), hardware, analysis (sessions, uploads, ZIP, examples, HF), estimate, benchmark, recommender |
| app/parsers/gpt2_inputs.py | active | MVP schema from `data/mvp_feature_contract.json`, validation, estimator objects |
| app/parsers/gpt2_extract.py | active | Static extraction (AST/JSON/YAML/CSV), file classification, conflicts, ZIP, HF snapshot, examples |
| app/services/estimators/estimate_gpt2.py | active | Analytical GPT-2 estimator (see ESTIMATOR.md); CLI |
| app/services/estimators/gpt2_predict.py | active | Fit, runtime, cost, hardware comparison, catalogue autofill |
| app/services/estimators/pricing.py | active | Cost from elapsed hours and price unit |
| app/parsers/detect_hardware_lookup.py | active | Joins hardware catalogue, Nebius platforms and Nebius prices for search |
| app/parsers/dataset_sample.py | active | Optional dataset-sample statistics |
| app/parsers/static_extract.py | partly | AST helpers active; profile mappers legacy |
| app/services/upload_cache.py, analysis_store.py | active | Per-analysis upload cache and session state/snapshots |
| app/services/benchmarks/ | active | Benchmark mode: config builder, local package, results validation, auth, durable store, Nebius jobs (see ../BENCHMARK.md) |
| ../benchmark/ | active | Trusted runner, model registry and GPT-2 decoder shared by the local package and the container image |
| app/clients/huggingface.py | active | config.json fetch (metadata only) |
| app/clients/nebius.py | placeholder | Future Token Factory adapter (not used for benchmarks) |
| app/parsers/normalize_inputs.py, collect_workload.py | legacy | MLP / classic-transformer schema and collector |
| app/services/estimators/gpu_memory.py, gpu_runtime.py, profiles.py, estimator.py | legacy | MLP / classic-transformer estimators |

Main endpoints (all under /api): `GET /health`, `GET /schema`, `GET /hf/search`, `GET /hardware/search`,
`GET /hardware/options/{id}`, `GET /hardware/local-command`, `POST /analysis`, `DELETE /analysis/{id}`,
`POST /analysis/{id}/close`, `POST /analysis/{id}/uploads`, `POST /analysis/{id}/huggingface`,
`POST /analysis/{id}/validate`, `POST /analysis/{id}/predict`, `POST /analysis/{id}/compare`,
`GET /analysis/{id}/files/estimator_input.json`, `POST /analysis/{id}/examples/{example_id}`, `POST /analysis/{id}/benchmark/{config|package|results}`,
`GET /bench/status`, `POST|GET /bench/session`, `POST|GET /bench/jobs`, `GET /bench/jobs/{id}`, `POST /bench/jobs/{id}/cancel`.
Cloud benchmark routes are off unless `BENCH_CLOUD_ENABLED=1`; settings in `.env.example`, deployment in ../BENCHMARK.md.

Upload cache: `$UPLOAD_CACHE_DIR` or `<system temp>/compute_cast_uploads/<analysis_id>/` (outside the repo, not served).
Limits: 1 MB per script/config, 25 MB per dataset sample, 30 MB per ZIP, 20 files / 60 MB per analysis. Cleared on reset, on page close
(best effort beacon) and by a 2-hour inactivity expiry sweep. Uploads are only parsed statically, never executed.

Collector CLI: `python -m app.parsers.collect_workload --profile mlp_regression --model-config m.json --training-config t.json --output workload.json`

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
# optional: run the generated benchmark script on CPU with a torch-enabled Python
BENCH_PYTHON=/path/to/python-with-torch python -m unittest discover -s tests -v
```
