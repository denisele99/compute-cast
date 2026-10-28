# Compute Cast — GPU training planner

Plan PyTorch training with uploaded Python scripts and JSON/YAML configs, or use
manual inputs. The connected planner provides input review, Nebius GPU snapshot
search, memory fit checks, prediction, and a local benchmark download/import flow.
**Only PyTorch is supported.** Uploaded Python is statically inspected and never
executed on the API server.

## Start development

Python 3.10+ and Node.js 22 LTS with npm are recommended. Install once:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Keep both processes running. The frontend's `/api` proxy connects to FastAPI on
port 8000. API documentation is available at `/docs` on the backend.

## Inputs and estimates

Review all extracted fields before proceeding. Unsupported or uncertain values
are marked in red; defaults appear in grey, and optional fields may remain empty.
The general configuration preserves original metadata, while a normalized subset
is sent to the existing memory and runtime estimators.

Search shows up to ten matching regional Nebius offers from the checked-in
snapshot files. Prices are USD per GPU-hour, not live quotes. Some priced
platforms have no memory definition in the supplied platform file; enter their
verified capacity manually. Manufacturer GB converts to GiB. The default usable
memory limit is 90% of each GPU's capacity.

Available GPUs and GPUs actually used by the script are separate. Select DDP or
an explicit sharding strategy for multiple active GPUs; this does not modify your
training script. Advanced CPU/RAM/SSD/NVMe capacities are saved as metadata, but
capacity alone cannot establish CPU demand, I/O throughput or offload behavior.

Runtime uses matching measured microbatch timing, or the existing metadata-only
runtime planner when explicit FLOPs, HBM traffic, hardware throughput/bandwidth
and data-ready assumptions are provided. Planning scenarios are not confidence
bounds. Incomplete inputs leave runtime/cost unknown. Peak VRAM requires a
matching measured peak, or parameter count plus activation and auxiliary storage
assumptions. Parameter count alone cannot establish full peak memory or fit.

Uploads, the immutable first-run snapshot and current inputs live only in page
memory. Refresh, close or New analysis clears them. The review/recommendation
bundle download includes uploaded source files. Optional code recommendations can
send cached source to the existing Nebius model provider after explicit consent;
paid API charges may apply. This needs the OpenAI SDK and NEBIUS_API_KEY on the
backend. Proposed files are reviewed and applied only to the page cache, never
automatically executed or written into the repository.

## Local benchmark

From the results page, select Benchmark, then download the runner and current
config. A trusted training file must expose `build_benchmark(model_path,
data_path)`, returning a callable for one complete training microbatch. Keep
model code, data, project imports and a CUDA-compatible PyTorch install available:

```bash
python run_benchmark.py --model model.py --train train.py --data_sample data --config analysis_config.json --output benchmark.json
```

The runner executes the adapter on the user's local CUDA GPU and supports one
GPU. Import its report into the planner, review it, then rerun the estimate.
The API rejects reports for a different configuration. Full-run initialization,
checkpoint peaks and other GPU processes are outside the measured scope.

Nebius cloud benchmarking is unavailable: no GPU jobs or paid calls are submitted
by estimation. Provisioning, billing consent, credentials and cleanup require a
separate integration.

## Checks

```bash
cd backend
python -m unittest discover -s tests -v
```

```bash
cd frontend
npm run build
npx playwright install chromium
npm run test:e2e
```

Keep backend and Vite running for browser tests. An existing browser can be set
with `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`; `PLANNER_BASE_URL` may target a local
production preview instead. See [the change report](docs/WORKFLOW_CHANGE_REPORT.md)
for calculation changes, test evidence and outstanding work.

The original `/api/estimate` and `/api/gpus` endpoints remain for compatibility,
including their illustrative defaults. The current planner uses `/api/analysis/*`
and `/api/hardware` instead.
