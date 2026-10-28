# GPU Planner


## What works

- **PyTorch only.** Two prediction profiles: a two-layer MLP regression (`Linear → ReLU → Linear`, MSE) and a classic GPT-2-style transformer LM (MHA, 4×H non-gated MLP, 16-bit activations, eager attention). Both require full training and no gradient checkpointing; anything else is reported as unsupported.
- Workload input (01 Model / 02 Training setup / 03 Dataset): upload model/training `.py` and JSON/YAML configs, dataset samples, or enter values manually. Python is parsed with `ast` only, never imported or executed. Hugging Face `config.json` lookup (metadata only, no weights, no remote code).
- Every value carries provenance (upload, static extraction, Hugging Face, dataset sample, catalogue, default, manual); uncertain/unresolved values are flagged in red and user edits are never silently overwritten.
- Hardware: search Nebius platforms/prices (`backend/data/nebius_*.json`) joined to the general GPU catalogue (`hardware_catalog.json`), or enter custom hardware. Usable VRAM defaults to 90%.
- Results: peak VRAM per GPU vs usable VRAM, fit, CPU/NVMe when applicable, fast/slow runtime scenarios, cost (elapsed hours × price unit), benchmark cost, hardware comparison, changes vs the original run, and “How we calculated this”.
- Benchmark mode: downloadable `run_benchmark.py` for a local GPU and result import. Running on Nebius is **not implemented** and is shown as unavailable.

Predictions are planning estimates from metadata and documented formulas, not measurements. See backend/README.md for endpoints.

## Prerequisites

Python 3.10+ and Node.js 22 LTS with npm. Run commands from the extracted project folder. Install dependencies from the internet on first setup.

## Start backend — terminal 1

```bash
cd backend
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API docs: http://127.0.0.1:8000/docs

## Start frontend — terminal 2

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and click **Open estimator**. Frontend requests use Vite's /api proxy to reach Python. Keep both processes running (one uvicorn worker: analysis sessions live in process memory).

## Calculation assumptions

Memory: `backend/app/services/estimators/gpu_memory.py` (model states, profile activation formulas, unsourced workspace/runtime/fragmentation allowances). Runtime: `gpu_runtime.py` (GEMM FLOP counts, roofline with editable efficiencies, data loading, optimizer, communication). Cost: `pricing.py` — `elapsed job hours × GPUs allocated × price per GPU-hour` (or per instance-hour), with startup counted once. Missing inputs produce explicit missing-information lists, not numbers.

See backend/README.md for endpoints.

## Prerequisites

Python 3.10+ and Node.js 22 LTS with npm. Run commands from the extracted project folder. Install dependencies from the internet on first setup.

## Start backend — terminal 1

```bash
cd backend
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Interactive API docs: http://127.0.0.1:8000/docs

## Start frontend — terminal 2

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and click **Open estimator**. Frontend requests use Vite's /api proxy to reach Python. Keep both processes running (one uvicorn worker: analysis sessions live in process memory).

## Calculation assumptions

Training batches = ceil(examples / batch_size) × epochs. Hours = batches × measured seconds per batch / 3600. One GPU, partial last batch retained, no gradient accumulation. Sequence length is context only: benchmarks must match actual model, lengths/padding, precision, optimizer, batch size, GPU and loading strategy. GPU-hours equal runtime only for this single-GPU case. Memory comparison is a simple capacity check, not a guarantee that training fits. Use consistent GB units for both measurements.

Time excludes provisioning/setup, validation, checkpoints and idle time. Cost excludes storage, transfers and taxes. EC2 bills instance uptime, including idle time. Update the estimator before claiming broader support.

See backend/README.md for endpoint details and explicit AI/benchmark placeholders.
