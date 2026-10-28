# GPU Planner


## What works

- Workload form → POST /api/estimate → runtime, GPU-hours and cost.
- Optional measured peak VRAM compared with user-entered GPU capacity.
- Input validation, API error messages and explicit assumptions.

This is a **benchmark extrapolation starter**, not a validated model-based resource predictor. Defaults are fictional examples. It does not automatically benchmark training, predict VRAM from architecture, fetch prices, allocate resources or call Nebius/NVIDIA models. Add those features separately if required by your hackathon.

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

Open http://localhost:5173 and click **Estimate resources**. Default output: 62,500 batches, 3.472 hours, $3.47. Frontend requests use Vite's /api proxy to reach Python. Keep both processes running. A package-lock.json is included; use npm ci for repeatable frontend installs.


## Calculation assumptions

Training batches = ceil(examples / batch_size) × epochs. Hours = batches × measured seconds per batch / 3600. One GPU, partial last batch retained, no gradient accumulation. Sequence length is context only: benchmarks must match actual model, lengths/padding, precision, optimizer, batch size, GPU and loading strategy. GPU-hours equal runtime only for this single-GPU case. Memory comparison is a simple capacity check, not a guarantee that training fits. Use consistent GB units for both measurements.

Time excludes provisioning/setup, validation, checkpoints and idle time. Cost excludes storage, transfers and taxes. EC2 bills instance uptime, including idle time. Update the estimator before claiming broader support.

See backend/README.md for endpoint details and explicit AI/benchmark placeholders.
