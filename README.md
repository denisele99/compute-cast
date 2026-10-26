# GPU Planner Starter

Minimal React + Vite frontend and Python + FastAPI backend. No cloud account, GPU, API key, database or paid service is needed to run this starter.

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

## Structure

```text
gpu-planner-starter/
  frontend/
    src/
      components/       # Place for reusable UI components
      api.js            # Backend HTTP requests
      App.jsx           # Current form and results
      main.jsx          # React entry point
      styles.css
    index.html
    vite.config.js
    package.json
    package-lock.json
  backend/
    app/
      main.py           # FastAPI application
      routes.py         # HTTP endpoints
      schemas.py        # Input/output validation
      config.py         # Environment configuration
      services/
        estimation.py   # Memory, runtime and pricing together
      clients/
        nebius.py       # AI API placeholder
        aws.py          # Cloud metadata/pricing placeholder
    tests/test_api.py
    .env.example
    requirements.txt
    requirements-dev.txt
    README.md
  .gitignore
  README.md
  TODO.md
```

Python package directories also contain `__init__.py`.

## Calculation assumptions

Training batches = ceil(examples / batch_size) × epochs. Hours = batches × measured seconds per batch / 3600. One GPU, partial last batch retained, no gradient accumulation. Sequence length is context only: benchmarks must match actual model, lengths/padding, precision, optimizer, batch size, GPU and loading strategy. GPU-hours equal runtime only for this single-GPU case. Memory comparison is a simple capacity check, not a guarantee that training fits. Use consistent GB units for both measurements.

Time excludes provisioning/setup, validation, checkpoints and idle time. Cost excludes storage, transfers and taxes. EC2 bills instance uptime, including idle time. Update the estimator before claiming broader support.

See backend/README.md for endpoint details and explicit AI/benchmark placeholders.

## GitHub

Create an EMPTY GitHub repository (without an initial README), then from this folder:

```bash
git init
git add .
git commit -m "Add GPU planner starter"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Keep secrets in backend environment variables, never frontend code or Git. This archive deliberately contains no .git history.

## AI coding tools and deployment

Ask your coding assistant to preserve the /api/estimate request/response contract while improving the React UI. Browser-based JS tools can edit/preview the frontend; they do not automatically run the Python backend. For a UI-only preview, use an explicitly labeled mock response until the API is available.

For production, run npm run build in frontend and serve dist through your frontend host. The development proxy is NOT part of the built site. Configure your host/reverse proxy to forward /api to the deployed backend, or change the frontend API URL and configure restricted CORS on FastAPI. Deploy the backend separately and keep API keys server-side. npm run preview previews static assets only; it is not the complete deployment setup.
