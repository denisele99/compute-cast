# Compute Cast — GPU planner (MVP)

Analytical, metadata-only estimates of **peak GPU memory, training time and cost** for fine-tuning a
**standard GPT-2** model in **PyTorch** on **one BF16-capable GPU**. Nothing you upload is executed,
no weights are downloaded and nothing runs on a GPU to produce the estimate.

- **Model:** pick `openai-community/gpt2` or `distilbert/distilgpt2` (config.json only), **or** upload your own config/script.
- **Files:** one drop zone takes multiple files or a ZIP; content decides whether a file feeds Model, Training or Data. Example file sets are one click away.
- **Unsupported settings** in files (FP16, gradient checkpointing, LoRA, dynamic padding, multi-GPU, …) are shown as conflicts; you can explicitly choose a supported hypothetical scenario.
- **Hardware:** Nebius GPUs with preemptible prices, the general GPU catalogue, or a custom GPU.
- **Results:** capacity vs usable VRAM, memory breakdown, training compute time, cost, benchmark cost, hardware comparison, changes vs the original run, and "How we calculated this".
- **Benchmark mode:** review a validated `benchmark_config.json`, then either download a local package (same trusted runner as the cloud image) and import `results.json`, or run it on Nebius AI Cloud Serverless Jobs (password-protected, allowance-limited, off by default). Predicted vs measured are shown side by side. See **[BENCHMARK.md](BENCHMARK.md)**.

Inputs, fixed assumptions, formulas and the change log: **[MVP.md](MVP.md)**. Estimator formulas: [ESTIMATOR.md](backend/app/services/estimators/ESTIMATOR.md).

## Prerequisites

Python 3.10+ and Node.js 22 LTS with npm.

## Start backend — terminal 1

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs. Use one uvicorn worker: analysis sessions live in process memory.

## Start frontend — terminal 2

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and click **Open estimator**. The frontend reaches the backend through Vite's `/api` proxy.

## Checks

```bash
cd backend && python -m pip install -r requirements-dev.txt && python -m unittest discover -s tests
cd frontend && npm test && npm run build
cd frontend && npx playwright install chromium && npm run e2e   # needs both servers running
```

Run the estimator without the app: `cd backend && python -m app.services.estimators.estimate_gpt2 ../examples/gpt2_mvp/gpt2_estimator_input.json`.
