# Publishing Compute Cast (free plans)

The app has two parts, hosted in two places:

```
GitHub Pages   https://denisele99.github.io/compute-cast/    the website (static React files, free)
      │ calls the API
      ▼
Render         https://compute-cast-api.onrender.com          the Python backend (FastAPI, free plan)
```

GitHub Pages can only serve files; it can't run Python. So the backend runs on Render and the website
on Pages calls it. Both deploy automatically from the `main` branch.

## 0. Put everything on `main`

Both deployments build from `main`, so merge `claude/benchmark-mode` into `main` first (one pull request
on GitHub).

## 1. Backend on Render (free)

1. Sign in at https://dashboard.render.com with GitHub.
2. Click **New**, then **Blueprint**, and pick `denisele99/compute-cast`. Render reads `render.yaml` and
   creates the free `compute-cast-api` web service.
3. Render asks for the secret values. Fill in what you have; you can add the rest later under the
   service's **Environment** tab.
   - `NEBIUS_API_KEY`: your Token Factory key (turns on Recommendations).
   - `BENCH_DEMO_PASSWORD`: a password people type in the app to start cloud benchmarks. Use a long one.
   - The `NEBIUS_*` benchmark values: needed for cloud benchmarks (see BENCHMARK.md). Also add the
     `credentials.json` Secret File under **Environment**, then **Secret Files**.
4. Click **Apply**. The first build takes a few minutes.
5. Open `https://<your-service>.onrender.com/api/health`. It should show `{"status":"ok"}`. Copy that
   base URL (without `/api/health`); it's needed in step 2.

**What the free plan means:**
- **It sleeps.** After about 15 minutes without visitors the service sleeps, so the next visit waits
  about a minute while it wakes up. An analysis that was open when it went to sleep is lost; start a
  new one.
- **No disk.** The benchmark database is lost on every restart or sleep, and the app's spending allowances
  reset with it. Cloud benchmarks are on (`BENCH_CLOUD_ENABLED=1` in `render.yaml`), so the hard spending
  limit is your Nebius quota: keep the benchmark project's GPU quota at 1. Set `BENCH_CLOUD_ENABLED` to
  `0` on Render to switch cloud benchmarks off.
- **Nebius settings are required.** Cloud benchmarks only start once the `NEBIUS_*` values and the
  `credentials.json` Secret File are set on Render; until then the Nebius tab lists what is missing.
- **For allowances that survive restarts,** either:
  - upgrade to a paid plan and add a disk (mount it at `/var/data` and set
    `BENCH_DATABASE_URL=sqlite:////var/data/bench.sqlite3`); or
  - set `BENCH_DATABASE_URL` to an external Postgres database.

## 2. Website on GitHub Pages (free for public repos)

1. On GitHub, open the repository's **Settings**, then **Pages**, and set **Source** to **GitHub Actions**.
2. Open **Settings**, then **Secrets and variables**, then **Actions**, then the **Variables** tab, then
   **New repository variable**:
   - Name: `VITE_API_BASE_URL`
   - Value: your Render URL from step 1.5, for example `https://compute-cast-api.onrender.com`, with no
     trailing slash.
3. Open **Actions**, then **Deploy frontend to GitHub Pages**, then **Run workflow**. Later pushes to
   `main` that change `frontend/` redeploy it automatically.
4. Open https://denisele99.github.io/compute-cast/ (the workflow's summary shows the link).

The backend only accepts requests from `https://denisele99.github.io` (`CORS_ALLOWED_ORIGINS` in
`render.yaml`). If the site moves to another address, change that setting on Render.

## 3. Checking the Nebius setup

The free plan has no Render Shell, so run the read-only check on your own computer. From `backend/`, with
your `backend/.env` filled in:

```bash
python scripts/check_nebius.py
```

## Updating

Merge to `main`. Render rebuilds the backend and the workflow republishes the website.
