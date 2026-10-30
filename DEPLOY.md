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
   - `BENCH_DEMO_PASSWORD`: anything for now (cloud benchmarks stay off on the free plan).
   - The `NEBIUS_*` benchmark values: can stay empty on the free plan.
4. Click **Apply**. The first build takes a few minutes.
5. Open `https://<your-service>.onrender.com/api/health`. It should show `{"status":"ok"}`. Copy that
   base URL (without `/api/health`); it's needed in step 2.

**What the free plan means:**
- **It sleeps.** After about 15 minutes without visitors the service sleeps, so the next visit waits
  about a minute while it wakes up. An analysis that was open when it went to sleep is lost; start a
  new one.
- **No disk.** The benchmark database is lost on every restart, which would also reset the spending
  allowances. So `BENCH_CLOUD_ENABLED` stays `0` on the free plan.
- **What still works:** estimates, recommendations and **local benchmarks** (download the package and
  run it on your own GPU).
- **To run Nebius benchmarks from the hosted app later,** do one of:
  - upgrade to a paid plan and add a disk (mount it at `/var/data` and set
    `BENCH_DATABASE_URL=sqlite:////var/data/bench.sqlite3`);
  - set `BENCH_DATABASE_URL` to an external Postgres database.

  Then set `BENCH_CLOUD_ENABLED=1`.

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
