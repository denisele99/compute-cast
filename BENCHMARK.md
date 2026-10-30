# Benchmark mode

Benchmark mode measures the real speed and memory of the GPT-2 workload from the planner, on one
GPU, using a single trusted runner. You can run it in two ways:

1. **Local package.** Download a ZIP, run `python benchmark.py --config benchmark_config.json` on
   your own GPU, then upload or paste `results.json`.
2. **Nebius AI Cloud Serverless Jobs.** The backend submits the job asynchronously and the
   frontend polls it. The results are read back from a bucket.

Both paths run **the same code**: `benchmark/benchmark.py`, `benchmark/model_registry.py` and
`benchmark/models/`. Compute Cast only ever generates a validated **JSON** file,
`benchmark_config.json`. It never generates per-benchmark scripts, never executes uploaded files,
never installs user dependencies and never enables Hugging Face remote code.

Runner details, local commands, exit codes and the container build are in
[benchmark/README.md](benchmark/README.md).

## How it works

```
Workload form ──► POST /api/analysis/{id}/benchmark/config  (review: provenance, missing fields, memory check)
                  ├─► POST .../benchmark/package  ──► ZIP (runner + models + requirements + README + config)
                  │       user runs locally ──► results.json ──► POST .../benchmark/results {text}
                  └─► POST /api/bench/jobs (bearer session, idempotency key, confirm_charges)
                          reserve allowance + active slot (one DB transaction) ──► 202
                          background: Nebius JobService.create(spec)        (no automatic retry)
                      GET /api/bench/jobs/{id}  ──► throttled JobService.get ──► state mapping
                          on completion: read <prefix>/<job>/results.json via S3 ──► validate ──► release reservation
                      POST .../benchmark/results {job_id} ──► predicted vs measured
```

### Config review

The review screen shows every field, tagged with where its value came from:

- **detected**: from your files or the Hugging Face config;
- **your choice**: you entered it in the form, or edited it on the review screen;
- **default**: a benchmark default you haven't changed, such as 3 warm-up and 10 measured updates;
- **assumption**: fixed by the runner profile, such as synthetic inputs, random init, BF16
  autocast, FP32 math attention, AdamW and one GPU.

Missing fields are listed before anything can run.

Edits made on the review screen are sent as overrides and re-applied each time the config is
regenerated. An edit that becomes invalid, for example a sequence length above the new model's
positions, is dropped and reported; it is never kept silently. Only measurement counts, the seed,
optimizer settings and input shapes are editable there.

### Validation

`model_registry.validate` is used by both the backend and the runner. It rejects:

- unknown fields (so a config can't add a command, image, path or dependency), bad types,
  booleans used as numbers, and NaN or Infinity;
- unsupported architectures, Hugging Face IDs or revisions, weights, precision, optimizer and
  attention modes, input sources, and `gpu_count` other than 1;
- Hugging Face dimensions that differ from the allowlist, heads that don't divide the hidden size,
  and a sequence length above the position count;
- anything over the conservative limits in `LIMITS`: layers, width, vocabulary, batch,
  tokens per microbatch, total microbatches per job and parameter count.

The backend also runs an `estimate_gpt2` memory estimate, which must fit within **VRAM × (1 −
`BENCH_MEMORY_SAFETY_FRACTION`)**. For cloud jobs this check is blocking. For local packages it is
a warning, because the local GPU is only known once the runner starts.

### Measurement scope

What is timed: complete optimizer updates after warm-up, between `cuda.synchronize()` calls. Each
update is the forward pass, FP32 loss and backward pass for every accumulation microbatch,
followed by the AdamW step.

| Not counted in training time | Where it is reported |
| --- | --- |
| Runner start-up and model construction | `durations.startup_seconds` |
| Total runner wall time | `durations.runner_wall_seconds` |
| Provisioning, image pull and scheduling | Provider timestamps (`provider_run_seconds`) |
| The actual bill | Only Nebius billing |

Because inputs are synthetic, the results say nothing about data loading or model quality.

### Results

`results.json` contains:

- schema and code versions, the config fingerprint and the effective config;
- software versions (Python, torch, CUDA, cuDNN);
- detected hardware: GPU name, memory, compute capability and SM count;
- settings, update counts, per-update times, throughput (with units) and memory peaks;
- scope and exclusions, status, and failure type and message.

The backend treats it as **untrusted data**:

- size limit of 256 KiB, NaN and Infinity rejected, finite non-negative numbers;
- the effective config is re-validated and must match its fingerprint;
- counts must match the config;
- a non-smoke result must come from CUDA, and multi-GPU results are rejected.

### Comparison and calibration

Measurements are shown **beside** a prediction made for exactly the measured config. The original
estimate in the Estimate tab is never modified.

Measured ÷ predicted ratios are shown only when **the config fingerprint equals the current
config and the measured GPU matches the predicted GPU**. Nothing is scaled to other shapes or GPUs.

A full-run figure (measured median update × planned updates) is labelled as an estimate. It appears
only for the same config and shapes.

CPU smoke tests are labelled as such and never used for calibration.

## Safeguards

| Safeguard | Where | Setting |
| --- | --- | --- |
| Kill switch: cloud submissions refused | `jobs.submit` | `BENCH_CLOUD_ENABLED` (default off) |
| One active GPU job globally | UNIQUE `bench_jobs.active_slot` | — |
| Duplicate submissions return the same job | UNIQUE `(session_id, idempotency_key)`, plus a provider-side name `cc-bench-<id>` | — |
| Per-session and global allowance, reserved atomically before submission | Conditional `UPDATE … WHERE used + reserved + amount <= limit` | `BENCH_SESSION_ALLOWANCE_USD`, `BENCH_GLOBAL_ALLOWANCE_USD`, `BENCH_RESERVATION_USD_PER_GPU_HOUR` |
| Runner time limit (default 5 min, including start-up and warm-up) | `benchmark.py --max-seconds` (SIGALRM plus a loop deadline) | `BENCH_RUNNER_MAX_SECONDS` |
| Backend deadline: start-up grace + runner limit, then cancel and mark `timed_out` | `jobs.refresh` | `BENCH_STARTUP_GRACE_SECONDS` |
| Platform timeout backstop | `JobSpec.timeout` | `BENCH_PLATFORM_TIMEOUT_HOURS` (minimum 1 h) |
| No automatic retries (invalid config, OOM, anything else) | `restart_attempts=0`, single create call | — |
| Memory estimate with a safety margin before submission | `config_builder.memory_check` | `BENCH_MEMORY_SAFETY_FRACTION` |
| One GPU per job, from allowed platform/preset pairs only | Registry plus `nebius_hardware` | `BENCH_ALLOWED_PLATFORMS` |
| Polling throttled per job | `jobs.refresh` | `BENCH_POLL_MIN_INTERVAL_SECONDS` |
| User-facing cancel (off until verified live) | `/api/bench/jobs/{id}/cancel` | `BENCH_CANCEL_ENABLED` |

**About the platform timeout.** The Nebius CLI documents a job `--timeout` of 1–168 hours (default
24 h). This was found through search results; `docs.nebius.com` itself could not be opened from
the development environment. The SDK field is `JobSpec.timeout` (a duration). The backend sends at
least one hour.

Whether that timeout counts from job creation, from provisioning or from container start still
needs **live verification**. Until then, treat it only as a backstop. The real limits are the
runner's 5 minutes and the backend's cancel at the deadline.

**About money.** Allowances are reservations inside Compute Cast. They are **not** a Nebius
spending cap, and estimated charges are **not bills**: they are calculated from provider
timestamps at `BENCH_RESERVATION_USD_PER_GPU_HOUR`.

Nebius budget alerts notify you; they do not stop spending. **No maximum bill is guaranteed.**
Limit exposure with a dedicated project, a restricted service account, low allowances and the
kill switch.

## Job states

| Compute Cast | Nebius `JobStatus.State` |
| --- | --- |
| `submitting` | (before `create` returns; reconciled with `get_by_name` after 60 s) |
| `queued` | created / `STATE_UNSPECIFIED` |
| `provisioning` | `PROVISIONING`, `STARTING`, `IMAGE_PULLING` |
| `running` | `RUNNING` |
| `cancelling` | `CANCELLING`, or after a user cancel request |
| `succeeded` | `COMPLETED`, with a valid `results.json` showing status succeeded |
| `failed` | `COMPLETED`, `FAILED`, `ERROR` or `DELETING`, with missing, invalid or failed results; or a submission error |
| `timed_out` | the runner reported `timeout`, or the backend deadline passed |
| `cancelled` | `CANCELLED` |

Jobs are not resumable. A failed job is not resubmitted; the user starts a new one.

## Nebius integration

The integration lives in `backend/app/services/benchmarks/nebius_client.py`, behind the
`JobsClient` and `ResultStore` protocols. `FakeJobsClient` (labelled "SIMULATED" in the UI) is used
by tests and `BENCH_NEBIUS_BACKEND=fake`.

It was built against the official Python SDK **nebius 0.6.31** from PyPI. The field names below
were checked by constructing the real messages; no request was ever sent.

| Purpose | SDK call or field |
| --- | --- |
| Create a job | `JobServiceClient.create(CreateJobRequest(metadata=ResourceMetadata(parent_id=NEBIUS_PROJECT_ID, name="cc-bench-<id>", labels=…), spec=JobSpec(…)))` |
| Image and command | `image=BENCH_IMAGE`, `container_command="python"`, `args="/app/benchmark.py --config /config/benchmark_config.json --output-dir /results --max-seconds 300"` |
| Hardware | `platform` and `preset` (1 GPU) from `BENCH_ALLOWED_PLATFORMS` |
| Limits and pricing | `timeout`, `restart_attempts=0`, `preemptible` with a matching `pricing_model` (`on_demand` or `follows_spot_price`) |
| Config delivery | `injected_files=[FileInjection(container_path="/config/benchmark_config.json", content=<config bytes ≤ 64 KiB>)]` |
| Results storage | `volumes=[VolumeMount(source=NEBIUS_RESULTS_BUCKET_ID, source_path="<prefix>/<job id>", container_path="/results", mode=READ_WRITE)]` |
| Private registry | `registry_credentials=RegistryCredentials(mysterybox_secret_version=…)`, only if set. The secret holds `REGISTRY_USERNAME` and `REGISTRY_PASSWORD`. |
| Status, lookup, cancel | `get(GetJobRequest(id))` → `status.state`, `status.state_details`, `status.started_at`, `status.finished_at`; `get_by_name(GetJobByNameRequest(parent_id, name))`; `cancel(CancelJobRequest(id))` |
| Authentication | `SDK(credentials_file_name=NEBIUS_CREDENTIALS_FILE)`, a service-account credentials JSON |

**No environment variables or secrets are passed to the container.** The runner needs none.
Results are read back with a separate **read-only** S3 key, using boto3 against
`NEBIUS_S3_ENDPOINT`.

### Service account and permissions

Use a **dedicated project** for benchmarks and a **dedicated service account**. This account is
separate from any Nebius Token Factory API key; never reuse a Token Factory key here. Grant it
only:

- permission to create, get and cancel AI jobs in that project;
- read access to the benchmark image in Container Registry, or use the MysteryBox secret for a
  private registry;
- permission to mount the results bucket into jobs.

Create the S3 access key for the backend as **read-only** on the results bucket.

Check the exact role names in the current Nebius IAM docs, then verify each permission with a
deliberately failing request before enabling the kill switch.

Credentials never reach the browser, the container, the local download or the logs:

- `Policy` hides secret fields from `repr`;
- `/api/bench/status` returns only the *names* of missing settings;
- tests assert that secrets are absent from packages, configs and responses.

**Workload identity and metadata access.** The JobSpec in SDK 0.6.31 has no service-account field,
so the job is not given a service account by Compute Cast. Before going live, verify that a running
job cannot obtain project credentials through an instance metadata endpoint or a default
identity. For example, run a throwaway job that tries the metadata address and the Nebius CLI
without credentials, and confirm that both fail. If anything is reachable, keep the project
dedicated and empty apart from the benchmark bucket.

## Deployment

### Local development

Start the backend and frontend as described in [README.md](README.md). No cloud settings are
needed; the Nebius tab shows "Switched off".

To try the cloud flow without Nebius, use the simulated provider (`BENCH_NEBIUS_BACKEND=fake`).
Simulated jobs end as `failed / results missing`, because no simulated results are fabricated.

```bash
cd backend
BENCH_CLOUD_ENABLED=1 BENCH_NEBIUS_BACKEND=fake BENCH_DEMO_PASSWORD=dev-pass BENCH_SESSION_SECRET=dev-secret-please-change python -m uvicorn app.main:app --port 8000
```

### Backend on Render

1. Create a **Web Service** from this repository.

   | Setting | Value |
   | --- | --- |
   | Root directory | `backend` |
   | Build command | `pip install -r requirements-cloud.txt` |
   | Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
   | Instances | **One** (analysis sessions are held in process memory) |

2. Add **durable storage**, using one of these:
   - a persistent disk mounted at `/var/data`, with
     `BENCH_DATABASE_URL=sqlite:////var/data/bench.sqlite3`;
   - Render Postgres, with `BENCH_DATABASE_URL=postgresql://…`.

   Do not rely on the default filesystem or process memory; both are lost on deploy and restart.
   The Postgres code path has not been run against a real database yet (see live verification).

3. Set the environment from `backend/.env.example`:
   - Add `BENCH_DEMO_PASSWORD`, `BENCH_SESSION_SECRET` and `NEBIUS_S3_SECRET_ACCESS_KEY` as secrets.
   - Upload the service-account credentials JSON as a **secret file**, for example
     `/etc/secrets/nebius-credentials.json`, and point `NEBIUS_CREDENTIALS_FILE` at it.
   - Set `CORS_ALLOWED_ORIGINS=https://<user>.github.io`. This is the origin only, with no path.
   - Keep `BENCH_CLOUD_ENABLED=0` until the live checks below pass.

4. Point `BENCH_IMAGE` at the published image **by digest**; see
   [benchmark/README.md](benchmark/README.md).

5. Login rate-limiting uses the last `X-Forwarded-For` entry, which is the one Render's proxy
   appends. If you put another proxy in front of Render, revisit `_client_ip` in
   `app/api/routes/benchmark.py`.

### Frontend on GitHub Pages

1. Build with the backend URL and the Pages base path. Settings are documented in
   `frontend/.env.example`; `VITE_*` values are public.

   ```bash
   cd frontend && VITE_API_BASE_URL=https://<service>.onrender.com VITE_BASE_PATH=/<repo>/ npm run build
   ```

2. Publish `frontend/dist`, for example with the `actions/upload-pages-artifact` and
   `actions/deploy-pages` actions. No workflow file is added in this change.

**Sessions across origins.** Logging in returns a signed bearer token
(`HMAC-SHA256(BENCH_SESSION_SECRET)`) with an expiry of `BENCH_SESSION_TTL_MINUTES`. Sessions are
also stored server-side.

The frontend keeps the token in `sessionStorage` and sends it as an `Authorization` header, so no
third-party cookies are involved. CORS allows only the configured origins and the
`Authorization` and `Content-Type` headers, without credentials mode.

The display name is only a label. Allowances and job ownership are keyed by the random session ID,
and other sessions get 404 for jobs they don't own.

### Turning cloud benchmarks off

Set `BENCH_CLOUD_ENABLED=0` and restart. New submissions are refused immediately, and local
packages keep working. Jobs that are already running continue until they finish or are cancelled
at the backend deadline, provided status polling continues, or until the platform timeout.

To stop a job immediately, cancel it in the Nebius console or with the CLI.

## Tests

Run the offline tests:

```bash
cd backend && python -m pytest -q tests
```

With a torch environment, `BENCH_PYTHON` also runs the real runner on CPU (smoke). On a GPU
machine, add `BENCH_CUDA=1` for the CUDA test, which is skipped without a GPU. These commands run
from `backend/`.

```bash
BENCH_PYTHON=/path/to/python-with-torch python -m pytest -q tests/test_benchmark.py
```

```bash
BENCH_PYTHON=/path/to/python-with-torch BENCH_CUDA=1 python -m pytest -q tests/test_benchmark.py -k Cuda
```

The end-to-end flow (`frontend/e2e/upload_flow.cjs`) covers the config review, package download,
CPU smoke run, results import and, against a simulated-provider backend, login and submission.
See its header for the environment variables it needs.

### Tested offline

- Registry and config validation: unknown or user-controlled fields, types, non-finite numbers,
  unsupported modes, Hugging Face allowlist and dimensions, heads/hidden size, positions,
  every limit, total work and parameter cap.
- Review provenance, override retention and dropping, missing fields, Nebius hardware validity and
  alternatives, and the memory safety margin.
- Package contents: the same runner bytes, the validated config, and no secrets.
- Results validation: schema, NaN, size, fingerprint, counts, CUDA vs smoke, multi-GPU.
- Comparison and calibration only for a compatible config and GPU; smoke results labelled.
- Auth: login, wrong password, rate limit, tampered or expired tokens, unconfigured server, and that
  status responses carry no secrets.
- Allowances, including atomic reservation under concurrent threads; the global single active job;
  idempotent resubmission; ownership.
- Mocked Nebius lifecycle:
  - success, failure without results, runner OOM and timeout, backend deadline cancel → `timed_out`;
  - cancel, and cancel disabled;
  - submission error reconciled by name, submission failure releasing the reservation;
  - poll throttling, kill switch and missing configuration.
- `CreateJobRequest` construction against the real `nebius` 0.6.31 SDK, including the 64 KiB
  injection limit.
- A real CPU smoke run of `benchmark.py`: results validated, parameter count checked, plus the
  invalid-config, timeout and no-GPU exit codes.
- The browser flow, end to end, with a simulated provider.

### Needs live verification (not done: no Nebius account was used and no paid job was submitted)

- The container image: it **has not been built or run**, because Docker was unavailable. Build,
  smoke-test, push and pin it by digest.
- A real job submission:
  - `create().wait().resource_id` returns the job ID;
  - `container_command` and `args` override the image entrypoint, and `args` is tokenised as
    expected;
  - `/config` file injection works;
  - bucket mount write permissions and the resulting object key `<prefix>/<job>/results.json`;
  - the S3 endpoint and region, and the read-only key.
- State timing and transitions, and what `started_at` and `finished_at` measure, compared with
  Nebius billing.
- Cancel semantics: how quickly billing stops. Keep `BENCH_CANCEL_ENABLED=0` until this is
  confirmed.
- How `JobSpec.timeout` is counted.
- Platform and preset availability for serverless jobs in your project's region.
- Service-account role names and least privilege; no workload identity or metadata credentials
  inside jobs.
- Postgres (`psycopg`) mode and the Render persistent disk; CORS from the real Pages origin.
- The official docs at `docs.nebius.com` could not be opened from the development sandbox; network
  policy blocked them. Integration details come from the SDK package itself and search summaries
  of the docs. Re-check them against the current documentation.
