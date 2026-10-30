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
| User-facing cancel (Nebius docs: cancel immediately stops the VM) | `/api/bench/jobs/{id}/cancel` | `BENCH_CANCEL_ENABLED` (default on) |
| Smaller job disk than the Nebius 250 GiB default | `JobSpec.disk` | `BENCH_DISK_GIB` (default 100, minimum 50) |

**About the platform timeout.** Per the Nebius "Managing jobs" docs, the job timeout ranges from
1 to 168 hours, with a default of 24 h; the job is cancelled if it hasn't completed by then. A job
that exceeds it ends `FAILED` with `state_details.code = TimeoutExceeded`, which Compute Cast maps
to `timed_out`. The backend sends 1 hour, the minimum.

The docs don't say whether the timeout counts from creation or from container start. It doesn't
matter much here, because the real limits are the runner's 5 minutes and the backend's cancel at
the deadline; the platform timeout is only a backstop.

**About provisioning.** If capacity isn't available, Nebius keeps the job in `PROVISIONING` for up
to 30 minutes, then ends it as `ERROR` with `NotEnoughResources`, and doesn't retry. Compute Cast
cancels earlier, at `BENCH_STARTUP_GRACE_SECONDS` (15 min) plus the runner limit, and reports
`timed_out`. Jobs are billed only while running, so waiting for capacity isn't charged.

**About money.** Allowances are reservations inside Compute Cast. They are **not** a Nebius
spending cap, and estimated charges are **not bills**: they are calculated from provider
timestamps at `BENCH_RESERVATION_USD_PER_GPU_HOUR`.

Nebius budget alerts notify you; they do not stop spending. The Nebius docs confirm this: "A
budget does not stop or cap your usage", and usage data updates only every few hours. **No maximum
bill is guaranteed.**

The one hard limit Nebius offers is **quotas**. The docs recommend requesting a quota *decrease* to
limit usage. For the dedicated benchmark project, ask for a GPU quota of 1 on each allowed
platform, and 1 for regular VMs (also preemptible VMs, if you use them). Even a misbehaving backend
then can't run more than one GPU at a time. Combine this with a low allowance, the kill switch and
a budget alert set at a low threshold.

## Job states

| Compute Cast | Nebius `JobStatus.State` |
| --- | --- |
| `submitting` | (before `create` returns; reconciled with `get_by_name` after 60 s) |
| `queued` | created / `STATE_UNSPECIFIED` |
| `provisioning` | `PROVISIONING`, `STARTING`, `IMAGE_PULLING` |
| `running` | `RUNNING` |
| `cancelling` | `CANCELLING`, or after a user cancel request |
| `succeeded` | `COMPLETED`, with a valid `results.json` showing status succeeded |
| `failed` | `COMPLETED`, `FAILED`, `ERROR` or `DELETING`, with missing, invalid or failed results (`state_details.code` `StartFailed`, `ContainerFailed` or `NotEnoughResources` is shown); or a submission error |
| `timed_out` | the runner reported `timeout`, the Nebius code was `TimeoutExceeded`, or the backend deadline passed |
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
| Hardware | `platform` and `preset` (1 GPU) from `BENCH_ALLOWED_PLATFORMS`; `disk` (Network SSD, `BENCH_DISK_GIB`); `subnet_id` from `NEBIUS_SUBNET_ID` (Nebius requires it only if the project has several subnets) |
| Limits and pricing | `timeout`, `restart_attempts=0`, `preemptible` with a matching `pricing_model` (`on_demand` or `follows_spot_price`) |
| Config delivery | `injected_files=[FileInjection(container_path="/config/benchmark_config.json", content=<config bytes ≤ 64 KiB>)]` |
| Results storage | `volumes=[VolumeMount(source=NEBIUS_RESULTS_BUCKET_ID, source_path="<prefix>/<job id>", container_path="/results", mode=READ_WRITE)]` |
| Private registry | Not needed for Nebius Container Registry in the same project (per the docs). For other registries: `registry_credentials=RegistryCredentials(mysterybox_secret_version=…)`. The SDK docstring requires secret keys `REGISTRY_USERNAME` and `REGISTRY_PASSWORD`; the CLI docs name them `registry_username` and `registry_password`. Check which one works on first use. |
| Status, lookup, cancel | `get(GetJobRequest(id))` → `status.state`, `status.state_details`, `status.started_at`, `status.finished_at`; `get_by_name(GetJobByNameRequest(parent_id, name))`; `cancel(CancelJobRequest(id))` |
| Authentication | The documented server-to-server method: `SDK(service_account_id=…, service_account_public_key_id=…, service_account_private_key_file_name=…)`, using an authorized key you upload for the service account. A credentials JSON (`NEBIUS_CREDENTIALS_FILE`) is still accepted as an alternative. |

**No environment variables or secrets are passed to the container.** The runner needs none.
Results are read back with a separate **read-only** S3 key, using boto3 against
`NEBIUS_S3_ENDPOINT`.

### Service account and permissions

Use a **dedicated project** for benchmarks, holding only the results bucket and, ideally, the image
registry. This matters because Nebius has **no Serverless-AI-specific role**: creating, viewing and
cancelling jobs needs the general `editor` role ("View, Create, Modify, Cancel, Delete" on jobs).
`editor` also covers most other resources, so scope it to that project only.

Set up two service accounts, both separate from any Token Factory API key:

1. **Job submitter** (used by the backend SDK). Put it in a custom group with an access permit for
   `editor` on the benchmark project only, never the tenant. Create an authorized key: generate an
   RSA key pair, upload the public key, and give the backend the private key file, the public key ID
   and the service account ID.
2. **Results reader** (used by the S3 client). Give it a custom group with `storage.object-viewer`
   ("download objects and view their metadata") on the results bucket. Create an access key for it,
   with an expiry date (`--expires-at`), and use that key's ID and secret as
   `NEBIUS_S3_ACCESS_KEY_ID` and `NEBIUS_S3_SECRET_ACCESS_KEY`.

Container Registry images in the same project need no pull credentials. Verify each permission with
a deliberately failing request before enabling the kill switch. Only `admin` can change quotas, so
keep quota changes with a human administrator, not either service account.

Credentials never reach the browser, the container, the local download or the logs:

- `Policy` hides secret fields from `repr`;
- `/api/bench/status` returns only the *names* of missing settings;
- tests assert that secrets are absent from packages, configs and responses.

**Workload identity and metadata access.** The Serverless AI job docs list no service-account or
identity setting for jobs, and the JobSpec in SDK 0.6.31 has no such field. Nebius "workload
identity federation" is for external identity providers, not for jobs. So the job isn't given an
identity by Compute Cast, and the docs don't document one being attached by default. Before going live, verify that a running
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
   - Upload the job submitter's private key as a **secret file**, for example
     `/etc/secrets/nebius-private.pem`. Set `NEBIUS_PRIVATE_KEY_FILE` to its path, and set
     `NEBIUS_PUBLIC_KEY_ID` and `NEBIUS_SERVICE_ACCOUNT_ID`.
   - Set `CORS_ALLOWED_ORIGINS=https://<user>.github.io`. This is the origin only, with no path.
   - Keep `BENCH_CLOUD_ENABLED=0` until the live checks below pass.

4. The job image defaults to `cr.eu-north1.nebius.cloud/e00yyx5y6aa4dfp7m4/benchmark@sha256:1de6db7c41473542e4e74997f3f068bb4b2d2f4c6a50f8c947f22386943becc9`
   (pushed as `benchmark:v1` in Nebius Container Registry, eu-north1). Same-project registry images need no pull credentials, so leave
   `NEBIUS_REGISTRY_SECRET_VERSION` empty. Run jobs in a project in the same region, and pin the image
   **by digest** with `BENCH_IMAGE` (the Nebius tab warns while it is a tag); see
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

### Checking the Nebius sign-in

After filling in the Nebius settings, run this read-only check from `backend/`. It creates nothing and costs
nothing:

```bash
python scripts/check_nebius.py
```

It signs in exactly as the backend does and shows which service account it is. It lists jobs in
`NEBIUS_PROJECT_ID`, and tries to download a non-existent object from the results bucket with the reader key.
Each step prints `OK` or `FAIL` with the reason. Secrets are never printed.

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

### Checked against docs.nebius.com

These Serverless AI pages were read: "Managing jobs", "Lifecycle and statuses", "Automatic job
restarts", "Debugging failed jobs" and "Pricing and quotas". So were the pages on IAM roles,
service-account authentication and access keys, Container Registry authentication, budgets and
quotas. They confirm the following:

- `container_command` is the entrypoint and `args` its arguments.
- Injected files are read-only, at most 64 KiB, at an absolute path.
- The timeout ranges from 1 to 168 h, with a 24 h default.
- `restart_attempts=0`, the default, runs the job once.
- `pricing_model` must match `preemptible`.
- The job states and the `state_details.code` values (`StartFailed`, `ContainerFailed`,
  `TimeoutExceeded`, `NotEnoughResources`).
- Waiting for capacity lasts up to 30 min, then the job ends in `ERROR`.
- Cancel immediately stops the VM and deletes the container disk.
- The VM is deleted when a job completes or fails; mounted volumes are kept.
- Billing covers only the time a job is running.
- `subnet_id` is needed only when the project has several subnets.
- Same-project Container Registry needs no credentials.
- Jobs need the `editor` role; `storage.object-viewer` is enough for reading results.
- Budgets don't cap spending; decreasing quotas does.

### Needs live verification (not done: no Nebius account was used and no paid job was submitted)

- The container image: it **has not been built or run**, because Docker was unavailable. Build,
  smoke-test, push and pin it by digest.
- A first real job:
  - `create().wait().resource_id` returns the job ID;
  - with a bucket volume, `source_path` (an SDK field the docs pages don't show) maps to the
    object key `<prefix>/<job>/results.json`;
  - files written to `/results` are in the bucket by the time the job reports `COMPLETED`;
  - the S3 endpoint (for example `https://storage.eu-north1.nebius.cloud`) works with the read-only key.
- What `status.started_at` and `finished_at` ("time when the job workload started/finished") cover,
  compared with the billed time.
- Whether the timeout is counted from creation or from container start (a backstop only).
- The registry secret key names, if you use a non-Nebius private registry (uppercase in the SDK,
  lowercase in the CLI docs).
- Platform and preset capacity for serverless jobs in your project's region.
- That a running job can't obtain credentials from a metadata endpoint, since the docs don't
  cover this.
- Postgres (`psycopg`) mode and the Render persistent disk; CORS from the real Pages origin.
