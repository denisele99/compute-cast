# Connected estimation workflow

Branch: `feature/connected-estimation-workflow`. Target: `main`.

## Implemented

- Rebased onto current `main`, corrected the latest API router imports and restored the legacy calculation behind the original endpoint. Newer backend estimator/parser directories and frontend helper/page modules are retained. Legacy contracts remain covered by their original tests.
- Explicitly load the TypeScript app. The starting snapshot contained conflicting legacy JS entrypoints; current main also removes those old files.
- Added a connected workload → hardware → results flow, advanced settings, field validation, required/optional labels, grey defaults, red extraction uncertainty and upload review confirmation.
- Python AST and safe JSON/YAML parsing populate the same typed form used by manual input. The server never imports uploaded code and does not save uploads. Conflicting, dynamic or invalid extracted values require review. Upload limits: 8 files, 250 KB per file, 1 MB combined.
- Cache uploads and metadata only in page memory. Save immutable original metadata on the first successful run; compare subsequent inputs and export original/current configuration and source in a review/recommendation bundle. Refresh, close, or New analysis discards the in-memory cache.
- Search up to 10 regional GPU offers from the two supplied Nebius snapshot files; display memory, region and USD/GPU-hour price, with autofill and custom overrides. The files contain 5 GPU platform definitions and 12 regional price offers; missing memory is never fabricated.
- Add optional per-GPU memory limits, available versus active GPU counts, explicit DDP/ZeRO/FSDP choices and optional CPU/RAM/SSD/NVMe metadata.
- Return prediction results, actual VRAM usage percentages when known, fit checks, indicative memory comparisons, assumptions, calculation details and normalized estimator configuration export.
- Add prediction/benchmark mode switching on the same results page. Download a local PyTorch CUDA runner and config, then import a matching report through the API. Inputs changed after a measurement clear stale timing/VRAM values.

## Backend logic changes

- New workflow uses a validated general config and a smaller normalized workload/hardware/deployment/assumptions/calibration/pricing object. Unsupported original config fields remain in metadata rather than silently influencing estimates.
- Manufacturer GB converts to GiB before comparison. The default usable limit is **90% of each GPU's physical memory**, not pooled multi-GPU VRAM.
- Only GPUs actually used by the script determine data partitioning and GPU-hour cost. Merely owning more GPUs does not accelerate training.
- Per-rank batches = `ceil(ceil(examples / active_GPUs) / batch_size) × epochs`. Accumulation changes optimizer updates and flushes a partial accumulation group at each epoch boundary.
- Runtime extrapolates matching whole-microbatch timing, or uses the existing metadata-only runtime planner when its explicit prerequisites are complete. Its formulas are unchanged. Cost = runtime hours × active GPUs × USD/GPU-hour. Missing timing or pricing produces **unknown**, replacing the new UI's former fictional timing/price defaults. The legacy `/api/estimate` keeps its original default behavior for compatibility.
- Memory prediction reuses `estimate_gpu_memory.py`, including its existing DDP/ZeRO/FSDP state accounting, workspace, allocator and runtime allowances. No changes were made to its formulas. Manual parameter metadata assumes fully trainable dense FP32 weights/gradients, FP32 optimizer states, no master copy/checkpointing, zero buffers and an aggregate largest tensor; these are disclosed assumptions, not inferred facts.
- Parameter count alone cannot predict complete activation/workspace memory. Require activation plus auxiliary storage overrides, or a matching measured peak, before claiming a peak or fit. CPU RAM requirements stay unknown without allocation metadata.
- The local runner synchronizes CUDA, warms up, times complete adapter microbatches and records this process's reserved-memory peak. The import API rejects mismatched input configs, invalid values and unsupported multi-GPU reports. User-supplied reports are not cryptographically attested.

## Approval and remaining work

- Review/merge this branch through a pull request. No merge to `main` or deployment is authorized by this implementation.
- **Nebius cloud benchmark execution is not implemented.** The UI is disabled and the endpoint returns 501 without submitting a job. Enabling it needs a separate implementation of provisioning, credentials, job isolation, billing consent/budget, cancellation and cleanup. No cloud charge was incurred.
- Local benchmarks require the user's CUDA GPU, PyTorch environment, project dependencies and a trusted `build_benchmark(model_path, data_path)` adapter. Distributed benchmark execution needs a distributed runner. Real GPU execution was not validated in this CPU environment.
- Arbitrary Python model construction is not executed during upload. Static parsing cannot reliably infer all parameter counts, tensor shapes or runtime behavior. Dynamic/unsupported model metadata requires manual review or a trusted external collection step.
- Metadata-only runtime reuses the existing runtime planner when users supply per-microbatch forward/backward FLOPs and HBM traffic, GPU compute/bandwidth specifications, and an explicit data-ready assumption. Multi-GPU runtime also needs effective link bandwidth. The UI reports fast/slow planning scenarios and uses the slow scenario for displayed cost; these are not statistical bounds. Missing data produces unknown total runtime. No timing is transferred between GPUs in comparisons.
- CPU RAM/SSD/NVMe capacities are saved, but CPU demand, disk throughput and offload planning are not inferred from capacity alone. No percentage is fabricated for unknown requirements.
- Code recommendations use the existing Nebius provider endpoint only after explicit source-sharing/paid-request consent. Cached source and original/current metadata are sent on request; proposed files can be reviewed, downloaded and applied to the page cache for re-parsing. This optional integration requires backend NEBIUS_API_KEY, the OpenAI SDK and HTTPS access to api.tokenfactory.nebius.com. Live provider calls were not tested and no charge was incurred. Generated changes are never executed or written to the repository automatically.
- Prices are the provided dated spot snapshots, not live quotes. Refresh/enrich catalogue data separately if all listed offers need verified memory specs.

## Validation

Validated after rebasing onto `main` at `6e97fff`: 27 backend tests, 5 retained frontend unit tests, a production build, and 7 connected browser tests against the production build. The code-recommendation provider boundary is mocked in its browser test. GPU execution and paid provider requests were not run. The newer main introduced Node-based TypeScript tests; adding Node type definitions restores their inclusion in the production type check.

Run backend tests from `backend`:

```bash
python -m unittest discover -s tests -v
```

Run `npm ci` and `npm run build` in `frontend`. Start the backend and Vite as described in the root README, install a Playwright Chromium browser (`npx playwright install chromium`), then run `npm run test:e2e`. An existing Chromium binary may be selected with `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`. `PLANNER_BASE_URL` can target a locally running production preview for the same connected checks.
