"""Stateless, metadata-only workflow. Uploaded Python is never executed here."""
import ast
import json
import math
from pathlib import Path
from typing import Literal

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .services.estimators.gpu_memory import estimate_gpu_memory
from .services.estimators.gpu_runtime import estimate_runtime

router = APIRouter()
DATA = Path(__file__).resolve().parents[1] / "data"
GIB = 1024 ** 3


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Training(StrictModel):
    examples: int = Field(default=100000, ge=1, le=10**12)
    batch_size: int = Field(default=16, ge=1, le=10**6)
    epochs: int = Field(default=10, ge=1, le=100000)
    sequence_length: int = Field(default=512, ge=1, le=10**7)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=100000)
    script_gpu_count: int = Field(default=1, ge=1, le=1024)
    strategy: Literal["single", "ddp", "zero1", "zero2", "zero3", "fsdp_full_shard"] = "single"
    seconds_per_batch: float | None = Field(default=None, gt=0, le=10**6)
    measured_peak_vram_gib: float | None = Field(default=None, gt=0, le=100000)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    optimizer: Literal["AdamW", "Adam", "SGD"] = "AdamW"
    assume_data_ready: bool = False

    @model_validator(mode="after")
    def check(self):
        if self.batch_size > self.examples:
            raise ValueError("batch_size cannot exceed examples")
        if self.script_gpu_count > 1 and self.strategy == "single":
            raise ValueError("strategy must be DDP or sharded when script_gpu_count exceeds 1")
        if self.script_gpu_count == 1 and self.strategy != "single":
            raise ValueError("strategy must be single when script_gpu_count is 1")
        return self


class ModelInfo(StrictModel):
    total_params: int | None = Field(default=None, ge=1, le=10**13)
    activation_gib: float | None = Field(default=None, ge=0, le=100000)
    auxiliary_gib: float | None = Field(default=None, ge=0, le=100000)
    flops_per_microbatch: float | None = Field(default=None, gt=0, le=1e30)
    hbm_bytes_per_microbatch: float | None = Field(default=None, gt=0, le=1e30)
    # Original config and optional collect_workload report are retained, never executed.
    config: dict = Field(default_factory=dict)


class Hardware(StrictModel):
    option_id: str = Field(default="custom", max_length=200)
    name: str = Field(default="Custom GPU", max_length=200)
    vram_gib: float = Field(default=24, gt=0, le=10000)
    memory_limit_gib: float | None = Field(default=None, gt=0, le=10000)
    available_gpus: int = Field(default=1, ge=1, le=1024)
    price_per_gpu_hour: float | None = Field(default=None, ge=0, le=10**6)
    cpu_ram_gib: float | None = Field(default=None, gt=0, le=10**6)
    cpu_cores: int | None = Field(default=None, ge=1, le=100000)
    nvme_gib: float | None = Field(default=None, ge=0, le=10**9)
    ssd_gib: float | None = Field(default=None, ge=0, le=10**9)
    cpu_specs: str = Field(default="", max_length=500)
    compute_tflops: float | None = Field(default=None, gt=0, le=1e9)
    fp32_tflops: float | None = Field(default=None, gt=0, le=1e9)
    memory_bandwidth_gbps: float | None = Field(default=None, gt=0, le=1e9)
    gpu_link_gbps: float | None = Field(default=None, gt=0, le=1e9)

    @model_validator(mode="after")
    def check(self):
        if self.memory_limit_gib is not None and self.memory_limit_gib > self.vram_gib:
            raise ValueError("memory_limit_gib cannot exceed vram_gib")
        return self


class Analysis(StrictModel):
    framework: Literal["pytorch"] = "pytorch"
    training: Training = Field(default_factory=Training)
    model: ModelInfo = Field(default_factory=ModelInfo)
    hardware: Hardware = Field(default_factory=Hardware)

    @model_validator(mode="after")
    def check(self):
        if self.training.script_gpu_count > self.hardware.available_gpus:
            raise ValueError("script_gpu_count cannot exceed available_gpus")
        return self


def catalog():
    platforms = {p["metadata"]["name"]: p for p in json.loads((DATA / "nebius_platforms.json").read_text())["items"]}
    offers = []
    for item in json.loads((DATA / "nebius_spot_prices.json").read_text())["items"]:
        key = item["computeInstance"]["platform"]
        platform = platforms.get(key, {})
        gb = platform.get("spec", {}).get("gpu_memory_gigabytes")
        price = item["price"]
        if price["currency"].lower() != "usd" or price["unitTitle"] != "GPU hour":
            continue
        offers.append({"id": item["skuId"], "platform": key, "name": item["productTitle"],
                       "region": item["region"], "vram_gib": gb * 10**9 / GIB if gb else None,
                       "memory_gb": gb, "price_per_gpu_hour": float(price["value"]),
                       "effective_time": item["effectiveTime"], "presets": platform.get("spec", {}).get("presets", [])})
    return offers


@router.get("/hardware")
def hardware(q: str = Query(default="", max_length=200), limit: int = Query(default=10, ge=1, le=100)):
    items = [i for i in catalog() if q.lower() in (i["name"] + i["platform"] + i["region"]).lower()]
    return {"items": items[:limit], "total": len(items), "note": "Saved Nebius spot-price snapshot, not a live quote. GPU-hour only; excludes CPU, storage, network, tax and interruption costs. Manufacturer GB converted to GiB. Unknown memory must be supplied manually."}


ALIASES = {
    "examples": ("training", "examples"), "num_examples": ("training", "examples"),
    "num_train_examples": ("training", "examples"), "num_samples": ("training", "examples"),
    "batch_size": ("training", "batch_size"), "batch_size_per_gpu": ("training", "batch_size"),
    "per_device_train_batch_size": ("training", "batch_size"),
    "epochs": ("training", "epochs"), "num_train_epochs": ("training", "epochs"),
    "sequence_length": ("training", "sequence_length"), "max_seq_length": ("training", "sequence_length"),
    "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
    "gpu_count": ("training", "script_gpu_count"), "world_size": ("training", "script_gpu_count"),
    "script_gpu_count": ("training", "script_gpu_count"),
    "distributed_strategy": ("training", "strategy"), "strategy": ("training", "strategy"),
    "precision": ("training", "precision"), "optimizer": ("training", "optimizer"),
    "seconds_per_batch": ("training", "seconds_per_batch"),
    "total_params": ("model", "total_params"), "num_parameters": ("model", "total_params"),
    "activation_gib": ("model", "activation_gib"), "auxiliary_gib": ("model", "auxiliary_gib"),
    "flops_per_microbatch": ("model", "flops_per_microbatch"),
    "hbm_bytes_per_microbatch": ("model", "hbm_bytes_per_microbatch"),
    "assume_data_ready": ("training", "assume_data_ready"),
}


class Upload(StrictModel):
    name: str = Field(min_length=1, max_length=200)
    content: str = Field(max_length=250000)


class ParseRequest(StrictModel):
    files: list[Upload] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def size(self):
        if sum(len(f.content.encode()) for f in self.files) > 1_000_000:
            raise ValueError("Combined uploads must be at most 1 MB")
        return self


@router.post("/analysis/parse")
def parse_uploads(request: ParseRequest):
    candidates, errors, metadata = {}, {}, {}

    def add(key, value, source):
        if key in ALIASES:
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                value = None
            path = ".".join(ALIASES[key])
            candidates.setdefault(path, []).append({"value": value, "source": source})

    def walk(value, source, depth=0):
        if depth > 30:
            raise ValueError("Config nesting exceeds 30 levels")
        if isinstance(value, dict):
            for key, val in value.items():
                add(key, val, source)
                walk(val, source, depth + 1)
        elif isinstance(value, list):
            for val in value:
                walk(val, source, depth + 1)

    for file in request.files:
        try:
            suffix = Path(file.name).suffix.lower()
            if suffix == ".py":
                tree = ast.parse(file.content)
                for node in ast.walk(tree):
                    pairs = []
                    if isinstance(node, ast.Assign):
                        pairs = [(t.id, node.value) for t in node.targets if isinstance(t, ast.Name)]
                    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
                        pairs = [(node.target.id, node.value)]
                    elif isinstance(node, ast.Call):
                        pairs = [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
                    for key, expr in pairs:
                        try:
                            val = ast.literal_eval(expr)
                        except (ValueError, TypeError, SyntaxError):
                            val = None
                        add(key, val, f"{file.name}:{node.lineno}")
                metadata[file.name] = {"kind": "static_python", "note": "Literal hints only; scope and control flow are not resolved. PyTorch only. Review all extracted values."}
            elif suffix in (".json", ".yaml", ".yml"):
                config = json.loads(file.content) if suffix == ".json" else yaml.safe_load(file.content)
                if not isinstance(config, dict):
                    raise ValueError("Config must contain an object")
                # Reject recursive aliases and non-JSON YAML values before walking.
                json.dumps(config, allow_nan=False, check_circular=True)
                walk(config, file.name)
                metadata[file.name] = config
            else:
                raise ValueError("Use Python scripts or JSON/YAML configs")
        except (SyntaxError, ValueError, TypeError, yaml.YAMLError, RecursionError) as exc:
            errors[file.name] = str(exc)[:500]
    values, uncertain = {}, {}
    for path, hints in candidates.items():
        distinct = {json.dumps(h["value"], sort_keys=True) for h in hints}
        if len(distinct) != 1 or hints[0]["value"] is None:
            uncertain[path] = "Conflicting or dynamic values; enter and review manually."
            continue
        group, key = path.split(".")
        value = hints[0]["value"]
        field = {"training": Training, "model": ModelInfo}[group].model_fields[key]
        from pydantic import TypeAdapter, ValidationError
        from typing import Annotated
        try:
            value = TypeAdapter(Annotated[field.annotation, *field.metadata]).validate_python(value)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Value must be finite")
            values.setdefault(group, {})[key] = value
        except (ValidationError, ValueError):
            uncertain[path] = "Extracted value is invalid; enter manually."
    for path in ("training.examples", "training.batch_size", "training.epochs", "model.total_params"):
        group, key = path.split(".")
        if key not in values.get(group, {}):
            uncertain.setdefault(path, "Not reliably extracted; verify a default or supply a value. Optional fields may remain empty.")
    return {"values": values, "uncertain": uncertain, "errors": errors, "metadata": metadata,
            "provenance": candidates, "review_required": True}


def normalize(config: Analysis):
    t, h, m = config.training, config.hardware, config.model
    limit = h.memory_limit_gib if h.memory_limit_gib is not None else h.vram_gib * .9
    training = {"num_examples": t.examples, "batch_size_per_gpu": t.batch_size,
                "epochs": t.epochs, "sequence_length": t.sequence_length,
                "gradient_accumulation_steps": t.gradient_accumulation_steps,
                "gpu_count": t.script_gpu_count, "distributed_strategy": t.strategy,
                "precision": t.precision, "optimizer": t.optimizer,
                "gradient_dtype": "fp32", "optimizer_state_dtype": "fp32",
                "master_weights": "none", "gradient_checkpointing": False}
    workload = None
    if m.total_params is not None:
        workload = {"model": {"total_params": m.total_params, "trainable_params": m.total_params,
                               "logical_buffer_bytes": 0},
                    "tensor_inventory": {"weights": [{"numel": m.total_params, "dtype": "fp32"}]},
                    "training": training}
    assumptions = {}
    if m.activation_gib is not None and m.auxiliary_gib is not None:
        assumptions = {"saved_activation_bytes": round(m.activation_gib * GIB),
                       "auxiliary_tensor_bytes": round(m.auxiliary_gib * GIB)}
    peaks = {}
    if h.compute_tflops is not None:
        peaks[t.precision] = h.compute_tflops
    if h.fp32_tflops is not None:
        peaks["fp32"] = h.fp32_tflops
    runtime_assumptions = {"runtime_model": "custom", "assume_data_ready": t.assume_data_ready}
    if m.flops_per_microbatch is not None:
        runtime_assumptions["forward_backward_flops_per_microbatch"] = m.flops_per_microbatch
    if m.hbm_bytes_per_microbatch is not None:
        runtime_assumptions["hbm_bytes_per_microbatch"] = m.hbm_bytes_per_microbatch
    return {"workload": workload, "training": training,
            "dataset": {"dataset": {"num_train_examples": t.examples}},
            "runtime_assumptions": runtime_assumptions,
            "memory_assumptions": assumptions,
            "hardware": {"gpu": {"peak_compute_tflops": peaks, "memory_bandwidth_gb_per_second": h.memory_bandwidth_gbps, "gpu_id": h.option_id, "vram_gib_per_gpu": h.vram_gib,
                                   "safety_reserve_gib_per_gpu": h.vram_gib - limit},
                         "cpu": {"available_ram_gib": h.cpu_ram_gib},
                         "storage": {"available_gib": h.nvme_gib},
                         "interconnect": {"gpu_gpu_effective_gb_per_second": h.gpu_link_gbps}},
            "deployment": {"strategy": t.strategy, "world_size": t.script_gpu_count,
                           "gpus_per_node": t.script_gpu_count},
            "calibration": {"whole_microbatch_seconds": t.seconds_per_batch,
                            "measured_peak_vram_gib": t.measured_peak_vram_gib},
            "pricing": {"usd_per_gpu_hour": h.price_per_gpu_hour,
                        "active_gpus": t.script_gpu_count},
            "memory_limit_gib": limit}


@router.post("/analysis/normalize")
def normalize_route(config: Analysis):
    return {"config": config.model_dump(), "normalized": normalize(config)}


def predict(config: Analysis):
    t, h, m = config.training, config.hardware, config.model
    n = normalize(config)
    steps = math.ceil(math.ceil(t.examples / t.script_gpu_count) / t.batch_size) * t.epochs
    updates = math.ceil(math.ceil(math.ceil(t.examples / t.script_gpu_count) / t.batch_size) / t.gradient_accumulation_steps) * t.epochs
    hours = steps * t.seconds_per_batch / 3600 if t.seconds_per_batch is not None else None
    notes = ["PyTorch only. Prediction never executes uploaded scripts or allocates GPUs.",
             "Identical GPUs, balanced data parallel ranks, partial batches retained; accumulation flushes each epoch.",
             "Available GPUs do not automatically accelerate the script. Only script GPU count participates in the workload.",
             "Runtime uses a user-supplied whole-microbatch measurement matching this workload, hardware, GPU count and sharding. No linear multi-GPU speedup is assumed.",
             "If matching timing is absent, explicit FLOPs, HBM traffic, hardware throughput/bandwidth and data-pipeline assumptions can drive the existing runtime planner. Otherwise runtime/cost stay unknown.",
             "Memory limit defaults to 90% per-GPU capacity. CPU RAM and storage requirements are unknown without allocation metadata.",
             "Manual memory assumptions: fully trainable dense FP32 stored weights and gradients, FP32 optimizer states, no extra master copy, no checkpointing; review advanced settings.",
             "Price is USD per GPU-hour from the saved spot snapshot or a manual override, excluding other infrastructure, retries and tax."]
    peak = t.measured_peak_vram_gib
    source = "measured" if peak is not None else "unknown"
    memory = None
    if peak is None and m.total_params is not None:
        # Use the repository's existing planner; do not create a new sharding formula.
        try:
            memory = estimate_gpu_memory(n["workload"], n["memory_assumptions"], n["hardware"], deployment=n["deployment"])
            peak = memory["gpu"]["peak_vram_gib"]
            if peak is not None:
                source = "planning estimate"
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, detail=f"Memory metadata: {exc}") from exc
        notes.append("Aggregate parameter inventory uses the full parameter count as the largest tensor: conservative optimizer scratch, not actual tensor layout. Buffers assumed zero. Supply activation and auxiliary storage to complete a peak estimate.")
    if peak is None:
        notes.append("Peak VRAM is unknown. Supply matching measured peak, or parameter count plus activation and auxiliary storage assumptions. Parameter count alone cannot establish fit.")
    runtime = None
    timing_source = "user-supplied whole-microbatch measurement" if hours is not None else "unknown"
    if hours is None and n["workload"] is not None:
        try:
            runtime = estimate_runtime(n["workload"], n["dataset"], n["hardware"],
                                       n["runtime_assumptions"], n["deployment"])
            if runtime["status"] == "estimated_with_assumptions":
                hours = runtime["scenarios"]["slow"]["training_hours"]
                timing_source = "metadata-only planning estimate (slow scenario, not an upper bound)"
                notes.extend(runtime.get("assumptions", []))
            else:
                notes.extend(runtime.get("missing_information", []))
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, detail=f"Runtime metadata: {exc}") from exc
    cost = hours * t.script_gpu_count * h.price_per_gpu_hour if hours is not None and h.price_per_gpu_hour is not None else None
    comparison = []
    for offer in catalog()[:10]:
        capacity = offer["vram_gib"]
        comparison.append({**offer, "fits": peak <= capacity * .9 if peak is not None and capacity else None,
                           "cost_usd": None, "note": "90% default limit. Runtime and cost require hardware-specific calibration."})
    return {"mode": "prediction", "config": config.model_dump(), "normalized": n,
            "training_steps": steps, "optimizer_updates": updates, "training_hours": hours,
            "timing_source": timing_source, "runtime_details": runtime,
            "gpu_hours": hours * t.script_gpu_count if hours is not None else None,
            "training_cost_usd": round(cost, 4) if cost is not None else None,
            "gpu": {"peak_vram_gib": peak, "source": source, "capacity_gib": h.vram_gib,
                    "limit_gib": n["memory_limit_gib"], "utilization_percent": peak / h.vram_gib * 100 if peak is not None else None,
                    "fits": peak <= n["memory_limit_gib"] if peak is not None else None},
            "cpu": {"required_gib": None, "capacity_gib": h.cpu_ram_gib},
            "memory_details": memory, "assumptions": notes, "comparison": comparison,
            "benchmark": {"cloud_available": False, "estimated_gpu_cost_usd": h.price_per_gpu_hour * t.script_gpu_count * 60 / 3600 if h.price_per_gpu_hour is not None else None,
                          "assumed_seconds": 60, "note": "Illustrative 60-second GPU-only cost; excludes provisioning, minimum billing, CPU, storage and failures. No cloud job has been submitted."}}


@router.post("/analysis/estimate")
def estimate_route(config: Analysis):
    return predict(config)


@router.get("/benchmark/script")
def benchmark_script():
    return FileResponse(Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py",
                        media_type="text/x-python", filename="run_benchmark.py")


@router.post("/benchmark/cloud")
def cloud_benchmark():
    raise HTTPException(501, "Nebius execution is not configured. No job was submitted and no charge was incurred. Use the local benchmark runner.")


class BenchmarkReport(StrictModel):
    schema_version: Literal[1]
    config: Analysis
    steps: int = Field(ge=1, le=10**9)
    seconds_per_batch: float = Field(gt=0, le=10**6)
    measured_peak_vram_gib: float = Field(gt=0, le=100000)
    gpu_name: str = Field(min_length=1, max_length=200)
    capacity_gib: float = Field(gt=0, le=10000)
    scope: Literal['whole_training_microbatch']
    note: str = Field(max_length=2000)


class BenchmarkImport(StrictModel):
    config: Analysis
    report: BenchmarkReport


@router.post('/benchmark/import')
def import_benchmark(request: BenchmarkImport):
    current, measured = request.config.model_dump(), request.report.config.model_dump()
    for value in (current, measured):
        value['training']['seconds_per_batch'] = None
        value['training']['measured_peak_vram_gib'] = None
    if current != measured:
        raise HTTPException(422, 'Benchmark workload/hardware differs from the current inputs. Restore the measured configuration or run a new benchmark.')
    if request.config.training.script_gpu_count != 1:
        raise HTTPException(422, 'The downloadable benchmark runner supports one GPU only.')
    capacity = request.config.hardware.vram_gib
    if abs(request.report.capacity_gib - capacity) / capacity > .1:
        raise HTTPException(422, 'Benchmark GPU capacity differs from the selected hardware.')
    if request.report.measured_peak_vram_gib > request.report.capacity_gib:
        raise HTTPException(422, 'Reported peak exceeds the measured GPU capacity.')
    result = request.config.model_dump()
    result['training']['seconds_per_batch'] = request.report.seconds_per_batch
    result['training']['measured_peak_vram_gib'] = request.report.measured_peak_vram_gib
    return {'config': Analysis.model_validate(result).model_dump(), 'gpu_name': request.report.gpu_name,
            'note': 'User-supplied local report, not independently verified. Review GPU identity and adapter contract. Reserved-memory peak excludes full-run initialization/checkpoint and other-process allocations.'}
