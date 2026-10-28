"""Compatibility implementation for the documented original /api/estimate contract."""
from math import ceil, isfinite
from ..schemas import Workload

def estimate_steps(examples: int, batch_size: int, epochs: int) -> int:
    """Single GPU; retain partial final batches; no gradient accumulation."""
    if min(examples, batch_size, epochs) < 1:
        raise ValueError("Examples, batch size and epochs must be positive")
    return ceil(examples / batch_size) * epochs


def estimate_runtime(steps: int, seconds_per_batch: float) -> float:
    if steps < 1 or not isfinite(seconds_per_batch) or seconds_per_batch <= 0:
        raise ValueError("Steps and benchmark time must be positive and finite")
    return steps * seconds_per_batch / 3600

def memory_status(measured_peak_gb: float | None, capacity_gb: float) -> str:
    if measured_peak_gb is None:
        return "unknown"
    if measured_peak_gb >= capacity_gb:
        return "exceeds_capacity"
    return "below_capacity_check_headroom"


"""No live prices. A provider adapter can supply validated dated rates later."""


def training_cost_usd(hours: float, hourly_rate: float) -> float:
    return round(hours * hourly_rate, 2)




def estimate_workload(workload: Workload) -> dict:
    steps = estimate_steps(workload.examples, workload.batch_size, workload.epochs)
    hours = estimate_runtime(steps, workload.seconds_per_batch)
    peak = workload.measured_peak_vram_gb
    return {
        "training_steps": steps,
        "training_hours": round(hours, 3),
        "gpu_hours": round(hours, 3),
        "training_cost_usd": training_cost_usd(hours, workload.hourly_price_usd),
        "memory_status": memory_status(peak, workload.target_vram_gb),
        "assumptions": [
            "One GPU; partial final batches are retained; no gradient accumulation.",
            "Benchmark time must include data loading, forward/backward passes and optimizer updates.",
            "Use measurements from the same model, GPU, sequence length, batch size and precision.",
            "Sequence length is recorded for context; it does not independently scale this estimate.",
            "Excludes setup, validation, checkpointing, idle time, storage, transfer and taxes.",
            "Price is manually entered, not a live cloud quote. Default benchmark and price are illustrative.",
            "Below-capacity memory usage does not guarantee fit; leave headroom and test longest inputs."
        ]
    }
