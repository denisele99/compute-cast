from dataclasses import dataclass
from pyexpat import model
from typing import Iterable

from ..schemas import Workload
from math import ceil, isfinite

class model:
    
    pass
class training:
    pass

DTYPE_BYTES = { #TODO check this is correct for PyTorch and TensorFlow
    "fp64": 8,
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int64": 8,
    "int32": 4,
    "int8": 1,
    "uint8": 1,
}

@dataclass(frozen=True)
class TensorSpec:
    name: str
    numel: int
    dtype: str

    @property
    def memory_bytes(self) -> int:
        if self.numel < 0:
            raise ValueError("numel must be nonnegative")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return self.numel * DTYPE_BYTES[self.dtype]


def estimate_model_state_memory(
    *,
    weights: Iterable[TensorSpec],
    gradients: Iterable[TensorSpec],
    optimizer_states: Iterable[TensorSpec],
    master_weights: Iterable[TensorSpec] = (),
) -> dict[str, float]:
    """Resident GPU tensor storage; excludes temporary workspace."""
    groups = {
        "weights": weights,
        "gradients": gradients,
        "optimizer_states": optimizer_states,
        "master_weights": master_weights,
    }

    sizes = {
        name: sum(t.memory_bytes for t in tensors)
        for name, tensors in groups.items()
    }
    sizes["total"] = sum(sizes.values())

    return {
        f"{name}_gib": size / 1024**3
        for name, size in sizes.items()
    }
    

#Calculate model-state memory
def estimate_model_state_memory(model, training):
    '''Estimate the number of model states (parameters) in memory during training
M_states = M_weights + M_gradients  +M_optimizer_tensors+M_master_weights
'''
    n_params = model.num_parameters
    
    
    return 0    

def estimate_activations(model, training, attention_backend):
    return 0

def estimate_workspace(model, training, software):
    #CUDA context, PyTorch caching allocator
    return 0

def estimate_training_peak(states, activations, workspace):
    return (0, 0)

def compare_with_capacity(peak_range, hardware):
    return 0

def estimate_gpu_peak_memory(model, training, attention_backend, software, hardware):
    
    states = estimate_model_states(model, training)
    activations = estimate_activations(model, training, attention_backend)
    workspace = estimate_workspace(model, training, software)

    peak_range = estimate_training_peak(
        states=states,
        activations=activations,
        workspace=workspace,
    )

    fit = compare_with_capacity(peak_range, hardware)


def estimate_steps(examples: int, batch_size: int, epochs: int) -> int:
    """Single GPU; retain partial final batches; no gradient accumulation."""
    if min(examples, batch_size, epochs) < 1:
        raise ValueError("Examples, batch size and epochs must be positive")
    return ceil(examples / batch_size) * epochs


def estimate_runtime(steps: int, seconds_per_batch: float) -> float:
    if steps < 1 or not isfinite(seconds_per_batch) or seconds_per_batch <= 0:
        raise ValueError("Steps and benchmark time must be positive and finite")
    return steps * seconds_per_batch / 3600

def estimate_memory(): #TODO
    return 0

#Estimate runtime given hardware class + model params + training params




"""Capacity check only. Replace with a calibrated memory estimator later."""


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
