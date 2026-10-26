from typing import Literal
from pydantic import BaseModel, Field




class Workload(BaseModel):
    examples: int = Field(default=100000, ge=1, le=10**12)
    batch_size: int = Field(default=16, ge=1, le=10**6)
    epochs: int = Field(default=10, ge=1, le=100000)
    sequence_length: int = Field(default=512, ge=1, le=10**7)
    seconds_per_batch: float = Field(default=0.2, gt=0, le=10**6, allow_inf_nan=False)
    hourly_price_usd: float = Field(default=1, ge=0, le=10**6, allow_inf_nan=False)
    measured_peak_vram_gb: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    target_vram_gb: float = Field(default=24, gt=0, le=10000, allow_inf_nan=False)





class EstimateResponse(BaseModel):
    training_steps: int
    training_hours: float
    gpu_hours: float
    training_cost_usd: float
    memory_status: Literal["unknown", "exceeds_capacity", "below_capacity_check_headroom"]
    assumptions: list[str]



class GPUOption(BaseModel):
    id: str
    label: str
    vram_gb: float
    gpu_count: int = 1
    hourly_price_usd: float | None = None
    illustrative: bool = True


class GPUCatalogResponse(BaseModel):
    items: list[GPUOption]
    note: str



class BenchmarkRequest(BaseModel):
    workload_description: str = Field(min_length=1, max_length=5000)
    gpu_option_id: str = Field(min_length=1, max_length=100)


class WorkloadDescription(BaseModel):
    description: str = Field(min_length=1, max_length=5000)
