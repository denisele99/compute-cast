"""HTTP entry points. Estimation logic lives in services/estimation.py."""
from fastapi import APIRouter, HTTPException
from ...schemas import Workload, EstimateResponse, GPUCatalogResponse, BenchmarkRequest, WorkloadDescription
from ...services.legacy_estimation import estimate_workload
from ...clients.nebius import parse_workload

router = APIRouter()


@router.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@router.post("/estimate", response_model=EstimateResponse, tags=["Estimates"])
def estimate(workload: Workload):
    return estimate_workload(workload)


@router.get("/gpus", response_model=GPUCatalogResponse, tags=["GPUs"])
def list_gpus():
    return {
        "items": [{"id": f"demo-{size}gb", "label": f"Illustrative single GPU — {size} GB",
                   "vram_gb": size, "illustrative": True} for size in (16, 24, 48)],
        "note": "Illustrative capacity tiers, not real cloud offerings or live prices."
    }


@router.post("/workloads/parse", tags=["AI parsing — placeholder"],
             responses={501: {"description": "Not implemented"}})
def parse_description(request: WorkloadDescription):
    try:
        return parse_workload(request.description)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.post("/benchmarks", tags=["Benchmarks — placeholder"],
             responses={501: {"description": "Not implemented"}})
def create_benchmark(request: BenchmarkRequest):
    # TODO: submit to an isolated GPU worker once benchmarking is implemented.
    raise HTTPException(status_code=501, detail="GPU benchmarking is not implemented; no job was submitted.")
