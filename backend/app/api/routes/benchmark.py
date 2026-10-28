from fastapi import APIRouter
from backend.app.models.schemas import BenchmarkRequest
from backend.app.services.benchmark_service import submit_benchmark

router = APIRouter(tags=["Benchmarks"])


@router.post("/benchmarks")
def create_benchmark(request: BenchmarkRequest):
    return submit_benchmark(request)