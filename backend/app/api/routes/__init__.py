"""HTTP routes. Combine all route modules under one router (mounted at /api)."""
from fastapi import APIRouter

from . import advisor, analysis, benchmark, core, estimate, hardware, parse, recommender

router = APIRouter()
router.include_router(core.router)
router.include_router(parse.router)
router.include_router(hardware.router)
router.include_router(analysis.router)
router.include_router(estimate.router)
router.include_router(benchmark.router)
router.include_router(recommender.router)
router.include_router(advisor.router)
