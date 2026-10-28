from fastapi import APIRouter

from backend.app.api.routes import (
    estimate,
    parse,
    hardware,
    recommender
)

router = APIRouter()

router.include_router(estimate.router)
router.include_router(parse.router)
router.include_router(hardware.router)
router.include_router(recommender.router)