"""HTTP routes. Combine core endpoints and the recommender under one router."""
from fastapi import APIRouter

from . import core, recommender

router = APIRouter()
router.include_router(core.router)
router.include_router(recommender.router)
