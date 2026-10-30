"""Health endpoint. Estimation endpoints live in estimate.py / analysis.py."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
