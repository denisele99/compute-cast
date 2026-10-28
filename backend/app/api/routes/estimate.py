from fastapi import APIRouter
from backend.app.schemas import EstimateInput
from backend.app.services.estimators.estimator import run_estimation

router = APIRouter(prefix="/estimate", tags=["Estimation"])


@router.post("")
def estimate(data: EstimateInput):
    return run_estimation(data)


@router.post("/compare")
def compare(data: EstimateInput):
    # Comparison logic
    pass


#POST /api/routes/estimate
#POST /api/routes/estimate/compare