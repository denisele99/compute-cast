from fastapi import APIRouter

router = APIRouter(prefix="/hardware", tags=["Hardware"])


@router.get("/gpus")
def get_gpus():
    # Return GPU specifications
    pass


@router.get("/pricing")
def get_pricing():
    # Return GPU pricing
    pass