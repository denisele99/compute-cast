from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/parse", tags=["Parsing"])


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Call appropriate parser
    pass


@router.post("/script")
def analyze_script():
    # Parse Python code
    pass

@router.post("/workloads/parse", tags=["AI parsing — placeholder"],
             responses={501: {"description": "Not implemented"}})
def parse_description(request: WorkloadDescription):
    try:
        return parse_workload(request.description)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc