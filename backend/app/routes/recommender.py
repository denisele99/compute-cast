"""POST /api/recommendations: code review via the hosted model. Needs NEBIUS_API_KEY server-side."""

import json
import logging

from fastapi import APIRouter, HTTPException
try:
    from openai import APIError
except ImportError:  # openai is optional until recommendations are used
    APIError = ()  # type: ignore[assignment,misc]; an empty tuple catches nothing
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from ..services.recommender import recommend, safe_path

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
)


class RecommendationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)

    # Relative filename -> complete file contents
    files: dict[str, str]

    # Hardware, workload, launch command, estimator results, etc.
    context: dict = Field(default_factory=dict)


@router.post("")
async def create_recommendations(request: RecommendationRequest):
    # Check user inputs before making a paid API request.
    try:
        if not request.prompt.strip() or not request.files:
            raise ValueError("Provide a prompt and at least one file.")

        for path in request.files:
            safe_path(path)

        payload = request.model_dump()
        if len(json.dumps(payload).encode("utf-8")) > 250_000:
            raise ValueError("Input exceeds the 250 KB limit.")

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        # recommend() is synchronous. Run it in a worker thread
        # so it doesn't block FastAPI's async event loop.
        result = await run_in_threadpool(
            recommend,
            prompt=request.prompt,
            files=request.files,
            context=request.context,
        )

        return result

    except APIError as exc:
        logger.warning("Nebius request failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="The model provider request failed.",
        ) from exc

    except (ValidationError, ValueError, SyntaxError) as exc:
        logger.warning("Invalid model output: %s", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="The model returned an invalid proposal. Try fewer files.",
        ) from exc