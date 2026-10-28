"""POST /api/recommendations: code review via the hosted model. Needs NEBIUS_API_KEY server-side."""

import json
import logging
import os
import importlib.util

from fastapi import APIRouter, HTTPException
try:
    from openai import APIError
except ImportError:  # openai is optional until recommendations are used
    APIError = ()  # type: ignore[assignment,misc]; an empty tuple catches nothing
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from ...services.recommender import recommend, safe_path

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
)


class RecommendationRequest(BaseModel):
    consent_to_paid_request: bool = False
    prompt: str = Field(min_length=1, max_length=10_000)

    # Relative filename -> complete file contents
    files: dict[str, str]

    # Hardware, workload, launch command, estimator results, etc.
    context: dict = Field(default_factory=dict)


@router.get("/status")
def recommendation_status():
    return {"configured": bool(os.getenv("NEBIUS_API_KEY")) and importlib.util.find_spec("openai") is not None,
            "note": "Uses the configured Nebius model provider. Source files are sent only after explicit consent; provider charges may apply."}


@router.post("")
async def create_recommendations(request: RecommendationRequest):
    if not request.consent_to_paid_request:
        raise HTTPException(403, "Explicit consent to send source to the paid model provider is required.")
    if not recommendation_status()["configured"]:
        raise HTTPException(503, "Recommendations need the OpenAI SDK and NEBIUS_API_KEY configured on the backend. No provider request was sent.")
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