from fastapi import FastAPI
from .config import get_settings
from .api.routes import router
from .services.analysis_store import AnalysisStore


# backend/app/main.py

def create_app(store: AnalysisStore | None = None) -> FastAPI:

    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.4.0"
    )
    # One in-memory store per process; run uvicorn with a single worker.
    app.state.store = store or AnalysisStore()
    app.state.store.sweep()

    app.include_router(
        router,
        prefix="/api"
    )

    return app


app = create_app()
