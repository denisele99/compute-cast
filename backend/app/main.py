from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import get_settings
from .api.routes import router
from .services.analysis_store import AnalysisStore


# backend/app/main.py

def create_app(store: AnalysisStore | None = None, bench=None) -> FastAPI:

    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.4.0"
    )
    # One in-memory store per process; run uvicorn with a single worker.
    app.state.store = store or AnalysisStore()
    app.state.store.sweep()
    # Benchmark job service (durable store + Nebius client); created lazily unless injected (tests).
    app.state.bench = bench

    # GitHub Pages frontend -> Render backend. Bearer tokens, not cookies, so no credentials mode.
    from .services.benchmarks.policy import load_policy
    origins = list(load_policy().cors_origins)
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                           allow_methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
                           allow_headers=['Authorization', 'Content-Type'], max_age=600)

    app.include_router(
        router,
        prefix="/api"
    )

    return app


app = create_app()
