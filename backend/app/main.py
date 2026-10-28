from fastapi import FastAPI
from .config import get_settings
from .api.routes import router
from .workflow import router as workflow_router


def create_app() -> FastAPI:
    app = FastAPI(title=get_settings().app_name, version="0.4.0")
    app.include_router(router, prefix="/api")
    app.include_router(workflow_router, prefix="/api")
    return app


app = create_app()
