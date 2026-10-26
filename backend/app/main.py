from fastapi import FastAPI
from .config import get_settings
from .routes import router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.3.0")
    app.include_router(router, prefix="/api")
    return app


app = create_app()
