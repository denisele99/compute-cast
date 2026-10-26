"""Server-side configuration. Never expose API keys through API responses."""
from dataclasses import dataclass, field
from functools import lru_cache
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    app_name: str
    log_level: str
    nebius_api_key: str | None = field(repr=False)
    nebius_model: str | None


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "GPU Planner Starter"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        nebius_api_key=os.getenv("NEBIUS_API_KEY") or None,
        nebius_model=os.getenv("NEBIUS_MODEL") or None,
    )
