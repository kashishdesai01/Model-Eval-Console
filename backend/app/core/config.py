from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEC_", env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://mec:mec@localhost:5438/mec"
    api_url: str = "http://localhost:8000"
    torch_threads: int = Field(default=2, ge=1, le=32)
    lease_seconds: int = Field(default=60, ge=3, le=3600)
    log_level: str = "INFO"
    code_version: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
