from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select, text

from app.api import comparisons, predict, registry, runs
from app.api.errors import install_handlers
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import EvalRun
from app.db.session import session_factory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.inference.service import InferenceService

    configure_logging(get_settings().log_level)
    app.state.inference = InferenceService()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Model Eval Console", version="0.1.0", lifespan=lifespan)
    install_handlers(app)
    for router in (registry.router, runs.router, comparisons.router, predict.router):
        app.include_router(router)

    @app.get("/healthz", include_in_schema=False)
    def health():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def ready():
        try:
            with session_factory()() as session:
                version = session.scalar(text("SELECT version_num FROM alembic_version"))
            if version != "0002":
                raise ValueError("Migrations are not current")
            return {"status": "ready"}
        except Exception:
            return JSONResponse(status_code=503, content={"status": "not-ready"})

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        with session_factory()() as session:
            depth = (
                session.scalar(
                    select(func.count())
                    .select_from(EvalRun)
                    .where(
                        EvalRun.status == "queued",
                    )
                )
                or 0
            )
        body = (
            generate_latest() + f"# TYPE mec_queue_depth gauge\nmec_queue_depth {depth}\n".encode()
        )
        return Response(body, media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
