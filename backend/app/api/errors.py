from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError


class Problem(Exception):
    def __init__(self, status: int, slug: str, detail: str):
        self.status, self.slug, self.detail = status, slug, detail


def problem_response(status: int, slug: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        content={
            "type": f"urn:mec:problem:{slug}",
            "title": slug.replace("-", " "),
            "status": status,
            "detail": detail,
        },
    )


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(Problem)
    async def handle_problem(_request: Request, exc: Problem) -> JSONResponse:
        return problem_response(exc.status, exc.slug, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return problem_response(422, "validation-error", details)

    @app.exception_handler(IntegrityError)
    async def handle_integrity(_request: Request, _exc: IntegrityError) -> JSONResponse:
        return problem_response(409, "conflict", "This immutable record already exists.")
