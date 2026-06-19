"""FastAPI application for FloodLens model and knowledge services."""

from contextlib import asynccontextmanager
import json
import time

from fastapi import Request
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.document_routes import router as document_router
from app.api.routes import router
from app.core.settings import settings
from app.services.document_service import reconcile_interrupted_documents
from app.services.system_monitoring_service import get_system_monitoring_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    reconcile_interrupted_documents()
    yield


app = FastAPI(title="FloodLens API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_http_telemetry(request: Request, call_next):
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        service = getattr(app.state, "system_monitoring_service", None)
        if service is None:
            service = get_system_monitoring_service()
        service.log_request(
            method=request.method,
            route=f"{request.method} {route_path}",
            status_code=status_code,
            duration_ms=duration_ms,
            model_version=current_model_version(),
        )


def current_model_version() -> str | None:
    if not settings.model_metadata_path.exists():
        return None
    try:
        return json.loads(settings.model_metadata_path.read_text(encoding="utf-8")).get(
            "model_version"
        )
    except (OSError, json.JSONDecodeError):
        return None


app.include_router(router)
app.include_router(document_router)
