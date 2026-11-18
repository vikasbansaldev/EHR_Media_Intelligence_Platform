"""FastAPI application: search, patient detail and the static clinician UI.

Run with ``ehr-intel serve`` or ``uvicorn ehr_intel.api.app:create_app --factory``.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ehr_intel import __version__
from ehr_intel.api.schemas import (
    FilterOptions,
    HealthResponse,
    PatientDetail,
    PatientListItem,
    ResourceSummary,
    SearchRequest,
)
from ehr_intel.config import Settings, get_settings
from ehr_intel.fhir.store import RECORD_RESOURCE_TYPES, BundleStore, record_kind
from ehr_intel.logging_config import configure_logging
from ehr_intel.search import SEARCHABLE_TYPES, IndexNotReadyError, SearchResponse, SearchService
from ehr_intel.summarization import ClinicalSummary, PatientNotFoundError, SummarizationService

logger = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    store: BundleStore
    summaries: SummarizationService
    search: SearchService


def build_services(settings: Settings) -> Services:
    settings.ensure_dirs()
    store = BundleStore(settings.bundle_db_path)
    summaries = SummarizationService(settings, store)
    search = SearchService(settings, store, summaries)
    return Services(settings, store, summaries, search)


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]
router = APIRouter()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic search across patient records and AI summaries",
)
def search(
    body: SearchRequest,
    services: ServicesDep,
    resource_type: Annotated[
        list[str] | None,
        Query(description=f"Repeatable. One of: {', '.join(SEARCHABLE_TYPES)}"),
    ] = None,
    date_from: Annotated[date | None, Query(description="Inclusive, YYYY-MM-DD")] = None,
    date_to: Annotated[date | None, Query(description="Inclusive, YYYY-MM-DD")] = None,
) -> SearchResponse:
    if resource_type:
        unknown = sorted(set(resource_type) - set(SEARCHABLE_TYPES))
        if unknown:
            raise HTTPException(
                422, f"unknown resource_type {unknown}; expected any of {list(SEARCHABLE_TYPES)}"
            )
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must be on or before date_to")
    try:
        return services.search.search(
            body.query,
            top_k=body.top_k,
            resource_types=resource_type,
            date_from=date_from,
            date_to=date_to,
        )
    except IndexNotReadyError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get(
    "/filters",
    response_model=FilterOptions,
    tags=["search"],
    summary="Values available for the UI filter controls",
)
def filters(services: ServicesDep) -> FilterOptions:
    lo, hi = services.search.date_bounds()
    return FilterOptions(resource_types=services.search.available_types(), date_min=lo, date_max=hi)


# ---------------------------------------------------------------------------
# Patients & resources
# ---------------------------------------------------------------------------


@router.get("/patients", response_model=list[PatientListItem], tags=["patients"])
def list_patients(services: ServicesDep) -> list[PatientListItem]:
    return [
        PatientListItem(
            mrn=p.mrn,
            name=p.name,
            birth_date=p.birth_date,
            gender=p.gender,
            bundle_valid=p.is_valid,
            identity_flags=p.flags,
        )
        for p in services.store.patients()
    ]


@router.get(
    "/patients/{mrn}",
    response_model=PatientDetail,
    tags=["patients"],
    summary="Patient demographics, AI summary and linked FHIR resources",
)
def patient_detail(mrn: str, services: ServicesDep) -> PatientDetail:
    patient = services.store.patient(mrn)
    if patient is None:
        raise HTTPException(404, f"patient {mrn} not found")
    resources = services.store.resources(mrn=mrn)
    linked = [
        ResourceSummary(
            resource_id=r.resource_id,
            resource_type=r.resource_type,
            kind=record_kind(r.resource) if r.resource_type in RECORD_RESOURCE_TYPES else None,
            title=r.title,
            date=r.effective_date[:10] if r.effective_date else None,
        )
        for r in resources
        if r.resource_type != "Patient"
    ]
    linked.sort(key=lambda r: r.date or "", reverse=True)
    return PatientDetail(
        mrn=patient.mrn,
        name=patient.name,
        birth_date=patient.birth_date,
        gender=patient.gender,
        bundle_valid=patient.is_valid,
        identity_flags=patient.flags,
        summary=services.summaries.latest(mrn),
        resources=linked,
        resource_counts=dict(Counter(r.resource_type for r in resources)),
    )


@router.get(
    "/patients/{mrn}/summary",
    response_model=ClinicalSummary,
    tags=["patients"],
    summary="AI summary (served from cache; generated on first request)",
)
def patient_summary(
    mrn: str, services: ServicesDep, refresh: bool = Query(False, description="Bypass the cache")
) -> ClinicalSummary:
    try:
        return services.summaries.summarize(mrn, force=refresh)
    except PatientNotFoundError as exc:
        raise HTTPException(404, f"patient {mrn} not found") from exc


@router.get("/patients/{mrn}/bundle", tags=["fhir"], summary="The patient's FHIR R4 Bundle")
def patient_bundle(mrn: str, services: ServicesDep) -> dict[str, Any]:
    bundle = services.store.get_bundle(mrn)
    if bundle is None:
        raise HTTPException(404, f"patient {mrn} not found")
    return bundle


@router.get("/resources/{resource_id}", tags=["fhir"], summary="A single FHIR resource")
def get_resource(resource_id: str, services: ServicesDep) -> dict[str, Any]:
    stored = services.store.resource(resource_id)
    if stored is None:
        raise HTTPException(404, f"resource {resource_id} not found")
    return stored.resource


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(services: ServicesDep) -> HealthResponse:
    return HealthResponse(
        status="ok" if services.search.ready else "degraded",
        patients=services.store.count(),
        indexed_documents=len(services.search.index),
        embedder=services.search.embedder.name,
        vector_backend=services.search.index.backend,
        summary_provider=services.summaries.provider.name,
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        if getattr(app.state, "services", None) is None:
            app.state.services = build_services(settings)
        svc: Services = app.state.services
        if svc.store.count() == 0:
            logger.warning("FHIR store is empty - run `ehr-intel run-all` to populate it")
        elif not svc.search.ready:
            svc.search.load_or_build()
        yield

    app = FastAPI(
        title="EHR Media Intelligence API",
        version=__version__,
        description="Semantic search and AI summaries over FHIR R4 patient records. "
        "All data is synthetic. Summaries are AI-generated and are not clinical "
        "decisions.",
        lifespan=lifespan,
    )
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)

    if settings.frontend_dir.is_dir():
        # Mounted last so API routes take precedence; html=True serves index.html at "/".
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="ui")
    return app
