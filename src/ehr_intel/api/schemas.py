"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from ehr_intel.summarization.models import ClinicalSummary


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, examples=["chest pain with elevated troponin"])
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class ResourceSummary(BaseModel):
    resource_id: str
    resource_type: str
    kind: str | None = None
    title: str | None
    date: str | None


class PatientListItem(BaseModel):
    mrn: str
    name: str | None
    birth_date: str | None
    gender: str | None
    bundle_valid: bool
    identity_flags: list[str] = []


class PatientDetail(PatientListItem):
    summary: ClinicalSummary | None
    resources: list[ResourceSummary]
    resource_counts: dict[str, int]


class FilterOptions(BaseModel):
    resource_types: list[str]
    date_min: str | None
    date_max: str | None


class HealthResponse(BaseModel):
    status: str
    patients: int
    indexed_documents: int
    embedder: str
    vector_backend: str
    summary_provider: str


class ErrorResponse(BaseModel):
    detail: str | list[dict[str, Any]]
