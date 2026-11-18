"""Turn stored FHIR resources and AI summaries into searchable documents."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ehr_intel.fhir.store import RECORD_RESOURCE_TYPES, BundleStore, PatientRow, record_kind
from ehr_intel.summarization.models import ClinicalSummary

SUMMARY_TYPE = "ClinicalSummary"
SearchableType = Literal["DocumentReference", "DiagnosticReport", "ClinicalSummary"]
SEARCHABLE_TYPES: tuple[str, ...] = (*RECORD_RESOURCE_TYPES, SUMMARY_TYPE)


class SearchDocument(BaseModel):
    doc_id: str
    mrn: str
    patient_name: str | None
    resource_id: str | None
    resource_type: str
    kind: str
    title: str
    date: str | None  # ISO date, YYYY-MM-DD
    text: str

    def embedding_text(self) -> str:
        """Text the embedder sees: title and kind give context to short records."""
        who = f"Patient {self.patient_name} MRN {self.mrn}." if self.patient_name else ""
        return f"{self.kind}: {self.title}. {self.text} {who}".strip()


def build_documents(
    store: BundleStore, summaries: dict[str, ClinicalSummary] | None = None
) -> list[SearchDocument]:
    patients: dict[str, PatientRow] = {p.mrn: p for p in store.patients()}
    docs: list[SearchDocument] = []
    last_date: dict[str, str] = {}

    for r in store.resources(types=RECORD_RESOURCE_TYPES):
        patient = patients.get(r.mrn)
        date = r.effective_date[:10] if r.effective_date else None
        if date and date > last_date.get(r.mrn, ""):
            last_date[r.mrn] = date
        docs.append(
            SearchDocument(
                doc_id=r.resource_id,
                mrn=r.mrn,
                patient_name=patient.name if patient else None,
                resource_id=r.resource_id,
                resource_type=r.resource_type,
                kind=record_kind(r.resource),
                title=r.title or record_kind(r.resource),
                date=date,
                text=" ".join((r.text or "").split()),
            )
        )

    for mrn, summary in sorted((summaries or {}).items()):
        docs.append(
            SearchDocument(
                doc_id=f"summary-{mrn}",
                mrn=mrn,
                patient_name=summary.patient_name,
                resource_id=None,
                resource_type=SUMMARY_TYPE,
                kind="AI summary",
                title=f"AI clinical summary - {summary.patient_name or mrn}",
                date=last_date.get(mrn),
                text=summary.render(),
            )
        )
    return docs
