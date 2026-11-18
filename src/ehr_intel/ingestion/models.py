"""Data models for the ingestion stage.

Two layers:

* :class:`RawRecord` - a loosely-typed, format-agnostic view of one source
  record exactly as a reader found it (all strings, anything may be missing).
* The *canonical* models (:class:`PatientDemographics`, :class:`Encounter`,
  :class:`ClinicalRecord`) - validated, normalized output ready for FHIR
  mapping. Each carries the audit trail of every change made to produce it.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SourceFormat(StrEnum):
    JSON = "json"
    CSV = "csv"
    TEXT = "text"


class Gender(StrEnum):
    """FHIR ``AdministrativeGender`` value set."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class RecordType(StrEnum):
    """Canonical clinical document categories."""

    DISCHARGE_SUMMARY = "discharge_summary"
    PROGRESS_NOTE = "progress_note"
    CLINICAL_NOTE = "clinical_note"
    RADIOLOGY_REPORT = "radiology_report"
    LAB_RESULT = "lab_result"
    SCANNED_DOCUMENT = "scanned_document"

    @property
    def is_diagnostic(self) -> bool:
        """Diagnostic results map to ``DiagnosticReport``; narratives to ``DocumentReference``."""
        return self in {RecordType.RADIOLOGY_REPORT, RecordType.LAB_RESULT}

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()


class EncounterClass(StrEnum):
    """Subset of HL7 v3 ``ActEncounterCode``."""

    INPATIENT = "IMP"
    AMBULATORY = "AMB"
    EMERGENCY = "EMER"
    UNKNOWN = "UNK"


class AuditAction(StrEnum):
    NORMALIZED = "normalized"  # value reformatted to canonical form
    DEFAULTED = "defaulted"  # missing value replaced by a safe default
    INFERRED = "inferred"  # missing value derived from other data
    LINKED = "linked"  # record attached to a patient by demographics
    CONFLICT = "conflict"  # disagreeing values; one chosen, the other recorded
    DEDUPLICATED = "deduplicated"  # duplicate source record merged into this one
    INVALID = "invalid"  # value unusable and discarded
    WARNING = "warning"  # needs human review, nothing changed
    REJECTED = "rejected"  # whole record excluded


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """One change (or observation) made while cleaning a record."""

    model_config = ConfigDict(frozen=True)

    field: str
    action: AuditAction
    original: str | None = None
    new: str | None = None
    reason: str
    source_id: str | None = None

    def __str__(self) -> str:
        change = f" {self.original!r} -> {self.new!r}" if self.original or self.new else ""
        return f"[{self.action}] {self.field}:{change} ({self.reason})"


# ---------------------------------------------------------------------------
# Raw (reader output)
# ---------------------------------------------------------------------------


class RawObservation(BaseModel):
    """A single lab analyte as found in the source."""

    test_name: str | None = None
    loinc: str | None = None
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    flag: str | None = None


class RawRecord(BaseModel):
    """Format-agnostic, unvalidated view of one source record."""

    source_id: str
    source_format: SourceFormat
    source_location: str = Field(description="File (and row/line) the record came from.")

    mrn: str | None = None
    name: str | dict[str, Any] | None = None
    dob: str | None = None
    sex: str | None = None

    encounter_id: str | None = None
    encounter_class: str | None = None
    encounter_start: str | None = None
    encounter_end: str | None = None

    doc_type: str | None = None
    title: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None
    observations: list[RawObservation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical (cleaner output)
# ---------------------------------------------------------------------------


class PatientDemographics(BaseModel):
    """Canonical patient identity, one per resolved MRN."""

    mrn: str = Field(pattern=r"^\d{4,12}$", description="Zero-padded numeric MRN.")
    given_name: str | None = None
    family_name: str | None = None
    birth_date: date | None = None
    gender: Gender = Gender.UNKNOWN
    source_ids: list[str] = Field(default_factory=list)
    audit: list[AuditEntry] = Field(default_factory=list)

    @property
    def display_name(self) -> str:
        parts = [p for p in (self.given_name, self.family_name) if p]
        return " ".join(parts) if parts else f"MRN {self.mrn}"


class Encounter(BaseModel):
    encounter_id: str
    mrn: str
    encounter_class: EncounterClass = EncounterClass.UNKNOWN
    start: datetime | None = None
    end: datetime | None = None


class LabObservation(BaseModel):
    """One normalized lab analyte."""

    test_name: str
    loinc: str | None = Field(default=None, pattern=r"^\d{1,7}-\d$")
    value: float | None = None
    value_text: str | None = None
    unit: str | None = None
    reference_low: float | None = None
    reference_high: float | None = None
    reference_text: str | None = None
    interpretation: Literal["H", "L", "N"] | None = None

    @property
    def is_abnormal(self) -> bool:
        return self.interpretation in {"H", "L"}

    def describe(self) -> str:
        value = f"{self.value:g}" if self.value is not None else (self.value_text or "?")
        unit = f" {self.unit}" if self.unit else ""
        ref = f" (ref {self.reference_text})" if self.reference_text else ""
        flag = {"H": " [HIGH]", "L": " [LOW]"}.get(self.interpretation or "", "")
        return f"{self.test_name}: {value}{unit}{ref}{flag}"


class ClinicalRecord(BaseModel):
    """A cleaned clinical document or result, ready for FHIR mapping."""

    record_id: str
    mrn: str
    record_type: RecordType
    title: str
    text: str = Field(min_length=1)
    effective_at: datetime | None = None
    author: str | None = None
    encounter_id: str | None = None
    source_format: SourceFormat
    source_ids: list[str] = Field(min_length=1)
    observations: list[LabObservation] = Field(default_factory=list)
    content_hash: str
    audit: list[AuditEntry] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class RejectedRecord(BaseModel):
    source_id: str
    source_location: str
    reason: str


class CleanedDataset(BaseModel):
    """Everything the ingestion stage produces."""

    patients: dict[str, PatientDemographics] = Field(default_factory=dict)
    encounters: dict[str, Encounter] = Field(default_factory=dict)
    records: list[ClinicalRecord] = Field(default_factory=list)
    rejected: list[RejectedRecord] = Field(default_factory=list)
    raw_record_count: int = 0

    def records_for(self, mrn: str) -> list[ClinicalRecord]:
        return [r for r in self.records if r.mrn == mrn]

    def encounters_for(self, mrn: str) -> list[Encounter]:
        return [e for e in self.encounters.values() if e.mrn == mrn]

    def audit_log(self) -> list[dict[str, Any]]:
        """Flattened audit trail across patients and records."""
        rows: list[dict[str, Any]] = []
        for patient in self.patients.values():
            rows += [
                {"scope": "patient", "id": patient.mrn, **a.model_dump()} for a in patient.audit
            ]
        for record in self.records:
            rows += [
                {"scope": "record", "id": record.record_id, **a.model_dump()} for a in record.audit
            ]
        return rows

    def stats(self) -> dict[str, Any]:
        actions: dict[str, int] = {}
        for row in self.audit_log():
            actions[row["action"]] = actions.get(row["action"], 0) + 1
        by_type: dict[str, int] = {}
        for record in self.records:
            by_type[record.record_type] = by_type.get(record.record_type, 0) + 1
        return {
            "raw_records": self.raw_record_count,
            "clean_records": len(self.records),
            "rejected_records": len(self.rejected),
            "patients": len(self.patients),
            "encounters": len(self.encounters),
            "records_by_type": dict(sorted(by_type.items())),
            "audit_actions": dict(sorted(actions.items())),
        }
