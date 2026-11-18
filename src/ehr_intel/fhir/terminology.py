"""Code systems and value sets used when building FHIR resources.

Kept in one place so terminology choices are reviewable and replaceable
(e.g. swapping LOINC document codes for a local catalogue).
"""

from __future__ import annotations

from typing import Final

from ehr_intel.ingestion.models import EncounterClass, RecordType

LOINC: Final = "http://loinc.org"
UCUM: Final = "http://unitsofmeasure.org"
V2_0203: Final = "http://terminology.hl7.org/CodeSystem/v2-0203"  # identifier type
V2_0074: Final = "http://terminology.hl7.org/CodeSystem/v2-0074"  # diagnostic service section
V3_ACT_CODE: Final = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
V3_NULL_FLAVOR: Final = "http://terminology.hl7.org/CodeSystem/v3-NullFlavor"
V3_INTERPRETATION: Final = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
OBSERVATION_CATEGORY: Final = "http://terminology.hl7.org/CodeSystem/observation-category"
US_CORE_DOC_CATEGORY: Final = (
    "http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category"
)

# LOINC document ontology codes for DocumentReference.type / DiagnosticReport.code.
RECORD_TYPE_CODES: Final[dict[RecordType, tuple[str, str]]] = {
    RecordType.DISCHARGE_SUMMARY: ("18842-5", "Discharge summary"),
    RecordType.PROGRESS_NOTE: ("11506-3", "Progress note"),
    RecordType.CLINICAL_NOTE: ("11488-4", "Consult note"),
    RecordType.SCANNED_DOCUMENT: ("34109-9", "Note"),
    RecordType.RADIOLOGY_REPORT: ("18748-4", "Diagnostic imaging study"),
    RecordType.LAB_RESULT: ("11502-2", "Laboratory report"),
}

DIAGNOSTIC_SECTIONS: Final[dict[RecordType, tuple[str, str]]] = {
    RecordType.RADIOLOGY_REPORT: ("RAD", "Radiology"),
    RecordType.LAB_RESULT: ("LAB", "Laboratory"),
}

ENCOUNTER_CLASS_DISPLAY: Final[dict[EncounterClass, str]] = {
    EncounterClass.INPATIENT: "inpatient encounter",
    EncounterClass.AMBULATORY: "ambulatory",
    EncounterClass.EMERGENCY: "emergency",
    EncounterClass.UNKNOWN: "unknown",
}

INTERPRETATION_DISPLAY: Final[dict[str, str]] = {"H": "High", "L": "Low", "N": "Normal"}

# Required-binding value sets (FHIR R4) that fhir.resources does not enforce.
REQUIRED_BINDINGS: Final[dict[tuple[str, str], frozenset[str]]] = {
    ("Patient", "gender"): frozenset({"male", "female", "other", "unknown"}),
    ("Encounter", "status"): frozenset(
        {
            "planned",
            "arrived",
            "triaged",
            "in-progress",
            "onleave",
            "finished",
            "cancelled",
            "entered-in-error",
            "unknown",
        }
    ),
    ("DocumentReference", "status"): frozenset({"current", "superseded", "entered-in-error"}),
    ("DocumentReference", "docStatus"): frozenset(
        {"preliminary", "final", "amended", "entered-in-error"}
    ),
    ("DiagnosticReport", "status"): frozenset(
        {
            "registered",
            "partial",
            "preliminary",
            "final",
            "amended",
            "corrected",
            "appended",
            "cancelled",
            "entered-in-error",
            "unknown",
        }
    ),
    ("Observation", "status"): frozenset(
        {
            "registered",
            "preliminary",
            "final",
            "amended",
            "corrected",
            "cancelled",
            "entered-in-error",
            "unknown",
        }
    ),
    ("Bundle", "type"): frozenset(
        {
            "document",
            "message",
            "transaction",
            "transaction-response",
            "batch",
            "batch-response",
            "history",
            "searchset",
            "collection",
        }
    ),
}
