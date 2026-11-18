"""Map cleaned records to HL7 FHIR R4 resources and assemble one Bundle per patient.

Mapping summary:

=====================  ==========================================================
Canonical model        FHIR resource(s)
=====================  ==========================================================
PatientDemographics    ``Patient`` (MRN identifier typed ``MR``)
Encounter              ``Encounter`` (class from v3 ActCode)
Narrative records      ``DocumentReference`` (LOINC type, text/plain attachment)
Radiology report       ``DiagnosticReport`` (RAD, conclusion + presentedForm)
Lab panel              ``DiagnosticReport`` (LAB) + one ``Observation`` per analyte
=====================  ==========================================================

Resources get deterministic UUIDv5 ids, so re-running the pipeline produces
byte-identical Bundles. Inside a Bundle every reference uses the target entry's
``urn:uuid:`` fullUrl, which FHIR servers resolve on import.

The mapper builds plain JSON-ready dicts and never raises on odd data; the
:mod:`ehr_intel.fhir.validator` module is the single place that judges validity.
"""

from __future__ import annotations

import base64
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ehr_intel.config import Settings
from ehr_intel.fhir import terminology as term
from ehr_intel.ingestion.models import (
    CleanedDataset,
    ClinicalRecord,
    Encounter,
    LabObservation,
    PatientDemographics,
    RecordType,
)

Resource = dict[str, Any]

# Stable namespace so ids survive re-runs (and differ from any other system's uuid5s).
_NAMESPACE = uuid.UUID("6f1c1e1a-8d2b-4a57-9f3e-2b1d7c0e4a10")
_IMPRESSION = re.compile(r"impression:\s*(.+)$", re.IGNORECASE)


def resource_uuid(kind: str, key: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{kind}/{key}"))


def urn(resource_id: str) -> str:
    return f"urn:uuid:{resource_id}"


@dataclass
class PatientBundle:
    """A patient's Bundle plus lookup metadata used by the store and search index."""

    mrn: str
    patient_id: str
    bundle: Resource
    # record_id -> FHIR resource id of the primary resource for that record
    record_resource_ids: dict[str, str] = field(default_factory=dict)

    @property
    def resources(self) -> list[Resource]:
        return [entry["resource"] for entry in self.bundle.get("entry", [])]


_UTC_NAMES = frozenset({"utc", "z", "etc/utc", "gmt", "etc/gmt"})


def resolve_timezone(name: str) -> tzinfo:
    """UTC needs no tz database; other IANA zones need one (``tzdata`` on Windows)."""
    if name.strip().lower() in _UTC_NAMES:
        return UTC
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Unknown time zone {name!r} for EHR_SOURCE_TIMEZONE. Use an IANA name such as "
            "'America/New_York'. On Windows the zone database comes from the 'tzdata' "
            "package: pip install tzdata"
        ) from exc


class FhirMapper:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tz = resolve_timezone(settings.source_timezone)

    # ------------------------------------------------------------- helpers --

    def _instant(self, value: datetime) -> str:
        """FHIR ``instant``/``dateTime`` with an explicit offset (required when a time is given)."""
        aware = value if value.tzinfo else value.replace(tzinfo=self._tz)
        return aware.isoformat(timespec="seconds")

    @staticmethod
    def _coding(system: str, code: str, display: str | None = None) -> dict[str, str]:
        coding = {"system": system, "code": code}
        if display:
            coding["display"] = display
        return coding

    def _concept(self, system: str, code: str, display: str, text: str | None = None) -> Resource:
        return {"coding": [self._coding(system, code, display)], "text": text or display}

    @staticmethod
    def _attachment(text: str, title: str, created: str | None) -> Resource:
        attachment: Resource = {
            "contentType": "text/plain; charset=utf-8",
            "language": "en",
            "data": base64.b64encode(text.encode("utf-8")).decode("ascii"),
            "title": title,
        }
        if created:
            attachment["creation"] = created
        return attachment

    # ----------------------------------------------------------- resources --

    def map_patient(self, patient: PatientDemographics) -> Resource:
        resource: Resource = {
            "resourceType": "Patient",
            "id": resource_uuid("Patient", patient.mrn),
            "identifier": [
                {
                    "use": "usual",
                    "type": self._concept(term.V2_0203, "MR", "Medical record number"),
                    "system": self.settings.mrn_system,
                    "value": patient.mrn,
                }
            ],
            "active": True,
            "gender": patient.gender.value,
        }
        if patient.family_name or patient.given_name:
            name: Resource = {"use": "official", "text": patient.display_name}
            if patient.family_name:
                name["family"] = patient.family_name
            if patient.given_name:
                name["given"] = patient.given_name.split()
            resource["name"] = [name]
        if patient.birth_date:
            resource["birthDate"] = patient.birth_date.isoformat()
        return resource

    def map_encounter(self, encounter: Encounter, patient_ref: str) -> Resource:
        cls = encounter.encounter_class
        system = term.V3_NULL_FLAVOR if cls.value == "UNK" else term.V3_ACT_CODE
        resource: Resource = {
            "resourceType": "Encounter",
            "id": resource_uuid("Encounter", encounter.encounter_id),
            "identifier": [
                {"system": "urn:ehr-intel:encounter-id", "value": encounter.encounter_id}
            ],
            "status": "finished",
            "class": self._coding(system, cls.value, term.ENCOUNTER_CLASS_DISPLAY[cls]),
            "subject": {"reference": patient_ref},
        }
        period = {
            k: self._instant(v)
            for k, v in (("start", encounter.start), ("end", encounter.end))
            if v
        }
        if period:
            resource["period"] = period
        return resource

    def _common_links(
        self, record: ClinicalRecord, patient_ref: str, encounter_refs: dict[str, str]
    ) -> tuple[Resource, Resource | None]:
        subject = {"reference": patient_ref}
        enc_ref = encounter_refs.get(record.encounter_id or "")
        return subject, ({"reference": enc_ref} if enc_ref else None)

    def map_document_reference(
        self, record: ClinicalRecord, patient_ref: str, encounter_refs: dict[str, str]
    ) -> Resource:
        code, display = term.RECORD_TYPE_CODES[record.record_type]
        subject, encounter = self._common_links(record, patient_ref, encounter_refs)
        when = self._instant(record.effective_at) if record.effective_at else None
        resource: Resource = {
            "resourceType": "DocumentReference",
            "id": resource_uuid("DocumentReference", record.record_id),
            "identifier": [
                {"system": "urn:ehr-intel:source-id", "value": sid} for sid in record.source_ids
            ],
            "status": "current",
            "docStatus": "final",
            "type": self._concept(term.LOINC, code, display, record.record_type.label),
            "category": [
                self._concept(term.US_CORE_DOC_CATEGORY, "clinical-note", "Clinical Note")
            ],
            "subject": subject,
            "description": record.title,
            "content": [{"attachment": self._attachment(record.text, record.title, when)}],
        }
        if when:
            resource["date"] = when
        if record.author:
            resource["author"] = [{"display": record.author}]
        context: Resource = {}
        if encounter:
            context["encounter"] = [encounter]
        if when:
            context["period"] = {"start": when}
        if context:
            resource["context"] = context
        return resource

    def map_observation(
        self,
        obs: LabObservation,
        record: ClinicalRecord,
        index: int,
        patient_ref: str,
        encounter: Resource | None,
    ) -> Resource:
        code: Resource = {"text": obs.test_name}
        if obs.loinc:
            code["coding"] = [self._coding(term.LOINC, obs.loinc, obs.test_name)]
        resource: Resource = {
            "resourceType": "Observation",
            "id": resource_uuid("Observation", f"{record.record_id}/{index}"),
            "status": "final",
            "category": [self._concept(term.OBSERVATION_CATEGORY, "laboratory", "Laboratory")],
            "code": code,
            "subject": {"reference": patient_ref},
        }
        if encounter:
            resource["encounter"] = encounter
        if record.effective_at:
            resource["effectiveDateTime"] = self._instant(record.effective_at)
        if obs.value is not None:
            quantity: Resource = {"value": obs.value}
            if obs.unit:
                quantity.update(unit=obs.unit, system=term.UCUM, code=obs.unit)
            resource["valueQuantity"] = quantity
        elif obs.value_text:
            resource["valueString"] = obs.value_text
        if obs.reference_low is not None or obs.reference_high is not None or obs.reference_text:
            ref_range: Resource = {}
            for key, bound in (("low", obs.reference_low), ("high", obs.reference_high)):
                if bound is not None:
                    ref_range[key] = {
                        "value": bound,
                        **(
                            {"unit": obs.unit, "system": term.UCUM, "code": obs.unit}
                            if obs.unit
                            else {}
                        ),
                    }
            if obs.reference_text:
                ref_range["text"] = obs.reference_text
            resource["referenceRange"] = [ref_range]
        if obs.interpretation:
            resource["interpretation"] = [
                self._concept(
                    term.V3_INTERPRETATION,
                    obs.interpretation,
                    term.INTERPRETATION_DISPLAY[obs.interpretation],
                )
            ]
        return resource

    def map_diagnostic_report(
        self, record: ClinicalRecord, patient_ref: str, encounter_refs: dict[str, str]
    ) -> list[Resource]:
        """Return the DiagnosticReport first, followed by any Observations it references."""
        code, display = term.RECORD_TYPE_CODES[record.record_type]
        section, section_display = term.DIAGNOSTIC_SECTIONS[record.record_type]
        subject, encounter = self._common_links(record, patient_ref, encounter_refs)
        when = self._instant(record.effective_at) if record.effective_at else None

        observations = [
            self.map_observation(obs, record, i, patient_ref, encounter)
            for i, obs in enumerate(record.observations)
        ]
        report: Resource = {
            "resourceType": "DiagnosticReport",
            "id": resource_uuid("DiagnosticReport", record.record_id),
            "identifier": [
                {"system": "urn:ehr-intel:source-id", "value": sid} for sid in record.source_ids
            ],
            "status": "final",
            "category": [self._concept(term.V2_0074, section, section_display)],
            "code": self._concept(term.LOINC, code, display, record.title),
            "subject": subject,
            "conclusion": self._conclusion(record),
            "presentedForm": [self._attachment(record.text, record.title, when)],
        }
        if encounter:
            report["encounter"] = encounter
        if when:
            report["effectiveDateTime"] = when
            report["issued"] = when
        if record.author:
            report["performer"] = [{"display": record.author}]
        if observations:
            report["result"] = [
                {"reference": urn(o["id"]), "display": o["code"]["text"]} for o in observations
            ]
        return [report, *observations]

    @staticmethod
    def _conclusion(record: ClinicalRecord) -> str:
        if record.record_type is RecordType.LAB_RESULT:
            abnormal = [o.describe() for o in record.observations if o.is_abnormal]
            return ("Abnormal: " + "; ".join(abnormal)) if abnormal else "All results within range."
        match = _IMPRESSION.search(record.text)
        return match.group(1).strip() if match else record.text[:500]

    # --------------------------------------------------------------- bundle --

    def build_bundle(
        self,
        patient: PatientDemographics,
        encounters: list[Encounter],
        records: list[ClinicalRecord],
    ) -> PatientBundle:
        patient_res = self.map_patient(patient)
        patient_ref = urn(patient_res["id"])
        resources: list[Resource] = [patient_res]

        encounter_refs: dict[str, str] = {}
        for enc in sorted(encounters, key=lambda e: e.encounter_id):
            enc_res = self.map_encounter(enc, patient_ref)
            encounter_refs[enc.encounter_id] = urn(enc_res["id"])
            resources.append(enc_res)

        record_ids: dict[str, str] = {}
        for record in records:
            if record.record_type.is_diagnostic:
                mapped = self.map_diagnostic_report(record, patient_ref, encounter_refs)
            else:
                mapped = [self.map_document_reference(record, patient_ref, encounter_refs)]
            record_ids[record.record_id] = mapped[0]["id"]
            resources.extend(mapped)

        bundle: Resource = {
            "resourceType": "Bundle",
            "id": resource_uuid("Bundle", patient.mrn),
            "meta": {
                "tag": [
                    {
                        "system": "urn:ehr-intel",
                        "code": "synthetic",
                        "display": "Generated by EHR Media Intelligence pipeline",
                    }
                ]
            },
            "identifier": {"system": self.settings.mrn_system, "value": patient.mrn},
            "type": "collection",
            "entry": [{"fullUrl": urn(r["id"]), "resource": r} for r in resources],
        }
        if records and (
            latest := max((r.effective_at for r in records if r.effective_at), default=None)
        ):
            bundle["timestamp"] = self._instant(latest)
        return PatientBundle(
            mrn=patient.mrn,
            patient_id=patient_res["id"],
            bundle=bundle,
            record_resource_ids=record_ids,
        )

    def build_all(self, dataset: CleanedDataset) -> list[PatientBundle]:
        return [
            self.build_bundle(patient, dataset.encounters_for(mrn), dataset.records_for(mrn))
            for mrn, patient in sorted(dataset.patients.items())
        ]
