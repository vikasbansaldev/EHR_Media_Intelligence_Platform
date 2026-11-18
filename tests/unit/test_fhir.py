"""Unit tests for FHIR mapping, validation and storage."""

from __future__ import annotations

import base64
import copy
from datetime import UTC, date, datetime
from typing import Any

import pytest

from ehr_intel.config import Settings
from ehr_intel.fhir.mapper import FhirMapper, PatientBundle, urn
from ehr_intel.fhir.store import BundleStore, resource_text
from ehr_intel.fhir.validator import BundleValidator, ValidationReport
from ehr_intel.ingestion.models import (
    ClinicalRecord,
    Encounter,
    EncounterClass,
    Gender,
    LabObservation,
    PatientDemographics,
    RecordType,
    SourceFormat,
)


def _record(record_type: RecordType, **kw: Any) -> ClinicalRecord:
    base: dict[str, Any] = {
        "record_id": f"00001234-{record_type.value[:6]}",
        "mrn": "00001234",
        "record_type": record_type,
        "title": record_type.label,
        "text": "Findings: small effusion. Impression: mild CHF.",
        "effective_at": datetime(2025, 1, 10, 9, 30),
        "author": "Dr. Test",
        "encounter_id": "ENC-1",
        "source_format": SourceFormat.JSON,
        "source_ids": ["SRC-1"],
        "content_hash": "x" * 64,
    }
    base.update(kw)
    return ClinicalRecord(**base)


@pytest.fixture
def patient_bundle(settings: Settings) -> PatientBundle:
    patient = PatientDemographics(
        mrn="00001234",
        given_name="Jane",
        family_name="Doe",
        birth_date=date(1962, 3, 14),
        gender=Gender.FEMALE,
    )
    encounter = Encounter(
        encounter_id="ENC-1",
        mrn="00001234",
        encounter_class=EncounterClass.INPATIENT,
        start=datetime(2025, 1, 9),
        end=datetime(2025, 1, 12),
    )
    lab = _record(
        RecordType.LAB_RESULT,
        record_id="00001234-lab",
        title="Lab panel: BNP, Sodium",
        text="BNP: 900 pg/mL [HIGH]; Sodium: 140 mmol/L",
        observations=[
            LabObservation(
                test_name="BNP",
                loinc="30934-4",
                value=900,
                unit="pg/mL",
                reference_low=0,
                reference_high=100,
                reference_text="0-100",
                interpretation="H",
            ),
            LabObservation(
                test_name="Sodium",
                loinc="2951-2",
                value=140,
                unit="mmol/L",
                reference_low=135,
                reference_high=145,
                interpretation="N",
            ),
        ],
    )
    records = [_record(RecordType.DISCHARGE_SUMMARY), _record(RecordType.RADIOLOGY_REPORT), lab]
    return FhirMapper(settings).build_bundle(patient, [encounter], records)


def _by_type(pb: PatientBundle, rtype: str) -> list[dict[str, Any]]:
    return [r for r in pb.resources if r["resourceType"] == rtype]


# --- Mapper -------------------------------------------------------------------------


class TestMapper:
    def test_bundle_shape(self, patient_bundle: PatientBundle) -> None:
        b = patient_bundle.bundle
        assert b["resourceType"] == "Bundle"
        assert b["type"] == "collection"
        types = [r["resourceType"] for r in patient_bundle.resources]
        assert types == [
            "Patient",
            "Encounter",
            "DocumentReference",
            "DiagnosticReport",
            "DiagnosticReport",
            "Observation",
            "Observation",
        ]
        assert all(e["fullUrl"] == urn(e["resource"]["id"]) for e in b["entry"])

    def test_patient_mapping(self, patient_bundle: PatientBundle) -> None:
        (p,) = _by_type(patient_bundle, "Patient")
        assert p["gender"] == "female"
        assert p["birthDate"] == "1962-03-14"
        assert p["name"][0] == {
            "use": "official",
            "text": "Jane Doe",
            "family": "Doe",
            "given": ["Jane"],
        }
        ident = p["identifier"][0]
        assert ident["value"] == "00001234"
        assert ident["type"]["coding"][0]["code"] == "MR"

    def test_subject_and_encounter_references(self, patient_bundle: PatientBundle) -> None:
        patient_ref = urn(patient_bundle.patient_id)
        (enc,) = _by_type(patient_bundle, "Encounter")
        enc_ref = urn(enc["id"])
        (doc,) = _by_type(patient_bundle, "DocumentReference")
        assert doc["subject"]["reference"] == patient_ref
        assert doc["context"]["encounter"][0]["reference"] == enc_ref
        for report in _by_type(patient_bundle, "DiagnosticReport"):
            assert report["subject"]["reference"] == patient_ref
            assert report["encounter"]["reference"] == enc_ref

    def test_document_content_round_trips(self, patient_bundle: PatientBundle) -> None:
        (doc,) = _by_type(patient_bundle, "DocumentReference")
        data = doc["content"][0]["attachment"]["data"]
        assert base64.b64decode(data).decode() == "Findings: small effusion. Impression: mild CHF."
        assert doc["date"] == "2025-01-10T09:30:00+00:00"  # instant carries an offset
        assert doc["type"]["coding"][0]["code"] == "18842-5"

    def test_radiology_conclusion_uses_impression(self, patient_bundle: PatientBundle) -> None:
        rad = next(
            r
            for r in _by_type(patient_bundle, "DiagnosticReport")
            if r["category"][0]["coding"][0]["code"] == "RAD"
        )
        assert rad["conclusion"] == "mild CHF."

    def test_lab_report_links_observations(self, patient_bundle: PatientBundle) -> None:
        lab = next(
            r
            for r in _by_type(patient_bundle, "DiagnosticReport")
            if r["category"][0]["coding"][0]["code"] == "LAB"
        )
        obs_refs = {urn(o["id"]) for o in _by_type(patient_bundle, "Observation")}
        assert {r["reference"] for r in lab["result"]} == obs_refs
        assert lab["conclusion"].startswith("Abnormal: BNP")
        bnp = next(o for o in _by_type(patient_bundle, "Observation") if o["code"]["text"] == "BNP")
        assert bnp["valueQuantity"] == {
            "value": 900,
            "unit": "pg/mL",
            "system": "http://unitsofmeasure.org",
            "code": "pg/mL",
        }
        assert bnp["interpretation"][0]["coding"][0]["code"] == "H"
        assert bnp["referenceRange"][0]["high"]["value"] == 100

    def test_ids_are_deterministic(self, settings: Settings, patient_bundle: PatientBundle) -> None:
        patient = PatientDemographics(
            mrn="00001234",
            given_name="Jane",
            family_name="Doe",
            birth_date=date(1962, 3, 14),
            gender=Gender.FEMALE,
        )
        again = FhirMapper(settings).build_bundle(patient, [], [])
        assert again.patient_id == patient_bundle.patient_id
        assert again.bundle["id"] == patient_bundle.bundle["id"]

    def test_record_without_encounter_or_date_still_maps(self, settings: Settings) -> None:
        patient = PatientDemographics(mrn="00000001")
        record = _record(
            RecordType.PROGRESS_NOTE,
            mrn="00000001",
            encounter_id=None,
            effective_at=None,
            author=None,
        )
        pb = FhirMapper(settings).build_bundle(patient, [], [record])
        (doc,) = _by_type(pb, "DocumentReference")
        assert "context" not in doc and "date" not in doc
        assert BundleValidator().validate(pb.bundle).is_valid


# --- Validator ------------------------------------------------------------------------


class TestValidator:
    def test_mapped_bundle_is_valid(self, patient_bundle: PatientBundle) -> None:
        result = BundleValidator().validate(patient_bundle.bundle, mrn="00001234")
        assert result.is_valid, [str(i) for i in result.issues]
        assert result.resource_counts["Observation"] == 2

    def _corrupt(self, pb: PatientBundle) -> dict[str, Any]:
        return copy.deepcopy(pb.bundle)

    def test_schema_error_is_located(self, patient_bundle: PatientBundle) -> None:
        bundle = self._corrupt(patient_bundle)
        bundle["entry"][0]["resource"]["birthDate"] = "1962-13-45"
        (issue,) = BundleValidator().validate(bundle).errors
        assert (issue.layer, issue.resource_type, issue.path) == ("schema", "Patient", "birthDate")

    def test_missing_required_element(self, patient_bundle: PatientBundle) -> None:
        bundle = self._corrupt(patient_bundle)
        del bundle["entry"][2]["resource"]["content"]
        errors = BundleValidator().validate(bundle).errors
        assert any(e.resource_type == "DocumentReference" and e.path == "content" for e in errors)

    def test_required_binding_enforced(self, patient_bundle: PatientBundle) -> None:
        bundle = self._corrupt(patient_bundle)
        bundle["entry"][0]["resource"]["gender"] = "M"
        (issue,) = BundleValidator().validate(bundle).errors
        assert issue.layer == "rules" and issue.path == "gender"
        assert "required value set" in issue.message

    def test_unresolved_reference(self, patient_bundle: PatientBundle) -> None:
        bundle = self._corrupt(patient_bundle)
        bundle["entry"][2]["resource"]["subject"]["reference"] = "urn:uuid:does-not-exist"
        (issue,) = BundleValidator().validate(bundle).errors
        assert "unresolved reference" in issue.message

    def test_subject_must_point_at_patient(self, patient_bundle: PatientBundle) -> None:
        bundle = self._corrupt(patient_bundle)
        encounter_url = bundle["entry"][1]["fullUrl"]
        bundle["entry"][2]["resource"]["subject"]["reference"] = encounter_url
        (issue,) = BundleValidator().validate(bundle).errors
        assert issue.message == "subject must reference Patient, found Encounter"

    def test_encounter_must_point_at_encounter(self, patient_bundle: PatientBundle) -> None:
        bundle = self._corrupt(patient_bundle)
        patient_url = bundle["entry"][0]["fullUrl"]
        bundle["entry"][2]["resource"]["context"]["encounter"][0]["reference"] = patient_url
        (issue,) = BundleValidator().validate(bundle).errors
        assert issue.path == "context.encounter[0]"

    def test_duplicate_full_url(self, patient_bundle: PatientBundle) -> None:
        bundle = self._corrupt(patient_bundle)
        bundle["entry"].append(copy.deepcopy(bundle["entry"][0]))
        assert any(
            "duplicate fullUrl" in e.message for e in BundleValidator().validate(bundle).errors
        )

    def test_unknown_resource_type(self, patient_bundle: PatientBundle) -> None:
        bundle = self._corrupt(patient_bundle)
        bundle["entry"].append({"fullUrl": "urn:uuid:x", "resource": {"resourceType": "Bogus"}})
        assert any(
            "unknown FHIR resource type" in e.message
            for e in BundleValidator().validate(bundle).errors
        )

    def test_report_summarizes_and_renders(self, patient_bundle: PatientBundle) -> None:
        good = BundleValidator().validate(patient_bundle.bundle, mrn="00001234")
        broken = self._corrupt(patient_bundle)
        broken["entry"][0]["resource"]["gender"] = "M"
        bad = BundleValidator().validate(broken, mrn="00009999")
        report = ValidationReport(results=[good, bad])
        assert report.summary()["invalid_bundles"] == 1
        md = report.to_markdown()
        assert "**Status:** FAIL" in md
        assert "MRN 00009999" in md and "required value set" in md


# --- Store ----------------------------------------------------------------------------


class TestStore:
    def test_save_and_retrieve(self, settings: Settings, patient_bundle: PatientBundle) -> None:
        store = BundleStore(settings.bundle_db_path)
        store.save(patient_bundle, BundleValidator().validate(patient_bundle.bundle))

        assert store.get_bundle("00001234") == patient_bundle.bundle
        (row,) = store.patients()
        assert (row.name, row.gender, row.is_valid) == ("Jane Doe", "female", True)
        docs = store.resources(mrn="00001234", types=("DocumentReference", "DiagnosticReport"))
        assert len(docs) == 3
        assert all(d.text for d in docs)
        assert store.resource(docs[0].resource_id) == docs[0]
        assert store.get_bundle("nope") is None

    def test_save_is_idempotent(self, settings: Settings, patient_bundle: PatientBundle) -> None:
        store = BundleStore(settings.bundle_db_path)
        result = BundleValidator().validate(patient_bundle.bundle)
        store.save(patient_bundle, result)
        store.save(patient_bundle, result)
        assert store.count() == 1
        assert len(store.resources()) == len(patient_bundle.resources)

    def test_resource_text_extraction(self, patient_bundle: PatientBundle) -> None:
        (doc,) = _by_type(patient_bundle, "DocumentReference")
        date_, title, text = resource_text(doc)
        assert date_ == "2025-01-10T09:30:00+00:00"
        assert title == "Discharge Summary"
        assert text is not None and "Impression" in text


class TestTimezones:
    """Windows has no system tz database: UTC must work without one (regression test)."""

    def test_utc_needs_no_tz_database(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from zoneinfo import ZoneInfoNotFoundError

        from ehr_intel.fhir import mapper as mapper_module

        def no_tz_database(key: str) -> None:
            raise ZoneInfoNotFoundError(f"No time zone found with key {key}")

        monkeypatch.setattr(mapper_module, "ZoneInfo", no_tz_database)
        for name in ("UTC", "utc", "Etc/UTC", "Z"):
            assert mapper_module.resolve_timezone(name) is UTC
        mapped = FhirMapper(settings).build_bundle(
            PatientDemographics(mrn="00000001"),
            [],
            [_record(RecordType.PROGRESS_NOTE, mrn="00000001", encounter_id=None)],
        )
        doc = next(r for r in mapped.resources if r["resourceType"] == "DocumentReference")
        assert doc["date"].endswith("+00:00")

    def test_missing_zone_gives_actionable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from zoneinfo import ZoneInfoNotFoundError

        from ehr_intel.fhir import mapper as mapper_module

        def no_tz_database(key: str) -> None:
            raise ZoneInfoNotFoundError(key)

        monkeypatch.setattr(mapper_module, "ZoneInfo", no_tz_database)
        with pytest.raises(ValueError, match="pip install tzdata"):
            mapper_module.resolve_timezone("America/New_York")

    def test_named_zone_offsets_applied(self, settings: Settings) -> None:
        try:
            ny = settings.model_copy(update={"source_timezone": "America/New_York"})
            mapper = FhirMapper(ny)
        except ValueError:
            pytest.skip("no tz database available on this machine")
        assert mapper._instant(datetime(2025, 1, 10, 9, 30)) == "2025-01-10T09:30:00-05:00"
