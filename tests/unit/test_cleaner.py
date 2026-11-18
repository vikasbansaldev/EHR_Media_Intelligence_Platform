"""Cleaning edge cases (Task 1 requirement: at least three; this file covers twelve)."""

from __future__ import annotations

from datetime import date, datetime

from ehr_intel.config import Settings
from ehr_intel.ingestion.cleaner import RecordCleaner
from ehr_intel.ingestion.models import AuditAction, CleanedDataset, Gender, RecordType, SourceFormat
from tests.conftest import RawFactory


def _clean(settings: Settings, *raws: object) -> CleanedDataset:
    return RecordCleaner(settings).clean(raws)  # type: ignore[arg-type]


def _actions(dataset: CleanedDataset) -> list[AuditAction]:
    return [AuditAction(row["action"]) for row in dataset.audit_log()]


# --- Duplicates -------------------------------------------------------------------


def test_exact_duplicate_records_are_merged(settings: Settings, make_raw: RawFactory) -> None:
    a = make_raw(text="Same body.")
    b = make_raw(text="Same body.")
    ds = _clean(settings, a, b)

    assert len(ds.records) == 1
    record = ds.records[0]
    assert record.source_ids == [a.source_id, b.source_id]
    assert any(
        e.action is AuditAction.DEDUPLICATED and e.source_id == b.source_id for e in record.audit
    )


def test_cross_format_duplicate_detected_despite_ocr_whitespace(
    settings: Settings, make_raw: RawFactory
) -> None:
    json_rec = make_raw(text="Findings: normal. Impression: no acute disease.", author=None)
    scanned = make_raw(
        source_format=SourceFormat.TEXT,
        source_location="scan.txt",
        mrn=" 1234 ",
        date="01/10/2025 09:30",
        text="Findings: normal.\n   Impression: no acute\n disease.",
        author="Dr. Scan",
    )
    ds = _clean(settings, json_rec, scanned)

    assert len(ds.records) == 1
    record = ds.records[0]
    assert "cross-format" in record.audit[-2].reason
    # Gap on the kept record is filled from the duplicate, and audited.
    assert record.author == "Dr. Scan"
    assert record.audit[-1].action is AuditAction.INFERRED


def test_same_text_on_different_days_is_not_a_duplicate(
    settings: Settings, make_raw: RawFactory
) -> None:
    ds = _clean(
        settings,
        make_raw(text="Stable.", date="2025-01-01"),
        make_raw(text="Stable.", date="2025-02-01"),
    )
    assert len(ds.records) == 2


# --- Missing fields -----------------------------------------------------------------


def test_missing_document_date_inferred_from_encounter(
    settings: Settings, make_raw: RawFactory
) -> None:
    ds = _clean(settings, make_raw(date=None, encounter_start="2025-03-01"))
    record = ds.records[0]
    assert record.effective_at == datetime(2025, 3, 1)
    assert any(e.field == "effective_at" and e.action is AuditAction.INFERRED for e in record.audit)


def test_missing_demographics_resolved_from_other_records(
    settings: Settings, make_raw: RawFactory
) -> None:
    ds = _clean(settings, make_raw(sex=None, dob=""), make_raw(), make_raw())
    patient = ds.patients["00001234"]
    assert patient.gender is Gender.FEMALE
    assert patient.birth_date == date(1962, 3, 14)
    inferred = [e for r in ds.records for e in r.audit if e.action is AuditAction.INFERRED]
    assert {e.field for e in inferred} == {"gender", "birth_date"}


def test_missing_title_and_gender_defaulted(settings: Settings, make_raw: RawFactory) -> None:
    ds = _clean(settings, make_raw(title=None, sex=None, doc_type="Progress Note"))
    assert ds.records[0].title == "Progress Note"
    assert ds.patients["00001234"].gender is Gender.UNKNOWN
    assert ds.patients["00001234"].audit[0].action is AuditAction.DEFAULTED


def test_record_without_text_is_rejected(settings: Settings, make_raw: RawFactory) -> None:
    ds = _clean(settings, make_raw(text="   \n  "), make_raw())
    assert len(ds.records) == 1
    assert len(ds.rejected) == 1
    assert "no clinical text" in ds.rejected[0].reason


# --- Inconsistent formats ---------------------------------------------------------------


def test_mixed_formats_normalize_to_one_patient(settings: Settings, make_raw: RawFactory) -> None:
    ds = _clean(
        settings,
        make_raw(mrn="MRN-00001234", name="DOE, JANE", dob="03/14/1962", sex="F"),
        make_raw(
            mrn="mrn: 1234", name={"given": "Jane", "family": "Doe"}, dob="14-Mar-1962", sex="2"
        ),
        make_raw(mrn="1234", name="Jane Doe", dob="19620314", sex="female"),
    )
    assert list(ds.patients) == ["00001234"]
    patient = ds.patients["00001234"]
    assert (patient.given_name, patient.family_name) == ("Jane", "Doe")
    assert patient.birth_date == date(1962, 3, 14)
    assert AuditAction.CONFLICT not in _actions(ds)


def test_future_birth_date_discarded_and_audited(settings: Settings, make_raw: RawFactory) -> None:
    ds = _clean(settings, make_raw(dob="2091-01-01"), make_raw())
    assert ds.patients["00001234"].birth_date == date(1962, 3, 14)
    bad = next(r for r in ds.records if any(e.action is AuditAction.INVALID for e in r.audit))
    assert "future" in next(e.reason for e in bad.audit if e.action is AuditAction.INVALID)


# --- Conflicting identifiers ----------------------------------------------------------


def test_conflicting_dob_for_same_mrn_uses_majority(
    settings: Settings, make_raw: RawFactory
) -> None:
    ds = _clean(settings, make_raw(), make_raw(), make_raw(dob="1994-03-14"))
    patient = ds.patients["00001234"]
    assert patient.birth_date == date(1962, 3, 14)
    conflict = next(e for e in patient.audit if e.action is AuditAction.CONFLICT)
    assert conflict.original == "1994-03-14"
    assert conflict.new == "1962-03-14"


def test_record_without_mrn_linked_by_name_and_dob(
    settings: Settings, make_raw: RawFactory
) -> None:
    orphan = make_raw(mrn=None, name="Jane Doe")
    unmatched = make_raw(mrn=None, name="Nobody Known", dob="01/01/1990")
    no_identity = make_raw(mrn=None, dob=None)
    ds = _clean(settings, make_raw(), orphan, unmatched, no_identity)

    linked = next(r for r in ds.records if orphan.source_id in r.source_ids)
    assert linked.mrn == "00001234"
    assert any(e.action is AuditAction.LINKED for e in linked.audit)
    assert {r.source_id for r in ds.rejected} == {unmatched.source_id, no_identity.source_id}


def test_same_person_under_two_mrns_flagged_not_merged(
    settings: Settings, make_raw: RawFactory
) -> None:
    ds = _clean(settings, make_raw(mrn="1234"), make_raw(mrn="9999"))
    assert set(ds.patients) == {"00001234", "00009999"}
    for patient in ds.patients.values():
        assert any(
            e.action is AuditAction.WARNING and "possible duplicate" in e.reason
            for e in patient.audit
        )


# --- Lab results ------------------------------------------------------------------------


def test_lab_panel_deduplicates_rows_and_trusts_reference_range(
    settings: Settings, make_raw: RawFactory
) -> None:
    obs = {
        "test_name": "TSH",
        "loinc": "3016-3",
        "value": "1.2",
        "unit": "mIU/L",
        "reference_range": "0.4-4.0",
        "flag": "H",
    }
    ds = _clean(settings, make_raw(doc_type="Lab Result", text=None, observations=[obs, dict(obs)]))
    record = ds.records[0]
    assert record.record_type is RecordType.LAB_RESULT
    assert len(record.observations) == 1
    assert record.observations[0].interpretation == "N"
    actions = {e.action for e in record.audit}
    assert {AuditAction.DEDUPLICATED, AuditAction.CONFLICT} <= actions
    assert "TSH: 1.2 mIU/L (ref 0.4-4.0)" in record.text
