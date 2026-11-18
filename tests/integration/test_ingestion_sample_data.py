"""Run ingestion over the bundled synthetic dataset and check every injected defect is handled."""

from __future__ import annotations

import json

import pytest

from ehr_intel.config import Settings
from ehr_intel.ingestion import AuditAction, CleanedDataset, IngestionPipeline

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def dataset() -> CleanedDataset:
    from tests.conftest import SAMPLE_DATA

    return IngestionPipeline(Settings(_env_file=None)).run(SAMPLE_DATA)  # type: ignore[call-arg]


def test_counts(dataset: CleanedDataset) -> None:
    stats = dataset.stats()
    assert stats["raw_records"] > stats["clean_records"] >= 50
    assert stats["patients"] == 13  # 12 people + 1 duplicate registration (flagged)
    assert set(stats["records_by_type"]) >= {
        "discharge_summary",
        "lab_result",
        "radiology_report",
        "progress_note",
    }


def test_every_injected_defect_is_audited(dataset: CleanedDataset) -> None:
    actions = {AuditAction(row["action"]) for row in dataset.audit_log()}
    assert {
        AuditAction.DEDUPLICATED,
        AuditAction.CONFLICT,
        AuditAction.LINKED,
        AuditAction.INFERRED,
        AuditAction.INVALID,
        AuditAction.WARNING,
        AuditAction.DEFAULTED,
    } <= actions
    assert [r.source_id for r in dataset.rejected] == ["EPIC-EMPTY-1"]


def test_cross_format_duplicate_merged(dataset: CleanedDataset) -> None:
    merged = [r for r in dataset.records if any("cross-format" in a.reason for a in r.audit)]
    assert len(merged) == 1
    assert merged[0].source_ids[0] == "EPIC-00001"


def test_records_reference_known_patients_and_encounters(dataset: CleanedDataset) -> None:
    for record in dataset.records:
        assert record.mrn in dataset.patients
        if record.encounter_id:
            assert dataset.encounters[record.encounter_id].mrn == record.mrn


def test_reports_written(dataset: CleanedDataset, settings: Settings) -> None:
    out = IngestionPipeline(settings).write_reports(dataset)
    report = json.loads((out / "ingestion_report.json").read_text())
    assert report["stats"]["clean_records"] == len(dataset.records)
    assert report["audit_log"]
    assert (out / "cleaned_dataset.json").exists()
