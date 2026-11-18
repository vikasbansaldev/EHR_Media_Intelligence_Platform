"""Ingest the sample dataset, normalize to FHIR, and check bundles, store and report."""

from __future__ import annotations

import json

import pytest

from ehr_intel.config import Settings
from ehr_intel.fhir import FhirNormalizer
from ehr_intel.ingestion import IngestionPipeline

pytestmark = pytest.mark.integration


def test_sample_bundles_all_valid_and_persisted(settings: Settings) -> None:
    dataset = IngestionPipeline(settings).run()
    normalizer = FhirNormalizer(settings)
    report = normalizer.run(dataset)

    summary = report.summary()
    assert summary["bundles"] == len(dataset.patients)
    assert summary["invalid_bundles"] == 0, normalizer.store and report.to_markdown()
    assert {"Patient", "Encounter", "DocumentReference", "DiagnosticReport", "Observation"} <= set(
        summary["resources"]
    )

    # One JSON bundle file per patient, identical to the stored copy.
    files = sorted(normalizer.bundles_dir.glob("patient-*.json"))
    assert len(files) == len(dataset.patients)
    first = json.loads(files[0].read_text())
    assert normalizer.store.get_bundle(first["identifier"]["value"]) == first

    # Every clinical record became exactly one primary resource in the store.
    records = normalizer.store.resources(types=("DocumentReference", "DiagnosticReport"))
    assert len(records) == len(dataset.records)

    assert (settings.reports_dir / "fhir_validation_report.md").read_text().count("valid |") == (
        len(dataset.patients)
    )


def test_rerun_is_deterministic(settings: Settings) -> None:
    dataset = IngestionPipeline(settings).run()
    normalizer = FhirNormalizer(settings)
    normalizer.run(dataset, write_files=False)
    first = {p.mrn: p.bundle_hash for p in normalizer.store.patients()}
    normalizer.run(dataset, write_files=False)
    assert {p.mrn: p.bundle_hash for p in normalizer.store.patients()} == first
