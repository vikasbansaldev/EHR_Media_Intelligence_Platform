"""Task 2 orchestration: map -> validate -> store -> report."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ehr_intel.config import Settings
from ehr_intel.fhir.mapper import FhirMapper, PatientBundle
from ehr_intel.fhir.store import BundleStore
from ehr_intel.fhir.validator import BundleValidator, ValidationReport
from ehr_intel.ingestion.models import AuditAction, CleanedDataset, PatientDemographics

logger = logging.getLogger(__name__)


def identity_flags(patient: PatientDemographics) -> list[str]:
    """Clinician-facing identity warnings from the cleaning audit log."""
    flags = []
    for entry in patient.audit:
        if entry.action is AuditAction.WARNING:
            flags.append(entry.reason[0].upper() + entry.reason[1:])
        elif entry.action is AuditAction.CONFLICT:
            label = entry.field.replace("_", " ")
            flags.append(
                f"Conflicting {label} in source records: using {entry.new}, "
                f"one record says {entry.original}"
            )
    return flags


class FhirNormalizer:
    def __init__(self, settings: Settings, store: BundleStore | None = None) -> None:
        self.settings = settings
        self.mapper = FhirMapper(settings)
        self.validator = BundleValidator()
        self.store = store or BundleStore(settings.bundle_db_path)

    @property
    def bundles_dir(self) -> Path:
        return self.settings.var_dir / "bundles"

    def run(self, dataset: CleanedDataset, *, write_files: bool = True) -> ValidationReport:
        bundles = self.mapper.build_all(dataset)
        results = []
        self.store.clear()
        if write_files:
            self.bundles_dir.mkdir(parents=True, exist_ok=True)
        for pb in bundles:
            result = self.validator.validate(pb.bundle, mrn=pb.mrn)
            results.append(result)
            for issue in result.errors:
                logger.error("FHIR validation: MRN %s: %s", pb.mrn, issue)
            # Invalid bundles are still stored (flagged) so they can be inspected and fixed.
            self.store.save(pb, result, identity_flags(dataset.patients[pb.mrn]))
            if write_files:
                self._write_bundle(pb)

        report = ValidationReport(results=results)
        self.settings.reports_dir.mkdir(parents=True, exist_ok=True)
        report.write(self.settings.reports_dir)
        logger.info("FHIR normalization complete: %s", report.summary())
        return report

    def _write_bundle(self, pb: PatientBundle) -> None:
        path = self.bundles_dir / f"patient-{pb.mrn}.json"
        path.write_text(json.dumps(pb.bundle, indent=2), encoding="utf-8")
