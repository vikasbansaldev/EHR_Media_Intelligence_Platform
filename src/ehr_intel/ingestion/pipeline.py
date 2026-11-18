"""Orchestrates reading every source file and cleaning the combined record stream."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from ehr_intel.config import Settings
from ehr_intel.ingestion.cleaner import RecordCleaner
from ehr_intel.ingestion.models import CleanedDataset, RawRecord, RejectedRecord
from ehr_intel.ingestion.readers import ReaderError, discover_files, read_file

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Read -> clean -> (optionally) write reports.

    Example::

        dataset = IngestionPipeline(settings).run(Path("data/raw"))
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cleaner = RecordCleaner(settings)
        self._file_errors: list[RejectedRecord] = []

    def read_all(self, source: Path) -> Iterator[RawRecord]:
        files = discover_files(source)
        if not files:
            logger.warning("No ingestible files found under %s", source)
        for path in files:
            try:
                records = list(read_file(path))
            except ReaderError as exc:
                logger.error("Skipping unreadable file: %s", exc)
                self._file_errors.append(
                    RejectedRecord(source_id=path.name, source_location=str(path), reason=str(exc))
                )
                continue
            logger.info("Read %d record(s) from %s", len(records), path.name)
            yield from records

    def run(self, source: Path | None = None) -> CleanedDataset:
        self._file_errors = []
        dataset = self.cleaner.clean(self.read_all(source or self.settings.raw_data_dir))
        dataset.rejected = self._file_errors + dataset.rejected
        return dataset

    def write_reports(self, dataset: CleanedDataset, out_dir: Path | None = None) -> Path:
        """Persist the cleaned dataset and its audit log as JSON; return the report directory."""
        out_dir = out_dir or self.settings.reports_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cleaned_dataset.json").write_text(
            dataset.model_dump_json(indent=2), encoding="utf-8"
        )
        report = {
            "stats": dataset.stats(),
            "rejected": [r.model_dump() for r in dataset.rejected],
            "audit_log": dataset.audit_log(),
        }
        (out_dir / "ingestion_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        return out_dir
