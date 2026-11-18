"""Format-specific readers that turn source files into :class:`RawRecord` objects.

Readers only *extract*; they do not clean. Adding a new input format means
writing one class with a ``read(path)`` method and registering its suffix in
:data:`READERS`.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Protocol

from ehr_intel.ingestion.models import RawObservation, RawRecord, SourceFormat

logger = logging.getLogger(__name__)


class ReaderError(Exception):
    """A source file is unreadable or structurally invalid."""


class RecordReader(Protocol):
    def read(self, path: Path) -> Iterator[RawRecord]: ...


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class JsonExportReader:
    """Reads a simulated EHR export: ``{"records": [...]}`` or a bare list of records.

    Each record has optional ``patient``, ``encounter`` and ``document`` objects.
    """

    def read(self, path: Path) -> Iterator[RawRecord]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReaderError(f"{path}: cannot parse JSON ({exc})") from exc

        items = payload.get("records") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ReaderError(f"{path}: expected a list of records or {{'records': [...]}}")

        for index, item in enumerate(items):
            location = f"{path.name}#records[{index}]"
            if not isinstance(item, dict):
                logger.warning("%s: skipping non-object record", location)
                continue
            patient = item.get("patient") or {}
            encounter = item.get("encounter") or {}
            document = item.get("document") or {}
            yield RawRecord(
                source_id=_str_or_none(item.get("source_id")) or location,
                source_format=SourceFormat.JSON,
                source_location=location,
                mrn=_str_or_none(patient.get("mrn")),
                name=patient.get("name"),
                dob=_str_or_none(patient.get("dob")),
                sex=_str_or_none(patient.get("sex") or patient.get("gender")),
                encounter_id=_str_or_none(encounter.get("id")),
                encounter_class=_str_or_none(encounter.get("class")),
                encounter_start=_str_or_none(encounter.get("start")),
                encounter_end=_str_or_none(encounter.get("end")),
                doc_type=_str_or_none(document.get("type")),
                title=_str_or_none(document.get("title")),
                date=_str_or_none(document.get("date")),
                author=_str_or_none(document.get("author")),
                text=document.get("text"),
            )


# ---------------------------------------------------------------------------
# Lab CSV
# ---------------------------------------------------------------------------


class LabCsvReader:
    """Reads one-row-per-analyte lab results and groups them into panels.

    Rows sharing a ``panel_id`` (or, if absent, the same MRN + collection time)
    become a single :class:`RawRecord` of type ``lab`` with many observations.
    """

    REQUIRED_COLUMNS: ClassVar[set[str]] = {"test_name", "value"}

    def read(self, path: Path) -> Iterator[RawRecord]:
        try:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                columns = {c.strip().lower() for c in reader.fieldnames or []}
                missing = self.REQUIRED_COLUMNS - columns
                if missing:
                    raise ReaderError(f"{path}: missing required columns {sorted(missing)}")
                rows = [
                    {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                    for row in reader
                ]
        except OSError as exc:
            raise ReaderError(f"{path}: cannot read CSV ({exc})") from exc

        panels: dict[str, list[tuple[int, dict[str, str]]]] = {}
        for line_no, row in enumerate(rows, start=2):  # header is line 1
            key = row.get("panel_id") or f"{row.get('mrn', '')}|{row.get('collected_at', '')}"
            panels.setdefault(key, []).append((line_no, row))

        for key, members in panels.items():
            first_line, first = members[0]
            lines = ",".join(str(n) for n, _ in members)
            tests = sorted({r.get("test_name", "") for _, r in members if r.get("test_name")})
            yield RawRecord(
                source_id=f"{path.stem}:{key}",
                source_format=SourceFormat.CSV,
                source_location=f"{path.name}:lines {lines}",
                mrn=first.get("mrn") or None,
                name=first.get("patient_name") or None,
                dob=first.get("dob") or None,
                sex=first.get("sex") or None,
                encounter_id=first.get("encounter_id") or None,
                doc_type="Lab Result",
                title=f"Lab panel: {', '.join(tests)}" if tests else None,
                date=first.get("collected_at") or None,
                author=first.get("ordering_provider") or None,
                observations=[
                    RawObservation(
                        test_name=r.get("test_name") or None,
                        loinc=r.get("loinc") or None,
                        value=r.get("value") or None,
                        unit=r.get("unit") or None,
                        reference_range=r.get("reference_range") or None,
                        flag=r.get("flag") or None,
                    )
                    for _, r in members
                ],
            )
            logger.debug(
                "Grouped %d lab rows into panel %s (line %d)", len(members), key, first_line
            )


# ---------------------------------------------------------------------------
# Plain-text scanned notes
# ---------------------------------------------------------------------------

# Header aliases -> RawRecord field. Matching is case-insensitive.
_HEADER_ALIASES: dict[str, str] = {
    "patient": "name",
    "patient name": "name",
    "name": "name",
    "mrn": "mrn",
    "medical record number": "mrn",
    "dob": "dob",
    "date of birth": "dob",
    "sex": "sex",
    "gender": "sex",
    "date of service": "date",
    "date": "date",
    "service date": "date",
    "document type": "doc_type",
    "type": "doc_type",
    "title": "title",
    "encounter": "encounter_id",
    "encounter id": "encounter_id",
    "author": "author",
    "provider": "author",
}
_HEADER_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z ]{1,40}?)\s*:\s*(.*?)\s*$")
_SEPARATOR = re.compile(r"^\s*[-=_]{5,}\s*$")


class TextNoteReader:
    """Reads a scanned/OCR'd note: ``Key: value`` header lines, a separator, then the body.

    If no separator line exists, the header ends at the first line that is not
    a recognised ``Key: value`` pair.
    """

    def read(self, path: Path) -> Iterator[RawRecord]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ReaderError(f"{path}: cannot read text ({exc})") from exc

        fields: dict[str, str] = {}
        lines = content.splitlines()
        body_start = len(lines)
        for i, line in enumerate(lines):
            if _SEPARATOR.match(line):
                body_start = i + 1
                break
            match = _HEADER_LINE.match(line)
            key = _HEADER_ALIASES.get(match.group(1).lower()) if match else None
            if match and key:
                fields.setdefault(key, match.group(2))
            elif line.strip():
                body_start = i
                break

        yield RawRecord(
            source_id=path.stem,
            source_format=SourceFormat.TEXT,
            source_location=path.name,
            mrn=fields.get("mrn") or None,
            name=fields.get("name") or None,
            dob=fields.get("dob") or None,
            sex=fields.get("sex") or None,
            encounter_id=fields.get("encounter_id") or None,
            doc_type=fields.get("doc_type") or "Scanned Document",
            title=fields.get("title") or None,
            date=fields.get("date") or None,
            author=fields.get("author") or None,
            text="\n".join(lines[body_start:]),
        )


# ---------------------------------------------------------------------------
# Registry / discovery
# ---------------------------------------------------------------------------

READERS: dict[str, RecordReader] = {
    ".json": JsonExportReader(),
    ".csv": LabCsvReader(),
    ".txt": TextNoteReader(),
}


def discover_files(root: Path) -> list[Path]:
    """All files under ``root`` that have a registered reader, in stable order."""
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in READERS)


def read_file(path: Path) -> Iterator[RawRecord]:
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ReaderError(f"{path}: no reader registered for {path.suffix!r}")
    yield from reader.read(path)
