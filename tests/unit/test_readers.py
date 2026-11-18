"""Unit tests for source-format readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ehr_intel.ingestion.models import SourceFormat
from ehr_intel.ingestion.readers import (
    JsonExportReader,
    LabCsvReader,
    ReaderError,
    TextNoteReader,
    discover_files,
)


def test_json_reader_accepts_wrapped_and_bare_lists(tmp_path: Path) -> None:
    record = {
        "source_id": "A1",
        "patient": {"mrn": "1", "name": "A B", "sex": "F"},
        "document": {"type": "Clinical Note", "text": "hello"},
    }
    wrapped = tmp_path / "a.json"
    wrapped.write_text(json.dumps({"records": [record]}))
    bare = tmp_path / "b.json"
    bare.write_text(json.dumps([record]))

    for path in (wrapped, bare):
        (raw,) = JsonExportReader().read(path)
        assert raw.source_id == "A1"
        assert raw.source_format is SourceFormat.JSON
        assert raw.mrn == "1"
        assert raw.text == "hello"


def test_json_reader_rejects_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    with pytest.raises(ReaderError, match="cannot parse JSON"):
        list(JsonExportReader().read(path))


def test_csv_reader_groups_rows_into_panels(tmp_path: Path) -> None:
    path = tmp_path / "labs.csv"
    path.write_text(
        "mrn,patient_name,panel_id,collected_at,test_name,loinc,value,unit,reference_range,flag\n"
        '1,"Doe, Jane",P1,2025-01-01 08:00,Sodium,2951-2,140,mmol/L,135-145,\n'
        '1,"Doe, Jane",P1,2025-01-01 08:00,Potassium,2823-3,5.9,mmol/L,3.5-5.1,H\n'
        '2,"Roe, Rick",P2,2025-01-02 08:00,Sodium,2951-2,131,mmol/L,135-145,L\n'
    )
    panels = list(LabCsvReader().read(path))
    assert len(panels) == 2
    first = panels[0]
    assert first.doc_type == "Lab Result"
    assert [o.test_name for o in first.observations] == ["Sodium", "Potassium"]
    assert first.title == "Lab panel: Potassium, Sodium"
    assert "lines 2,3" in first.source_location


def test_csv_reader_requires_core_columns(tmp_path: Path) -> None:
    path = tmp_path / "labs.csv"
    path.write_text("mrn,unit\n1,mg\n")
    with pytest.raises(ReaderError, match="missing required columns"):
        list(LabCsvReader().read(path))


def test_text_reader_parses_headers_and_body(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text(
        "patient name:  Jane Doe\nMRN:   mrn: 42\nDOB: 01/02/1970\nDocument Type: Progress Note\n"
        "-----------\n  Line one.\n  Line two.\n"
    )
    (raw,) = TextNoteReader().read(path)
    assert raw.name == "Jane Doe"
    assert raw.mrn == "mrn: 42"
    assert raw.dob == "01/02/1970"
    assert raw.doc_type == "Progress Note"
    assert raw.text is not None and "Line one." in raw.text and "Line two." in raw.text


def test_text_reader_without_separator_and_unknown_type(tmp_path: Path) -> None:
    path = tmp_path / "scan.txt"
    path.write_text("MRN: 7\nSome free text without a separator line.\n")
    (raw,) = TextNoteReader().read(path)
    assert raw.mrn == "7"
    assert raw.doc_type == "Scanned Document"
    assert raw.text == "Some free text without a separator line."


def test_discover_files_filters_by_registered_suffix(tmp_path: Path) -> None:
    for name in ("a.json", "b.csv", "notes/c.txt", "ignore.pdf"):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    assert [p.name for p in discover_files(tmp_path)] == ["a.json", "b.csv", "c.txt"]
