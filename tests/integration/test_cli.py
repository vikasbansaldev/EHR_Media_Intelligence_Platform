"""End-to-end: drive the CLI exactly as a user would."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ehr_intel.cli import main
from tests.conftest import SAMPLE_DATA

pytestmark = pytest.mark.integration


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # no stray .env
    var = tmp_path / "var"
    monkeypatch.setenv("EHR_RAW_DATA_DIR", str(SAMPLE_DATA))
    monkeypatch.setenv("EHR_VAR_DIR", str(var))
    monkeypatch.setenv("EHR_EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("EHR_SUMMARY_PROVIDER", "rule_based")
    return var


def test_run_all_then_search(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run-all"]) == 0
    out = capsys.readouterr().out
    assert "13/13 bundles valid" in out
    for artifact in (
        "reports/ingestion_report.json",
        "reports/fhir_validation_report.md",
        "reports/summaries.json",
        "index/manifest.json",
        "fhir_store.sqlite3",
    ):
        assert (cli_env / artifact).exists(), artifact

    # Second run: every summary comes from the cache.
    assert main(["summarize"]) == 0
    assert "cache hits 13, generated 0" in capsys.readouterr().out

    assert main(["search", "pulmonary embolism", "--type", "DiagnosticReport", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["patient_name"] == "Parker Stubbins"
    assert all(r["resource_type"] == "DiagnosticReport" for r in payload["results"])

    assert main(["search", "fracture", "-k", "1"]) == 0
    assert "Taylor Demoson" in capsys.readouterr().out


def test_individual_stages(cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ingest"]) == 0
    assert main(["fhir"]) == 0
    assert main(["index"]) == 0
    out = capsys.readouterr().out
    assert "[1/4] Ingestion" in out and "[2/4] FHIR" in out and "[4/4] Search index" in out


def test_generate_data(cli_env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_dir = tmp_path / "generated"
    assert main(["generate-data", "--out", str(out_dir), "--seed", "42"]) == 0
    # Byte-identical on every OS (the generator writes LF explicitly).
    for rel in ("ehr_export.json", "lab_results.csv", "notes/progress_10001.txt"):
        assert (out_dir / rel).read_bytes() == (SAMPLE_DATA / rel).read_bytes(), rel
    assert len(list((out_dir / "notes").glob("*.txt"))) == 18
