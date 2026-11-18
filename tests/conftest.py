"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from ehr_intel.config import Settings, get_settings
from ehr_intel.ingestion.models import RawObservation, RawRecord, SourceFormat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "data" / "raw"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep tests hermetic: no real API keys, no cached settings, no .env leakage."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Offline settings writing all artifacts into a temp directory."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        raw_data_dir=SAMPLE_DATA,
        var_dir=tmp_path / "var",
        summary_provider="rule_based",
        embedding_provider="hashing",
        frontend_dir=PROJECT_ROOT / "frontend",
    )


@pytest.fixture(scope="session")
def pipeline_var(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run ingestion + FHIR once per session; tests copy the resulting var dir."""
    from ehr_intel.fhir import FhirNormalizer
    from ehr_intel.ingestion import IngestionPipeline

    var = tmp_path_factory.mktemp("pipeline") / "var"
    s = Settings(_env_file=None, raw_data_dir=SAMPLE_DATA, var_dir=var)  # type: ignore[call-arg]
    FhirNormalizer(s).run(IngestionPipeline(s).run(), write_files=False)
    return var


@pytest.fixture
def populated_settings(settings: Settings, pipeline_var: Path) -> Settings:
    """Settings whose var dir already holds the sample FHIR store (fresh copy per test)."""
    import shutil

    shutil.copytree(pipeline_var, settings.var_dir)
    return settings


RawFactory = Callable[..., RawRecord]


@pytest.fixture
def make_raw() -> RawFactory:
    """Build a valid :class:`RawRecord`, overriding any field via kwargs."""
    counter = {"n": 0}

    def factory(**overrides: Any) -> RawRecord:
        counter["n"] += 1
        observations = overrides.pop("observations", None)
        base: dict[str, Any] = {
            "source_id": f"SRC-{counter['n']}",
            "source_format": SourceFormat.JSON,
            "source_location": f"test.json#records[{counter['n']}]",
            "mrn": "MRN-00001234",
            "name": "DOE, JANE",
            "dob": "03/14/1962",
            "sex": "F",
            "encounter_id": "ENC-1",
            "encounter_class": "inpatient",
            "doc_type": "Discharge Summary",
            "title": "Discharge summary",
            "date": "2025-01-10T09:30:00",
            "author": "Dr. Test",
            "text": f"Discharge summary body number {counter['n']}.",
        }
        base.update(overrides)
        if observations is not None:
            base["observations"] = [RawObservation(**o) for o in observations]
        return RawRecord(**base)

    return factory
