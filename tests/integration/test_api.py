"""HTTP API tests (FastAPI TestClient) over the full sample pipeline."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ehr_intel.api import build_services, create_app
from ehr_intel.config import Settings

pytestmark = pytest.mark.integration
MRN = "00010012"  # synthetic pulmonary embolism patient


@pytest.fixture
def client(populated_settings: Settings) -> Iterator[TestClient]:
    services = build_services(populated_settings)
    services.summaries.summarize_all()
    services.search.build()
    with TestClient(create_app(populated_settings, services)) as c:
        yield c


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["patients"] == 13
    assert body["indexed_documents"] >= 50
    assert body["embedder"].startswith("hashing")


def test_search_returns_top5_with_scores(client: TestClient) -> None:
    res = client.post("/search", json={"query": "pulmonary embolism chest pain"})
    assert res.status_code == 200
    body = res.json()
    results = body["results"]
    assert len(results) == 5
    assert results[0]["mrn"] == MRN
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    for key in ("patient_name", "mrn", "date", "resource_type", "score", "summary_snippet"):
        assert key in results[0]


def test_search_filters_as_query_params(client: TestClient) -> None:
    res = client.post(
        "/search?resource_type=DiagnosticReport&date_from=2025-03-01&date_to=2025-05-31",
        json={"query": "imaging findings", "top_k": 10},
    )
    assert res.status_code == 200
    results = res.json()["results"]
    assert results
    assert all(r["resource_type"] == "DiagnosticReport" for r in results)
    assert all("2025-03-01" <= r["date"] <= "2025-05-31" for r in results)


def test_search_multiple_resource_types(client: TestClient) -> None:
    res = client.post(
        "/search?resource_type=DocumentReference&resource_type=ClinicalSummary",
        json={"query": "discharge", "top_k": 20},
    )
    types = {r["resource_type"] for r in res.json()["results"]}
    assert types <= {"DocumentReference", "ClinicalSummary"}


@pytest.mark.parametrize(
    ("url", "payload", "fragment"),
    [
        ("/search", {"query": "   "}, "blank"),
        ("/search", {}, "query"),
        ("/search", {"query": "x", "top_k": 0}, "top_k"),
        ("/search?resource_type=Bogus", {"query": "x"}, "unknown resource_type"),
        ("/search?date_from=2025-05-01&date_to=2025-01-01", {"query": "x"}, "date_from"),
        ("/search?date_from=not-a-date", {"query": "x"}, "date_from"),
    ],
)
def test_search_validation_errors(
    client: TestClient, url: str, payload: dict[str, object], fragment: str
) -> None:
    res = client.post(url, json=payload)
    assert res.status_code == 422
    assert fragment in res.text


def test_patients_and_detail(client: TestClient) -> None:
    patients = client.get("/patients").json()
    assert len(patients) == 13 and all(p["bundle_valid"] for p in patients)

    detail = client.get(f"/patients/{MRN}").json()
    assert detail["name"] == "Parker Stubbins"
    assert detail["summary"]["disclaimer"].startswith("AI-generated")
    assert detail["resource_counts"]["Patient"] == 1
    kinds = {r["resource_type"] for r in detail["resources"]}
    assert {"Encounter", "DocumentReference", "DiagnosticReport", "Observation"} <= kinds
    dates = [r["date"] for r in detail["resources"] if r["date"]]
    assert dates == sorted(dates, reverse=True)

    assert client.get("/patients/99999999").status_code == 404


def test_summary_bundle_and_resource_endpoints(client: TestClient) -> None:
    summary = client.get(f"/patients/{MRN}/summary").json()
    assert summary["cached"] is True
    assert summary["ai_generated"] is True

    bundle = client.get(f"/patients/{MRN}/bundle").json()
    assert bundle["resourceType"] == "Bundle"
    resource_id = bundle["entry"][1]["resource"]["id"]
    assert client.get(f"/resources/{resource_id}").json()["id"] == resource_id
    assert client.get("/resources/nope").status_code == 404
    assert client.get("/patients/nope/bundle").status_code == 404


def test_filters_endpoint(client: TestClient) -> None:
    body = client.get("/filters").json()
    assert body["resource_types"][-1] == "ClinicalSummary"
    assert "DiagnosticReport" in body["resource_types"]
    assert body["date_min"] < body["date_max"]


def test_ui_is_served(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_empty_store_reports_degraded(settings: Settings) -> None:
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/search", json={"query": "x"}).status_code == 503


@pytest.mark.performance
def test_search_latency_under_two_seconds(client: TestClient) -> None:
    """Requirement: results in < 2 s for 50 records. We index 77 docs and measure p95."""
    queries = [
        "chest pain",
        "renal ultrasound",
        "elevated white count",
        "mammogram mass",
        "shortness of breath",
        "fracture",
        "troponin",
        "stroke",
    ]
    timings = []
    for _ in range(5):
        for q in queries:
            start = time.perf_counter()
            assert (
                client.post("/search?resource_type=DiagnosticReport", json={"query": q}).status_code
                == 200
            )
            timings.append(time.perf_counter() - start)
    p95 = statistics.quantiles(timings, n=20)[-1]
    assert max(timings) < 2.0
    assert p95 < 0.5, f"p95 {p95:.3f}s"


def test_identity_flags_surface_in_detail_and_search(client: TestClient) -> None:
    detail = client.get("/patients/00090004").json()
    assert any("possible duplicate patient" in f.lower() for f in detail["identity_flags"])
    conflict = client.get("/patients/00010003").json()["identity_flags"]
    assert any("Conflicting birth date" in f for f in conflict)
    assert client.get("/patients/00010012").json()["identity_flags"] == []

    results = client.post("/search", json={"query": "breast surgery consult biopsy"}).json()
    flagged = [r for r in results["results"] if r["mrn"] == "00090004"]
    assert flagged and flagged[0]["patient_flags"]


def test_ui_markup_is_accessible(client: TestClient) -> None:
    html = client.get("/").text
    for marker in (
        'role="search"',
        'aria-live="polite"',
        'role="dialog"',
        'aria-modal="true"',
        'for="query"',
        'for="filter-type"',
        'for="filter-from"',
        'for="filter-to"',
        "Skip to results",
    ):
        assert marker in html, marker
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/styles.css").status_code == 200
