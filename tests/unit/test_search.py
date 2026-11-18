"""Tests for embedders, the FAISS index and the search service."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from ehr_intel.config import Settings
from ehr_intel.fhir.store import BundleStore
from ehr_intel.search import (
    HashingEmbedder,
    IndexNotReadyError,
    SearchDocument,
    SearchFilters,
    SearchService,
    VectorIndex,
    build_embedder,
)
from ehr_intel.search import embeddings as emb_module
from ehr_intel.search.index import (
    EmbedderMismatchError,
    Hit,
    NumpyBackend,
    faiss_available,
    make_backend,
)
from ehr_intel.search.service import best_snippet
from ehr_intel.summarization import SummarizationService

# --- Embedders ----------------------------------------------------------------------------


def _cos(e: HashingEmbedder, a: str, b: str) -> float:
    va, vb = e.embed([a, b])
    return float(va @ vb)


def test_hashing_embedder_is_normalized_and_deterministic() -> None:
    e = HashingEmbedder(256)
    vectors = e.embed(["acute myocardial infarction", "", "renal failure"])
    assert vectors.shape == (3, 256) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors[[0, 2]], axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(
        vectors, HashingEmbedder(256).embed(["acute myocardial infarction", "", "renal failure"])
    )
    assert e.embed([]).shape == (0, 256)


def test_hashing_embedder_bridges_lay_and_clinical_terms() -> None:
    e = HashingEmbedder()
    assert _cos(e, "heart attack", "NSTEMI myocardial infarction troponin") > _cos(
        e, "heart attack", "distal radius fracture"
    )
    assert _cos(e, "blood clots in the lungs", "acute pulmonary embolism") > _cos(
        e, "blood clots in the lungs", "iron deficiency anemia"
    )


def test_build_embedder_falls_back_when_model_unavailable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(_name: str) -> None:
        raise ImportError("no sentence-transformers")

    monkeypatch.setattr(emb_module, "SentenceTransformerEmbedder", unavailable)
    auto = settings.model_copy(update={"embedding_provider": "auto"})
    assert isinstance(build_embedder(auto), HashingEmbedder)
    strict = settings.model_copy(update={"embedding_provider": "sentence_transformer"})
    with pytest.raises(ImportError):
        build_embedder(strict)


# --- Index --------------------------------------------------------------------------------


def _doc(
    i: int,
    text: str,
    rtype: str = "DocumentReference",
    day: str | None = "2025-01-01",
    mrn: str = "00000001",
) -> SearchDocument:
    return SearchDocument(
        doc_id=f"d{i}",
        mrn=mrn,
        patient_name="Test Patient",
        resource_id=f"r{i}",
        resource_type=rtype,
        kind="Note",
        title=f"Doc {i}",
        date=day,
        text=text,
    )


BACKENDS = [
    "numpy",
    pytest.param(
        "faiss", marks=pytest.mark.skipif(not faiss_available(), reason="faiss-cpu not installed")
    ),
]


@pytest.fixture(params=BACKENDS)
def small_index(request: pytest.FixtureRequest) -> VectorIndex:
    index = VectorIndex(HashingEmbedder(), request.param)
    assert index.backend == request.param
    index.build(
        [
            _doc(0, "Acute pulmonary embolism on CT angiogram", "DiagnosticReport", "2025-01-05"),
            _doc(1, "Pneumonia right lower lobe consolidation", "DiagnosticReport", "2025-02-10"),
            _doc(2, "Discharge summary for pulmonary embolism, anticoagulated", day="2025-03-01"),
            _doc(3, "Wrist fracture after fall", day=None),
        ]
    )
    return index


def test_index_ranks_by_similarity(small_index: VectorIndex) -> None:
    hits = small_index.search("pulmonary embolism", k=4)
    assert {hits[0].doc.doc_id, hits[1].doc.doc_id} == {"d0", "d2"}
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_index_filters_inside_search(small_index: VectorIndex) -> None:
    by_type = small_index.search(
        "pulmonary embolism",
        k=5,
        filters=SearchFilters(resource_types=frozenset({"DocumentReference"})),
    )
    assert by_type[0].doc.doc_id == "d2"
    assert all(h.doc.resource_type == "DocumentReference" for h in by_type)

    by_date = small_index.search(
        "embolism",
        k=5,
        filters=SearchFilters(date_from=date(2025, 2, 1), date_to=date(2025, 2, 28)),
    )
    assert [h.doc.doc_id for h in by_date] == ["d1"]  # undated docs excluded by date filter

    none = small_index.search("anything", k=5, filters=SearchFilters(date_from=date(2030, 1, 1)))
    assert none == []


def test_index_persistence_round_trip(small_index: VectorIndex, tmp_path: Path) -> None:
    small_index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, HashingEmbedder())
    assert len(loaded) == 4
    assert [h.doc.doc_id for h in loaded.search("fracture", k=1)] == ["d3"]
    with pytest.raises(EmbedderMismatchError):
        VectorIndex.load(tmp_path, HashingEmbedder(128))
    with pytest.raises(IndexNotReadyError):
        VectorIndex.load(tmp_path / "missing", HashingEmbedder())


def _by_score(hits: list[Hit]) -> dict[float, set[str]]:
    groups: dict[float, set[str]] = {}
    for h in hits:
        groups.setdefault(round(h.score, 5), set()).add(h.doc.doc_id)
    return groups


@pytest.mark.skipif(not faiss_available(), reason="faiss-cpu not installed")
def test_backends_return_identical_rankings(small_index: VectorIndex) -> None:
    docs = small_index.documents
    faiss_idx, numpy_idx = (
        VectorIndex(HashingEmbedder(), "faiss"),
        VectorIndex(HashingEmbedder(), "numpy"),
    )
    faiss_idx.build(docs)
    numpy_idx.build(docs)
    for query in ("pulmonary embolism", "fracture", "pneumonia consolidation"):
        for filters in (None, SearchFilters(resource_types=frozenset({"DiagnosticReport"}))):
            a = faiss_idx.search(query, len(docs), filters)
            b = numpy_idx.search(query, len(docs), filters)
            # Same scores in the same order; documents tied on score may come in any order.
            assert [round(h.score, 5) for h in a] == [round(h.score, 5) for h in b], query
            assert _by_score(a) == _by_score(b), query


def test_index_saved_by_one_backend_loads_with_the_other(
    small_index: VectorIndex, tmp_path: Path
) -> None:
    small_index.save(tmp_path)
    other = (
        "numpy" if small_index.backend == "faiss" else ("faiss" if faiss_available() else "numpy")
    )
    loaded = VectorIndex.load(tmp_path, HashingEmbedder(), other)  # type: ignore[arg-type]
    assert loaded.backend == other
    assert [h.doc.doc_id for h in loaded.search("fracture", k=1)] == ["d3"]


def test_auto_backend_falls_back_to_numpy_without_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "faiss", None)  # makes `import faiss` raise ImportError
    assert isinstance(make_backend("auto", 8), NumpyBackend)
    with pytest.raises(ImportError):
        make_backend("faiss", 8)


def test_empty_index_raises() -> None:
    with pytest.raises(IndexNotReadyError):
        VectorIndex(HashingEmbedder()).search("x", k=5)


def test_best_snippet_picks_matching_sentence() -> None:
    text = "Patient admitted. Vitals stable. CT shows acute pulmonary embolism. Discharged home."
    assert best_snippet(text, "embolism").startswith("CT shows acute pulmonary embolism.")
    assert len(best_snippet("word " * 200, "word", size=50)) <= 50


# --- Service over the sample dataset ------------------------------------------------------


@pytest.fixture
def service(populated_settings: Settings) -> SearchService:
    store = BundleStore(populated_settings.bundle_db_path)
    summaries = SummarizationService(populated_settings, store)
    summaries.summarize_all()
    svc = SearchService(populated_settings, store, summaries)
    svc.build()
    return svc


@pytest.mark.parametrize(
    ("query", "expected_patient"),
    [
        ("blood clot in the lungs", "Parker Stubbins"),
        ("heart attack with elevated troponin", "Rowan Ipsumfield"),
        ("suspicious breast mass needing biopsy", "Casey Fakenham"),
        ("broken wrist after a fall", "Taylor Demoson"),
        ("stroke with slurred speech", "Morgan Synthby"),
        ("kidney failure creatinine", "Riley Mockford"),
    ],
)
def test_natural_language_queries_find_the_right_patient(
    service: SearchService, query: str, expected_patient: str
) -> None:
    response = service.search(query)
    names = [r.patient_name for r in response.results[:3]]
    assert expected_patient in names, (query, names)


def test_response_shape(service: SearchService) -> None:
    response = service.search("chest x-ray")
    assert len(response.results) == 5
    assert [r.rank for r in response.results] == [1, 2, 3, 4, 5]
    top = response.results[0]
    assert top.mrn and top.title and top.snippet and top.resource_type
    assert top.summary_snippet, "results carry the patient's AI summary snippet"
    assert response.total_indexed > 50


def test_summaries_are_indexed_and_filterable(service: SearchService) -> None:
    response = service.search("heart failure", resource_types=["ClinicalSummary"])
    assert response.results
    assert all(r.resource_type == "ClinicalSummary" for r in response.results)
    assert "ClinicalSummary" in service.available_types()


def test_load_or_build_reuses_persisted_index(
    service: SearchService, populated_settings: Settings
) -> None:
    fresh = SearchService(
        populated_settings,
        service.store,
        service.summaries,
        embedder=HashingEmbedder(populated_settings.hashing_dim),
    )
    fresh.load_or_build()
    assert len(fresh.index) == len(service.index)
    assert fresh.date_bounds()[0] is not None


def test_best_snippet_ignores_stopwords() -> None:
    text = "Discharged home in the evening. Filling defects consistent with embolism in lungs."
    assert best_snippet(text, "clot in the lungs").startswith("Filling defects")
