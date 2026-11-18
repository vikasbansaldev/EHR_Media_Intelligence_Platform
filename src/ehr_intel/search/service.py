"""Search service: builds/loads the index and answers ranked, filtered queries."""

from __future__ import annotations

import logging
import re
import time
from datetime import date

from pydantic import BaseModel, Field

from ehr_intel.config import Settings
from ehr_intel.fhir.store import BundleStore
from ehr_intel.search.documents import SUMMARY_TYPE, build_documents
from ehr_intel.search.embeddings import Embedder, build_embedder
from ehr_intel.search.index import (
    EmbedderMismatchError,
    IndexNotReadyError,
    SearchFilters,
    VectorIndex,
)
from ehr_intel.summarization.models import ClinicalSummary
from ehr_intel.summarization.service import SummarizationService

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)
SNIPPET_CHARS = 240


class SearchResult(BaseModel):
    rank: int
    score: float = Field(description="Cosine similarity of query and document (higher is better).")
    doc_id: str
    mrn: str
    patient_name: str | None
    resource_id: str | None
    resource_type: str
    kind: str
    title: str
    date: str | None
    snippet: str
    summary_snippet: str | None = None
    summary_confidence: str | None = None
    patient_flags: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_indexed: int
    took_ms: float
    embedder: str
    vector_backend: str


def best_snippet(text: str, query: str, size: int = SNIPPET_CHARS) -> str:
    """The sentence window that shares the most words with the query."""
    sentences = [s.strip() for s in re.split(r"(?<=[.;!?])\s+", text) if s.strip()]
    if not sentences:
        return ""
    terms = set(_WORD.findall(query.lower())) - _STOPWORDS
    best = max(
        range(len(sentences)),
        key=lambda i: (len(terms & set(_WORD.findall(sentences[i].lower()))), -i),
    )
    snippet = sentences[best]
    for extra in sentences[best + 1 :]:
        if len(snippet) + len(extra) + 1 > size:
            break
        snippet = f"{snippet} {extra}"
    return snippet if len(snippet) <= size else snippet[: size - 1].rstrip() + "…"


def summary_snippet(summary: ClinicalSummary, size: int = 200) -> str:
    text = summary.chief_concern.rstrip(".")
    if summary.key_diagnoses:
        text += ". Dx: " + "; ".join(summary.key_diagnoses[:2])
    return text if len(text) <= size else text[: size - 1].rstrip() + "…"


class SearchService:
    def __init__(
        self,
        settings: Settings,
        store: BundleStore,
        summaries: SummarizationService,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.summaries = summaries
        self.embedder = embedder or build_embedder(settings)
        self.index = VectorIndex(self.embedder, settings.vector_backend)
        self._summary_by_mrn: dict[str, ClinicalSummary] = {}
        self._flags_by_mrn: dict[str, list[str]] = {}

    # ------------------------------------------------------------ lifecycle --

    def _load_summaries(self) -> dict[str, ClinicalSummary]:
        found = {}
        patients = self.store.patients()
        self._flags_by_mrn = {p.mrn: p.flags for p in patients if p.flags}
        for patient in patients:
            if (s := self.summaries.latest(patient.mrn)) is not None:
                found[patient.mrn] = s
        return found

    def build(self, *, persist: bool = True) -> int:
        self._summary_by_mrn = self._load_summaries()
        documents = build_documents(self.store, self._summary_by_mrn)
        self.index.build(documents)
        if persist:
            self.index.save(self.settings.index_dir)
        return len(documents)

    def load_or_build(self) -> None:
        try:
            self.index = VectorIndex.load(
                self.settings.index_dir, self.embedder, self.settings.vector_backend
            )
            self._summary_by_mrn = self._load_summaries()
            logger.info("Loaded search index (%d documents)", len(self.index))
        except (IndexNotReadyError, EmbedderMismatchError) as exc:
            logger.info("Building search index: %s", exc)
            self.build()

    @property
    def ready(self) -> bool:
        return len(self.index) > 0

    # --------------------------------------------------------------- search --

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        resource_types: list[str] | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> SearchResponse:
        started = time.perf_counter()
        filters = SearchFilters(
            resource_types=frozenset(resource_types) if resource_types else None,
            date_from=date_from,
            date_to=date_to,
        )
        hits = self.index.search(query, top_k or self.settings.search_top_k, filters)

        results = []
        for rank, hit in enumerate(hits, start=1):
            doc = hit.doc
            summary = self._summary_by_mrn.get(doc.mrn)
            results.append(
                SearchResult(
                    rank=rank,
                    score=round(hit.score, 4),
                    doc_id=doc.doc_id,
                    mrn=doc.mrn,
                    patient_name=doc.patient_name,
                    resource_id=doc.resource_id,
                    resource_type=doc.resource_type,
                    kind=doc.kind,
                    title=doc.title,
                    date=doc.date,
                    snippet=best_snippet(doc.text, query),
                    summary_snippet=summary_snippet(summary) if summary else None,
                    summary_confidence=summary.confidence if summary else None,
                    patient_flags=self._flags_by_mrn.get(doc.mrn, []),
                )
            )
        took_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(
            query=query,
            results=results,
            total_indexed=len(self.index),
            took_ms=round(took_ms, 2),
            embedder=self.embedder.name,
            vector_backend=self.index.backend,
        )

    def available_types(self) -> list[str]:
        types = {d.resource_type for d in self.index.documents}
        return sorted(types, key=lambda t: (t == SUMMARY_TYPE, t))

    def date_bounds(self) -> tuple[str | None, str | None]:
        dates = [d.date for d in self.index.documents if d.date]
        return (min(dates), max(dates)) if dates else (None, None)
