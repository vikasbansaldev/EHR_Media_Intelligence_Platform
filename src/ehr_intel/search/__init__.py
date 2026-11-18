"""Task 4 - semantic search over FHIR documents and AI summaries."""

from ehr_intel.search.documents import SEARCHABLE_TYPES, SUMMARY_TYPE, SearchDocument
from ehr_intel.search.embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)
from ehr_intel.search.index import IndexNotReadyError, SearchFilters, VectorIndex
from ehr_intel.search.service import SearchResponse, SearchResult, SearchService

__all__ = [
    "SEARCHABLE_TYPES",
    "SUMMARY_TYPE",
    "Embedder",
    "HashingEmbedder",
    "IndexNotReadyError",
    "SearchDocument",
    "SearchFilters",
    "SearchResponse",
    "SearchResult",
    "SearchService",
    "SentenceTransformerEmbedder",
    "VectorIndex",
    "build_embedder",
]
