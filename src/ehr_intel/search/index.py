"""Vector index with metadata and filter-aware exact cosine search.

Two interchangeable backends compute the same result:

* :class:`FaissBackend` - FAISS ``IndexFlatIP``; filters are passed to FAISS as
  an ``IDSelectorBatch`` so top-k is computed only over matching documents.
  Used whenever ``faiss-cpu`` is installed.
* :class:`NumpyBackend` - a single matrix-vector product with a boolean mask.
  Used where FAISS has no wheel (e.g. Python 3.15 on Windows) or when
  ``EHR_VECTOR_BACKEND=numpy``.

Both do exact search over L2-normalized vectors (inner product = cosine), the
right choice up to ~100k documents: no training, no recall loss. Vectors are
persisted as a plain ``.npy`` matrix, so an index built with one backend loads
with the other.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from ehr_intel.search.documents import SearchDocument
from ehr_intel.search.embeddings import Embedder, Matrix

logger = logging.getLogger(__name__)

VECTORS_FILE = "vectors.npy"
DOCS_FILE = "documents.json"
MANIFEST_FILE = "manifest.json"

BackendName = Literal["auto", "faiss", "numpy"]
IdArray = npt.NDArray[np.int64]


class IndexNotReadyError(RuntimeError):
    pass


class EmbedderMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchFilters:
    resource_types: frozenset[str] | None = None
    date_from: date | None = None
    date_to: date | None = None

    def matches(self, doc: SearchDocument) -> bool:
        if self.resource_types and doc.resource_type not in self.resource_types:
            return False
        if self.date_from or self.date_to:
            if doc.date is None:
                return False
            d = date.fromisoformat(doc.date)
            if self.date_from and d < self.date_from:
                return False
            if self.date_to and d > self.date_to:
                return False
        return True

    @property
    def active(self) -> bool:
        return bool(self.resource_types or self.date_from or self.date_to)


@dataclass(frozen=True)
class Hit:
    doc: SearchDocument
    score: float


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class VectorBackend(Protocol):
    name: str

    def add(self, vectors: Matrix) -> None: ...

    def search(self, query: Matrix, k: int, allowed: IdArray | None) -> list[tuple[int, float]]: ...


class NumpyBackend:
    """Exact cosine search with NumPy: ``scores = V @ q``, masked, then partial sort."""

    name = "numpy"

    def __init__(self, dim: int) -> None:
        self._vectors: Matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: Matrix) -> None:
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query: Matrix, k: int, allowed: IdArray | None) -> list[tuple[int, float]]:
        scores = self._vectors @ query[0]
        candidates = allowed if allowed is not None else np.arange(len(scores), dtype=np.int64)
        if candidates.size == 0:
            return []
        k = min(k, int(candidates.size))
        sub = scores[candidates]
        top = np.argpartition(-sub, k - 1)[:k]
        top = top[np.argsort(-sub[top], kind="stable")]
        return [(int(candidates[i]), float(sub[i])) for i in top]


class FaissBackend:
    """FAISS ``IndexFlatIP`` with in-index filtering via ``IDSelectorBatch``."""

    name = "faiss"

    def __init__(self, dim: int) -> None:
        import faiss  # optional dependency; ImportError -> caller falls back

        self._faiss: Any = faiss
        self._index: Any = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, vectors: Matrix) -> None:
        if len(vectors):
            self._index.add_with_ids(vectors, np.arange(len(vectors), dtype=np.int64))

    def search(self, query: Matrix, k: int, allowed: IdArray | None) -> list[tuple[int, float]]:
        params = None
        if allowed is not None:
            if allowed.size == 0:
                return []
            params = self._faiss.SearchParameters()
            params.sel = self._faiss.IDSelectorBatch(allowed)
            k = min(k, int(allowed.size))
        k = min(k, int(self._index.ntotal))
        scores, ids = self._index.search(query, k, params=params)
        return [(int(i), float(s)) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


def make_backend(name: BackendName, dim: int) -> VectorBackend:
    """``auto`` prefers FAISS and falls back to NumPy when FAISS is not installed."""
    if name == "numpy":
        return NumpyBackend(dim)
    try:
        return FaissBackend(dim)
    except ImportError:
        if name == "faiss":
            raise
        logger.info("faiss-cpu not installed; using the NumPy vector backend")
        return NumpyBackend(dim)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class VectorIndex:
    def __init__(self, embedder: Embedder, backend: BackendName = "auto") -> None:
        self.embedder = embedder
        self.backend_name: BackendName = backend
        self._backend: VectorBackend = make_backend(backend, embedder.dim)
        self._vectors: Matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self.documents: list[SearchDocument] = []
        self.built_at: datetime | None = None

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def backend(self) -> str:
        return self._backend.name

    def _load_vectors(self, vectors: Matrix, documents: list[SearchDocument]) -> None:
        backend = make_backend(self.backend_name, self.embedder.dim)
        backend.add(vectors)
        self._backend, self._vectors, self.documents = backend, vectors, list(documents)

    # ------------------------------------------------------------- building --

    def build(self, documents: list[SearchDocument]) -> None:
        vectors: Matrix = self.embedder.embed([d.embedding_text() for d in documents])
        self._load_vectors(vectors, documents)
        self.built_at = datetime.now(UTC)
        logger.info(
            "Indexed %d documents with %s (%s backend)",
            len(documents),
            self.embedder.name,
            self.backend,
        )

    # --------------------------------------------------------------- search --

    def search(self, query: str, k: int, filters: SearchFilters | None = None) -> list[Hit]:
        if not self.documents:
            raise IndexNotReadyError("search index is empty; run `ehr-intel index` first")
        filters = filters or SearchFilters()
        allowed: IdArray | None = None
        if filters.active:
            allowed = np.array(
                [i for i, d in enumerate(self.documents) if filters.matches(d)], dtype=np.int64
            )
            if allowed.size == 0:
                return []
        k = min(k, len(self.documents))
        hits = self._backend.search(self.embedder.embed([query]), k, allowed)
        return [Hit(doc=self.documents[i], score=score) for i, score in hits]

    # ---------------------------------------------------------- persistence --

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self._vectors)
        (directory / DOCS_FILE).write_text(
            json.dumps([d.model_dump() for d in self.documents]), encoding="utf-8"
        )
        manifest = {
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "documents": len(self.documents),
            "built_with_backend": self.backend,
            "built_at": (self.built_at or datetime.now(UTC)).isoformat(),
        }
        (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, directory: Path, embedder: Embedder, backend: BackendName = "auto"
    ) -> VectorIndex:
        manifest_path = directory / MANIFEST_FILE
        if not manifest_path.exists() or not (directory / VECTORS_FILE).exists():
            raise IndexNotReadyError(f"no index at {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["embedder"] != embedder.name:
            raise EmbedderMismatchError(
                f"index built with {manifest['embedder']}, current embedder is {embedder.name}"
            )
        index = cls(embedder, backend)
        vectors = np.load(directory / VECTORS_FILE).astype(np.float32, copy=False)
        docs = json.loads((directory / DOCS_FILE).read_text(encoding="utf-8"))
        index._load_vectors(vectors, [SearchDocument.model_validate(d) for d in docs])
        index.built_at = datetime.fromisoformat(manifest["built_at"])
        return index
