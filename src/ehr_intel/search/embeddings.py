"""Text embedders.

* :class:`SentenceTransformerEmbedder` - ``all-MiniLM-L6-v2`` (384-d), the
  default when ``sentence-transformers`` is installed and the model is available.
* :class:`HashingEmbedder` - dependency-free, deterministic feature-hashing
  embedder with light clinical synonym expansion. Used automatically when the
  model cannot be loaded (offline CI, air-gapped hosts) and in tests.

All embedders return L2-normalized ``float32`` matrices, so inner product
equals cosine similarity.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
import re
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from ehr_intel.config import Settings

logger = logging.getLogger(__name__)

Matrix = npt.NDArray[np.float32]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> Matrix: ...


def _normalize(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model: Any = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name.split('/')[-1]}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> Matrix:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return cast(Matrix, np.asarray(vectors, dtype=np.float32))


# Lay terms -> clinical vocabulary, so natural-language queries meet chart language.
_SYNONYMS: dict[str, str] = {
    "heart attack": "myocardial infarction nstemi troponin",
    "heart failure": "chf congestive cardiomegaly bnp",
    "chf": "congestive heart failure",
    "mi": "myocardial infarction",
    "stroke": "infarct cerebral mca ischemic",
    "blood clot": "embolism thrombosis d-dimer",
    "clot": "embolism thrombosis",
    "pe": "pulmonary embolism",
    "kidney": "renal creatinine ckd",
    "renal": "kidney",
    "lung": "pulmonary chest",
    "breathing": "dyspnea respiratory shortness breath",
    "shortness of breath": "dyspnea",
    "sugar": "glucose diabetes a1c",
    "diabetes": "diabetic a1c hyperglycemia glucose",
    "infection": "sepsis pneumonia leukocytosis wbc",
    "x-ray": "radiograph xray imaging",
    "xray": "x-ray radiograph imaging",
    "scan": "ct mri imaging",
    "imaging": "radiology x-ray ct mri ultrasound mammogram",
    "broken": "fracture",
    "bone": "fracture",
    "cancer": "mass malignancy suspicious biopsy bi-rads",
    "tumor": "mass",
    "lump": "mass",
    "anemia": "hemoglobin iron",
    "low blood": "anemia hemoglobin",
    "abnormal": "high low elevated",
    "elevated": "high",
    "labs": "lab laboratory panel",
    "blood test": "lab laboratory panel",
}
_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "es", "s", "al", "ic")


def _stem(token: str) -> str:
    if len(token) <= 4 or token.isdigit():
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


class HashingEmbedder:
    """Unigrams + bigrams + character trigrams, hashed into a fixed-size vector.

    Deterministic across processes (uses BLAKE2b, not Python's salted ``hash``).
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _expand(self, text: str) -> str:
        lowered = text.lower()
        extra = [v for k, v in _SYNONYMS.items() if re.search(rf"\b{re.escape(k)}s?\b", lowered)]
        return " ".join([lowered, *extra])

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, (1.0 if (value >> 63) & 1 else -1.0)

    def _vector(self, text: str) -> npt.NDArray[np.float32]:
        tokens = [_stem(t) for t in _TOKEN.findall(self._expand(text))]
        counts: dict[str, float] = {}
        for t in tokens:
            counts[f"w:{t}"] = counts.get(f"w:{t}", 0.0) + 1.0
            if len(t) >= 5:
                for i in range(len(t) - 2):
                    counts[f"c:{t[i : i + 3]}"] = counts.get(f"c:{t[i : i + 3]}", 0.0) + 0.25
        for a, b in itertools.pairwise(tokens):
            counts[f"b:{a}_{b}"] = counts.get(f"b:{a}_{b}", 0.0) + 0.5
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature, count in counts.items():
            index, sign = self._bucket(feature)
            vec[index] += sign * (1.0 + math.log(count)) if count >= 1 else sign * count
        return vec

    def embed(self, texts: list[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.vstack([self._vector(t) for t in texts]))


def build_embedder(settings: Settings) -> Embedder:
    """Resolve the configured embedder, falling back to hashing when the model is unavailable."""
    if settings.embedding_provider == "hashing":
        return HashingEmbedder(settings.hashing_dim)
    try:
        return SentenceTransformerEmbedder(settings.embedding_model)
    except Exception as exc:  # ImportError, network/OSError while downloading, ...
        if settings.embedding_provider == "sentence_transformer":
            raise
        logger.warning(
            "sentence-transformers unavailable (%s); using hashing embedder", exc.__class__.__name__
        )
        return HashingEmbedder(settings.hashing_dim)
