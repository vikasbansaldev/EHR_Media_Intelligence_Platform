"""SQLite cache for generated summaries.

The key is a hash of ``(MRN, Bundle content hash, provider, model, prompt
version)``. Any change to the patient's records, the model or the prompt
produces a new key, so stale summaries are never served and unchanged
patients never trigger a second API call.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ehr_intel.summarization.models import ClinicalSummary

_SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    cache_key    TEXT PRIMARY KEY,
    mrn          TEXT NOT NULL,
    source_hash  TEXT NOT NULL,
    provider     TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_summaries_mrn ON summaries(mrn, created_at);
"""


def cache_key(mrn: str, source_hash: str, provider: str, model: str, prompt_version: str) -> str:
    return hashlib.sha256(
        "|".join((mrn, source_hash, provider, model, prompt_version)).encode()
    ).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class SummaryCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stats = CacheStats()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get(self, key: str) -> ClinicalSummary | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary_json FROM summaries WHERE cache_key = ?", (key,)
            ).fetchone()
        if row is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return ClinicalSummary.model_validate_json(row[0]).model_copy(update={"cached": True})

    def put(self, key: str, summary: ClinicalSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?,?)",
                (
                    key,
                    summary.mrn,
                    summary.source_hash,
                    summary.provider,
                    summary.model_copy(update={"cached": False}).model_dump_json(),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def latest_for(self, mrn: str, source_hash: str | None = None) -> ClinicalSummary | None:
        """Most recent summary for a patient (optionally only for the current Bundle)."""
        sql = "SELECT summary_json FROM summaries WHERE mrn = ?"
        params: list[str] = [mrn]
        if source_hash:
            sql += " AND source_hash = ?"
            params.append(source_hash)
        with self._connect() as conn:
            row = conn.execute(sql + " ORDER BY created_at DESC LIMIT 1", params).fetchone()
        return ClinicalSummary.model_validate_json(row[0]) if row else None

    def __len__(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0])
