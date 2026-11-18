"""SQLite store for FHIR Bundles and a flattened, queryable index of their resources.

Schema:

* ``bundles``   - one row per patient: the full Bundle JSON plus validity.
* ``resources`` - one row per resource, with the columns downstream stages
  need (type, date, title, plain text) so search and the UI never have to
  re-parse Bundles.

Connections are opened per operation, which keeps the store safe to use from
FastAPI's thread pool without sharing a connection across threads.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ehr_intel.fhir.mapper import PatientBundle
from ehr_intel.fhir.validator import BundleValidationResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bundles (
    mrn           TEXT PRIMARY KEY,
    bundle_id     TEXT NOT NULL,
    patient_id    TEXT NOT NULL,
    patient_name  TEXT,
    birth_date    TEXT,
    gender        TEXT,
    bundle_json   TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    is_valid      INTEGER NOT NULL,
    error_count   INTEGER NOT NULL,
    flags_json    TEXT NOT NULL DEFAULT '[]',
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resources (
    resource_id    TEXT PRIMARY KEY,
    mrn            TEXT NOT NULL REFERENCES bundles(mrn) ON DELETE CASCADE,
    resource_type  TEXT NOT NULL,
    effective_date TEXT,
    title          TEXT,
    text           TEXT,
    resource_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_resources_mrn ON resources(mrn);
CREATE INDEX IF NOT EXISTS ix_resources_type_date ON resources(resource_type, effective_date);
"""

# Resource types that represent a clinical "record" a clinician would open.
RECORD_RESOURCE_TYPES = ("DocumentReference", "DiagnosticReport")


class StoredResource(BaseModel):
    resource_id: str
    mrn: str
    resource_type: str
    effective_date: str | None
    title: str | None
    text: str | None
    resource: dict[str, Any]


class PatientRow(BaseModel):
    mrn: str
    patient_id: str
    name: str | None
    birth_date: str | None
    gender: str | None
    is_valid: bool
    error_count: int
    bundle_hash: str
    flags: list[str] = []


def record_kind(resource: dict[str, Any]) -> str:
    """Clinician-facing category: 'Imaging', 'Lab', or the document type text."""
    if resource.get("resourceType") == "DiagnosticReport":
        code = resource.get("category", [{}])[0].get("coding", [{}])[0].get("code")
        return {"RAD": "Imaging", "LAB": "Lab"}.get(code or "", "Diagnostic report")
    return str(resource.get("type", {}).get("text") or "Document")


def _decode_attachment(attachment: dict[str, Any]) -> str:
    data = attachment.get("data")
    return base64.b64decode(data).decode("utf-8") if data else ""


def resource_text(resource: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Extract ``(effective_date, title, plain_text)`` from a clinical resource."""
    rtype = resource.get("resourceType")
    if rtype == "DocumentReference":
        attachments = [c.get("attachment", {}) for c in resource.get("content", [])]
        text = "\n".join(_decode_attachment(a) for a in attachments)
        return resource.get("date"), resource.get("description"), text
    if rtype == "DiagnosticReport":
        forms = resource.get("presentedForm", [])
        report_text = "\n".join(_decode_attachment(a) for a in forms)
        return (
            resource.get("effectiveDateTime"),
            resource.get("code", {}).get("text"),
            report_text or resource.get("conclusion"),
        )
    if rtype == "Observation":
        return (resource.get("effectiveDateTime"), resource.get("code", {}).get("text"), None)
    if rtype == "Encounter":
        return resource.get("period", {}).get("start"), None, None
    return None, None, None


def bundle_hash(bundle: dict[str, Any]) -> str:
    """Stable digest of a Bundle's content (used as the summary-cache key)."""
    canonical = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class BundleStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    # --------------------------------------------------------------- writes --

    def save(
        self,
        pb: PatientBundle,
        validation: BundleValidationResult,
        flags: list[str] | None = None,
    ) -> None:
        """Persist a Bundle; ``flags`` are identity warnings a clinician must see."""
        patient = pb.resources[0]
        name = (patient.get("name") or [{}])[0].get("text")
        with self._connect() as conn:
            conn.execute("DELETE FROM bundles WHERE mrn = ?", (pb.mrn,))
            conn.execute(
                "INSERT INTO bundles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    pb.mrn,
                    pb.bundle["id"],
                    pb.patient_id,
                    name,
                    patient.get("birthDate"),
                    patient.get("gender"),
                    json.dumps(pb.bundle),
                    bundle_hash(pb.bundle),
                    int(validation.is_valid),
                    len(validation.errors),
                    json.dumps(flags or []),
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            rows = []
            for resource in pb.resources:
                effective, title, text = resource_text(resource)
                rows.append(
                    (
                        resource["id"],
                        pb.mrn,
                        resource["resourceType"],
                        effective,
                        title,
                        text,
                        json.dumps(resource),
                    )
                )
            conn.executemany("INSERT INTO resources VALUES (?,?,?,?,?,?,?)", rows)

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM resources")
            conn.execute("DELETE FROM bundles")

    # ---------------------------------------------------------------- reads --

    def get_bundle(self, mrn: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT bundle_json FROM bundles WHERE mrn = ?", (mrn,)).fetchone()
        return json.loads(row["bundle_json"]) if row else None

    def patients(self) -> list[PatientRow]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM bundles ORDER BY mrn").fetchall()
        return [self._patient(r) for r in rows]

    def patient(self, mrn: str) -> PatientRow | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM bundles WHERE mrn = ?", (mrn,)).fetchone()
        return self._patient(row) if row else None

    def resources(
        self, mrn: str | None = None, types: tuple[str, ...] | None = None
    ) -> list[StoredResource]:
        clauses, params = [], []
        if mrn:
            clauses.append("mrn = ?")
            params.append(mrn)
        if types:
            clauses.append(f"resource_type IN ({','.join('?' * len(types))})")
            params.extend(types)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM resources {where} ORDER BY mrn, effective_date"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._resource(r) for r in rows]

    def resource(self, resource_id: str) -> StoredResource | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM resources WHERE resource_id = ?", (resource_id,)
            ).fetchone()
        return self._resource(row) if row else None

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM bundles").fetchone()[0])

    @staticmethod
    def _patient(row: sqlite3.Row) -> PatientRow:
        return PatientRow(
            mrn=row["mrn"],
            patient_id=row["patient_id"],
            name=row["patient_name"],
            birth_date=row["birth_date"],
            gender=row["gender"],
            is_valid=bool(row["is_valid"]),
            error_count=row["error_count"],
            bundle_hash=row["content_hash"],
            flags=json.loads(row["flags_json"]),
        )

    @staticmethod
    def _resource(row: sqlite3.Row) -> StoredResource:
        return StoredResource(
            resource_id=row["resource_id"],
            mrn=row["mrn"],
            resource_type=row["resource_type"],
            effective_date=row["effective_date"],
            title=row["title"],
            text=row["text"],
            resource=json.loads(row["resource_json"]),
        )
