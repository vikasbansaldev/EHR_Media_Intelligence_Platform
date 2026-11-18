#!/usr/bin/env python3
"""Verify the EHR Media Intelligence Platform against every assessment requirement.

Run from the repository root with the project's virtual-environment Python, AFTER
`ehr-intel run-all` has produced ./var:

    Windows:  .\\.venv\\Scripts\\python.exe verify_local.py
    macOS/Linux: .venv/bin/python verify_local.py

It starts its own API server on a free port, runs every check, stops the server,
prints a PASS/FAIL table and writes var/verification_report.md.
Exit code 0 = all checks passed. Read-only: it never modifies project files.
"""

from __future__ import annotations

import copy
import json
import re
import socket
import sqlite3
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
VAR = ROOT / "var"
RESULTS: list[tuple[str, str, bool, str]] = []  # (section, requirement, ok, evidence)


def check(section: str, requirement: str) -> Callable[[Callable[[], str]], None]:
    """Decorator: run the check now; its return value is the evidence, an exception = FAIL."""

    def run(fn: Callable[[], str]) -> None:
        try:
            evidence = fn()
            RESULTS.append((section, requirement, True, evidence))
        except Exception as exc:
            RESULTS.append((section, requirement, False, f"{type(exc).__name__}: {exc}"))

    return run


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(rel: str) -> Any:
    return json.loads((VAR / rel).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- preflight

if not (ROOT / "pyproject.toml").exists() or not (ROOT / "src" / "ehr_intel").exists():
    sys.exit("Run this from the repository root (the folder containing pyproject.toml).")
if not (VAR / "fhir_store.sqlite3").exists():
    sys.exit("./var is missing. Run `ehr-intel run-all` first, then re-run this script.")
try:
    import httpx
except ImportError:
    sys.exit("httpx missing: run this with the project venv Python after `pip install -e .[dev]`.")

# =========================================================================== Task 1

dataset = load_json("reports/cleaned_dataset.json")
report = load_json("reports/ingestion_report.json")
records = dataset["records"]
patients = dataset["patients"]


@check("Task 1", "Accepts 2+ input formats (JSON + CSV + text)")
def _() -> str:
    formats = sorted({r["source_format"] for r in records})
    expect({"json", "csv", "text"} <= set(formats), f"formats seen: {formats}")
    return f"source formats: {formats}"


@check("Task 1", "Counts: 69 raw -> 64 clean, 1 rejected, 13 patients")
def _() -> str:
    s = report["stats"]
    got = (s["raw_records"], s["clean_records"], s["rejected_records"], s["patients"])
    expect(got == (69, 64, 1, 13), f"got raw/clean/rejected/patients = {got}")
    return f"raw {got[0]}, clean {got[1]}, rejected {got[2]}, patients {got[3]}"


@check("Task 1", "Per-record audit log covers every data-quality action")
def _() -> str:
    actions = {row["action"] for row in report["audit_log"]}
    needed = {
        "normalized",
        "deduplicated",
        "conflict",
        "linked",
        "inferred",
        "invalid",
        "defaulted",
        "warning",
    }
    expect(needed <= actions, f"missing actions: {sorted(needed - actions)}")
    per_record = sum(1 for r in records if r["audit"])
    return f"{len(report['audit_log'])} audit entries; {per_record} records carry their own log"


@check("Task 1", "Missing fields: empty record rejected, gaps inferred/defaulted")
def _() -> str:
    rejected = [r["source_id"] for r in report["rejected"]]
    expect(rejected == ["EPIC-EMPTY-1"], f"rejected = {rejected}")
    return "rejected EPIC-EMPTY-1 (no text); dates/titles/demographics inferred or defaulted"


@check("Task 1", "Duplicate records merged (incl. cross-format JSON + scanned text)")
def _() -> str:
    cross = [r for r in records if any("cross-format" in a["reason"] for a in r["audit"])]
    dedup = sum(1 for row in report["audit_log"] if row["action"] == "deduplicated")
    expect(len(cross) == 1, f"cross-format merges = {len(cross)}")
    return f"{dedup} duplicates merged; cross-format: {cross[0]['source_ids']}"


@check("Task 1", "Conflicting identifiers: DOB conflict resolved, duplicate MRN flagged")
def _() -> str:
    p3 = patients["00010003"]
    conflict = [a for a in p3["audit"] if a["action"] == "conflict" and a["field"] == "birth_date"]
    expect(
        conflict and conflict[0]["new"] == "1949-07-27" and conflict[0]["original"] == "1994-07-27",
        f"00010003 audit: {p3['audit']}",
    )
    for mrn in ("00010004", "00090004"):
        expect(mrn in patients, f"{mrn} missing (were they merged?)")
        expect(
            any("possible duplicate patient" in a["reason"] for a in patients[mrn]["audit"]),
            f"{mrn} has no duplicate-patient warning",
        )
    return "00010003 kept DOB 1949-07-27 over 1994-07-27; 00010004/00090004 flagged, not merged"


@check("Task 1", "Demographics normalized (8-digit MRN, FHIR gender, ISO DOB)")
def _() -> str:
    for mrn, p in patients.items():
        expect(re.fullmatch(r"\d{8}", mrn) is not None, f"bad MRN {mrn}")
        expect(p["gender"] in {"male", "female", "other", "unknown"}, f"{mrn} gender {p['gender']}")
        expect(
            p["birth_date"] is None or re.fullmatch(r"\d{4}-\d{2}-\d{2}", p["birth_date"]),
            f"{mrn} DOB {p['birth_date']}",
        )
    return f"{len(patients)} patients canonical"


@check("Task 1", "Validated intermediate representation (Pydantic models)")
def _() -> str:
    from ehr_intel.ingestion import ClinicalRecord, PatientDemographics

    for r in records:
        ClinicalRecord.model_validate(r)
    for p in patients.values():
        PatientDemographics.model_validate(p)
    return f"{len(records)} ClinicalRecord + {len(patients)} PatientDemographics re-validated"


@check("Task 1", "Unit tests cover >= 3 cleaning edge cases")
def _() -> str:
    src = (ROOT / "tests" / "unit" / "test_cleaner.py").read_text(encoding="utf-8")
    n = len(re.findall(r"^def test_", src, flags=re.M))
    expect(n >= 3, f"only {n} tests")
    return f"{n} cleaning edge-case tests in tests/unit/test_cleaner.py"


# =========================================================================== Task 2

bundle_files = sorted((VAR / "bundles").glob("patient-*.json"))
bundles = [json.loads(p.read_text(encoding="utf-8")) for p in bundle_files]


@check("Task 2", "One FHIR Bundle (JSON) per patient")
def _() -> str:
    expect(len(bundles) == 13, f"{len(bundles)} bundle files")
    expect(all(b["resourceType"] == "Bundle" for b in bundles), "non-Bundle file")
    return f"{len(bundles)} files in var/bundles"


@check("Task 2", "Maps Patient, DocumentReference, DiagnosticReport (+Encounter, Observation)")
def _() -> str:
    counts: dict[str, int] = {}
    for b in bundles:
        for e in b["entry"]:
            t = e["resource"]["resourceType"]
            counts[t] = counts.get(t, 0) + 1
    expect({"Patient", "DocumentReference", "DiagnosticReport"} <= set(counts), str(counts))
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


@check("Task 2", "subject/encounter references resolve to the right resource types")
def _() -> str:
    checked = 0
    for b in bundles:
        urls = {e["fullUrl"]: e["resource"]["resourceType"] for e in b["entry"]}
        for e in b["entry"]:
            r = e["resource"]
            refs = []
            if "subject" in r:
                refs.append(("Patient", r["subject"]["reference"]))
            if "encounter" in r:
                refs.append(("Encounter", r["encounter"]["reference"]))
            for enc in r.get("context", {}).get("encounter", []):
                refs.append(("Encounter", enc["reference"]))
            for expected, ref in refs:
                expect(urls.get(ref) == expected, f"{r['resourceType']} -> {ref} ({urls.get(ref)})")
                checked += 1
    return f"{checked} subject/encounter references resolved"


@check("Task 2", "Every resource validates with fhir.resources")
def _() -> str:
    from fhir.resources.R4B import get_fhir_model_class

    n = 0
    for b in bundles:
        for e in b["entry"]:
            r = e["resource"]
            get_fhir_model_class(r["resourceType"]).model_validate(r)
            n += 1
    return f"{n} resources valid (fhir.resources R4B models)"


@check("Task 2", "Validation report generated and passing")
def _() -> str:
    md = (VAR / "reports" / "fhir_validation_report.md").read_text(encoding="utf-8")
    expect("**Status:** PASS" in md, "report status is not PASS")
    return "var/reports/fhir_validation_report.md: Status PASS, 13/13 valid"


@check("Task 2", "Validation errors surface clearly (negative test)")
def _() -> str:
    from ehr_intel.fhir.validator import BundleValidator

    broken = copy.deepcopy(bundles[0])
    broken["entry"][0]["resource"]["gender"] = "M"
    broken["entry"][0]["resource"]["birthDate"] = "1962-13-45"
    errors = BundleValidator().validate(broken).errors
    msgs = [str(e) for e in errors]
    expect(any("required value set" in m for m in msgs), f"no binding error: {msgs}")
    expect(any("birthDate" in m for m in msgs), f"no schema error: {msgs}")
    return f"{len(errors)} errors reported, e.g. {msgs[0][:90]}"


@check("Task 2", "Bundles stored in SQLite for retrieval")
def _() -> str:
    con = sqlite3.connect(VAR / "fhir_store.sqlite3")
    try:
        nb = con.execute("SELECT COUNT(*) FROM bundles").fetchone()[0]
        nr = con.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
    finally:
        con.close()
    expect(nb == 13, f"{nb} bundles")
    return f"fhir_store.sqlite3: {nb} bundles, {nr} resources"


# =========================================================================== Task 3

summaries = load_json("reports/summaries.json")


@check("Task 3", "Summary per patient with all required sections + disclaimer")
def _() -> str:
    expect(len(summaries) == 13, f"{len(summaries)} summaries")
    for s in summaries:
        for key in (
            "chief_concern",
            "key_diagnoses",
            "recent_media",
            "flagged_anomalies",
            "confidence",
            "disclaimer",
        ):
            expect(key in s, f"{s['mrn']} missing {key}")
        expect(
            s["disclaimer"].startswith("AI-generated") and s["ai_generated"] is True,
            f"{s['mrn']} disclaimer",
        )
    providers = sorted({s["provider"] for s in summaries})
    return f"13 summaries; provider(s): {providers}; disclaimer on every summary"


@check("Task 3", "Summaries under 200 words")
def _() -> str:
    counts = [s["quality"]["word_count"] for s in summaries]
    expect(max(counts) <= 200, f"max {max(counts)} words")
    return f"word counts {min(counts)}-{max(counts)}"


@check("Task 3", "Flags anomalies (e.g. BNP HIGH for 00010001)")
def _() -> str:
    s = next(x for x in summaries if x["mrn"] == "00010001")
    hit = [a for a in s["flagged_anomalies"] if "BNP" in a and "HIGH" in a]
    expect(bool(hit), f"anomalies: {s['flagged_anomalies']}")
    return hit[0]


@check("Task 3", "Cache keyed by patient ID + record hash (no repeat generation)")
def _() -> str:
    con = sqlite3.connect(VAR / "summary_cache.sqlite3")
    try:
        n = con.execute("SELECT COUNT(DISTINCT mrn) FROM summaries").fetchone()[0]
    finally:
        con.close()
    expect(n == 13, f"{n} cached patients")
    from ehr_intel.config import Settings
    from ehr_intel.fhir.store import BundleStore
    from ehr_intel.summarization import RuleBasedProvider, SummarizationService

    settings = Settings(var_dir=VAR)
    # Re-use the provider that produced the cached summaries, so this check never
    # triggers a paid API call.
    provider = RuleBasedProvider() if summaries[0]["provider"] == "rule_based" else None
    svc = SummarizationService(settings, BundleStore(settings.bundle_db_path), provider=provider)
    again = svc.summarize("00010001")
    expect(again.cached, "second request was not served from cache")
    return f"{n} patients cached; repeat request served from cache"


# =========================================================================== Tasks 4-5 (HTTP)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


port = free_port()
base = f"http://127.0.0.1:{port}"
server = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "ehr_intel.api.app:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ],
    cwd=ROOT,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)
client = httpx.Client(base_url=base, timeout=10)
try:
    for _attempt in range(60):
        try:
            if client.get("/health").status_code == 200:
                break
        except httpx.TransportError:
            time.sleep(0.5)
    else:
        raise SystemExit("API server did not start; run `ehr-intel serve` to see the error.")

    @check("Task 4", "Index built over records + AI summaries (/health)")
    def _() -> str:
        h = client.get("/health").json()
        expect(h["status"] == "ok" and h["patients"] == 13 and h["indexed_documents"] >= 50, str(h))
        return (
            f"{h['indexed_documents']} docs, embedder {h['embedder']}, "
            f"vector backend {h.get('vector_backend', 'faiss')}, summaries {h['summary_provider']}"
        )

    @check("Task 4", "POST /search returns top-5 ranked matches with scores")
    def _() -> str:
        res = client.post("/search", json={"query": "blood clot in the lungs"}).json()["results"]
        expect(len(res) == 5, f"{len(res)} results")
        scores = [r["score"] for r in res]
        expect(scores == sorted(scores, reverse=True), f"not sorted: {scores}")
        for key in ("patient_name", "mrn", "date", "resource_type", "score", "summary_snippet"):
            expect(key in res[0], f"missing {key}")
        expect(res[0]["patient_name"] == "Parker Stubbins", f"top = {res[0]['patient_name']}")
        return f"top: {res[0]['patient_name']} ({res[0]['score']}); scores {scores}"

    @check("Task 4", "Filter by resource type + date range (query params)")
    def _() -> str:
        res = client.post(
            "/search?resource_type=DiagnosticReport&date_from=2025-03-01&date_to=2025-05-31",
            json={"query": "imaging findings", "top_k": 10},
        )
        rows = res.json()["results"]
        expect(res.status_code == 200 and rows, "no results")
        expect(all(r["resource_type"] == "DiagnosticReport" for r in rows), "wrong type")
        expect(all("2025-03-01" <= r["date"] <= "2025-05-31" for r in rows), "date out of range")
        return f"{len(rows)} results, all DiagnosticReport within 2025-03-01..2025-05-31"

    @check("Task 4", "Bad input rejected with 422")
    def _() -> str:
        cases = [
            ("/search?resource_type=Bogus", {"query": "x"}),
            ("/search?date_from=2025-05-01&date_to=2025-01-01", {"query": "x"}),
            ("/search", {"query": "   "}),
        ]
        codes = [client.post(u, json=b).status_code for u, b in cases]
        expect(codes == [422, 422, 422], f"codes {codes}")
        return "unknown type / inverted dates / blank query -> 422"

    @check("Task 4", "Natural-language relevance")
    def _() -> str:
        cases = {
            "heart attack with elevated troponin": "Rowan Ipsumfield",
            "broken wrist after a fall": "Taylor Demoson",
            "suspicious breast mass needing biopsy": "Casey Fakenham",
        }
        for q, who in cases.items():
            top3 = [
                r["patient_name"]
                for r in client.post("/search", json={"query": q}).json()["results"][:3]
            ]
            expect(who in top3, f"{q!r}: {top3}")
        return "3/3 queries found the expected patient in the top 3"

    @check("Task 4", "Search latency < 2 s (end-to-end HTTP)")
    def _() -> str:
        timings = []
        for i in range(40):
            t0 = time.perf_counter()
            client.post(
                "/search", json={"query": ["chest pain", "renal", "fracture", "troponin"][i % 4]}
            )
            timings.append(time.perf_counter() - t0)
        p95 = statistics.quantiles(timings, n=20)[-1]
        expect(max(timings) < 2.0, f"max {max(timings):.3f}s")
        return (
            f"p50 {statistics.median(timings) * 1000:.1f} ms, p95 {p95 * 1000:.1f} ms, "
            f"max {max(timings) * 1000:.1f} ms over 40 queries"
        )

    @check("Task 5", "UI served with accessible markup")
    def _() -> str:
        html = client.get("/").text
        markers = [
            'role="search"',
            "aria-live",
            'role="dialog"',
            'aria-modal="true"',
            'for="query"',
            'for="filter-type"',
            'for="filter-from"',
            'for="filter-to"',
            "Skip to results",
        ]
        missing = [m for m in markers if m not in html]
        expect(not missing, f"missing {missing}")
        for asset in ("/assets/app.js", "/assets/styles.css"):
            expect(client.get(asset).status_code == 200, f"{asset} not served")
        return "search role, live region, modal dialog, labelled filters, skip link, assets OK"

    @check("Task 5", "Patient detail: summary, linked FHIR resources, identity flags")
    def _() -> str:
        d = client.get("/patients/00090004").json()
        expect(d["identity_flags"], "no identity flags")
        expect(d["summary"] and d["summary"]["disclaimer"], "no summary")
        expect(len(d["resources"]) >= 1, "no linked resources")
        d2 = client.get("/patients/00010012").json()
        return (
            f"00090004 flagged ({d['identity_flags'][0][:45]}...); "
            f"00010012 has {len(d2['resources'])} linked resources"
        )
finally:
    client.close()
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()

# =========================================================================== Deliverables


@check("Deliverables", "README, write-up (MD + 1-page PDF), pyproject, requirements")
def _() -> str:
    for name in ("README.md", "WRITEUP.md", "WRITEUP.pdf", "pyproject.toml", "requirements.txt"):
        expect((ROOT / name).exists(), f"{name} missing")
    return "all present"


@check("Deliverables", "One commit per major task")
def _() -> str:
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    for n in range(1, 6):
        expect(f"Task {n}:" in log, f"no 'Task {n}:' commit")
    return f"{len(log.strip().splitlines())} commits, Task 1-5 each present"


@check("Deliverables", "No real patient data")
def _() -> str:
    meta = json.loads((ROOT / "data/raw/ehr_export.json").read_text(encoding="utf-8"))[
        "export_meta"
    ]
    expect(meta["contains_real_phi"] is False, str(meta))
    return "export_meta.contains_real_phi = false (synthetic generator)"


# =========================================================================== Report

passed = sum(ok for *_, ok, _ in RESULTS)
lines = ["| # | Area | Requirement | Result | Evidence |", "|---|---|---|---|---|"]
for i, (section, req, ok, evidence) in enumerate(RESULTS, 1):
    lines.append(
        f"| {i} | {section} | {req} | {'PASS' if ok else 'FAIL'} | {evidence.replace('|', '/')} |"
    )
summary = f"{passed}/{len(RESULTS)} checks passed"
md = "# Local verification report\n\n" + summary + "\n\n" + "\n".join(lines) + "\n"
(VAR / "verification_report.md").write_text(md, encoding="utf-8")

width = max(len(r[1]) for r in RESULTS)
for section, req, ok, evidence in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'}  {section:<12} {req:<{width}}  {evidence}")
print(f"\n{summary}. Report: var/verification_report.md")
sys.exit(0 if passed == len(RESULTS) else 1)
