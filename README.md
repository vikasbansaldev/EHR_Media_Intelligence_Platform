# EHR Media Intelligence Platform

An AI pipeline that ingests messy EHR exports (JSON, lab CSVs, scanned text notes), cleans them, normalizes them to **HL7 FHIR R4**, writes a **clinical summary** per patient, and lets clinicians **search every record in plain language** from a web UI.

```
 data/raw/                Task 1                 Task 2                    Task 3               Task 4             Task 5
 ┌──────────────┐   ┌────────────────┐   ┌────────────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌─────────────┐
 │ ehr_export   │   │ readers        │   │ FhirMapper         │   │ Claude / rule-   │   │ embeddings   │   │ Tailwind +  │
 │  .json       │──▶│ normalizers    │──▶│ BundleValidator    │──▶│ based provider   │──▶│ FAISS index  │──▶│ vanilla JS  │
 │ lab_results  │   │ RecordCleaner  │   │ SQLite BundleStore │   │ quality gates    │   │ POST /search │   │ search UI   │
 │  .csv        │   │ + audit log    │   │ + validation report│   │ SQLite cache     │   │ (FastAPI)    │   │             │
 │ notes/*.txt  │   └────────────────┘   └────────────────────┘   └──────────────────┘   └──────────────┘   └─────────────┘
 └──────────────┘
```

All data in this repository is **synthetic**: names, identifiers and clinical details are fabricated by `scripts/generate_synthetic_data.py`. No real patient data is used.

---

## Quick start

Requirements: Python 3.11 to 3.15 (tested on 3.11, 3.13 and 3.15). Node is needed only if you want to rebuild the CSS; the compiled stylesheet is committed.

> **Python 3.15:** `faiss-cpu` has no 3.15 wheels yet, so it is skipped automatically on 3.15 and search uses the NumPy backend, which runs the same exact cosine search and returns the same ranking. Once FAISS publishes 3.15 wheels, `pip install -e ".[faiss]"` switches it back on with no other change.

```bash
git clone <repo-url> && cd ehr-media-intelligence
make install          # creates .venv, installs package + dev tools
make pipeline         # ingest -> FHIR -> summarize -> index   (≈1 s on the sample data)
make serve            # open http://127.0.0.1:8000
make test             # 187 tests, ~7 s
```

Without `make`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ehr-intel run-all
ehr-intel serve
pytest
```

Windows (PowerShell, no `make` needed):

```powershell
$env:PYTHONUTF8 = "1"
py -3.15 -m venv .venv                      # any Python 3.11-3.15
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\ehr-intel.exe run-all
.\.venv\Scripts\python.exe verify_local.py  # 31 requirement checks
.\.venv\Scripts\ehr-intel.exe serve        # http://127.0.0.1:8000 (use --port 8010 if XAMPP holds 8000)
```

### Optional: real LLM summaries and MiniLM embeddings

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # summaries via Claude (default model: claude-haiku-5-5)
pip install -e ".[ml]"                   # sentence-transformers, all-MiniLM-L6-v2 (pulls in torch)
ehr-intel run-all --force                # regenerate summaries, rebuild index with new embedder
```

Both upgrades are optional by design. Without a key, summaries come from a deterministic extractive summarizer. Without `sentence-transformers` (or with no network access to download the model), search uses an offline hashing embedder with clinical synonym expansion. Every component runs, and every test passes, fully offline.

---

## Using it

### CLI

| Command | What it does |
|---|---|
| `ehr-intel run-all [--force]` | All stages; `--force` regenerates summaries |
| `ehr-intel ingest` | Task 1 only; writes `var/reports/ingestion_report.json` |
| `ehr-intel fhir` | Tasks 1–2; writes `var/bundles/patient-<MRN>.json` and the validation report |
| `ehr-intel summarize [--force]` | Task 3 for every stored patient (cached) |
| `ehr-intel index` | Task 4: (re)build the vector index |
| `ehr-intel search "query" [--type T] [--date-from D] [--date-to D] [--json]` | Search from the terminal |
| `ehr-intel serve [--reload]` | API + UI |
| `ehr-intel generate-data [--seed N]` | Regenerate the synthetic dataset |

### HTTP API

Interactive docs: `http://127.0.0.1:8000/docs`.

```bash
curl -X POST 'http://127.0.0.1:8000/search?resource_type=DiagnosticReport&date_from=2025-01-01&date_to=2025-12-31' \
     -H 'Content-Type: application/json' -d '{"query": "blood clot in the lungs", "top_k": 5}'
```

| Endpoint | Purpose |
|---|---|
| `POST /search` | Body `{query, top_k=5}`. Query params `resource_type` (repeatable: `DocumentReference`, `DiagnosticReport`, `ClinicalSummary`), `date_from`, `date_to` |
| `GET /patients` | Patient list with identity flags |
| `GET /patients/{mrn}` | Demographics, AI summary, linked FHIR resources, identity flags |
| `GET /patients/{mrn}/summary?refresh=` | Summary (cached; generated on first request) |
| `GET /patients/{mrn}/bundle` | The patient's FHIR Bundle |
| `GET /resources/{id}` | One FHIR resource |
| `GET /filters`, `GET /health` | UI filter options; status of the index and providers |

### Configuration

Every setting is an environment variable with the `EHR_` prefix, or lives in a `.env` file (see [`.env.example`](.env.example)). Defaults are in [`src/ehr_intel/config.py`](src/ehr_intel/config.py). The ones you are most likely to change:

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Enables Claude summaries |
| `EHR_SUMMARY_PROVIDER` | `auto` | `auto` \| `claude` \| `rule_based` |
| `EHR_CLAUDE_MODEL` | `claude-haiku-5-5` | Any Messages-API model |
| `EHR_EMBEDDING_PROVIDER` | `auto` | `auto` \| `sentence_transformer` \| `hashing` |
| `EHR_VECTOR_BACKEND` | `auto` | `auto` (FAISS if installed, else NumPy) \| `faiss` \| `numpy` |
| `EHR_DATE_ORDER` | `MDY` | How to read ambiguous dates like `03/04/2024` |
| `EHR_SOURCE_TIMEZONE` | `UTC` | Zone for naive source timestamps (FHIR instants need an offset) |
| `EHR_MRN_SYSTEM` | placeholder OID | Your facility's MRN `Identifier.system` |
| `EHR_VAR_DIR` | `var` | All generated artifacts (DBs, index, reports) |

---

## Project layout

```
src/ehr_intel/
  config.py               typed settings (pydantic-settings)
  cli.py                  `ehr-intel` command
  ingestion/              Task 1
    readers.py              JSON export, lab CSV (rows -> panels), scanned text notes
    normalizers.py          pure functions: dates, gender, MRN, names, lab ranges
    cleaner.py              4-pass cleaner: normalize, resolve patients, encounters, dedupe
    models.py               RawRecord -> PatientDemographics / Encounter / ClinicalRecord + AuditEntry
  fhir/                   Task 2
    mapper.py               Patient, Encounter, DocumentReference, DiagnosticReport, Observation, Bundle
    validator.py            fhir.resources schema + R4 rules; JSON/Markdown report
    store.py                SQLite: bundles + flattened resources table
    terminology.py          LOINC / HL7 code systems in one reviewable place
  summarization/          Task 3
    providers.py            ClaudeProvider (forced tool call) + RuleBasedProvider
    prompts.py              system prompt, tool schema, PROMPT_VERSION
    quality.py              word limit, completeness, grounding, traceability
    cache.py                SQLite cache keyed by MRN + bundle hash + model + prompt version
  search/                 Task 4
    embeddings.py           MiniLM or offline hashing embedder
    index.py                FAISS IndexFlatIP (or NumPy fallback) with in-index filtering
    service.py              ranking, snippets, summary snippets
  api/app.py              FastAPI app; serves frontend/ at /
frontend/                 Task 5: index.html, assets/app.js, Tailwind source in src/input.css
scripts/generate_synthetic_data.py
tests/unit, tests/integration
```

---

## Design decisions by task

### Task 1: Ingestion and cleaning

- **Two-layer model.** Readers only *extract* into a loose `RawRecord`. All judgement lives in `RecordCleaner`, so a new input format is one small reader class registered by file suffix.
- **Every change is audited.** Each `ClinicalRecord` and `PatientDemographics` carries `AuditEntry` items with one of nine action types: `normalized`, `defaulted`, `inferred`, `linked`, `conflict`, `deduplicated`, `invalid`, `warning`, `rejected`. Each entry records the original value, the new value, the reason and the source ID. The flattened log goes to `var/reports/ingestion_report.json`.
- **Patient identity is resolved conservatively.**
  - Demographics conflicting within one MRN are settled by majority vote. The losing value is kept in the audit log.
  - Records with no MRN are linked only on an *exact* name + DOB match.
  - Two MRNs that share a name and DOB are **flagged and never auto-merged**. Merging patients is an irreversible clinical-safety decision that belongs to health-information-management staff. These flags flow through to the API and UI ("Confirm patient identity before acting on these records").
- **Dates.** Explicit format list (no fuzzy parsing). Ambiguous `NN/NN/YYYY` dates follow `EHR_DATE_ORDER`, but an impossible month (`27/07/1949`) forces day-first. Future or >130-year-old DOBs are rejected.
- **De-duplication** uses a hash of MRN + record type + day + whitespace-normalized text, so the same discharge summary exported as JSON *and* re-scanned as OCR text is merged. Missing fields on the kept copy (author, encounter) are filled from the duplicate.
- **Lab results** are grouped into panels. Abnormal flags are recomputed from the reference range, and a disagreeing source flag is recorded as a conflict.

### Task 2: FHIR R4

- **Mapping:**
  - Narrative documents become `DocumentReference` (LOINC document type, base64 `text/plain` attachment, `context.encounter`).
  - Imaging becomes `DiagnosticReport` (RAD, `conclusion` taken from the Impression section, `presentedForm`).
  - Lab panels become `DiagnosticReport` (LAB) with one `Observation` per analyte (`valueQuantity` in UCUM, `referenceRange`, `interpretation`).
  - `Encounter` resources exist so that `subject` / `encounter` references resolve.
- **Bundles** are `collection` type, one per patient, with deterministic UUIDv5 ids and `urn:uuid:` references. Re-running produces byte-identical output (tested).
- **Validation runs in two layers.**
  - `fhir.resources` checks each entry separately, so errors point at a resource and path.
  - A rules layer covers what that library does not enforce: required value-set bindings (it accepts `gender: "xx"`), reference resolution inside the Bundle, reference target types, and unique fullUrls that match resource ids.
  - Results go to `var/reports/fhir_validation_report.{md,json}`. Invalid Bundles are still stored, flagged, so they can be inspected.
- **Version note:** `fhir.resources` ≥ 7 (the Pydantic v2 line the brief requires) ships R5 plus `R4B` (4.3.0). R4B is structurally identical to R4 4.0.1 for every resource used here, so it is used for schema validation. A strict 4.0.1 conformance run would use the HL7 Java validator (see *Improvements*).

### Task 3: Summaries

- **Structured output by construction.** Claude is called with `tool_choice` forced to a `record_clinical_summary` tool whose JSON schema mirrors the Pydantic model, so responses always parse. Free-text JSON parsing is avoided entirely.
- **The prompt** requires grounding in the supplied records only, verbatim numbers and dates, under 200 words, no treatment advice, and the resource id cited for each media item.
- **Quality gates run on every summary**, from any provider:
  - word count (one corrective retry, then a deterministic trim of the least important content first);
  - required sections present;
  - **numeric and date grounding**: every number and ISO date in the summary must appear in the source, which catches the most dangerous hallucination class;
  - cited resource ids exist in the patient's Bundle.

  Any failure downgrades confidence to `low` and the rationale names the failed check.
- **Cache:** SQLite, keyed by `sha256(MRN | bundle hash | provider | model | prompt version)`. Changed records, a new model or an edited prompt invalidate the entry automatically. Unchanged patients never cost a second API call.
- **Disclaimer:** every summary carries a disclaimer, `ai_generated: true`, a confidence level with rationale, provenance, and the quality results.
- **Fallback:** if the API fails, the rule-based provider produces the summary, and it is cached under its own key so a later run with a working key retries the API.

### Task 4: Search

- **What gets indexed:** every `DocumentReference` and `DiagnosticReport`, plus one document per AI summary. Lab Observations are already rendered into their report's text.
- **FAISS `IndexFlatIP`** over L2-normalized vectors gives exact cosine similarity. Where FAISS has no wheel (Python 3.15), an equivalent NumPy backend (one matrix-vector product plus a mask) is used automatically. Tests check that both backends return the same ranking. Vectors are saved as `.npy`, so an index built with one backend loads with the other. That is the right choice below roughly 100k documents: no training and no recall loss. Filters are resolved against metadata and passed to FAISS as an `IDSelectorBatch`, so top-k is computed **within** the filtered set rather than post-filtered (post-filtering can return fewer than k).
- **Embedders** sit behind a protocol: `all-MiniLM-L6-v2` when available, otherwise a BLAKE2-based hashing embedder. The fallback uses unigrams, bigrams and character trigrams, plus a small lay-to-clinical synonym map ("blood clot" → embolism, "kidney" → renal/creatinine). The index manifest records which embedder built it, and a mismatch triggers a rebuild.
- **Latency:** about 1–8 ms per query on 77 documents. The test suite asserts p95 < 0.5 s and max < 2 s through the HTTP layer.

### Task 5: UI

- **Stack and real-time search:** vanilla JS and Tailwind v4, compiled to static CSS with a self-hosted font, so it works on an air-gapped network. Search is debounced, superseded requests are cancelled with `AbortController`, and the query and filters are kept in the URL so a search can be shared.
- **Result cards** show patient name and MRN, date, a record-type badge (colour encodes type), a relevance bar scaled to the best match with the numeric score, a highlighted snippet, and the patient's AI-summary snippet once per patient.
- **Drawer:** identity warnings, the disclaimer and confidence, chief concern, diagnoses, anomalies (red is reserved for out-of-range values), recent imaging and labs, and linked FHIR resources grouped by type with links to the raw JSON.
- **Accessibility:**
  - labelled controls, a skip link, and `aria-live` result announcements;
  - `/` focuses search; ↑/↓/Home/End move through results; Enter opens a result;
  - the drawer is a focus-trapped `role="dialog"`; Esc closes it and returns focus to the result;
  - reduced-motion is respected.
- **States covered:** loading skeletons, intro with example queries, no-match (with a "Clear filters" action when filters are active), service offline, and index not built.

---

## Testing

```bash
make check     # ruff + mypy --strict + pytest with coverage
python verify_local.py   # after `ehr-intel run-all`: 31 requirement checks, writes var/verification_report.md
```

187 tests at 90%+ line coverage (on Python 3.15 the 6 FAISS-only tests are skipped):

- **Normalizers:** dates, gender codes, MRNs, names, reference ranges.
- **Readers.**
- **13 cleaning edge cases:** exact and cross-format duplicates; missing date, sex, DOB and title; empty records; future DOB; conflicting DOB; MRN-less linking; duplicate-patient flagging; lab flag conflicts.
- **FHIR:** mapping, deliberately corrupted Bundles for each validator rule, store round-trips, determinism.
- **Summarization:** against a fake Anthropic client (structured call, over-length retry, invalid output, API failure fallback, hallucinated values, cache hits and invalidation).
- **Search:** relevance checks on six natural-language queries, in-index filtering, persistence.
- **API:** all endpoints, validation errors, latency budget, accessibility markers.
- **CLI:** end-to-end.

The Claude code path is exercised against a mocked client. Run `ehr-intel summarize --force` with a key to exercise the live API.

## Data and privacy

The repository contains only synthetic data. The generator documents every injected defect. `var/` (generated databases, Bundles, reports) is git-ignored. Summaries are sent to the configured LLM provider only when `ANTHROPIC_API_KEY` is set; real deployments would require a BAA with that provider.

See [WRITEUP.md](WRITEUP.md) for trade-offs, future improvements and how summary quality was validated, and [DEMO.md](DEMO.md) for a step-by-step client demo script.
