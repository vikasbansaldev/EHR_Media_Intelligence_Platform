# EHR Media Intelligence Platform: write-up

**Pipeline:** ingest JSON, CSV and scanned text → clean with a per-record audit log → FHIR R4 Bundles (validated, stored in SQLite) → Claude summaries with quality gates and a cache → FAISS semantic search → Tailwind UI. All data is synthetic. Every stage runs offline and is covered by tests (pytest, mypy `--strict`, ruff).

## Trade-offs and why

- **Never auto-merge patients.** Two MRNs with the same name and DOB are *flagged* and shown to clinicians, not merged. A wrong merge mixes two people's charts, which is worse than a duplicate.
- **Offline fallbacks for every external dependency.** The rule-based summarizer and hashing embedder keep the system usable, and the tests deterministic, without an API key or a model download. The cost is weaker semantic recall until MiniLM is installed.
- **Exact FAISS search (`IndexFlatIP`)** over approximate indexes: no recall loss and no training at this scale. Filters run *inside* the search so top-k is never short. On Python 3.15, where FAISS has no wheels yet, a NumPy backend computes the same exact cosine search.
- **R4B models for validation.** `fhir.resources` v7+ (required for Pydantic v2) has no pure-R4 package. R4B is wire-compatible for the five resources used, and I added a rules layer for what the library does not check (value-set bindings, reference integrity).
- **SQLite, one connection per operation.** Simple and thread-safe for FastAPI. It is not built for many concurrent writers.
- **Vanilla JS over React.** One file, no bundler, compiled CSS committed. React would pay off once the UI gains more views.

## What I would improve with more time

1. Run the official HL7 FHIR Validator (Java) in CI for strict 4.0.1 and US Core profile conformance. Add real terminology lookups (LOINC/SNOMED) instead of the static table.
2. Hybrid retrieval (BM25 + embeddings with reciprocal-rank fusion) and a clinical embedding model. Evaluate with a labelled query set and report recall@5 / nDCG.
3. Probabilistic patient matching (Fellegi–Sunter style) with a human-review queue, instead of exact-match linking.
4. OCR for real PDFs and images (e.g. Tesseract or a document AI service) feeding the text reader.
5. Production hardening: SMART on FHIR / OAuth2 auth, audit logging of every record access (HIPAA), PHI-safe logs, Postgres with pgvector, and async summarization jobs.
6. Clinician feedback on summaries ("accurate / missing / wrong") fed into prompt revisions and the evaluation set.

## FHIR and clinical concepts I researched

- The `DocumentReference` (narrative documents) vs `DiagnosticReport` + `Observation` (results) split, and `presentedForm` vs `content.attachment`.
- `instant` requires a timezone, but `date` does not.
- Encounter class codes come from v3 ActCode (IMP / AMB / EMER), and MRN identifiers are typed with v2-0203 `MR`.
- Bundle `fullUrl` and `urn:uuid` reference resolution, and `collection` vs `transaction` Bundles.
- LOINC document codes (e.g. 18842-5 discharge summary), UCUM units, and v3 ObservationInterpretation H/L/N.
- Why source abnormal flags should be checked against reference ranges.
- BI-RADS categories, and which findings count as critical (embolism, infarct, suspicious mass).

## How I validated AI summary quality

- **Structure:** a forced tool call against a JSON schema means output always parses into the Pydantic model. Invalid output raises an error and falls back to the rule-based summary.
- **Automated gates on every summary:** at most 200 words (one corrective retry, then a deterministic trim); all required sections present; cited resource ids exist in the Bundle. Any failure downgrades confidence to *low* and the rationale names the cause.
- **Grounding check:** every number and date in the summary must appear verbatim in the source records, which catches invented lab values.
- **Tests** inject hallucinated values, over-length output, malformed JSON and API failures, and assert each one is caught.
- **Baseline:** the extractive rule-based summarizer serves as a cannot-hallucinate reference to compare LLM output against. I spot-checked all 13 synthetic patients' summaries against their source records, covering chief concern, diagnoses and abnormal labs.
- **Safety framing:** every summary carries an "AI-generated, not a clinical decision" disclaimer plus provenance (model, prompt version, source hash).
