# Client demo script (about 8 minutes)

## Setup (before the client joins)

```powershell
cd C:\xampp\htdocs\EHR_Media_Intelligence_Platform
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\ehr-intel.exe run-all        # rebuilds everything from the raw files (~1 s)
.\.venv\Scripts\ehr-intel.exe serve --port 8000
```

- Open **http://127.0.0.1:8000** in Chrome or Edge. If port 8000 is busy (XAMPP), use `--port 8010`.
- Open a second tab with **http://127.0.0.1:8000/docs** (the API).
- Keep `var\reports\fhir_validation_report.md` and `var\verification_report.md` open in VS Code.
- Zoom the browser to 110–125% so the screen share is readable.

All patients are synthetic. Say so up front.

---

## 1. The problem (30 s)

Clinicians lose time digging through lab PDFs, imaging notes, discharge summaries and scans that sit outside structured EHR fields. This tool takes those raw exports, cleans them, converts them to the FHIR standard, summarizes each patient, and makes everything searchable in plain language.

## 2. Plain-language search (2 min)

| Type this | What to point out |
|---|---|
| `blood clot in the lungs` | Results appear as you type. Nobody wrote "blood clot": the record says *pulmonary embolism*. The search understands lay terms. The top hit is Parker Stubbins' CT angiogram and discharge summary. |
| `heart attack with elevated troponin` | Finds the NSTEMI patient's echo and troponin labs. |
| `broken wrist after a fall` | Finds the distal radius fracture X-ray. |

On each result card, point out:

- name and MRN
- date
- coloured **record-type badge** (note, imaging, lab, AI summary)
- **relevance bar and score**
- the matching sentence with search terms highlighted
- a one-line **AI summary** of the patient

Mention the speed: the result count in the corner shows a few milliseconds.

## 3. Patient view (2 min)

Click Parker Stubbins (or press ↓ then Enter). The drawer shows:

1. **The AI disclaimer and confidence.** It is not a clinical decision, and it says how sure it is and why.
2. **Chief concern, key diagnoses.**
3. **Flagged anomalies in red:** D-dimer 4.4 (ref 0.0–0.5) and the CT finding. Red is used *only* for out-of-range or critical results.
4. **Recent imaging and labs.**
5. **Linked FHIR resources.** Click one to show the raw, standards-compliant FHIR JSON. That is what makes this interoperable with Epic and other EHRs.

Press **Esc** to close. Focus returns to the same result, so the tool is fully keyboard-usable.

## 4. Safety: identity checks (1 min)

Search `breast surgery consult`.

- Casey Fakenham appears under **two MRNs**, with an amber **"Identity needs review"** label.
- Open her record: **"Confirm patient identity before acting on these records."**
- Key message: the system **never auto-merges** two patients. A wrong merge mixes two people's charts. It flags them for health-information staff.
- Note the **low-confidence** summary with its reason ("missing key diagnoses"). The AI tells you when it has too little to go on.

## 5. Filters (1 min)

1. Search `imaging findings`.
2. Set **Record type = Imaging & labs**. Only diagnostic reports remain.
3. Set **From = 2030-01-01** to show the clear "No records match" message with a one-click **Clear filters**.
4. Copy the browser URL. Searches are shareable links.

Optional: narrow the window to phone width to show it works on mobile.

## 6. Behind the scenes (1.5 min)

Show `var\reports\fhir_validation_report.md`: **13/13 FHIR bundles valid**, 160 resources (Patient, Encounter, DocumentReference, DiagnosticReport, Observation).

The raw data was deliberately messy: mixed date formats, mixed gender codes, mixed MRN formats, duplicates (including the same document arriving as JSON *and* as a scan), missing fields, and conflicting birth dates. Every fix is recorded in an audit log: `var\reports\ingestion_report.json`.

Show `var\verification_report.md`: **31/31 requirement checks passed**, plus 187 automated tests.

## Likely questions

| Question | Answer |
|---|---|
| Is this using a real AI model? | With an Anthropic API key set it uses Claude. Without one, as in this demo, it uses a built-in extractive summarizer that copies phrases verbatim, so it cannot invent facts. Both pass the same automated checks. |
| How do you stop the AI making things up? | Every number and date in a summary must appear in the source records, or the summary is downgraded to low confidence. It must also stay under 200 words and cite the records it used. |
| Does it work with Epic? | Output is HL7 FHIR R4, the standard Epic's APIs use. Each patient is a FHIR Bundle that can be loaded into a FHIR server. |
| Will it scale? | Exact vector search handles up to roughly 100k documents. Beyond that it moves to an approximate index and Postgres with pgvector. Summaries are cached, so the AI is called only when a patient's records change. |
| Real patient data? | Not in this demo. Production would add sign-in (SMART on FHIR), access audit logging and a BAA with the AI provider. |
