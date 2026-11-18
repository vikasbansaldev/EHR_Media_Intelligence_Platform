#!/usr/bin/env python3
"""Generate a deterministic, *synthetic* messy EHR export for development and tests.

No real patient data is used: every name, identifier and clinical detail is
fabricated. The output intentionally contains the data-quality problems the
ingestion pipeline must handle:

* three input formats - JSON export, lab-result CSV, plain-text scanned notes
* mixed date formats (ISO, US, day-first, ``14-Mar-2024``, ``20240314``, long form)
* mixed gender codes (``F``, ``female``, ``2``, ``W``, missing)
* mixed MRN formats (``MRN-00012345``, ``12345``, ``mrn: 12345``)
* exact duplicate records, including one duplicated across formats
* missing fields (sex, DOB, document date, title, MRN)
* conflicting identifiers (same MRN with a different DOB; same person with two MRNs)
* one unusable record (empty body) and one impossible DOB in the future
* lab rows whose source abnormal flag contradicts the reference range (TSH)

Usage::

    python scripts/generate_synthetic_data.py --out data/raw --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Synthetic patient profiles
# ---------------------------------------------------------------------------


@dataclass
class Profile:
    mrn: int
    given: str
    family: str
    dob: date
    sex: str  # "M" | "F"
    conditions: list[str]
    encounters: list[str] = field(default_factory=list)


PROFILES: list[Profile] = [
    Profile(10001, "Avery", "Testwell", date(1958, 4, 12), "F", ["chf", "t2dm"]),
    Profile(10002, "Jordan", "Samplecroft", date(1971, 11, 3), "M", ["pneumonia", "copd"]),
    Profile(10003, "Riley", "Mockford", date(1949, 7, 27), "F", ["ckd", "t2dm"]),
    Profile(10004, "Casey", "Fakenham", date(1983, 2, 18), "F", ["breast_mass"]),
    Profile(10005, "Morgan", "Synthby", date(1966, 9, 30), "M", ["stroke", "afib"]),
    Profile(10006, "Taylor", "Demoson", date(1992, 5, 6), "M", ["fracture"]),
    Profile(10007, "Quinn", "Placeholder", date(1955, 1, 21), "F", ["copd", "chf"]),
    Profile(10008, "Harper", "Lorem", date(1978, 12, 9), "F", ["anemia"]),
    Profile(10009, "Rowan", "Ipsumfield", date(1961, 3, 14), "M", ["mi", "t2dm"]),
    Profile(10010, "Emerson", "Nullbury", date(2001, 8, 25), "M", ["appendicitis"]),
    Profile(10011, "Sage", "Exampleton", date(1940, 10, 2), "F", ["sepsis", "ckd"]),
    Profile(10012, "Parker", "Stubbins", date(1987, 6, 17), "F", ["pe"]),
]

# Clinical content templates keyed by condition. Each entry provides material
# for narrative notes, an imaging study and a lab panel.
CONDITIONS: dict[str, dict[str, Any]] = {
    "chf": {
        "dx": "Congestive heart failure, NYHA class III",
        "chief": "progressive dyspnea on exertion and bilateral leg swelling",
        "imaging": (
            "Chest X-ray",
            "Cardiomegaly with small bilateral pleural effusions and "
            "pulmonary vascular congestion. No focal consolidation.",
        ),
        "lab": ("BNP", "30934-4", (850, 1900), "pg/mL", "0-100"),
    },
    "t2dm": {
        "dx": "Type 2 diabetes mellitus with hyperglycemia",
        "chief": "polyuria and elevated home glucose readings",
        "imaging": None,
        "lab": ("Hemoglobin A1c", "4548-4", (8.1, 11.4), "%", "4.0-5.6"),
    },
    "pneumonia": {
        "dx": "Community-acquired pneumonia, right lower lobe",
        "chief": "fever, productive cough and pleuritic chest pain",
        "imaging": (
            "Chest X-ray",
            "Right lower lobe airspace consolidation consistent with pneumonia. No pneumothorax.",
        ),
        "lab": ("White blood cell count", "6690-2", (13.5, 19.8), "10*3/uL", "4.5-11.0"),
    },
    "copd": {
        "dx": "Chronic obstructive pulmonary disease with acute exacerbation",
        "chief": "worsening shortness of breath and wheeze",
        "imaging": ("CT chest", "Centrilobular emphysema, upper-lobe predominant. No mass."),
        "lab": ("pCO2 arterial", "2019-8", (52, 64), "mm[Hg]", "35-45"),
    },
    "ckd": {
        "dx": "Chronic kidney disease stage 4",
        "chief": "fatigue and reduced urine output",
        "imaging": (
            "Renal ultrasound",
            "Bilateral echogenic, small kidneys consistent with "
            "chronic medical renal disease. No hydronephrosis.",
        ),
        "lab": ("Creatinine", "2160-0", (2.9, 4.6), "mg/dL", "0.6-1.2"),
    },
    "breast_mass": {
        "dx": "Palpable left breast mass, BI-RADS 4",
        "chief": "self-detected left breast lump",
        "imaging": (
            "Diagnostic mammogram",
            "1.8 cm irregular mass, left breast upper outer "
            "quadrant. BI-RADS 4 - suspicious. Ultrasound-guided biopsy recommended.",
        ),
        "lab": None,
    },
    "stroke": {
        "dx": "Acute ischemic stroke, left MCA territory",
        "chief": "sudden right-sided weakness and slurred speech",
        "imaging": (
            "MRI brain",
            "Acute infarct in the left middle cerebral artery territory "
            "on diffusion-weighted imaging. No hemorrhage.",
        ),
        "lab": ("LDL cholesterol", "13457-7", (162, 201), "mg/dL", "0-100"),
    },
    "afib": {
        "dx": "Atrial fibrillation with rapid ventricular response",
        "chief": "palpitations",
        "imaging": None,
        "lab": ("TSH", "3016-3", (0.9, 2.4), "mIU/L", "0.4-4.0"),
    },
    "fracture": {
        "dx": "Closed displaced fracture of the right distal radius",
        "chief": "right wrist pain after a fall on an outstretched hand",
        "imaging": (
            "X-ray right wrist",
            "Displaced, dorsally angulated fracture of the distal "
            "radius. No intra-articular extension.",
        ),
        "lab": None,
    },
    "anemia": {
        "dx": "Iron deficiency anemia",
        "chief": "fatigue and exertional lightheadedness",
        "imaging": None,
        "lab": ("Hemoglobin", "718-7", (7.2, 8.9), "g/dL", "12.0-15.5"),
    },
    "mi": {
        "dx": "NSTEMI (non-ST elevation myocardial infarction)",
        "chief": "substernal chest pressure radiating to the left arm",
        "imaging": (
            "Echocardiogram",
            "Hypokinesis of the inferior wall. Estimated ejection "
            "fraction 40%. No pericardial effusion.",
        ),
        "lab": ("Troponin I", "10839-9", (2.1, 6.8), "ng/mL", "0.00-0.04"),
    },
    "appendicitis": {
        "dx": "Acute appendicitis",
        "chief": "right lower quadrant abdominal pain with nausea",
        "imaging": (
            "CT abdomen and pelvis",
            "Dilated, thick-walled appendix measuring 11 mm "
            "with periappendiceal fat stranding. No abscess.",
        ),
        "lab": ("White blood cell count", "6690-2", (14.1, 17.3), "10*3/uL", "4.5-11.0"),
    },
    "sepsis": {
        "dx": "Sepsis secondary to urinary tract infection",
        "chief": "fever, confusion and hypotension",
        "imaging": None,
        "lab": ("Lactate", "2524-7", (3.4, 5.9), "mmol/L", "0.5-2.0"),
    },
    "pe": {
        "dx": "Acute pulmonary embolism, right lower lobe segmental",
        "chief": "acute pleuritic chest pain and tachycardia",
        "imaging": (
            "CT pulmonary angiogram",
            "Filling defects in right lower lobe segmental "
            "pulmonary arteries consistent with acute pulmonary embolism. No right "
            "heart strain.",
        ),
        "lab": ("D-dimer", "48065-7", (2.8, 6.1), "mg/L FEU", "0.0-0.5"),
    },
}

AUTHORS = ["Dr. A. Placeholder", "Dr. B. Fictional", "Dr. C. Synthetic", "Dr. D. Example"]

# Date renderers - each produces a different (valid) string format.
DATE_STYLES = [
    lambda d: d.strftime("%Y-%m-%d"),
    lambda d: d.strftime("%m/%d/%Y"),
    lambda d: d.strftime("%d-%b-%Y"),
    lambda d: d.strftime("%Y%m%d"),
    lambda d: d.strftime("%B %d, %Y"),
]
DATETIME_STYLES = [
    lambda d: d.strftime("%Y-%m-%dT%H:%M:%S"),
    lambda d: d.strftime("%m/%d/%Y %H:%M"),
    lambda d: d.strftime("%Y-%m-%d %H:%M"),
    lambda d: d.strftime("%d-%b-%Y %H:%M"),
]
SEX_CODES = {"F": ["F", "female", "Female", "2", "W"], "M": ["M", "male", "MALE", "1"]}


def mrn_variant(rng: random.Random, mrn: int) -> str:
    return rng.choice([f"MRN-{mrn:08d}", f"{mrn}", f"mrn: {mrn}", f" {mrn:08d} ", f"MRN{mrn}"])


def name_variant(rng: random.Random, p: Profile) -> Any:
    style = rng.randrange(3)
    if style == 0:
        return f"{p.family.upper()}, {p.given.upper()}"
    if style == 1:
        return f"{p.given} {p.family}"
    return {"given": p.given, "family": p.family}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def discharge_text(p: Profile, admit: date, discharge: date) -> str:
    primary = CONDITIONS[p.conditions[0]]
    secondary = [CONDITIONS[c]["dx"] for c in p.conditions[1:]]
    sec = f" Secondary diagnoses: {'; '.join(secondary)}." if secondary else ""
    return (
        f"DISCHARGE SUMMARY. Admitted {admit:%Y-%m-%d}, discharged {discharge:%Y-%m-%d}. "
        f"Chief complaint: {primary['chief']}. Principal diagnosis: {primary['dx']}.{sec} "
        f"Hospital course: patient was treated per protocol, monitored on telemetry and "
        f"improved clinically. Discharged home in stable condition with outpatient follow-up "
        f"in 7-10 days. Medications reconciled at discharge."
    )


def progress_text(p: Profile, rng: random.Random) -> str:
    c = CONDITIONS[rng.choice(p.conditions)]
    return (
        f"PROGRESS NOTE. Patient seen for follow-up of {c['dx'].lower()}. Reports "
        f"{rng.choice(['improvement', 'no change', 'mild worsening'])} of {c['chief']}. "
        f"Vitals reviewed. Plan: continue current management, reassess at next visit."
    )


def build(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    json_records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    notes: list[tuple[str, str]] = []
    base = datetime(2025, 1, 6, 8, 0)
    src = 0

    def next_src() -> str:
        nonlocal src
        src += 1
        return f"EPIC-{src:05d}"

    for i, p in enumerate(PROFILES):
        admit = (base + timedelta(days=17 * i + rng.randrange(5))).replace(
            hour=rng.randrange(7, 20)
        )
        discharge = admit + timedelta(days=rng.randrange(2, 7))
        enc_inpt = f"ENC-{p.mrn}-1"
        enc_opd = f"ENC-{p.mrn}-2"
        p.encounters = [enc_inpt, enc_opd]
        sex_code = rng.choice(SEX_CODES[p.sex])

        def patient_block(_p: Profile = p, _sex: str = sex_code) -> dict[str, Any]:
            return {
                "mrn": mrn_variant(rng, _p.mrn),
                "name": name_variant(rng, _p),
                "dob": rng.choice(DATE_STYLES)(_p.dob),
                "sex": _sex,
            }

        # 1) Discharge summary (JSON)
        json_records.append(
            {
                "source_id": next_src(),
                "patient": patient_block(),
                "encounter": {
                    "id": enc_inpt,
                    "class": "inpatient",
                    "start": admit.strftime("%Y-%m-%d"),
                    "end": discharge.strftime("%Y-%m-%d"),
                },
                "document": {
                    "type": "Discharge Summary",
                    "title": f"Discharge Summary - {CONDITIONS[p.conditions[0]]['dx']}",
                    "date": rng.choice(DATETIME_STYLES)(discharge),
                    "author": rng.choice(AUTHORS),
                    "text": discharge_text(p, admit.date(), discharge.date()),
                },
            }
        )

        # 2) Imaging reports (JSON or scanned text), one per condition with imaging
        for cond in p.conditions:
            img = CONDITIONS[cond]["imaging"]
            if not img:
                continue
            modality, finding = img
            when = admit + timedelta(hours=rng.randrange(2, 30))
            body = (
                f"{modality.upper()} REPORT. Indication: {CONDITIONS[cond]['chief']}. "
                f"Findings: {finding} Impression: findings consistent with "
                f"{CONDITIONS[cond]['dx'].lower()}."
            )
            if rng.random() < 0.5:
                json_records.append(
                    {
                        "source_id": next_src(),
                        "patient": patient_block(),
                        "encounter": {"id": enc_inpt, "class": "inpatient"},
                        "document": {
                            "type": "Radiology Report",
                            "title": f"{modality} report",
                            "date": rng.choice(DATETIME_STYLES)(when),
                            "author": "Dr. R. Radiologist",
                            "text": body,
                        },
                    }
                )
            else:
                notes.append(
                    (
                        f"scan_{p.mrn}_{cond}.txt",
                        render_note(
                            rng,
                            p,
                            {
                                "DATE OF SERVICE": rng.choice(DATE_STYLES)(when.date()),
                                "DOCUMENT TYPE": "Radiology Report",
                                "TITLE": f"{modality} report",
                                "ENCOUNTER": enc_inpt,
                                "AUTHOR": "Dr. R. Radiologist",
                            },
                            body,
                        ),
                    )
                )

        # 3) Lab panels (CSV): one row per analyte, grouped into a panel per draw
        for draw in range(2):
            when = admit + timedelta(hours=6 + 24 * draw)
            panel = f"{p.family.upper()}-PANEL-{draw + 1}"
            for cond in p.conditions:
                lab = CONDITIONS[cond]["lab"]
                if not lab:
                    continue
                name, loinc, (lo, hi), unit, ref = lab
                value = round(rng.uniform(lo, hi), 1)
                # Source flags: low for anemia, otherwise "H". The TSH flag is
                # deliberately wrong (value is in range) to exercise conflict handling.
                flag = "L" if cond == "anemia" else "H"
                csv_rows.append(
                    _lab_row(rng, p, when, panel, enc_inpt, name, loinc, value, unit, ref, flag)
                )
            # A routine normal analyte so not every result is abnormal.
            csv_rows.append(
                _lab_row(
                    rng,
                    p,
                    when,
                    panel,
                    enc_inpt,
                    "Sodium",
                    "2951-2",
                    float(rng.randrange(136, 144)),
                    "mmol/L",
                    "135-145",
                    "",
                )
            )

        # 4) Outpatient progress note (scanned text)
        visit = discharge + timedelta(days=rng.randrange(8, 14))
        notes.append(
            (
                f"progress_{p.mrn}.txt",
                render_note(
                    rng,
                    p,
                    {
                        "DATE OF SERVICE": rng.choice(DATE_STYLES)(visit.date()),
                        "DOCUMENT TYPE": "Progress Note",
                        "ENCOUNTER": enc_opd,
                        "AUTHOR": rng.choice(AUTHORS),
                    },
                    progress_text(p, rng),
                ),
            )
        )

    inject_quality_issues(rng, json_records, csv_rows, notes)
    return json_records, csv_rows, notes


def _lab_row(
    rng: random.Random,
    p: Profile,
    when: datetime,
    panel: str,
    enc: str,
    test: str,
    loinc: str,
    value: float,
    unit: str,
    ref: str,
    flag: str,
) -> dict[str, str]:
    return {
        "mrn": mrn_variant(rng, p.mrn),
        "patient_name": f"{p.family}, {p.given}",
        "dob": rng.choice(DATE_STYLES)(p.dob),
        "sex": rng.choice(SEX_CODES[p.sex]),
        "panel_id": panel,
        "encounter_id": enc,
        "collected_at": rng.choice(DATETIME_STYLES)(when),
        "test_name": test,
        "loinc": loinc,
        "value": f"{value:g}",
        "unit": unit,
        "reference_range": ref,
        "flag": flag,
    }


def render_note(rng: random.Random, p: Profile, headers: dict[str, str], body: str) -> str:
    """Render a 'scanned' note: inconsistent header casing, OCR-ish spacing."""
    lines = [
        f"{rng.choice(['PATIENT', 'Patient', 'patient name'])}:  {p.given} {p.family}",
        f"MRN:{rng.choice([' ', '   '])}{mrn_variant(rng, p.mrn)}",
        f"DOB: {rng.choice(DATE_STYLES)(p.dob)}",
        f"Sex: {rng.choice(SEX_CODES[p.sex])}",
    ]
    lines += [f"{k}: {v}" for k, v in headers.items()]
    wrapped = body.replace(". ", ".\n  ")  # simulate OCR line breaks
    return "\n".join(lines) + "\n----------------------------------------\n  " + wrapped + "\n"


def inject_quality_issues(
    rng: random.Random,
    records: list[dict[str, Any]],
    rows: list[dict[str, str]],
    notes: list[tuple[str, str]],
) -> None:
    """Deliberately corrupt a copy of the data in documented ways."""
    # Exact duplicates (re-exported records).
    for idx in (0, 3, 7):
        dup = json.loads(json.dumps(records[idx]))
        dup["source_id"] = records[idx]["source_id"] + "-DUP"
        records.append(dup)

    # Missing sex and missing DOB (recoverable from the patient's other records).
    records[1]["patient"].pop("sex", None)
    records[2]["patient"]["dob"] = ""

    # Missing document date -> fall back to encounter start.
    records[4]["document"].pop("date", None)

    # Missing title -> derived from document type.
    records[5]["document"].pop("title", None)

    # Missing MRN, but name + DOB match a known patient -> linked.
    p = PROFILES[1]
    records.append(
        {
            "source_id": "EPIC-ORPHAN-1",
            "patient": {
                "name": f"{p.given} {p.family}",
                "dob": p.dob.strftime("%m/%d/%Y"),
                "sex": "M",
            },
            "encounter": {"id": f"ENC-{p.mrn}-2", "class": "ambulatory"},
            "document": {
                "type": "Clinical Note",
                "title": "Pulmonary clinic note",
                "date": "2025-02-20 10:15",
                "author": "Dr. B. Fictional",
                "text": "PULMONARY CLINIC NOTE. Follow-up after pneumonia. Lungs clear on "
                "auscultation. Repeat chest imaging in 6 weeks to document resolution.",
            },
        }
    )

    # Same MRN, conflicting DOB (data-entry error) -> majority value wins, conflict audited.
    p = PROFILES[2]
    records.append(
        {
            "source_id": "EPIC-CONFLICT-1",
            "patient": {
                "mrn": f"MRN-{p.mrn:08d}",
                "name": f"{p.given} {p.family}",
                "dob": "1994-07-27",
                "sex": "F",
            },
            "encounter": {"id": f"ENC-{p.mrn}-2", "class": "ambulatory"},
            "document": {
                "type": "Clinical Note",
                "title": "Nephrology clinic note",
                "date": "03/02/2025 09:00",
                "author": "Dr. C. Synthetic",
                "text": "NEPHROLOGY NOTE. eGFR trending down. Discussed renal replacement "
                "therapy options. Avoid nephrotoxic agents.",
            },
        }
    )

    # Same person registered under a second MRN -> flagged as a possible duplicate patient.
    p = PROFILES[3]
    records.append(
        {
            "source_id": "EPIC-ALTMRN-1",
            "patient": {
                "mrn": "MRN-00090004",
                "name": f"{p.family}, {p.given}",
                "dob": p.dob.isoformat(),
                "sex": "female",
            },
            "encounter": {"id": "ENC-90004-1", "class": "ambulatory"},
            "document": {
                "type": "Clinical Note",
                "title": "Breast surgery consult",
                "date": "2025-02-03T11:00:00",
                "author": "Dr. D. Example",
                "text": "BREAST SURGERY CONSULT. Reviewed mammogram showing BI-RADS 4 mass. "
                "Ultrasound-guided core biopsy scheduled.",
            },
        }
    )

    # Unusable record: no clinical text -> rejected.
    records.append(
        {
            "source_id": "EPIC-EMPTY-1",
            "patient": {"mrn": "10006", "name": "Taylor Demoson", "dob": "05/06/1992", "sex": "M"},
            "document": {"type": "Scanned Document", "date": "2025-03-01", "text": "   "},
        }
    )

    # Impossible DOB in the future on one record (the patient's other records are fine).
    records[9]["patient"]["dob"] = "2091-01-01"

    # Cross-format duplicate: a JSON discharge summary also arrives as a scanned note.
    src = records[0]
    p0 = PROFILES[0]
    notes.append(
        (
            f"rescan_{p0.mrn}_discharge.txt",
            render_note(
                rng,
                p0,
                {
                    "DATE OF SERVICE": src["document"]["date"],
                    "DOCUMENT TYPE": "Discharge Summary",
                    "TITLE": src["document"]["title"],
                    "ENCOUNTER": src["encounter"]["id"],
                    "AUTHOR": src["document"]["author"],
                },
                src["document"]["text"],
            ),
        )
    )

    # Duplicate CSV row (same analyte exported twice).
    rows.append(dict(rows[0]))


def write(out: Path, seed: int) -> None:
    records, rows, notes = build(seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "notes").mkdir(exist_ok=True)
    for stale in (out / "notes").glob("*.txt"):
        stale.unlink()

    export = {
        "export_meta": {
            "system": "SIMULATED-EHR",
            "generated_by": "generate_synthetic_data.py",
            "seed": seed,
            "contains_real_phi": False,
        },
        "records": records,
    }
    # newline="\n": identical bytes on every OS (Windows would otherwise write CRLF).
    (out / "ehr_export.json").write_text(
        json.dumps(export, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    with (out / "lab_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    for name, text in notes:
        (out / "notes" / name).write_text(text, encoding="utf-8", newline="\n")

    print(f"Wrote {len(records)} JSON records, {len(rows)} lab rows, {len(notes)} notes -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write(args.out, args.seed)


if __name__ == "__main__":
    main()
