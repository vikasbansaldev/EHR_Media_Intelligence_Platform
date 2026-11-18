"""Build a compact, deterministic text view of a patient's FHIR Bundle for summarization.

The same :class:`PatientContext` feeds the LLM prompt, the rule-based
summarizer and the grounding checks, so all three see identical facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ehr_intel.fhir.store import record_kind, resource_text

MAX_TEXT_CHARS = 1200
MAX_RECORDS = 25


@dataclass(frozen=True)
class LabResult:
    date: str | None
    test: str
    value: str
    unit: str
    reference: str | None
    interpretation: str | None

    @property
    def is_abnormal(self) -> bool:
        return self.interpretation in {"H", "L"}

    def render(self) -> str:
        ref = f" (ref {self.reference})" if self.reference else ""
        flag = {"H": " HIGH", "L": " LOW"}.get(self.interpretation or "", "")
        return f"{self.test} {self.value} {self.unit}{ref}{flag}".replace("  ", " ").strip()


@dataclass(frozen=True)
class ContextRecord:
    resource_id: str
    resource_type: str
    kind: str
    date: str | None
    title: str
    text: str
    conclusion: str | None = None


@dataclass
class PatientContext:
    mrn: str
    name: str | None
    birth_date: str | None
    gender: str | None
    encounters: list[str] = field(default_factory=list)
    records: list[ContextRecord] = field(default_factory=list)  # newest first
    labs: list[LabResult] = field(default_factory=list)  # newest first

    @property
    def resource_ids(self) -> set[str]:
        return {r.resource_id for r in self.records}

    def render(self) -> str:
        """Prompt-ready text. Records are tagged with their resource id for traceability."""
        lines = [
            f"PATIENT: {self.name or 'unknown'} | MRN {self.mrn} | DOB {self.birth_date or '?'}"
            f" | gender {self.gender or '?'}",
        ]
        if self.encounters:
            lines.append("ENCOUNTERS: " + "; ".join(self.encounters))
        lines.append("")
        lines.append(f"RECORDS (newest first, {len(self.records)} total):")
        for r in self.records[:MAX_RECORDS]:
            text = r.text if len(r.text) <= MAX_TEXT_CHARS else r.text[:MAX_TEXT_CHARS] + " [...]"
            lines.append(f"[{r.resource_id}] {r.date or 'undated'} | {r.kind} | {r.title}")
            lines.append(f"  {text}")
        if self.labs:
            lines.append("")
            lines.append("LAB RESULTS (newest first):")
            lines += [f"  {lab.date or 'undated'}: {lab.render()}" for lab in self.labs]
        return "\n".join(lines)


def _day(value: str | None) -> str | None:
    return value[:10] if value else None


def build_context(bundle: dict[str, Any]) -> PatientContext:
    resources = [e["resource"] for e in bundle.get("entry", [])]
    patient: dict[str, Any] = next((r for r in resources if r["resourceType"] == "Patient"), {})
    name = (patient.get("name") or [{}])[0].get("text")
    identifier = (patient.get("identifier") or [{}])[0].get("value", "?")
    ctx = PatientContext(
        mrn=identifier, name=name, birth_date=patient.get("birthDate"), gender=patient.get("gender")
    )

    for r in resources:
        rtype = r["resourceType"]
        if rtype == "Encounter":
            period = r.get("period", {})
            label = f"{r['class'].get('display', '?')} {_day(period.get('start')) or '?'}"
            if period.get("end"):
                label += f" to {_day(period['end'])}"
            ctx.encounters.append(label)
        elif rtype in {"DocumentReference", "DiagnosticReport"}:
            date, title, text = resource_text(r)
            ctx.records.append(
                ContextRecord(
                    resource_id=r["id"],
                    resource_type=rtype,
                    kind=record_kind(r),
                    date=_day(date),
                    title=title or record_kind(r),
                    text=" ".join((text or "").split()),
                    conclusion=r.get("conclusion"),
                )
            )
        elif rtype == "Observation":
            qty = r.get("valueQuantity", {})
            value = qty.get("value", r.get("valueString", "?"))
            ctx.labs.append(
                LabResult(
                    date=_day(r.get("effectiveDateTime")),
                    test=r.get("code", {}).get("text", "?"),
                    value=f"{value:g}" if isinstance(value, float) else str(value),
                    unit=qty.get("unit", ""),
                    reference=(r.get("referenceRange") or [{}])[0].get("text"),
                    interpretation=(r.get("interpretation") or [{}])[0]
                    .get("coding", [{}])[0]
                    .get("code"),
                )
            )

    ctx.records.sort(key=lambda x: x.date or "", reverse=True)
    ctx.labs.sort(key=lambda x: x.date or "", reverse=True)
    return ctx
