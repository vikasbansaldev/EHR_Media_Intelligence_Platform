"""Structured clinical summary schema shared by every summarization provider."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "AI-generated summary for informational purposes only. It is not a clinical decision, "
    "diagnosis or recommendation; verify all details against the source records."
)

Confidence = Literal["high", "medium", "low"]


class MediaItem(BaseModel):
    """A recent imaging study, lab panel or scanned document."""

    date: str | None = Field(default=None, description="ISO date (YYYY-MM-DD) of the record.")
    kind: str = Field(description="e.g. 'Imaging', 'Lab', 'Discharge summary'.")
    title: str
    finding: str = Field(description="One-line key finding.")
    resource_id: str | None = Field(default=None, description="Source FHIR resource id.")

    def render(self) -> str:
        when = f"{self.date} " if self.date else ""
        return f"{when}{self.kind} - {self.title}: {self.finding}"


class SummaryContent(BaseModel):
    """The clinically meaningful part of a summary (what the LLM produces)."""

    chief_concern: str = Field(description="Main presenting problem in one sentence.")
    key_diagnoses: list[str] = Field(default_factory=list, max_length=8)
    recent_media: list[MediaItem] = Field(default_factory=list, max_length=6)
    flagged_anomalies: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Abnormal results, critical findings or data-quality concerns.",
    )
    confidence: Confidence = "medium"
    confidence_rationale: str = ""

    def render(self) -> str:
        """Plain-text rendering, used for word counting, embedding and display."""
        parts = [f"Chief concern: {self.chief_concern}"]
        if self.key_diagnoses:
            parts.append("Key diagnoses: " + "; ".join(self.key_diagnoses))
        if self.recent_media:
            parts.append("Recent media: " + " | ".join(m.render() for m in self.recent_media))
        if self.flagged_anomalies:
            parts.append("Flagged anomalies: " + "; ".join(self.flagged_anomalies))
        return "\n".join(parts)

    @property
    def word_count(self) -> int:
        return len(self.render().split())


class QualityCheck(BaseModel):
    """Automated post-generation checks (see :mod:`ehr_intel.summarization.quality`)."""

    passed: bool
    word_count: int
    within_word_limit: bool
    missing_sections: list[str] = Field(default_factory=list)
    ungrounded_values: list[str] = Field(
        default_factory=list, description="Numbers/dates in the summary absent from the source."
    )
    unknown_resource_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ClinicalSummary(SummaryContent):
    """A summary plus provenance, quality results and the mandatory disclaimer."""

    mrn: str
    patient_name: str | None = None
    disclaimer: str = DISCLAIMER
    ai_generated: bool = True
    provider: str
    model: str
    prompt_version: str
    source_hash: str = Field(description="Hash of the FHIR Bundle the summary was built from.")
    source_resource_count: int = 0
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    quality: QualityCheck | None = None
    cached: bool = False
