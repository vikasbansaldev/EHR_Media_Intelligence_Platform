"""Prompt templates for LLM summarization.

Bump :data:`PROMPT_VERSION` whenever the prompt or tool schema changes: it is
part of the cache key, so stale summaries are regenerated automatically.
"""

from __future__ import annotations

from typing import Any

PROMPT_VERSION = "v1"
TOOL_NAME = "record_clinical_summary"

SYSTEM_PROMPT = """\
You are a clinical documentation assistant helping physicians triage a patient's \
records quickly. You summarize ONLY what the supplied records state.

Rules:
1. Ground every statement in the records. Never infer diagnoses, values, dates or \
treatments that are not written there. If something is unclear, say so briefly.
2. Copy numbers, units and dates exactly as they appear. Do not round or convert.
3. Use standard clinical abbreviations sparingly; prefer plain clinical English.
4. Total length of all fields combined must be under {max_words} words. Be terse: \
fragments are fine.
5. chief_concern: the main presenting problem, one sentence.
6. key_diagnoses: up to 6 distinct diagnoses, most significant first.
7. recent_media: up to 4 of the most recent imaging studies, lab panels or scanned \
documents, newest first, each with a one-line key finding and the resource id shown \
in square brackets in the records.
8. flagged_anomalies: abnormal lab values (with value and reference range), critical or \
suspicious imaging findings, and contradictions or gaps in the record. Empty list if none.
9. confidence: "high" if records are consistent and complete, "medium" if partially \
documented, "low" if sparse or contradictory; explain briefly in confidence_rationale.
10. You must not give treatment recommendations.

Respond only by calling the {tool_name} tool."""

USER_TEMPLATE = """\
Summarize this patient's record history.

<records>
{context}
</records>"""

RETRY_TEMPLATE = (
    "Your previous summary was {word_count} words, which exceeds the {max_words}-word "
    "limit. Call the tool again with a shorter summary: fewer media items, terser findings."
)


def summary_tool_schema() -> dict[str, Any]:
    """JSON schema for the forced tool call (mirrors :class:`SummaryContent`)."""
    return {
        "name": TOOL_NAME,
        "description": "Record a structured clinical summary of the patient's records.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chief_concern": {"type": "string"},
                "key_diagnoses": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                "recent_media": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "YYYY-MM-DD"},
                            "kind": {"type": "string"},
                            "title": {"type": "string"},
                            "finding": {"type": "string"},
                            "resource_id": {"type": "string"},
                        },
                        "required": ["kind", "title", "finding"],
                    },
                },
                "flagged_anomalies": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "confidence_rationale": {"type": "string"},
            },
            "required": [
                "chief_concern",
                "key_diagnoses",
                "recent_media",
                "flagged_anomalies",
                "confidence",
                "confidence_rationale",
            ],
        },
    }


def build_system_prompt(max_words: int) -> str:
    return SYSTEM_PROMPT.format(max_words=max_words, tool_name=TOOL_NAME)
