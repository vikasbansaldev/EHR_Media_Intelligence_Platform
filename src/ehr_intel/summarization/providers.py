"""Summary providers: Claude (LLM) and a deterministic rule-based fallback.

Both implement :class:`SummaryProvider`, so the service, cache and tests are
provider-agnostic. Adding OpenAI or a local model means one new class.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from pydantic import ValidationError

from ehr_intel.summarization import prompts
from ehr_intel.summarization.context import PatientContext
from ehr_intel.summarization.models import MediaItem, SummaryContent

logger = logging.getLogger(__name__)


class SummarizationError(RuntimeError):
    """The provider could not produce a valid summary."""


class SummaryProvider(Protocol):
    name: str
    model: str

    def summarize(self, ctx: PatientContext) -> SummaryContent: ...


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class ClaudeProvider:
    """Claude via the Messages API, using a forced tool call for structured output."""

    name = "claude"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_words: int = 200,
        timeout: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_words = max_words
        if client is None:
            import anthropic  # imported lazily so the package works without network access

            client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self._client: Any = client

    def summarize(self, ctx: PatientContext) -> SummaryContent:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompts.USER_TEMPLATE.format(context=ctx.render())},
        ]
        content, tool_block = self._call(messages)
        if content.word_count > self.max_words:
            logger.info(
                "Claude summary for %s was %d words; asking for a shorter one",
                ctx.mrn,
                content.word_count,
            )
            messages += [
                {"role": "assistant", "content": [tool_block]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block["id"],
                            "content": prompts.RETRY_TEMPLATE.format(
                                word_count=content.word_count, max_words=self.max_words
                            ),
                            "is_error": True,
                        }
                    ],
                },
            ]
            content, _ = self._call(messages)
        return content

    def _call(self, messages: list[dict[str, Any]]) -> tuple[SummaryContent, dict[str, Any]]:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=prompts.build_system_prompt(self.max_words),
                tools=[prompts.summary_tool_schema()],
                tool_choice={"type": "tool", "name": prompts.TOOL_NAME},
                messages=messages,
            )
        except Exception as exc:  # SDK raises many subclasses; normalize them
            raise SummarizationError(f"Claude API call failed: {exc}") from exc

        block = next((b for b in response.content if getattr(b, "type", None) == "tool_use"), None)
        if block is None:
            raise SummarizationError("Claude response contained no tool call")
        try:
            content = SummaryContent.model_validate(block.input)
        except ValidationError as exc:
            raise SummarizationError(f"Claude returned an invalid summary: {exc}") from exc
        return content, {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }


# ---------------------------------------------------------------------------
# Rule-based (offline, deterministic)
# ---------------------------------------------------------------------------

_CHIEF = re.compile(r"chief complaint:\s*([^.]+)\.", re.IGNORECASE)
_INDICATION = re.compile(r"indication:\s*([^.]+)\.", re.IGNORECASE)
_PRINCIPAL = re.compile(r"principal diagnosis:\s*([^.]+)\.", re.IGNORECASE)
_SECONDARY = re.compile(r"secondary diagnoses:\s*([^.]+)\.", re.IGNORECASE)
_CONSISTENT = re.compile(r"consistent with\s+([^.]+)\.", re.IGNORECASE)
_FOLLOW_UP = re.compile(r"follow-up of\s+([^.]+)\.", re.IGNORECASE)
_CRITICAL = re.compile(
    r"\b(suspicious|acute|infarct|embol|hemorrhage|fracture|consolidation|mass|biopsy|"
    r"bi-rads\s*[45]|effusion|ejection fraction\s*[1-3]\d%)",
    re.IGNORECASE,
)


_FINDINGS = re.compile(r"findings:\s*(.+?)(?:\s+impression:|$)", re.IGNORECASE)


def _first_sentence(text: str, max_words: int = 18) -> str:
    """First sentence that is not an all-caps heading such as 'PROGRESS NOTE.'."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s", text.strip()) if s]
    sentence = next((s for s in sentences if not s.isupper()), sentences[0] if sentences else "")
    words = sentence.split()
    return sentence if len(words) <= max_words else " ".join(words[:max_words]) + "..."


def _add_unique(items: list[str], value: str) -> None:
    """Append ``value`` unless an equal or more specific entry exists.

    "Acute pulmonary embolism" is dropped when "Acute pulmonary embolism, right lower
    lobe segmental" is already listed, and upgraded in place if it arrives first.
    """
    value = value.strip().rstrip(".")
    if not value:
        return
    value = value[0].upper() + value[1:]
    low = value.lower()
    for i, existing in enumerate(items):
        if low in existing.lower():
            return
        if existing.lower() in low:
            items[i] = value
            return
    items.append(value)


class RuleBasedProvider:
    """Extractive summarizer: pulls key phrases verbatim, so it cannot hallucinate.

    Used when no API key is configured, as a fallback when the LLM call fails,
    and as a deterministic baseline in tests.
    """

    name = "rule_based"
    model = "extractive-v1"

    def summarize(self, ctx: PatientContext) -> SummaryContent:
        texts = [r.text for r in ctx.records]

        chief = next((m.group(1) for t in texts if (m := _CHIEF.search(t))), None)
        chief = chief or next((m.group(1) for t in texts if (m := _INDICATION.search(t))), None)
        if chief is None:
            chief = _first_sentence(texts[0]) if texts else "No clinical documents on file."

        diagnoses: list[str] = []
        for text in texts:
            if m := _PRINCIPAL.search(text):
                _add_unique(diagnoses, m.group(1))
            if m := _SECONDARY.search(text):
                for part in m.group(1).split(";"):
                    _add_unique(diagnoses, part)
        for pattern in (_CONSISTENT, _FOLLOW_UP):
            for text in texts:
                if m := pattern.search(text):
                    _add_unique(diagnoses, m.group(1))

        def finding(text: str, conclusion: str | None) -> str:
            m = _FINDINGS.search(text)
            return _first_sentence(m.group(1) if m else (conclusion or text), 14)

        diagnostic = [r for r in ctx.records if r.resource_type == "DiagnosticReport"]
        # Imaging and labs are the "media" of interest; fall back to scanned documents.
        sources = diagnostic or ctx.records
        media = [
            MediaItem(
                date=r.date,
                kind=r.kind,
                title=r.title,
                resource_id=r.resource_id,
                finding=finding(r.text, r.conclusion),
            )
            for r in sources
        ][:4]

        anomalies: list[str] = []
        seen_tests: set[str] = set()
        for lab in ctx.labs:  # newest first: report the latest abnormal value per test
            if lab.is_abnormal and lab.test not in seen_tests:
                seen_tests.add(lab.test)
                anomalies.append(f"{lab.render()} ({lab.date})")
        for r in ctx.records:
            if r.kind != "Imaging":
                continue
            m = _FINDINGS.search(r.text)
            findings = m.group(1) if m else r.text
            if _CRITICAL.search(findings):
                anomalies.append(f"{r.title} {r.date}: {_first_sentence(findings, 14)}")

        documented = len(ctx.records)
        confidence = "medium" if documented >= 3 else "low"
        return SummaryContent(
            chief_concern=chief[0].upper() + chief[1:] if chief else chief,
            key_diagnoses=diagnoses[:6],
            recent_media=media,
            flagged_anomalies=anomalies[:6],
            confidence=confidence,
            confidence_rationale=(
                f"Extractive rule-based summary (no LLM) over {documented} record(s); "
                "phrases are copied from source documents."
            ),
        )
