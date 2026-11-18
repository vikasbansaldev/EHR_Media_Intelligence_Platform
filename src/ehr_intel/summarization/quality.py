"""Automated quality gates applied to every generated summary.

These are cheap, deterministic checks that catch the most damaging LLM failure
modes before a clinician sees the output:

* **Length** - total words must not exceed the configured limit
  (:func:`enforce_word_limit` trims deterministically if the model overshoots).
* **Completeness** - the four required sections are present.
* **Grounding** - every number and ISO date in the summary must appear
  verbatim in the source context. An invented value ("BNP 1200" when the record
  says 1439.3) is the most dangerous hallucination in a clinical summary.
* **Traceability** - cited resource ids must exist in the patient's Bundle.

Any failed check downgrades the summary's confidence to ``low`` and is listed
in :class:`QualityCheck` so the UI can show it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ehr_intel.summarization.context import PatientContext
from ehr_intel.summarization.models import QualityCheck, SummaryContent

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")


def _numbers(text: str) -> set[str]:
    """Numbers in canonical form (``'1439.30'`` -> ``'1439.3'``), ignoring ISO dates."""
    stripped = _DATE.sub(" ", text)
    return {f"{float(n):g}" for n in _NUMBER.findall(stripped)}


def check_quality(content: SummaryContent, ctx: PatientContext, max_words: int) -> QualityCheck:
    rendered = content.render()
    source = ctx.render()

    missing = []
    if not content.chief_concern.strip():
        missing.append("chief_concern")
    if not content.key_diagnoses:
        missing.append("key_diagnoses")
    if not content.recent_media and any(r.resource_type == "DiagnosticReport" for r in ctx.records):
        missing.append("recent_media")

    ungrounded_dates = sorted(set(_DATE.findall(rendered)) - set(_DATE.findall(source)))
    ungrounded_numbers = sorted(_numbers(rendered) - _numbers(source))
    unknown_ids = sorted(
        {
            m.resource_id
            for m in content.recent_media
            if m.resource_id and m.resource_id not in ctx.resource_ids
        }
    )

    word_count = content.word_count
    notes = []
    if not content.flagged_anomalies:
        notes.append("no anomalies flagged")
    passed = (
        word_count <= max_words
        and not missing
        and not ungrounded_dates
        and not ungrounded_numbers
        and not unknown_ids
    )
    return QualityCheck(
        passed=passed,
        word_count=word_count,
        within_word_limit=word_count <= max_words,
        missing_sections=missing,
        ungrounded_values=ungrounded_dates + ungrounded_numbers,
        unknown_resource_ids=unknown_ids,
        notes=notes,
    )


def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit]) + "..."


def _pop_if(items: list[Any], keep: int) -> bool:
    if len(items) > keep:
        items.pop()
        return True
    return False


def _shorten_findings(c: SummaryContent, limit: int = 12) -> bool:
    changed = False
    for m in c.recent_media:
        short = _truncate_words(m.finding, limit)
        changed |= short != m.finding
        m.finding = short
    return changed


# Ordered from least to most clinically important content to give up.
_REDUCERS: tuple[Callable[[SummaryContent], bool], ...] = (
    lambda c: _pop_if(c.recent_media, 2),
    _shorten_findings,
    lambda c: _pop_if(c.flagged_anomalies, 3),
    lambda c: _pop_if(c.key_diagnoses, 3),
    lambda c: _pop_if(c.recent_media, 1),
    lambda c: _pop_if(c.key_diagnoses, 2),
    lambda c: _pop_if(c.flagged_anomalies, 2),
)


def enforce_word_limit(content: SummaryContent, max_words: int) -> SummaryContent:
    """Deterministically shrink a summary until it fits, least-important content first.

    Every section keeps at least one entry; if list reduction is not enough, all
    free-text fields are truncated with a progressively tighter word budget.
    """
    c = content.model_copy(deep=True)
    for reduce in _REDUCERS:
        while c.word_count > max_words and reduce(c):
            pass
    budget = 12
    while c.word_count > max_words and budget >= 3:
        c.chief_concern = _truncate_words(c.chief_concern, budget + 4)
        c.key_diagnoses = [_truncate_words(d, budget) for d in c.key_diagnoses]
        c.flagged_anomalies = [_truncate_words(a, budget) for a in c.flagged_anomalies]
        for m in c.recent_media:
            m.title = _truncate_words(m.title, budget)
            m.finding = _truncate_words(m.finding, budget)
        budget -= 2
    return c
