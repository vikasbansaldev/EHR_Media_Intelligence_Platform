"""Summarization service: cache lookup -> provider -> quality gates -> cache write."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ehr_intel.config import Settings
from ehr_intel.fhir.store import BundleStore, bundle_hash
from ehr_intel.summarization.cache import SummaryCache, cache_key
from ehr_intel.summarization.context import build_context
from ehr_intel.summarization.models import ClinicalSummary, QualityCheck, SummaryContent
from ehr_intel.summarization.prompts import PROMPT_VERSION
from ehr_intel.summarization.providers import (
    ClaudeProvider,
    RuleBasedProvider,
    SummarizationError,
    SummaryProvider,
)
from ehr_intel.summarization.quality import check_quality, enforce_word_limit

logger = logging.getLogger(__name__)


class PatientNotFoundError(KeyError):
    pass


def build_provider(settings: Settings) -> SummaryProvider:
    """Pick the provider from settings: Claude when a key is configured, else rule-based."""
    choice = settings.summary_provider
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    if choice == "claude" and not key:
        raise ValueError("EHR_SUMMARY_PROVIDER=claude requires ANTHROPIC_API_KEY")
    if choice == "claude" or (choice == "auto" and key):
        assert key is not None
        return ClaudeProvider(
            api_key=key,
            model=settings.claude_model,
            max_words=settings.summary_max_words,
            timeout=settings.summary_timeout_s,
        )
    return RuleBasedProvider()


class SummarizationService:
    def __init__(
        self,
        settings: Settings,
        store: BundleStore,
        cache: SummaryCache | None = None,
        provider: SummaryProvider | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.cache = cache or SummaryCache(settings.summary_cache_path)
        self.provider = provider or build_provider(settings)
        self.fallback: SummaryProvider = RuleBasedProvider()

    def summarize(self, mrn: str, *, force: bool = False) -> ClinicalSummary:
        bundle = self.store.get_bundle(mrn)
        if bundle is None:
            raise PatientNotFoundError(mrn)
        source_hash = bundle_hash(bundle)

        provider = self.provider
        key = cache_key(mrn, source_hash, provider.name, provider.model, PROMPT_VERSION)
        if not force and (cached := self.cache.get(key)) is not None:
            logger.debug("Summary cache hit for %s", mrn)
            return cached

        ctx = build_context(bundle)
        try:
            content = provider.summarize(ctx)
        except SummarizationError as exc:
            if provider is self.fallback:
                raise
            logger.warning(
                "%s failed for MRN %s (%s); using %s fallback",
                provider.name,
                mrn,
                exc,
                self.fallback.name,
            )
            provider = self.fallback
            key = cache_key(mrn, source_hash, provider.name, provider.model, PROMPT_VERSION)
            content = provider.summarize(ctx)

        max_words = self.settings.summary_max_words
        if content.word_count > max_words:
            logger.info(
                "Trimming %s summary for %s from %d words", provider.name, mrn, content.word_count
            )
            content = enforce_word_limit(content, max_words)
        quality = check_quality(content, ctx, max_words)
        content = self._apply_quality(content, quality)

        summary = ClinicalSummary(
            **content.model_dump(),
            mrn=mrn,
            patient_name=ctx.name,
            provider=provider.name,
            model=provider.model,
            prompt_version=PROMPT_VERSION,
            source_hash=source_hash,
            source_resource_count=len(ctx.records),
            quality=quality,
        )
        self.cache.put(key, summary)
        return summary

    @staticmethod
    def _apply_quality(content: SummaryContent, quality: QualityCheck) -> SummaryContent:
        """Downgrade confidence and say exactly which automated checks failed."""
        if quality.passed:
            return content
        problems = []
        if quality.missing_sections:
            missing = ", ".join(s.replace("_", " ") for s in quality.missing_sections)
            problems.append(f"missing {missing}")
        if quality.ungrounded_values:
            problems.append("values not found in source: " + ", ".join(quality.ungrounded_values))
        if quality.unknown_resource_ids:
            problems.append("cites records not in the bundle")
        if not quality.within_word_limit:
            problems.append(f"{quality.word_count} words exceeds limit")
        reason = "; ".join(problems) or "automated quality check failed"
        rationale = f"{content.confidence_rationale} Downgraded: {reason}.".strip()
        return content.model_copy(update={"confidence": "low", "confidence_rationale": rationale})

    def summarize_all(self, *, force: bool = False) -> list[ClinicalSummary]:
        summaries = []
        for patient in self.store.patients():
            summaries.append(self.summarize(patient.mrn, force=force))
        logger.info(
            "Summarized %d patients (cache hits %d, misses %d)",
            len(summaries),
            self.cache.stats.hits,
            self.cache.stats.misses,
        )
        return summaries

    def latest(self, mrn: str) -> ClinicalSummary | None:
        """Cached summary for the patient's *current* Bundle, without generating one."""
        bundle = self.store.get_bundle(mrn)
        if bundle is None:
            return None
        return self.cache.latest_for(mrn, bundle_hash(bundle))

    @staticmethod
    def write_report(summaries: list[ClinicalSummary], out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "summaries.json"
        path.write_text(
            json.dumps([s.model_dump(mode="json") for s in summaries], indent=2), encoding="utf-8"
        )
        return path
