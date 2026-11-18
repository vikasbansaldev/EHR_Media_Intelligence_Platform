"""Task 3 - AI-powered clinical summaries with caching and quality gates."""

from ehr_intel.summarization.cache import SummaryCache
from ehr_intel.summarization.models import DISCLAIMER, ClinicalSummary, MediaItem, SummaryContent
from ehr_intel.summarization.providers import (
    ClaudeProvider,
    RuleBasedProvider,
    SummarizationError,
    SummaryProvider,
)
from ehr_intel.summarization.service import (
    PatientNotFoundError,
    SummarizationService,
    build_provider,
)

__all__ = [
    "DISCLAIMER",
    "ClaudeProvider",
    "ClinicalSummary",
    "MediaItem",
    "PatientNotFoundError",
    "RuleBasedProvider",
    "SummarizationError",
    "SummarizationService",
    "SummaryCache",
    "SummaryContent",
    "SummaryProvider",
    "build_provider",
]
