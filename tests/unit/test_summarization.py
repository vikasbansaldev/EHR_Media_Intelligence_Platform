"""Tests for clinical summarization: providers, quality gates, caching and fallback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from ehr_intel.config import Settings
from ehr_intel.fhir.store import BundleStore
from ehr_intel.summarization import (
    DISCLAIMER,
    ClaudeProvider,
    MediaItem,
    PatientNotFoundError,
    RuleBasedProvider,
    SummarizationError,
    SummarizationService,
    SummaryContent,
    build_provider,
)
from ehr_intel.summarization.context import PatientContext, build_context
from ehr_intel.summarization.prompts import TOOL_NAME
from ehr_intel.summarization.quality import check_quality, enforce_word_limit

MRN = "00010001"  # synthetic CHF + diabetes patient


class FakeMessages:
    """Stands in for ``anthropic.Anthropic().messages``; replays scripted tool inputs."""

    def __init__(self, *payloads: dict[str, Any] | Exception) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        payload = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        if isinstance(payload, Exception):
            raise payload
        block = SimpleNamespace(
            type="tool_use", id=f"tu_{len(self.calls)}", name=TOOL_NAME, input=payload
        )
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok"), block])


def fake_claude(*payloads: dict[str, Any] | Exception) -> tuple[ClaudeProvider, FakeMessages]:
    messages = FakeMessages(*payloads)
    provider = ClaudeProvider(
        api_key="test",
        model="claude-test",
        max_words=200,
        client=SimpleNamespace(messages=messages),
    )
    return provider, messages


def good_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chief_concern": "Progressive dyspnea on exertion and leg swelling.",
        "key_diagnoses": ["Congestive heart failure, NYHA class III", "Type 2 diabetes"],
        "recent_media": [
            {
                "date": "2025-01-06",
                "kind": "Imaging",
                "title": "Chest X-ray",
                "finding": "Cardiomegaly with small bilateral effusions.",
            }
        ],
        "flagged_anomalies": ["BNP 1439.3 pg/mL (ref 0-100) HIGH"],
        "confidence": "high",
        "confidence_rationale": "Consistent documentation.",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def store(populated_settings: Settings) -> BundleStore:
    return BundleStore(populated_settings.bundle_db_path)


@pytest.fixture
def ctx(store: BundleStore) -> PatientContext:
    bundle = store.get_bundle(MRN)
    assert bundle is not None
    return build_context(bundle)


# --- Context ----------------------------------------------------------------------------


def test_context_is_newest_first_and_tagged(ctx: PatientContext) -> None:
    assert ctx.name == "Avery Testwell"
    dates = [r.date for r in ctx.records if r.date]
    assert dates == sorted(dates, reverse=True)
    rendered = ctx.render()
    assert all(f"[{r.resource_id}]" in rendered for r in ctx.records)
    assert "BNP" in rendered and "HIGH" in rendered


# --- Rule-based provider ----------------------------------------------------------------


def test_rule_based_summary_has_required_sections(ctx: PatientContext) -> None:
    content = RuleBasedProvider().summarize(ctx)
    assert "dyspnea" in content.chief_concern.lower()
    assert any("heart failure" in d.lower() for d in content.key_diagnoses)
    assert any(m.kind == "Imaging" for m in content.recent_media)
    assert any(a.startswith("BNP") and "HIGH" in a for a in content.flagged_anomalies)
    quality = check_quality(content, ctx, 200)
    assert quality.passed, quality


def test_rule_based_is_deterministic(ctx: PatientContext) -> None:
    assert RuleBasedProvider().summarize(ctx) == RuleBasedProvider().summarize(ctx)


# --- Claude provider (mocked client) -----------------------------------------------------


def test_claude_request_forces_structured_tool_call(ctx: PatientContext) -> None:
    provider, messages = fake_claude(good_payload())
    content = provider.summarize(ctx)

    assert content.confidence == "high"
    (call,) = messages.calls
    assert call["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert call["max_tokens"] == 1024
    assert "under 200 words" in call["system"]
    assert "Avery Testwell" in call["messages"][0]["content"]


def test_claude_retries_once_when_too_long(ctx: PatientContext) -> None:
    too_long = good_payload(chief_concern="word " * 230)
    provider, messages = fake_claude(too_long, good_payload())
    content = provider.summarize(ctx)

    assert content.word_count <= 200
    assert len(messages.calls) == 2
    retry_turn = messages.calls[1]["messages"][-1]["content"][0]
    assert retry_turn["type"] == "tool_result"
    assert "exceeds the 200-word limit" in retry_turn["content"]


def test_claude_invalid_output_raises(ctx: PatientContext) -> None:
    provider, _ = fake_claude({"chief_concern": "x", "confidence": "certain"})
    with pytest.raises(SummarizationError, match="invalid summary"):
        provider.summarize(ctx)


def test_claude_api_error_wrapped(ctx: PatientContext) -> None:
    provider, _ = fake_claude(RuntimeError("503 overloaded"))
    with pytest.raises(SummarizationError, match="503 overloaded"):
        provider.summarize(ctx)


# --- Quality gates -----------------------------------------------------------------------


def test_hallucinated_value_is_detected(ctx: PatientContext) -> None:
    content = SummaryContent.model_validate(
        good_payload(flagged_anomalies=["BNP 2200 pg/mL (ref 0-100) HIGH on 2031-05-05"])
    )
    quality = check_quality(content, ctx, 200)
    assert not quality.passed
    assert set(quality.ungrounded_values) == {"2200", "2031-05-05"}


def test_unknown_resource_id_is_detected(ctx: PatientContext) -> None:
    content = SummaryContent(
        chief_concern="x",
        key_diagnoses=["y"],
        recent_media=[MediaItem(kind="Imaging", title="t", finding="f", resource_id="made-up")],
    )
    assert check_quality(content, ctx, 200).unknown_resource_ids == ["made-up"]


def test_enforce_word_limit_trims_least_important_first(ctx: PatientContext) -> None:
    content = RuleBasedProvider().summarize(ctx)
    trimmed = enforce_word_limit(content, 60)
    assert content.word_count > 60
    assert trimmed.word_count <= 60
    assert trimmed.chief_concern  # never dropped
    assert trimmed.key_diagnoses


# --- Service: cache, fallback, provenance --------------------------------------------------


def test_service_caches_by_patient_and_record_hash(
    populated_settings: Settings, store: BundleStore
) -> None:
    provider, messages = fake_claude(good_payload())
    svc = SummarizationService(populated_settings, store, provider=provider)

    first = svc.summarize(MRN)
    second = svc.summarize(MRN)
    assert len(messages.calls) == 1, "second call must be served from cache"
    assert (first.cached, second.cached) == (False, True)
    assert second.chief_concern == first.chief_concern
    assert svc.cache.stats.hits == 1

    svc.summarize(MRN, force=True)
    assert len(messages.calls) == 2

    # A new process (fresh cache object) still hits the persisted cache.
    again = SummarizationService(populated_settings, store, provider=provider).summarize(MRN)
    assert again.cached and len(messages.calls) == 2


def test_changed_records_invalidate_cache(populated_settings: Settings, store: BundleStore) -> None:
    from ehr_intel.fhir.mapper import PatientBundle
    from ehr_intel.fhir.validator import BundleValidator

    provider, messages = fake_claude(good_payload())
    svc = SummarizationService(populated_settings, store, provider=provider)
    svc.summarize(MRN)

    bundle = store.get_bundle(MRN)
    assert bundle is not None
    bundle["entry"][0]["resource"]["active"] = False  # any content change
    pb = PatientBundle(mrn=MRN, patient_id=bundle["entry"][0]["resource"]["id"], bundle=bundle)
    store.save(pb, BundleValidator().validate(bundle))

    assert not svc.summarize(MRN).cached
    assert len(messages.calls) == 2


def test_summary_carries_disclaimer_and_provenance(
    populated_settings: Settings, store: BundleStore
) -> None:
    provider, _ = fake_claude(good_payload())
    summary = SummarizationService(populated_settings, store, provider=provider).summarize(MRN)
    assert summary.disclaimer == DISCLAIMER
    assert summary.ai_generated is True
    assert (summary.provider, summary.model) == ("claude", "claude-test")
    assert summary.quality is not None and summary.quality.passed
    assert summary.source_resource_count > 0


def test_ungrounded_llm_output_downgrades_confidence(
    populated_settings: Settings, store: BundleStore
) -> None:
    provider, _ = fake_claude(good_payload(flagged_anomalies=["Troponin 9.9 ng/mL HIGH"]))
    summary = SummarizationService(populated_settings, store, provider=provider).summarize(MRN)
    assert summary.confidence == "low"
    assert "9.9" in summary.confidence_rationale


def test_api_failure_falls_back_to_rule_based(
    populated_settings: Settings, store: BundleStore
) -> None:
    provider, _ = fake_claude(RuntimeError("network down"))
    summary = SummarizationService(populated_settings, store, provider=provider).summarize(MRN)
    assert summary.provider == "rule_based"
    assert summary.chief_concern


def test_unknown_patient(populated_settings: Settings, store: BundleStore) -> None:
    svc = SummarizationService(populated_settings, store)
    with pytest.raises(PatientNotFoundError):
        svc.summarize("99999999")


def test_summarize_all_under_word_limit(populated_settings: Settings, store: BundleStore) -> None:
    summaries = SummarizationService(populated_settings, store).summarize_all()
    assert len(summaries) == len(store.patients())
    assert all(s.word_count <= populated_settings.summary_max_words for s in summaries)


# --- Provider selection --------------------------------------------------------------------


def test_build_provider_selection(settings: Settings) -> None:
    auto = settings.model_copy(update={"summary_provider": "auto"})
    assert build_provider(auto).name == "rule_based"
    keyed = auto.model_copy(update={"anthropic_api_key": SecretStr("sk-test")})
    assert build_provider(keyed).name == "claude"
    forced = settings.model_copy(update={"summary_provider": "claude"})
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_provider(forced)


def test_rule_based_diagnoses_keep_most_specific_form() -> None:
    from ehr_intel.summarization.providers import _add_unique

    items: list[str] = []
    _add_unique(items, "acute pulmonary embolism.")
    _add_unique(items, "Acute pulmonary embolism, right lower lobe segmental")
    _add_unique(items, "Acute pulmonary embolism")
    _add_unique(items, "Sepsis")
    assert items == ["Acute pulmonary embolism, right lower lobe segmental", "Sepsis"]
