"""Turn :class:`RawRecord` objects into a validated, de-duplicated :class:`CleanedDataset`.

Cleaning runs in four passes, each recording what it changed in the audit log:

1. **Normalize** each record independently (dates, gender, MRN, names, text,
   lab values). Records with no clinical content or no usable patient
   identifier are rejected.
2. **Resolve patients**: group records by canonical MRN, pick each demographic
   attribute by majority vote (conflicts are audited), link MRN-less records
   by exact name + DOB match, and flag different MRNs that look like the same
   person. Patients are *never* merged automatically across MRNs - that is a
   clinical-safety decision for health-information-management staff.
3. **Build encounters** from the encounter references on records.
4. **De-duplicate** records with identical content (same patient, type, day
   and normalized text), merging their source IDs.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, TypeVar

from pydantic import ValidationError

from ehr_intel.config import Settings
from ehr_intel.ingestion import normalizers as norm
from ehr_intel.ingestion.models import (
    AuditAction,
    AuditEntry,
    CleanedDataset,
    ClinicalRecord,
    Encounter,
    EncounterClass,
    Gender,
    LabObservation,
    PatientDemographics,
    RawObservation,
    RawRecord,
    RecordType,
    RejectedRecord,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class _RejectError(Exception):
    """Internal signal: drop the current record with the given reason."""


@dataclass
class _Candidate:
    """A record after pass 1 - normalized but not yet linked or de-duplicated."""

    raw: RawRecord
    mrn: str | None = None
    given: str | None = None
    family: str | None = None
    birth_date: date | None = None
    gender: Gender | None = None
    record_type: RecordType = RecordType.SCANNED_DOCUMENT
    title: str = ""
    text: str = ""
    effective_at: datetime | None = None
    author: str | None = None
    encounter_id: str | None = None
    encounter_class: EncounterClass = EncounterClass.UNKNOWN
    encounter_start: datetime | None = None
    encounter_end: datetime | None = None
    observations: list[LabObservation] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)

    def log(
        self,
        field_name: str,
        action: AuditAction,
        reason: str,
        original: Any = None,
        new: Any = None,
    ) -> None:
        self.audit.append(
            AuditEntry(
                field=field_name,
                action=action,
                reason=reason,
                original=None if original is None else str(original),
                new=None if new is None else str(new),
                source_id=self.raw.source_id,
            )
        )

    @property
    def identity_key(self) -> tuple[str, str, date] | None:
        if self.family and self.given and self.birth_date:
            return self.family.lower(), self.given.lower(), self.birth_date
        return None


def _majority(values: list[T]) -> tuple[T, list[T]]:
    """Most common value (ties broken by first occurrence) and the distinct losers."""
    counts = Counter(values)
    best = max(counts, key=lambda v: (counts[v], -values.index(v)))
    losers = [v for v in dict.fromkeys(values) if v != best]
    return best, losers


class RecordCleaner:
    """Stateless cleaner; one instance can clean any number of datasets."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------ API --

    def clean(self, raws: Iterable[RawRecord]) -> CleanedDataset:
        dataset = CleanedDataset()
        candidates: list[_Candidate] = []
        for raw in raws:
            dataset.raw_record_count += 1
            try:
                candidates.append(self._normalize(raw))
            except _RejectError as exc:
                self._reject(dataset, raw, str(exc))

        dataset.patients = self._resolve_patients(candidates, dataset)
        linked = [c for c in candidates if c.mrn in dataset.patients]
        dataset.encounters = self._build_encounters(linked)
        dataset.records = self._deduplicate([self._to_record(c) for c in linked])
        logger.info("Cleaning complete: %s", dataset.stats())
        return dataset

    # -------------------------------------------------------- pass 1: normalize --

    def _normalize(self, raw: RawRecord) -> _Candidate:
        c = _Candidate(raw=raw)
        order = self.settings.date_order

        # --- Patient identifiers ---------------------------------------------
        if raw.mrn is not None:
            try:
                c.mrn = norm.normalize_mrn(raw.mrn, self.settings.mrn_digits)
                if c.mrn != raw.mrn:
                    c.log(
                        "mrn", AuditAction.NORMALIZED, "canonical zero-padded MRN", raw.mrn, c.mrn
                    )
            except norm.NormalizationError as exc:
                c.log("mrn", AuditAction.INVALID, str(exc), raw.mrn)

        c.given, c.family = norm.parse_name(raw.name)
        if (
            isinstance(raw.name, str)
            and raw.name.strip() != f"{c.given or ''} {c.family or ''}".strip()
        ):
            c.log(
                "name",
                AuditAction.NORMALIZED,
                "split into given/family and re-cased",
                raw.name,
                f"{c.given} | {c.family}",
            )

        try:
            parsed_dob = norm.parse_date(raw.dob, order)
            c.birth_date = norm.validate_birth_date(parsed_dob)
            if c.birth_date and raw.dob != c.birth_date.isoformat():
                c.log("birth_date", AuditAction.NORMALIZED, "ISO-8601 date", raw.dob, c.birth_date)
        except norm.NormalizationError as exc:
            c.log("birth_date", AuditAction.INVALID, str(exc), raw.dob)

        try:
            c.gender = norm.normalize_gender(raw.sex)
            if c.gender and raw.sex != c.gender.value:
                c.log(
                    "gender", AuditAction.NORMALIZED, "FHIR AdministrativeGender", raw.sex, c.gender
                )
        except norm.NormalizationError as exc:
            c.log("gender", AuditAction.INVALID, str(exc), raw.sex)

        if c.mrn is None and c.identity_key is None:
            raise _RejectError("no usable patient identifier (needs MRN, or full name + DOB)")

        # --- Document ----------------------------------------------------------
        try:
            c.record_type = norm.normalize_record_type(raw.doc_type) or RecordType.SCANNED_DOCUMENT
            if raw.doc_type is None:
                c.log(
                    "record_type",
                    AuditAction.DEFAULTED,
                    "document type missing",
                    None,
                    c.record_type,
                )
        except norm.NormalizationError as exc:
            c.log("record_type", AuditAction.DEFAULTED, str(exc), raw.doc_type, c.record_type)

        c.observations = self._normalize_observations(c, raw.observations)
        if c.observations:
            c.text = "; ".join(o.describe() for o in c.observations)
        else:
            c.text = norm.normalize_text(raw.text)
        if not c.text:
            raise _RejectError("record has no clinical text or results")

        c.title = (raw.title or "").strip()
        if not c.title:
            c.title = c.record_type.label
            c.log(
                "title",
                AuditAction.DEFAULTED,
                "title missing; derived from record type",
                None,
                c.title,
            )

        c.author = raw.author

        # --- Encounter -----------------------------------------------------------
        c.encounter_id = raw.encounter_id
        c.encounter_class = norm.normalize_encounter_class(raw.encounter_class)
        for attr, value in (
            ("encounter_start", raw.encounter_start),
            ("encounter_end", raw.encounter_end),
        ):
            try:
                setattr(c, attr, norm.parse_datetime(value, order))
            except norm.NormalizationError as exc:
                c.log(attr, AuditAction.INVALID, str(exc), value)

        # --- Effective date (with fallback) ------------------------------------------
        try:
            c.effective_at = norm.parse_datetime(raw.date, order)
            if c.effective_at and raw.date != c.effective_at.isoformat():
                c.log(
                    "effective_at",
                    AuditAction.NORMALIZED,
                    "ISO-8601 timestamp",
                    raw.date,
                    c.effective_at.isoformat(),
                )
        except norm.NormalizationError as exc:
            c.log("effective_at", AuditAction.INVALID, str(exc), raw.date)

        if c.effective_at is None:
            if c.encounter_start is not None:
                c.effective_at = c.encounter_start
                c.log(
                    "effective_at",
                    AuditAction.INFERRED,
                    "document date missing; used encounter start",
                    None,
                    c.effective_at,
                )
            else:
                c.log(
                    "effective_at",
                    AuditAction.WARNING,
                    "document date missing and no encounter date to infer from",
                )
        return c

    def _normalize_observations(
        self, c: _Candidate, raws: list[RawObservation]
    ) -> list[LabObservation]:
        result: list[LabObservation] = []
        seen: set[tuple[Any, ...]] = set()
        for raw in raws:
            if not raw.test_name:
                c.log("observation", AuditAction.INVALID, "lab row without test name", raw.value)
                continue
            key = (raw.test_name.lower(), raw.loinc, raw.value, raw.unit)
            if key in seen:
                c.log(
                    "observation",
                    AuditAction.DEDUPLICATED,
                    "identical analyte row repeated in panel",
                    raw.test_name,
                )
                continue
            seen.add(key)

            value = norm.parse_numeric(raw.value)
            low, high = norm.parse_reference_range(raw.reference_range)
            interpretation = norm.interpret_value(value, low, high, raw.flag)
            source_flag = (raw.flag or "").strip().upper()[:1] or None
            if source_flag and interpretation and source_flag != interpretation:
                c.log(
                    f"observation[{raw.test_name}].interpretation",
                    AuditAction.CONFLICT,
                    "source flag disagrees with reference range; range used",
                    source_flag,
                    interpretation,
                )
            try:
                result.append(
                    LabObservation(
                        test_name=raw.test_name.strip(),
                        loinc=raw.loinc,
                        value=value,
                        value_text=None if value is not None else raw.value,
                        unit=raw.unit,
                        reference_low=low,
                        reference_high=high,
                        reference_text=raw.reference_range,
                        interpretation=interpretation,
                    )
                )
            except ValidationError:
                c.log(
                    f"observation[{raw.test_name}].loinc",
                    AuditAction.INVALID,
                    "malformed LOINC code dropped",
                    raw.loinc,
                )
                result.append(
                    LabObservation(
                        test_name=raw.test_name.strip(),
                        value=value,
                        unit=raw.unit,
                        value_text=None if value is not None else raw.value,
                        reference_low=low,
                        reference_high=high,
                        reference_text=raw.reference_range,
                        interpretation=interpretation,
                    )
                )
        return result

    # ------------------------------------------------------ pass 2: patients --

    def _resolve_patients(
        self, candidates: list[_Candidate], dataset: CleanedDataset
    ) -> dict[str, PatientDemographics]:
        groups: dict[str, list[_Candidate]] = {}
        for c in candidates:
            if c.mrn:
                groups.setdefault(c.mrn, []).append(c)

        patients = {mrn: self._consolidate(mrn, group) for mrn, group in groups.items()}

        # Link records that have no MRN by exact (family, given, DOB) match.
        by_identity: dict[tuple[str, str, date], list[str]] = {}
        for p in patients.values():
            if p.family_name and p.given_name and p.birth_date:
                key = (p.family_name.lower(), p.given_name.lower(), p.birth_date)
                by_identity.setdefault(key, []).append(p.mrn)

        for c in candidates:
            if c.mrn:
                continue
            matches = by_identity.get(c.identity_key, []) if c.identity_key else []
            if len(matches) == 1:
                c.mrn = matches[0]
                c.log(
                    "mrn",
                    AuditAction.LINKED,
                    "MRN missing; linked by exact name + DOB match",
                    None,
                    c.mrn,
                )
                patients[c.mrn].source_ids.append(c.raw.source_id)
            else:
                why = "matches several patients" if matches else "matches no known patient"
                self._reject(dataset, c.raw, f"MRN missing and name + DOB {why}")

        # Same person registered under several MRNs: flag, never auto-merge.
        for mrns in by_identity.values():
            if len(mrns) > 1:
                for mrn in mrns:
                    others = ", ".join(m for m in mrns if m != mrn)
                    patients[mrn].audit.append(
                        AuditEntry(
                            field="mrn",
                            action=AuditAction.WARNING,
                            reason=(
                                f"possible duplicate patient: same name and DOB as MRN {others}; "
                                "not merged automatically - needs identity review"
                            ),
                        )
                    )
        return patients

    def _consolidate(self, mrn: str, group: list[_Candidate]) -> PatientDemographics:
        patient = PatientDemographics(mrn=mrn, source_ids=[c.raw.source_id for c in group])
        for attr, target in (
            ("given", "given_name"),
            ("family", "family_name"),
            ("birth_date", "birth_date"),
            ("gender", "gender"),
        ):
            values = [getattr(c, attr) for c in group if getattr(c, attr) is not None]
            if not values:
                if attr == "gender":
                    patient.audit.append(
                        AuditEntry(
                            field="gender",
                            action=AuditAction.DEFAULTED,
                            new=Gender.UNKNOWN.value,
                            reason="no record supplied a gender",
                        )
                    )
                continue
            chosen, losers = _majority(values)
            setattr(patient, target, chosen)
            for loser in losers:
                n_loser = values.count(loser)
                reason = (
                    f"conflicting values across {len(values)} records; kept majority "
                    f"value ({values.count(chosen)}x) over minority ({n_loser}x)"
                )
                patient.audit.append(
                    AuditEntry(
                        field=target,
                        action=AuditAction.CONFLICT,
                        original=str(loser),
                        new=str(chosen),
                        reason=reason,
                    )
                )
                for c in group:
                    if getattr(c, attr) == loser:
                        c.log(target, AuditAction.CONFLICT, reason, loser, chosen)
            for c in group:
                if getattr(c, attr) is None:
                    c.log(
                        target,
                        AuditAction.INFERRED,
                        "missing or invalid in this record; resolved from patient's other records",
                        None,
                        chosen,
                    )
        return patient

    # ---------------------------------------------------- pass 3: encounters --

    @staticmethod
    def _build_encounters(candidates: list[_Candidate]) -> dict[str, Encounter]:
        encounters: dict[str, Encounter] = {}
        for c in candidates:
            if not c.encounter_id:
                continue
            assert c.mrn is not None
            enc = encounters.get(c.encounter_id)
            if enc is None:
                enc = encounters[c.encounter_id] = Encounter(encounter_id=c.encounter_id, mrn=c.mrn)
            elif enc.mrn != c.mrn:
                c.log(
                    "encounter_id",
                    AuditAction.CONFLICT,
                    f"encounter belongs to MRN {enc.mrn}; reference removed",
                    c.encounter_id,
                    None,
                )
                c.encounter_id = None
                continue
            if enc.encounter_class is EncounterClass.UNKNOWN:
                enc.encounter_class = c.encounter_class
            starts = [d for d in (enc.start, c.encounter_start) if d]
            ends = [d for d in (enc.end, c.encounter_end) if d]
            enc.start = min(starts) if starts else None
            enc.end = max(ends) if ends else None

        # Encounters with no explicit period: bound them by their documents' dates.
        for enc in encounters.values():
            if enc.start is None:
                dates = [
                    c.effective_at
                    for c in candidates
                    if c.encounter_id == enc.encounter_id and c.effective_at
                ]
                enc.start = min(dates) if dates else None
        return encounters

    # ------------------------------------------------- pass 4: records + dedupe --

    @staticmethod
    def _content_hash(c: _Candidate) -> str:
        day = c.effective_at.date().isoformat() if c.effective_at else ""
        basis = f"{c.mrn}|{c.record_type}|{day}|{c.text.lower()}"
        return hashlib.sha256(basis.encode()).hexdigest()

    def _to_record(self, c: _Candidate) -> ClinicalRecord:
        assert c.mrn is not None
        digest = self._content_hash(c)
        return ClinicalRecord(
            record_id=f"{c.mrn}-{digest[:10]}",
            mrn=c.mrn,
            record_type=c.record_type,
            title=c.title,
            text=c.text,
            effective_at=c.effective_at,
            author=c.author,
            encounter_id=c.encounter_id,
            source_format=c.raw.source_format,
            source_ids=[c.raw.source_id],
            observations=c.observations,
            content_hash=digest,
            audit=c.audit,
        )

    @staticmethod
    def _deduplicate(records: list[ClinicalRecord]) -> list[ClinicalRecord]:
        kept: dict[str, ClinicalRecord] = {}
        for record in records:
            existing = kept.get(record.content_hash)
            if existing is None:
                kept[record.content_hash] = record
                continue
            dup_id = record.source_ids[0]
            existing.source_ids.append(dup_id)
            cross = (
                f" (cross-format: {existing.source_format} + {record.source_format})"
                if existing.source_format != record.source_format
                else ""
            )
            existing.audit.append(
                AuditEntry(
                    field="record",
                    action=AuditAction.DEDUPLICATED,
                    source_id=dup_id,
                    reason=f"identical content to {existing.source_ids[0]}; merged{cross}",
                )
            )
            # Fill gaps on the kept record from the duplicate.
            for attr in ("author", "encounter_id"):
                if getattr(existing, attr) is None and getattr(record, attr) is not None:
                    setattr(existing, attr, getattr(record, attr))
                    existing.audit.append(
                        AuditEntry(
                            field=attr,
                            action=AuditAction.INFERRED,
                            source_id=dup_id,
                            new=str(getattr(record, attr)),
                            reason="filled from merged duplicate",
                        )
                    )
        return sorted(
            kept.values(), key=lambda r: (r.mrn, r.effective_at or datetime.min, r.record_id)
        )

    @staticmethod
    def _reject(dataset: CleanedDataset, raw: RawRecord, reason: str) -> None:
        logger.warning("Rejected %s (%s): %s", raw.source_id, raw.source_location, reason)
        dataset.rejected.append(
            RejectedRecord(
                source_id=raw.source_id, source_location=raw.source_location, reason=reason
            )
        )
