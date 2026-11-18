"""Unit tests for pure normalization functions."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from ehr_intel.ingestion import normalizers as norm
from ehr_intel.ingestion.models import EncounterClass, Gender, RecordType


class TestParseDate:
    @pytest.mark.parametrize(
        "raw",
        [
            "1962-03-14",
            "03/14/1962",
            "3/14/1962",
            "14-Mar-1962",
            "19620314",
            "March 14, 1962",
            "1962-03-14T08:30:00",
            "14 Mar 1962",
            "1962/03/14",
        ],
    )
    def test_supported_formats_agree(self, raw: str) -> None:
        assert norm.parse_date(raw) == date(1962, 3, 14)

    def test_ambiguous_date_follows_configured_order(self) -> None:
        assert norm.parse_date("03/04/2024", "MDY") == date(2024, 3, 4)
        assert norm.parse_date("03/04/2024", "DMY") == date(2024, 4, 3)

    def test_impossible_month_forces_day_first(self) -> None:
        # 27 cannot be a month, so even in MDY mode this must be 27 July.
        assert norm.parse_date("27/07/1949", "MDY") == date(1949, 7, 27)

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_blank_is_none(self, raw: str | None) -> None:
        assert norm.parse_date(raw) is None

    @pytest.mark.parametrize("raw", ["not a date", "2024-13-45", "02/30/2024"])
    def test_garbage_raises(self, raw: str) -> None:
        with pytest.raises(norm.NormalizationError):
            norm.parse_date(raw)


class TestParseDatetime:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2025-01-06T13:00:00", datetime(2025, 1, 6, 13, 0)),
            ("01/06/2025 13:00", datetime(2025, 1, 6, 13, 0)),
            ("06-Jan-2025 13:00", datetime(2025, 1, 6, 13, 0)),
            ("2025-01-06 13:00", datetime(2025, 1, 6, 13, 0)),
            ("2025-01-06", datetime(2025, 1, 6, 0, 0)),
        ],
    )
    def test_formats(self, raw: str, expected: datetime) -> None:
        assert norm.parse_datetime(raw) == expected

    def test_offset_aware_converted_to_utc(self) -> None:
        assert norm.parse_datetime("2025-01-06T13:00:00+05:30") == datetime(2025, 1, 6, 7, 30)


class TestBirthDate:
    def test_future_dob_rejected(self) -> None:
        with pytest.raises(norm.NormalizationError, match="future"):
            norm.validate_birth_date(date(2091, 1, 1), today=date(2025, 1, 1))

    def test_implausibly_old_dob_rejected(self) -> None:
        with pytest.raises(norm.NormalizationError, match="implausibly"):
            norm.validate_birth_date(date(1850, 1, 1), today=date(2025, 1, 1))

    def test_valid_dob_passes(self) -> None:
        assert norm.validate_birth_date(date(1980, 5, 5)) == date(1980, 5, 5)


class TestGender:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("F", Gender.FEMALE),
            ("female", Gender.FEMALE),
            ("2", Gender.FEMALE),
            ("W", Gender.FEMALE),
            ("M", Gender.MALE),
            ("MALE", Gender.MALE),
            ("1", Gender.MALE),
            ("U", Gender.UNKNOWN),
            ("O", Gender.OTHER),
            (" f ", Gender.FEMALE),
        ],
    )
    def test_codes(self, raw: str, expected: Gender) -> None:
        assert norm.normalize_gender(raw) is expected

    def test_blank(self) -> None:
        assert norm.normalize_gender("") is None

    def test_unknown_code_raises(self) -> None:
        with pytest.raises(norm.NormalizationError):
            norm.normalize_gender("purple")


class TestMrn:
    @pytest.mark.parametrize(
        "raw", ["MRN-00012345", "12345", "mrn: 12345", " 00012345 ", "MRN12345", "MRN#12345"]
    )
    def test_variants_canonicalize(self, raw: str) -> None:
        assert norm.normalize_mrn(raw, digits=8) == "00012345"

    @pytest.mark.parametrize("raw", ["ABC-123", "123456789", "0000", "12-34"])
    def test_invalid(self, raw: str) -> None:
        with pytest.raises(norm.NormalizationError):
            norm.normalize_mrn(raw, digits=8)


class TestNames:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("DOE, JANE", ("Jane", "Doe")),
            ("Jane Doe", ("Jane", "Doe")),
            ("Mary Ann  Smith", ("Mary Ann", "Smith")),
            ({"given": "JANE", "family": "DOE"}, ("Jane", "Doe")),
            ("Jane McDonald", ("Jane", "McDonald")),
            ("Cher", (None, "Cher")),
            ("", (None, None)),
        ],
    )
    def test_parse(
        self, raw: str | dict[str, str], expected: tuple[str | None, str | None]
    ) -> None:
        assert norm.parse_name(raw) == expected


class TestDocuments:
    def test_record_type_synonyms(self) -> None:
        assert norm.normalize_record_type("Radiology Report") is RecordType.RADIOLOGY_REPORT
        assert norm.normalize_record_type("imaging") is RecordType.RADIOLOGY_REPORT
        assert norm.normalize_record_type("discharge_summary") is RecordType.DISCHARGE_SUMMARY

    def test_record_type_unknown_raises(self) -> None:
        with pytest.raises(norm.NormalizationError):
            norm.normalize_record_type("horoscope")

    def test_encounter_class(self) -> None:
        assert norm.normalize_encounter_class("Inpatient") is EncounterClass.INPATIENT
        assert norm.normalize_encounter_class("ED") is EncounterClass.EMERGENCY
        assert norm.normalize_encounter_class(None) is EncounterClass.UNKNOWN

    def test_text_whitespace_collapsed(self) -> None:
        assert norm.normalize_text("  A.\n   B\t\tC  ") == "A. B C"


class TestLabValues:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0-100", (0.0, 100.0)),
            ("4.0 - 5.6", (4.0, 5.6)),
            ("<5", (None, 5.0)),
            (">=60", (60.0, None)),
            ("negative", (None, None)),
            (None, (None, None)),
        ],
    )
    def test_reference_range(
        self, raw: str | None, expected: tuple[float | None, float | None]
    ) -> None:
        assert norm.parse_reference_range(raw) == expected

    @pytest.mark.parametrize(
        ("value", "low", "high", "flag", "expected"),
        [
            (150, 0, 100, None, "H"),
            (50, 0, 100, "H", "N"),
            (7.5, 12, 15.5, None, "L"),
            (None, None, None, "H", "H"),
            (None, None, None, None, None),
        ],
    )
    def test_interpretation(
        self,
        value: float | None,
        low: float | None,
        high: float | None,
        flag: str | None,
        expected: str | None,
    ) -> None:
        assert norm.interpret_value(value, low, high, flag) == expected
