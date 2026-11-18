"""Pure, side-effect-free normalization functions.

Each function takes one raw value and returns the canonical value (or ``None``
when the input is empty or unusable). Functions never guess silently: when a
value cannot be interpreted they raise :class:`NormalizationError`, and the
cleaner decides how to record that in the audit log.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any, Literal

from ehr_intel.config import DateOrder
from ehr_intel.ingestion.models import EncounterClass, Gender, RecordType


class NormalizationError(ValueError):
    """Raised when a raw value is present but cannot be interpreted."""


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

# Unambiguous formats, tried in order.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y%m%d",
    "%d-%b-%Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%Y/%m/%d",
)
_TIME_FORMATS = ("%H:%M:%S", "%H:%M")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$")


def _parse_numeric(text: str, order: DateOrder) -> date | None:
    """Parse ``NN/NN/YYYY``; resolve MDY/DMY ambiguity, honouring impossible values."""
    match = _NUMERIC_DATE.match(text)
    if not match:
        return None
    first, second, year = (int(g) for g in match.groups())
    if order == "MDY":
        month, day = first, second
        if month > 12 and day <= 12:  # e.g. 27/07/1949 can only be day-first
            month, day = day, month
    else:
        day, month = first, second
        if month > 12 and day <= 12:
            month, day = day, month
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise NormalizationError(f"invalid calendar date {text!r}") from exc


def parse_date(value: str | None, order: DateOrder = "MDY") -> date | None:
    """Parse a date in any supported format. ``None`` for blank input."""
    if is_blank(value):
        return None
    assert value is not None
    text = value.strip()
    # A datetime string is acceptable: keep the date part.
    if (dt := _try_datetime(text, order)) is not None:
        return dt.date()
    if (parsed := _parse_numeric(text, order)) is not None:
        return parsed
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise NormalizationError(f"unrecognized date format {text!r}")


_HAS_TIME = re.compile(r"\d{1,2}:\d{2}")


def _try_datetime(text: str, order: DateOrder) -> datetime | None:
    """Parse ``text`` if it carries a time component, else return ``None``."""
    if not _HAS_TIME.search(text):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    date_part, _, time_part = text.rpartition(" ")
    if not date_part or ":" not in time_part:
        return None
    day = parse_date(date_part, order)
    for fmt in _TIME_FORMATS:
        try:
            t = datetime.strptime(time_part, fmt).time()
        except ValueError:
            continue
        assert day is not None
        return datetime.combine(day, t)
    raise NormalizationError(f"unrecognized time {time_part!r} in {text!r}")


def parse_datetime(value: str | None, order: DateOrder = "MDY") -> datetime | None:
    """Parse a timestamp; date-only input becomes midnight. Result is naive (local)."""
    if is_blank(value):
        return None
    assert value is not None
    text = value.strip()
    if (dt := _try_datetime(text, order)) is not None:
        # Offset-aware input is converted to UTC so all timestamps share one clock.
        return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt
    parsed = parse_date(text, order)
    assert parsed is not None
    return datetime.combine(parsed, datetime.min.time())


def validate_birth_date(dob: date | None, today: date | None = None) -> date | None:
    """Reject impossible birth dates (future, or older than 130 years)."""
    if dob is None:
        return None
    today = today or date.today()
    if dob > today:
        raise NormalizationError(f"birth date {dob.isoformat()} is in the future")
    if today.year - dob.year > 130:
        raise NormalizationError(f"birth date {dob.isoformat()} is implausibly old")
    return dob


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

# Covers FHIR codes, HL7 v2 (M/F/O/U/A/N), ISO/IEC 5218 numerics (1/2/0/9) and
# common free-text variants. "W" (woman) appears in some European exports.
_GENDER_MAP: dict[str, Gender] = {
    **dict.fromkeys(("m", "male", "man", "1"), Gender.MALE),
    **dict.fromkeys(("f", "female", "woman", "w", "2"), Gender.FEMALE),
    **dict.fromkeys(("o", "other", "a", "x", "non-binary", "nonbinary"), Gender.OTHER),
    **dict.fromkeys(("u", "unknown", "unk", "0", "9", "n", "not specified"), Gender.UNKNOWN),
}


def normalize_gender(value: str | None) -> Gender | None:
    """Map any known gender code to FHIR AdministrativeGender. ``None`` for blank."""
    if is_blank(value):
        return None
    assert value is not None
    key = value.strip().lower()
    if key not in _GENDER_MAP:
        raise NormalizationError(f"unrecognized gender code {value!r}")
    return _GENDER_MAP[key]


_MRN_PREFIX = re.compile(r"^\s*(?:mrn)?\s*[:#-]?\s*", re.IGNORECASE)


def normalize_mrn(value: str | None, digits: int = 8) -> str | None:
    """Canonicalize an MRN to zero-padded digits: ``'mrn: 123'`` -> ``'00000123'``."""
    if is_blank(value):
        return None
    assert value is not None
    core = _MRN_PREFIX.sub("", value).strip()
    if not core.isdigit():
        raise NormalizationError(f"MRN {value!r} contains non-numeric characters")
    if len(core.lstrip("0")) > digits:
        raise NormalizationError(f"MRN {value!r} exceeds {digits} digits")
    if int(core) == 0:
        raise NormalizationError(f"MRN {value!r} is all zeros")
    return core.lstrip("0").zfill(digits)


def _tidy_name_part(part: str) -> str | None:
    part = " ".join(part.split())
    if not part:
        return None
    # Only re-case names that are entirely upper/lower case; keep "McDonald" intact.
    return part.title() if part.isupper() or part.islower() else part


def parse_name(value: str | dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Return ``(given, family)`` from ``'FAMILY, GIVEN'``, ``'Given Family'`` or a dict."""
    if value is None:
        return None, None
    if isinstance(value, dict):
        given = value.get("given") or value.get("first")
        family = value.get("family") or value.get("last")
        return (
            _tidy_name_part(str(given)) if given else None,
            _tidy_name_part(str(family)) if family else None,
        )
    text = value.strip()
    if not text:
        return None, None
    if "," in text:
        family, _, given = text.partition(",")
        return _tidy_name_part(given), _tidy_name_part(family)
    parts = text.split()
    if len(parts) == 1:
        return None, _tidy_name_part(parts[0])
    return _tidy_name_part(" ".join(parts[:-1])), _tidy_name_part(parts[-1])


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_RECORD_TYPE_SYNONYMS: dict[str, RecordType] = {
    "discharge summary": RecordType.DISCHARGE_SUMMARY,
    "discharge": RecordType.DISCHARGE_SUMMARY,
    "progress note": RecordType.PROGRESS_NOTE,
    "progress": RecordType.PROGRESS_NOTE,
    "clinical note": RecordType.CLINICAL_NOTE,
    "consult": RecordType.CLINICAL_NOTE,
    "consult note": RecordType.CLINICAL_NOTE,
    "radiology report": RecordType.RADIOLOGY_REPORT,
    "imaging": RecordType.RADIOLOGY_REPORT,
    "imaging report": RecordType.RADIOLOGY_REPORT,
    "radiology": RecordType.RADIOLOGY_REPORT,
    "lab": RecordType.LAB_RESULT,
    "lab result": RecordType.LAB_RESULT,
    "laboratory": RecordType.LAB_RESULT,
    "scanned document": RecordType.SCANNED_DOCUMENT,
    "scan": RecordType.SCANNED_DOCUMENT,
}


def normalize_record_type(value: str | None) -> RecordType | None:
    if is_blank(value):
        return None
    assert value is not None
    key = " ".join(value.lower().replace("_", " ").split())
    if key in _RECORD_TYPE_SYNONYMS:
        return _RECORD_TYPE_SYNONYMS[key]
    raise NormalizationError(f"unrecognized document type {value!r}")


_ENCOUNTER_CLASS_MAP: dict[str, EncounterClass] = {
    **dict.fromkeys(("inpatient", "imp", "ip", "acute"), EncounterClass.INPATIENT),
    **dict.fromkeys(("ambulatory", "amb", "outpatient", "op", "clinic"), EncounterClass.AMBULATORY),
    **dict.fromkeys(("emergency", "emer", "ed", "er"), EncounterClass.EMERGENCY),
}


def normalize_encounter_class(value: str | None) -> EncounterClass:
    if is_blank(value):
        return EncounterClass.UNKNOWN
    assert value is not None
    return _ENCOUNTER_CLASS_MAP.get(value.strip().lower(), EncounterClass.UNKNOWN)


def normalize_text(value: str | None) -> str:
    """Collapse OCR line breaks and runs of whitespace into single spaces."""
    if value is None:
        return ""
    return " ".join(value.split())


# ---------------------------------------------------------------------------
# Lab values
# ---------------------------------------------------------------------------

_RANGE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*$")
_UPPER = re.compile(r"^\s*(?:<|<=|≤)\s*(-?\d+(?:\.\d+)?)\s*$")
_LOWER = re.compile(r"^\s*(?:>|>=|≥)\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_reference_range(value: str | None) -> tuple[float | None, float | None]:
    """Parse ``'0-100'``, ``'<5'`` or ``'>60'`` into ``(low, high)``."""
    if is_blank(value):
        return None, None
    assert value is not None
    if m := _RANGE.match(value):
        return float(m.group(1)), float(m.group(2))
    if m := _UPPER.match(value):
        return None, float(m.group(1))
    if m := _LOWER.match(value):
        return float(m.group(1)), None
    return None, None


def parse_numeric(value: str | None) -> float | None:
    if is_blank(value):
        return None
    assert value is not None
    try:
        return float(value.strip().replace(",", ""))
    except ValueError:
        return None


def interpret_value(
    value: float | None, low: float | None, high: float | None, flag: str | None
) -> Literal["H", "L", "N"] | None:
    """Derive an H/L/N interpretation, trusting the numeric range over a source flag."""
    if value is not None and (low is not None or high is not None):
        if high is not None and value > high:
            return "H"
        if low is not None and value < low:
            return "L"
        return "N"
    code = (flag or "").strip().upper()[:1]
    if code == "H":
        return "H"
    if code == "L":
        return "L"
    if code == "N":
        return "N"
    return None
