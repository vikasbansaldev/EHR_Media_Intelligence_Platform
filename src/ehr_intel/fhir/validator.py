"""Validate Bundles against FHIR R4 and produce a human-readable report.

Two layers run on every Bundle:

1. **Schema** - ``fhir.resources`` (R4B models, structurally identical to R4 for
   the resources used) checks cardinality, data types and primitive formats.
   Each entry is validated on its own so errors point at a specific resource.
2. **Rules** - checks the library does not perform: required value-set
   bindings, reference integrity inside the Bundle (every ``urn:uuid`` resolves,
   ``subject`` points at a Patient, ``encounter`` at an Encounter), unique
   fullUrls and fullUrl/id agreement.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from fhir.resources.R4B import get_fhir_model_class
from pydantic import BaseModel, Field, ValidationError

from ehr_intel.fhir.terminology import REQUIRED_BINDINGS

Resource = dict[str, Any]

# Reference fields whose target type is fixed for our profiles.
_TYPED_REFERENCES: dict[str, str] = {"subject": "Patient", "encounter": "Encounter"}
FHIR_VERSION = "4.0.1 (validated with fhir.resources R4B models)"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    severity: Severity
    layer: str = Field(description="'schema' or 'rules'")
    resource_type: str
    resource_id: str | None
    path: str
    message: str

    def __str__(self) -> str:
        where = f"{self.resource_type}/{self.resource_id or '?'}"
        return f"{self.severity.upper():7} {where} {self.path}: {self.message}"


class BundleValidationResult(BaseModel):
    bundle_id: str
    mrn: str | None = None
    resource_counts: dict[str, int] = Field(default_factory=dict)
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _walk_references(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(path, reference)`` for every Reference.reference in a resource."""
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            if key == "reference" and isinstance(value, str):
                yield path, value
            else:
                yield from _walk_references(value, child)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _walk_references(item, f"{path}[{i}]")


def _pydantic_issues(
    exc: ValidationError, resource_type: str, resource_id: str | None
) -> list[ValidationIssue]:
    return [
        ValidationIssue(
            severity=Severity.ERROR,
            layer="schema",
            resource_type=resource_type,
            resource_id=resource_id,
            path=".".join(str(p) for p in err["loc"]) or "(root)",
            message=err["msg"],
        )
        for err in exc.errors()
    ]


class BundleValidator:
    def validate(self, bundle: Resource, mrn: str | None = None) -> BundleValidationResult:
        result = BundleValidationResult(bundle_id=str(bundle.get("id", "?")), mrn=mrn)
        entries = bundle.get("entry") or []
        result.resource_counts = dict(
            Counter(e.get("resource", {}).get("resourceType", "?") for e in entries)
        )

        # Bundle shell (validated without entries so entry errors are reported per resource).
        shell = {k: v for k, v in bundle.items() if k != "entry"}
        try:
            get_fhir_model_class("Bundle").model_validate(shell)
        except ValidationError as exc:
            result.issues += _pydantic_issues(exc, "Bundle", shell.get("id"))
        result.issues += self._check_bindings(bundle)

        full_urls: dict[str, str] = {}
        for index, entry in enumerate(entries):
            resource = entry.get("resource") or {}
            rtype = resource.get("resourceType", "?")
            rid = resource.get("id")
            result.issues += self._schema(resource, rtype, rid)
            result.issues += self._check_bindings(resource)

            full_url = entry.get("fullUrl")
            if not full_url:
                result.issues.append(
                    self._rule(rtype, rid, f"entry[{index}].fullUrl", "entry has no fullUrl")
                )
                continue
            if full_url in full_urls:
                result.issues.append(
                    self._rule(
                        rtype, rid, f"entry[{index}].fullUrl", f"duplicate fullUrl {full_url}"
                    )
                )
            full_urls[full_url] = rtype
            if rid and full_url.startswith("urn:uuid:") and full_url != f"urn:uuid:{rid}":
                result.issues.append(
                    self._rule(
                        rtype, rid, f"entry[{index}].fullUrl", "fullUrl does not match resource id"
                    )
                )

        for entry in entries:
            resource = entry.get("resource") or {}
            result.issues += self._check_references(resource, full_urls)
        return result

    # ---------------------------------------------------------------- layers --

    @staticmethod
    def _schema(resource: Resource, rtype: str, rid: str | None) -> list[ValidationIssue]:
        try:
            model = get_fhir_model_class(rtype)
        except (KeyError, ValueError):
            return [
                ValidationIssue(
                    severity=Severity.ERROR,
                    layer="schema",
                    resource_type=rtype,
                    resource_id=rid,
                    path="resourceType",
                    message=f"unknown FHIR resource type {rtype!r}",
                )
            ]
        try:
            model.model_validate(resource)
        except ValidationError as exc:
            return _pydantic_issues(exc, rtype, rid)
        return []

    @staticmethod
    def _check_bindings(resource: Resource) -> list[ValidationIssue]:
        rtype = resource.get("resourceType", "?")
        issues = []
        for (bound_type, element), allowed in REQUIRED_BINDINGS.items():
            if bound_type != rtype or element not in resource:
                continue
            value = resource[element]
            if value not in allowed:
                issues.append(
                    BundleValidator._rule(
                        rtype,
                        resource.get("id"),
                        element,
                        f"{value!r} is not in the required value set "
                        f"({', '.join(sorted(allowed))})",
                    )
                )
        return issues

    @staticmethod
    def _check_references(resource: Resource, full_urls: dict[str, str]) -> list[ValidationIssue]:
        rtype = resource.get("resourceType", "?")
        rid = resource.get("id")
        issues = []
        for path, ref in _walk_references(resource):
            if ref.startswith("urn:uuid:"):
                target = full_urls.get(ref)
                if target is None:
                    issues.append(
                        BundleValidator._rule(rtype, rid, path, f"unresolved reference {ref}")
                    )
                    continue
            else:
                target = ref.split("/")[0] if "/" in ref else None
            # Element name owning the Reference, e.g. "context.encounter[0]" -> "encounter".
            element = path.rsplit(".", 1)[-1].split("[", 1)[0]
            expected = _TYPED_REFERENCES.get(element)
            if expected and target and target != expected:
                issues.append(
                    BundleValidator._rule(
                        rtype, rid, path, f"{element} must reference {expected}, found {target}"
                    )
                )
        return issues

    @staticmethod
    def _rule(
        rtype: str, rid: str | None, path: str, message: str, severity: Severity = Severity.ERROR
    ) -> ValidationIssue:
        return ValidationIssue(
            severity=severity,
            layer="rules",
            resource_type=rtype,
            resource_id=rid,
            path=path,
            message=message,
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class ValidationReport(BaseModel):
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    fhir_version: str = FHIR_VERSION
    results: list[BundleValidationResult]

    @property
    def total_bundles(self) -> int:
        return len(self.results)

    @property
    def valid_bundles(self) -> int:
        return sum(r.is_valid for r in self.results)

    @property
    def total_errors(self) -> int:
        return sum(len(r.errors) for r in self.results)

    def summary(self) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        for r in self.results:
            counts.update(r.resource_counts)
        return {
            "bundles": self.total_bundles,
            "valid_bundles": self.valid_bundles,
            "invalid_bundles": self.total_bundles - self.valid_bundles,
            "errors": self.total_errors,
            "warnings": sum(len(r.issues) - len(r.errors) for r in self.results),
            "resources": dict(sorted(counts.items())),
        }

    def to_markdown(self) -> str:
        s = self.summary()
        status = "PASS" if s["invalid_bundles"] == 0 else "FAIL"
        lines = [
            "# FHIR R4 Bundle Validation Report",
            "",
            f"- **Status:** {status}",
            f"- **Generated:** {self.generated_at.isoformat(timespec='seconds')}",
            f"- **FHIR version:** {self.fhir_version}",
            f"- **Bundles:** {s['valid_bundles']}/{s['bundles']} valid, "
            f"{s['errors']} error(s), {s['warnings']} warning(s)",
            "- **Resources:** " + ", ".join(f"{k}: {v}" for k, v in s["resources"].items()),
            "",
            "| MRN | Bundle | Resources | Result |",
            "|---|---|---|---|",
        ]
        for r in self.results:
            n = sum(r.resource_counts.values())
            verdict = "valid" if r.is_valid else f"**{len(r.errors)} error(s)**"
            lines.append(f"| {r.mrn or '-'} | `{r.bundle_id}` | {n} | {verdict} |")
        failing = [r for r in self.results if r.issues]
        if failing:
            lines += ["", "## Issues", ""]
            for r in failing:
                lines.append(f"### MRN {r.mrn} - bundle `{r.bundle_id}`")
                lines += [f"- [{i.layer}] {i}" for i in r.issues]
                lines.append("")
        return "\n".join(lines) + "\n"

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "fhir_validation_report.json"
        md_path = out_dir / "fhir_validation_report.md"
        payload = {"summary": self.summary(), **self.model_dump(mode="json")}
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path
