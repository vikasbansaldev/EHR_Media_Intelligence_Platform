"""Task 2 - map cleaned records to FHIR R4 Bundles, validate them, and store them."""

from ehr_intel.fhir.mapper import FhirMapper, PatientBundle
from ehr_intel.fhir.pipeline import FhirNormalizer
from ehr_intel.fhir.store import BundleStore, StoredResource
from ehr_intel.fhir.validator import BundleValidationResult, BundleValidator, ValidationReport

__all__ = [
    "BundleStore",
    "BundleValidationResult",
    "BundleValidator",
    "FhirMapper",
    "FhirNormalizer",
    "PatientBundle",
    "StoredResource",
    "ValidationReport",
]
