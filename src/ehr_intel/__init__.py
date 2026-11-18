"""EHR Media Intelligence Platform.

Pipeline stages, each in its own subpackage:

1. :mod:`ehr_intel.ingestion`     - read and clean heterogeneous EHR exports
2. :mod:`ehr_intel.fhir`          - map to HL7 FHIR R4 Bundles, validate, store
3. :mod:`ehr_intel.summarization` - LLM clinical summaries with a cache
4. :mod:`ehr_intel.search`        - embeddings + vector index for semantic search
5. :mod:`ehr_intel.api`           - FastAPI app serving search and the web UI
"""

__version__ = "0.1.0"
