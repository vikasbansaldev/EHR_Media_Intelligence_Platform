"""HTTP API (FastAPI) exposing search, patient detail and the web UI."""

from ehr_intel.api.app import Services, build_services, create_app

__all__ = ["Services", "build_services", "create_app"]
