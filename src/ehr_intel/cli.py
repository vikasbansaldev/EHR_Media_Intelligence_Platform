"""Command-line interface for every pipeline stage.

Examples::

    ehr-intel run-all                 # ingest -> FHIR -> summarize -> index
    ehr-intel search "chest pain with elevated troponin" --type DiagnosticReport
    ehr-intel serve --reload
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from ehr_intel.config import Settings, get_settings
from ehr_intel.logging_config import configure_logging

if TYPE_CHECKING:
    from ehr_intel.ingestion import CleanedDataset
    from ehr_intel.search import SearchService

logger = logging.getLogger("ehr_intel.cli")


def _ingest(settings: Settings, source: Path | None = None) -> CleanedDataset:
    from ehr_intel.ingestion import IngestionPipeline

    pipeline = IngestionPipeline(settings)
    dataset = pipeline.run(source)
    out = pipeline.write_reports(dataset)
    print(f"[1/4] Ingestion: {json.dumps(dataset.stats())}\n      report -> {out}")
    return dataset


def _fhir(settings: Settings, dataset: CleanedDataset) -> bool:
    from ehr_intel.fhir import FhirNormalizer

    report = FhirNormalizer(settings).run(dataset)
    s = report.summary()
    print(
        f"[2/4] FHIR: {s['valid_bundles']}/{s['bundles']} bundles valid, "
        f"{s['errors']} errors, resources {s['resources']}\n"
        f"      report -> {settings.reports_dir / 'fhir_validation_report.md'}"
    )
    return bool(s["invalid_bundles"] == 0)


def _summarize(settings: Settings, force: bool) -> None:
    from ehr_intel.fhir import BundleStore
    from ehr_intel.summarization import SummarizationService

    svc = SummarizationService(settings, BundleStore(settings.bundle_db_path))
    summaries = svc.summarize_all(force=force)
    path = svc.write_report(summaries, settings.reports_dir)
    failed = sum(1 for s in summaries if s.quality and not s.quality.passed)
    print(
        f"[3/4] Summaries: {len(summaries)} via {svc.provider.name} "
        f"(cache hits {svc.cache.stats.hits}, generated {svc.cache.stats.misses}, "
        f"quality flags {failed})\n      report -> {path}"
    )


def _index(settings: Settings) -> SearchService:
    from ehr_intel.fhir import BundleStore
    from ehr_intel.search import SearchService
    from ehr_intel.summarization import SummarizationService

    store = BundleStore(settings.bundle_db_path)
    svc = SearchService(settings, store, SummarizationService(settings, store))
    count = svc.build()
    print(
        f"[4/4] Search index: {count} documents embedded with {svc.embedder.name}, "
        f"{svc.index.backend} vector backend "
        f"-> {settings.index_dir}"
    )
    return svc


def cmd_run_all(args: argparse.Namespace, settings: Settings) -> int:
    dataset = _ingest(settings, args.source)
    ok = _fhir(settings, dataset)
    _summarize(settings, args.force)
    _index(settings)
    print("\nDone. Start the UI with: ehr-intel serve")
    return 0 if ok else 1


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    _ingest(settings, args.source)
    return 0


def cmd_fhir(args: argparse.Namespace, settings: Settings) -> int:
    return 0 if _fhir(settings, _ingest(settings, args.source)) else 1


def cmd_summarize(args: argparse.Namespace, settings: Settings) -> int:
    _summarize(settings, args.force)
    return 0


def cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    _index(settings)
    return 0


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    if args.rebuild:
        svc = _index(settings)
    else:
        from ehr_intel.fhir import BundleStore
        from ehr_intel.search import SearchService
        from ehr_intel.summarization import SummarizationService

        store = BundleStore(settings.bundle_db_path)
        svc = SearchService(settings, store, SummarizationService(settings, store))
        svc.load_or_build()
    response = svc.search(
        args.query,
        top_k=args.top_k,
        resource_types=args.type,
        date_from=args.date_from,
        date_to=args.date_to,
    )
    if args.json:
        print(response.model_dump_json(indent=2))
        return 0
    print(
        f'"{response.query}" - {len(response.results)} result(s) in {response.took_ms} ms '
        f"({response.embedder})"
    )
    for r in response.results:
        print(
            f"\n#{r.rank} [{r.score:.3f}] {r.patient_name} (MRN {r.mrn}) - {r.kind}: {r.title}"
            f" [{r.date or 'undated'}]\n    {r.snippet}"
        )
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    uvicorn.run(
        "ehr_intel.api.app:create_app",
        factory=True,
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
    )
    return 0


def cmd_generate(args: argparse.Namespace, settings: Settings) -> int:
    import runpy

    script = Path(__file__).resolve().parents[2] / "scripts" / "generate_synthetic_data.py"
    sys.argv = [
        str(script),
        "--out",
        str(args.out or settings.raw_data_dir),
        "--seed",
        str(args.seed),
    ]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ehr-intel", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run-all", help="ingest, normalize, summarize and index")
    p.add_argument("--source", type=Path, help="input file or directory (default: raw_data_dir)")
    p.add_argument("--force", action="store_true", help="regenerate summaries (ignore cache)")
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("ingest", help="Task 1: ingest and clean only")
    p.add_argument("--source", type=Path)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("fhir", help="Tasks 1-2: ingest and build/validate FHIR bundles")
    p.add_argument("--source", type=Path)
    p.set_defaults(func=cmd_fhir)

    p = sub.add_parser("summarize", help="Task 3: summarize every stored patient")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("index", help="Task 4: (re)build the search index")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="query the index from the terminal")
    p.add_argument("query")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--type", action="append", help="resource type filter (repeatable)")
    p.add_argument("--date-from", type=date.fromisoformat)
    p.add_argument("--date-to", type=date.fromisoformat)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("serve", help="Task 5: run the API and web UI")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("generate-data", help="regenerate the synthetic sample dataset")
    p.add_argument("--out", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.WARNING)
    settings = get_settings()
    settings.ensure_dirs()
    return int(args.func(args, settings))


if __name__ == "__main__":
    raise SystemExit(main())
