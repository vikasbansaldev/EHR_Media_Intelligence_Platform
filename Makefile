.DEFAULT_GOAL := help
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install install-ml data pipeline serve test test-fast lint format typecheck check frontend clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:
	$(PY) -m venv $(VENV)

install: venv ## Install package + dev tools (offline hashing embedder)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev]"

install-ml: install ## Also install sentence-transformers (all-MiniLM-L6-v2)
	$(BIN)/pip install -e ".[dev,ml]"

data: ## Regenerate the synthetic sample dataset
	$(BIN)/ehr-intel generate-data

pipeline: ## Ingest -> FHIR -> summarize -> index
	$(BIN)/ehr-intel run-all

serve: ## Run the API + UI on http://127.0.0.1:8000
	$(BIN)/ehr-intel serve

test: ## Run the full test suite with coverage
	$(BIN)/pytest --cov --cov-report=term-missing

test-fast: ## Unit tests only
	$(BIN)/pytest tests/unit -q

lint: ## Ruff lint + format check
	$(BIN)/ruff check src tests scripts verify_local.py
	$(BIN)/ruff format --check src tests scripts verify_local.py

format: ## Auto-format
	$(BIN)/ruff format src tests scripts
	$(BIN)/ruff check --fix src tests scripts

typecheck: ## mypy --strict
	$(BIN)/mypy

check: lint typecheck test ## Everything CI would run

frontend: ## Rebuild Tailwind CSS (needs Node 18+)
	cd frontend && npm ci && npm run build

clean: ## Remove generated artifacts
	rm -rf var .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
