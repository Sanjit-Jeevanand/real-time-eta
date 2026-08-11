SHELL := /bin/bash
.DEFAULT_GOAL := help

export PYTHONPATH := $(CURDIR)/src

UV      ?= uv
COMPOSE ?= docker compose
PROD    := docker/compose.prod.yml
OSRM    := docker/compose.osrm.yml

.PHONY: help setup lock lint format typecheck test test-cov ci \
        data data-download data-trips data-weather data-enrich data-card route route-fetch route-zones route-matrix route-embeddings route-detour train calib bench deploy \
        osrm-up osrm-down up down logs clean

help:
	@echo 'quality : lint typecheck leakage parity test ci'
	@echo 'data    : data data-download data-trips data-weather data-enrich data-card'
	@echo 'pipeline: route route-fetch route-zones route-matrix route-embeddings route-detour train calib bench deploy'
	@echo 'env     : setup lock format clean'
	@echo 'docker  : up down logs osrm-up osrm-down'

setup:
	$(UV) sync --extra dev --extra geo
	$(UV) run pre-commit install

lock:
	$(UV) lock
	@echo "uv.lock regenerated -- commit it alongside pyproject.toml"

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest

leakage:
	$(UV) run pytest -m leakage

parity:
	$(UV) run pytest -m parity

replay:
	$(UV) run python -m eta.features.replay_cli --month 2023-06

test-cov:
	$(UV) run pytest --cov --cov-report=term-missing --cov-report=xml

ci: lint typecheck leakage parity test

define not_yet
	@echo ""
	@echo "  make $(1) is not implemented yet -- arrives in Phase $(2)."
	@echo "  $(3)"
	@echo ""
	@exit 1
endef

data:
	$(UV) run python -m eta.data all

data-download:
	$(UV) run python -m eta.data download

data-trips:
	$(UV) run python -m eta.data trips

data-weather:
	$(UV) run python -m eta.data weather

data-enrich:
	$(UV) run python -m eta.data enrich

data-card:
	$(UV) run python -m eta.data card

route-fetch:
	mkdir -p data/osrm
	curl -fSL --progress-bar -o data/osrm/new-york-latest.osm.pbf \
		https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf

route:
	$(UV) run python -m eta.routing all

route-zones:
	$(UV) run python -m eta.routing zones

route-matrix:
	$(UV) run python -m eta.routing matrix

route-embeddings:
	$(UV) run python -m eta.routing embeddings

route-detour:
	$(UV) run python -m eta.routing detour

route-unused:
	$(call not_yet,route,3,Run `make route-fetch` then `make osrm-up`; this target precomputes the 265x265 zone-pair matrix)

train:
	$(call not_yet,train,6,LightGBM quantile models at the 5 configured levels x 3 seeds)

calib:
	$(call not_yet,calib,7,Per-segment isotonic + CQR on the dedicated calibration split)

bench:
	$(call not_yet,bench,9,k6 ramp against eta-bench from a separate load box)

deploy:
	$(call not_yet,deploy,8,docker build -> GHCR -> ssh eta-prod -> compose pull && up -d)

osrm-up:
	$(COMPOSE) -f $(OSRM) --profile osrm up -d osrm-routed

osrm-down:
	$(COMPOSE) -f $(OSRM) --profile osrm down

up:
	$(COMPOSE) -f $(PROD) up -d

down:
	$(COMPOSE) -f $(PROD) down

logs:
	$(COMPOSE) -f $(PROD) logs -f --tail=100

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
