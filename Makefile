# HybridIR convenience targets. Everything runs in Docker except `test-local` / `lint-local`.
SHELL := /bin/bash
COMPOSE ?= docker compose
RUN_BACKEND := $(COMPOSE) run --rm seed

.DEFAULT_GOAL := help
.PHONY: help up down logs ps seed reseed eval ablations test test-local lint lint-local load \
        load-mixed results readme clean

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

up: ## build and start everything (first run downloads models + dataset)
	$(COMPOSE) up --build -d
	@echo "web: http://localhost:$${WEB_PORT:-3000}   api: http://localhost:$${API_PORT:-8000}/docs"

down: ## stop containers (keeps volumes)
	$(COMPOSE) down

logs: ## follow logs of all services
	$(COMPOSE) logs -f --tail=100

ps: ## service status
	$(COMPOSE) ps

seed: ## run the idempotent seed again (no-op if nothing changed)
	$(RUN_BACKEND)

reseed: ## rebuild index, embeddings, collection and results from scratch
	$(COMPOSE) run --rm -e FORCE_RESEED=1 seed
	$(COMPOSE) restart api

eval: ## rerun the baseline evaluation (all strategies) and rewrite results/
	$(RUN_BACKEND) python -m scripts.run_eval

ablations: ## rerun every ablation grid (ONLY=bm25_grid,fusion,... to restrict)
	$(RUN_BACKEND) python -m scripts.run_ablations $(if $(ONLY),--only $(ONLY),)

test: ## backend test suite inside the backend image (no network needed)
	$(COMPOSE) run --rm --no-deps seed python -m pytest -q -p no:cacheprovider

test-local: ## backend tests on the host (pip install -r backend/requirements-base.txt -r backend/requirements-dev.txt)
	cd backend && python -m pytest -q

lint: lint-local ## lint + type-check backend and frontend (host toolchains)

lint-local:
	cd backend && ruff check . && ruff format --check . && mypy
	cd frontend && npm run lint && npm run typecheck

load: ## k6 load test per strategy -> results/k6_summary.json
	$(COMPOSE) --profile loadtest run --rm k6

load-mixed: ## k6 mixed workload (70/20/10)
	K6_SCRIPT=k6_mixed.js $(COMPOSE) --profile loadtest run --rm k6

results: ## copy results/ out of the Docker volume into ./results
	mkdir -p results
	$(COMPOSE) run --rm --no-deps -v "$(CURDIR)/results:/export" --user "$$(id -u):$$(id -g)" seed \
		sh -c 'cp -r /results/. /export/'

readme: ## render README results block from the results volume
	$(COMPOSE) run --rm --no-deps -v "$(CURDIR):/repo" --user "$$(id -u):$$(id -g)" seed \
		python -m scripts.render_readme --results-dir /results --readme /repo/README.md

clean: ## stop and DELETE all volumes (index, embeddings, results, model cache)
	$(COMPOSE) --profile loadtest down -v --remove-orphans
	rm -rf results
