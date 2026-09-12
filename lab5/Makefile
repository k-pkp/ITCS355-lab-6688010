# ITCS355 Lab 1
# `make reproduce` is the one command a grader runs. Keep it working.

SHELL := /bin/bash
IMAGE ?= itcs355-lab1
TAG   ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
# Passed into the container: it has no .git, so the run cannot find its own commit.
COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
PLATFORM ?= linux/amd64
SEED ?= 20260101

.PHONY: help setup cloud-check data test portability-audit train image image-push reproduce verify clean teardown \
        tune compare reload-check register promote serve serve-image loadtest drift inject-drift \
        pipeline cost swap-check price-check llm-eval llm-gate llm-eval-mine llm-gate-mine \
        llm-cap-check llm-record canary

help:
	@grep -E "^[a-zA-Z_-]+:.*?## .*$$" $(MAKEFILE_LIST) | awk -F":.*?## " "{printf \"  %-20s %s\\n\", \$$1, \$$2}"

setup: ## Install dependencies and print environment status
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "environment ok"

cloud-check: ## Resolve the eight capability slots
	python scripts/cloud_check.py

data: ## Generate the default dataset (deterministic)
	python scripts/make_dataset.py --seed $(SEED)

test: ## Run data contract and split property tests
	pytest -q tests/

portability-audit: ## Fail if provider strings leak into src/
	python scripts/portability_audit.py

train: ## Train locally, outside the container
	python -m src.train --seed $(SEED) --metrics-out reports/metrics.json

image: ## Build the training image for linux/amd64
	docker buildx build --platform $(PLATFORM) -t $(IMAGE):$(TAG) --load .

image-push: image ## Push to CONTAINER_REGISTRY via your adapter
	@mkdir -p reports
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	print(get_adapter(config.load()).push_image(\"$(IMAGE):$(TAG)\"))" | tee reports/image-digest.txt
	@echo "digest recorded in reports/image-digest.txt — Lab 2 lineage reads it from there"

reproduce: data image ## THE ONE COMMAND. Grader runs this.
	@mkdir -p reports
	# --user and -w are what make this work on a machine that is not the author's.
	# The image runs as uid 10001, but reports/ on a fresh clone belongs to whoever
	# cloned it, mode 755 — so the container cannot write MLflow's sqlite file and the
	# run dies with "unable to open database file". Running as the invoking user fixes
	# the ownership mismatch and still runs non-root; -w puts MLflow's relative artifact
	# directory inside the mount rather than in root-owned /app.
	docker run --rm \
	  -v "$$PWD/data:/app/data:ro" \
	  -v "$$PWD/reports:/app/reports" \
	  -e MLFLOW_TRACKING_URI=sqlite:////app/reports/mlflow.db \
	  -e GIT_COMMIT=$(COMMIT) \
	  -e HOME=/tmp \
	  --user "$$(id -u):$$(id -g)" \
	  -w /app/reports \
	  $(IMAGE):$(TAG) --seed $(SEED) --metrics-out /app/reports/metrics.json

verify: ## Check the produced metric against the README claim
	python scripts/verify_metric.py

teardown: ## Delete every resource tagged course=itcs355 for this lab
	python -c "from src import config; from cloudlayer.factory import get_adapter; \
	cfg=config.load(); print(get_adapter(cfg).teardown(cfg.tags(1)))"

clean: ## Remove local artifacts
	rm -rf mlruns mlartifacts mlflow.db reports/metrics.json .pytest_cache

# --- Lab 2 -------------------------------------------------------------------
tune: ## Budgeted hyperparameter study (>=12 trials)
	python -m src.tune --trials 12 --budget-thb 150

compare: ## Rank runs by metric and by cost per point
	python scripts/compare_runs.py --experiment itcs355-lab2

reload-check: ## Load the registered model by version and score rows
	python scripts/reload_check.py --name $(MODEL_REGISTRY_NAME) --version $(VERSION)

register: ## Train the chosen configuration and register it with all eight lineage fields
	python scripts/register_model.py --stage Staging

price-check: ## Re-pull every GCP rate in src/costs.py from the Cloud Billing Catalog
	python scripts/price_check.py --show-working

# --- Lab 3 -------------------------------------------------------------------
serve: ## Run the inference service locally on :8080
	python scripts/export_model.py --out reports/model.joblib
	MODEL_PATH=reports/model.joblib MODEL_VERSION=local uvicorn service.app:app --port 8080

serve-image: ## Build the serving image
	docker buildx build --platform $(PLATFORM) -f service/Dockerfile.serve -t itcs355-serve:$(TAG) --load .

loadtest: ## Load test at three concurrency levels
	@for vus in 1 10 50; do \
	  echo "=== $$vus VUs ==="; \
	  k6 run -e TARGET=$(TARGET) -e VUS=$$vus loadtest/k6.js || true; \
	done

# --- Lab 4 -------------------------------------------------------------------
inject-drift: ## Shift a feature's distribution on purpose
	python scripts/inject_drift.py --feature temp_c --mode shift --magnitude 6

drift: ## Score drift against the reference window
	python -m monitoring.drift --current data/current.csv

# --- Lab 5 -------------------------------------------------------------------
pipeline: ## Compile pipeline/pipeline.yaml for your provider
	python -c "from cloudlayer.pipelines import compile_for; from src import config; \
	print('compiled to', compile_for(config.load().provider))"

llm-eval: ## Run the LLM golden set against recorded responses (offline, free)
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json

llm-gate: ## Prove the gate fails on a degraded set — expected to exit non-zero
	python scripts/llm_eval.py --out reports/llm_eval-baseline.json >/dev/null
	python scripts/llm_eval.py --responses evals/fixtures/triage-regressed.jsonl \
	  --out reports/llm_eval.json --baseline reports/llm_eval-baseline.json

GOLDEN ?= evals/golden/maintenance-triage.jsonl
DEGRADED ?= evals/fixtures/maintenance-triage-terse-prompt.jsonl

llm-eval-mine: ## Run OUR golden set (13 cases) against the baseline responses
	python scripts/llm_eval.py --golden $(GOLDEN) \
	  --responses evals/fixtures/maintenance-triage-baseline.jsonl \
	  --out reports/llm_eval-mt-baseline.json

llm-gate-mine: llm-eval-mine ## Prove OUR gate fails on the degraded set — exits non-zero
	python scripts/llm_eval.py --golden $(GOLDEN) --responses $(DEGRADED) \
	  --out reports/llm_eval-mt-terse.json --baseline reports/llm_eval-mt-baseline.json

llm-cap-check: llm-eval-mine ## Show that a 32-token output cap breaks the answers
	python scripts/llm_eval.py --golden $(GOLDEN) \
	  --responses evals/fixtures/maintenance-triage-cap32.jsonl \
	  --out reports/llm_eval-mt-cap32.json --baseline reports/llm_eval-mt-baseline.json

llm-record: ## Re-record the fixture live through the adapter, with real usage fields
	python scripts/llm_eval.py --golden $(GOLDEN) --live \
	  --out reports/llm_eval-mt-live.json

cost: ## Build the cost report
	python scripts/cost_report.py --estimate $(EST) --actual $(ACT) --rps $(RPS) --instance $(INSTANCE)

swap-check: ## Prove the portability seam against a second provider
	python scripts/portability_swap_check.py --second-provider $(SECOND)
