.PHONY: install lint test train serve docker-build docker-run clean mlflow-ui

# ── Environment ──────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"
	pip install -r requirements.txt

# ── Code Quality ─────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# ── Data ──────────────────────────────────────────────────────────────────────
download-data:
	python scripts/download_data.py

process-data:
	python scripts/process_data.py

# ── Training ──────────────────────────────────────────────────────────────────
train:
	python src/training/train.py

train-experiment:
	python src/training/train.py experiment=cf_experiment

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

promote-model:
	python scripts/promote_model.py

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	uvicorn src.serving.server:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.serving.server:app --host 0.0.0.0 --port 8000 --workers 4

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile.train -t cinerecops-train:latest .
	docker build -f docker/Dockerfile.serve -t cinerecops-serve:latest .

docker-run-train:
	docker run --rm -v $(PWD)/mlruns:/app/mlruns cinerecops-train:latest

docker-run-serve:
	docker run --rm -p 8000:8000 cinerecops-serve:latest

docker-compose-up:
	docker compose -f docker/docker-compose.yml up -d

docker-compose-down:
	docker compose -f docker/docker-compose.yml down

# ── Monitoring ────────────────────────────────────────────────────────────────
monitoring-up:
	docker compose -f monitoring/docker-compose.monitoring.yml up -d

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info
