# CineRecOps

**Production-Ready Movie Recommendation Platform with MLOps**

End-to-end recommendation engine built on TensorFlow and MLflow, featuring reproducible training, experiment tracking, model registry versioning, CI/CD automation, and scalable inference APIs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CineRecOps                           │
│                                                             │
│  ┌──────────┐   ┌────────────┐   ┌──────────────────────┐  │
│  │   Data   │──▶│  Features  │──▶│  Model Training      │  │
│  │ Pipeline │   │Engineering │   │  (TF + MLflow)       │  │
│  └──────────┘   └────────────┘   └──────────┬───────────┘  │
│                                             │               │
│                                  ┌──────────▼───────────┐  │
│                                  │  MLflow Model        │  │
│                                  │  Registry            │  │
│                                  └──────────┬───────────┘  │
│                                             │               │
│  ┌──────────────────────────────────────────▼───────────┐  │
│  │            FastAPI Inference Server                  │  │
│  │  /recommend  │  /predict  │  /health  │  /metrics   │  │
│  └──────────────────────────┬───────────────────────────┘  │
│                             │                               │
│  ┌──────────────────────────▼───────────────────────────┐  │
│  │     Monitoring: Prometheus + Grafana                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Models

| Model | Architecture | Use Case |
|---|---|---|
| **Two-Tower** | Dual encoder + dot product | Large-scale retrieval |
| **NCF** | GMF + MLP (NeuMF fusion) | Ranking / rating prediction |
| **Hybrid** | ID embeddings + content features | Cold-start + warm users |

All models are trained with:
- L2 regularization + dropout
- Early stopping on validation loss
- Learning rate reduction on plateau
- MLflow autologging + model registry

---

## Project Structure

```
CineRecOps/
├── src/
│   ├── data/                   # Ingestion & preprocessing
│   │   ├── ingestion.py        # MovieLens download + loading
│   │   └── preprocessing.py    # Filter, encode, temporal split
│   ├── features/
│   │   └── feature_engineering.py  # User/item feature matrices
│   ├── models/
│   │   ├── two_tower.py        # Two-Tower (dual encoder)
│   │   ├── ncf.py              # Neural Collaborative Filtering
│   │   └── hybrid.py           # Hybrid CF + content-based
│   ├── training/
│   │   ├── train.py            # Main training entry point (Hydra)
│   │   ├── dataset.py          # tf.data.Dataset builders
│   │   └── callbacks.py        # MLflow logging callbacks
│   ├── evaluation/
│   │   └── metrics.py          # RMSE, MAE, NDCG@K, Precision@K
│   └── serving/
│       ├── server.py           # FastAPI app
│       ├── model_loader.py     # MLflow registry loader
│       └── schemas.py          # Pydantic request/response models
├── mlflow/
│   ├── registry.py             # Promote, archive, alias helpers
│   └── mlflow_config.py        # Experiment + registry setup
├── pipelines/
│   └── full_pipeline.py        # End-to-end orchestration
├── configs/
│   ├── config.yaml             # Base Hydra config
│   └── experiment/             # Per-experiment overrides
├── tests/
│   ├── unit/                   # Preprocessing, models, metrics
│   └── integration/            # API endpoint tests
├── docker/
│   ├── Dockerfile.train        # Training image (GPU)
│   ├── Dockerfile.serve        # Serving image
│   └── docker-compose.yml      # Full stack (MLflow + API + trainer)
├── monitoring/
│   ├── prometheus.yml          # Scrape config
│   ├── docker-compose.monitoring.yml
│   └── grafana/                # Dashboards + datasources
├── .github/workflows/
│   ├── ci.yml                  # Lint, unit tests, integration tests, Docker build
│   └── cd.yml                  # Train, promote, deploy (scheduled + manual)
└── scripts/
    ├── download_data.py
    ├── process_data.py
    └── promote_model.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Start MLflow tracking server

```bash
make mlflow-ui
# → http://localhost:5000
```

### 3. Download data

```bash
make download-data
```

### 4. Train a model

```bash
# Default: two-tower model
make train

# Override experiment config
make train-experiment   # uses configs/experiment/cf_experiment.yaml
```

### 5. Promote the best model to Production

```bash
make promote-model
```

### 6. Serve the inference API

```bash
make serve
# → http://localhost:8000/docs
```

---

## API Reference

### `POST /recommend`

```json
{
  "user_id": 42,
  "top_k": 10
}
```

Response:
```json
{
  "user_id": 42,
  "recommendations": [
    {
      "movie_id": 260,
      "title": "Star Wars: Episode IV - A New Hope",
      "genres": "Action|Adventure|Sci-Fi",
      "predicted_rating": 4.51,
      "score": 0.8752
    }
  ],
  "model_version": "3",
  "latency_ms": 42.1
}
```

### `POST /predict`

```json
{ "user_id": 42, "movie_id": 260 }
```

### `GET /health`

Returns model status, version, and uptime.

### `GET /metrics`

Prometheus metrics scrape endpoint.

---

## Docker Stack

```bash
# Start MLflow + API
docker compose -f docker/docker-compose.yml up -d

# Run a training job
docker compose -f docker/docker-compose.yml --profile training up trainer

# Start monitoring (Prometheus + Grafana)
make monitoring-up
# Grafana → http://localhost:3000  (admin / cinerecops)
# Prometheus → http://localhost:9090
```

---

## Running Tests

```bash
make test           # All tests with coverage
make test-unit      # Unit tests only
make test-integration  # API integration tests
```

---

## CI/CD

| Workflow | Trigger | Steps |
|---|---|---|
| **CI** (`ci.yml`) | Push / PR | Lint → Unit tests → Integration tests → Docker build |
| **CD** (`cd.yml`) | Manual / Nightly cron | Train → Evaluate → Promote → Build image → Deploy |

The CD pipeline uses GitHub Environments for production gate approval.

---

## MLOps Practices

- **Experiment tracking**: every hyperparameter, metric, and artifact is logged to MLflow
- **Model versioning**: models are registered in the MLflow Model Registry with semantic stages (Staging → Production → Archived)
- **Reproducibility**: Hydra config snapshots + random seed locking + temporal data splits
- **Monitoring**: Prometheus metrics for request rate, latency (p50/p95/p99), and error rate
- **Automated promotion**: CI/CD promotes the best-RMSE model to Production automatically
- **Health checks**: Docker HEALTHCHECK + `/health` endpoint for orchestrators

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow 2.15, Keras |
| Experiment Tracking | MLflow 2.9 |
| Config Management | Hydra + OmegaConf |
| API Serving | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Data | MovieLens-1M |
