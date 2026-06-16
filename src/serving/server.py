"""
FastAPI inference server — exposes recommendation and rating-prediction endpoints.
Includes Prometheus metrics, health checks, and structured logging.

Endpoints:
  GET  /health              — liveness / readiness probe
  POST /recommend           — top-K recommendations for a user
  POST /predict             — single rating prediction
  GET  /metrics             — Prometheus metrics (scrape target)
  GET  /model/info          — current model version and metadata
"""
from __future__ import annotations

import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .model_loader import ModelLoader
from .schemas import (
    HealthResponse,
    RatingPredictionRequest,
    RatingPredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
    MovieRecommendation,
)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "cinerecops_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
RECOMMENDATION_LATENCY = Histogram(
    "cinerecops_recommendation_latency_seconds",
    "Time to generate recommendations",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)
PREDICTION_LATENCY = Histogram(
    "cinerecops_prediction_latency_seconds",
    "Time for single rating prediction",
)

_start_time = time.time()


# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    loader = ModelLoader.get_instance()
    loader.load(
        model_name=os.getenv("MLFLOW_MODEL_NAME", "cinerecops-recommender"),
        stage=os.getenv("MODEL_STAGE", "Production"),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    # Load metadata for movie lookup
    _load_metadata(app)
    logger.info("CineRecOps API is ready.")
    yield
    logger.info("Shutting down CineRecOps API.")


def _load_metadata(app: FastAPI) -> None:
    """Load movie metadata and encoder mappings into app state."""
    processed_path = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
    try:
        import pandas as pd
        movies_path = processed_path / "movies.parquet"
        if movies_path.exists():
            app.state.movies_df = pd.read_parquet(movies_path)
        else:
            app.state.movies_df = None
            logger.warning("movies.parquet not found; movie titles won't be available.")

        with open(processed_path / "movie_encoder.pkl", "rb") as f:
            app.state.movie_encoder = pickle.load(f)

        app.state.all_movie_idxs = np.arange(len(app.state.movie_encoder.classes_), dtype=np.int32)
    except Exception as e:
        logger.warning(f"Could not load metadata: {e}. Some features may be degraded.")
        app.state.movies_df = None
        app.state.movie_encoder = None
        app.state.all_movie_idxs = np.arange(1000, dtype=np.int32)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CineRecOps",
    description="Production Movie Recommendation API powered by TensorFlow & MLflow",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware ────────────────────────────────────────────────────────────────
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    loader = ModelLoader.get_instance()
    return HealthResponse(
        status="healthy" if loader.is_loaded else "degraded",
        model_loaded=loader.is_loaded,
        model_version=loader.version,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/metrics", tags=["Operations"], include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", tags=["Model"])
async def model_info():
    loader = ModelLoader.get_instance()
    return {
        "model_name": os.getenv("MLFLOW_MODEL_NAME", "cinerecops-recommender"),
        "model_version": loader.version,
        "stage": os.getenv("MODEL_STAGE", "Production"),
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    }


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend(request: RecommendationRequest, req: Request) -> RecommendationResponse:
    """Return top-K movie recommendations for a given user."""
    loader = ModelLoader.get_instance()
    if not loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.time()

    # Resolve user index
    user_idx = request.user_id - 1  # simple 1-based → 0-based
    all_item_idxs = req.app.state.all_movie_idxs

    # Score all items
    user_idxs = np.full(len(all_item_idxs), user_idx, dtype=np.int32)
    scores = loader.predict(user_idxs, all_item_idxs)

    # Select top-K
    top_k_idxs = np.argsort(-scores)[: request.top_k]
    top_k_items = all_item_idxs[top_k_idxs]
    top_k_scores = scores[top_k_idxs]

    recommendations = _build_recommendations(
        top_k_items, top_k_scores, req.app.state.movies_df, req.app.state.movie_encoder
    )

    latency_ms = (time.time() - start) * 1000
    RECOMMENDATION_LATENCY.observe((time.time() - start))

    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        model_version=loader.version,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict", response_model=RatingPredictionResponse, tags=["Predictions"])
async def predict_rating(request: RatingPredictionRequest) -> RatingPredictionResponse:
    """Predict the rating a user would give a specific movie."""
    loader = ModelLoader.get_instance()
    if not loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.time()

    user_idx = np.array([request.user_id - 1], dtype=np.int32)
    item_idx = np.array([request.movie_id - 1], dtype=np.int32)
    score = loader.predict(user_idx, item_idx)[0]

    predicted_rating = float(score * 4.0 + 1.0)  # denormalize [0,1] → [1,5]
    PREDICTION_LATENCY.observe(time.time() - start)

    return RatingPredictionResponse(
        user_id=request.user_id,
        movie_id=request.movie_id,
        predicted_rating=round(predicted_rating, 2),
        confidence=round(float(score), 4),
        model_version=loader.version,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_recommendations(
    item_idxs: np.ndarray,
    scores: np.ndarray,
    movies_df,
    movie_encoder,
) -> list[MovieRecommendation]:
    recs = []
    for idx, score in zip(item_idxs, scores):
        try:
            movie_id = int(movie_encoder.inverse_transform([idx])[0]) if movie_encoder else int(idx)
        except Exception:
            movie_id = int(idx)

        title, genres = "Unknown", "Unknown"
        if movies_df is not None:
            row = movies_df[movies_df["movie_id"] == movie_id]
            if not row.empty:
                title = str(row.iloc[0].get("title_clean", "Unknown"))
                genres = str(row.iloc[0].get("genres", "Unknown"))

        recs.append(
            MovieRecommendation(
                movie_id=movie_id,
                title=title,
                genres=genres,
                predicted_rating=round(float(score) * 4.0 + 1.0, 2),
                score=round(float(score), 4),
            )
        )
    return recs


def main():
    import uvicorn
    uvicorn.run(
        "serving.server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=int(os.getenv("API_WORKERS", "4")),
    )


if __name__ == "__main__":
    main()
