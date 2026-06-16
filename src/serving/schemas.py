"""
Pydantic request/response schemas for the recommendation API.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="Internal user identifier", gt=0)
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations to return")
    exclude_seen: bool = Field(default=True, description="Exclude movies the user has already rated")

    model_config = {"json_schema_extra": {"example": {"user_id": 42, "top_k": 10}}}


class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    predicted_rating: float = Field(..., ge=0.0, le=5.0)
    score: float = Field(..., description="Raw model score (0–1)")

    model_config = {"json_schema_extra": {
        "example": {
            "movie_id": 260,
            "title": "Star Wars: Episode IV - A New Hope",
            "genres": "Action|Adventure|Sci-Fi",
            "predicted_rating": 4.5,
            "score": 0.87,
        }
    }}


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[MovieRecommendation]
    model_version: str
    latency_ms: float


class RatingPredictionRequest(BaseModel):
    user_id: int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0)


class RatingPredictionResponse(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
