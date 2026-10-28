"""FastAPI application for customer purchase propensity predictions."""

from contextlib import asynccontextmanager
import os
import json
import hashlib
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.train import ARTIFACTS, FEATURES


MODEL_PATH = ARTIFACTS / "purchase_propensity_pipeline.joblib"
REGISTRY_PATH = ARTIFACTS / "model_registry.json"
MODEL_OVERRIDE = os.getenv("CUSTOMER_PURCHASE_MODEL")
THRESHOLD = float(os.getenv("CUSTOMER_PURCHASE_THRESHOLD", "0.5"))
if not 0.0 <= THRESHOLD <= 1.0:
    raise ValueError("CUSTOMER_PURCHASE_THRESHOLD must be between 0 and 1")


class CustomerFeatures(BaseModel):
    """Customer features calculated from the previous 90 days of transactions."""

    line_count: int = Field(ge=1)
    order_count: int = Field(ge=1)
    total_spend: float = Field(gt=0)
    average_line_value: float = Field(gt=0)
    product_diversity: int = Field(ge=1)
    total_quantity: int = Field(ge=1)
    recency_days: float = Field(ge=0, le=90)
    average_order_value: float = Field(gt=0)
    mean_days_between_orders: Optional[float] = Field(default=None, ge=0, le=90)


class PredictionRequest(BaseModel):
    customer_id: Optional[str] = None
    features: CustomerFeatures


class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    purchase_probability: float
    predicted_purchase: bool
    threshold: float
    probability_note: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(MODEL_OVERRIDE) if MODEL_OVERRIDE else MODEL_PATH
    if not MODEL_OVERRIDE and REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        active = registry.get("active")
        if active and active.get("model_path"):
            model_path = Path(active["model_path"])
            expected_hash = active.get("model_sha256")
            if expected_hash:
                digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
                if digest != expected_hash:
                    raise RuntimeError(f"Active model checksum mismatch at {model_path}")
    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found at {model_path}. Train or promote a model first."
        )
    app.state.model = joblib.load(model_path)
    app.state.model_path = str(model_path.resolve())
    yield


app = FastAPI(
    title="Customer Intelligence ML API",
    description="Predict 30-day purchase propensity from 90-day customer features.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    """Report service and loaded model status."""
    model = getattr(request.app.state, "model", None)
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_type": (
            type(model.named_steps["classifier"]).__name__ if model is not None else None
        ),
        "model_path": getattr(request.app.state, "model_path", None),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Return one customer's raw purchase score and thresholded class."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    feature_values = (
        payload.features.model_dump()
        if hasattr(payload.features, "model_dump")
        else payload.features.dict()
    )
    feature_row = pd.DataFrame([feature_values], columns=FEATURES)
    probability = float(model.predict_proba(feature_row)[0, 1])
    return PredictionResponse(
        customer_id=payload.customer_id,
        purchase_probability=probability,
        predicted_purchase=probability >= THRESHOLD,
        threshold=THRESHOLD,
        probability_note=(
            "Raw model score; November evaluation found probabilities under-calibrated."
        ),
    )
