from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from src import api


def valid_features():
    return api.CustomerFeatures(
        line_count=20,
        order_count=2,
        total_spend=680.50,
        average_line_value=34.03,
        product_diversity=16,
        total_quantity=240,
        recency_days=12.5,
        average_order_value=340.25,
        mean_days_between_orders=31.0,
    )


def test_features_reject_negative_recency():
    values = valid_features().model_dump()
    values["recency_days"] = -1

    with pytest.raises(ValidationError):
        api.CustomerFeatures(**values)


def test_predict_returns_score_class_and_feature_order(monkeypatch):
    class StubModel:
        def predict_proba(self, frame):
            assert list(frame.columns) == api.FEATURES
            return np.array([[0.25, 0.75]])

    monkeypatch.setattr(api, "THRESHOLD", 0.7)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model=StubModel())))
    payload = api.PredictionRequest(customer_id="12346", features=valid_features())

    response = api.predict(payload, request)

    assert response.customer_id == "12346"
    assert response.purchase_probability == 0.75
    assert response.predicted_purchase is True
    assert response.threshold == 0.7
    assert "under-calibrated" in response.probability_note


def test_predict_returns_false_below_threshold(monkeypatch):
    class StubModel:
        def predict_proba(self, frame):
            return np.array([[0.8, 0.2]])

    monkeypatch.setattr(api, "THRESHOLD", 0.5)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(model=StubModel())))
    response = api.predict(
        api.PredictionRequest(features=valid_features()), request
    )

    assert response.predicted_purchase is False
