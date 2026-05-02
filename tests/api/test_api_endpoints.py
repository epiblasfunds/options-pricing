from fastapi.testclient import TestClient
import numpy as np

from src.api.main import app
from src.api.main import get_model_service


class _FakeModelService:
    def __init__(self):
        self.predict_request = None
        self.sample_request = None

    def predict(self, request):
        self.predict_request = request
        return {
            "modelo": request.modelo,
            "prediction": 0.24,
            "input": {"StrikePrice": request.caracteristicas.strikePrice},
        }

    def sample_explainability(self, request):
        self.sample_request = request
        return {
            "modelo": request.modelo,
            "prediction": np.float64(0.31),
            "input": {"StrikePrice": request.caracteristicas.strikePrice},
            "reference_sample_index": None,
            "waterfall_image": "data:image/png;base64,abc",
            "local_explanation": {"feature_names": ["rate"], "value": np.float64(1.2)},
            "neighbors": [{"index": np.int64(10), "distance": np.float64(0.0)}],
            "neighbor_distances": [{"row_id": "10", "distance": np.float64(0.0)}],
        }


def _payload():
    return {
        "modelo": "random_forest",
        "caracteristicas": {
            "optionType": "CALL",
            "strikePrice": 10000.0,
            "underlyingPrice": 10100.0,
            "timeToExpiration": 30.0,
            "rate": 0.02,
        },
    }


def test_predict_endpoint_accepts_defaults():
    service = _FakeModelService()
    app.dependency_overrides[get_model_service] = lambda: service
    try:
        response = TestClient(app).post("/run_model/predict/", json=_payload())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["prediction"] == 0.24
    assert service.predict_request.caracteristicas.strikePrice == 10000.0


def test_sample_explainability_endpoint_returns_dashboard_payload():
    service = _FakeModelService()
    app.dependency_overrides[get_model_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/run_model/sample_explainability/",
            json=_payload(),
        )
    finally:
        app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 200
    assert body["prediction"] == 0.31
    assert body["reference_sample_index"] is None
    assert body["waterfall_image"].startswith("data:image/png;base64,")
    assert body["neighbors"][0]["distance"] == 0.0
