import pandas as pd

from src.volatility_models.model_explainability.config import (
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_METRICS_REGISTRY,
)
from src.volatility_models.model_explainability.services.surrogate_service import (
    SurrogateService,
)


class _PredictionServiceStub:
    def load_bundle(self, _model_id):
        return object()

    def resolve_model_input_features(self, _bundle):
        return [
            "TimeToExpiration",
            "Rate",
            "UnderlyingPrice",
            "StrikePrice",
            "OptionType",
            "ExecHour",
            "ExecWeekday",
        ]

    def predict_frame(self, _model_id, frame):
        option_type = frame["OptionType"].map({"C": 0.05, "P": 0.15})
        regime = (frame["UnderlyingPrice"].astype(float) > 9250).astype(float) * 0.25
        weekday_regime = (frame["ExecWeekday"].astype(float) >= 4).astype(float) * 0.10
        return (
            0.1
            + regime
            + weekday_regime
            + 0.002 * frame["TimeToExpiration"].astype(float)
            + option_type
        )


def test_surrogate_service_trains_tree_on_model_predictions():
    frame = pd.DataFrame(
        {
            "TimeToExpiration": [10, 20, 30, 40, 50, 60] * 120,
            "Rate": [1.0] * 720,
            "UnderlyingPrice": [9000, 9100, 9200, 9300, 9400, 9500] * 120,
            "StrikePrice": [8900, 9000, 9100, 9200, 9300, 9400] * 120,
            "OptionType": ["C", "P", "C", "P", "C", "P"] * 120,
            "ExecHour": [9, 10, 11, 12, 13, 14] * 120,
            "ExecWeekday": [1, 2, 3, 4, 5, 1] * 120,
        }
    )

    service = SurrogateService(
        prediction_service=_PredictionServiceStub(),
        feature_schema=DEFAULT_FEATURE_SCHEMA,
        metrics_registry=DEFAULT_METRICS_REGISTRY,
    )

    result = service.train("stub-model", frame)

    assert result.tree_depth > 0
    assert result.n_leaves > 1
    assert not result.feature_importances.empty
