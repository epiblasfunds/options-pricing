from types import SimpleNamespace

import pandas as pd

from src.dashboard.domain import build_feature_schema
from src.dashboard.services.sample_explainability import NeighborsService


def test_rank_neighbors_uses_training_reference_frame():
    dashboard_model = SimpleNamespace(
        transformed_feature_names=["StrikePrice", "UnderlyingPrice"],
        metadata={"model_input_features": ["StrikePrice", "UnderlyingPrice"]},
        training_reference_frame=pd.DataFrame(
            {
                "StrikePrice": [100.0, 250.0],
                "UnderlyingPrice": [101.0, 251.0],
                "ImpliedVolatility": [0.2, 0.6],
            },
            index=[10, 20],
        ),
        dataset_frame=pd.DataFrame(
            {
                "StrikePrice": [1000.0, 2000.0],
                "UnderlyingPrice": [1001.0, 2001.0],
                "ImpliedVolatility": [1.0, 2.0],
            },
            index=[1, 2],
        ),
    )
    prediction_service = SimpleNamespace(
        load_bundle=lambda model_id: SimpleNamespace(dashboard_model=dashboard_model)
    )
    service = NeighborsService(
        prediction_service=prediction_service,
        feature_schema=build_feature_schema(),
    )

    ranked = service.rank_neighbors(
        "model",
        pd.DataFrame({"StrikePrice": [102.0], "UnderlyingPrice": [103.0]}),
    )

    assert ranked.index.tolist()[0] == 10
