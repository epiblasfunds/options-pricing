import pandas as pd

from src.volatility_models.model_explainability.config import DEFAULT_FEATURE_SCHEMA
from src.volatility_models.model_explainability.services.data_provider import (
    VolatilityDataProvider,
)
from src.volatility_models.model_explainability.utils.preprocessing import (
    build_similarity_preprocessor,
)
from src.volatility_models.model_explainability.utils.validation import ensure_columns


def test_validation_and_preprocessing_pipeline(tmp_path):
    dataset_path = tmp_path / "VOLATILITY_DB.csv"
    frame = pd.DataFrame(
        {
            "ExecDatetime": [
                "2020-01-06 09:15:00.000000",
                "2020-01-07 10:30:00.000000",
            ],
            "UnderlyingExecDatetime": [
                "2020-01-06 09:10:00.000000",
                "2020-01-07 10:29:00.000000",
            ],
            "MaturityDatetime": [
                "2020-01-17 17:30:00.000000",
                "2020-01-24 17:30:00.000000",
            ],
            "TimeToExpiration": [10.0, 17.0],
            "Rate": [1.0, 1.1],
            "UnderlyingPrice": [9000.0, 9050.0],
            "StrikePrice": [8900.0, 9150.0],
            "OptionType": ["C", "P"],
            "ImpliedVolatility": [0.2, 0.25],
        }
    )
    frame.to_csv(dataset_path, sep=";", index=False)

    provider = VolatilityDataProvider(dataset_path, DEFAULT_FEATURE_SCHEMA)
    loaded = provider.load_dataset()
    ensure_columns(loaded, ["ExecHour", "ExecWeekday", "Moneyness"])

    preprocessor = build_similarity_preprocessor(
        DEFAULT_FEATURE_SCHEMA,
        [
            "TimeToExpiration",
            "Rate",
            "UnderlyingPrice",
            "StrikePrice",
            "OptionType",
            "ExecHour",
            "ExecWeekday",
        ],
    )
    transformed = preprocessor.fit_transform(
        loaded[
            [
                "TimeToExpiration",
                "Rate",
                "UnderlyingPrice",
                "StrikePrice",
                "OptionType",
                "ExecHour",
                "ExecWeekday",
            ]
        ]
    )

    assert transformed.shape[0] == 2
