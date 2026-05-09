from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import shap
from src.api.models import ModelRequest, PredictionFeatures, ApiOptionTypeEnum
from src.api.services.cache import ApiModelCache

from src.api.services.model_runtime import ApiModelService
from src.dashboard.plots.local_plots import neighbors_projection_figure
from src.model2dashboard import artifact_builders
from src.model2dashboard.features import EXPLAINABILITY_FEATURE_NAMES
from src.python_models.dashboard.artifacts import StoredShapExplanation
from src.enums.volatility_model_enums import ModelNameEnum


class _LinearModel:
    def predict(self, matrix):
        return 0.01 * matrix[:, 0] + 0.02 * matrix[:, 1]


class _FakeExplainer:
    def __init__(
        self,
        model,
        masker,
        algorithm,
        feature_names,
        seed,
    ) -> None:
        self.model = model
        self.masker = masker
        self.feature_names = feature_names

    def __call__(self, encoded_frame, max_evals, silent):
        predictions = np.asarray(self.model(encoded_frame), dtype="float64").reshape(-1)
        base_value = float(
            np.asarray(self.model(self.masker), dtype="float64").reshape(-1).mean()
        )
        values = np.zeros(
            (len(encoded_frame), len(self.feature_names)), dtype="float64"
        )
        values[:, 0] = predictions - base_value
        return shap.Explanation(
            values=values,
            base_values=np.full(len(encoded_frame), base_value, dtype="float64"),
            data=encoded_frame.to_numpy(),
            feature_names=self.feature_names,
        )


def _raw_frame():
    return pd.DataFrame(
        [
            {
                "OptionType": "C",
                "StrikePrice": 10000.0,
                "UnderlyingPrice": 10100.0,
                "TimeToExpiration": 30.0,
                "Rate": 0.02,
            },
            {
                "OptionType": "P",
                "StrikePrice": 9900.0,
                "UnderlyingPrice": 10000.0,
                "TimeToExpiration": 45.0,
                "Rate": 0.03,
            },
        ]
    )


def _fake_predict_raw_frame(_runtime, raw_frame):
    strike = pd.to_numeric(raw_frame["StrikePrice"], errors="coerce")
    option_is_put = (raw_frame["OptionType"].astype(str) == "P").astype(float)
    return (strike * 0.1 + option_is_put).to_numpy(dtype="float64")


def _request(*, option_type=ApiOptionTypeEnum.CALL, contract_code="CIBX 10000X26", implied_volatility=0.2):
    return ModelRequest(
        modelo=ModelNameEnum.RANDOM_FOREST,
        caracteristicas=PredictionFeatures(
            optionContractCode=contract_code,
            optionType=option_type,
            strikePrice=10000.0,
            underlyingPrice=10100.0,
            timeToExpiration=30.0,
            rate=0.02,
            impliedVolatility=implied_volatility,
        ),
    )


def test_runtime_shap_explanation_uses_manual_row_not_stored_sample():
    raw = _raw_frame()
    runtime = SimpleNamespace(
        model=_LinearModel(),
        model_input_features=["TTEYears", "sqrtTTEYears"],
        scaler=None,
        is_keras=False,
    )
    dashboard_model = SimpleNamespace(
        dataset_frame=raw,
        raw_feature_names=list(EXPLAINABILITY_FEATURE_NAMES),
        metadata={"explainability_feature_names": list(EXPLAINABILITY_FEATURE_NAMES)},
    )

    stored = ApiModelService._runtime_shap_explanation(
        training_runtime=runtime,
        dashboard_model=dashboard_model,
        raw_frame=raw.iloc[[0]],
        prediction=0.123,
    )

    assert stored.index == [0]
    assert stored.method == "shap.Explainer(permutation, runtime)"
    assert stored.feature_names == list(EXPLAINABILITY_FEATURE_NAMES)
    assert stored.values.shape == (1, len(EXPLAINABILITY_FEATURE_NAMES))
    assert stored.predictions.tolist() == [0.123]


def test_runtime_shap_explanation_matches_persisted_background_base_value(monkeypatch):
    raw = _raw_frame()
    runtime = SimpleNamespace()
    predictions = pd.Series([0.2, 0.3], index=raw.index, name="PredictedVolatility")
    dashboard_model = SimpleNamespace(
        dataset_frame=raw,
        raw_feature_names=list(EXPLAINABILITY_FEATURE_NAMES),
        metadata={"explainability_feature_names": list(EXPLAINABILITY_FEATURE_NAMES)},
    )

    monkeypatch.setattr(artifact_builders.shap, "Explainer", _FakeExplainer)
    monkeypatch.setattr(artifact_builders, "predict_raw_frame", _fake_predict_raw_frame)
    monkeypatch.setattr(
        "src.api.services.model_runtime.shap.Explainer",
        _FakeExplainer,
    )
    monkeypatch.setattr(
        "src.api.services.model_runtime.predict_raw_frame",
        _fake_predict_raw_frame,
    )
    monkeypatch.setattr(
        artifact_builders,
        "sample_frame",
        lambda frame, max_rows, random_state: frame.head(1),
    )
    monkeypatch.setattr(
        "src.api.services.model_runtime.sample_frame",
        lambda frame, max_rows, random_state: frame.head(1),
    )

    _global_shap, local_shap = artifact_builders.build_shap_artifacts(
        runtime=runtime,
        dataset_frame=raw,
        raw_frame=raw,
        predictions=predictions,
        sample_indices=[1],
    )
    runtime_stored = ApiModelService._runtime_shap_explanation(
        training_runtime=runtime,
        dashboard_model=dashboard_model,
        raw_frame=raw.iloc[[1]],
        prediction=0.3,
    )

    assert float(local_shap.base_values[0]) == float(runtime_stored.base_values[0])


def test_neighbors_projection_figure_uses_api_neighbor_index_column():
    sample = pd.DataFrame(
        {"StrikePrice": [100.0], "UnderlyingPrice": [101.0], "distance": [0.0]}
    )
    figure = neighbors_projection_figure(
        sample,
        pd.DataFrame(
            {
                "index": [101, 205],
                "distance": [0.1, 0.3],
                "StrikePrice": [99.0, 103.0],
                "UnderlyingPrice": [100.0, 104.0],
            }
        ),
        feature_names=["StrikePrice", "UnderlyingPrice"],
    )

    assert figure.data[1].customdata[0][0] == "101"
    assert figure.data[1].customdata[1][0] == "205"


def test_runtime_neighbors_use_training_reference_frame():
    service = ApiModelService.__new__(ApiModelService)
    dashboard_model = SimpleNamespace(
        transformed_feature_names=["StrikePrice", "UnderlyingPrice"],
        metadata={"model_input_features": ["StrikePrice", "UnderlyingPrice"]},
        training_reference_frame=pd.DataFrame(
            {
                "StrikePrice": [100.0, 220.0],
                "UnderlyingPrice": [101.0, 221.0],
                "ImpliedVolatility": [0.2, 0.5],
            },
            index=[50, 60],
        ),
        dataset_frame=pd.DataFrame(
            {
                "StrikePrice": [1000.0, 2000.0],
                "UnderlyingPrice": [1001.0, 2001.0],
                "ImpliedVolatility": [1.0, 2.0],
            },
            index=[5, 6],
        ),
    )
    sample = pd.DataFrame({"StrikePrice": [102.0], "UnderlyingPrice": [103.0]})

    neighbors = service._find_runtime_neighbors(
        dashboard_model=dashboard_model,
        sample_frame=sample,
        k=1,
    )

    assert neighbors.index.tolist() == [50]


def test_stored_shap_to_result_uses_waterfall_final_prediction():
    stored = StoredShapExplanation(
        method="shap.Explainer(permutation, runtime)",
        feature_names=["StrikePrice", "Rate"],
        index=[7],
        values=np.asarray([[0.20, -0.05]]),
        base_values=np.asarray([0.30]),
        data=np.asarray([[9100.0, -0.6]]),
        display_data=np.asarray([[9100.0, -0.6]]),
        predictions=np.asarray([9.99]),
        mean_abs_shap={"StrikePrice": 0.20, "Rate": 0.05},
    )

    result = ApiModelService._stored_shap_to_result(stored)
    payload = ApiModelService._stored_shap_to_payload(stored)

    assert result.predictions.loc[7] == 0.45
    assert payload["predictions"] == [0.45]
    assert payload["display_data"] == [[9100.0, -0.6]]


def test_sample_explainability_builds_complete_runtime_payload(monkeypatch):
    service = ApiModelService.__new__(ApiModelService)
    service.neighbors_k = 2
    service.feature_schema = object()
    loaded = SimpleNamespace(training_runtime=object(), dashboard_model=object())
    stored = StoredShapExplanation(
        method="runtime",
        feature_names=["StrikePrice"],
        index=[0],
        values=np.asarray([[0.2]]),
        base_values=np.asarray([0.1]),
        data=np.asarray([[10000.0]]),
        display_data=np.asarray([[10000.0]]),
        predictions=np.asarray([0.3]),
        mean_abs_shap={"StrikePrice": 0.2},
    )
    monkeypatch.setattr(service, "_load_model", lambda model_name: loaded)
    monkeypatch.setattr(
        "src.api.services.model_runtime.predict_raw_frame",
        lambda runtime, raw_frame: np.asarray([0.31]),
    )
    monkeypatch.setattr(
        service,
        "_find_runtime_neighbors",
        lambda **kwargs: pd.DataFrame({"distance": [np.float64(0.1)]}, index=[10]),
    )
    monkeypatch.setattr(
        service,
        "_runtime_shap_explanation",
        lambda **kwargs: stored,
    )
    monkeypatch.setattr(
        service,
        "_stored_shap_to_result",
        lambda _stored: SimpleNamespace(predictions=pd.Series([0.45], index=[0])),
    )
    monkeypatch.setattr(
        service,
        "_stored_shap_to_payload",
        lambda _stored: {"value": np.float64(1.2)},
    )
    monkeypatch.setattr(
        "src.api.services.model_runtime.waterfall_image",
        lambda explanation_result, row_index, feature_schema: "img-src",
    )

    payload = service.sample_explainability(_request())

    assert payload["modelo"] == "random_forest"
    assert payload["prediction"] == 0.45
    assert payload["waterfall_image"] == "img-src"
    assert payload["neighbors"][0]["index"] == 10
    assert payload["neighbor_distances"] == [{"row_id": "10", "distance": 0.1}]
    assert payload["local_explanation"] == {"value": 1.2}


def test_request_to_raw_frame_and_sample_frame_include_optional_fields():
    service = ApiModelService.__new__(ApiModelService)

    raw_with_all_fields = service._request_to_raw_frame(_request())
    raw_without_optionals = service._request_to_raw_frame(
        _request(option_type=ApiOptionTypeEnum.PUT, contract_code="", implied_volatility=None)
    )
    sample = service._build_dashboard_sample_frame(raw_with_all_fields, prediction=0.25)

    assert raw_with_all_fields.loc[0, "OptionType"] == "C"
    assert raw_with_all_fields.loc[0, "OptionContractCode"] == "CIBX 10000X26"
    assert "ImpliedVolatility" in raw_with_all_fields.columns
    assert raw_without_optionals.loc[0, "OptionType"] == "P"
    assert "OptionContractCode" not in raw_without_optionals.columns
    assert "ImpliedVolatility" not in raw_without_optionals.columns
    assert sample.loc[0, "PredictedVolatility"] == 0.25
    assert "Residual" in sample.columns
    assert "AbsoluteError" in sample.columns


def test_load_model_and_uncached_loading_delegate_to_cache_and_disk(monkeypatch):
    service = ApiModelService.__new__(ApiModelService)
    service.cache = ApiModelCache(max_entries=2)
    service.storage = SimpleNamespace(
        prepare_model=lambda model_name: SimpleNamespace(
            trained_models_dir="trained-dir",
            retrained_metadata_dir="metadata-dir",
            dashboard_model_dir="dashboard-dir",
        )
    )

    monkeypatch.setattr(
        "src.api.services.model_runtime.load_training_runtime",
        lambda **kwargs: {"family": kwargs["family_name"]},
    )
    monkeypatch.setattr(
        "src.api.services.model_runtime.DashboardModel.load",
        lambda path: {"dashboard": path},
    )

    loaded = service._load_uncached("random_forest")
    cached = service._load_model("random_forest")

    assert loaded.training_runtime == {"family": "random_forest"}
    assert loaded.dashboard_model == {"dashboard": "dashboard-dir"}
    assert cached.training_runtime == {"family": "random_forest"}


def test_neighbor_feature_names_and_runtime_neighbors_cover_empty_and_metadata_fallback():
    service = ApiModelService.__new__(ApiModelService)
    dashboard_model = SimpleNamespace(
        transformed_feature_names=[],
        metadata={"model_input_features": ["StrikePrice"]},
        training_reference_frame=pd.DataFrame({"Other": [1.0]}, index=[1]),
    )
    sample = pd.DataFrame({"StrikePrice": [100.0]})

    assert service._neighbor_feature_names(dashboard_model, sample) == []
    assert service._find_runtime_neighbors(
        dashboard_model=dashboard_model,
        sample_frame=sample,
        k=1,
    ).empty


def test_runtime_shap_explanation_requires_at_least_one_feature(monkeypatch):
    monkeypatch.setattr(
        "src.api.services.model_runtime.EXPLAINABILITY_FEATURE_NAMES",
        [],
    )

    with pytest.raises(RuntimeError, match="without features"):
        ApiModelService._runtime_shap_explanation(
            training_runtime=object(),
            dashboard_model=SimpleNamespace(
                metadata={"explainability_feature_names": []},
                raw_feature_names=[],
                dataset_frame=pd.DataFrame(),
            ),
            raw_frame=pd.DataFrame([{"OptionType": "C"}]),
            prediction=0.1,
        )


def test_predict_explainability_values_and_json_safe_cover_supported_types(monkeypatch):
    class _Encoder:
        def reconstruct_raw_frame(self, values):
            return pd.DataFrame({"StrikePrice": [values[0][0]]})

    monkeypatch.setattr(
        "src.api.services.model_runtime.predict_raw_frame",
        lambda runtime, raw_frame: np.asarray([raw_frame.loc[0, "StrikePrice"]]),
    )

    predicted = ApiModelService._predict_explainability_values(
        object(),
        _Encoder(),
        [[12.5]],
    )
    converted = ApiModelService._json_safe(
        {
            "enum": ApiOptionTypeEnum.CALL,
            "scalar": np.float64(1.5),
            "list": [np.int64(2)],
            "tuple": (np.int64(3),),
            "array": np.asarray([4.0]),
            "timestamp_naive": pd.Timestamp("2026-05-09 10:00:00"),
            "timestamp_tz": pd.Timestamp("2026-05-09 10:00:00", tz="Europe/Madrid"),
            "datetime": datetime(2026, 5, 9, 10, 0, tzinfo=timezone.utc),
            "inf": float("inf"),
            "nan": np.nan,
        }
    )

    assert predicted.tolist() == [12.5]
    assert converted["enum"] == "CALL"
    assert converted["scalar"] == 1.5
    assert converted["list"] == [2]
    assert converted["tuple"] == [3]
    assert converted["array"] == [4.0]
    assert converted["timestamp_naive"] == "2026-05-09T10:00:00"
    assert converted["timestamp_tz"].endswith("+00:00")
    assert converted["datetime"].endswith("+00:00")
    assert converted["inf"] is None
    assert converted["nan"] is None


def test_frame_records_includes_index_column():
    frame = pd.DataFrame({"distance": [0.1]}, index=[55])

    records = ApiModelService._frame_records(frame)

    assert records == [{"index": 55, "distance": 0.1}]
