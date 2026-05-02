from types import SimpleNamespace

import numpy as np
import pandas as pd
import shap

from src.api.services.model_runtime import ApiModelService
from src.dashboard.plots.local_plots import neighbors_distance_figure
from src.model2dashboard import artifact_builders
from src.model2dashboard.features import EXPLAINABILITY_FEATURE_NAMES
from src.python_models.dashboard.artifacts import StoredShapExplanation


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


def test_neighbors_distance_figure_uses_api_neighbor_index_column():
    figure = neighbors_distance_figure(
        pd.DataFrame({"index": [101, 205], "distance": [0.1, 0.3]})
    )

    assert list(figure.data[0].x) == ["101", "205"]


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
