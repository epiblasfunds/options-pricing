from types import SimpleNamespace

import numpy as np
import pandas as pd
import shap

from src.model2dashboard import artifact_builders
from src.model2dashboard.artifact_builders import _normalize_symbolic_equation_table


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


class _FakeRegressor:
    def __init__(self, equations: pd.DataFrame, best_equation: str):
        self.equations_ = equations
        self._best_equation = best_equation

    def get_best(self) -> pd.Series:
        return self.equations_.loc[
            self.equations_["equation"] == self._best_equation
        ].iloc[0]


def test_build_shap_artifacts_use_shared_background_base_value(monkeypatch):
    def fake_predict_raw_frame(_runtime, raw_frame):
        exec_dt = pd.to_datetime(raw_frame["ExecDatetime"], errors="coerce")
        option_is_put = (raw_frame["OptionType"].astype(str) == "P").astype(float)
        return (
            exec_dt.dt.hour.astype(float) * 100.0
            + raw_frame["Quantity"].astype(float)
            + option_is_put
        ).to_numpy(dtype="float64")

    monkeypatch.setattr(artifact_builders.shap, "Explainer", _FakeExplainer)
    monkeypatch.setattr(artifact_builders, "predict_raw_frame", fake_predict_raw_frame)
    monkeypatch.setattr(
        artifact_builders,
        "sample_frame",
        lambda frame, max_rows, random_state: frame.head(1),
    )

    raw_frame = pd.DataFrame(
        [
            {
                "ExecDatetime": "2026-04-22T10:00:00+00:00",
                "OptionContractCode": "CIBX 9000X26",
                "OptionType": "C",
                "Quantity": 1,
                "StrikePrice": 9000.0,
                "UnderlyingPrice": 9050.0,
                "TimeToExpiration": 15.0,
                "Rate": -0.5,
                "ImpliedVolatility": 0.20,
            },
            {
                "ExecDatetime": "2026-04-23T15:00:00+00:00",
                "OptionContractCode": "PIBX 9100X26",
                "OptionType": "P",
                "Quantity": 7,
                "StrikePrice": 9100.0,
                "UnderlyingPrice": 9000.0,
                "TimeToExpiration": 20.0,
                "Rate": -0.6,
                "ImpliedVolatility": 0.21,
            },
        ],
        index=[10, 11],
    )
    dataset_frame = raw_frame.copy()
    predictions = pd.Series(
        [0.2, 0.3], index=raw_frame.index, name="PredictedVolatility"
    )

    _global_shap, local_shap = artifact_builders.build_shap_artifacts(
        runtime=SimpleNamespace(),
        dataset_frame=dataset_frame,
        raw_frame=raw_frame,
        predictions=predictions,
        sample_indices=[10, 11],
    )

    assert local_shap.index == [10, 11]
    assert local_shap.base_values.tolist() == [1303.5, 1303.5]


def test_normalize_symbolic_equation_table_persists_at_least_five_candidates():
    equations = pd.DataFrame(
        [
            {"complexity": 1, "loss": 0.50, "score": 0.00, "equation": "c0"},
            {"complexity": 2, "loss": 0.40, "score": 0.01, "equation": "c1"},
            {"complexity": 2, "loss": 0.39, "score": 0.02, "equation": "c1b"},
            {"complexity": 3, "loss": 0.30, "score": 0.03, "equation": "c2"},
            {"complexity": 4, "loss": 0.25, "score": 0.04, "equation": "c3"},
            {"complexity": 5, "loss": 0.20, "score": 0.05, "equation": "c4"},
            {"complexity": 6, "loss": 0.15, "score": 0.06, "equation": "best_eq"},
            {"complexity": 7, "loss": 0.14, "score": 0.07, "equation": "c6"},
        ]
    )
    regressor = _FakeRegressor(equations=equations, best_equation="best_eq")

    normalized = _normalize_symbolic_equation_table(regressor, min_equations=5)

    assert len(normalized) >= 5
    assert normalized["equation"].is_unique
    assert normalized["selected"].sum() == 1
    assert "best_eq" in set(normalized["equation"])
