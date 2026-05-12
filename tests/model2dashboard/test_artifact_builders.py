from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import shap
import pytest

from src.model2dashboard import artifact_builders
from src.model2dashboard.artifact_builders import _normalize_symbolic_equation_table
from src.model2dashboard.features import TARGET_COLUMN


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


class _FakePySRRegressor:
    fit_rows = None
    predict_rows = None
    kwargs = None

    def __init__(self, **kwargs) -> None:
        type(self).kwargs = dict(kwargs)
        self.model_selection = kwargs["model_selection"]
        self.equations_ = pd.DataFrame(
            [
                {
                    "complexity": 1,
                    "loss": 0.0,
                    "score": 1.0,
                    "equation": "Rate",
                }
            ]
        )

    def fit(self, X, y, variable_names):
        type(self).fit_rows = len(X)
        self.variable_names = list(variable_names)
        self.rate_position = self.variable_names.index("Rate")
        return self

    def predict(self, X):
        type(self).predict_rows = len(X)
        return X[:, self.rate_position]

    def get_best(self) -> pd.Series:
        return self.equations_.iloc[0]

    def sympy(self):
        return "Rate"

    def latex(self, precision):
        return "Rate"


def _surrogate_frame(index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OptionType": [
                "C" if position % 2 == 0 else "P"
                for position, _ in enumerate(index)
            ],
            "StrikePrice": [
                9000.0 + 10.0 * position for position, _ in enumerate(index)
            ],
            "UnderlyingPrice": [
                9050.0 + 8.0 * position for position, _ in enumerate(index)
            ],
            "TimeToExpiration": [10.0 + position for position, _ in enumerate(index)],
            "Rate": [0.01 + 0.001 * position for position, _ in enumerate(index)],
        },
        index=index,
    )


def test_build_shap_artifacts_use_shared_background_base_value(monkeypatch):
    def fake_predict_raw_frame(_runtime, raw_frame):
        strike = pd.to_numeric(raw_frame["StrikePrice"], errors="coerce")
        option_is_put = (raw_frame["OptionType"].astype(str) == "P").astype(float)
        return (strike * 0.1 + option_is_put).to_numpy(dtype="float64")

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
    assert local_shap.base_values.tolist() == [900.0, 900.0]


def test_build_surrogate_tree_models_fit_full_train_and_validate_full_test(monkeypatch):
    monkeypatch.setattr(
        artifact_builders.config.dashboard_models_config,
        "build_config",
        replace(
            artifact_builders.config.dashboard_models_config.build_config,
            surrogate_depths=(2,),
        ),
    )
    monkeypatch.setattr(
        artifact_builders.config.dashboard_models_config,
        "surrogate_min_samples_leaf",
        1,
    )
    train_frame = _surrogate_frame([10, 11, 12, 13])
    test_frame = _surrogate_frame([20, 21, 22])
    train_predictions = pd.Series(
        [0.11, 0.12, 0.13, 0.14],
        index=train_frame.index,
        name="PredictedVolatility",
    )
    test_predictions = pd.Series(
        [0.21, 0.22, 0.23],
        index=test_frame.index,
        name="PredictedVolatility",
    )

    tree_models = artifact_builders.build_surrogate_tree_models(
        runtime=SimpleNamespace(family_name="unit-test"),
        train_reference_frame=train_frame,
        train_predictions=train_predictions,
        test_reference_frame=test_frame,
        test_predictions=test_predictions,
    )

    tree_model = tree_models[2]
    assert tree_model.model.tree_.n_node_samples[0] == len(train_frame)
    assert len(tree_model.fidelity_frame) == len(test_frame)
    assert tree_model.fidelity_frame["model_prediction"].tolist() == [
        0.21,
        0.22,
        0.23,
    ]
    assert "full-train predictions" in tree_model.interpretation
    assert "full-test predictions" in tree_model.interpretation


def test_build_symbolic_regressor_model_fit_full_train_and_validate_full_test(
    monkeypatch,
):
    monkeypatch.setattr(artifact_builders, "PySRRegressor", _FakePySRRegressor)
    train_frame = _surrogate_frame([10, 11, 12, 13])
    test_frame = _surrogate_frame([20, 21, 22])
    train_predictions = pd.Series(
        [0.11, 0.12, 0.13, 0.14],
        index=train_frame.index,
        name="PredictedVolatility",
    )
    test_predictions = pd.Series(
        [0.21, 0.22, 0.23],
        index=test_frame.index,
        name="PredictedVolatility",
    )

    symbolic_model = artifact_builders.build_symbolic_regressor_model(
        runtime=SimpleNamespace(family_name="unit-test"),
        train_reference_frame=train_frame,
        train_predictions=train_predictions,
        test_reference_frame=test_frame,
        test_predictions=test_predictions,
    )

    assert _FakePySRRegressor.fit_rows == len(train_frame)
    assert _FakePySRRegressor.predict_rows == len(test_frame)
    assert _FakePySRRegressor.kwargs["parsimony"] == (
        artifact_builders.config.dashboard_models_config.symbolic_parsimony
    )
    assert len(symbolic_model.fidelity_frame) == len(test_frame)
    assert symbolic_model.fidelity_frame["model_prediction"].tolist() == [
        0.21,
        0.22,
        0.23,
    ]
    assert "full-train predictions" in symbolic_model.interpretation
    assert "full-test predictions" in symbolic_model.interpretation


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


def test_build_neighbors_projection_pca_returns_reusable_transform():
    artifact = artifact_builders.build_neighbors_projection_pca(
        training_reference_frame=pd.DataFrame(
            {
                "TTEYears": [0.05, 0.10, 0.20],
                "sqrtTTEYears": [0.22, 0.31, 0.45],
                "isCall": [1.0, 0.0, 1.0],
            }
        ),
        feature_names=["TTEYears", "sqrtTTEYears", "isCall"],
    )

    coords = artifact.transform_frame(
        pd.DataFrame(
            {
                "TTEYears": [0.15],
                "sqrtTTEYears": [0.39],
                "isCall": [0.0],
            }
        ),
        dimensions=3,
    )

    assert artifact.feature_names == ["TTEYears", "sqrtTTEYears", "isCall"]
    assert artifact.components.shape[1] == 3
    assert coords.shape == (1, 3)


def test_build_diagnosis_artifact_uses_full_test_for_metrics_and_heatmap(monkeypatch):
    dataset = pd.DataFrame(
        {
            TARGET_COLUMN: [0.10, 0.20, 0.40],
            "PredictedVolatility": [0.10, 0.25, 0.30],
            "Moneyness": [0.90, 1.00, 1.10],
            "TimeToExpiration": [10.0, 20.0, 30.0],
            "AbsoluteError": [0.00, 0.05, 0.10],
        }
    )

    monkeypatch.setattr(
        artifact_builders,
        "sample_frame",
        lambda frame, max_rows, random_state: frame.head(1).copy(),
    )

    artifact = artifact_builders.build_diagnosis_artifact(
        dataset_frame=dataset,
        financial_warnings=[],
    )

    assert artifact.metrics["rmse"] == pytest.approx(
        np.sqrt(((0.0**2) + (0.05**2) + (0.10**2)) / 3.0)
    )
    assert len(artifact.plot_frame) == 1
    assert int(artifact.error_heatmap["AbsoluteError"].notna().sum()) == 3
