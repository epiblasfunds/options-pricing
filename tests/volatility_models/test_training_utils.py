import numpy as np
import pandas as pd

from src.enums.volatility_model_enums.training_phase import TrainingPhase
from src.python_models.volatility_models.volatility_model_family import MONEYNESS_COL
from src.python_models.volatility_models.volatility_model_family import ModelFitResult
from src.python_models.volatility_models.volatility_model_family import (
    VolatilityModelFamilyABC,
)
from src.volatility_models.training_utils import Trainer


class _EchoFamily(VolatilityModelFamilyABC):
    @staticmethod
    def get_family_name() -> str:
        return "echo"

    @staticmethod
    def get_fixed_params() -> dict:
        return {}

    @staticmethod
    def get_hyperparameter_search_space() -> dict:
        return {}

    @staticmethod
    def instantiate_model(*, input_dim: int, model_params: dict):
        _ = input_dim, model_params
        return object()

    @staticmethod
    def get_n_iter():
        return 1

    @classmethod
    def fit_model(
        cls,
        *,
        model,
        model_params: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        progressive_training: bool,
        phase: TrainingPhase = TrainingPhase.CV,
        shuffle: bool = True,
    ) -> ModelFitResult:
        _ = cls, model, model_params, X_val, progressive_training, phase, shuffle
        return ModelFitResult(
            model=model,
            train_predictions=np.asarray(y_train, dtype=float),
            validation_predictions=np.zeros(len(X_val), dtype=float),
        )

    @staticmethod
    def save_model(
        model,
        scaler=None,
        family_name_override: str | None = None,
    ) -> None:
        _ = model, scaler, family_name_override


def test_fit_model_restores_train_prediction_order_after_shuffle():
    trainer = Trainer(_EchoFamily)
    X_train = pd.DataFrame({"feature": [10.0, 20.0, 30.0, 40.0]})
    y_train = np.asarray([1.0, 2.0, 3.0, 4.0])
    X_val = pd.DataFrame({"feature": [50.0]})

    np.random.seed(7)
    fit_result, _ = trainer.fit_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        phase=TrainingPhase.FINAL_TEST,
        model_params={},
        shuffle=True,
    )

    assert fit_result.train_predictions.tolist() == y_train.tolist()


def test_build_progressive_sample_weights_prioritizes_atm_and_normalizes_mean():
    X_train = pd.DataFrame(
        {
            MONEYNESS_COL: [-0.50, -0.20, -0.05, 0.03, 0.18, 0.45],
            "feature": np.arange(6, dtype=float),
        }
    )

    weights = _EchoFamily.build_progressive_sample_weights(X_train)

    assert np.isclose(weights.mean(), 1.0)
    atm_idx = int(np.argmin(np.abs(X_train[MONEYNESS_COL].to_numpy())))
    otm_idx = int(np.argmax(np.abs(X_train[MONEYNESS_COL].to_numpy())))
    assert weights[atm_idx] > weights[otm_idx]


def test_build_progressive_phase_datasets_grow_monotonically_and_hold_out_eval_split():
    n_rows = 20
    X_train = pd.DataFrame(
        {
            MONEYNESS_COL: np.linspace(-0.5, 0.5, n_rows),
            "feature": np.arange(n_rows, dtype=float),
        }
    )
    y_train = np.arange(n_rows, dtype=float)

    phase_datasets, X_full, y_full, X_es, y_es = _EchoFamily.build_progressive_phase_datasets(
        X_train,
        y_train,
    )

    phase_sizes = [len(x_phase) for x_phase, _ in phase_datasets]

    assert len(phase_datasets) > 1
    assert phase_sizes == sorted(phase_sizes)
    assert phase_sizes[-1] + len(X_es) == len(X_train)
    assert len(X_full) == len(y_full) == len(X_train)
    assert len(X_es) == len(y_es)
