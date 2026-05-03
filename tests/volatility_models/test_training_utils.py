import numpy as np
import pandas as pd

from src.enums.volatility_model_enums.training_phase import TrainingPhase
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
        phase: TrainingPhase = TrainingPhase.CV,
        shuffle: bool = True,
    ) -> ModelFitResult:
        _ = cls, model, model_params, X_val, phase, shuffle
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
