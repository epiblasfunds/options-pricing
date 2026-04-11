import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import keras
import numpy as np
import tensorflow as tf
from keras import callbacks, layers, regularizers
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.volatility_models.data_utils import BASE_NUMERIC_FEATURE_COLS
from src.volatility_models.visualization_utils import Visualizer


class TrainnigPhase(StrEnum):
    CV = "cv"
    TRAIN_VAL = "train_val"
    FINAL_TEST = "final_test"


@dataclass
class ModelFitResult:
    model: t.Any
    train_predictions: np.ndarray
    validation_predictions: np.ndarray
    best_iteration: int | None = None
    best_score: float | None = None
    epoch_history: t.Dict[str, t.List[float]] | None = None


class VolatilityModelFamilyABC(ABC):
    RANDOM_SEED = 42

    @staticmethod
    @abstractmethod
    def get_family_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_fixed_params() -> t.Dict:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_hyperparameter_search_space() -> t.Dict:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def instantiate_model(*, input_dim: int, model_params: t.Dict):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_n_iter():
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def fit_model(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        phase: TrainnigPhase,
    ) -> ModelFitResult:
        raise NotImplementedError

    @classmethod
    def plots(cls, kwargs):
        pass

    @classmethod
    def build_model_candidates(
        cls,
    ) -> t.Dict[str, t.Dict[str, t.Any]]:
        search_space = cls.get_hyperparameter_search_space()
        if not search_space:
            return {cls.get_family_name(): cls.get_fixed_params()}

        rng = np.random.default_rng(cls.RANDOM_SEED)
        candidates: t.Dict[str, t.Dict[str, t.Any]] = {}
        for model_index in range(cls.get_n_iter()):
            sampled_params = {}
            for param_name, values in search_space.items():
                value_list = list(values)
                sampled_params[param_name] = value_list[
                    int(rng.integers(0, len(value_list)))
                ]
            candidates[f"{cls.get_family_name()}_{model_index + 1:03d}"] = {
                **cls.get_fixed_params(),
                **sampled_params,
            }
        return candidates

    @staticmethod
    def temporal_inner_split_for_early_stopping(
        X_train,
        y_train,
        valid_fraction: float = 0.20,
        min_valid_samples: int = 256,
    ):
        """
        Crea un split temporal interno dentro de X_train para el early stopping.
        El fold-valid externo queda reservado únicamente para evaluación final de métricas,
        evitando así cualquier sesgo de overfitting hacia ese conjunto.
        """
        n_samples = len(X_train)

        n_valid = max(int(np.ceil(n_samples * valid_fraction)), min_valid_samples)
        split_idx = n_samples - n_valid

        return (
            X_train[:split_idx],
            y_train[:split_idx],
            X_train[split_idx:],
            y_train[split_idx:],
        )


class LinearRegressionFamily(VolatilityModelFamilyABC):
    @staticmethod
    def get_family_name() -> str:
        return "linear_regression"

    @staticmethod
    def get_fixed_params() -> t.Dict:
        return {
            "fit_intercept": True,
            "copy_X": True,
            "n_jobs": None,
            "positive": False,
        }

    @staticmethod
    def get_hyperparameter_search_space() -> t.Dict:
        return {}

    @staticmethod
    def instantiate_model(
        *, input_dim: int, model_params: dict[str, t.Any]
    ) -> LinearRegression:
        _ = input_dim
        return LinearRegression(**model_params)

    @staticmethod
    def get_n_iter():
        return 1

    @classmethod
    def fit_model(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        phase: TrainnigPhase,
    ) -> ModelFitResult:
        _ = model_params, y_val, phase
        model.fit(X_train, y_train)
        return ModelFitResult(
            model=model,
            train_predictions=np.asarray(model.predict(X_train), dtype=float),
            validation_predictions=np.asarray(model.predict(X_val), dtype=float),
        )


class XGBoostFamily(VolatilityModelFamilyABC):
    @staticmethod
    def get_family_name() -> str:
        return "xgboost"

    @staticmethod
    def get_fixed_params() -> t.Dict:
        return {
            "booster": "gbtree",
            "objective": "reg:pseudohubererror",
            "eval_metric": "rmse",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "predictor": "auto",
            "enable_categorical": False,
            "verbosity": 0,
            "sampling_method": "uniform",
            "colsample_bylevel": 1.0,
            "colsample_bynode": 1.0,
            "max_delta_step": 0.0,
            "grow_policy": "depthwise",
            "base_score": 0.20,
        }

    @staticmethod
    def get_hyperparameter_search_space() -> t.Dict:
        return {
            "n_estimators": [400, 600, 800, 1000, 1400],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
            "max_depth": [3, 4, 5, 6],
            "min_child_weight": [1, 3, 5, 8, 10, 15, 20],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "gamma": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
            "reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
            "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
            "max_bin": [128, 256, 512],
            "num_parallel_tree": [1, 2, 4],
            "early_stopping_rounds": [30, 50, 80],
        }

    @staticmethod
    def instantiate_model(
        *, input_dim: int, model_params: dict[str, t.Any]
    ) -> LinearRegression:
        _ = input_dim
        return XGBRegressor(**model_params)

    @staticmethod
    def get_n_iter():
        return 200

    @classmethod
    def fit_model(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        phase: TrainnigPhase,
    ) -> ModelFitResult:
        _ = model_params

        if phase in {TrainnigPhase.CV, TrainnigPhase.FINAL_TEST}:
            X_fit, y_fit, X_es, y_es = cls.temporal_inner_split_for_early_stopping(
                X_train,
                y_train,
            )
            model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_es, y_es)],
                verbose=False,
            )
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

        best_iteration = getattr(model, "best_iteration", None)
        best_score = getattr(model, "best_score", None)

        return ModelFitResult(
            model=model,
            train_predictions=np.asarray(model.predict(X_train), dtype=float),
            validation_predictions=np.asarray(model.predict(X_val), dtype=float),
            best_iteration=(
                int(best_iteration)
                if best_iteration is not None and best_iteration >= 0
                else None
            ),
            best_score=float(best_score) if best_score is not None else None,
        )


class SequentialNNFamily(VolatilityModelFamilyABC):
    @staticmethod
    def get_family_name() -> str:
        return "sequential_nn"

    @staticmethod
    def get_fixed_params() -> t.Dict:
        return {
            "patience": 12,
            "epochs": 150,
            "verbose": 0,
        }

    @staticmethod
    def get_hyperparameter_search_space() -> t.Dict:
        return {
            "hidden_layers": [
                [32],
                [64],
                [128],
                [256],
                [512],
                [32, 32],
                [64, 32],
                [64, 64],
                [128, 64],
                [128, 128],
                [256, 128],
                [256, 256],
                [128, 64, 32],
                [128, 128, 64],
                [128, 128, 128],
                [256, 128, 64],
                [256, 256, 128],
                [256, 128, 64, 32],
                [256, 256, 128, 64],
                [512, 256, 128, 64],
            ],
            "dropout_rate": [0.0, 0.05, 0.1, 0.2, 0.3],
            "l2_reg": [0.0, 1e-5, 1e-4, 5e-4, 1e-3],
            "learning_rate": [5e-5, 1e-4, 5e-4, 1e-3, 3e-3],
            "batch_size": [128, 256, 512, 1024, 2048],
            "activation": ["relu", "gelu"],
            "kernel_initializer": ["he_normal", "he_uniform"],
            "use_batch_norm": [True, False],
            "use_lr_scheduler": [True, False],
            "loss": ["huber", "mse"],
        }

    @staticmethod
    def _scale_numeric_features(
        *,
        X_fit_raw: np.ndarray,
        X_valid_raw: np.ndarray,
        X_train_full_raw: np.ndarray,
        X_eval_raw: np.ndarray,
        numeric_col_indices: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        scaler = StandardScaler()
        indices = list(numeric_col_indices)

        X_fit_scaled = X_fit_raw.copy()
        X_valid_scaled = X_valid_raw.copy()
        X_train_full_scaled = X_train_full_raw.copy()
        X_eval_scaled = X_eval_raw.copy()

        X_fit_scaled[:, indices] = scaler.fit_transform(X_fit_raw[:, indices])
        X_valid_scaled[:, indices] = scaler.transform(X_valid_raw[:, indices])
        X_train_full_scaled[:, indices] = scaler.transform(X_train_full_raw[:, indices])
        X_eval_scaled[:, indices] = scaler.transform(X_eval_raw[:, indices])

        return X_fit_scaled, X_valid_scaled, X_train_full_scaled, X_eval_scaled, scaler
    
    @staticmethod
    def temporal_inner_split_for_early_stopping(
        X_train,
        y_train,
        valid_fraction: float = 0.20,
        min_valid_samples: int = 256,
    ):
        """
        Crea un split temporal interno dentro de X_train para el early stopping.
        El fold-valid externo queda reservado únicamente para evaluación final de métricas,
        evitando así cualquier sesgo de overfitting hacia ese conjunto.
        """
        n_samples = len(X_train)

        n_valid = max(int(np.ceil(n_samples * valid_fraction)), min_valid_samples)
        split_idx = n_samples - n_valid

        return X_train[:split_idx], y_train[:split_idx], X_train[split_idx:], y_train[split_idx:]

    @staticmethod
    def instantiate_model(
        *, input_dim: int, model_params: dict[str, t.Any]
    ) -> LinearRegression:
        keras.backend.clear_session()
        tf.random.set_seed(42)

        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))

        for units in model_params["hidden_layers"]:
            model.add(
                layers.Dense(
                    units,
                    activation=None,
                    kernel_initializer=model_params["kernel_initializer"],
                    kernel_regularizer=regularizers.l2(model_params["l2_reg"]),
                )
            )
            if model_params["use_batch_norm"]:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(model_params["activation"]))
            if model_params["dropout_rate"] > 0.0:
                model.add(layers.Dropout(model_params["dropout_rate"]))

        model.add(layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=model_params["learning_rate"]
            ),
            loss=model_params["loss"],
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        return model

    @staticmethod
    def get_n_iter():
        return 200

    @classmethod
    def fit_model(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        phase: TrainnigPhase,
    ) -> ModelFitResult:

        if phase in {"cv", "final_test"}:
            X_fit_raw, y_fit, X_es_raw, y_es = cls.temporal_inner_split_for_early_stopping(
                X_train,
                y_train,
            )
            (
                X_fit_scaled,
                X_es_scaled,
                X_train_scaled,
                X_val_scaled,
                scaler,
            ) = cls._scale_numeric_features(
                X_fit_raw=X_fit_raw,
                X_val_raw=X_es_raw,
                X_train_full_raw=X_train,
                X_eval_raw=X_val,
                numeric_col_indices=BASE_NUMERIC_FEATURE_COLS,
            )
            y_fit_target = y_fit
            validation_data = (X_es_scaled, y_es)
        else:
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_val_scaled = X_val.copy()
            indices = list(BASE_NUMERIC_FEATURE_COLS)
            X_train_scaled[:, indices] = scaler.fit_transform(X_train[:, indices])
            X_val_scaled[:, indices] = scaler.transform(X_val[:, indices])
            X_fit_scaled = X_train_scaled
            y_fit_target = y_train
            validation_data = (X_val_scaled, y_val)

        history = model.fit(
            X_fit_scaled,
            y_fit_target,
            epochs=model_params["epochs"],
            batch_size=model_params["batch_size"],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=model_params["verbose"],
        )

        val_rmse_history = history.history.get("val_rmse", [])
        best_epoch = int(np.argmin(val_rmse_history)) + 1 if val_rmse_history else None
        best_score = float(np.min(val_rmse_history)) if val_rmse_history else None

        return ModelFitResult(
            model=model,
            train_predictions=model.predict(X_train_scaled, verbose=0).flatten(),
            validation_predictions=model.predict(X_val_scaled, verbose=0).flatten(),
            best_iteration=best_epoch,
            best_score=best_score,
            epoch_history={
                "rmse": [float(value) for value in history.history.get("rmse", [])],
                "val_rmse": [float(value) for value in val_rmse_history],
            },
            preprocessor=scaler,
        )

    @classmethod
    def plots(cls, kwargs):
        args = {
            "training_registry": kwargs.get("training_registry"),
            "best_model_row": kwargs.get("best_model_row"),
            "family_name": cls.get_family_name(),
        }
        Visualizer.plot_nn_learning_curves(**args)
