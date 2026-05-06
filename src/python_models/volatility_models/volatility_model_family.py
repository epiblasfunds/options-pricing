import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod

import joblib
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, ops, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config.config import VOLATILITY_TRAINED_MODELS_DIR_PATH
from src.enums.volatility_model_enums.training_phase import TrainingPhase
from src.volatility_models.data_utils import (
    BASE_NUMERIC_FEATURE_COLS,
)
from src.volatility_models.visualization_utils import Visualizer
from src.config.config import config

logger = logging.getLogger(__name__)

MONEYNESS_COL = config.volatility_models_config.training_data_config.moneyness_column
N_SEGMENTS = int(config.volatility_models_config.training_data_config.n_segments)


@dataclass
class ModelFitResult:
    model: t.Any
    train_predictions: np.ndarray
    validation_predictions: np.ndarray
    best_iteration: int | None = None
    best_score: float | None = None
    epoch_history: t.Dict[str, t.List[float]] | None = None
    feature_scaler: StandardScaler | None = None


@keras.utils.register_keras_serializable(package="volatility_models")
class TensorTrainLayer(keras.layers.Layer):
    """Tensor-network feature extractor based on a tensor-train factorization."""

    def __init__(
        self,
        input_dims: t.Sequence[int],
        tt_ranks: t.Sequence[int],
        output_dim: int,
        activation: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dims = tuple(int(dim) for dim in input_dims)
        self.tt_ranks = tuple(int(rank) for rank in tt_ranks)
        self.output_dim = int(output_dim)
        self.activation = keras.activations.get(activation)

        if len(self.tt_ranks) != len(self.input_dims) + 1:
            raise ValueError("tt_ranks must have length len(input_dims) + 1.")
        if self.tt_ranks[0] != 1 or self.tt_ranks[-1] != 1:
            raise ValueError("TensorTrainLayer requires TT boundary ranks [1, ..., 1].")
        if any(dim <= 0 for dim in self.input_dims):
            raise ValueError("input_dims must contain only positive integers.")
        if any(rank <= 0 for rank in self.tt_ranks):
            raise ValueError("tt_ranks must contain only positive integers.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")

    def build(self, input_shape):
        expected_shape = tuple(self.input_dims)
        received_shape = tuple(
            int(dim) if dim is not None else None for dim in input_shape[1:]
        )

        if len(received_shape) != len(expected_shape):
            raise ValueError(
                f"Expected a tensor of order {len(expected_shape)}, but got shape={input_shape}."
            )

        for expected_dim, received_dim in zip(expected_shape, received_shape):
            if received_dim is not None and received_dim != expected_dim:
                raise ValueError(
                    f"Shape incompatible. Esperado {expected_shape} y recibido {received_shape}."
                )

        initializer = keras.initializers.GlorotUniform()
        self.cores: list[tuple[t.Any, ...]] = []
        for out_idx in range(self.output_dim):
            output_chain = []
            for site_idx, mode_dim in enumerate(self.input_dims):
                output_chain.append(
                    self.add_weight(
                        name=f"tt_core_{out_idx}_{site_idx}",
                        shape=(
                            self.tt_ranks[site_idx],
                            mode_dim,
                            self.tt_ranks[site_idx + 1],
                        ),
                        initializer=initializer,
                        trainable=True,
                    )
                )
            self.cores.append(tuple(output_chain))
        super().build(input_shape)

    def _contract_chain(self, inputs, chain: tuple[t.Any, ...]):
        state = ops.tensordot(inputs, chain[0], axes=[[1], [1]])
        state = ops.squeeze(state, axis=-2)
        for core in chain[1:]:
            state = ops.tensordot(state, core, axes=[[1, -1], [1, 0]])
        return ops.reshape(state, (-1,))

    def call(self, inputs):
        outputs = ops.stack(
            [self._contract_chain(inputs, chain) for chain in self.cores],
            axis=-1,
        )
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dims": list(self.input_dims),
                "tt_ranks": list(self.tt_ranks),
                "output_dim": self.output_dim,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return config


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
    def _segment_sorted_indices(
        cls,
        X_train: pd.DataFrame,
    ) -> list[np.ndarray]:
        sorted_positions = np.argsort(np.abs(X_train[MONEYNESS_COL].to_numpy()))
        segment_size = len(X_train) // N_SEGMENTS
        segments: list[np.ndarray] = []
        for idx in range(N_SEGMENTS - 1):
            start = idx * segment_size
            end = (idx + 1) * segment_size
            segments.append(sorted_positions[start:end])
        segments.append(sorted_positions[(N_SEGMENTS - 1) * segment_size:])
        return segments

    @classmethod
    def build_progressive_phase_datasets(
        cls,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> tuple[
        list[tuple[np.ndarray, np.ndarray]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        segment_indices = cls._segment_sorted_indices(X_train)
        sorted_indices = np.concatenate(segment_indices)
        segment_ids = np.empty(len(sorted_indices), dtype=int)
        start = 0
        for segment_id, segment in enumerate(segment_indices):
            end = start + len(segment)
            segment_ids[start:end] = segment_id
            start = end

        X_sorted = X_train.iloc[sorted_indices].to_numpy()
        y_train = np.asarray(y_train)
        y_sorted = y_train[sorted_indices]

        rng = np.random.default_rng(cls.RANDOM_SEED)
        permutation = rng.permutation(len(X_sorted))
        X_shuffled = X_sorted[permutation]
        y_shuffled = y_sorted[permutation]
        segment_ids = segment_ids[permutation]

        n_samples = len(X_shuffled)
        n_valid = max(int(np.ceil(n_samples * 0.20)), 256)
        if n_samples > 1:
            n_valid = min(n_valid, n_samples - 1)
        split_idx = n_samples - n_valid

        X_fit = X_shuffled[:split_idx]
        y_fit = y_shuffled[:split_idx]
        segment_ids_fit = segment_ids[:split_idx]
        X_es = X_shuffled[split_idx:]
        y_es = y_shuffled[split_idx:]

        phase_datasets: list[tuple[np.ndarray, np.ndarray]] = []
        for segment_id in range(N_SEGMENTS):
            mask = segment_ids_fit <= segment_id
            phase_datasets.append((X_fit[mask], y_fit[mask]))

        return phase_datasets, X_train.to_numpy(), y_train, X_es, y_es

    @classmethod
    def build_progressive_sample_weights(
        cls,
        X_train: pd.DataFrame,
    ) -> np.ndarray:
        sample_weights = np.empty(len(X_train), dtype=float)
        for segment_id, segment_indices in enumerate(cls._segment_sorted_indices(X_train)):
            sample_weights[segment_indices] = float(N_SEGMENTS - segment_id)
        sample_weights /= sample_weights.mean()
        return sample_weights

    @classmethod
    def fit_model(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        progressive_training: bool,
        phase: TrainingPhase = TrainingPhase.CV,
        shuffle: bool = True,
    ) -> ModelFitResult:
        inverse_indices = None
        if shuffle:
            rng = np.random.default_rng(cls.RANDOM_SEED)
            indices = rng.permutation(len(X_train))
            inverse_indices = np.empty_like(indices)
            inverse_indices[indices] = np.arange(len(indices))
            X_train = X_train.iloc[indices]
            y_train = y_train[indices]

        fit_result = cls._fit_model_family(
            model=model,
            model_params=model_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            progressive_training=progressive_training,
            phase=phase,
            # if it's progressive training, then don't shuffle
            shuffle=False,
        )

        if inverse_indices is not None:
            fit_result.train_predictions = np.asarray(
                fit_result.train_predictions,
                dtype=float,
            )[inverse_indices]

        return fit_result

    @classmethod
    @abstractmethod
    def _fit_model_family(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        progressive_training: bool,
        phase: TrainingPhase = TrainingPhase.CV,
        shuffle: bool = True,
    ) -> ModelFitResult:
        raise NotImplementedError

    @classmethod
    def plots(cls, kwargs):
        pass

    @classmethod
    def get_model_path(cls):
        return (
            VOLATILITY_TRAINED_MODELS_DIR_PATH
            / f"{cls.get_family_name()}{cls.MODEL_EXTENSION}"
        )

    @staticmethod
    @abstractmethod
    def save_model(
        model: t.Any,
        scaler: StandardScaler | None = None,
        family_name_override: str | None = None,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def build_model_candidates(
        cls,
    ) -> t.Dict[str, t.Dict[str, t.Any]]:
        search_space = cls.get_hyperparameter_search_space()
        max_configurations = cls.get_max_configurations()

        if not search_space:
            logger.info(
                "Family '%s' has no hyperparameters to explore."
                " A single model will be trained with the fixed parameters.",
                cls.get_family_name(),
            )
            return {cls.get_family_name(): cls.get_fixed_params()}

        requested_iterations = int(cls.get_n_iter())
        explored_configurations = min(requested_iterations, max_configurations)
        explored_ratio = (
            (100.0 * explored_configurations / max_configurations)
            if max_configurations > 0
            else 0.0
        )

        logger.info(
            "Hyperparameter exploration for '%s': %s/%s configurations (%.2f%%).",
            cls.get_family_name(),
            explored_configurations,
            max_configurations,
            explored_ratio,
        )

        rng = np.random.default_rng(cls.RANDOM_SEED)
        candidates: t.Dict[str, t.Dict[str, t.Any]] = {}
        for model_index in range(requested_iterations):
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

    @classmethod
    def get_max_configurations(cls) -> int:
        search_space = cls.get_hyperparameter_search_space()
        if not search_space:
            return 1

        cardinalities = [len(list(values)) for values in search_space.values()]
        if any(cardinality <= 0 for cardinality in cardinalities):
            raise ValueError(
                f"Family '{cls.get_family_name()}' has hyperparameters with an empty domain."
            )
        return int(prod(cardinalities))

    @staticmethod
    def temporal_inner_split_for_early_stopping(
        X_train,
        y_train,
        valid_fraction: float = 0.20,
        min_valid_samples: int = 256,
    ):
        """
        Creates an internal split within X_train for early stopping, taking the
        last `valid_fraction` rows by position as the inner validation set.

        When called after a global shuffle (shuffle=True in fit_model), the split
        is effectively random, not temporal. When called on ordered data (e.g.
        progressive training where order is meaningful), the split is temporal.
        The outer val/test folds are never touched here.
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

    @staticmethod
    def transform_numeric_features(
        *,
        X_raw: np.ndarray,
        numeric_col_indices: tuple[int, ...],
        scaler: StandardScaler,
    ) -> np.ndarray:
        X_scaled = X_raw.copy()
        indices = list(numeric_col_indices)
        X_scaled[:, indices] = scaler.transform(X_raw[:, indices])
        return X_scaled


class LinearRegressionFamily(VolatilityModelFamilyABC):
    MODEL_EXTENSION = ".joblib"

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
    def _fit_model_family(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        progressive_training: bool,
        phase: TrainingPhase = TrainingPhase.CV,
        shuffle: bool = True,
    ) -> ModelFitResult:
        _, _, _ = model_params, phase, shuffle
        fit_kwargs = {}
        if progressive_training:
            fit_kwargs["sample_weight"] = cls.build_progressive_sample_weights(X_train)
        model.fit(X_train, y_train, **fit_kwargs)
        return ModelFitResult(
            model=model,
            train_predictions=np.asarray(model.predict(X_train), dtype=float),
            validation_predictions=np.asarray(model.predict(X_val), dtype=float),
        )

    @staticmethod
    def save_model(
        model: t.Any,
        scaler: StandardScaler | None = None,
        family_name_override: str | None = None,
    ) -> None:
        _ = scaler
        VOLATILITY_TRAINED_MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

        family_name = family_name_override or LinearRegressionFamily.get_family_name()
        model_path = VOLATILITY_TRAINED_MODELS_DIR_PATH / f"{family_name}.joblib"
        joblib.dump(model, model_path)

        Visualizer.info_confirmation(
            model_path,
            label="Model saved to",
        )


class RandomForestFamily(VolatilityModelFamilyABC):
    MODEL_EXTENSION = ".joblib"

    @staticmethod
    def get_family_name() -> str:
        return "random_forest"

    @staticmethod
    def get_fixed_params() -> t.Dict:
        return {
            "criterion": "squared_error",
            "random_state": VolatilityModelFamilyABC.RANDOM_SEED,
            "n_jobs": -1,
            "min_weight_fraction_leaf": 0.0,
            "verbose": 0,
            "bootstrap": True,
        }

    @staticmethod
    def get_hyperparameter_search_space() -> t.Dict:
        return {
            "n_estimators": [300, 400, 500],
            "max_depth": [None, 8, 12, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
            "bootstrap": [True],
            "max_samples": [None, 0.6, 0.8, 0.9],
            "min_impurity_decrease": [0.0, 1e-6, 1e-5, 1e-4],
            "ccp_alpha": [0.0, 1e-6, 1e-5, 1e-4],
        }

    @staticmethod
    def instantiate_model(
        *, input_dim: int, model_params: dict[str, t.Any]
    ) -> RandomForestRegressor:
        _ = input_dim
        return RandomForestRegressor(**model_params)

    @staticmethod
    def get_n_iter():
        return 120

    @classmethod
    def _fit_model_family(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        progressive_training: bool,
        phase: TrainingPhase = TrainingPhase.CV,
        shuffle: bool = True,
    ) -> ModelFitResult:
        _, _, _ = model_params, phase, shuffle
        fit_kwargs = {}
        if progressive_training:
            fit_kwargs["sample_weight"] = cls.build_progressive_sample_weights(X_train)
        model.fit(X_train, y_train, **fit_kwargs)
        return ModelFitResult(
            model=model,
            train_predictions=np.asarray(model.predict(X_train), dtype=float),
            validation_predictions=np.asarray(model.predict(X_val), dtype=float),
        )

    @staticmethod
    def save_model(
        model: t.Any,
        scaler: StandardScaler | None = None,
        family_name_override: str | None = None,
    ) -> None:
        _ = scaler
        VOLATILITY_TRAINED_MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

        family_name = family_name_override or RandomForestFamily.get_family_name()
        model_path = VOLATILITY_TRAINED_MODELS_DIR_PATH / f"{family_name}.joblib"
        joblib.dump(model, model_path)

        Visualizer.info_confirmation(
            model_path,
            label="Model saved to",
        )


class XGBoostFamily(VolatilityModelFamilyABC):
    MODEL_EXTENSION = ".joblib"

    @staticmethod
    def get_family_name() -> str:
        return "xgboost"

    @staticmethod
    def get_fixed_params() -> t.Dict:
        return {
            "booster": "gbtree",
            "objective": "reg:pseudohubererror",
            "eval_metric": "rmse",
            "random_state": VolatilityModelFamilyABC.RANDOM_SEED,
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
            "base_score": 0.20,  # Adjusted to the target mean to improve convergence
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
    ) -> XGBRegressor:
        _ = input_dim
        return XGBRegressor(**model_params)

    @staticmethod
    def get_n_iter():
        return 200

    @classmethod
    def _fit_model_family(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        progressive_training: bool,
        phase: TrainingPhase = TrainingPhase.CV,
        shuffle: bool = True,
    ) -> ModelFitResult:
        _, _, _ = model_params, phase, shuffle

        if progressive_training:
            phase_datasets, X_train_full_raw, _, X_es, y_es = (
                cls.build_progressive_phase_datasets(X_train, y_train)
            )

            total_estimators = int(model_params["n_estimators"])
            base_estimators = total_estimators // len(phase_datasets)
            remainder = total_estimators % len(phase_datasets)
            phase_estimators = [
                base_estimators + (1 if idx < remainder else 0)
                for idx in range(len(phase_datasets))
            ]

            current_model = model
            previous_booster = None
            for idx, ((X_phase, y_phase), n_estimators) in enumerate(
                zip(phase_datasets, phase_estimators)
            ):
                phase_params = dict(model_params)
                phase_params["n_estimators"] = n_estimators
                current_model = cls.instantiate_model(
                    input_dim=X_train.shape[1],
                    model_params=phase_params,
                )
                fit_kwargs = {
                    "X": X_phase,
                    "y": y_phase,
                    "eval_set": [(X_es, y_es)],
                    "verbose": False,
                }
                if previous_booster is not None:
                    fit_kwargs["xgb_model"] = previous_booster
                current_model.fit(**fit_kwargs)
                previous_booster = current_model.get_booster()

            model = current_model
            X_train_for_predictions = X_train_full_raw
        else:
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
            X_train_for_predictions = X_train

        best_iteration = getattr(model, "best_iteration", None)
        best_score = getattr(model, "best_score", None)

        return ModelFitResult(
            model=model,
            train_predictions=np.asarray(
                model.predict(X_train_for_predictions),
                dtype=float,
            ),
            validation_predictions=np.asarray(model.predict(X_val), dtype=float),
            best_iteration=(
                int(best_iteration)
                if best_iteration is not None and best_iteration >= 0
                else None
            ),
            best_score=float(best_score) if best_score is not None else None,
        )

    @staticmethod
    def save_model(
        model: t.Any,
        scaler: StandardScaler | None = None,
        family_name_override: str | None = None,
    ) -> None:
        _ = scaler
        VOLATILITY_TRAINED_MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

        family_name = family_name_override or XGBoostFamily.get_family_name()
        model_path = VOLATILITY_TRAINED_MODELS_DIR_PATH / f"{family_name}.joblib"
        joblib.dump(model, model_path)

        Visualizer.info_confirmation(
            model_path,
            label="Model saved to",
        )


class SequentialNNFamily(VolatilityModelFamilyABC):
    MODEL_EXTENSION = ".keras"

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
        phase: TrainingPhase,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        scaler = StandardScaler()
        indices = list(numeric_col_indices)

        X_fit_scaled = X_fit_raw.copy()
        X_valid_scaled = X_valid_raw.copy()
        X_train_full_scaled = X_train_full_raw.copy()
        X_eval_scaled = X_eval_raw.copy()

        scale_reference = (
            X_train_full_raw if phase is TrainingPhase.FINAL_TEST else X_fit_raw
        )
        scaler.fit(scale_reference[:, indices])

        X_fit_scaled[:, indices] = scaler.transform(X_fit_raw[:, indices])
        X_valid_scaled[:, indices] = scaler.transform(X_valid_raw[:, indices])
        X_train_full_scaled[:, indices] = scaler.transform(X_train_full_raw[:, indices])
        X_eval_scaled[:, indices] = scaler.transform(X_eval_raw[:, indices])

        return X_fit_scaled, X_valid_scaled, X_train_full_scaled, X_eval_scaled, scaler

    @staticmethod
    def _resolve_numeric_col_indices(*, data: pd.DataFrame) -> tuple[int, ...]:
        numeric_cols = list(BASE_NUMERIC_FEATURE_COLS)
        return tuple(int(data.columns.get_loc(col)) for col in numeric_cols)

    @staticmethod
    def instantiate_model(
        *, input_dim: int, model_params: dict[str, t.Any]
    ) -> keras.Sequential:
        keras.backend.clear_session()
        tf.random.set_seed(VolatilityModelFamilyABC.RANDOM_SEED)

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
        return 120

    @classmethod
    def _fit_model_family(
        cls,
        *,
        model: t.Any,
        model_params: t.Dict[str, t.Any],
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        progressive_training: bool,
        phase: TrainingPhase = TrainingPhase.CV,
        shuffle: bool = True,
    ) -> ModelFitResult:
        numeric_col_indices = cls._resolve_numeric_col_indices(data=X_train)
        X_train_full_raw = np.asarray(X_train)
        y_train_full = np.asarray(y_train)
        X_val = np.asarray(X_val)

        if progressive_training:
            phase_datasets, _, _, X_es_raw, y_es = cls.build_progressive_phase_datasets(
                X_train,
                y_train,
            )
            X_fit_raw = phase_datasets[-1][0]
        else:
            phase_datasets = [(X_train_full_raw, y_train_full)]
            X_fit_raw, y_fit, X_es_raw, y_es = cls.temporal_inner_split_for_early_stopping(
                phase_datasets[-1][0],
                phase_datasets[-1][1],
            )
            phase_datasets[-1] = (X_fit_raw, y_fit)

        _, X_es_scaled, X_train_scaled, X_val_scaled, feature_scaler = (
            cls._scale_numeric_features(
                X_fit_raw=X_fit_raw,
                X_valid_raw=X_es_raw,
                X_train_full_raw=X_train_full_raw,
                X_eval_raw=X_val,
                numeric_col_indices=numeric_col_indices,
                phase=phase,
            )
        )

        early_stop = EarlyStopping(
            monitor="val_rmse",
            mode="min",
            patience=model_params["patience"],
            restore_best_weights=True,
            verbose=0,
        )
        callbacks = [early_stop]

        if model_params.get("use_lr_scheduler", False):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor="val_rmse",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0,
                )
            )

        combined_history = {}
        for x_raw, y_phase in phase_datasets:
            x = cls.transform_numeric_features(
                X_raw=x_raw,
                numeric_col_indices=numeric_col_indices,
                scaler=feature_scaler,
            )
            history = model.fit(
                x,
                y_phase,
                epochs=model_params["epochs"],
                batch_size=model_params["batch_size"],
                validation_data=(X_es_scaled, y_es),
                callbacks=callbacks,
                verbose=model_params["verbose"],
                shuffle=shuffle,
            )

            for key, values in history.history.items():
                combined_history.setdefault(key, []).extend(values)

        train_rmse_history = [float(value) for value in combined_history.get("rmse", [])]
        val_rmse_history = [
            float(value) for value in combined_history.get("val_rmse", [])
        ]

        metric_history = val_rmse_history if val_rmse_history else train_rmse_history
        best_epoch = int(np.argmin(metric_history)) + 1 if metric_history else None
        best_score = float(np.min(metric_history)) if metric_history else None

        return ModelFitResult(
            model=model,
            train_predictions=model.predict(X_train_scaled, verbose=0).flatten(),
            validation_predictions=model.predict(X_val_scaled, verbose=0).flatten(),
            best_iteration=best_epoch,
            best_score=best_score,
            epoch_history={
                "rmse": train_rmse_history,
                "val_rmse": val_rmse_history,
            },
            feature_scaler=feature_scaler,
        )

    @classmethod
    def get_scaler_path(cls):
        return (
            VOLATILITY_TRAINED_MODELS_DIR_PATH
            / f"{cls.get_family_name()}_scaler.joblib"
        )

    @staticmethod
    def save_model(
        model: t.Any,
        scaler: StandardScaler | None = None,
        family_name_override: str | None = None,
    ) -> None:
        VOLATILITY_TRAINED_MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

        family_name = family_name_override or SequentialNNFamily.get_family_name()
        model_path = VOLATILITY_TRAINED_MODELS_DIR_PATH / f"{family_name}.keras"

        model.save(model_path)

        if scaler is not None:
            scaler_path = (
                VOLATILITY_TRAINED_MODELS_DIR_PATH / f"{family_name}_scaler.joblib"
            )
            joblib.dump(scaler, scaler_path)
            Visualizer.info_confirmation(
                scaler_path,
                label="Scaler saved to",
            )

        Visualizer.info_confirmation(
            model_path,
            label="Model saved to",
        )

    @classmethod
    def plots(cls, kwargs):
        args = {
            "training_registry": kwargs.get("training_registry"),
            "best_model_name": kwargs.get("best_model_name"),
            "family_name": kwargs.get("family_name", cls.get_family_name()),
            "training_information": kwargs.get("training_information"),
            "phase": kwargs.get("phase"),
        }
        Visualizer.plot_nn_learning_curves(**args)


class QuantumInspiredNNFamily(SequentialNNFamily):
    MODEL_EXTENSION = ".keras"

    @staticmethod
    def get_family_name() -> str:
        return "quantum_inspired_nn"

    @staticmethod
    def get_fixed_params() -> t.Dict:
        return {
            "patience": 12,
            "epochs": 150,
            "verbose": 0,
            "embedding_activation": "swish",
            "post_tt_activation": "swish",
            "use_batch_norm": True,
            "kernel_initializer": "glorot_uniform",
        }

    @staticmethod
    def _get_tt_configurations() -> list[dict[str, t.Any]]:
        return [
            {
                "tensor_dims": [2, 2, 2, 2],
                "tt_ranks": [1, 2, 2, 2, 1],
                "tt_output_dim": 8,
            },
            {
                "tensor_dims": [2, 2, 2, 2],
                "tt_ranks": [1, 4, 4, 4, 1],
                "tt_output_dim": 12,
            },
            {
                "tensor_dims": [2, 2, 2, 3],
                "tt_ranks": [1, 3, 3, 3, 1],
                "tt_output_dim": 12,
            },
            {
                "tensor_dims": [2, 2, 2, 4],
                "tt_ranks": [1, 4, 4, 4, 1],
                "tt_output_dim": 16,
            },
            {
                "tensor_dims": [2, 2, 3, 3],
                "tt_ranks": [1, 3, 4, 3, 1],
                "tt_output_dim": 16,
            },
        ]

    @staticmethod
    def get_hyperparameter_search_space() -> t.Dict:
        return {
            "tt_configuration": QuantumInspiredNNFamily._get_tt_configurations(),
            "dense_units": [16, 24, 32, 48, 64],
            "dropout_rate": [0.0, 0.05, 0.1, 0.15],
            "l2_reg": [0.0, 1e-6, 1e-5, 1e-4],
            "learning_rate": [3e-4, 5e-4, 8e-4, 1e-3],
            "batch_size": [128, 256, 512, 1024],
            "loss": ["huber", "mse"],
            "use_lr_scheduler": [True, False],
            "kernel_initializer": ["glorot_uniform", "he_uniform"],
        }

    @staticmethod
    def instantiate_model(
        *, input_dim: int, model_params: dict[str, t.Any]
    ) -> keras.Model:
        keras.backend.clear_session()
        keras.utils.set_random_seed(VolatilityModelFamilyABC.RANDOM_SEED)

        tt_configuration = dict(model_params["tt_configuration"])
        tensor_dims = tuple(int(dim) for dim in tt_configuration["tensor_dims"])
        tt_ranks = tuple(int(rank) for rank in tt_configuration["tt_ranks"])
        tt_output_dim = int(tt_configuration["tt_output_dim"])
        embedding_dim = int(np.prod(tensor_dims))
        kernel_regularizer = regularizers.l2(model_params["l2_reg"])

        inputs = keras.Input(shape=(input_dim,), name="qi_features")
        x = inputs

        if model_params.get("use_batch_norm", True):
            x = layers.BatchNormalization(name="qi_batch_norm")(x)

        x = layers.Dense(
            embedding_dim,
            activation=model_params["embedding_activation"],
            kernel_initializer=model_params["kernel_initializer"],
            kernel_regularizer=kernel_regularizer,
            name="qi_embedding",
        )(x)
        x = layers.Reshape(tensor_dims, name="qi_tensorize")(x)
        x = TensorTrainLayer(
            input_dims=tensor_dims,
            tt_ranks=tt_ranks,
            output_dim=tt_output_dim,
            activation=model_params["post_tt_activation"],
            name="qi_tensor_train",
        )(x)
        x = layers.Dense(
            model_params["dense_units"],
            activation=model_params["post_tt_activation"],
            kernel_initializer=model_params["kernel_initializer"],
            kernel_regularizer=kernel_regularizer,
            name="qi_post_tt_dense",
        )(x)

        if model_params["dropout_rate"] > 0.0:
            x = layers.Dropout(model_params["dropout_rate"], name="qi_post_tt_dropout")(
                x
            )

        outputs = layers.Dense(1, activation="linear", name="qi_output")(x)

        model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=f"quantum_inspired_tt_{'_'.join(map(str, tensor_dims))}",
        )
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
        return 50

    @staticmethod
    def save_model(
        model: t.Any,
        scaler: StandardScaler | None = None,
        family_name_override: str | None = None,
    ) -> None:
        VOLATILITY_TRAINED_MODELS_DIR_PATH.mkdir(parents=True, exist_ok=True)

        family_name = family_name_override or QuantumInspiredNNFamily.get_family_name()
        model_path = VOLATILITY_TRAINED_MODELS_DIR_PATH / f"{family_name}.keras"

        model.save(model_path)

        if scaler is not None:
            scaler_path = (
                VOLATILITY_TRAINED_MODELS_DIR_PATH / f"{family_name}_scaler.joblib"
            )
            joblib.dump(scaler, scaler_path)
            Visualizer.info_confirmation(
                scaler_path,
                label="Scaler saved to",
            )

        Visualizer.info_confirmation(
            model_path,
            label="Model saved to",
        )
