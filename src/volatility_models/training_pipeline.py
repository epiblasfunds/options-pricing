from dataclasses import dataclass
from pathlib import Path

import keras
import pandas as pd
from keras import callbacks, layers, models, optimizers, regularizers
from sklearn.preprocessing import StandardScaler

from src.config.config import (
    VOLATILITY_MODEL_DATA_DIR_PATH,
    VOLATILITY_TRAINED_MODELS_DIR_PATH,
    config,
)
from src.dashboard.domain import build_metrics_registry
from src.data_management.loaders.volatility_step_loader import VolatilityStepLoader
from src.enums.volatility_model_enums import ModelFormatEnum
from src.volatility_models import (
    MODEL_FEATURE_NAMES,
    TARGET_COLUMN,
    build_feature_frame_from_trades,
    select_trade_columns,
)
from src.volatility_models.trained_model import (
    TrainedModel,
    TrainedModelMetadata,
    TrainingHistory,
)

METRICS_REGISTRY = build_metrics_registry()


@dataclass(frozen=True)
class DatasetSplits:
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame


def load_volatility_trade_frame(force_reload: bool = False) -> pd.DataFrame:
    frame = VolatilityStepLoader.load(force_reload=force_reload)
    selected = select_trade_columns(frame)
    return selected.sort_values("ExecDatetime").reset_index(drop=True)


def split_trade_frame(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> DatasetSplits:
    ordered = frame.sort_values("ExecDatetime").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_ratio)
    validation_end = int(n_rows * (train_ratio + validation_ratio))
    return DatasetSplits(
        train_frame=ordered.iloc[:train_end].copy(),
        validation_frame=ordered.iloc[train_end:validation_end].copy(),
        test_frame=ordered.iloc[validation_end:].copy(),
    )


def save_dataset_splits(
    splits: DatasetSplits,
    output_dir: Path = VOLATILITY_MODEL_DATA_DIR_PATH,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits.train_frame.to_csv(output_dir / "train.csv", sep=";", index=False)
    splits.validation_frame.to_csv(output_dir / "validation.csv", sep=";", index=False)
    splits.test_frame.to_csv(output_dir / "test.csv", sep=";", index=False)


def build_training_matrices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = build_feature_frame_from_trades(frame)
    y = frame[str(TARGET_COLUMN)].astype("float64")
    return X.loc[:, MODEL_FEATURE_NAMES], y


def build_baseline_mlp(
    input_dim: int,
    hidden_units: tuple[int, ...] = (256, 128, 64),
    dropout_rate: float = 0.15,
    learning_rate: float = 5e-4,
    l2_strength: float = 1e-5,
) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for units in hidden_units:
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_strength),
        )(x)
        x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, name="predicted_iv")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="baseline_mlp")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_residual_mlp(
    input_dim: int,
    width: int = 192,
    depth: int = 3,
    dropout_rate: float = 0.1,
    learning_rate: float = 5e-4,
    l2_strength: float = 5e-5,
) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(
        width,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_strength),
    )(inputs)
    for _ in range(max(depth - 1, 0)):
        residual = x
        block = layers.BatchNormalization()(x)
        block = layers.Dense(
            width,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_strength),
        )(block)
        block = layers.Dropout(dropout_rate)(block)
        x = layers.Add()([residual, block])
    outputs = layers.Dense(1, name="predicted_iv")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="residual_mlp")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_trained_model(
    *,
    model_id: str,
    model_name: str,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    model_builder,
    epochs: int = 40,
    batch_size: int = 1024,
    output_dir: Path = VOLATILITY_TRAINED_MODELS_DIR_PATH,
) -> tuple[TrainedModel, dict[str, float]]:
    X_train, y_train = build_training_matrices(train_frame)
    X_validation, y_validation = build_training_matrices(validation_frame)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    model = model_builder(X_train.shape[1])
    history = model.fit(
        X_train_scaled.astype("float32"),
        y_train.to_numpy(dtype="float32"),
        validation_data=(
            X_validation_scaled.astype("float32"),
            y_validation.to_numpy(dtype="float32"),
        ),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True,
            )
        ],
    )
    validation_predictions = model.predict(
        X_validation_scaled.astype("float32"),
        verbose=0,
    ).reshape(-1)
    validation_metrics = METRICS_REGISTRY.compute_metrics(
        y_validation.reset_index(drop=True),
        pd.Series(validation_predictions),
        config.dashboard_models_config.error_metrics,
    )
    model_path = output_dir / model_id
    trained_model = TrainedModel(
        model=model,
        metadata=TrainedModelMetadata(
            model_id=model_id,
            name=model_name,
            path=model_path,
            format=ModelFormatEnum.KERAS,
            feature_names=tuple(MODEL_FEATURE_NAMES),
            target_column=str(TARGET_COLUMN),
            loss_name="loss",
            metadata={
                "validation_metrics": {
                    name: float(value) for name, value in validation_metrics.items()
                },
                "model_input_features": list(MODEL_FEATURE_NAMES),
                "target_column": str(TARGET_COLUMN),
            },
        ),
        history=TrainingHistory.from_keras_history(history, loss_name="loss"),
        preprocessor=scaler,
    )
    trained_model.save(model_path)
    return trained_model, {
        name: float(value) for name, value in validation_metrics.items()
    }
