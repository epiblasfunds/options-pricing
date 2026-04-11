import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from src.config.config import VOLATILITY_FAMILY_METADATA_DIR_PATH, config
from src.python_models.volatility_models.volatility_model_family import (
    TrainnigPhase,
    VolatilityModelFamilyABC,
)
from src.volatility_models.data_utils import (
    BASE_FEATURE_COLS,
    TrainingDataHandler,
    TrainingDataSplitEnum,
)
from src.volatility_models.visualization_utils import Visualizer

TARGET_COLUMN = config.volatility_models_config.training_data_config.target_column
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model_family: VolatilityModelFamilyABC):
        self.model_family = model_family

    @staticmethod
    def add_custom_error1(metrics_table: pd.DataFrame, round_digits: int = 6):
        """
        Añade una métrica compuesta de selección que penaliza inestabilidad y sobreajuste.
        """
        custom_error_1_config = config.volatility_models_config.training_data_config.custom_error_1

        base_metric = custom_error_1_config["base_metric"]
        alpha = custom_error_1_config["alpha"]
        beta = custom_error_1_config["beta"]

        train_metric_col = f"train_{base_metric}"
        valid_metric_col = f"val_{base_metric}"
        valid_metric_std_col = f"{valid_metric_col}_std"

        train_metric = pd.to_numeric(metrics_table[train_metric_col])
        valid_metric = pd.to_numeric(metrics_table[valid_metric_col])
        valid_metric_std = pd.to_numeric(metrics_table[valid_metric_std_col])

        overfit_gap = np.maximum(0.0, valid_metric - train_metric)

        metrics_table = metrics_table.copy()
        metrics_table[custom_error_1_config["metric"]] = np.round(
            valid_metric + alpha * valid_metric_std + beta * overfit_gap,
            round_digits,
        )

        return metrics_table

    @classmethod
    def add_model_metrics_row(
        cls,
        metrics_dict,
        model_name="model",
        metrics_table=None,
        round_digits: int = 6,
    ):
        """
        Agrega una fila con métricas de un modelo a la tabla comparativa.
        """
        ordered_keys = []
        for k in config.volatility_models_config.training_data_config.metric_order:
            if k in metrics_dict:
                ordered_keys.append(k)
                if f"{k}_std" in metrics_dict:
                    ordered_keys.append(f"{k}_std")

        model_row = pd.DataFrame(
            [{k: float(np.round(metrics_dict[k], round_digits)) for k in ordered_keys}],
            index=[model_name],
        )
        model_row.index.name = "model"

        if metrics_table is None:
            metrics_table = model_row
        else:
            metrics_table = pd.concat([metrics_table, model_row], axis=0)

        metrics_table = cls.add_custom_error1(metrics_table, round_digits=round_digits)

        # Orden de columnas derivado directamente de METRIC_ORDER, con _std intercalado tras cada métrica
        ordered_cols = []
        for k in config.volatility_models_config.training_data_config.metric_order:
            if k in metrics_table.columns:
                ordered_cols.append(k)
            if f"{k}_std" in metrics_table.columns:
                ordered_cols.append(f"{k}_std")
        ordered_cols += [c for c in metrics_table.columns if c not in ordered_cols]
        metrics_table = metrics_table[ordered_cols]

        selected_metric = (
            config.volatility_models_config.training_data_config.custom_error_1[
                "metric"
            ]
        )
        selected_mode = (
            config.volatility_models_config.training_data_config.custom_error_1["mode"]
        )

        if selected_metric in metrics_table.columns:
            metrics_table = metrics_table.sort_values(
                by=selected_metric, ascending=(selected_mode == "min")
            )

        return metrics_table

    @staticmethod
    def calculate_metrics_across_folds(fold_metrics_list, round_digits: int = 6):
        """
        Calcula estadísticas agregadas (media y desviación estándar) de métricas a través de los folds.
        """
        metric_order = config.volatility_models_config.training_data_config.metric_order
        metric_keys = [k for k in metric_order if k in fold_metrics_list[0]]
        agg_metrics = {}

        for key in metric_keys:
            values = np.asarray([fm[key] for fm in fold_metrics_list], dtype=float)
            agg_metrics[key] = float(np.round(values.mean(), round_digits))
            agg_metrics[f"{key}_std"] = float(np.round(values.std(), round_digits))

        return agg_metrics

    @staticmethod
    def calculate_regression_metrics(
        y_train_true,
        y_train_pred,
        y_valid_true,
        y_valid_pred,
        evaluation_label: TrainingDataSplitEnum,
        round_digits: int = 6,
    ) -> dict[str, float]:
        y_train_true = np.asarray(y_train_true)
        y_train_pred = np.asarray(y_train_pred)
        y_valid_true = np.asarray(y_valid_true)
        y_valid_pred = np.asarray(y_valid_pred)

        metrics = {
            "train_mae": mean_absolute_error(y_train_true, y_train_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
            "train_r2": r2_score(y_train_true, y_train_pred),
            f"{evaluation_label.value}_mae": mean_absolute_error(y_valid_true, y_valid_pred),
            f"{evaluation_label.value}_rmse": np.sqrt(mean_squared_error(y_valid_true, y_valid_pred)),
            f"{evaluation_label.value}_r2": r2_score(y_valid_true, y_valid_pred),
        }
        return {name: float(np.round(value, round_digits)) for name, value in metrics.items()}

    def run_kfolds_training_for_specific_model(
        self,
        model_name,
        model_params,
        phase: TrainnigPhase,
        evaluation_label: TrainingDataSplitEnum = TrainingDataSplitEnum.VAL,
    ):
        training_registry = {}
        folds = TrainingDataHandler.load_kfolds(False)
        model_folds_metrics = []
        for k, v in folds.items():
            fold_name = k
            train_fold = v["train"]
            val_fold = v["val"]

            X_train_fold = train_fold[BASE_FEATURE_COLS].to_numpy()
            y_train_fold = train_fold[TARGET_COLUMN].to_numpy()
            X_val_fold = val_fold[BASE_FEATURE_COLS].to_numpy()
            y_val_fold = val_fold[TARGET_COLUMN].to_numpy()

            model = self.model_family.instantiate_model(
                input_dim=X_train_fold.shape[1],
                model_params=model_params,
            )

            fit_result = self.model_family.fit_model(
                model=model,
                model_params=model_params,
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                phase=phase,
            )

            metrics = self.calculate_regression_metrics(
                y_train_fold,
                fit_result.train_predictions,
                y_val_fold,
                fit_result.validation_predictions,
                evaluation_label=evaluation_label,
            )
            model_folds_metrics.append(metrics)

            training_registry[f"{model_name}_{fold_name}"] = {
                "model": model,
                "best_iteration": fit_result.best_iteration,
                "best_score": fit_result.best_score,
                "metrics": metrics,
                "y_val_true": y_val_fold.tolist(),
                "y_val_pred": fit_result.validation_predictions.tolist(),
            }

        metrics = self.calculate_metrics_across_folds(model_folds_metrics)

        return training_registry, metrics

    @staticmethod
    def select_best_model_from_metrics_table(
        metrics_table: pd.DataFrame,
        training_registry=None,
        family_name=None,
        selection_config=None,
    ):
        """
        Selecciona el mejor modelo desde `metrics_table` según criterio configurable.
        """
        selected_metric = selection_config["metric"]
        selected_mode = selection_config["mode"]

        metric_series = pd.to_numeric(metrics_table[selected_metric])

        best_idx = (
            metric_series.idxmin() if selected_mode == "min" else metric_series.idxmax()
        )
        best_model_row = metrics_table.loc[[best_idx]].copy()

        Visualizer.best_model_family_graphics(
            training_registry, best_model_row, family_name
        )

        return best_model_row, best_idx

    @classmethod
    def _to_builtin(cls, obj):
        """Convierte tipos numpy/pandas a tipos nativos para serializar en JSON."""
        if isinstance(obj, dict):
            return {str(k): cls._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._to_builtin(v) for v in obj]
        if isinstance(obj, tuple):
            return [cls._to_builtin(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    @classmethod
    def _training_registry_for_json(cls, training_registry: dict) -> dict:
        """Elimina objetos no serializables (p.ej. modelos) y conserva métricas/históricos."""
        serializable = {}
        for key, record in training_registry.items():
            serializable[key] = {
                "best_iteration": cls._to_builtin(record.get("best_iteration", None)),
                "best_score": cls._to_builtin(record.get("best_score", None)),
                "metrics": cls._to_builtin(record.get("metrics", {})),
                "epoch_history": cls._to_builtin(record.get("epoch_history", {})),
                "y_val_true": cls._to_builtin(record.get("y_val_true", [])),
                "y_val_pred": cls._to_builtin(record.get("y_val_pred", [])),
            }
        return serializable

    @classmethod
    def save_family_metadata(
        cls,
        family_name: str,
        metrics_table: pd.DataFrame,
        model_params_registry: dict,
        training_registry: dict,
        best_model_row: pd.DataFrame,
    ):
        """Guarda metadatos de una familia para reanudar trabajo en otra sesión."""
        if metrics_table is None or len(metrics_table) == 0:
            raise ValueError(
                f"{family_name}: metrics_table vacío, no se puede persistir metadata."
            )

        best_model_name = str(best_model_row.index[0])
        best_model_metrics = {
            k: cls._to_builtin(v) for k, v in best_model_row.iloc[0].to_dict().items()
        }

        payload = {
            "family_name": family_name,
            "best_model_name": best_model_name,
            "best_model_metrics": best_model_metrics,
            "best_model_params": cls._to_builtin(
                model_params_registry[best_model_name]
            ),
            "metrics_table": cls._to_builtin(metrics_table.to_dict(orient="split")),
            "model_params_registry": cls._to_builtin(model_params_registry),
            "training_registry": cls._training_registry_for_json(training_registry),
        }

        file_path = VOLATILITY_FAMILY_METADATA_DIR_PATH / f"{family_name}_metadata.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Metadata guardada: {file_path}")

    def run_kfolds_training(self, phase: TrainnigPhase):
        models = self.model_family.build_model_candidates()
        total_family_models = len(models)
        family_models_metrics_table = None
        training_registry = {}
        model_params_registry = {}

        for model_i_name, model_i_params in tqdm(
            models.items(),
            total=total_family_models,
            desc=f"Entrenando familia: {self.model_family.get_family_name()}",
        ):
            model_params_registry[model_i_name] = model_i_params
            tr, metrics = self.run_kfolds_training_for_specific_model(
                model_name=model_i_name,
                model_params=model_i_params,
                phase=phase,
            )
            family_models_metrics_table = self.add_model_metrics_row(
                metrics,
                model_name=model_i_name,
                metrics_table=family_models_metrics_table,
            )
            training_registry.update(tr)

        best_model, best_idx = self.select_best_model_from_metrics_table(
            family_models_metrics_table,
            training_registry,
            family_name=self.model_family.get_family_name(),
            selection_config=config.volatility_models_config.training_data_config.custom_error_1,
        )

        self.model_family.plots(
            {
                "training_registry": training_registry,
                "best_model_row": best_model,
            }
        )

        self.save_family_metadata(
            family_name=self.model_family.get_family_name(),
            metrics_table=family_models_metrics_table,
            model_params_registry=model_params_registry,
            training_registry=training_registry,
            best_model_row=best_model,
        )
        return family_models_metrics_table, best_idx
