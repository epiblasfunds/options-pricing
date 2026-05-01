import json
import logging
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from src.config.config import (
    VOLATILITY_FAMILY_METADATA_DIR_PATH,
    VOLATILITY_RETRAINED_METADATA_DIR_PATH,
    config,
)
from src.enums.volatility_model_enums.training_data_split import TrainingDataSplitEnum
from src.enums.volatility_model_enums.training_phase import TrainingPhase
from src.python_models.volatility_models.volatility_model_family import (
    ModelFitResult,
    VolatilityModelFamilyABC,
)
from src.volatility_models.data_utils import BASE_FEATURE_COLS, TrainingDataHandler
from src.volatility_models.visualization_utils import Visualizer

TARGET_COLUMN = config.volatility_models_config.training_data_config.target_column
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model_family: VolatilityModelFamilyABC):
        self.model_family = model_family

    @staticmethod
    def _metric_column(split_label: TrainingDataSplitEnum, metric_name: str) -> str:
        return f"{split_label}_{metric_name}"

    @staticmethod
    def _compute_base_metric(metric_name: str, y_true, y_pred) -> float:
        if metric_name == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        if metric_name == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        if metric_name == "r2":
            return float(r2_score(y_true, y_pred))

    @classmethod
    def _metrics_for_splits(
        cls,
        split_labels: list[TrainingDataSplitEnum],
    ) -> list[str]:
        base_metrics = config.volatility_models_config.training_data_config.models_metrics
        return [
            cls._metric_column(split_label, metric_name)
            for split_label in split_labels
            for metric_name in base_metrics
        ]

    @staticmethod
    def add_custom_error1(
        metrics_table: pd.DataFrame,
        round_digits: int = 6
    ):
        """
        Añade una métrica compuesta de selección que penaliza inestabilidad y sobreajuste.
        """
        custom_error_1_config = config.volatility_models_config.training_data_config.custom_error_1

        base_metric = custom_error_1_config["base_metric"]
        alpha = custom_error_1_config["alpha"]
        beta = custom_error_1_config["beta"]

        train_metric_col = Trainer._metric_column(TrainingDataSplitEnum.TRAIN, base_metric)
        valid_metric_col = Trainer._metric_column(TrainingDataSplitEnum.VAL, base_metric)
        valid_metric_std_col = f"{valid_metric_col}_std"

        train_metric = pd.to_numeric(metrics_table[train_metric_col])
        valid_metric = pd.to_numeric(metrics_table[valid_metric_col])
        valid_metric_std = pd.to_numeric(metrics_table[valid_metric_std_col])

        overfit_gap = np.maximum(0.0, valid_metric - train_metric)

        metrics_table = metrics_table.copy()
        metric_col = custom_error_1_config["metric"]
        metrics_table[metric_col] = np.round(
            valid_metric + alpha * valid_metric_std + beta * overfit_gap,
            round_digits,
        )
        cols = [metric_col] + [c for c in metrics_table.columns if c != metric_col]

        return metrics_table[cols]

    @staticmethod
    def add_custom_error2(
        metrics_dict: dict[str, float],
        custom_error_1: float,
        round_digits: int = 6,
    ) -> dict[str, float]:
        """
        Añade una métrica compuesta de selección que penaliza inestabilidad y sobreajuste.
        """
        custom_error_2_config = config.volatility_models_config.training_data_config.custom_error_2

        base_metric = custom_error_2_config["base_metric"]
        gamma = custom_error_2_config["gamma"]
        beta = custom_error_2_config["beta"]

        train_metric_col = Trainer._metric_column(TrainingDataSplitEnum.TRAIN, base_metric)
        valid_metric_col = Trainer._metric_column(TrainingDataSplitEnum.VAL, base_metric)

        train_metric = float(metrics_dict[train_metric_col])
        valid_metric = float(metrics_dict[valid_metric_col])

        overfit_gap = np.maximum(0.0, valid_metric - train_metric)

        metric_col = custom_error_2_config["metric"]
        result = {
            metric_col: float(
                np.round(
                    valid_metric + gamma * custom_error_1 + beta * overfit_gap,
                    round_digits,
                )
            )
        }
        result.update({k: float(v) for k, v in metrics_dict.items() if k != metric_col})
        return result


    @staticmethod
    def calculate_metrics_across_folds(
        fold_metrics_list,
        round_digits: int = 6,
    ):
        """
        Calcula estadísticas agregadas (media y desviación estándar) de métricas a través de los folds.
        """
        metric_keys = [
            k for k in Trainer._metrics_for_splits([TrainingDataSplitEnum.TRAIN, TrainingDataSplitEnum.VAL])
            if k in fold_metrics_list[0]
        ]
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
        train_label: TrainingDataSplitEnum = TrainingDataSplitEnum.TRAIN,
        round_digits: int = 6,
    ) -> dict[str, float]:
        y_train_true = np.asarray(y_train_true)
        y_train_pred = np.asarray(y_train_pred)
        y_valid_true = np.asarray(y_valid_true)
        y_valid_pred = np.asarray(y_valid_pred)

        base_metrics = list(config.volatility_models_config.training_data_config.models_metrics)
        metrics = {}
        for metric_name in base_metrics:
            train_metric_col = Trainer._metric_column(train_label, metric_name)
            eval_metric_col = Trainer._metric_column(evaluation_label, metric_name)
            metrics[train_metric_col] = Trainer._compute_base_metric(metric_name, y_train_true, y_train_pred)
            metrics[eval_metric_col] = Trainer._compute_base_metric(metric_name, y_valid_true, y_valid_pred)

        return {name: float(np.round(value, round_digits)) for name, value in metrics.items()}

    def run_kfolds_training_for_specific_model(
        self,
        model_name,
        model_params,
        phase: TrainingPhase,
        evaluation_label: TrainingDataSplitEnum = TrainingDataSplitEnum.VAL,
        folds: dict | None = None,
    ):
        training_registry = {}
        if folds is None:
            folds = TrainingDataHandler.load_kfolds(verbose=False)
        model_folds_metrics = []
        for k, v in folds.items():
            fold_name = k
            train_fold = v["train"]
            val_fold = v["val"]

            X_train_fold = train_fold[BASE_FEATURE_COLS]
            y_train_fold = train_fold[TARGET_COLUMN].to_numpy()
            X_val_fold = val_fold[BASE_FEATURE_COLS]
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
                phase=phase,
                shuffle=True,
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
                "epoch_history": fit_result.epoch_history or {},
                "metrics": metrics,
                "y_val_true": y_val_fold.tolist(),
                "y_val_pred": fit_result.validation_predictions.tolist(),
            }

        metrics = self.calculate_metrics_across_folds(model_folds_metrics)

        return training_registry, metrics

    @staticmethod
    def select_best_model_from_metrics_table(
        metrics_table: pd.DataFrame,
        training_registry: dict,
        family_name: str,
        selection_config=None,
    ):
        """
        Selecciona el mejor modelo desde `metrics_table` según criterio configurable.
        """
        selected_metric = selection_config["metric"]
        selected_mode = selection_config["mode"]

        metric_series = pd.to_numeric(metrics_table[selected_metric])

        best_model_name = (
            metric_series.idxmin() if selected_mode == "min" else metric_series.idxmax()
        )
        best_model_info_df = metrics_table.loc[[best_model_name]].copy()

        Visualizer.best_model_family_graphics(
        training_registry=training_registry,
        best_model_row=best_model_info_df,
        family_name=family_name
        )

        return best_model_name

    @classmethod
    def _to_builtin(cls, obj):
        if isinstance(obj, dict):
            return {str(k): cls._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._to_builtin(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    @classmethod
    def _training_registry_for_json(cls, training_registry: dict) -> dict:
        return {
            k: {
                "best_iteration": cls._to_builtin(v.get("best_iteration")),
                "best_score": cls._to_builtin(v.get("best_score")),
                "metrics": cls._to_builtin(v.get("metrics", {})),
                "epoch_history": cls._to_builtin(v.get("epoch_history", {})),
                "y_val_true": cls._to_builtin(v.get("y_val_true", [])),
                "y_val_pred": cls._to_builtin(v.get("y_val_pred", [])),
            }
            for k, v in training_registry.items()
        }

    @classmethod
    def save_family_metadata(
        cls,
        family_name: str,
        best_model_name: str,
        metrics_table: pd.DataFrame,
        model_params_registry: dict,
        training_registry: dict,
    ):
        """Guarda metadatos estructurados de la familia entrenada en JSON."""

        payload = {
            "family_name": family_name,
            "best_model_name": best_model_name,
            "best_model_metrics": cls._to_builtin(metrics_table.loc[best_model_name].to_dict()),
            "best_model_params": cls._to_builtin(model_params_registry[best_model_name]),
            "metrics_table": cls._to_builtin(metrics_table.to_dict(orient="split")),
            "model_params_registry": cls._to_builtin(model_params_registry),
            "training_registry": cls._training_registry_for_json(training_registry),
        }

        file_path = VOLATILITY_FAMILY_METADATA_DIR_PATH / f"{family_name}_metadata.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        Visualizer.info_confirmation(file_path, label="Metadata guardada")

    @classmethod
    def save_retrained_metadata(
        cls,
        family_name: str,
        phase: TrainingPhase,
        model_params: dict,
        training_information: dict,
    ):
        """Guarda metadatos del modelo reentrenado en JSON."""

        phase_suffix = str(getattr(phase, "value", phase)).lower().replace(" ", "_")

        payload = {
            "family_name": family_name,
            "model_params": cls._to_builtin(model_params),
            "result_metrics": cls._to_builtin(
                training_information.get("result_series", pd.Series(dtype=float)).to_dict()
            ),
            "training_information": {
                "best_iteration": cls._to_builtin(training_information.get("best_iteration")),
                "best_score": cls._to_builtin(training_information.get("best_score")),
                "epoch_history": cls._to_builtin(training_information.get("epoch_history", {})),
                "y_val_true": cls._to_builtin(training_information.get("y_val_true", [])),
                "y_val_pred": cls._to_builtin(training_information.get("y_val_pred", [])),
            },
        }

        file_path = VOLATILITY_RETRAINED_METADATA_DIR_PATH / f"{family_name}_{phase_suffix}_retrained_metadata.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        Visualizer.info_confirmation(file_path, label="Metadata de reentrenamiento guardada")
    
    @staticmethod
    def _build_progressive_training_data(
        segments: list,
        feature_cols: list,
        target_col: str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Construye bloques acumulativos ATM->menos ATM.
        En cada iteracion se concatena hasta el segmento actual, se aplica shuffle
        sobre ese acumulado, y al final se concatenan todos los acumulados.
        """
        progressive_blocks = []
        for idx in range(len(segments)):
            accumulated_segment = pd.concat(segments[: idx + 1], ignore_index=True)
            accumulated_segment = accumulated_segment.sample(frac=1.0).reset_index(drop=True)
            progressive_blocks.append(accumulated_segment)

        accumulated_data = pd.concat(progressive_blocks, ignore_index=True)
        X_train = accumulated_data[feature_cols]
        y_train = accumulated_data[target_col].to_numpy()
        
        return X_train, y_train
    
    @classmethod
    def create_empty_metrics_table(
        cls,
        selection_config
    ) -> pd.DataFrame:
        model_metrics = cls._metrics_for_splits(
            [TrainingDataSplitEnum.TRAIN, TrainingDataSplitEnum.VAL]
        )
        columns = model_metrics.copy()
        
        if selection_config is config.volatility_models_config.training_data_config.custom_error_1:
            columns += [f"{col}_std" for col in model_metrics]

        return pd.DataFrame(columns=columns)

    def run_kfolds_training(
        self,
        force_reload: bool = False,
    ) -> tuple[pd.DataFrame, str]:
        family_name = self.model_family.get_family_name()
        metadata_path = VOLATILITY_FAMILY_METADATA_DIR_PATH / f"{family_name}_metadata.json"

        if not force_reload and metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            family_models_metrics_table = pd.DataFrame(**payload["metrics_table"])
            training_registry = payload["training_registry"]
            best_model_name = payload["best_model_name"]

            model_params_registry = payload.get("model_params_registry", {})
            explored_configurations = int(len(model_params_registry))
            max_configurations = int(self.model_family.get_max_configurations())
            explored_ratio = (
                100.0 * explored_configurations / max_configurations
                if max_configurations > 0
                else 0.0
            )
            logger.info(
                "Exploracion de hiperparametros para '%s' (metadata cache): %s/%s configuraciones (%.2f%%).",
                family_name,
                explored_configurations,
                max_configurations,
                explored_ratio,
            )

            Visualizer.top_n_family_models_table(
                family_models_metrics_table=family_models_metrics_table,
                n=15,
                cache_path=metadata_path,
            )
            self.select_best_model_from_metrics_table(
                metrics_table=family_models_metrics_table,
                training_registry=training_registry,
                family_name=family_name,
                selection_config=config.volatility_models_config.training_data_config.custom_error_1,
            )
            self.model_family.plots(
                {
                    "training_registry": training_registry,
                    "best_model_name": best_model_name,
                    "family_name": family_name,
                }
            )
            return family_models_metrics_table, best_model_name

        phase = TrainingPhase.CV
        models = self.model_family.build_model_candidates()
        total_family_models = len(models)
        family_models_metrics_table = self.create_empty_metrics_table(
            selection_config=config.volatility_models_config.training_data_config.custom_error_1
        )
        training_registry = {}
        model_params_registry = {}
        folds = TrainingDataHandler.load_kfolds(verbose=False)

        for model_i_name, model_i_params in tqdm(
            models.items(),
            total=total_family_models,
            desc=f"Entrenando familia: {family_name}",
        ):
            model_params_registry[model_i_name] = model_i_params
            tr, metrics = self.run_kfolds_training_for_specific_model(
                model_name=model_i_name,
                model_params=model_i_params,
                phase=phase,
                folds=folds,
            )
            family_models_metrics_table.loc[model_i_name] = metrics
            training_registry.update(tr)
        
        family_models_metrics_table = self.add_custom_error1(
            metrics_table=family_models_metrics_table,
        )

        family_models_metrics_table = family_models_metrics_table.sort_values(
            by=config.volatility_models_config.training_data_config.custom_error_1["metric"],
            ascending=(config.volatility_models_config.training_data_config.custom_error_1["mode"] == "min")
        )

        Visualizer.top_n_family_models_table(
            family_models_metrics_table=family_models_metrics_table,
            n=15,
        )

        best_model_name = self.select_best_model_from_metrics_table(
            metrics_table=family_models_metrics_table,
            training_registry=training_registry,
            family_name=family_name,
            selection_config=config.volatility_models_config.training_data_config.custom_error_1,
        )

        self.model_family.plots(
            {
                "training_registry": training_registry,
                "best_model_name": best_model_name,
                "family_name": family_name,
            }
        )

        self.save_family_metadata(
            family_name=family_name,
            best_model_name=best_model_name,
            metrics_table=family_models_metrics_table,
            model_params_registry=model_params_registry,
            training_registry=training_registry,
        )
        return family_models_metrics_table, best_model_name
    
    def fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        phase: TrainingPhase,
        model_params: dict,
        shuffle: bool = True,
    ) -> tuple:
        if shuffle:
            indices = np.random.permutation(len(X_train))
            X_train = X_train.iloc[indices]
            y_train = y_train[indices]

        model = self.model_family.instantiate_model(
            input_dim=X_train.shape[1],
            model_params=model_params,
        )

        fit_result = self.model_family.fit_model(
            model=model,
            model_params=model_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            phase=phase,
            shuffle=shuffle,
        )

        return fit_result, model

    def retrain_best_params_family(
        self,
        phase: TrainingPhase = TrainingPhase.TRAIN_VAL,
        upload_to_gcp: bool = True,
    ):
        family_name = self.model_family.get_family_name()
        model_name = f"{family_name}_best"
        metadata_path = VOLATILITY_FAMILY_METADATA_DIR_PATH / f"{family_name}_metadata.json"
        phase_suffix = str(getattr(phase, "value", phase)).lower()
        is_train_val = phase_suffix == TrainingPhase.TRAIN_VAL.value
        is_final_test = phase_suffix == TrainingPhase.FINAL_TEST.value
        retrained_metadata_path = (
            VOLATILITY_RETRAINED_METADATA_DIR_PATH
            / f"{family_name}_{phase_suffix}_retrained_metadata.json"
        )

        can_use_retrained_cache = (
            is_train_val and retrained_metadata_path.exists()
        )

        if can_use_retrained_cache:
            with open(retrained_metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            result_series = pd.Series(payload.get("result_metrics", {}), name=model_name)
            Visualizer.best_model_family_retrained(
                result_series=result_series,
                phase=phase,
                cache_path=retrained_metadata_path,
            )

            training_information = payload.get("training_information", {})
            self.model_family.plots(
                {
                    "training_information": training_information,
                    "best_model_name": model_name,
                    "family_name": family_name,
                    "phase": phase,
                }
            )

            return result_series

        if not metadata_path.exists():
            logger.warning("Metadata de familia no encontrada para '%s'. Ejecutando entrenamiento con k-folds...", family_name)
            Trainer(self.model_family).run_kfolds_training()

        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        best_model_params = payload["best_model_params"]
        custom_error_1 = payload["best_model_metrics"][config.volatility_models_config.training_data_config.custom_error_1["metric"]]

        if is_train_val:
            train_data, val_data, _ = TrainingDataHandler.load_full_features_splitted_data(verbose=False)

            X_train = train_data[BASE_FEATURE_COLS]
            y_train = train_data[TARGET_COLUMN].to_numpy()
            X_val = val_data[BASE_FEATURE_COLS]
            y_val = val_data[TARGET_COLUMN].to_numpy()

            train_label = TrainingDataSplitEnum.TRAIN
            evaluation_label = TrainingDataSplitEnum.VAL

        elif is_final_test:
            train_data, test_data = TrainingDataHandler.load_full_features_trainval_test_data(verbose=False)

            X_train = train_data[BASE_FEATURE_COLS]
            y_train = train_data[TARGET_COLUMN].to_numpy()
            X_val = test_data[BASE_FEATURE_COLS]
            y_val = test_data[TARGET_COLUMN].to_numpy()

            train_label = TrainingDataSplitEnum.TRAIN_VAL
            evaluation_label = TrainingDataSplitEnum.TEST

        logger.info(
            "Reentrenamiento '%s' | phase=%s | input fit: train_rows=%s eval_rows=%s total_rows=%s",
            family_name,
            phase.value if hasattr(phase, "value") else phase,
            len(X_train),
            len(X_val),
            len(X_train) + len(X_val),
        )

        fit_result, model = self.fit_model(
            model_params=best_model_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            phase=phase,
            shuffle=True,
        )
        
        metrics_dict = self.calculate_regression_metrics(
            y_train,
            fit_result.train_predictions,
            y_val,
            fit_result.validation_predictions,
            evaluation_label=evaluation_label,
            train_label=train_label,
        )

        if is_train_val:
            metrics_dict = self.add_custom_error2(
                metrics_dict=metrics_dict,
                custom_error_1=custom_error_1,
            )

        result_series = pd.Series(metrics_dict, name=model_name)

        Visualizer.best_model_family_retrained(
            result_series=result_series,
            phase=phase,
        )

        training_information = {
            "model": model,
            "best_iteration": fit_result.best_iteration,
            "best_score": fit_result.best_score,
            "epoch_history": fit_result.epoch_history or {},
            "result_series": result_series,
            "y_val_true": y_val.tolist(),
            "y_val_pred": fit_result.validation_predictions.tolist(),
        }

        self.model_family.plots(
            {
                "training_information": training_information,
                "best_model_name": model_name,
                "family_name": family_name,
                "phase": phase,
            }
        )
        
        self.save_retrained_metadata(
            family_name=family_name,
            phase=phase,
            model_params=best_model_params,
            training_information=training_information,
        )

        if is_final_test:
            self.model_family.save_model(
                model=model,
                scaler=fit_result.feature_scaler,
            )

            model_path = self.model_family.get_model_path()
            get_scaler_path = getattr(self.model_family, "get_scaler_path", None)
            scaler_path = get_scaler_path() if callable(get_scaler_path) else None

            _run_model2dashboard_pipeline(family_name=family_name)

            if upload_to_gcp:
                _upload_models_to_gcp(
                    family_name=family_name,
                    model_path=model_path,
                    scaler_path=scaler_path,
                    retrained_metadata_path=retrained_metadata_path,
                )

        return result_series

    def retrain_best_params_family_progressive(
        self,
        phase: TrainingPhase = TrainingPhase.TRAIN_VAL,
        upload_to_gcp: bool = True,
    ):
        family_name = self.model_family.get_family_name()
        progressive_family_name = f"{family_name}_retrained_progressive"
        phase_value = str(getattr(phase, "value", phase)).lower()
        is_train_val = phase_value == TrainingPhase.TRAIN_VAL.value
        is_final_test = phase_value == TrainingPhase.FINAL_TEST.value
        metadata_path = VOLATILITY_FAMILY_METADATA_DIR_PATH / f"{family_name}_metadata.json"
        retrained_metadata_path = (
            VOLATILITY_RETRAINED_METADATA_DIR_PATH
            / f"{family_name}_{phase_value}_retrained_progressive_metadata.json"
        )

        can_use_retrained_cache = (
            is_train_val and retrained_metadata_path.exists()
        )

        if can_use_retrained_cache:
            with open(retrained_metadata_path, "r", encoding="utf-8") as f:
                cached_payload = json.load(f)

            result_series = pd.Series(
                cached_payload.get("result_metrics", {}),
                name=f"{progressive_family_name}_best",
            )
            Visualizer.best_model_family_retrained(
                result_series=result_series,
                phase=phase,
                cache_path=retrained_metadata_path,
            )

            training_information = cached_payload.get("training_information", {})
            self.model_family.plots(
                {
                    "training_information": training_information,
                    "best_model_name": f"{progressive_family_name}_best",
                    "family_name": progressive_family_name,
                    "phase": phase,
                }
            )

            progressive_results = {
                "progressive": {
                    "metrics": cached_payload.get("result_metrics", {}),
                    "best_iteration": training_information.get("best_iteration"),
                    "best_score": training_information.get("best_score"),
                    "epoch_history": training_information.get("epoch_history", {}),
                    "y_val_true": training_information.get("y_val_true", []),
                    "y_val_pred": training_information.get("y_val_pred", []),
                }
            }
            return progressive_results

        if not is_final_test and upload_to_gcp:
            upload_to_gcp = False
        
        if not metadata_path.exists():
            logger.warning("Metadata de familia no encontrada para '%s'. Ejecutando entrenamiento con k-folds...", family_name)
            Trainer(self.model_family).run_kfolds_training()

        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        best_model_params = payload["best_model_params"]
        custom_error_1 = payload["best_model_metrics"][config.volatility_models_config.training_data_config.custom_error_1["metric"]]

        # Load data
        if is_train_val:
            train_data, val_data, _ = TrainingDataHandler.load_full_features_splitted_data(verbose=False)
            train_label = TrainingDataSplitEnum.TRAIN
            evaluation_label = TrainingDataSplitEnum.VAL
        elif is_final_test:
            train_data, test_data = TrainingDataHandler.load_full_features_trainval_test_data(verbose=False)
            train_data = train_data.copy()
            test_data = test_data.copy()
            train_label = TrainingDataSplitEnum.TRAIN_VAL
            evaluation_label = TrainingDataSplitEnum.TEST
            val_data = test_data

        PROGRESSIVE_TRAINING_CONFIG = config.volatility_models_config.training_data_config
        n_segments = int(PROGRESSIVE_TRAINING_CONFIG.n_segments)
        moneyness_col = PROGRESSIVE_TRAINING_CONFIG.moneyness_column

        # Segment by distance to ATM: increasing |log(S/K)| (ATM -> further from ATM)
        train_data_sorted = train_data.assign(_atm_distance=train_data[moneyness_col].abs())
        train_data_sorted = train_data_sorted.sort_values("_atm_distance").drop(columns=["_atm_distance"]).reset_index(drop=True)
        segment_size = len(train_data_sorted) // n_segments
        segments = [
            train_data_sorted.iloc[i * segment_size : (i + 1) * segment_size]
            for i in range(n_segments - 1)
        ]
        segments.append(train_data_sorted.iloc[(n_segments - 1) * segment_size :])

        # Prepare training data progressively and fit model
        X_train, y_train = self._build_progressive_training_data(
            segments=segments,
            feature_cols=BASE_FEATURE_COLS,
            target_col=TARGET_COLUMN,
        )
        X_val = val_data[BASE_FEATURE_COLS]
        y_val = val_data[TARGET_COLUMN].to_numpy()

        logger.info(
            "Reentrenamiento progresivo '%s' | phase=%s | n_segments=%s | input fit: train_rows=%s eval_rows=%s total_rows=%s",
            progressive_family_name,
            phase.value if hasattr(phase, "value") else phase,
            n_segments,
            len(X_train),
            len(X_val),
            len(X_train) + len(X_val),
        )

        fit_result, model = self.fit_model(
            model_params=best_model_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            phase=phase,
            shuffle=False,
        )

        metrics_dict = self.calculate_regression_metrics(
            y_train,
            fit_result.train_predictions,
            y_val,
            fit_result.validation_predictions,
            evaluation_label=evaluation_label,
            train_label=train_label,
        )

        if is_train_val:
            metrics_dict = self.add_custom_error2(
                metrics_dict=metrics_dict,
                custom_error_1=custom_error_1,
            )

        result_series = pd.Series(
            metrics_dict,
            name=f"{progressive_family_name}_best",
        )
        
        progressive_results = {
            "progressive": {
                "metrics": result_series.to_dict(),
                "n_train_samples": len(X_train),
                "best_iteration": fit_result.best_iteration,
                "best_score": fit_result.best_score,
                "epoch_history": fit_result.epoch_history or {},
                "y_val_true": y_val.tolist(),
                "y_val_pred": fit_result.validation_predictions.tolist(),
            }
        }
        
        Visualizer.best_model_family_retrained(
            result_series=result_series,
            phase=phase,
            cache_path=None,
        )

        training_information = {
            "model": model,
            "best_iteration": fit_result.best_iteration,
            "best_score": fit_result.best_score,
            "epoch_history": fit_result.epoch_history or {},
            "result_series": result_series,
            "y_val_true": y_val.tolist(),
            "y_val_pred": fit_result.validation_predictions.tolist(),
        }

        self.model_family.plots(
            {
                "training_information": training_information,
                "best_model_name": f"{progressive_family_name}_best",
                "family_name": progressive_family_name,
                "phase": phase,
            }
        )

        payload = {
            "family_name": progressive_family_name,
            "model_params": self._to_builtin(best_model_params),
            "result_metrics": self._to_builtin(result_series.to_dict()),
            "training_information": {
                "best_iteration": self._to_builtin(training_information.get("best_iteration")),
                "best_score": self._to_builtin(training_information.get("best_score")),
                "epoch_history": self._to_builtin(training_information.get("epoch_history", {})),
                "y_val_true": training_information.get("y_val_true", []),
                "y_val_pred": training_information.get("y_val_pred", []),
            },
        }

        with open(retrained_metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        Visualizer.info_confirmation(retrained_metadata_path, label="Metadata progresiva guardada")

        if is_final_test:
            self.model_family.save_model(
                model=model,
                scaler=fit_result.feature_scaler,
                family_name_override=progressive_family_name,
            )

            model_path_progressive = self.model_family.get_model_path()
            model_path_progressive = model_path_progressive.with_name(
                f"{progressive_family_name}{model_path_progressive.suffix}"
            )
            
            get_scaler_path = getattr(self.model_family, "get_scaler_path", None)
            scaler_path_progressive = None
            if callable(get_scaler_path):
                scaler_path_base = get_scaler_path()
                if scaler_path_base:
                    scaler_path_progressive = scaler_path_base.with_name(
                        f"{progressive_family_name}_scaler{scaler_path_base.suffix}"
                    )

            _run_model2dashboard_pipeline(family_name=progressive_family_name)

            if upload_to_gcp:
                _upload_models_to_gcp(
                    family_name=progressive_family_name,
                    model_path=model_path_progressive,
                    scaler_path=scaler_path_progressive,
                    retrained_metadata_path=retrained_metadata_path,
                )

        return progressive_results

def _run_model2dashboard_pipeline(family_name: str):
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.model2dashboard.pipeline",
                "--family",
                family_name,
            ],
            check=True,
        )
        logger.info(
            "Pipeline de model2dashboard para familia '%s' ejecutada correctamente.",
            family_name,
        )
    except Exception as e:
        logger.exception("Error ejecutando pipeline de model2dashboard: %s", e)
        raise

def _find_gcloud():
    for name in ("gcloud", "gcloud.cmd", "gcloud.exe"):
        gcloud = shutil.which(name)
        if gcloud:
            return gcloud

    possible_paths = []

    # Windows
    if os.name == "nt":
        possible_paths.extend([
            r"%LOCALAPPDATA%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            r"%PROGRAMFILES%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            r"%PROGRAMFILES(X86)%\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            r"%PROGRAMDATA%\chocolatey\bin\gcloud.cmd",
            r"%USERPROFILE%\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        ])

    else: # Linux y macOS
        possible_paths.extend([
            "/usr/local/bin/gcloud",
            "/opt/homebrew/bin/gcloud",
        ])
        possible_paths.append(os.path.expanduser("~/google-cloud-sdk/bin/gcloud"))
        possible_paths.append("/snap/bin/gcloud")
        possible_paths.extend([
            "/usr/bin/gcloud",
            "/usr/local/sbin/gcloud",
            "/opt/google-cloud-sdk/bin/gcloud",
        ])

    for path in possible_paths:
        expanded = os.path.expandvars(os.path.expanduser(path))
        if os.path.exists(expanded) and os.access(expanded, os.X_OK):
            return expanded

    logger.debug("No se encontro un ejecutable funcional de gcloud en PATH ni en rutas conocidas.")

    return None


def _upload_models_to_gcp(
    family_name: str,
    model_path,
    scaler_path=None,
    retrained_metadata_path=None,
):
    env = os.environ.copy()
    gcloud_path = _find_gcloud()

    if not gcloud_path:
        logger.warning(
            "No se encontró 'gcloud'. Se omite subida a GCP para la familia '%s'.",
            family_name,
        )
        return

    if "VIRTUAL_ENV" in env:
        python_rel = os.path.join("Scripts", "python.exe") if os.name == "nt" else os.path.join("bin", "python")
        env["CLOUDSDK_PYTHON"] = os.path.join(env["VIRTUAL_ENV"], python_rel)

    credentials_path = getattr(config.clientserver_config.gcp_storage, "credentials_path", None)
    if credentials_path:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

    dashboard_family_path = os.path.join("src", "dashboard", "saved_models", family_name)

    gcp_storage = config.clientserver_config.gcp_storage
    VOLATILITY_BUCKET = gcp_storage.trained_models_bucket
    DASHBOARD_BUCKET = gcp_storage.explainability_artifacts_bucket
    TRAINED_MODELS_PREFIX = gcp_storage.trained_models_prefix.strip("/")
    RETRAINED_METADATA_PREFIX = gcp_storage.retrained_metadata_prefix.strip("/")

    model_destination = (
        f"gs://{VOLATILITY_BUCKET}/{TRAINED_MODELS_PREFIX}/"
        if TRAINED_MODELS_PREFIX
        else f"gs://{VOLATILITY_BUCKET}/"
    )
    metadata_destination = (
        f"gs://{VOLATILITY_BUCKET}/{RETRAINED_METADATA_PREFIX}/"
        if RETRAINED_METADATA_PREFIX
        else f"gs://{VOLATILITY_BUCKET}/"
    )
    final_test_metadata_path = retrained_metadata_path

    if not VOLATILITY_BUCKET:
        logger.warning(
            "trained_models_bucket no está configurado. Se omite subida de modelo/scaler/metadatos para '%s'.",
            family_name,
        )

    if not DASHBOARD_BUCKET:
        logger.warning(
            "explainability_artifacts_bucket no está configurado. Se omite subida de artefactos dashboard para '%s'.",
            family_name,
        )

    try:
        if VOLATILITY_BUCKET and model_path and os.path.isfile(model_path):
            subprocess.run(
                [
                    gcloud_path,
                    "storage",
                    "cp",
                    str(model_path),
                    model_destination,
                ],
                check=True,
                env=env,
                shell=(os.name == "nt")
            )
            logger.info(
                "Modelo entrenado '%s' subido a GCP: %s -> %s",
                family_name,
                model_path,
                model_destination,
            )
        else:
            logger.warning("No existe modelo entrenado para subir: %s", model_path)

        if VOLATILITY_BUCKET and scaler_path and os.path.isfile(scaler_path):
            subprocess.run(
                [
                    gcloud_path,
                    "storage",
                    "cp",
                    str(scaler_path),
                    model_destination,
                ],
                check=True,
                env=env,
                shell=(os.name == "nt")
            )
            logger.info(
                "Scaler de modelo entrenado '%s' subido a GCP: %s -> %s",
                family_name,
                scaler_path,
                model_destination,
            )

        if VOLATILITY_BUCKET and final_test_metadata_path and os.path.isfile(final_test_metadata_path):
            subprocess.run(
                [
                    gcloud_path,
                    "storage",
                    "cp",
                    str(final_test_metadata_path),
                    metadata_destination,
                ],
                check=True,
                env=env,
                shell=(os.name == "nt"),
            )
            logger.info(
                "Metadata final-test '%s' subida a GCP: %s -> %s",
                family_name,
                final_test_metadata_path,
                metadata_destination,
            )
        else:
            logger.warning(
                "No existe metadata final-test para subir: %s",
                final_test_metadata_path,
            )

        if DASHBOARD_BUCKET and os.path.isdir(dashboard_family_path):
            dashboard_destination = f"gs://{DASHBOARD_BUCKET}/{family_name}/"
            subprocess.run(
                [
                    gcloud_path,
                    "storage",
                    "cp",
                    "--recursive",
                    os.path.join(dashboard_family_path, "*"),
                    dashboard_destination,
                ],
                check=True,
                env=env,
                shell=(os.name == "nt"),
            )
            logger.info(
                "Artefactos de dashboard '%s' subidos a GCP: %s -> %s",
                family_name,
                dashboard_family_path,
                dashboard_destination,
            )
        else:
            logger.warning("No existe carpeta de dashboard para '%s', omitiendo.", family_name)

    except Exception as e:
        logger.exception("Error al subir modelos a GCP: %s", e)
        raise