import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from src.config.config import (
    VOLATILITY_FAMILY_METADATA_DIR_PATH,
    VOLATILITY_RETRAINED_METADATA_DIR_PATH,
    VOLATILITY_TRAINED_MODELS_DIR_PATH,
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


class Trainer:
    def __init__(self, model_family: VolatilityModelFamilyABC):
        self.model_family = model_family

    @staticmethod
    def _dataset_suffix(use_atm: bool = False) -> str:
        return "_atm" if use_atm else ""

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

        train_metric_col = f"train_{base_metric}"
        valid_metric_col = f"val_{base_metric}"
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

        train_metric_col = f"train_{base_metric}"
        valid_metric_col = f"val_{base_metric}"

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
    def calculate_metrics_across_folds(fold_metrics_list, round_digits: int = 6):
        """
        Calcula estadísticas agregadas (media y desviación estándar) de métricas a través de los folds.
        """
        models_metrics = config.volatility_models_config.training_data_config.models_metrics
        metric_keys = [k for k in models_metrics if k in fold_metrics_list[0]]
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
        phase: TrainingPhase,
        evaluation_label: TrainingDataSplitEnum = TrainingDataSplitEnum.VAL,
        use_atm: bool = False,
        folds: dict | None = None,
    ):
        training_registry = {}
        if folds is None:
            folds = TrainingDataHandler.load_kfolds(False, use_atm=use_atm)
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
        use_atm: bool = False,
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

        file_path = VOLATILITY_FAMILY_METADATA_DIR_PATH / f"{family_name}{cls._dataset_suffix(use_atm)}_metadata.json"
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
        use_atm: bool = False,
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

        file_path = VOLATILITY_RETRAINED_METADATA_DIR_PATH / f"{family_name}{cls._dataset_suffix(use_atm)}_{phase_suffix}_retrained_metadata.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        Visualizer.info_confirmation(file_path, label="Metadata de reentrenamiento guardada")
    
    @classmethod
    def rename_metrics(
        cls,
        metrics_series: pd.Series,
    ) -> pd.Series:
        renamed_metrics = {}
        for k, v in metrics_series.items():
            if k.startswith("val_"):
                new_key = k.replace("val_", "test_")
            else:
                new_key = k
            renamed_metrics[new_key] = v
        return pd.Series(renamed_metrics, name=metrics_series.name)

    @classmethod
    def create_empty_metrics_table(
        cls,
        selection_config
    ) -> pd.DataFrame:
        model_metrics = list(config.volatility_models_config.training_data_config.models_metrics)
        columns = model_metrics.copy()
        
        if selection_config is config.volatility_models_config.training_data_config.custom_error_1:
            columns += [f"{col}_std" for col in model_metrics]

        return pd.DataFrame(columns=columns)

    def run_kfolds_training(
        self,
        force_reload: bool = False,
        use_atm: bool = False,
    ) -> tuple[pd.DataFrame, str]:
        family_name = self.model_family.get_family_name()
        metadata_path = VOLATILITY_FAMILY_METADATA_DIR_PATH / f"{family_name}{self._dataset_suffix(use_atm)}_metadata.json"

        if not force_reload and metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            family_models_metrics_table = pd.DataFrame(**payload["metrics_table"])
            training_registry = payload["training_registry"]
            best_model_name = payload["best_model_name"]

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
        folds = TrainingDataHandler.load_kfolds(False, use_atm=use_atm)

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
                use_atm=use_atm,
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
            use_atm=use_atm,
        )
        return family_models_metrics_table, best_model_name
    
    def fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        phase: TrainingPhase,
        model_params: dict,
    ) -> ModelFitResult:
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
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
        )

        return fit_result, model

    def retrain_best_params_family(
        self,
        phase: TrainingPhase = TrainingPhase.TRAIN_VAL,
        force_reload: bool = False,
        use_atm: bool = False,
    ):
        family_name = self.model_family.get_family_name()
        model_name = f"{family_name}_best"
        metadata_path = VOLATILITY_FAMILY_METADATA_DIR_PATH / f"{family_name}{self._dataset_suffix(use_atm)}_metadata.json"
        phase_suffix = str(getattr(phase, "value", phase)).lower().replace(" ", "_")
        retrained_metadata_path = (
            VOLATILITY_RETRAINED_METADATA_DIR_PATH
            / f"{family_name}{self._dataset_suffix(use_atm)}_{phase_suffix}_retrained_metadata.json"
        )

        if not force_reload and retrained_metadata_path.exists():
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

            if phase is TrainingPhase.FINAL_TEST:
                Visualizer.info_confirmation(self.model_family.get_model_path(), label="Modelo guardado previamente en")
                get_scaler_path = getattr(self.model_family, "get_scaler_path", None)
                scaler_path = None
                if callable(get_scaler_path):
                    scaler_path = get_scaler_path()
                elif family_name == "sequential_nn":
                    scaler_path = (
                        VOLATILITY_TRAINED_MODELS_DIR_PATH
                        / f"{family_name}_scaler.joblib"
                    )

                if scaler_path is not None and scaler_path.exists():
                    Visualizer.info_confirmation(
                        scaler_path,
                        label="Scaler guardado previamente en",
                    )
            return result_series

        if not metadata_path.exists():
            Visualizer.missing_metadata_warning(family_name)
            Trainer(self.model_family).run_kfolds_training(use_atm=use_atm)

        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        best_model_params = payload["best_model_params"]
        custom_error_1 = payload["best_model_metrics"][config.volatility_models_config.training_data_config.custom_error_1["metric"]]

        if phase is TrainingPhase.TRAIN_VAL:
            train_data, val_data, _ = TrainingDataHandler.load_full_features_splitted_data(verbose=False, use_atm=use_atm)

            X_train = train_data[BASE_FEATURE_COLS].to_numpy()      
            y_train = train_data[TARGET_COLUMN].to_numpy()
            X_val = val_data[BASE_FEATURE_COLS].to_numpy()
            y_val = val_data[TARGET_COLUMN].to_numpy()

            evaluation_label = TrainingDataSplitEnum.VAL

        if phase is TrainingPhase.FINAL_TEST:
            train_data, val_data, test_data = TrainingDataHandler.load_full_features_splitted_data(verbose=False, use_atm=use_atm)
            train_data = pd.concat([train_data, val_data], ignore_index=True)

            X_train = train_data[BASE_FEATURE_COLS].to_numpy()
            y_train = train_data[TARGET_COLUMN].to_numpy()
            X_val = test_data[BASE_FEATURE_COLS].to_numpy()
            y_val = test_data[TARGET_COLUMN].to_numpy()

            evaluation_label = TrainingDataSplitEnum.TEST

        fit_result, model = self.fit_model(
            model_params=best_model_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            phase=phase,
        )
        
        metrics_dict = self.calculate_regression_metrics(
            y_train,
            fit_result.train_predictions,
            y_val,
            fit_result.validation_predictions,
            evaluation_label=evaluation_label,
        )

        if phase is TrainingPhase.TRAIN_VAL:
            metrics_dict = self.add_custom_error2(
                metrics_dict=metrics_dict,
                custom_error_1=custom_error_1,
            )

        result_series = pd.Series(metrics_dict, name=model_name)

        if phase is TrainingPhase.FINAL_TEST:
            result_series = self.rename_metrics(result_series)

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
            use_atm=use_atm,
        )

        if phase is TrainingPhase.FINAL_TEST:
            self.model_family.save_model(
                model=model,
                scaler=fit_result.feature_scaler,
            )

        return result_series