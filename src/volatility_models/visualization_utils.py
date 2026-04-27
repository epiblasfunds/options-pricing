import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.config.config import config
from src.enums.volatility_model_enums.training_phase import TrainingPhase

logger = logging.getLogger(__name__)


class Visualizer:

    @staticmethod
    def info_confirmation(info_path, label="Información guardada en"):
        logger.info(f"{label}: {info_path}")

    @staticmethod
    def missing_metadata_warning(family_name: str):
        logger.error(
            f"No se encontró metadata para la familia {family_name}. "
            "Se requiere ejecutar `run_kfolds_training`."
        )

    @staticmethod
    def best_model_family_retrained(result_series, phase, cache_path=None):
        if cache_path is not None:
            logger.info(f"Metadata de reentrenamiento cargada desde cache: {cache_path}")
        if phase is TrainingPhase.TRAIN_VAL:
            title = "Resultados del mejor modelo de la familia tras el reentrenamiento:"
        elif phase is TrainingPhase.FINAL_TEST:
            title = "Resultados del mejor modelo de la familia tras el reentrenamiento con train + val:"
        else:
            title = "Resultados del mejor modelo de la familia:"
        result_df = pd.DataFrame([result_series], index=["retrained_best"])
        logger.info("%s\n%s", title, result_df.to_string())

    @staticmethod
    def top_n_family_models_table(
        family_models_metrics_table,
        n=10,
        cache_path=None,
    ):
        if cache_path is not None:
            logger.info(f"Metadata cargada desde cache: {cache_path}")
        top_n_models = family_models_metrics_table.head(n)
        logger.info("Mostrando top %s modelos de la familia\n%s", n, top_n_models.to_string())

    @staticmethod
    def progressive_training_segments_graphics(
        progressive_results: dict,
        family_name: str,
        phase: TrainingPhase,
    ):
        if not progressive_results:
            logger.info("No hay resultados progresivos para mostrar.")
            return

        rows = []
        for segment_name, segment_payload in progressive_results.items():
            try:
                segment_idx = int(str(segment_name).split("_")[1])
            except Exception:
                segment_idx = len(rows)

            row = {
                "segment": segment_name,
                "segment_idx": segment_idx,
                "n_train_samples": segment_payload.get("n_train_samples", np.nan),
            }
            row.update(segment_payload.get("metrics", {}))
            rows.append(row)

        progressive_df = pd.DataFrame(rows).sort_values("segment_idx").reset_index(drop=True)
        logger.info(
            "Evolucion del reentrenamiento progresivo (%s)\n%s",
            family_name,
            progressive_df.to_string(index=False),
        )

        eval_prefix = "test" if phase is TrainingPhase.FINAL_TEST else "val"
        eval_rmse_col = f"{eval_prefix}_rmse"
        eval_mae_col = f"{eval_prefix}_mae"
        eval_r2_col = f"{eval_prefix}_r2"

        if eval_rmse_col not in progressive_df.columns:
            return

        segment_labels = progressive_df["segment"].tolist()

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        if "train_rmse" in progressive_df.columns:
            axes[0, 0].plot(segment_labels, progressive_df["train_rmse"], marker="o", label="train RMSE")
        axes[0, 0].plot(segment_labels, progressive_df[eval_rmse_col], marker="o", label=f"{eval_prefix} RMSE")
        axes[0, 0].set_title(f"{family_name} - RMSE por segmento")
        axes[0, 0].set_xlabel("Segmento")
        axes[0, 0].set_ylabel("RMSE")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.25)

        if "train_mae" in progressive_df.columns and eval_mae_col in progressive_df.columns:
            axes[0, 1].plot(segment_labels, progressive_df["train_mae"], marker="o", label="train MAE")
            axes[0, 1].plot(segment_labels, progressive_df[eval_mae_col], marker="o", label=f"{eval_prefix} MAE")
            axes[0, 1].set_title(f"{family_name} - MAE por segmento")
            axes[0, 1].set_xlabel("Segmento")
            axes[0, 1].set_ylabel("MAE")
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.25)
        else:
            axes[0, 1].set_visible(False)

        if "train_r2" in progressive_df.columns and eval_r2_col in progressive_df.columns:
            x = np.arange(len(segment_labels))
            width = 0.38
            axes[1, 0].bar(x - width / 2, progressive_df["train_r2"], width=width, label="train R2")
            axes[1, 0].bar(x + width / 2, progressive_df[eval_r2_col], width=width, label=f"{eval_prefix} R2")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(segment_labels, rotation=20)
            axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
            axes[1, 0].set_title(f"{family_name} - R2 por segmento")
            axes[1, 0].set_xlabel("Segmento")
            axes[1, 0].set_ylabel("R2")
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.25)
        else:
            axes[1, 0].set_visible(False)

        selected_segment_name = segment_labels[-1]
        if phase is TrainingPhase.TRAIN_VAL:
            custom_error_2_metric = (
                config.volatility_models_config.training_data_config.custom_error_2["metric"]
            )
            candidate_segment_labels = [
                segment_name
                for segment_name in segment_labels
                if custom_error_2_metric
                in progressive_results.get(segment_name, {}).get("metrics", {})
            ]
            if candidate_segment_labels:
                selected_segment_name = min(
                    candidate_segment_labels,
                    key=lambda segment_name: progressive_results[segment_name]["metrics"][
                        custom_error_2_metric
                    ],
                )

        selected_segment = progressive_results.get(selected_segment_name, {})
        y_true = np.asarray(selected_segment.get("y_val_true", []), dtype=float)
        y_pred = np.asarray(selected_segment.get("y_val_pred", []), dtype=float)
        if y_true.size > 0 and y_pred.size > 0:
            axes[1, 1].scatter(y_true, y_pred, s=8, alpha=0.35, edgecolor="none")
            lower = float(np.nanmin([y_true.min(), y_pred.min()]))
            upper = float(np.nanmax([y_true.max(), y_pred.max()]))
            axes[1, 1].plot([lower, upper], [lower, upper], "k--")
            axes[1, 1].set_title(f"{family_name} - Real vs Pred ({selected_segment_name})")
            axes[1, 1].set_xlabel("Real")
            axes[1, 1].set_ylabel("Pred")
            axes[1, 1].grid(alpha=0.25)
        else:
            axes[1, 1].set_visible(False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_nn_learning_curves(
        training_registry=None,
        best_model_name=None,
        family_name=None,
        training_information=None,
        phase=None,
    ):
        """
        Muestra las curvas de aprendizaje (train vs val por epoch) del mejor modelo
        en cada fold, con una línea vertical en best_epoch.
        """
        plot_title = family_name

        if training_information is not None:
            progressive_results = training_information.get("progressive_results", {})
            if progressive_results:
                segment_keys = sorted(
                    progressive_results.keys(),
                    key=lambda key: int(str(key).split("_")[1]),
                )

                n_segments = len(segment_keys)
                n_cols = 3
                n_rows = int(np.ceil(n_segments / n_cols))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
                axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
                validation_label = "test RMSE" if phase is TrainingPhase.FINAL_TEST else "val RMSE"

                for ax_idx, segment_key in enumerate(segment_keys):
                    ax = axes_flat[ax_idx]
                    segment_payload = progressive_results[segment_key]
                    epoch_history = segment_payload.get("epoch_history", {})
                    train_rmse = epoch_history.get("rmse", [])
                    val_rmse = epoch_history.get("val_rmse", [])
                    best_epoch = segment_payload.get("best_iteration")

                    if train_rmse:
                        ax.plot(
                            range(1, len(train_rmse) + 1),
                            train_rmse,
                            label="train RMSE",
                            linewidth=1.2,
                            alpha=0.85,
                        )
                    if val_rmse:
                        ax.plot(
                            range(1, len(val_rmse) + 1),
                            val_rmse,
                            label=validation_label,
                            linewidth=1.2,
                            alpha=0.85,
                        )

                    max_epochs = max(len(train_rmse), len(val_rmse))
                    if best_epoch and best_epoch <= max_epochs:
                        ax.axvline(
                            best_epoch,
                            color="red",
                            linestyle="--",
                            linewidth=1,
                            label=f"best epoch ({best_epoch})",
                        )

                    ax.set_title(segment_key)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("RMSE")
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.25)

                for ax_idx in range(n_segments, len(axes_flat)):
                    axes_flat[ax_idx].set_visible(False)

                fig.suptitle(f"{plot_title} - curvas de aprendizaje por segmento", y=1.02)
                plt.tight_layout()
                plt.show()
                return

            epoch_history = training_information.get("epoch_history", {})
            train_rmse = epoch_history.get("rmse", [])
            val_rmse = epoch_history.get("val_rmse", [])
            best_epoch = training_information.get("best_iteration", None)

            if not train_rmse and not val_rmse:
                logger.info("No hay historial de epocas para graficar curvas de aprendizaje.")
                return

            phase_value = phase.value
            validation_label = "test RMSE" if phase is TrainingPhase.FINAL_TEST else "val RMSE"
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))

            if train_rmse:
                ax.plot(
                    range(1, len(train_rmse) + 1),
                    train_rmse,
                    label="train RMSE",
                    linewidth=1.4,
                    alpha=0.9,
                )
            if val_rmse:
                ax.plot(
                    range(1, len(val_rmse) + 1),
                    val_rmse,
                    label=validation_label,
                    linewidth=1.4,
                    alpha=0.9,
                )

            max_epochs = max(len(train_rmse), len(val_rmse))
            if best_epoch and best_epoch <= max_epochs:
                ax.axvline(
                    best_epoch,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label=f"best epoch ({best_epoch})",
                )

            ax.set_title(f"{plot_title} - curvas de aprendizaje ({phase_value})")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("RMSE")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()
            return

        if training_registry is None or best_model_name is None:
            logger.info("Faltan datos para graficar curvas de aprendizaje.")
            return

        fold_keys = sorted(
            [k for k in training_registry.keys() if k.startswith(f"{best_model_name}_fold-")],
            key=lambda x: int(x.split("fold-")[-1]),
        )

        n_folds = len(fold_keys)
        n_cols = 3
        n_rows = int(np.ceil(n_folds / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax_idx, fold_key in enumerate(fold_keys):
            ax = axes_flat[ax_idx]
            record = training_registry[fold_key]
            fold_name = fold_key.split("_")[-1]
            epoch_history = record.get("epoch_history", {})

            val_rmse = epoch_history.get("val_rmse", [])
            train_rmse = epoch_history.get("rmse", [])
            best_epoch = record.get("best_iteration", None)

            train_epochs = range(1, len(train_rmse) + 1)
            val_epochs = range(1, len(val_rmse) + 1)
            max_epochs = max(len(train_rmse), len(val_rmse))

            if train_rmse:
                ax.plot(
                    train_epochs,
                    train_rmse,
                    label="train RMSE",
                    linewidth=1.2,
                    alpha=0.85,
                )
            if val_rmse:
                ax.plot(
                    val_epochs,
                    val_rmse,
                    label="val RMSE",
                    linewidth=1.2,
                    alpha=0.85,
                )

            if best_epoch and best_epoch <= max_epochs:
                ax.axvline(
                    best_epoch, color="red", linestyle="--", linewidth=1,
                    label=f"best epoch ({best_epoch})"
                )

            ax.set_title(fold_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("RMSE")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

        for ax_idx in range(n_folds, len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        fig.suptitle(f"{plot_title} - curvas de aprendizaje por fold", y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def best_model_family_graphics(
        training_registry,
        best_model_row,
        family_name=None,
        sample_size: int = 7000,
    ):
        """
        Gráficas representativas del mejor modelo a lo largo de los folds:
        evolución de métricas train/val, gap de sobreajuste y scatter agregado real vs pred.
        """

        best_model_name = str(best_model_row.index[0])
        plot_title = family_name if family_name is not None else best_model_name
        fold_keys = sorted(
            [k for k in training_registry.keys() if k.startswith(f"{best_model_name}_fold-")],
            key=lambda x: int(x.split("fold-")[-1]),
        )

        fold_rows = []
        all_val_true = []
        all_val_pred = []

        for fold_key in fold_keys:
            record = training_registry[fold_key]
            fold_name = fold_key.split("_")[-1]
            fold_metrics = record["metrics"]

            fold_rows.append(
                {
                    "fold": fold_name,
                    "train_mae": fold_metrics["train_mae"],
                    "train_rmse": fold_metrics["train_rmse"],
                    "train_r2": fold_metrics["train_r2"],
                    "val_mae": fold_metrics["val_mae"],
                    "val_rmse": fold_metrics["val_rmse"],
                    "val_r2": fold_metrics["val_r2"],
                    "rmse_gap": fold_metrics["val_rmse"] - fold_metrics["train_rmse"],
                }
            )

            all_val_true.extend(record["y_val_true"])
            all_val_pred.extend(record["y_val_pred"])

        fold_metrics_df = pd.DataFrame(fold_rows)

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        axes[0, 0].plot(
            fold_metrics_df["fold"], fold_metrics_df["train_rmse"], marker="o", label="train RMSE"
        )
        axes[0, 0].plot(
            fold_metrics_df["fold"], fold_metrics_df["val_rmse"], marker="o", label="val RMSE"
        )
        axes[0, 0].set_title(f"{plot_title} - RMSE por fold")
        axes[0, 0].set_xlabel("Fold")
        axes[0, 0].set_ylabel("RMSE")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.25)

        axes[0, 1].plot(
            fold_metrics_df["fold"], fold_metrics_df["train_mae"], marker="o", label="train MAE"
        )
        axes[0, 1].plot(
            fold_metrics_df["fold"], fold_metrics_df["val_mae"], marker="o", label="val MAE"
        )
        axes[0, 1].set_title(f"{plot_title} - MAE por fold")
        axes[0, 1].set_xlabel("Fold")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.25)

        x = np.arange(len(fold_metrics_df["fold"]))
        width = 0.38
        axes[1, 0].bar(x - width / 2, fold_metrics_df["train_r2"], width=width, label="train R2")
        axes[1, 0].bar(x + width / 2, fold_metrics_df["val_r2"], width=width, label="val R2")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(fold_metrics_df["fold"])
        axes[1, 0].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1, 0].set_title(f"{plot_title} - R2 por fold")
        axes[1, 0].set_xlabel("Fold")
        axes[1, 0].set_ylabel("R2")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.25)

        # El scatter se construye con el fold de último índice
        max_fold_key = (
            max(fold_keys, key=lambda k: int(k.split("fold-")[-1]))
        )
        last_fold_record = training_registry[max_fold_key] if max_fold_key else {}
        last_fold_valid_true = last_fold_record.get("y_val_true", last_fold_record.get("y_valid_true", []))
        last_fold_valid_pred = last_fold_record.get("y_val_pred", last_fold_record.get("y_valid_pred", []))
        if last_fold_valid_true and last_fold_valid_pred:
            scatter_df = pd.DataFrame(
                {
                    "y_true": np.asarray(last_fold_valid_true, dtype=float),
                    "y_pred": np.asarray(last_fold_valid_pred, dtype=float),
                }
            )
            if len(scatter_df) > sample_size:
                scatter_df = scatter_df.sample(sample_size, random_state=42)

            min_value = float(min(scatter_df["y_true"].min(), scatter_df["y_pred"].min()))
            max_value = float(max(scatter_df["y_true"].max(), scatter_df["y_pred"].max()))

            axes[1, 1].scatter(
                scatter_df["y_true"],
                scatter_df["y_pred"],
                s=10,
                alpha=0.30,
            )
            axes[1, 1].plot([min_value, max_value], [min_value, max_value], "k--")
            axes[1, 1].set_title(f"{plot_title} - valid: real vs pred ({best_model_name})")
            axes[1, 1].set_xlabel("IV real")
            axes[1, 1].set_ylabel("IV pred")
            axes[1, 1].grid(alpha=0.25)

        fig.suptitle(f"{plot_title} - resumen del mejor candidato", y=1.02)
        plt.tight_layout()
        plt.show()
