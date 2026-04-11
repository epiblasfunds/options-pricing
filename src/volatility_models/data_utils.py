import logging
import typing as t
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.config import (
    TRAINING_DATA_SPLITTED_DIR,
    TRAINING_DATA_SPLITTED_FEATURES_DIR,
    config,
)
from src.data_management.loaders.volatility_step_loader import VolatilityStepLoader
from src.enums.data_enums import OptionTypeEnum
from src.enums.data_enums.database_schema.training_data_enum import TrainingDataEnum
from src.enums.data_enums.database_schema.volatility_db_enum import VolatilityDBEnum

logger = logging.getLogger(__name__)

TRAINING_DATA_CONFIG = config.volatility_models_config.training_data_config

SELECTED_TRADE_COLUMNS = TRAINING_DATA_CONFIG.vol_db_cols
TARGET_COL = VolatilityDBEnum.IMPLIED_VOLATILITY
BASE_NUMERIC_FEATURE_COLS = TRAINING_DATA_CONFIG.numeric_features
AUX_CONTEXT_COLS = TRAINING_DATA_CONFIG.aux_context_cols
BASE_CATEGORICAL_FEATURE_COLS = [
    enum_value
    for enum_value in TrainingDataEnum
    if enum_value not in BASE_NUMERIC_FEATURE_COLS and enum_value != TARGET_COL
]

BASE_FEATURE_COLS = BASE_NUMERIC_FEATURE_COLS + BASE_CATEGORICAL_FEATURE_COLS


class TrainingDataSplitEnum(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class TrainingDataHandler:
    @staticmethod
    def _get_splitted_data_filename(split: TrainingDataSplitEnum):
        filename = f"{split.value}_splitted_data.csv"
        return TRAINING_DATA_SPLITTED_DIR / filename

    @staticmethod
    def _get_splitted_features_data_filename(split: TrainingDataSplitEnum):
        filename = f"{split.value}_splitted_features_data.csv"
        return TRAINING_DATA_SPLITTED_FEATURES_DIR / filename

    @staticmethod
    def _get_kfolds_filename(split: TrainingDataSplitEnum, kfold_name: str):
        if split == TrainingDataSplitEnum.TEST:
            raise ValueError
        filename = f"{split.value}_kfolds_{kfold_name}.csv"
        return TRAINING_DATA_SPLITTED_DIR / filename

    @staticmethod
    def get_df_dates(data_df: pd.DataFrame):
        return np.array(
            sorted(data_df[VolatilityDBEnum.EXEC_DATETIME].dt.date.unique())
        )

    @staticmethod
    def build_features_from_trade(tr) -> dict:
        """
        Build the new features given the volatility database features
        """

        tte_years = tr[VolatilityDBEnum.TIME_TO_EXPIRATION] / 365.0
        sqrt_tte_years = np.sqrt(tte_years)

        underlying_price = tr[VolatilityDBEnum.UNDERLYING_PRICE]
        strike_price = tr[VolatilityDBEnum.STRIKE_PRICE]
        log_moneyness = np.log(underlying_price / strike_price)
        log_moneyness_sq = log_moneyness**2
        log_moneyness_x_sqrt_tte = log_moneyness * sqrt_tte_years

        rate = tr[VolatilityDBEnum.RATE]
        forward_price = underlying_price * np.exp(rate * tte_years)
        log_forward_moneyness = np.log(forward_price / strike_price)

        is_call = float(tr[VolatilityDBEnum.OPTION_TYPE].upper() == OptionTypeEnum.CALL)
        is_put = float(tr[VolatilityDBEnum.OPTION_TYPE].upper() == OptionTypeEnum.PUT)

        quantity_raw = tr[VolatilityDBEnum.QUANTITY]
        quantity_log1p = np.log1p(quantity_raw)

        underlying_lag_minutes = tr[VolatilityDBEnum.UNDERLYING_LAG_MINUTES]

        features = {
            TrainingDataEnum.TTE_YEARS: tte_years,
            TrainingDataEnum.SQRT_TTE_YEARS: sqrt_tte_years,
            TrainingDataEnum.LOG_MONEYNESS: log_moneyness,
            TrainingDataEnum.LOG_MONEYNESS_SQ: log_moneyness_sq,
            TrainingDataEnum.LOG_MONEYNESS_X_SQRT_TTE: log_moneyness_x_sqrt_tte,
            TrainingDataEnum.LOG_FORWARD_MONEYNESS: log_forward_moneyness,
            TrainingDataEnum.RATE: rate,
            TrainingDataEnum.UNDERLYING_LAG_MINUTES: underlying_lag_minutes,
            TrainingDataEnum.QUANTITY_LOG1P: quantity_log1p,
            TrainingDataEnum.IS_CALL: is_call,
            TrainingDataEnum.IS_PUT: is_put,
        }

        trade_value = str(tr[VolatilityDBEnum.TRADE_TYPE])
        for col in [
            TrainingDataEnum.TRADE_TYPE_3,
            TrainingDataEnum.TRADE_TYPE_H,
            TrainingDataEnum.TRADE_TYPE_M,
            TrainingDataEnum.TRADE_TYPE_S,
            TrainingDataEnum.TRADE_TYPE_W,
            TrainingDataEnum.TRADE_TYPE_X,
        ]:
            features[col.value] = float(f"tradeType{trade_value}" == col.value)

        exec_dt = tr[VolatilityDBEnum.EXEC_DATETIME]
        exec_hour = float(exec_dt.hour)
        exec_weekday = float(exec_dt.weekday())
        for col in [
            TrainingDataEnum.EXEC_HOUR_9,
            TrainingDataEnum.EXEC_HOUR_10,
            TrainingDataEnum.EXEC_HOUR_11,
            TrainingDataEnum.EXEC_HOUR_12,
            TrainingDataEnum.EXEC_HOUR_13,
            TrainingDataEnum.EXEC_HOUR_14,
            TrainingDataEnum.EXEC_HOUR_15,
            TrainingDataEnum.EXEC_HOUR_16,
            TrainingDataEnum.EXEC_HOUR_17,
            TrainingDataEnum.EXEC_HOUR_18,
            TrainingDataEnum.EXEC_HOUR_19,
        ]:
            features[col.value] = float(f"execHour{exec_hour}" == col.value)

        for col in [
            TrainingDataEnum.EXEC_WEEKDAY_0,
            TrainingDataEnum.EXEC_WEEKDAY_1,
            TrainingDataEnum.EXEC_WEEKDAY_2,
            TrainingDataEnum.EXEC_WEEKDAY_3,
            TrainingDataEnum.EXEC_WEEKDAY_4,
        ]:
            features[col.value] = float(f"execWeekday{exec_weekday}" == col.value)

        return features

    @staticmethod
    def split_train_test(
        data_df: pd.DataFrame,
    ) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
        train_size = TRAINING_DATA_CONFIG.train_test_split_config.train_size
        lag = TRAINING_DATA_CONFIG.train_test_split_config.lag

        exec_dates = data_df[VolatilityDBEnum.EXEC_DATETIME].dt.date
        unique_dates_total = np.array(sorted(exec_dates.unique()))
        n_dates_total = len(unique_dates_total)
        train_end = int(n_dates_total * train_size)

        train_dates = unique_dates_total[:train_end]
        test_dates = unique_dates_total[train_end + lag:]

        train_df = data_df[exec_dates.isin(train_dates)].copy()
        test_df = data_df[exec_dates.isin(test_dates)].copy()

        train_df = train_df.sort_values(VolatilityDBEnum.EXEC_DATETIME)
        test_df = test_df.sort_values(VolatilityDBEnum.EXEC_DATETIME)

        return train_df, test_df

    @classmethod
    def add_features(cls, data_df: pd.DataFrame):
        new_features = data_df.apply(
            lambda row: pd.Series(cls.build_features_from_trade(row)), axis=1
        )
        columns = BASE_FEATURE_COLS + [TARGET_COL] + AUX_CONTEXT_COLS
        logger.info(f"Rows after feature engineering: {len(data_df)}")
        result = pd.concat([data_df, new_features], axis=1)[columns]
        return result

    @staticmethod
    def load_volatility_db() -> pd.DataFrame:
        volatility_df = VolatilityStepLoader.load()
        volatility_df = volatility_df[SELECTED_TRADE_COLUMNS]
        volatility_df = volatility_df.sort_values(VolatilityDBEnum.EXEC_DATETIME)
        volatility_df = volatility_df.reset_index(drop=True)
        return volatility_df

    @classmethod
    def load_splitted_data(cls, verbose: bool = True):
        volatility_df = cls.load_volatility_db()
        trainval_df, test_df = cls.split_train_test(volatility_df)
        train_df, val_df = cls.split_train_test(trainval_df)

        if verbose:
            DataInfoDisplay.display_split_info(train_df, val_df, test_df)
        return train_df, val_df, test_df

    @staticmethod
    def read_features_splitted_data(split: TrainingDataSplitEnum):
        df = pd.read_csv(
            TrainingDataHandler._get_splitted_features_data_filename(split),
            sep=";",
            header=0,
            dtype="string",
        )

        for col in (BASE_FEATURE_COLS + [TARGET_COL]):
            df[col] = df[col].astype("float64")

        if VolatilityDBEnum.EXEC_DATETIME.value in df.columns:
            col = VolatilityDBEnum.EXEC_DATETIME.value
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S.%f")

        return df

    @staticmethod
    def _to_csv(df: pd.DataFrame, filename: Path):
        # Format Datetimes
        if VolatilityDBEnum.EXEC_DATETIME.value in df.columns:
            col = VolatilityDBEnum.EXEC_DATETIME.value
            df[col] = df[col].dt.strftime(date_format="%Y-%m-%d %H:%M:%S.%f")

        df.to_csv(
            filename,
            encoding="utf-8",
            sep=";",
            index=False,
        )

    @classmethod
    def load_full_features_splitted_data(cls, verbose: bool = True):
        train_df, val_df, test_df = cls.load_splitted_data(verbose=False)
        for data_df, split in [
            (train_df, TrainingDataSplitEnum.TRAIN),
            (val_df, TrainingDataSplitEnum.VAL),
            (test_df, TrainingDataSplitEnum.TEST),
        ]:
            filename = cls._get_splitted_features_data_filename(split)
            if not filename.exists():
                df = cls.add_features(data_df)
                cls._to_csv(df, filename=filename)

        train_df = cls.read_features_splitted_data(TrainingDataSplitEnum.TRAIN)
        val_df = cls.read_features_splitted_data(TrainingDataSplitEnum.VAL)
        test_df = cls.read_features_splitted_data(TrainingDataSplitEnum.TEST)
        if verbose:
            DataInfoDisplay.describe_dataset(
                pd.concat([train_df, val_df, test_df], axis=0)
            )
        return train_df, val_df, test_df

    @classmethod
    def load_kfolds(cls, verbose: bool = True):
        train_df, _, _ = cls.load_full_features_splitted_data(verbose=False)
        n_folds = TRAINING_DATA_CONFIG.kfolds_config.n_folds
        extra_blocks = TRAINING_DATA_CONFIG.kfolds_config.extra_blocks

        n_blocks = n_folds + extra_blocks  # t0..t6 => 7 bloques
        block_size = train_df.shape[0] // n_blocks

        folds = {}
        for i in range(n_folds):
            fold_name = f"fold-{n_folds-i}"
            kfold_full_df = train_df.iloc[: -block_size * i] if i > 0 else train_df
            fold_train_df, fold_val_df = cls.split_train_test(kfold_full_df)
            folds[fold_name] = {"train": fold_train_df, "val": fold_val_df}

        if verbose:
            DataInfoDisplay.display_kfolds_info(folds=folds, full_df=train_df)

        return folds


class DataInfoDisplay:
    @staticmethod
    def describe_dataset(data_df: pd.DataFrame):
        feature_inventory = pd.DataFrame(
            {
                "block": ["numeric", "categorical", "target", "aux_context"],
                "n_columns": [
                    len(BASE_NUMERIC_FEATURE_COLS),
                    len(BASE_CATEGORICAL_FEATURE_COLS),
                    1,
                    len(AUX_CONTEXT_COLS),
                ],
            }
        )

        logger.info("Resumen de columnas del dataset:")
        logger.info(feature_inventory)

        logger.info("\nResumen de features:")
        logger.info(
            f"  - Numéricas ({len(BASE_NUMERIC_FEATURE_COLS)}): {BASE_NUMERIC_FEATURE_COLS}"
        )
        logger.info(
            f"  - Categóricas ({len(BASE_CATEGORICAL_FEATURE_COLS)}): {BASE_CATEGORICAL_FEATURE_COLS}"
        )
        logger.info(f"  - Total features modelo: {len(BASE_FEATURE_COLS)}")
        logger.info(
            f"\nColumnas auxiliares NO usadas para entrenar: {AUX_CONTEXT_COLS}"
        )

        logger.info("\n" + "=" * 60)
        logger.info("FEATURE STATISTICS (numéricas):")
        logger.info("=" * 60)
        logger.info(data_df[BASE_NUMERIC_FEATURE_COLS].describe())

        logger.info("\n" + "=" * 60)
        logger.info("FEATURE DISTRIBUTION (categóricas):")
        logger.info("=" * 60)

        categorical_summary_groups = {
            "is_call": ["isCall"],
            "trade_type": list(
                config.volatility_models_config.training_data_config.trade_type_to_feature.values()
            ),
            "exec_hour": [
                enum_value.value
                for enum_value in TrainingDataEnum
                if enum_value.value.startswith("execHour")
            ],
            "exec_weekday": [
                enum_value.value
                for enum_value in TrainingDataEnum
                if enum_value.value.startswith("execWeekday")
            ],
        }

        for group_name, group_cols in categorical_summary_groups.items():
            logger.info(f"\n{group_name}:")
            if len(group_cols) == 1:
                logger.info(data_df[group_cols[0]].value_counts().sort_index())
            else:
                group_counts = (
                    data_df[group_cols].sum().sort_values(ascending=False).astype(int)
                )
                logger.info(group_counts.to_frame(name="count"))

        logger.info("\n" + "=" * 60)
        logger.info("TARGET VARIABLE (Implied Volatility):")
        logger.info("=" * 60)
        logger.info(data_df[[TARGET_COL]].describe())

        logger.info("\n" + "=" * 60)
        logger.info("DATA QUALITY:")
        logger.info("=" * 60)
        logger.info(f"Total rows: {len(data_df)}")
        logger.info(
            f"Date range: {data_df[VolatilityDBEnum.EXEC_DATETIME].min()} "
            f"to {data_df[VolatilityDBEnum.EXEC_DATETIME].max()}"
        )
        missing_counts = data_df[BASE_FEATURE_COLS + [TARGET_COL]].isna().sum()
        if missing_counts.sum() > 0:
            logger.info("\nMissing values in features/target:")
            logger.info(missing_counts[missing_counts > 0])
        else:
            logger.info("No missing values in features/target")

        logger.info("\nCorrelation with target (top 8):")
        correlations = (
            data_df[BASE_FEATURE_COLS + [TARGET_COL]]
            .corr()[TARGET_COL]
            .drop(TARGET_COL)
            .abs()
            .sort_values(ascending=False)
        )
        logger.info(correlations.head(8))

    @staticmethod
    def display_split_info(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        train_dates = TrainingDataHandler.get_df_dates(train_df)
        val_dates = TrainingDataHandler.get_df_dates(val_df)
        test_dates = TrainingDataHandler.get_df_dates(test_df)

        splits = [
            ("Train", train_df, train_dates),
            ("Valid", val_df, val_dates),
            ("Test", test_df, test_dates),
        ]

        total_rows = sum([len(t[1]) for t in splits])
        total_dates = sum([len(t[2]) for t in splits])

        logger.info("=" * 95)
        logger.info("SPLIT TEMPORAL INICIAL (sobre todo el dataset): Train / Valid / Test")
        logger.info("=" * 95)
        logger.info(f"Total filas: {total_rows:,} | Total fechas: {total_dates:,}")

        for name, split_df, split_dates in splits:
            rows = len(split_df)
            dates = len(split_dates)
            start_date = split_dates[0] if dates > 0 else "N/A"
            end_date = split_dates[-1] if dates > 0 else "N/A"
            logger.info(
                f"{name:<5} | rows: {rows:>8,} ({rows / total_rows:>7.2%}) "
                f"| dates: {dates:>5,} ({dates / total_dates:>7.2%}) "
                f"| range: {start_date} -> {end_date}"
            )

    @staticmethod
    def display_kfolds_info(folds: t.Dict[str, pd.DataFrame], full_df: pd.DataFrame):
        full_df_dates = TrainingDataHandler.get_df_dates(full_df)
        logger.info("\n" + "=" * 95)
        logger.info(
            "K-FOLDS TEMPORALES (solo dentro de Train) para validación por familia de modelos"
        )
        logger.info("=" * 95)
        logger.info(
            f"Train global -> filas: {len(full_df):,} | fechas: {len(full_df_dates):,}"
        )
        for fold_name in sorted(folds.keys(), key=lambda x: int(x.split("-")[-1])):
            fold_train_df = folds[fold_name]["train"]
            fold_val_df = folds[fold_name]["val"]

            fold_train_dates = TrainingDataHandler.get_df_dates(fold_train_df)
            fold_val_dates = TrainingDataHandler.get_df_dates(fold_val_df)

            train_rows = len(fold_train_df)
            val_rows = len(fold_val_df)
            train_n_dates = len(fold_train_dates)
            val_n_dates = len(fold_val_dates)

            train_start = min(fold_train_dates)
            train_end_date = max(fold_train_dates)
            val_start = min(fold_val_dates)
            val_end_date = max(fold_val_dates)

            logger.info(f"{fold_name.upper()}")
            logger.info(
                f"  TRAIN | rows: {train_rows:>8,} ({train_rows / len(full_df):>7.2%} de train) "
                f"| dates: {train_n_dates:>5,} ({train_n_dates / len(full_df_dates):>7.2%} de train) "
                f"| range: {train_start} -> {train_end_date}"
            )
            logger.info(
                f"  VALID | rows: {val_rows:>8,} ({val_rows / len(full_df):>7.2%} de train) "
                f"| dates: {val_n_dates:>5,} ({val_n_dates / len(full_df_dates):>7.2%} de train) "
                f"| range: {val_start} -> {val_end_date}"
            )
