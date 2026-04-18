import typing as t

import numpy as np
import pandas as pd

from src.config.config import config
from src.enums.data_enums import OptionTypeEnum, TrainingDataEnum, VolatilityDBEnum


def column_name(column: t.Any) -> str:
    return str(column.value) if hasattr(column, "value") else str(column)


TARGET_COLUMN = column_name(
    config.volatility_models_config.training_data_config.target_column
)
RAW_INPUT_FEATURE_NAMES = [
    column_name(column)
    for column in config.volatility_models_config.training_data_config.raw_model_input
]
RAW_TRADE_COLUMN_NAMES = [
    column_name(column)
    for column in config.volatility_models_config.training_data_config.vol_db_cols
]
BASE_NUMERIC_FEATURE_NAMES = [
    column_name(column)
    for column in config.volatility_models_config.training_data_config.numeric_features
]
BASE_CATEGORICAL_FEATURE_NAMES = [
    column_name(enum_value)
    for enum_value in TrainingDataEnum
    if column_name(enum_value) not in BASE_NUMERIC_FEATURE_NAMES
    and column_name(enum_value) != TARGET_COLUMN
]
MODEL_INPUT_FEATURE_NAMES = BASE_NUMERIC_FEATURE_NAMES + BASE_CATEGORICAL_FEATURE_NAMES
TRADE_TYPE_TO_FEATURE = {
    str(key): str(value)
    for key, value in config.volatility_models_config.training_data_config.trade_type_to_feature.items()
}
ANALYSIS_FEATURE_NAMES = [
    "Moneyness",
    column_name(VolatilityDBEnum.TIME_TO_EXPIRATION),
    column_name(VolatilityDBEnum.UNDERLYING_LAG_MINUTES),
    column_name(VolatilityDBEnum.QUANTITY),
    column_name(VolatilityDBEnum.RATE),
]


def load_test_trade_frame(*, verbose: bool = False, use_atm: bool = False) -> pd.DataFrame:
    from src.volatility_models.data_utils import TrainingDataHandler

    _, _, test_frame = TrainingDataHandler.load_splitted_data(
        verbose=verbose,
        use_atm=use_atm,
    )
    return _normalize_trade_frame(test_frame).reset_index(drop=True)


def _normalize_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [column_name(column) for column in normalized.columns]
    present = [column for column in RAW_TRADE_COLUMN_NAMES if column in normalized.columns]
    return normalized.loc[:, present].copy()


def build_feature_frame_from_trades(raw_frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_trade_frame(raw_frame)
    feature_frame = _build_features_vectorized(normalized)
    for feature_name in MODEL_INPUT_FEATURE_NAMES:
        if feature_name not in feature_frame.columns:
            feature_frame[feature_name] = 0.0
    return feature_frame.loc[:, MODEL_INPUT_FEATURE_NAMES].astype("float64")


def add_dashboard_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [column_name(column) for column in result.columns]
    underlying = result[column_name(VolatilityDBEnum.UNDERLYING_PRICE)].astype(float)
    strike = result[column_name(VolatilityDBEnum.STRIKE_PRICE)].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        moneyness = underlying / strike
        log_moneyness = np.log(moneyness)
    result["Moneyness"] = pd.Series(moneyness, index=result.index).replace(
        [np.inf, -np.inf], np.nan
    )
    result["LogMoneyness"] = pd.Series(log_moneyness, index=result.index).replace(
        [np.inf, -np.inf], np.nan
    )
    result["AbsLogMoneyness"] = result["LogMoneyness"].abs()

    exec_dt_col = column_name(VolatilityDBEnum.EXEC_DATETIME)
    if exec_dt_col in result.columns:
        exec_dt = pd.to_datetime(result[exec_dt_col], errors="coerce")
        result[exec_dt_col] = exec_dt
        result["ExecHour"] = exec_dt.dt.hour.astype("float64")
        result["ExecWeekday"] = exec_dt.dt.weekday.astype("float64")
    return result


def build_dashboard_dataset(
    raw_test_frame: pd.DataFrame,
    predictions: pd.Series | np.ndarray,
) -> pd.DataFrame:
    dataset = _normalize_trade_frame(raw_test_frame).reset_index(drop=True)
    feature_frame = build_feature_frame_from_trades(dataset)
    for feature_name in feature_frame.columns:
        dataset[feature_name] = feature_frame[feature_name].to_numpy()
    dataset = add_dashboard_derived_features(dataset)
    dataset["PredictedVolatility"] = np.asarray(predictions, dtype="float64").reshape(-1)
    if TARGET_COLUMN in dataset.columns:
        dataset["Residual"] = (
            dataset[TARGET_COLUMN].astype(float) - dataset["PredictedVolatility"]
        )
        dataset["AbsoluteError"] = dataset["Residual"].abs()
    return dataset


def apply_feature_override(
    raw_frame: pd.DataFrame,
    feature_name: str,
    value: float,
) -> pd.DataFrame:
    adjusted = _normalize_trade_frame(raw_frame)
    if feature_name == "Moneyness":
        underlying = adjusted[column_name(VolatilityDBEnum.UNDERLYING_PRICE)].astype(float)
        adjusted[column_name(VolatilityDBEnum.STRIKE_PRICE)] = underlying / float(value)
    elif feature_name == column_name(VolatilityDBEnum.TIME_TO_EXPIRATION):
        adjusted[column_name(VolatilityDBEnum.TIME_TO_EXPIRATION)] = float(value)
    elif feature_name == column_name(VolatilityDBEnum.UNDERLYING_LAG_MINUTES):
        adjusted[column_name(VolatilityDBEnum.UNDERLYING_LAG_MINUTES)] = float(value)
    elif feature_name == column_name(VolatilityDBEnum.QUANTITY):
        adjusted[column_name(VolatilityDBEnum.QUANTITY)] = float(value)
    elif feature_name == column_name(VolatilityDBEnum.RATE):
        adjusted[column_name(VolatilityDBEnum.RATE)] = float(value)
    return adjusted


def _build_features_from_trade(trade: pd.Series) -> dict[str, float]:
    tte_years = float(trade[column_name(VolatilityDBEnum.TIME_TO_EXPIRATION)]) / 365.0
    sqrt_tte_years = float(np.sqrt(tte_years))

    underlying_price = float(trade[column_name(VolatilityDBEnum.UNDERLYING_PRICE)])
    strike_price = float(trade[column_name(VolatilityDBEnum.STRIKE_PRICE)])
    log_moneyness = float(np.log(underlying_price / strike_price))
    log_moneyness_sq = log_moneyness**2
    log_moneyness_x_sqrt_tte = log_moneyness * sqrt_tte_years

    rate = float(trade[column_name(VolatilityDBEnum.RATE)])
    forward_price = underlying_price * float(np.exp(rate * tte_years))
    log_forward_moneyness = float(np.log(forward_price / strike_price))

    option_type = str(trade[column_name(VolatilityDBEnum.OPTION_TYPE)]).upper()
    trade_value = str(trade[column_name(VolatilityDBEnum.TRADE_TYPE)])
    exec_dt = pd.to_datetime(trade[column_name(VolatilityDBEnum.EXEC_DATETIME)])
    exec_hour = int(exec_dt.hour)
    exec_weekday = int(exec_dt.weekday())

    features: dict[str, float] = {
        "TTEYears": tte_years,
        "sqrtTTEYears": sqrt_tte_years,
        "logMoneyness": log_moneyness,
        "logMoneynessSq": log_moneyness_sq,
        "logMoneynessXSqrtTTE": log_moneyness_x_sqrt_tte,
        "logForwardMoneyness": log_forward_moneyness,
        "rate": rate,
        "underlyingLagMinutes": float(
            trade[column_name(VolatilityDBEnum.UNDERLYING_LAG_MINUTES)]
        ),
        "quantityLog1p": float(
            np.log1p(float(trade[column_name(VolatilityDBEnum.QUANTITY)]))
        ),
        "isCall": float(option_type == OptionTypeEnum.CALL.value),
        "isPut": float(option_type == OptionTypeEnum.PUT.value),
    }
    for trade_type, feature_column in TRADE_TYPE_TO_FEATURE.items():
        features[feature_column] = float(trade_value == trade_type)
    for hour in range(9, 20):
        features[f"execHour{hour}"] = float(exec_hour == hour)
    for weekday in range(5):
        features[f"execWeekday{weekday}"] = float(exec_weekday == weekday)
    return features


def _build_features_vectorized(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    tte_days = frame[column_name(VolatilityDBEnum.TIME_TO_EXPIRATION)].astype(float)
    tte_years = tte_days / 365.0
    sqrt_tte_years = np.sqrt(tte_years)
    underlying = frame[column_name(VolatilityDBEnum.UNDERLYING_PRICE)].astype(float)
    strike = frame[column_name(VolatilityDBEnum.STRIKE_PRICE)].astype(float)
    rate = frame[column_name(VolatilityDBEnum.RATE)].astype(float)
    quantity = frame[column_name(VolatilityDBEnum.QUANTITY)].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_moneyness = np.log(underlying / strike)
        forward_price = underlying * np.exp(rate * tte_years)
        log_forward_moneyness = np.log(forward_price / strike)

    result["TTEYears"] = tte_years
    result["sqrtTTEYears"] = sqrt_tte_years
    result["logMoneyness"] = log_moneyness
    result["logMoneynessSq"] = log_moneyness**2
    result["logMoneynessXSqrtTTE"] = log_moneyness * sqrt_tte_years
    result["logForwardMoneyness"] = log_forward_moneyness
    result["rate"] = rate
    result["underlyingLagMinutes"] = frame[
        column_name(VolatilityDBEnum.UNDERLYING_LAG_MINUTES)
    ].astype(float)
    result["quantityLog1p"] = np.log1p(quantity)

    option_type = frame[column_name(VolatilityDBEnum.OPTION_TYPE)].astype(str).str.upper()
    result["isCall"] = (option_type == OptionTypeEnum.CALL.value).astype(float)
    result["isPut"] = (option_type == OptionTypeEnum.PUT.value).astype(float)

    trade_type = frame[column_name(VolatilityDBEnum.TRADE_TYPE)].astype(str)
    for trade_value, feature_column in TRADE_TYPE_TO_FEATURE.items():
        result[feature_column] = (trade_type == trade_value).astype(float)

    exec_dt = pd.to_datetime(frame[column_name(VolatilityDBEnum.EXEC_DATETIME)])
    exec_hour = exec_dt.dt.hour
    exec_weekday = exec_dt.dt.weekday
    for hour in range(9, 20):
        result[f"execHour{hour}"] = (exec_hour == hour).astype(float)
    for weekday in range(5):
        result[f"execWeekday{weekday}"] = (exec_weekday == weekday).astype(float)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
