from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.enums.data_enums import OptionTypeEnum
from src.enums.data_enums import VolatilityDBEnum


SELECTED_TRADE_COLUMNS = [
    VolatilityDBEnum.EXEC_DATETIME,
    VolatilityDBEnum.OPTION_CONTRACT_CODE,
    VolatilityDBEnum.OPTION_TYPE,
    VolatilityDBEnum.QUANTITY,
    VolatilityDBEnum.STRIKE_PRICE,
    VolatilityDBEnum.TRADE_TYPE,
    VolatilityDBEnum.UNDERLYING_LAG_MINUTES,
    VolatilityDBEnum.UNDERLYING_PRICE,
    VolatilityDBEnum.TIME_TO_EXPIRATION,
    VolatilityDBEnum.RATE,
    VolatilityDBEnum.IMPLIED_VOLATILITY,
]

RAW_INPUT_FEATURES = [
    VolatilityDBEnum.EXEC_DATETIME,
    VolatilityDBEnum.OPTION_TYPE,
    VolatilityDBEnum.QUANTITY,
    VolatilityDBEnum.STRIKE_PRICE,
    VolatilityDBEnum.TRADE_TYPE,
    VolatilityDBEnum.UNDERLYING_LAG_MINUTES,
    VolatilityDBEnum.UNDERLYING_PRICE,
    VolatilityDBEnum.TIME_TO_EXPIRATION,
    VolatilityDBEnum.RATE,
]

TRADE_TYPE_TO_FEATURE = {
    "3": "trade_type_3",
    "H": "trade_type_h",
    "M": "trade_type_m",
    "S": "trade_type_s",
    "W": "trade_type_w",
    "X": "trade_type_x",
}

TARGET_COLUMN = VolatilityDBEnum.IMPLIED_VOLATILITY
DASHBOARD_DERIVED_FEATURES = [
    "ExecHour",
    "ExecWeekday",
    "Moneyness",
    "LogMoneyness",
    "AbsLogMoneyness",
]
ANALYSIS_FEATURE_NAMES = [
    str(VolatilityDBEnum.TIME_TO_EXPIRATION),
    str(VolatilityDBEnum.RATE),
    str(VolatilityDBEnum.UNDERLYING_PRICE),
    str(VolatilityDBEnum.STRIKE_PRICE),
    str(VolatilityDBEnum.QUANTITY),
    str(VolatilityDBEnum.UNDERLYING_LAG_MINUTES),
    "ExecHour",
    "Moneyness",
    "LogMoneyness",
]
MODEL_FEATURE_NAMES = [
    "tte_years",
    "sqrt_tte_years",
    "log_moneyness",
    "log_moneyness_sq",
    "log_moneyness_x_sqrt_tte",
    "log_forward_moneyness",
    "rate",
    "is_call",
    "exec_hour",
    "exec_weekday",
    "underlying_lag_minutes",
    "quantity_log1p",
    *TRADE_TYPE_TO_FEATURE.values(),
]


def build_features_from_trade(tr) -> dict:
    """
    Construye el diccionario de features por trade.
    """

    tte_years = tr[VolatilityDBEnum.TIME_TO_EXPIRATION] / 365.0
    sqrt_tte_years = np.sqrt(tte_years)

    underlying_price = tr[VolatilityDBEnum.UNDERLYING_PRICE]
    strike_price = tr[VolatilityDBEnum.STRIKE_PRICE]
    log_moneyness = np.log(underlying_price / strike_price)
    log_moneyness_sq = log_moneyness ** 2
    log_moneyness_x_sqrt_tte = log_moneyness * sqrt_tte_years

    rate = tr[VolatilityDBEnum.RATE]
    forward_price = underlying_price * np.exp(rate * tte_years)
    log_forward_moneyness = np.log(forward_price / strike_price)

    is_call = float(str(tr[VolatilityDBEnum.OPTION_TYPE]).upper() == OptionTypeEnum.CALL)

    exec_dt = tr[VolatilityDBEnum.EXEC_DATETIME]
    exec_hour = float(exec_dt.hour)
    exec_weekday = float(exec_dt.weekday())

    quantity_raw = tr[VolatilityDBEnum.QUANTITY]
    quantity_log1p = np.log1p(quantity_raw)

    underlying_lag_minutes = tr[VolatilityDBEnum.UNDERLYING_LAG_MINUTES]

    features = {
        "tte_years": tte_years,
        "sqrt_tte_years": sqrt_tte_years,
        "log_moneyness": log_moneyness,
        "log_moneyness_sq": log_moneyness_sq,
        "log_moneyness_x_sqrt_tte": log_moneyness_x_sqrt_tte,
        "log_forward_moneyness": log_forward_moneyness,
        "rate": rate,
        "is_call": is_call,
        "exec_hour": exec_hour,
        "exec_weekday": exec_weekday,
        "underlying_lag_minutes": underlying_lag_minutes,
        "quantity_log1p": quantity_log1p,
    }

    trade_value = str(tr[VolatilityDBEnum.TRADE_TYPE])
    for tid, col_name in TRADE_TYPE_TO_FEATURE.items():
        features[col_name] = float(trade_value == tid)

    return features


def select_trade_columns(frame: pd.DataFrame) -> pd.DataFrame:
    available_columns = [str(column) for column in SELECTED_TRADE_COLUMNS if str(column) in frame.columns]
    return frame.loc[:, available_columns].copy()


def add_dashboard_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    derived = frame.copy()

    if VolatilityDBEnum.EXEC_DATETIME in derived.columns:
        exec_dt = pd.to_datetime(
            derived[VolatilityDBEnum.EXEC_DATETIME],
            format="mixed",
            errors="coerce",
        )
        derived["ExecHour"] = exec_dt.dt.hour.astype("float64")
        derived["ExecWeekday"] = exec_dt.dt.weekday.astype("float64")

    if {
        VolatilityDBEnum.UNDERLYING_PRICE,
        VolatilityDBEnum.STRIKE_PRICE,
    }.issubset(derived.columns):
        ratio = (
            pd.to_numeric(derived[VolatilityDBEnum.UNDERLYING_PRICE], errors="coerce")
            / pd.to_numeric(derived[VolatilityDBEnum.STRIKE_PRICE], errors="coerce")
        )
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        safe_ratio = ratio.where(ratio > 0.0)
        derived["Moneyness"] = ratio
        derived["LogMoneyness"] = np.log(safe_ratio)
        derived["AbsLogMoneyness"] = np.abs(np.log(safe_ratio))

    return derived


def build_feature_frame_from_trades(frame: pd.DataFrame) -> pd.DataFrame:
    trades = select_trade_columns(frame)
    if trades.empty:
        return pd.DataFrame(index=frame.index, columns=MODEL_FEATURE_NAMES, dtype="float64")

    normalized = trades.copy()
    normalized[VolatilityDBEnum.EXEC_DATETIME] = pd.to_datetime(
        normalized[VolatilityDBEnum.EXEC_DATETIME],
        format="mixed",
        errors="coerce",
    )

    rows = [build_features_from_trade(row) for _, row in normalized.iterrows()]
    features = pd.DataFrame(rows, index=normalized.index)
    for feature_name in MODEL_FEATURE_NAMES:
        if feature_name not in features.columns:
            features[feature_name] = 0.0
    return features.loc[:, MODEL_FEATURE_NAMES]


def build_model_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    raw_frame = select_trade_columns(frame)
    dashboard_frame = add_dashboard_derived_features(raw_frame)
    model_features = build_feature_frame_from_trades(raw_frame)
    return pd.concat([dashboard_frame, model_features], axis=1)


def apply_feature_override(frame: pd.DataFrame, feature_name: str, value: Any) -> pd.DataFrame:
    updated = frame.copy()

    if feature_name == "ExecHour" and VolatilityDBEnum.EXEC_DATETIME in updated.columns:
        exec_dt = pd.to_datetime(updated[VolatilityDBEnum.EXEC_DATETIME], format="mixed", errors="coerce")
        updated[VolatilityDBEnum.EXEC_DATETIME] = (
            exec_dt.dt.floor("D") + pd.to_timedelta(float(value), unit="h")
        )
        return add_dashboard_derived_features(updated)

    if feature_name == "ExecWeekday" and VolatilityDBEnum.EXEC_DATETIME in updated.columns:
        exec_dt = pd.to_datetime(updated[VolatilityDBEnum.EXEC_DATETIME], format="mixed", errors="coerce")
        updated[VolatilityDBEnum.EXEC_DATETIME] = (
            exec_dt + pd.to_timedelta(float(value) - exec_dt.dt.weekday, unit="D")
        )
        return add_dashboard_derived_features(updated)

    if feature_name in updated.columns:
        updated[feature_name] = value
        return add_dashboard_derived_features(updated)

    if feature_name == "Moneyness":
        updated[VolatilityDBEnum.STRIKE_PRICE] = (
            pd.to_numeric(updated[VolatilityDBEnum.UNDERLYING_PRICE], errors="coerce")
            / float(value)
        )
        return add_dashboard_derived_features(updated)

    if feature_name == "LogMoneyness":
        moneyness = float(np.exp(float(value)))
        updated[VolatilityDBEnum.STRIKE_PRICE] = (
            pd.to_numeric(updated[VolatilityDBEnum.UNDERLYING_PRICE], errors="coerce")
            / moneyness
        )
        return add_dashboard_derived_features(updated)

    raise KeyError(f"Unsupported override feature: {feature_name}")
