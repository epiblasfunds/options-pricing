import typing as t
from dataclasses import dataclass

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
EXPLAINABILITY_FEATURE_NAMES = list(RAW_INPUT_FEATURE_NAMES)
MAIN_EXPLAINABILITY_FEATURE_NAMES = list(EXPLAINABILITY_FEATURE_NAMES)
CONTEXT_FEATURE_NAMES: list[str] = []
VISIBLE_RAW_INPUT_FEATURE_NAMES = list(EXPLAINABILITY_FEATURE_NAMES)
LEGACY_MAIN_EXPLAINABILITY_FEATURE_NAMES = [
    column_name(VolatilityDBEnum.OPTION_TYPE),
    column_name(VolatilityDBEnum.STRIKE_PRICE),
    column_name(VolatilityDBEnum.UNDERLYING_PRICE),
    column_name(VolatilityDBEnum.TIME_TO_EXPIRATION),
    column_name(VolatilityDBEnum.RATE),
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
ANALYSIS_FEATURE_NAMES = [
    column_name(VolatilityDBEnum.STRIKE_PRICE),
    column_name(VolatilityDBEnum.UNDERLYING_PRICE),
    column_name(VolatilityDBEnum.TIME_TO_EXPIRATION),
    column_name(VolatilityDBEnum.RATE),
]
_DATETIME_EXPLAINABILITY_FEATURE_NAMES: set[str] = set()
_NUMERIC_EXPLAINABILITY_FEATURE_NAMES = {
    column_name(VolatilityDBEnum.STRIKE_PRICE),
    column_name(VolatilityDBEnum.UNDERLYING_PRICE),
    column_name(VolatilityDBEnum.TIME_TO_EXPIRATION),
    column_name(VolatilityDBEnum.RATE),
}
_CATEGORICAL_EXPLAINABILITY_FEATURE_NAMES = tuple(
    feature_name
    for feature_name in EXPLAINABILITY_FEATURE_NAMES
    if feature_name not in _DATETIME_EXPLAINABILITY_FEATURE_NAMES
    and feature_name not in _NUMERIC_EXPLAINABILITY_FEATURE_NAMES
)
_MISSING_CATEGORY_TOKEN = "__MISSING__"


@dataclass(frozen=True)
class ExplainabilityEncoder:
    feature_names: tuple[str, ...]
    categorical_levels: dict[str, tuple[str, ...]]
    raw_defaults: dict[str, t.Any]

    def encode_frame(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        explain_frame = build_explainability_frame(
            raw_frame,
            feature_names=list(self.feature_names),
        )
        encoded = pd.DataFrame(index=explain_frame.index)
        for feature_name in self.feature_names:
            series = explain_frame[feature_name]
            if feature_name in _NUMERIC_EXPLAINABILITY_FEATURE_NAMES:
                encoded[feature_name] = pd.to_numeric(series, errors="coerce")
            elif feature_name in _DATETIME_EXPLAINABILITY_FEATURE_NAMES:
                encoded[feature_name] = _encode_datetime_series(series)
            else:
                encoded[feature_name] = _encode_categorical_series(
                    series,
                    self.categorical_levels.get(
                        feature_name, (_MISSING_CATEGORY_TOKEN,)
                    ),
                )
        return encoded.astype("float64")

    def decode_values(self, values: t.Any) -> pd.DataFrame:
        frame = (
            values.copy()
            if isinstance(values, pd.DataFrame)
            else pd.DataFrame(values, columns=list(self.feature_names))
        )
        decoded = pd.DataFrame(index=frame.index)
        for feature_name in self.feature_names:
            series = frame[feature_name]
            if feature_name in _NUMERIC_EXPLAINABILITY_FEATURE_NAMES:
                decoded[feature_name] = pd.to_numeric(series, errors="coerce")
            elif feature_name in _DATETIME_EXPLAINABILITY_FEATURE_NAMES:
                decoded[feature_name] = _decode_datetime_series(series)
            else:
                decoded[feature_name] = _decode_categorical_series(
                    series,
                    self.categorical_levels.get(
                        feature_name, (_MISSING_CATEGORY_TOKEN,)
                    ),
                )
        return decoded

    def reconstruct_raw_frame(self, values: t.Any) -> pd.DataFrame:
        decoded = self.decode_values(values)
        raw_frame = pd.DataFrame(index=decoded.index)
        for column_name in RAW_TRADE_COLUMN_NAMES:
            raw_frame[column_name] = self.raw_defaults.get(column_name, np.nan)
        for feature_name in self.feature_names:
            raw_frame[feature_name] = decoded[feature_name]
        return raw_frame


def load_test_trade_frame(*, verbose: bool = False) -> pd.DataFrame:
    from src.volatility_models.data_utils import TrainingDataHandler

    _, _, test_frame = TrainingDataHandler.load_splitted_data(
        verbose=verbose,
    )
    return _normalize_trade_frame(test_frame).reset_index(drop=True)


def _normalize_trade_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [column_name(column) for column in normalized.columns]
    present = [
        column for column in RAW_TRADE_COLUMN_NAMES if column in normalized.columns
    ]
    return normalized.loc[:, present].copy()


def build_explainability_frame(
    raw_frame: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    selected = list(feature_names or EXPLAINABILITY_FEATURE_NAMES)
    explain_frame = _normalize_trade_frame(raw_frame)
    for feature_name in selected:
        if feature_name not in explain_frame.columns:
            explain_frame[feature_name] = np.nan
    return explain_frame.loc[:, selected].copy()


def build_explainability_encoder(
    reference_frame: pd.DataFrame,
    feature_names: list[str] | None = None,
    defaults_override: dict[str, t.Any] | None = None,
) -> ExplainabilityEncoder:
    selected = tuple(feature_names or EXPLAINABILITY_FEATURE_NAMES)
    normalized_reference = _normalize_trade_frame(reference_frame)
    explain_frame = build_explainability_frame(normalized_reference, list(selected))
    categorical_levels: dict[str, tuple[str, ...]] = {}
    for feature_name in selected:
        if feature_name not in _CATEGORICAL_EXPLAINABILITY_FEATURE_NAMES:
            continue
        raw_values = (
            explain_frame[feature_name]
            .astype("string")
            .fillna(_MISSING_CATEGORY_TOKEN)
            .tolist()
        )
        levels = tuple(dict.fromkeys(str(value) for value in raw_values))
        categorical_levels[feature_name] = levels or (_MISSING_CATEGORY_TOKEN,)
    raw_defaults = _build_raw_defaults(
        normalized_reference,
        defaults_override=defaults_override,
    )
    return ExplainabilityEncoder(
        feature_names=selected,
        categorical_levels=categorical_levels,
        raw_defaults=raw_defaults,
    )


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
    dataset["PredictedVolatility"] = np.asarray(predictions, dtype="float64").reshape(
        -1
    )
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
        underlying = adjusted[column_name(VolatilityDBEnum.UNDERLYING_PRICE)].astype(
            float
        )
        adjusted[column_name(VolatilityDBEnum.STRIKE_PRICE)] = underlying / float(value)
    elif feature_name == column_name(VolatilityDBEnum.TIME_TO_EXPIRATION):
        adjusted[column_name(VolatilityDBEnum.TIME_TO_EXPIRATION)] = float(value)
    elif feature_name == column_name(VolatilityDBEnum.STRIKE_PRICE):
        adjusted[column_name(VolatilityDBEnum.STRIKE_PRICE)] = float(value)
    elif feature_name == column_name(VolatilityDBEnum.UNDERLYING_PRICE):
        adjusted[column_name(VolatilityDBEnum.UNDERLYING_PRICE)] = float(value)
    elif feature_name == column_name(VolatilityDBEnum.RATE):
        adjusted[column_name(VolatilityDBEnum.RATE)] = float(value)
    return adjusted


def _encode_datetime_series(series: pd.Series) -> pd.Series:
    datetimes = pd.to_datetime(series, errors="coerce", utc=True)
    encoded = (
        pd.Series(
            datetimes.astype("int64"),
            index=series.index,
            dtype="float64",
        )
        / 1_000_000_000.0
    )
    encoded.loc[datetimes.isna()] = np.nan
    return encoded


def _decode_datetime_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")


def _encode_categorical_series(
    series: pd.Series,
    levels: tuple[str, ...],
) -> pd.Series:
    values = series.astype("string").fillna(_MISSING_CATEGORY_TOKEN)
    mapping = {level: float(index) for index, level in enumerate(levels)}
    encoded = values.map(mapping).astype("float64")
    return encoded.fillna(mapping.get(_MISSING_CATEGORY_TOKEN, 0.0))


def _decode_categorical_series(
    series: pd.Series,
    levels: tuple[str, ...],
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    def _decode(value: float) -> str | float:
        if pd.isna(value):
            return np.nan
        index = int(round(float(value)))
        if index < 0 or index >= len(levels):
            return np.nan
        decoded = levels[index]
        return np.nan if decoded == _MISSING_CATEGORY_TOKEN else decoded

    return numeric.map(_decode)


def _build_raw_defaults(
    frame: pd.DataFrame,
    *,
    defaults_override: dict[str, t.Any] | None = None,
) -> dict[str, t.Any]:
    defaults: dict[str, t.Any] = {
        column_name(VolatilityDBEnum.OPTION_CONTRACT_CODE): np.nan,
        column_name(VolatilityDBEnum.IMPLIED_VOLATILITY): np.nan,
    }
    for feature_name in RAW_TRADE_COLUMN_NAMES:
        if feature_name in frame.columns:
            series = frame[feature_name].dropna()
            if not series.empty:
                if feature_name == column_name(VolatilityDBEnum.EXEC_DATETIME):
                    defaults[feature_name] = pd.to_datetime(
                        series.iloc[0],
                        errors="coerce",
                        utc=True,
                    )
                elif feature_name in {
                    column_name(VolatilityDBEnum.STRIKE_PRICE),
                    column_name(VolatilityDBEnum.UNDERLYING_PRICE),
                    column_name(VolatilityDBEnum.TIME_TO_EXPIRATION),
                    column_name(VolatilityDBEnum.RATE),
                    column_name(VolatilityDBEnum.IMPLIED_VOLATILITY),
                }:
                    defaults[feature_name] = float(
                        pd.to_numeric(series, errors="coerce").median()
                    )
                else:
                    defaults[feature_name] = series.mode().iloc[0]
    if defaults_override:
        defaults.update(defaults_override)
    return defaults


def _build_features_vectorized(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    tte_days = frame[column_name(VolatilityDBEnum.TIME_TO_EXPIRATION)].astype(float)
    tte_years = tte_days / 365.0
    sqrt_tte_years = np.sqrt(tte_years)
    underlying = frame[column_name(VolatilityDBEnum.UNDERLYING_PRICE)].astype(float)
    strike = frame[column_name(VolatilityDBEnum.STRIKE_PRICE)].astype(float)
    rate = frame[column_name(VolatilityDBEnum.RATE)].astype(float)

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

    option_type = (
        frame[column_name(VolatilityDBEnum.OPTION_TYPE)].astype(str).str.upper()
    )
    result["isCall"] = (option_type == OptionTypeEnum.CALL.value).astype(float)
    result["isPut"] = (option_type == OptionTypeEnum.PUT.value).astype(float)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
