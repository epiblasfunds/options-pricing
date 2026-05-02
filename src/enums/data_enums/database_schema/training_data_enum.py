from enum import StrEnum


class TrainingDataEnum(StrEnum):
    # Numeric features
    IMPLIED_VOLATILITY = "ImpliedVolatility"

    TTE_YEARS = "TTEYears"
    SQRT_TTE_YEARS = "sqrtTTEYears"
    LOG_MONEYNESS = "logMoneyness"
    LOG_MONEYNESS_SQ = "logMoneynessSq"
    LOG_MONEYNESS_X_SQRT_TTE = "logMoneynessXSqrtTTE"
    LOG_FORWARD_MONEYNESS = "logForwardMoneyness"
    RATE = "rate"

    # Categorical features
    IS_CALL = "isCall"
    IS_PUT = "isPut"
    EXEC_HOUR_9 = "execHour9"
    EXEC_HOUR_10 = "execHour10"
    EXEC_HOUR_11 = "execHour11"
    EXEC_HOUR_12 = "execHour12"
    EXEC_HOUR_13 = "execHour13"
    EXEC_HOUR_14 = "execHour14"
    EXEC_HOUR_15 = "execHour15"
    EXEC_HOUR_16 = "execHour16"
    EXEC_HOUR_17 = "execHour17"
    EXEC_HOUR_18 = "execHour18"
    EXEC_HOUR_19 = "execHour19"
    EXEC_WEEKDAY_0 = "execWeekday0"
    EXEC_WEEKDAY_1 = "execWeekday1"
    EXEC_WEEKDAY_2 = "execWeekday2"
    EXEC_WEEKDAY_3 = "execWeekday3"
    EXEC_WEEKDAY_4 = "execWeekday4"
