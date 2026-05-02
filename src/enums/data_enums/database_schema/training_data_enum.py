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
