from enum import StrEnum


class VolatilityDBEnum(StrEnum):
    EXEC_DATETIME = "ExecDatetime"
    EXEC_TIME = "ExecTime"
    FUTURE_CONTRACT_CODE = "FutureContractCode"
    MARKET_CODE = "MarketCode"
    MATURITY_DATETIME = "MaturityDatetime"
    OPTION_CONTRACT_CODE = "OptionContractCode"
    OPTION_TYPE = "OptionType"
    QUANTITY = "Quantity"
    SESSION_DATE = "SessionDate"
    STRIKE_PRICE = "StrikePrice"
    TRADE_EXEC_ID = "TradeExecID"
    TRADE_PRICE_OPTION = "TradePriceOption"
    TRADE_TYPE = "TradeType"
    UNDERLYING_EXEC_DATETIME = "UnderlyingExecDatetime"
    UNDERLYING_LAG_MINUTES = "UnderlyingLagMinutes"
    UNDERLYING_PRICE = "UnderlyingPrice"
    TIME_TO_EXPIRATION = "TimeToExpiration"
    RATE = "Rate"
    IMPLIED_VOLATILITY = "ImpliedVolatility"
