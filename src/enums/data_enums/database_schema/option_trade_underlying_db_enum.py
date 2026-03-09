from enum import StrEnum


class OptionTradesUnderlyingDBEnum(StrEnum):
    EXEC_DATETIME = "ExecDateTime"
    EXEC_TIME = "ExecTime"
    FUTURE_CONTRACT_CODE = "FutureContractCode"
    MARKET_CODE = "MarketCode"
    MATURITY_DATE = "MaturityDate"
    OPTION_CONTRACT_CODE = "OptionContractCode"
    QUANTITY = "Quantity"
    SESSION_DATE = "SessionDate"
    STRIKE_PRICE = "StrikePrice"
    TRADE_EXEC_ID = "TradeExecID"
    TRADE_PRICE_OPTION = "TradePriceOption"
    TRADE_TYPE = "TradeType"
    UNDERLYING_EXEC_DATETIME = "UnderlyingExecDateTime"
    UNDERLYING_PRICE = "UnderlyingPrice"
    TIME_TO_EXPIRATION = "TimeToExpiration"