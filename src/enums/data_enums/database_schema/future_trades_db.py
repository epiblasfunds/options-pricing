from enum import StrEnum


class FuturesTradeIbexDBEnum(StrEnum):
    FUTURE_CONTRACT_CODE = "FutureContractCode"
    SESSION_DATE = "SessionDate"
    MARKET_CODE = "MarketCode"
    TRADE_EXEC_ID = "TradeExecID"
    EXEC_TIME = "ExecTime"
    EXEC_DATETIME = "ExecDatetime"
    TRADE_PRICE = "TradePrice"
    QUANTITY = "Quantity"
    TRADE_TYPE = "TradeType"
    STRIKE_PRICE = "StrikePrice"
    MATURITY_DATETIME = "MaturityDatetime"
    TIME_TO_EXPIRATION = "TimeToExpiration"
