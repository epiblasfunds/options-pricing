from enum import StrEnum


class TradeIbexDBEnum(StrEnum):
    SESSION_DATE = "SessionDate"
    CONTRACT_CODE = "ContractCode"
    MARKET_CODE = "MarketCode"
    TRADE_EXEC_ID = "TradeExecID"
    EXEC_TIME = "ExecTime"
    EXEC_DATETIME = "ExecDatetime"
    TRADE_PRICE = "TradePrice"
    QUANTITY = "Quantity"
    TRADE_TYPE = "TradeType"
    STRIKE_PRICE = "StrikePrice"
    MATURITY_DATETIME = "MaturityDatetime"
    CONTRACT_TYPE = "ContractType"
    TIME_TO_EXPIRATION = "TimeToExpiration"
