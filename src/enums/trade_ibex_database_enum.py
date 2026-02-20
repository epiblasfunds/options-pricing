from enum import Enum


class TradeIbexDatabaseEnum(Enum):
    SESSION_DATE = "SessionDate"
    CONTRACT_CODE = "ContractCode"
    MARKET_CODE = "MarketCode"
    TRADE_EXEC_ID = "TradeExecID"
    EXEC_TIME = "ExecTime"
    TRADE_PRICE = "TradePrice"
    QUANTITY = "Quantity"
    TRADE_TYPE = "TradeType"
    STRIKE_PRICE = "StrikePrice"
    MATURITY_DATE = "MaturityDate"
    CONTRACT_Type = "ContractType"
