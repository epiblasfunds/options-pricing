from enum import Enum


class OptionsTradeIbexDatabaseEnum(Enum):
    OPTION_CONTRACT_CODE = "OptionContractCode"
    SESSION_DATE = "SessionDate"
    MARKET_CODE = "MarketCode"
    TRADE_EXEC_ID = "TradeExecID"
    EXEC_TIME = "ExecTime"
    TRADE_PRICE = "TradePrice"
    QUANTITY = "Quantity"
    TRADE_TYPE = "TradeType"
    STRIKE_PRICE = "StrikePrice"
    MATURITY_DATE = "MaturityDate"
