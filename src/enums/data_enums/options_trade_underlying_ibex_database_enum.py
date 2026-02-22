from enum import Enum


class OptionsTradeIbexUnderlyingDatabaseEnum(Enum):
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
    UNDERLYING_EXEC_TIME = "UnderelayingExecTime"
    UNDERLYING_PRICE = "UnderelayingPrice"
