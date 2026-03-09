from enum import StrEnum


class OptionUnderlyingDBEnum(StrEnum):
    OPTION_CONTRACT_CODE = "OptionContractCode"
    FUTURE_CONTRACT_CODE = "FutureContractCode"
    MATURITY_DATETIME = "MaturityDatetime"
