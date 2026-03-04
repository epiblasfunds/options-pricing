class DataError(Exception):
    def __init__(self, msg: str = "Data error default msg"):
        super().__init__(msg)


class MissingValuesError(DataError):
    def __init__(
        self,
        msg: str = "Missing values (NAs) found.",
    ):
        super().__init__(msg)


class DuplicatedPrimaryKeysError(DataError):
    def __init__(
        self,
        msg: str = "Primary Keys are duplicated.",
    ):
        super().__init__(msg)


class ContractCcodeLengthError(DataError):
    def __init__(self, msg: str = "The contract code has an unexpected len."):
        super().__init__(msg)


class ContractCodeMaturityYearError(DataError):
    def __init__(
        self,
        msg: str = "ContractCode indicates a maturity year that differs from MaturityDate year.",
    ):
        super().__init__(msg)


class ContractCodeMaturityMonthError(DataError):
    def __init__(
        self,
        msg: str = "ContractCode indicates a maturity month that differs from MaturityDate month.",
    ):
        super().__init__(msg)


class ContractCodeStrikeError(DataError):
    def __init__(
        self,
        msg: str = "ContractCode indicates a strike that differs from StrikePrice.",
    ):
        super().__init__(msg)


class NegativeTradePriceError(DataError):
    def __init__(self, msg: str = "TradePrice contains non-positive (<= 0) values."):
        super().__init__(msg)


class NegativeQuantityError(DataError):
    def __init__(self, msg: str = "Quantity contains non-positive (<= 0) values."):
        super().__init__(msg)


class SessionAfterMaturityError(DataError):
    def __init__(self, msg: str = "SessionDate occurs after MaturityDate."):
        super().__init__(msg)


class UnderlyingExecDatetimeAfterExecDatetimeError(DataError):
    def __init__(self, msg: str = "UnderlyingExecDatetime occurs after ExecDatetime."):
        super().__init__(msg)


class UnderlyingExecDatetimeOutOfRangeError(DataError):
    def __init__(self, msg: str = "UnderlyingExecDatetime is outside the valid range."):
        super().__init__(msg)


class RatesOutOfRangeError(DataError):
    def __init__(self, msg: str = "Interest rates are outside the valid range."):
        super().__init__(msg)


class TimeToMaturityOutOfRangeError(DataError):
    def __init__(self, msg: str = "Time to maturity is outside the valid range."):
        super().__init__(msg)
