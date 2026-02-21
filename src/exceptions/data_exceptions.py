class DataError(Exception):
    def __init__(self, msg: str = "Data error default msg"):
        super().__init__(msg)


class ContractCcodeLengthError(DataError):
    def __init__(self, msg: str = "The contract code has an unexpected len."):
        super().__init__(msg)
