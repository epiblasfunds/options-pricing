class DataError(Exception):
    def __init__(self, msg: str = "Data error defauilt msg"):
        super().__init__(msg)
