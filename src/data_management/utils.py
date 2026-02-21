from datetime import date

from src.config.config import config
from src.exceptions.data_exceptions import ContractCcodeLengthError


def calculate_maturity_from_contract_code(session_date: date, contract_code: str) -> date:
    if len(contract_code) == config.contract_code_config.options_code_len:
        # Options
        ...
    elif len(contract_code) == config.contract_code_config.futures_code_len:
        # Futures
        month_code = contract_code[-2]
        year_code = contract_code[-1]
        
    else:
        raise ContractCcodeLengthError(
            f"The contract code {contract_code} has an unexpected len."
        )


def calculalte_strike_from_contract_code(contract_code: str) -> int:
    ...
