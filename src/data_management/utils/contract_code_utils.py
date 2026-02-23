from datetime import date

import pandas as pd

from src.config.config import config
from src.enums.data_enums import ContractTypeEnum
from src.exceptions.data_exceptions import (
    ContractCcodeLengthError,
    ContractCodeMaturityMonthError,
    ContractCodeMaturityYearError,
    ContractCodeStrikeError,
)


def compute_ibex_mask(contract_code_series: pd.Series) -> pd.Series:
    return contract_code_series.str.startswith(
            tuple(config.data_config.contract_code_config.contracts_prefixes),
            na=False,
        )


def compute_monthly_maturity_mask(contract_code_series: pd.Series) -> pd.Series:
    # Select only futures and options with monthly maturity
    # We can identify them because their number of characters
    return contract_code_series.str.len().isin(
            [
                config.data_config.contract_code_config.futures_code_len,
                config.data_config.contract_code_config.options_code_len,
            ]
        )


def validate_maturity_contract_code_month(
    contract_type: ContractTypeEnum,
    month_code_idx: int,
    cc_series: pd.Series,
    m_series: pd.Series,
):
    invalid_months = (
        cc_series.str[month_code_idx].map(
            config.data_config.contract_code_config.futures_code_month
        )
        != pd.to_datetime(m_series).dt.month
    )
    if invalid_months.any():
        first_invalid = cc_series[invalid_months].iloc[0]
        raise ContractCodeMaturityMonthError(
            f"There are {contract_type.value} contract codes with maturity month "
            "that does not match the maturity date month. "
            f"First invalid option code: {first_invalid}."
        )


def validate_maturity_contract_code_year(
    contract_type: ContractTypeEnum,
    year_code_idx: int,
    cc_series: pd.Series,
    m_series: pd.Series,
    sd_series: pd.Series,
):
    invalid_years = (
        cc_series.str[year_code_idx:].astype("Int64")
        != pd.to_datetime(m_series).dt.year % (10 ** abs(year_code_idx))
    ) & (
        (contract_type == ContractTypeEnum.OPTIONS)
        | (
            # As Futures ContractCode have just one number for the year, it indicates
            # the next year ending in this number, son can't be further than 10 years
            (contract_type == ContractTypeEnum.FUTURES)
            & (
                pd.to_datetime(m_series).dt.year - pd.to_datetime(sd_series).dt.year
                <= 10
            )
        )
    )
    if invalid_years.any():
        first_invalid = cc_series[invalid_years].iloc[0]
        raise ContractCodeMaturityYearError(
            f"There are {contract_type.value} contract codes with maturity year "
            "that does not match the maturity date year. "
            f"First invalid option code: {first_invalid}."
        )


def validate_maturity_contract_code(
    contract_type: ContractTypeEnum,
    contract_code_series: pd.Series,
    maturity_series: pd.Series,
    session_date_series: pd.Series,
):
    if contract_type == ContractTypeEnum.OPTIONS:
        type_len = config.data_config.contract_code_config.options_code_len
        month_code_idx = config.data_config.contract_code_config.month_options_code_idx
        year_code_idx = config.data_config.contract_code_config.year_options_code_idx
    else:
        type_len = config.data_config.contract_code_config.futures_code_len
        month_code_idx = config.data_config.contract_code_config.month_futures_code_idx
        year_code_idx = config.data_config.contract_code_config.year_futures_code_idx

    cc_series = contract_code_series[contract_code_series.str.len() == type_len]
    m_series = maturity_series[contract_code_series.str.len() == type_len]
    sd_series = session_date_series[contract_code_series.str.len() == type_len]

    validate_maturity_contract_code_month(
        contract_type=contract_type,
        month_code_idx=month_code_idx,
        cc_series=cc_series,
        m_series=m_series,
    )
    validate_maturity_contract_code_year(
        contract_type=contract_type,
        year_code_idx=year_code_idx,
        cc_series=cc_series,
        m_series=m_series,
        sd_series=sd_series,
    )


def validate_strike_contract_code(
    contract_code_series: pd.Series,
    strike_series: pd.Series,
):
    # Filter by options
    type_len = config.data_config.contract_code_config.options_code_len

    cc_series = contract_code_series[contract_code_series.str.len() == type_len]
    s_series = strike_series[contract_code_series.str.len() == type_len]

    # Validate
    strike_starts = config.data_config.contract_code_config.strike_starts
    strike_ends = config.data_config.contract_code_config.strike_ends
    invalid_strikes = (
        cc_series.str[strike_starts:strike_ends].astype("float") != s_series.astype("float")
    )
    if invalid_strikes.any():
        first_invalid_cc = cc_series[invalid_strikes].iloc[0]
        first_invalid_cc_converted = cc_series.str[strike_starts:strike_ends].astype("float").iloc[0]
        first_invalid_s = s_series[invalid_strikes].iloc[0]
        raise ContractCodeStrikeError(
            f"There are option contract codes with strike "
            "that does not match the strike column. "
            f"First invalid option code: {first_invalid_cc} ({first_invalid_cc_converted})"
            f" with strike {first_invalid_s}."
        )


def third_friday(year: int, month: int) -> int:
    """
    Returns the day of the month corresponding to the third Friday.
    """
    first_day = date(year, month, 1)
    first_friday_offset = (4 - first_day.weekday()) % 7
    first_friday = 1 + first_friday_offset
    third_friday = first_friday + 14
    return third_friday


def calculate_maturity_from_option_contract_code(
    contract_code: str,
) -> date:
    year_code_idx = config.data_config.contract_code_config.year_options_code_idx
    month_code_idx = config.data_config.contract_code_config.month_options_code_idx

    year_code = contract_code[year_code_idx:]
    year = int(f"20{year_code}")

    month_code = contract_code[month_code_idx]
    month = config.data_config.contract_code_config.futures_code_month[month_code]

    day = third_friday(year, month)

    return date(year, month, day)


def calculate_maturity_from_future_contract_code(
    contract_code: str,
    session_date: date,
) -> date:
    month_code_idx = config.data_config.contract_code_config.month_futures_code_idx
    year_code_idx = config.data_config.contract_code_config.year_futures_code_idx

    month_code = contract_code[month_code_idx]
    month = config.data_config.contract_code_config.futures_code_month[month_code]

    year_code = int(contract_code[year_code_idx:])
    delta = (year_code - (session_date.year % 10)) % 10
    year = session_date.year + delta

    day = third_friday(year, month)

    maturity_date_candidate = date(year, month, day)
    if maturity_date_candidate < session_date:
        year += 10
        day = third_friday(year, month)

    return date(year, month, day)


def calculate_maturity_from_contract_code(
    contract_code: str,
    session_date: date,
) -> date:
    if len(contract_code) == config.data_config.contract_code_config.options_code_len:
        # Options
        maturity_date = calculate_maturity_from_option_contract_code(
            contract_code=contract_code,
        )
    elif len(contract_code) == config.data_config.contract_code_config.futures_code_len:
        # Futures
        maturity_date = calculate_maturity_from_future_contract_code(
            contract_code=contract_code, session_date=session_date
        )
    else:
        raise ContractCcodeLengthError(
            f"The contract code {contract_code} has an unexpected len."
        )

    return maturity_date


def calculalte_strike_from_contract_code(contract_code: str) -> float:
    if len(contract_code) == config.data_config.contract_code_config.options_code_len:
        # Options
        strike_starts = config.data_config.contract_code_config.strike_starts
        strike_ends = config.data_config.contract_code_config.strike_ends
        strike_str = contract_code[strike_starts:strike_ends]
        return float(strike_str)
    elif len(contract_code) == config.data_config.contract_code_config.futures_code_len:
        # Futures don't have strike
        return 0.0
    else:
        raise ContractCcodeLengthError(
            f"The contract code {contract_code} has an unexpected len."
        )


def get_contract_type(code: str) -> str:
    if pd.isna(code):
        return pd.NA
    elif len(code) == config.data_config.contract_code_config.options_code_len:
        return ContractTypeEnum.OPTIONS.value
    elif len(code) == config.data_config.contract_code_config.futures_code_len:
        return ContractTypeEnum.FUTURES.value
    else:
        return pd.NA
