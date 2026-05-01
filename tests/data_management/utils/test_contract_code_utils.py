from datetime import date

import pandas as pd
import pytest

from src.data_management.utils.contract_code_utils import calculate_maturity_from_contract_code
from src.data_management.utils.contract_code_utils import calculate_maturity_from_future_contract_code
from src.data_management.utils.contract_code_utils import calculate_maturity_from_option_contract_code
from src.data_management.utils.contract_code_utils import calculalte_strike_from_contract_code
from src.data_management.utils.contract_code_utils import compute_ibex_mask
from src.data_management.utils.contract_code_utils import compute_monthly_maturity_mask
from src.data_management.utils.contract_code_utils import get_contract_type
from src.data_management.utils.contract_code_utils import third_friday
from src.data_management.utils.contract_code_utils import validate_maturity_contract_code
from src.data_management.utils.contract_code_utils import validate_strike_contract_code
from src.enums.data_enums import ContractTypeEnum
from src.exceptions.data_exceptions import ContractCcodeLengthError
from src.exceptions.data_exceptions import ContractCodeMaturityMonthError
from src.exceptions.data_exceptions import ContractCodeMaturityYearError
from src.exceptions.data_exceptions import ContractCodeStrikeError


def test_contract_code_masks_detect_supported_ibex_contracts():
    codes = pd.Series(["CIBX 9000X26", "FIBXX6", "OTHER"])

    assert compute_ibex_mask(codes).tolist() == [True, True, False]
    assert compute_monthly_maturity_mask(codes).tolist() == [True, True, False]


def test_contract_code_date_and_strike_helpers_cover_options_and_futures():
    session_date = date(2026, 11, 18)

    assert third_friday(2026, 11) == 20
    assert calculate_maturity_from_option_contract_code("CIBX 9000X26") == date(
        2026, 11, 20
    )
    assert calculate_maturity_from_future_contract_code("FIBXX6", session_date) == date(
        2026, 11, 20
    )
    assert calculate_maturity_from_contract_code("CIBX 9000X26", session_date) == date(
        2026, 11, 20
    )
    assert calculalte_strike_from_contract_code("CIBX 9000X26") == 9000.0
    assert calculalte_strike_from_contract_code("FIBXX6") == 0.0
    assert get_contract_type("CIBX 9000X26") == ContractTypeEnum.OPTIONS
    assert get_contract_type("FIBXX6") == ContractTypeEnum.FUTURES
    assert pd.isna(get_contract_type("bad"))
    assert pd.isna(get_contract_type(pd.NA))


def test_validate_maturity_contract_code_accepts_consistent_series():
    validate_maturity_contract_code(
        contract_type=ContractTypeEnum.OPTIONS,
        contract_code_series=pd.Series(["CIBX 9000X26"]),
        maturity_series=pd.Series([date(2026, 11, 20)]),
        session_date_series=pd.Series([date(2026, 1, 2)]),
    )


def test_validate_maturity_contract_code_raises_for_month_mismatch():
    with pytest.raises(ContractCodeMaturityMonthError):
        validate_maturity_contract_code(
            contract_type=ContractTypeEnum.OPTIONS,
            contract_code_series=pd.Series(["CIBX 9000X26"]),
            maturity_series=pd.Series([date(2026, 10, 16)]),
            session_date_series=pd.Series([date(2026, 1, 2)]),
        )


def test_validate_maturity_contract_code_raises_for_year_mismatch():
    with pytest.raises(ContractCodeMaturityYearError):
        validate_maturity_contract_code(
            contract_type=ContractTypeEnum.OPTIONS,
            contract_code_series=pd.Series(["CIBX 9000X26"]),
            maturity_series=pd.Series([date(2027, 11, 19)]),
            session_date_series=pd.Series([date(2026, 1, 2)]),
        )


def test_validate_strike_contract_code_raises_for_mismatch():
    with pytest.raises(ContractCodeStrikeError):
        validate_strike_contract_code(
            contract_code_series=pd.Series(["CIBX 9000X26"]),
            strike_series=pd.Series([9050.0]),
        )


def test_contract_code_helpers_raise_for_unexpected_lengths():
    with pytest.raises(ContractCcodeLengthError):
        calculate_maturity_from_contract_code("BAD", date(2026, 1, 1))
    with pytest.raises(ContractCcodeLengthError):
        calculalte_strike_from_contract_code("BAD")
