import logging

from src.data_management.product_split_step.abstract_product_trade_ibex_loader import (
    AbstractProductTradeIbexLoader,
)
from src.enums.data_enums import ContractTypeEnum

logger = logging.getLogger(__name__)


class FuturesTradeIbexLoader(AbstractProductTradeIbexLoader):
    @classmethod
    def _get_contract_type(cls) -> ContractTypeEnum:
        return ContractTypeEnum.FUTURES
