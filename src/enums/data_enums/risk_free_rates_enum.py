from enum import Enum


class RiskFreeRatesEnum(Enum):
    DATE = "Date"
    EONIA_RATE = "EoniaRate"
    STR_RATE = "STRRate"
    OVERNIGHT_RATE = "OvernightRate"
    EURIBOR_3M_RATE = "Euribor3MRate"
    EURIBOR_6M_RATE = "Euribor6MRate"
    EURIBOR_12M_RATE = "Euribor12MRate"
