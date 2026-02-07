

CCONTRACTS_C2_COLUMNS_LIST = [
    "SessionDate",                     # 1 Fecha de sesión
    "ClearingHouseCode",               # 2 Código de cámara
    "ContractCode",                    # 3 Código de contrato
    "ContractGroupCode",               # 4 Grupo del contrato
    "ContractTypeCode",                # 5 Tipo del contrato
    "StrikePrice",                     # 6 Precio de ejercicio
    "MaturityDate",                    # 7 Fecha de vencimiento
    "TradingEndDate",                  # 8 Fecha de fin de negociación
    "ExerciseUnderlyingContractCode",  # 9 Código contrato subyacente (ejercicio)
    "MarginUnderlyingContractCode",    # 10 Código contrato subyacente (garantías)
    "ArrayCode",                       # 11 Código de matriz de garantías
    "ExpiryNumber",                    # 12 Nº vencimiento de liquidación
    "OffsetNumber",                    # 13 Nº compensación
    "ExpirySpan",                      # 14 Tipo de vencimiento (S/L)
    "MaturityMonthYear",               # 15 Identificador del vencimiento
    "ISINCode"                         # 16 Código ISIN
]

CCONTRACTS_C2_COLUMNS_SELECTED_DICT = {
    "SessionDate": DataType.DATE,       # 1 Fecha de sesión
    "ContractCode": DataType.TEXT,      # 3 Código de contrato
    "StrikePrice": DataType.FLOAT,      # 6 Precio de ejercicio
    "MaturityDate": DataType.DATE,      # 7 Fecha de vencimiento
}

# CCONTRACTS_C2
df = build_data_raw(
    columns_list=CCONTRACTS_C2_COLUMNS_LIST,
    selected_columns_dict=CCONTRACTS_C2_COLUMNS_SELECTED_DICT,
    file_prefix="CCONTRACTS_C2",
    )

TGENTRADES_COLUMNS_LIST = [
    "SessionDate",    # 1 Fecha de sesión
    "MarketCode",     # 2 Código de mercado
    "TradeExecID",    # 3 Número de registro de negociación
    "ContractCode",   # 4 Código de contrato
    "ExecTime",       # 5 Hora de ejecución
    "TradePrice",     # 6 Precio
    "Quantity",       # 7 Volumen
    "TradeType"       # 8 Tipo de operación
]

TGENTRADES_COLUMNS_SELECTED_DICT = {
    "SessionDate": DataType.DATE,       # 1 Fecha de sesión
    "MarketCode": DataType.TEXT,        # 2 Código de mercado
    "TradeExecID": DataType.TEXT,       # 3 Número de registro de negociación
    "ContractCode": DataType.TEXT,      # 4 Código de contrato
    "ExecTime": DataType.DATETIME,      # 5 Hora de ejecución
    "TradePrice": DataType.FLOAT,       # 6 Precio
    "Quantity": DataType.INT,           # 7 Volumen
    "TradeType": DataType.TEXT          # 8 Tipo de operación
}

# TGENTRADES
df = build_data_raw(
    columns_list=TGENTRADES_COLUMNS_LIST,
    selected_columns_dict=TGENTRADES_COLUMNS_SELECTED_DICT,
    file_prefix="TGENTRADES",
    custom_processing_func=tgentrades_custom_process,
    )