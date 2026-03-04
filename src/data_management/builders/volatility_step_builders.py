import logging
import math

import numpy as np
import pandas as pd
from py_vollib.black.implied_volatility import implied_volatility as iv_lib

from src.config.config import VOLATILITY_DATA_STEP_DIR_PATH, config
from src.enums.data_enums.options_trade_underlying_rates_ibex_database_enum import (
    OptionsTradeUnderlyingRatesIbexDatabaseEnum,
)

logger = logging.getLogger(__name__)

class OptionsTradeVolatilityIbexBuilder:
    OUTPUT_FILENAME = (
        VOLATILITY_DATA_STEP_DIR_PATH
        / f"{config.data_config.volatility_config.output_filename}.csv"
    )

    @staticmethod
    def get_output_filename():
        return OptionsTradeVolatilityIbexBuilder.OUTPUT_FILENAME

    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _no_arbitrage_bounds(
        F: float,
        K: float,
        T: float,
        r: float,
        c_type: str
    ) -> tuple[float, float]:
        # Calculate no-arbitrage bounds for option price based on Black-76 model
        discount = math.exp(-r * T)
        if c_type == "c":
            lower = max(discount * (F - K), 0.0)
            upper = discount * F
        else:
            lower = max(discount * (K - F), 0.0)
            upper = discount * K
        return lower, upper

    @staticmethod
    def black76_price(
        F: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        c_type: str,
    ) -> float:
        
        # Validate inputs
        if any([
            sigma <= 0,
            T <= 0,
            F <= 0,
            K <= 0
        ]):
            return 0.0
        
        # Calculate d1 and d2
        sqrt_t = math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        discount = math.exp(-r * T)

        # Calculate option price based on contract type
        if c_type == "c":
            return discount * (F * OptionsTradeVolatilityIbexBuilder.norm_cdf(d1) - K * OptionsTradeVolatilityIbexBuilder.norm_cdf(d2))
        else:
            return discount * (K * OptionsTradeVolatilityIbexBuilder.norm_cdf(-d2) - F * OptionsTradeVolatilityIbexBuilder.norm_cdf(-d1))    
    
    @staticmethod
    def _validate_bisection_bounds(f_low: float, f_high: float) -> bool:
        """Check if bisection bounds are valid."""
        return np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high < 0

    @staticmethod
    def _bisection_iteration(
        objective,
        low: float, 
        high: float, 
        f_low: float, 
        f_high: float,
        tol: float,
    ) -> tuple[float, float, float, float, bool]:
        """Perform one bisection iteration. Returns (low, high, f_low, f_high, converged)."""
        mid = 0.5 * (low + high)
        f_mid = objective(mid)

        if not np.isfinite(f_mid):
            return low, high, f_low, f_high, False
        
        if abs(f_mid) < tol or abs(high - low) < tol:
            return mid, mid, f_mid, f_mid, True
        
        if f_mid == 0:
            return mid, mid, f_mid, f_mid, True
        
        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
        
        return low, high, f_low, f_high, False

    @staticmethod
    def implied_vol_fallback(
        price: float,
        F: float,
        K: float,
        T: float,
        r: float,
        c_type: str,
    ) -> tuple[float, dict]:
        """
        Fallback bisection method. 
        Returns (implied_vol, metadata) where metadata contains convergence info.
        """
        metadata = {
            "method": "fallback_bisection",
            "converged": False,
            "iterations": 0,
            "no_arbitrage_violated": False,
        }
        
        # Define objective function for finding root
        def objective(sigma: float) -> float:
            return OptionsTradeVolatilityIbexBuilder.black76_price(F, K, T, r, sigma, c_type) - price

        # Validate no-arbitrage bounds
        lower_bound, upper_bound = OptionsTradeVolatilityIbexBuilder._no_arbitrage_bounds(F, K, T, r, c_type)
        if price < lower_bound - 1e-4 or price > upper_bound + 1e-4:
            metadata["no_arbitrage_violated"] = True
            return np.nan, metadata

        # Set bounds for implied volatility
        low = config.data_config.volatility_config.solver_min_sigma
        high = config.data_config.volatility_config.solver_max_sigma
        f_low = objective(low)
        f_high = objective(high)

        # Check if the function values at the bounds are valid
        if not OptionsTradeVolatilityIbexBuilder._validate_bisection_bounds(f_low, f_high):
            return np.nan, metadata
        
        tol = config.data_config.volatility_config.solver_tol
        max_iter = config.data_config.volatility_config.solver_max_iter
        
        # Bisection method to find implied volatility
        for iteration in range(max_iter):
            low, high, f_low, f_high, converged = OptionsTradeVolatilityIbexBuilder._bisection_iteration(
                objective, low, high, f_low, f_high, tol
            )
            metadata["iterations"] = iteration + 1
            
            if converged:
                metadata["converged"] = True
                return low, metadata

        return 0.5 * (low + high), metadata

    @staticmethod
    def implied_vol_engine(
        price: float,
        F: float,
        K: float,
        T: float,
        r: float,
        c_type: str,
    ) -> tuple[float, dict]:
        '''
        Calculates the implied volatility of an option using the Black-76 model.
        It first attempts to use the py_vollib library for calculation, and if the result is not valid,
        it falls back to a custom implementation based on the bisection method.
        '''
        metadata = {"method": None, "converged": False, "iterations": 0}

        # Validate inputs
        if  any(
            [
                pd.isna(price),
                pd.isna(F),
                pd.isna(K),
                pd.isna(T),
                pd.isna(r),
                price <= 0,
                F <= 0,
                K <= 0,
                T <= 0,
                c_type not in {"c", "p"},
            ]
        ):
            return np.nan, metadata
        
        # Calculate implied volatility using py_vollib
        try:
            iv = iv_lib(price, F, K, r, T, c_type)

            if(
                np.isfinite(iv)
                and iv >= config.data_config.volatility_config.solver_min_sigma
                and iv <= config.data_config.volatility_config.pyvollib_max_sigma
                ):
                metadata["method"] = "pyvollib"
                metadata["converged"] = True
                return iv, metadata
        except Exception:
            pass
        
        # Fall back to bisection
        iv, fallback_metadata = OptionsTradeVolatilityIbexBuilder.implied_vol_fallback(
            price, F, K, T, r, c_type
        )
        metadata.update(fallback_metadata)
        return iv, metadata
     
    @staticmethod
    def build(
        options_trade_underlying_rates_ibex_df: pd.DataFrame,
    ) -> pd.DataFrame:
        
        # Select columns
        price_col_name = OptionsTradeUnderlyingRatesIbexDatabaseEnum.TRADE_PRICE_OPTION.value
        underlying_col_name = OptionsTradeUnderlyingRatesIbexDatabaseEnum.UNDERLYING_PRICE.value
        strike_col_name = OptionsTradeUnderlyingRatesIbexDatabaseEnum.STRIKE_PRICE.value
        risk_free_col_name = OptionsTradeUnderlyingRatesIbexDatabaseEnum.RISK_FREE_RATE.value
        time_to_maturity_col_name = OptionsTradeUnderlyingRatesIbexDatabaseEnum.TIME_TO_MATURITY.value
        option_contract_code_col_name = OptionsTradeUnderlyingRatesIbexDatabaseEnum.OPTION_CONTRACT_CODE.value
        
        # Convert to numeric
        for col in [
            price_col_name,
            underlying_col_name,
            strike_col_name,
            risk_free_col_name,
            time_to_maturity_col_name,
        ]:
            options_trade_underlying_rates_ibex_df[col] = pd.to_numeric(
                options_trade_underlying_rates_ibex_df[col]
            )
        
        # Calculate time to maturity in years, risk-free rate in decimals, and contract type
        t_in_years = options_trade_underlying_rates_ibex_df[time_to_maturity_col_name] / 365.0
        r_in_decimals = options_trade_underlying_rates_ibex_df[risk_free_col_name] / 100.0
        contract_type = options_trade_underlying_rates_ibex_df[option_contract_code_col_name].astype(str).str[0].str.lower()

        # Calculate implied volatility
        iv_values: list[float] = []
        stats = {
            "total_rows": len(options_trade_underlying_rates_ibex_df),
            "pyvollib_success": 0,
            "fallback_converged": 0,
            "fallback_max_iter": 0,
            "no_arbitrage_violated": 0,
            "invalid_inputs": 0,
        }
        
        for row in zip(
            options_trade_underlying_rates_ibex_df[price_col_name].values,
            options_trade_underlying_rates_ibex_df[underlying_col_name].values,
            options_trade_underlying_rates_ibex_df[strike_col_name].values,
            t_in_years.values,
            r_in_decimals.values,
            contract_type.values,
        ):
            price, F, K, T, r, c_type = row
            iv, metadata = OptionsTradeVolatilityIbexBuilder.implied_vol_engine(
                price=price,
                F=F,
                K=K,
                T=T,
                r=r,
                c_type=c_type,
            )
            iv_values.append(iv)
            
            # Update statistics
            if metadata["method"] is None:
                stats["invalid_inputs"] += 1
            elif metadata["method"] == "pyvollib":
                stats["pyvollib_success"] += 1
            elif metadata["method"] == "fallback_bisection":
                if metadata.get("no_arbitrage_violated"):
                    stats["no_arbitrage_violated"] += 1
                elif metadata["converged"]:
                    stats["fallback_converged"] += 1
                else:
                    stats["fallback_max_iter"] += 1

        # Create output DataFrame
        options_trade_volatility_ibex_df = options_trade_underlying_rates_ibex_df.copy()
        options_trade_volatility_ibex_df[config.data_config.volatility_config.implied_volatility_column] = iv_values

        # Save CSV
        options_trade_volatility_ibex_df.to_csv(
            OptionsTradeVolatilityIbexBuilder.get_output_filename(),
            encoding="utf-8",
            sep=";",
        )
        logger.info(
            f"OptionsTradeVolatilityIbex (with shape {options_trade_volatility_ibex_df.shape}) saved in: {OptionsTradeVolatilityIbexBuilder.get_output_filename()}."
        )
        logger.info(
            f"IV Calculation Statistics: "
            f"Total={stats['total_rows']}, "
            f"PyVolLib={stats['pyvollib_success']}, "
            f"Fallback_Converged={stats['fallback_converged']}, "
            f"Fallback_MaxIter={stats['fallback_max_iter']}, "
            f"NoArbitrage_Violated={stats['no_arbitrage_violated']}, "
            f"Invalid_Inputs={stats['invalid_inputs']}"
        )

        return options_trade_volatility_ibex_df
        


