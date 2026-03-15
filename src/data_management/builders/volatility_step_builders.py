import logging
import math

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.config import VOLATILITY_DATA_STEP_DIR_PATH, config
from src.enums.data_enums import (
    OptionTradesUnderlyingDBEnum,
    RatesEnum,
    VolatilityDBEnum,
)

logger = logging.getLogger(__name__)


class VolatilityBuilder:
    OUTPUT_FILENAME = (
        VOLATILITY_DATA_STEP_DIR_PATH
        / f"{config.data_config.volatility_config.output_filename}.csv"
    )

    @staticmethod
    def get_output_filename():
        return VolatilityBuilder.OUTPUT_FILENAME

    @staticmethod
    def calculate_compounded_rate(
        exec_date: pd.Timestamp,
        maturity_date: pd.Timestamp,
        time_to_expiration: float,
        rates_df: pd.DataFrame,
    ) -> float:
        """
        Calculate the compounded ESTR average rate based on the formula:

        Compound interest = [∏(1 + r_i × n_i / N) - 1] × N / d_c

        where:
        - d_b = number of TARGET2 business days in the interest period
        - d_c = number of calendar days in the interest period (time_to_expiration)
        - r_i = ESTR rate published on business day i
        - n_i = number of calendar days for which rate r_i applies (usually 1,
                except on each Monday within the interest period when it will be 3
                to account for the weekend)
        - N = number of days in the year (360 for European money market)
        """
        N = 360  # European money market convention
        d_c = time_to_expiration

        if d_c <= 0:
            return 0.0

        # Get business days between exec_date and maturity_date
        business_days = pd.bdate_range(
            start=exec_date.date(), end=maturity_date.date(), freq="B"
        )

        if len(business_days) == 0:
            return 0.0

        # Calculate the compounded product
        compound_product = 1.0

        for business_day in business_days:
            date_rate = business_day.date()

            mask = (rates_df[RatesEnum.SESSION_DATE] == date_rate)
            while not mask.any():
                date_rate -= pd.Timedelta(days=1)
                mask = (rates_df[RatesEnum.SESSION_DATE] == date_rate)

            r_i = float(rates_df.loc[mask, RatesEnum.RATE].iloc[0])

            # Calculate n_i: number of calendar days this rate applies to
            # If it's Monday (weekday==0), it covers 3 days (Sat, Sun, Mon)
            # Otherwise, it's 1 day
            if business_day.weekday() == 0:  # Monday
                n_i = 3
            else:
                n_i = 1

            # Apply the formula: multiply by (1 + r_i × n_i / N)
            compound_product *= 1 + (r_i * n_i / N)

        # Final calculation: [product - 1] × N / d_c
        compounded_rate = (compound_product - 1) * (N / d_c)

        return compounded_rate

    @staticmethod
    def create_rate_column(df: pd.DataFrame, rates_df: pd.DataFrame) -> pd.DataFrame:
        rate_values: list[float] = []
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Calculating compounded rate",
            unit="row",
        ):
            compounded_rate = VolatilityBuilder.calculate_compounded_rate(
                exec_date=pd.to_datetime(
                    row[OptionTradesUnderlyingDBEnum.EXEC_DATETIME]
                ),
                maturity_date=pd.to_datetime(
                    row[OptionTradesUnderlyingDBEnum.MATURITY_DATETIME]
                ),
                time_to_expiration=float(
                    row[OptionTradesUnderlyingDBEnum.TIME_TO_EXPIRATION]
                ),
                rates_df=rates_df,
            )
            rate_values.append(compounded_rate)

        df[VolatilityDBEnum.RATE] = rate_values

        return df

    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

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
        if any([sigma <= 0, T <= 0, F <= 0, K <= 0]):
            return 0.0

        # Calculate d1 and d2
        sqrt_t = math.sqrt(T)
        d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        discount = math.exp(-r * T)

        # Calculate option price based on contract type
        if c_type == "c":
            return discount * (
                F * VolatilityBuilder.norm_cdf(d1) - K * VolatilityBuilder.norm_cdf(d2)
            )
        else:
            return discount * (
                K * VolatilityBuilder.norm_cdf(-d2)
                - F * VolatilityBuilder.norm_cdf(-d1)
            )

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

        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

        return low, high, f_low, f_high, False

    @staticmethod
    def implied_vol(
        price: float,
        F: float,
        K: float,
        T: float,
        r: float,
        c_type: str,
    ) -> float:
        """
        Bisection method.
        Returns implied volatility.
        """
        # Define objective function for finding root
        def objective(sigma: float) -> float:
            return VolatilityBuilder.black76_price(F, K, T, r, sigma, c_type) - price

        # Set bounds for implied volatility
        low = config.data_config.volatility_config.solver_min_sigma
        high = config.data_config.volatility_config.solver_max_sigma
        f_low = objective(low)
        f_high = objective(high)

        # Check if the function values at the bounds are valid
        if not VolatilityBuilder._validate_bisection_bounds(f_low, f_high):
            return np.nan

        tol = config.data_config.volatility_config.solver_tol

        # Bisection method to find implied volatility
        converged = False
        while not converged:
            low, high, f_low, f_high, converged = (
                VolatilityBuilder._bisection_iteration(
                    objective, low, high, f_low, f_high, tol
                )
            )
        return low

    @staticmethod
    def build(
        options_trades_underlying_df: pd.DataFrame,
        rates_df: pd.DataFrame,
    ) -> pd.DataFrame:
        # Calculate compound rate
        volatility_df = VolatilityBuilder.create_rate_column(
            df=options_trades_underlying_df, rates_df=rates_df
        )

        # Add auxiliar columns for calculating the volatility
        t_in_years_col = "t_in_years"
        r_in_decimals_col = "r_in_decimals"
        option_type_col = "option_type"

        volatility_df[t_in_years_col] = (
            volatility_df[OptionTradesUnderlyingDBEnum.TIME_TO_EXPIRATION] / 365.0
        )
        volatility_df[r_in_decimals_col] = volatility_df[VolatilityDBEnum.RATE] / 100.0
        volatility_df[option_type_col] = (
            volatility_df[OptionTradesUnderlyingDBEnum.OPTION_CONTRACT_CODE]
            .astype(str)
            .str[0]
            .str.lower()
        )

        # Calculate implied volatility
        iv_values: list[float] = []

        subset_cols = [
            OptionTradesUnderlyingDBEnum.TRADE_PRICE_OPTION,
            OptionTradesUnderlyingDBEnum.UNDERLYING_PRICE,
            OptionTradesUnderlyingDBEnum.STRIKE_PRICE,
            t_in_years_col,
            r_in_decimals_col,
            option_type_col,
        ]
        for _, row in tqdm(
            volatility_df[subset_cols].iterrows(),
            total=len(volatility_df),
            desc="Calculating implied volatility",
            unit="row",
        ):
            iv = VolatilityBuilder.implied_vol(
                price=row[OptionTradesUnderlyingDBEnum.TRADE_PRICE_OPTION],
                F=row[OptionTradesUnderlyingDBEnum.UNDERLYING_PRICE],
                K=row[OptionTradesUnderlyingDBEnum.STRIKE_PRICE],
                T=row[t_in_years_col],
                r=row[r_in_decimals_col],
                c_type=row[option_type_col],
            )
            iv_values.append(iv)

        # Create output DataFrame
        volatility_df[VolatilityDBEnum.IMPLIED_VOLATILITY] = iv_values

        # Save CSV
        volatility_df.to_csv(
            VolatilityBuilder.get_output_filename(),
            encoding="utf-8",
            sep=";",
            index=False
        )
        logger.info(
            f"OptionsTradeVolatilityIbex (with shape {volatility_df.shape}) "
            + f"saved in: {VolatilityBuilder.get_output_filename()}."
        )

        return volatility_df
