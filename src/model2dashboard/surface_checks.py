"""Shared financial heuristics for precomputed volatility surfaces."""

import pandas as pd


def financial_checks_from_surface(surface_frame: pd.DataFrame) -> list[str]:
    if surface_frame.empty:
        return ["No local surface could be generated for the financial checks."]
    pivot = surface_frame.pivot_table(
        index="TimeToExpiration",
        columns="Moneyness",
        values="PredictedVolatility",
    ).sort_index()
    smile_diff = pivot.diff(axis=1).abs().max().max()
    term_diff = pivot.diff(axis=0).abs().max().max()
    warnings: list[str] = []
    if pd.notna(smile_diff) and smile_diff > 0.20:
        warnings.append(
            "Heuristic warning: adjacent smile points show large volatility jumps."
        )
    if pd.notna(term_diff) and term_diff > 0.20:
        warnings.append(
            "Heuristic warning: adjacent maturity points show large term-structure jumps."
        )
    if not warnings:
        warnings.append("No large discontinuities were detected by the heuristic checks.")
    return warnings
