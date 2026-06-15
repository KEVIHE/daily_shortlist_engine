"""Filtering helpers for the stock shortlist universe."""

from __future__ import annotations

import pandas as pd


EXCLUDED_TICKERS = {"SOUN", "TOP", "HKD", "GNS"}


def apply_universe_filters(
    df: pd.DataFrame,
    min_price: float,
    min_dollar_volume: float,
    min_abs_move_percent: float,
) -> pd.DataFrame:
    """Remove illiquid or obscure names and attach liquidity labels."""
    if df.empty:
        return _empty_output()

    filtered = df.copy()
    filtered["liquidity_flag"] = "liquid"
    filtered.loc[
        (filtered["current_price"] < min_price)
        | (filtered["dollar_volume"] < min_dollar_volume)
        | (filtered["ticker"].isin(EXCLUDED_TICKERS)),
        "liquidity_flag",
    ] = "filtered_out"

    filtered = filtered[
        (filtered["current_price"] >= min_price)
        & (filtered["dollar_volume"] >= min_dollar_volume)
        & (filtered["move_percent"].abs() >= min_abs_move_percent)
        & (~filtered["ticker"].isin(EXCLUDED_TICKERS))
    ].copy()

    if filtered.empty:
        return _empty_output()

    return filtered


def _empty_output() -> pd.DataFrame:
    """Return an empty dataframe with the required export columns."""
    columns = [
        "list_type",
        "ticker",
        "current_price",
        "move_percent",
        "catalyst",
        "liquidity_flag",
        "setup_type",
        "risk_note",
        "score",
        "news_score",
        "liquidity_score",
        "setup_score",
        "tradeable",
        "tradeable_reason",
        "not_tradeable_reason",
    ]
    return pd.DataFrame(columns=columns)
