"""Label engineering for model training and replay validation."""

from __future__ import annotations

import pandas as pd


def add_ml_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add ranking, classification, and regression targets from realized outcomes."""
    if df.empty:
        return df.copy()
    labeled = df.copy()
    for column in [
        "horizon_5m_high",
        "horizon_5m_low",
        "horizon_15m_high",
        "horizon_15m_low",
        "horizon_30m_high",
        "horizon_30m_low",
        "breakout_low",
        "breakout_high",
        "invalidation",
        "last_price",
        "spread_pct",
    ]:
        if column not in labeled.columns:
            labeled[column] = 0.0
        labeled[column] = pd.to_numeric(labeled[column], errors="coerce").fillna(0.0)

    price = labeled["last_price"].replace(0, pd.NA)
    labeled["future_return_5m"] = ((labeled["horizon_5m_high"] / price) - 1.0).fillna(0.0) * 100.0
    labeled["future_return_15m"] = ((labeled["horizon_15m_high"] / price) - 1.0).fillna(0.0) * 100.0
    labeled["future_max_upside_5m"] = ((labeled["horizon_5m_high"] / price) - 1.0).fillna(0.0) * 100.0
    labeled["future_max_upside_15m"] = ((labeled["horizon_15m_high"] / price) - 1.0).fillna(0.0) * 100.0
    labeled["future_max_drawdown_5m"] = ((labeled["horizon_5m_low"] / price) - 1.0).fillna(0.0) * 100.0
    labeled["future_max_drawdown_15m"] = ((labeled["horizon_15m_low"] / price) - 1.0).fillna(0.0) * 100.0

    spread_cost = labeled["spread_pct"].fillna(0.0) * 0.5
    drawdown_penalty = labeled["future_max_drawdown_15m"].abs() * 0.6
    labeled["rank_target"] = labeled["future_return_15m"] - drawdown_penalty - spread_cost

    labeled["breakout_first_5m"] = ((labeled["horizon_5m_high"] >= labeled["breakout_low"]) & (labeled["horizon_5m_low"] > labeled["invalidation"]))
    labeled["breakout_first_15m"] = ((labeled["horizon_15m_high"] >= labeled["breakout_low"]) & (labeled["horizon_15m_low"] > labeled["invalidation"]))
    labeled["invalidation_first_5m"] = labeled["horizon_5m_low"] <= labeled["invalidation"]
    labeled["invalidation_first_15m"] = labeled["horizon_15m_low"] <= labeled["invalidation"]
    labeled["profitable_move_after_cost_5m"] = labeled["future_return_5m"] > spread_cost
    labeled["profitable_move_after_cost_15m"] = labeled["future_return_15m"] > spread_cost
    return labeled
