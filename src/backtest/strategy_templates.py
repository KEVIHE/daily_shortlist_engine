"""Positive-expectancy candidate template evaluation."""

from __future__ import annotations

import pandas as pd

from src.backtest.evaluation import evaluate_strategy


TEMPLATES = {
    "breakout_high_score": lambda df: (df["status_tag"] == "Breakout Watch") & (df["total_score"] >= 6.0),
    "pullback_tradeable": lambda df: (df["status_tag"] == "Pullback Watch") & (df["tradeable"] == 1),
    "news_momentum": lambda df: df["catalyst_type"].isin(["earnings", "contract", "m_and_a", "price_action"]),
    "ml_supported": lambda df: pd.to_numeric(df.get("ml_score", 0), errors="coerce").fillna(0.0) >= 5.0,
}


def evaluate_strategy_templates(df: pd.DataFrame) -> pd.DataFrame:
    """Return candidate template metrics on realized future-return targets."""
    if df.empty or "future_return_15m" not in df.columns:
        return pd.DataFrame(columns=["template", "sample_count", "win_rate", "average_return", "average_loss", "payoff_ratio", "max_drawdown"])
    rows = []
    for name, fn in TEMPLATES.items():
        try:
            mask = fn(df)
        except Exception:
            continue
        metrics = evaluate_strategy(df, "future_return_15m", mask)
        metrics["template"] = name
        rows.append(metrics)
    if not rows:
        return pd.DataFrame(columns=["template", "sample_count", "win_rate", "average_return", "average_loss", "payoff_ratio", "max_drawdown"])
    frame = pd.DataFrame(rows)
    return frame.sort_values(["average_return", "win_rate", "sample_count"], ascending=[False, False, False]).reset_index(drop=True)
