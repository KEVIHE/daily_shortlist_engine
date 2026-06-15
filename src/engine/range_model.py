"""Rule-based range model for intraday execution context."""

from __future__ import annotations

import pandas as pd


def apply_range_model(df: pd.DataFrame) -> pd.DataFrame:
    """Add base/breakout/pullback/invalidation ranges and confidence."""
    if df.empty:
        return df.copy()

    enriched = df.copy()
    effective_vwap = enriched["vwap"].where(enriched["vwap"] > 0, enriched["current_price"])
    centers = 0.5 * enriched["current_price"] + 0.5 * effective_vwap
    widths = pd.concat(
        [
            enriched["atr_5m"],
            enriched["recent_15m_range"] * 0.35,
            enriched["current_price"] * 0.003,
        ],
        axis=1,
    ).max(axis=1)
    enriched["base_low"] = (centers - widths).clip(lower=0)
    enriched["base_high"] = centers + widths
    enriched["breakout_low"] = enriched["recent_resistance"].where(enriched["recent_resistance"] > 0, enriched["current_price"])
    enriched["breakout_high"] = enriched["breakout_low"] + 0.8 * enriched["atr_5m"].where(enriched["atr_5m"] > 0, enriched["current_price"] * 0.002)
    enriched["pullback_low"] = pd.concat(
        [
            enriched["vwap"] - 0.3 * enriched["atr_5m"],
            enriched["recent_support"],
        ],
        axis=1,
    ).max(axis=1)
    enriched["pullback_high"] = enriched["vwap"] + 0.2 * enriched["atr_5m"].where(enriched["atr_5m"] > 0, enriched["current_price"] * 0.002)
    enriched["invalidation"] = (enriched["recent_support"] - 0.3 * enriched["atr_5m"]).clip(lower=0)
    predicted_upside = pd.to_numeric(enriched.get("predicted_upside_pct", 0.0), errors="coerce").fillna(0.0)
    predicted_downside = pd.to_numeric(enriched.get("predicted_downside_pct", 0.0), errors="coerce").fillna(0.0)
    predicted_upper = enriched["current_price"] * (1.0 + predicted_upside.clip(lower=0.0) / 100.0)
    predicted_lower = enriched["current_price"] * (1.0 + predicted_downside.clip(upper=0.0) / 100.0)
    enriched["predicted_upper_band"] = predicted_upper.where(predicted_upper > 0, enriched["base_high"]).fillna(enriched["base_high"])
    enriched["predicted_lower_band"] = predicted_lower.where(predicted_lower > 0, enriched["base_low"]).fillna(enriched["base_low"])
    enriched["confidence"] = enriched.apply(_confidence_label, axis=1)
    return enriched


def _confidence_label(row: pd.Series) -> str:
    completeness = float(row.get("data_completeness", 0.0))
    liquidity = float(row.get("liquidity_score", 0.0))
    news_score = float(row.get("news_score", 0.0))
    structure = float(row.get("setup_score", 0.0))
    if completeness >= 0.8 and liquidity >= 6.0 and news_score >= 6.0 and structure >= 6.0:
        return "high"
    if completeness >= 0.6 and liquidity >= 4.0 and structure >= 4.5:
        return "medium"
    return "low"
