"""Structured ML feature preparation for shortlist candidates."""

from __future__ import annotations

from typing import Any

import pandas as pd

NUMERIC_FEATURES = [
    "last_price",
    "current_price",
    "gap_pct",
    "move_percent",
    "volume_today",
    "relative_volume",
    "dollar_volume",
    "spread_pct",
    "ret_1m",
    "ret_5m",
    "ret_15m",
    "range_5m",
    "range_15m",
    "distance_to_vwap_pct",
    "distance_to_intraday_high_pct",
    "distance_to_premarket_high_pct",
    "breakout_ready_score",
    "trade_count_recent",
    "news_count_30m",
    "freshness_minutes",
    "headline_strength",
    "source_count",
    "volatility_1m",
    "volatility_5m",
    "volatility_15m",
    "overextension_score",
    "wickiness_score",
    "halt_risk_proxy",
    "market_trend_strength",
    "sector_strength",
    "breadth_proxy",
    "vwap_regime_flag",
    "news_score",
    "setup_score",
    "liquidity_score",
    "risk_score",
]
CATEGORICAL_FEATURES = [
    "catalyst_type",
    "status_tag",
    "setup_bias",
    "list_type",
    "index_regime",
    "session",
    "mover_group",
]
ML_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["minutes_from_open"]


def prepare_ml_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize candidate features for LightGBM models."""
    if df.empty:
        return pd.DataFrame(columns=ML_FEATURE_COLUMNS)
    frame = df.copy()
    for column in NUMERIC_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_FEATURES:
        if column not in frame.columns:
            frame[column] = "unknown"
        frame[column] = frame[column].fillna("unknown").astype(str).str.strip().replace({"": "unknown"}).astype("category")
    frame["minutes_from_open"] = frame.apply(_minutes_from_open, axis=1).astype(float)
    return frame[ML_FEATURE_COLUMNS].copy()


def prepare_ml_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the feature columns needed for inference."""
    prepared = prepare_ml_feature_frame(df)
    return prepared


def _minutes_from_open(row: pd.Series) -> float:
    text = str(row.get("last_update_time") or "").strip().replace("Z", "+00:00")
    if not text:
        return 0.0
    try:
        ts = pd.Timestamp(text)
        et = ts.tz_convert("America/New_York") if ts.tzinfo else ts.tz_localize("UTC").tz_convert("America/New_York")
        open_ts = et.normalize() + pd.Timedelta(hours=9, minutes=30)
        return max((et - open_ts).total_seconds() / 60.0, 0.0)
    except Exception:
        return 0.0
