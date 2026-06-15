"""SQLite schema definitions for shortlist replay data."""

from __future__ import annotations

CANDIDATE_SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    rank INTEGER,
    list_type TEXT,
    last_price REAL,
    gap_pct REAL,
    move_percent REAL,
    volume REAL,
    relative_volume REAL,
    spread_pct REAL,
    dollar_volume REAL,
    catalyst TEXT,
    catalyst_type TEXT,
    news_score REAL,
    setup_score REAL,
    liquidity_score REAL,
    risk_score REAL,
    ml_score REAL,
    total_score REAL,
    status_tag TEXT,
    tradeable INTEGER,
    base_low REAL,
    base_high REAL,
    breakout_low REAL,
    breakout_high REAL,
    pullback_low REAL,
    pullback_high REAL,
    invalidation REAL,
    predicted_upper_band REAL,
    predicted_lower_band REAL,
    breakout_probability REAL,
    failure_probability REAL,
    confidence TEXT,
    vwap REAL,
    last_update_time TEXT,
    action_note TEXT,
    raw_json TEXT,
    UNIQUE(timestamp, symbol)
);
"""

NEWS_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS news_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT,
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT,
    catalyst_type TEXT,
    sentiment_or_strength REAL,
    raw_json TEXT,
    UNIQUE(timestamp, symbol, headline, source)
);
"""

RANGE_OUTCOMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS range_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    prediction_timestamp TEXT NOT NULL,
    horizon_5m_high REAL,
    horizon_5m_low REAL,
    horizon_15m_high REAL,
    horizon_15m_low REAL,
    horizon_30m_high REAL,
    horizon_30m_low REAL,
    base_range_hit INTEGER,
    breakout_zone_hit INTEGER,
    invalidation_hit_first INTEGER,
    updated_at TEXT,
    UNIQUE(symbol, prediction_timestamp)
);
"""

MODEL_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    train_start TEXT,
    train_end TEXT,
    valid_start TEXT,
    valid_end TEXT,
    test_start TEXT,
    test_end TEXT,
    metrics_json TEXT,
    feature_importance_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(model_name, model_version)
);
"""
