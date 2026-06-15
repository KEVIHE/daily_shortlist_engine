"""SQLite persistence for shortlist snapshots, news, replay metrics, and model runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.strategy_templates import evaluate_strategy_templates
from src.data.alpaca_market import AlpacaMarketDataClient
from src.storage.models import CANDIDATE_SNAPSHOTS_SCHEMA, MODEL_RUNS_SCHEMA, NEWS_EVENTS_SCHEMA, RANGE_OUTCOMES_SCHEMA


CANDIDATE_EXTRA_COLUMNS = {
    "ml_score": "ALTER TABLE candidate_snapshots ADD COLUMN ml_score REAL",
    "predicted_upper_band": "ALTER TABLE candidate_snapshots ADD COLUMN predicted_upper_band REAL",
    "predicted_lower_band": "ALTER TABLE candidate_snapshots ADD COLUMN predicted_lower_band REAL",
    "breakout_probability": "ALTER TABLE candidate_snapshots ADD COLUMN breakout_probability REAL",
    "failure_probability": "ALTER TABLE candidate_snapshots ADD COLUMN failure_probability REAL",
}


def initialize_database(db_path: Path) -> None:
    """Ensure the sqlite database and required tables exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(CANDIDATE_SNAPSHOTS_SCHEMA)
        connection.execute(NEWS_EVENTS_SCHEMA)
        connection.execute(RANGE_OUTCOMES_SCHEMA)
        connection.execute(MODEL_RUNS_SCHEMA)
        _ensure_columns(connection, "candidate_snapshots", CANDIDATE_EXTRA_COLUMNS)
        connection.commit()


def record_candidate_snapshots(db_path: Path, shortlist_df: pd.DataFrame, generated_at: datetime) -> None:
    """Persist the current shortlist snapshot for later replay analysis."""
    if shortlist_df.empty:
        return
    timestamp = generated_at.isoformat(timespec="seconds")
    rows = []
    for _, row in shortlist_df.iterrows():
        payload = row.to_dict()
        rows.append(
            (
                timestamp,
                str(payload.get("symbol") or payload.get("ticker") or ""),
                int(payload.get("rank") or 0),
                str(payload.get("list_type") or ""),
                float(payload.get("last_price") or payload.get("current_price") or 0.0),
                float(payload.get("gap_pct") or 0.0),
                float(payload.get("move_percent") or 0.0),
                float(payload.get("volume_today") or payload.get("volume") or 0.0),
                float(payload.get("relative_volume") or 0.0),
                float(payload.get("spread_pct") or 0.0),
                float(payload.get("dollar_volume") or 0.0),
                str(payload.get("catalyst") or ""),
                str(payload.get("catalyst_type") or ""),
                float(payload.get("news_score") or 0.0),
                float(payload.get("setup_score") or 0.0),
                float(payload.get("liquidity_score") or 0.0),
                float(payload.get("risk_score") or 0.0),
                float(payload.get("ml_score") or 0.0),
                float(payload.get("total_score") or 0.0),
                str(payload.get("status_tag") or ""),
                1 if bool(payload.get("tradeable")) else 0,
                float(payload.get("base_low") or 0.0),
                float(payload.get("base_high") or 0.0),
                float(payload.get("breakout_low") or 0.0),
                float(payload.get("breakout_high") or 0.0),
                float(payload.get("pullback_low") or 0.0),
                float(payload.get("pullback_high") or 0.0),
                float(payload.get("invalidation") or 0.0),
                float(payload.get("predicted_upper_band") or 0.0),
                float(payload.get("predicted_lower_band") or 0.0),
                float(payload.get("breakout_probability") or 0.0),
                float(payload.get("failure_probability") or 0.0),
                str(payload.get("confidence") or ""),
                float(payload.get("vwap") or 0.0),
                str(payload.get("last_update_time") or ""),
                str(payload.get("action_note") or ""),
                json.dumps(payload, ensure_ascii=True),
            )
        )
    with sqlite3.connect(db_path) as connection:
        _ensure_columns(connection, "candidate_snapshots", CANDIDATE_EXTRA_COLUMNS)
        connection.executemany(
            """
            INSERT OR REPLACE INTO candidate_snapshots (
                timestamp, symbol, rank, list_type, last_price, gap_pct, move_percent, volume,
                relative_volume, spread_pct, dollar_volume, catalyst, catalyst_type, news_score,
                setup_score, liquidity_score, risk_score, ml_score, total_score, status_tag, tradeable,
                base_low, base_high, breakout_low, breakout_high, pullback_low, pullback_high,
                invalidation, predicted_upper_band, predicted_lower_band, breakout_probability,
                failure_probability, confidence, vwap, last_update_time, action_note, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def record_news_events(db_path: Path, news_df: pd.DataFrame) -> int:
    """Persist normalized news and filing events."""
    if news_df.empty:
        return 0
    rows = []
    for _, row in news_df.iterrows():
        rows.append(
            (
                str(row.get("timestamp") or ""),
                str(row.get("symbol") or ""),
                str(row.get("headline") or ""),
                str(row.get("source") or ""),
                str(row.get("url") or ""),
                str(row.get("catalyst_type") or ""),
                float(row.get("headline_strength") or row.get("sentiment_or_strength") or 0.0),
                str(row.get("raw_json") or ""),
            )
        )
    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO news_events (
                timestamp, symbol, headline, source, url, catalyst_type, sentiment_or_strength, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
        inserted = connection.execute("SELECT changes()").fetchone()[0]
    return int(inserted)


def record_model_run(
    db_path: Path,
    model_name: str,
    model_version: str,
    split_meta: dict[str, Any],
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
) -> None:
    """Persist one model training metadata record."""
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO model_runs (
                model_name, model_version, train_start, train_end, valid_start, valid_end,
                test_start, test_end, metrics_json, feature_importance_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_name,
                model_version,
                split_meta.get("train_start", ""),
                split_meta.get("train_end", ""),
                split_meta.get("valid_start", ""),
                split_meta.get("valid_end", ""),
                split_meta.get("test_start", ""),
                split_meta.get("test_end", ""),
                json.dumps(metrics, ensure_ascii=True),
                json.dumps(feature_importance, ensure_ascii=True),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        connection.commit()


def backfill_range_outcomes(db_path: Path, market_client: AlpacaMarketDataClient, now_ts: datetime) -> int:
    """Backfill realized range outcomes for snapshots older than 30 minutes."""
    pending = load_pending_outcomes(db_path, now_ts)
    if pending.empty:
        return 0

    inserted = 0
    with sqlite3.connect(db_path) as connection:
        for _, row in pending.iterrows():
            prediction_ts = _parse_timestamp(row["timestamp"])
            if prediction_ts is None:
                continue
            start = prediction_ts
            end = prediction_ts + timedelta(minutes=31)
            bars_map = market_client.fetch_historical_bars([row["symbol"]], "1Min", start, end, limit=120)
            bars = pd.DataFrame(bars_map.get(row["symbol"], []))
            if bars.empty:
                continue
            for column in ["h", "l"]:
                bars[column] = pd.to_numeric(bars[column], errors="coerce").fillna(0.0)
            bars["t"] = pd.to_datetime(bars.get("t"), errors="coerce", utc=True)
            bars = bars.dropna(subset=["t"]).sort_values("t")
            outcome = _compute_outcome(row, bars, now_ts)
            connection.execute(
                """
                INSERT OR REPLACE INTO range_outcomes (
                    symbol, prediction_timestamp, horizon_5m_high, horizon_5m_low, horizon_15m_high,
                    horizon_15m_low, horizon_30m_high, horizon_30m_low, base_range_hit,
                    breakout_zone_hit, invalidation_hit_first, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                outcome,
            )
            inserted += 1
        connection.commit()
    return inserted


def load_pending_outcomes(db_path: Path, now_ts: datetime) -> pd.DataFrame:
    """Return candidate snapshots that are ready for replay backfill."""
    threshold = (now_ts - timedelta(minutes=30)).isoformat(timespec="seconds")
    query = """
        SELECT c.*
        FROM candidate_snapshots c
        LEFT JOIN range_outcomes r
          ON c.symbol = r.symbol AND c.timestamp = r.prediction_timestamp
        WHERE c.timestamp <= ? AND r.id IS NULL
        ORDER BY c.timestamp DESC
        LIMIT 100
    """
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(query, connection, params=(threshold,))


def load_replay_metrics(db_path: Path) -> dict[str, Any]:
    """Load aggregate replay, model-run, and strategy metrics for the UI."""
    if not db_path.exists():
        return {
            "overview": {},
            "by_catalyst": pd.DataFrame(),
            "by_status": pd.DataFrame(),
            "recent_predictions": pd.DataFrame(),
            "by_ml_bucket": pd.DataFrame(),
            "strategy_candidates": pd.DataFrame(),
            "model_runs": pd.DataFrame(),
        }
    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        overview = pd.read_sql_query(
            """
            SELECT
                COUNT(*) AS total_predictions,
                AVG(CASE WHEN total_score >= 6.0 THEN breakout_zone_hit END) AS top_score_hit_rate,
                AVG(base_range_hit) AS base_range_hit_rate,
                AVG(breakout_zone_hit) AS breakout_zone_hit_rate,
                AVG(invalidation_hit_first) AS invalidation_first_rate
            FROM candidate_snapshots c
            JOIN range_outcomes r
              ON c.symbol = r.symbol AND c.timestamp = r.prediction_timestamp
            """,
            connection,
        )
        joined = pd.read_sql_query(
            """
            SELECT c.timestamp, c.symbol, c.total_score, c.ml_score, c.status_tag, c.catalyst_type,
                   c.tradeable, r.base_range_hit, r.breakout_zone_hit, r.invalidation_hit_first,
                   r.horizon_15m_high, r.horizon_15m_low, c.last_price, c.spread_pct, c.raw_json
            FROM candidate_snapshots c
            JOIN range_outcomes r
              ON c.symbol = r.symbol AND c.timestamp = r.prediction_timestamp
            ORDER BY c.timestamp DESC, c.rank ASC
            """,
            connection,
        )
        model_runs = pd.read_sql_query(
            "SELECT model_name, model_version, train_start, train_end, valid_start, valid_end, test_start, test_end, metrics_json, feature_importance_json, created_at FROM model_runs ORDER BY created_at DESC LIMIT 20",
            connection,
        )
    overview_row = overview.iloc[0].to_dict() if not overview.empty else {}
    if joined.empty:
        return {
            "overview": overview_row,
            "by_catalyst": pd.DataFrame(),
            "by_status": pd.DataFrame(),
            "recent_predictions": pd.DataFrame(),
            "by_ml_bucket": pd.DataFrame(),
            "strategy_candidates": pd.DataFrame(),
            "model_runs": model_runs,
        }
    joined["future_return_15m"] = (((pd.to_numeric(joined["horizon_15m_high"], errors="coerce").fillna(0.0) / pd.to_numeric(joined["last_price"], errors="coerce").replace(0, pd.NA)) - 1.0).fillna(0.0) * 100.0)
    joined["ml_bucket"] = pd.cut(pd.to_numeric(joined["ml_score"], errors="coerce").fillna(0.0), bins=[-0.1, 2.5, 5.0, 7.5, 10.0], labels=["0-2.5", "2.5-5", "5-7.5", "7.5-10"])
    by_catalyst = joined.groupby("catalyst_type", dropna=False, observed=False).agg(
        sample_count=("symbol", "count"),
        base_range_hit_rate=("base_range_hit", "mean"),
        breakout_zone_hit_rate=("breakout_zone_hit", "mean"),
        invalidation_first_rate=("invalidation_hit_first", "mean"),
        avg_return_15m=("future_return_15m", "mean"),
    ).reset_index().sort_values(["sample_count", "breakout_zone_hit_rate"], ascending=[False, False])
    by_status = joined.groupby("status_tag", dropna=False, observed=False).agg(
        sample_count=("symbol", "count"),
        base_range_hit_rate=("base_range_hit", "mean"),
        breakout_zone_hit_rate=("breakout_zone_hit", "mean"),
        invalidation_first_rate=("invalidation_hit_first", "mean"),
        avg_return_15m=("future_return_15m", "mean"),
    ).reset_index().sort_values(["sample_count", "breakout_zone_hit_rate"], ascending=[False, False])
    by_ml_bucket = joined.groupby("ml_bucket", dropna=False, observed=False).agg(
        sample_count=("symbol", "count"),
        breakout_zone_hit_rate=("breakout_zone_hit", "mean"),
        invalidation_first_rate=("invalidation_hit_first", "mean"),
        avg_return_15m=("future_return_15m", "mean"),
    ).reset_index()
    strategy_candidates = evaluate_strategy_templates(joined)
    recent_predictions = joined[["timestamp", "symbol", "total_score", "ml_score", "status_tag", "catalyst_type", "base_range_hit", "breakout_zone_hit", "invalidation_hit_first", "future_return_15m"]].head(100)
    return {
        "overview": overview_row,
        "by_catalyst": by_catalyst,
        "by_status": by_status,
        "recent_predictions": recent_predictions,
        "by_ml_bucket": by_ml_bucket,
        "strategy_candidates": strategy_candidates,
        "model_runs": model_runs,
    }


def load_news_count_today(db_path: Path) -> int:
    """Return today's ingested news event count."""
    if not db_path.exists():
        return 0
    initialize_database(db_path)
    start = datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM news_events WHERE substr(timestamp, 1, 10) = ?",
            (start,),
        ).fetchone()
    return int(row[0] if row else 0)


def load_recent_news_events(db_path: Path, limit: int = 50) -> pd.DataFrame:
    """Return recent normalized news events for the UI."""
    if not db_path.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "headline", "source", "catalyst_type"])
    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT timestamp, symbol, headline, source, catalyst_type FROM news_events ORDER BY timestamp DESC LIMIT ?",
            connection,
            params=(limit,),
        )


def _ensure_columns(connection: sqlite3.Connection, table: str, statements: dict[str, str]) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
    for column, statement in statements.items():
        if column not in existing:
            connection.execute(statement)


def _compute_outcome(row: pd.Series, bars: pd.DataFrame, now_ts: datetime) -> tuple[Any, ...]:
    prediction_ts = str(row["timestamp"])
    symbol = str(row["symbol"])
    prediction_dt = _parse_timestamp(prediction_ts)
    if prediction_dt is None:
        prediction_dt = now_ts
    horizon_5 = bars[bars["t"] <= prediction_dt + timedelta(minutes=5)]
    horizon_15 = bars[bars["t"] <= prediction_dt + timedelta(minutes=15)]
    horizon_30 = bars[bars["t"] <= prediction_dt + timedelta(minutes=30)]
    base_range_hit = int(_zone_hit(horizon_30, float(row["base_low"]), float(row["base_high"])))
    breakout_zone_hit = int(_zone_hit(horizon_30, float(row["breakout_low"]), float(row["breakout_high"])))
    invalidation_time = _first_hit_time(horizon_30, float(row["invalidation"]), "below")
    breakout_time = _first_zone_hit_time(horizon_30, float(row["breakout_low"]), float(row["breakout_high"]))
    invalidation_hit_first = int(bool(invalidation_time and (not breakout_time or invalidation_time <= breakout_time)))
    return (
        symbol,
        prediction_ts,
        _high(horizon_5),
        _low(horizon_5),
        _high(horizon_15),
        _low(horizon_15),
        _high(horizon_30),
        _low(horizon_30),
        base_range_hit,
        breakout_zone_hit,
        invalidation_hit_first,
        now_ts.isoformat(timespec="seconds"),
    )


def _zone_hit(bars: pd.DataFrame, low: float, high: float) -> bool:
    if bars.empty or low <= 0 or high <= 0:
        return False
    return bool(((bars["h"] >= low) & (bars["l"] <= high)).any())


def _first_hit_time(bars: pd.DataFrame, level: float, direction: str) -> datetime | None:
    if bars.empty or level <= 0:
        return None
    if direction == "below":
        hits = bars[bars["l"] <= level]
    else:
        hits = bars[bars["h"] >= level]
    if hits.empty:
        return None
    return hits.iloc[0]["t"].to_pydatetime()


def _first_zone_hit_time(bars: pd.DataFrame, low: float, high: float) -> datetime | None:
    if bars.empty:
        return None
    hits = bars[(bars["h"] >= low) & (bars["l"] <= high)]
    if hits.empty:
        return None
    return hits.iloc[0]["t"].to_pydatetime()


def _high(bars: pd.DataFrame) -> float | None:
    return float(bars["h"].max()) if not bars.empty else None


def _low(bars: pd.DataFrame) -> float | None:
    return float(bars["l"].min()) if not bars.empty else None


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
