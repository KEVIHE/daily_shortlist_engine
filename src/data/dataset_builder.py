"""Training dataset assembly from sqlite snapshot and outcome tables."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from src.engine.ml_targets import add_ml_targets


def build_training_dataset(db_path: Path) -> pd.DataFrame:
    """Join candidate snapshots with realized outcomes and expand raw features."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query(
            """
            SELECT c.*, r.horizon_5m_high, r.horizon_5m_low, r.horizon_15m_high, r.horizon_15m_low,
                   r.horizon_30m_high, r.horizon_30m_low, r.base_range_hit, r.breakout_zone_hit,
                   r.invalidation_hit_first
            FROM candidate_snapshots c
            JOIN range_outcomes r
              ON c.symbol = r.symbol AND c.timestamp = r.prediction_timestamp
            ORDER BY c.timestamp ASC, c.rank ASC
            """,
            connection,
        )
    if frame.empty:
        return frame
    expanded_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        raw_payload = _parse_raw_json(row.get("raw_json"))
        payload = row.to_dict()
        if isinstance(raw_payload, dict):
            for key, value in raw_payload.items():
                payload.setdefault(key, value)
        expanded_rows.append(payload)
    dataset = pd.DataFrame(expanded_rows)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], errors="coerce", utc=True)
    dataset = dataset.dropna(subset=["timestamp"]).reset_index(drop=True)
    return add_ml_targets(dataset)


def _parse_raw_json(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
