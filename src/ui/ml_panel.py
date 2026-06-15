"""ML summary and validation panels for the workstation."""

from __future__ import annotations

from typing import Any

import json
import pandas as pd
import streamlit as st

from src.ui.dashboard import render_metric_cards, render_section_header


def render_ml_summary(run_status: dict[str, Any], latest_df: pd.DataFrame) -> None:
    """Render lightweight ML summary cards on the dashboard."""
    if latest_df.empty:
        render_metric_cards([
            ("ML Status", "Fallback", "No candidates available"),
            ("Avg ML Score", "0.00", "No rows in current shortlist"),
            ("Avg Breakout Prob", "0.00", "No rows in current shortlist"),
            ("Avg Failure Prob", "0.00", "No rows in current shortlist"),
        ])
        return
    render_metric_cards([
        ("ML Status", str(run_status.get("ml_status") or "fallback").title(), str(run_status.get("ml_message") or "No model metadata recorded.")),
        ("Avg ML Score", f"{pd.to_numeric(latest_df.get('ml_score'), errors='coerce').fillna(0.0).mean():.2f}", "Current shortlist mean"),
        ("Avg Breakout Prob", f"{pd.to_numeric(latest_df.get('breakout_probability'), errors='coerce').fillna(0.0).mean() * 100:.1f}%", "15-minute breakout probability"),
        ("Avg Failure Prob", f"{pd.to_numeric(latest_df.get('failure_probability'), errors='coerce').fillna(0.0).mean() * 100:.1f}%", "15-minute failure probability"),
    ])


def render_model_runs(model_runs: pd.DataFrame) -> None:
    """Show recent model training metadata and feature importance."""
    render_section_header("Model Validation", "Recent train/validation/test splits and feature importance.")
    if model_runs.empty:
        st.info("The model has not trained successfully yet. The system is currently falling back to the rule-based scoring layer.")
        return
    display = model_runs.copy()
    st.dataframe(display[["model_name", "model_version", "train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end", "created_at"]], hide_index=True, width="stretch")
    selected = display.iloc[0]
    with st.expander("Latest model metrics", expanded=True):
        st.json(_safe_json_parse(selected.get("metrics_json")))
    with st.expander("Latest feature importance", expanded=False):
        importance = _safe_json_parse(selected.get("feature_importance_json"))
        if isinstance(importance, list) and importance:
            st.dataframe(pd.DataFrame(importance), hide_index=True, width="stretch")
        else:
            st.info("No feature importance recorded.")


def render_prediction_panel(row: dict[str, Any]) -> None:
    """Render the candidate-level ML outputs in the detail panel."""
    render_section_header("ML Layer", "Ranking, probability estimates, and upper/lower band forecasts. Falls back automatically when data is insufficient.")
    render_metric_cards([
        ("ML Score", _fmt_score(row.get("ml_score")), "Model-assisted ranking"),
        ("Breakout Prob", _fmt_prob(row.get("breakout_probability")), "Future 15m breakout-first estimate"),
        ("Failure Prob", _fmt_prob(row.get("failure_probability")), "Future 15m invalidation-first estimate"),
        ("Predicted Upper", _fmt_price(row.get("predicted_upper_band")), "Model upper band"),
        ("Predicted Lower", _fmt_price(row.get("predicted_lower_band")), "Model lower band"),
    ])


def _safe_json_parse(value: Any) -> Any:
    if not value:
        return {}
    try:
        payload = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return {}
    return payload


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _fmt_prob(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _fmt_price(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "--"
