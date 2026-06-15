"""Replay page rendering for shortlist outcome statistics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.storage.db import load_replay_metrics
from src.ui.dashboard import render_section_header
from src.ui.ml_panel import render_model_runs


def render_replay_page(db_path) -> None:
    """Render replay metrics, model validation, and strategy research outcomes."""
    metrics = load_replay_metrics(db_path)
    overview = metrics["overview"]
    render_section_header("Replay / Evaluation", "命中率、模型验证和策略候选模板都在这里看。")

    cards = st.columns(4)
    cards[0].metric("Predictions", int(overview.get("total_predictions") or 0))
    cards[1].metric("Top Score Hit", _fmt_pct(overview.get("top_score_hit_rate")))
    cards[2].metric("Base Range Hit", _fmt_pct(overview.get("base_range_hit_rate")))
    cards[3].metric("Breakout Hit", _fmt_pct(overview.get("breakout_zone_hit_rate")))
    st.metric("Invalidation First", _fmt_pct(overview.get("invalidation_first_rate")))

    render_section_header("By Catalyst Type", "按催化类型拆分看命中率和未来 15 分钟平均表现。")
    by_catalyst = metrics["by_catalyst"]
    if by_catalyst.empty:
        st.info("Replay outcomes are not available yet. Let the system accumulate more snapshots and backfilled bars.")
    else:
        st.dataframe(_format_metric_frame(by_catalyst), hide_index=True, width="stretch")

    render_section_header("By Status Tag", "看 Breakout / Pullback / Risky 等状态标签在历史上表现如何。")
    by_status = metrics["by_status"]
    if by_status.empty:
        st.info("No status-tag replay metrics are available yet.")
    else:
        st.dataframe(_format_metric_frame(by_status), hide_index=True, width="stretch")

    render_section_header("By ML Bucket", "按 ML Score 分层观察 breakout hit 与 invalidation 先触发比例。")
    by_ml_bucket = metrics["by_ml_bucket"]
    if by_ml_bucket.empty:
        st.info("ML score bucket analysis is not available yet.")
    else:
        st.dataframe(_format_metric_frame(by_ml_bucket), hide_index=True, width="stretch")

    render_section_header("Strategy Candidates", "这些只是候选正期望模板，不等于可直接实盘的保证盈利策略。")
    strategy_candidates = metrics["strategy_candidates"]
    if strategy_candidates.empty:
        st.info("还没有足够的历史样本去评估策略模板。")
    else:
        st.dataframe(_format_metric_frame(strategy_candidates), hide_index=True, width="stretch")

    render_model_runs(metrics["model_runs"])

    render_section_header("Recent Predictions", "最近记录下来的候选与其后续 15 分钟 outcome。")
    recent = metrics["recent_predictions"]
    if recent.empty:
        st.info("No candidate snapshots have been recorded yet.")
    else:
        recent_display = recent.copy()
        if "timestamp" in recent_display.columns:
            recent_display["timestamp"] = recent_display["timestamp"].astype(str).str.slice(0, 16)
        for column in ["base_range_hit", "breakout_zone_hit", "invalidation_hit_first"]:
            if column in recent_display.columns:
                recent_display[column] = recent_display[column].map(lambda value: "Yes" if bool(value) else "No")
        st.dataframe(recent_display, hide_index=True, width="stretch")


def _format_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Format replay metric frames for readable display."""
    display = frame.copy()
    pct_columns = [column for column in display.columns if column.endswith("_rate")]
    for column in pct_columns:
        display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0.0).map(lambda value: f"{value * 100:.1f}%")
    if "avg_return_15m" in display.columns:
        display["avg_return_15m"] = pd.to_numeric(display["avg_return_15m"], errors="coerce").fillna(0.0).map(lambda value: f"{value:.2f}%")
    if "win_rate" in display.columns:
        display["win_rate"] = pd.to_numeric(display["win_rate"], errors="coerce").fillna(0.0).map(lambda value: f"{value * 100:.1f}%")
    for column in ["average_return", "average_loss", "payoff_ratio", "max_drawdown"]:
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0.0).map(lambda value: f"{value:.2f}")
    return display


def _fmt_pct(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "--"
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "--"
