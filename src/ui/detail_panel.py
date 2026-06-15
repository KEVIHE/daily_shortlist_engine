"""Detail panel helpers for shortlist candidates."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.ui.dashboard import format_timestamp, render_detail_panel, render_metric_cards
from src.ui.ml_panel import render_prediction_panel


def render_candidate_detail(row: dict[str, Any], context: dict[str, Any] | None = None) -> None:
    """Render the candidate detail panel with quotes, zones, bars, and catalyst context."""
    context = context or {}
    latest_quote = context.get("latest_quote", {}) if isinstance(context, dict) else {}
    intraday_bars = context.get("intraday_bars", []) if isinstance(context, dict) else []
    news_items = context.get("news", []) if isinstance(context, dict) else []

    render_metric_cards(
        [
            ("Symbol", str(row.get("symbol") or "--"), str(row.get("status_tag") or "Ignore")),
            ("Last Price", _fmt_price(row.get("last_price")), f"VWAP {_fmt_price(row.get('vwap'))}"),
            ("Bid / Ask", f"{_fmt_price(_quote_or_row(latest_quote, 'bp', row.get('bid_price')))} / {_fmt_price(_quote_or_row(latest_quote, 'ap', row.get('ask_price')))}", f"Spread {_fmt_pct(row.get('spread_pct'))}"),
            ("Total Score", _fmt_score(row.get("total_score")), f"Confidence {str(row.get('confidence') or 'low').upper()}"),
            ("Tradeable", "Yes" if bool(row.get("tradeable")) else "No", str(row.get("setup_bias") or "neutral").title()),
        ]
    )
    render_prediction_panel(row)

    top_left, top_right = st.columns([1.05, 0.95])
    with top_left:
        render_detail_panel(
            "Execution Zones",
            [
                ("Latest Refresh", format_timestamp(row.get("last_update_time") or row.get("quote_time"))),
                ("Base Range", _fmt_range(row.get("base_low"), row.get("base_high"))),
                ("Breakout Zone", _fmt_range(row.get("breakout_low"), row.get("breakout_high"))),
                ("Pullback Zone", _fmt_range(row.get("pullback_low"), row.get("pullback_high"))),
                ("Invalidation", _fmt_price(row.get("invalidation"))),
                ("Predicted Upper", _fmt_price(row.get("predicted_upper_band"))),
                ("Predicted Lower", _fmt_price(row.get("predicted_lower_band"))),
                ("Why This Range", _range_reason(row)),
            ],
        )
        render_detail_panel(
            "Execution Context",
            [
                ("Selection Reason", row.get("selection_reason") or "No summary available"),
                ("Catalyst", row.get("catalyst") or "No fresh catalyst"),
                ("Catalyst Type", row.get("catalyst_type") or "none"),
                ("SEC Filing", row.get("sec_filing_type") or "None"),
                ("Action Note", row.get("action_note") or "No action note"),
                ("Tradeable Reason", row.get("tradeable_reason") or "N/A"),
                ("Not Tradeable", row.get("not_tradeable_reason") or "N/A"),
                ("Risk", row.get("risk_note") or "Manual review required"),
            ],
        )
        _render_score_breakdown(row)
    with top_right:
        st.markdown("**Recent 1-Min Structure**")
        bars_frame = _bars_frame(intraday_bars)
        if bars_frame.empty:
            st.info("Recent bars are not available in the latest context snapshot.")
        else:
            st.altair_chart(_candlestick_chart(bars_frame, row), width="stretch")
            st.dataframe(
                bars_frame.tail(12),
                hide_index=True,
                width="stretch",
                column_config={
                    "time": st.column_config.TextColumn("Time"),
                    "open": st.column_config.NumberColumn("Open", format="%.2f"),
                    "high": st.column_config.NumberColumn("High", format="%.2f"),
                    "low": st.column_config.NumberColumn("Low", format="%.2f"),
                    "close": st.column_config.NumberColumn("Close", format="%.2f"),
                    "volume": st.column_config.NumberColumn("Volume", format="%.0f"),
                },
            )

    st.markdown("**Latest Catalyst Feed**")
    if news_items:
        news_frame = pd.DataFrame(news_items)
        keep = [column for column in ["timestamp", "source", "catalyst_type", "headline", "url"] if column in news_frame.columns]
        if "timestamp" in news_frame.columns:
            news_frame["timestamp"] = news_frame["timestamp"].map(format_timestamp)
        st.dataframe(news_frame[keep], hide_index=True, width="stretch")
    else:
        st.info("No symbol-specific news items are attached to the latest context snapshot.")

    with st.expander("Debug data", expanded=False):
        st.json(
            {
                "latest_quote": _safe_debug(latest_quote),
                "latest_bar_count": len(intraday_bars),
                "news_count": len(news_items),
            }
        )


def _render_score_breakdown(row: dict[str, Any]) -> None:
    frame = pd.DataFrame(
        [
            {"component": "News", "value": float(row.get("news_score") or 0.0)},
            {"component": "Setup", "value": float(row.get("setup_score") or 0.0)},
            {"component": "Liquidity", "value": float(row.get("liquidity_score") or 0.0)},
            {"component": "Risk", "value": float(row.get("risk_score") or 0.0)},
            {"component": "ML", "value": float(row.get("ml_score") or 0.0)},
        ]
    )
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 10]), title="Score"),
            y=alt.Y("component:N", sort=["News", "Setup", "Liquidity", "Risk", "ML"], title=None),
            color=alt.Color("component:N", legend=None, scale=alt.Scale(range=["#2b6cb0", "#0f6a50", "#8b5cf6", "#b91c1c", "#99601b"])),
            tooltip=["component", alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=180)
    )
    st.markdown("**Score Breakdown**")
    st.altair_chart(chart, width="stretch")


def _candlestick_chart(frame: pd.DataFrame, row: dict[str, Any]) -> alt.Chart:
    color = alt.condition("datum.open <= datum.close", alt.value("#0f6a50"), alt.value("#b91c1c"))
    base = alt.Chart(frame)
    rule = base.mark_rule().encode(x="time:T", y="low:Q", y2="high:Q", color=color)
    bar = base.mark_bar(size=6).encode(x="time:T", y="open:Q", y2="close:Q", color=color)

    overlay_rows = []
    for label, value in [
        ("VWAP", row.get("vwap")),
        ("Current", row.get("last_price")),
        ("Breakout", row.get("breakout_low")),
        ("Invalidation", row.get("invalidation")),
        ("Pred Upper", row.get("predicted_upper_band")),
        ("Pred Lower", row.get("predicted_lower_band")),
    ]:
        try:
            overlay_rows.append({"label": label, "value": float(value)})
        except (TypeError, ValueError):
            continue
    overlay = alt.Chart(pd.DataFrame(overlay_rows)).mark_rule(strokeDash=[6, 4]).encode(
        y="value:Q",
        color=alt.Color("label:N", legend=alt.Legend(orient="bottom")),
        tooltip=["label", alt.Tooltip("value:Q", format=".2f")],
    ) if overlay_rows else alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_rule()

    pullback = alt.Chart(
        pd.DataFrame([
            {
                "y1": float(row.get("pullback_low") or 0.0),
                "y2": float(row.get("pullback_high") or 0.0),
                "x1": frame["time"].min(),
                "x2": frame["time"].max(),
            }
        ])
    ).mark_rect(opacity=0.12, color="#d97706").encode(x="x1:T", x2="x2:T", y="y1:Q", y2="y2:Q") if not frame.empty else alt.Chart(pd.DataFrame()).mark_rect()

    return (pullback + rule + bar + overlay).properties(height=320)


def _bars_frame(bars: list[dict[str, Any]]) -> pd.DataFrame:
    if not isinstance(bars, list) or not bars:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(bars)
    if frame.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    frame["time"] = pd.to_datetime(frame.get("t"), errors="coerce", utc=True)
    return pd.DataFrame(
        {
            "time": frame["time"],
            "open": pd.to_numeric(frame.get("o"), errors="coerce").fillna(0.0),
            "high": pd.to_numeric(frame.get("h"), errors="coerce").fillna(0.0),
            "low": pd.to_numeric(frame.get("l"), errors="coerce").fillna(0.0),
            "close": pd.to_numeric(frame.get("c"), errors="coerce").fillna(0.0),
            "volume": pd.to_numeric(frame.get("v"), errors="coerce").fillna(0.0),
        }
    ).dropna(subset=["time"])


def _fmt_price(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "--"


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "--"


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "--"


def _fmt_range(low: Any, high: Any) -> str:
    try:
        return f"{float(low):.2f} - {float(high):.2f}"
    except (TypeError, ValueError):
        return "--"


def _quote_or_row(quote: dict[str, Any], key: str, fallback: Any) -> Any:
    if isinstance(quote, dict) and quote.get(key) not in {None, ""}:
        return quote.get(key)
    return fallback


def _range_reason(row: dict[str, Any]) -> str:
    setup = str(row.get("setup_bias") or "neutral")
    if setup == "breakout":
        return "The structure leans toward breakout continuation, so the focus is on whether the upper resistance area gets confirmed."
    if setup == "pullback":
        return "The structure leans toward a pullback setup, so the first thing to watch is whether buyers hold near VWAP."
    if setup == "extended":
        return "Momentum is still present, but price is already too far from a reasonable chase zone."
    return "The name is better treated as watch-only until the structure becomes clearer."


def _safe_debug(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in (payload or {}).items():
        if any(token in str(key).lower() for token in ["key", "secret", "token", "auth"]):
            safe[str(key)] = "[REDACTED]"
        else:
            safe[str(key)] = value
    return safe
