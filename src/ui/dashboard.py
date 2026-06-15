"""Shared dashboard rendering helpers."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def inject_styles() -> None:
    """Apply a compact workstation theme that stays readable in normal laptop widths."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7fa;
            --panel: rgba(255,255,255,0.96);
            --line: rgba(15,23,42,0.10);
            --ink: #132033;
            --muted: #5d6b7d;
            --good: #0f6a50;
            --good-soft: #eaf6f1;
            --warn: #99601b;
            --warn-soft: #fbf2e6;
            --bad: #992f2f;
            --bad-soft: #faecec;
            --neutral: #23425f;
            --neutral-soft: #e9eef4;
        }
        .stApp { background: linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%); color: var(--ink); }
        .block-container { max-width: 1380px; padding-top: 1.1rem; padding-bottom: 2.2rem; }
        .hero, .panel, .metric-card, .status-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }
        .hero { padding: 18px 20px; margin-bottom: 10px; display: flex; justify-content: space-between; gap: 12px; align-items: start; }
        .hero-title { margin: 0; font-size: 1.72rem; letter-spacing: -0.01em; }
        .hero-subtitle { margin-top: 6px; color: var(--muted); max-width: 760px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 10px 0 14px; }
        .status-card { padding: 12px 14px; min-height: 88px; }
        .status-label { font-size: .72rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
        .status-value { margin-top: 6px; font-size: 1rem; font-weight: 800; line-height: 1.28; white-space: normal; word-break: break-word; }
        .tone-good { background: linear-gradient(180deg, var(--good-soft) 0%, #fff 100%); }
        .tone-warn { background: linear-gradient(180deg, var(--warn-soft) 0%, #fff 100%); }
        .tone-bad { background: linear-gradient(180deg, var(--bad-soft) 0%, #fff 100%); }
        .tone-neutral { background: linear-gradient(180deg, var(--neutral-soft) 0%, #fff 100%); }
        .section-head { margin: 14px 0 8px; }
        .section-title { font-size: .94rem; font-weight: 800; text-transform: uppercase; letter-spacing: .06em; }
        .section-subtitle { font-size: .88rem; color: var(--muted); margin-top: 3px; }
        .metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 12px; margin-bottom: 14px; }
        .metric-card { padding: 14px; min-height: 96px; }
        .metric-label { font-size: .74rem; font-weight: 800; text-transform: uppercase; color: var(--muted); letter-spacing: .08em; }
        .metric-value { margin-top: 8px; font-size: 1.38rem; font-weight: 800; line-height: 1.2; white-space: normal; word-break: break-word; }
        .metric-sub { margin-top: 5px; color: var(--muted); font-size: .82rem; line-height: 1.35; }
        .detail-panel { padding: 14px; }
        .detail-grid { display: grid; grid-template-columns: 150px 1fr; gap: 8px 12px; }
        .detail-k { font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 800; }
        .detail-v { word-break: break-word; line-height: 1.4; }
        .panel { padding: 14px; margin-bottom: 12px; }
        [data-testid="stSidebar"] { min-width: 300px; }
        @media (max-width: 1024px) {
            .hero { flex-direction: column; }
            .detail-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='section-head'><div class='section-title'>{html.escape(title)}</div><div class='section-subtitle'>{html.escape(subtitle)}</div></div>",
        unsafe_allow_html=True,
    )


def render_metric_cards(metrics: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, subtitle in metrics:
        cards.append(
            f"<div class='metric-card'><div class='metric-label'>{html.escape(label)}</div><div class='metric-value'>{html.escape(value)}</div><div class='metric-sub'>{html.escape(subtitle)}</div></div>"
        )
    st.markdown("<div class='metric-row'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_status_cards(items: list[tuple[str, str, str]]) -> None:
    cards = []
    for label, value, tone in items:
        cards.append(
            f"<div class='status-card tone-{tone}'><div class='status-label'>{html.escape(label)}</div><div class='status-value'>{html.escape(value)}</div></div>"
        )
    st.markdown("<div class='status-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_detail_panel(title: str, rows: list[tuple[str, Any]]) -> None:
    body = []
    for label, value in rows:
        body.append(f"<div class='detail-k'>{html.escape(str(label))}</div><div class='detail-v'>{html.escape(str(value or 'Unavailable'))}</div>")
    st.markdown(
        f"<div class='panel detail-panel'><div class='section-title'>{html.escape(title)}</div><div class='detail-grid'>{''.join(body)}</div></div>",
        unsafe_allow_html=True,
    )


def format_timestamp(value: Any) -> str:
    if not value:
        return "Unavailable"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text


def shorten_path(value: Any, keep_parts: int = 3) -> str:
    if not value:
        return "Unavailable"
    parts = Path(str(value)).parts
    if len(parts) <= keep_parts + 1:
        return str(value)
    return str(Path("...", *parts[-keep_parts:]))


def status_tone(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"connected", "live", "success", "succeeded", "yes", "high", "good"}:
        return "good"
    if text in {"mixed", "warning", "warn", "medium", "mock"}:
        return "warn"
    if text in {"failed", "error", "bad", "no"}:
        return "bad"
    return "neutral"


def preview_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    subset = df.copy()
    for column in columns:
        if column not in subset.columns:
            subset[column] = ""
    return subset[columns]
