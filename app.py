"""Streamlit workstation for intraday shortlist analysis."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config.settings import load_settings
from src.project_env import load_project_env
from src.storage.db import load_news_count_today, load_recent_news_events
from src.ui.dashboard import (
    format_timestamp,
    inject_styles,
    preview_frame,
    render_detail_panel,
    render_metric_cards,
    render_section_header,
    render_status_cards,
    shorten_path,
    status_tone,
)
from src.ui.detail_panel import render_candidate_detail
from src.ui.ml_panel import render_ml_summary
from src.ui.replay_page import render_replay_page
from src.workstation_data import (
    append_activity_event,
    ensure_data_directories,
    get_file_summary,
    load_activity_log,
    load_context_snapshot,
    load_history_file,
    load_history_index,
    load_latest_shortlist,
    load_run_status,
)


PROJECT_ROOT = Path(__file__).resolve().parent
PAGES = ["Dashboard", "Shortlist", "Replay", "History", "Activity", "Files"]
PAGE_DESCRIPTIONS = {
    "Dashboard": "Pre-market and intraday command center for status, candidates, zones, and catalysts.",
    "Shortlist": "Current candidate pool with filtering, sorting, and single-symbol inspection.",
    "Replay": "Replay hit rates, model validation, and historical outcome summaries.",
    "History": "Preview archived shortlist outputs.",
    "Activity": "Recent run, probe, refresh, and archive events.",
    "Files": "Current local outputs, status files, and sqlite index.",
}
SORT_OPTIONS = {
    "total_score": "Total Score",
    "ml_score": "ML Score",
    "breakout_probability": "Breakout Probability",
    "news_score": "News Score",
    "setup_score": "Setup Score",
    "liquidity_score": "Liquidity Score",
    "gap_pct": "Gap %",
    "relative_volume": "Relative Volume",
}

st.set_page_config(page_title="Daily Shortlist Workstation", layout="wide", page_icon="DS")


def render_page() -> None:
    inject_styles()
    _init_state()
    runtime = load_runtime_config()
    workspace = load_workspace(runtime)
    page = render_shell(workspace)
    render_refresh_feedback()

    if page == "Dashboard":
        render_dashboard_page(workspace)
    elif page == "Shortlist":
        render_shortlist_page(workspace)
    elif page == "Replay":
        render_replay_page(workspace["db_path"])
    elif page == "History":
        render_history_page(workspace)
    elif page == "Activity":
        render_activity_page(workspace)
    else:
        render_files_page(workspace)


def _init_state() -> None:
    if st.session_state.get("workspace_page") == "Overview":
        st.session_state["workspace_page"] = "Dashboard"
    st.session_state.setdefault("workspace_page", "Dashboard")
    st.session_state.setdefault("refresh_state", None)
    st.session_state.setdefault("selected_symbol", None)
    st.session_state.setdefault("show_tradeable_only", False)
    st.session_state.setdefault("min_price", 5.0)
    st.session_state.setdefault("max_price", 500.0)
    st.session_state.setdefault("min_relative_volume", 1.0)
    st.session_state.setdefault("max_spread_pct", 1.0)
    st.session_state.setdefault("minimum_total_score", 0.0)
    st.session_state.setdefault("catalyst_filter", "all")
    st.session_state.setdefault("setup_filter", "all")
    st.session_state.setdefault("sort_by", "total_score")


def load_runtime_config() -> dict[str, Any]:
    config = load_project_env(PROJECT_ROOT)
    settings = load_settings(PROJECT_ROOT)
    key_text = str(config["alpaca_api_key"] or "").strip().upper()
    return {
        "mock_mode": bool(config["mock_mode"]),
        "mode_label": "MOCK MODE" if bool(config["mock_mode"]) else "LIVE MODE",
        "alpaca_configured": bool(config["alpaca_api_key"]) and bool(config["alpaca_api_secret"]),
        "alpaca_credential_hint": "broker_like" if key_text.startswith("C") else ("configured" if key_text else "missing"),
        "source_output": str(config["output_csv"]),
        "settings": settings,
    }


def load_workspace(runtime: dict[str, Any]) -> dict[str, Any]:
    paths = ensure_data_directories(PROJECT_ROOT)
    run_status_bundle = load_run_status(paths["run_status"])
    latest_bundle = load_latest_shortlist(PROJECT_ROOT, run_status_bundle["data"].get("latest_file") or paths["latest_shortlist"])
    history_bundle = load_history_index(paths["history_dir"])
    activity_bundle = load_activity_log(paths["activity_log"])
    context_bundle = load_context_snapshot(paths["latest_context"])
    db_path = runtime["settings"].db_path
    news_count_today = load_news_count_today(db_path) if db_path.exists() else 0
    recent_news = load_recent_news_events(db_path, 30) if db_path.exists() else pd.DataFrame()
    return {
        "runtime": runtime,
        "paths": paths,
        "run_status": run_status_bundle,
        "latest": latest_bundle,
        "history": history_bundle,
        "activity": activity_bundle,
        "context": context_bundle,
        "db_path": db_path,
        "news_count_today": news_count_today,
        "recent_news": recent_news,
    }


def render_shell(workspace: dict[str, Any]) -> str:
    run_status = workspace["run_status"]["data"]
    current_page = st.session_state.get("workspace_page", "Dashboard")
    with st.sidebar:
        if st.button("Rerun main.py and refresh", width="stretch"):
            with st.spinner("Running main.py..."):
                st.session_state["refresh_state"] = run_engine()
            st.rerun()
        st.caption(f"Data root: {workspace['paths']['data_dir']}")
        st.caption(f"Latest run: {format_timestamp(run_status.get('generated_at'))}")
        if current_page in {"Dashboard", "Shortlist"}:
            st.markdown("---")
            render_shortlist_filters(workspace["latest"]["data"])

    st.markdown(
        f"<div class='hero'><div><h1 class='hero-title'>Daily Shortlist Workstation</h1><div class='hero-subtitle'>{PAGE_DESCRIPTIONS[current_page]}</div></div><div><strong>{workspace['runtime']['mode_label']}</strong></div></div>",
        unsafe_allow_html=True,
    )
    page = st.radio("Workspace", options=PAGES, key="workspace_page", horizontal=True, label_visibility="collapsed")
    st.caption(PAGE_DESCRIPTIONS[page])
    if workspace["runtime"]["alpaca_credential_hint"] == "broker_like" and not workspace["runtime"]["mock_mode"]:
        st.warning("The current .env appears to contain broker-style Alpaca credentials. This project only supports Trading API / Market Data API header authentication, so please replace ALPACA_API_KEY and ALPACA_API_SECRET with Trading API credentials.")
    return page


def render_shortlist_filters(latest_df: pd.DataFrame) -> None:
    catalyst_options = sorted(set(latest_df.get("catalyst_type", pd.Series(dtype="object")).dropna().astype(str).tolist()))
    setup_options = ["all", "Breakout Watch", "Pullback Watch"]
    st.session_state["show_tradeable_only"] = st.toggle("show only tradeable ideas", value=bool(st.session_state["show_tradeable_only"]))
    st.session_state["min_price"] = st.number_input("min price", min_value=0.0, value=float(st.session_state["min_price"]), step=1.0)
    st.session_state["max_price"] = st.number_input("max price", min_value=0.0, value=float(st.session_state["max_price"]), step=1.0)
    st.session_state["min_relative_volume"] = st.number_input("min relative volume", min_value=0.0, value=float(st.session_state["min_relative_volume"]), step=0.1)
    st.session_state["max_spread_pct"] = st.number_input("max spread %", min_value=0.0, value=float(st.session_state["max_spread_pct"]), step=0.1)
    st.session_state["minimum_total_score"] = st.number_input("minimum total score", min_value=0.0, value=float(st.session_state["minimum_total_score"]), step=0.25)
    catalyst_values = ["all"] + catalyst_options if catalyst_options else ["all"]
    if st.session_state["catalyst_filter"] not in catalyst_values:
        st.session_state["catalyst_filter"] = "all"
    st.session_state["catalyst_filter"] = st.selectbox("catalyst type filter", options=catalyst_values)
    st.session_state["setup_filter"] = st.selectbox("breakout only / pullback only", options=setup_options)
    st.session_state["sort_by"] = st.selectbox("sort shortlist", options=list(SORT_OPTIONS), format_func=lambda value: SORT_OPTIONS[value])


def render_refresh_feedback() -> None:
    refresh_state = st.session_state.get("refresh_state")
    if not refresh_state:
        return
    if refresh_state["success"]:
        st.success("Refresh completed successfully.")
    else:
        st.error("Refresh failed. Existing data remains available.")
    if refresh_state.get("output"):
        with st.expander("Refresh log"):
            st.code(refresh_state["output"])


def render_dashboard_page(workspace: dict[str, Any]) -> None:
    run_status = workspace["run_status"]["data"]
    latest_df = workspace["latest"]["data"]
    render_status_cards([
        ("Configured Mode", workspace["runtime"]["mode_label"], status_tone("warn" if workspace["runtime"]["mock_mode"] else "good")),
        ("Data Mode", str(run_status.get("data_mode") or "mock").upper(), status_tone(run_status.get("data_mode") or "mock")),
        ("Alpaca", _provider_label(run_status.get("alpaca_status")), status_tone(run_status.get("alpaca_status") or "unavailable")),
        ("News", _provider_label(run_status.get("news_status")), status_tone(run_status.get("news_status") or "unavailable")),
        ("SEC RSS", _provider_label(run_status.get("sec_status")), status_tone(run_status.get("sec_status") or "unavailable")),
        ("Last Refresh", format_timestamp(run_status.get("generated_at")), "neutral"),
        ("Candidates", str(run_status.get("candidate_count") or workspace["latest"]["row_count"]), "neutral"),
        ("News Today", str(workspace["news_count_today"]), "neutral"),
    ])
    render_metric_cards([
        ("Total Ideas", str(len(latest_df)), "Current shortlist rows"),
        ("Tradeable Ideas", str(int(latest_df.get("tradeable", pd.Series(dtype="bool")).sum()) if not latest_df.empty else 0), "Rows passing hard tradeability rules"),
        ("Average Score", f"{float(latest_df['total_score'].mean()):.2f}" if not latest_df.empty and "total_score" in latest_df.columns else "0.00", "Mean total score"),
        ("High Score", str(int((pd.to_numeric(latest_df.get('total_score'), errors='coerce').fillna(0.0) >= 6.0).sum()) if not latest_df.empty else 0), "Score >= 6.0"),
        ("Breakout Watch", str(int((latest_df.get("status_tag", pd.Series(dtype="object")) == "Breakout Watch").sum()) if not latest_df.empty else 0), "Breakout-leaning structures"),
    ])
    render_ml_summary(run_status, latest_df)

    render_section_header("Top Candidates", "The highest-priority shortlist names, followed directly by single-symbol detail.")
    _render_candidate_workspace(workspace, limit=12)

    provider_left, provider_right = st.columns([1.0, 1.0])
    with provider_left:
        render_section_header("Provider Health", "Latest health status for market data, news, and SEC data sources.")
        render_detail_panel(
            "Providers",
            [
                ("Alpaca Market", f"{_provider_label(run_status.get('alpaca_status'))} | {run_status.get('alpaca_message') or 'Unavailable'}"),
                ("Alpaca News", f"{_provider_label(run_status.get('news_status'))} | {run_status.get('news_message') or 'Unavailable'}"),
                ("SEC RSS", f"{_provider_label(run_status.get('sec_status'))} | {run_status.get('sec_message') or 'Unavailable'}"),
                ("Probe Time", format_timestamp(run_status.get('api_probe_time'))),
                ("Fallback Used", "Yes" if run_status.get('used_fallback_data') else "No"),
                ("ML Status", f"{str(run_status.get('ml_status') or 'fallback').title()} | {run_status.get('ml_message') or 'No model metadata recorded.'}"),
            ],
        )
    with provider_right:
        render_section_header("Run Summary", "Latest output summary and local file write status.")
        render_detail_panel(
            "Outputs",
            [
                ("Run Mode", run_status.get("run_mode") or "unknown"),
                ("Data Mode", str(run_status.get("data_mode") or "mock").upper()),
                ("Latest File", shorten_path(run_status.get("latest_file"))),
                ("Context File", shorten_path(run_status.get("context_file"))),
                ("DB File", shorten_path(run_status.get("db_file"))),
                ("Error", run_status.get("error") or "None"),
            ],
        )

    news_left, news_right = st.columns([1.0, 1.0])
    with news_left:
        render_section_header("Recent News", "Most recently ingested Alpaca News and SEC events.")
        if workspace["recent_news"].empty:
            st.info("No news events have been successfully ingested in the current environment yet.")
        else:
            st.dataframe(workspace["recent_news"], hide_index=True, width="stretch")
    with news_right:
        render_section_header("Recent Activity", "Recent run, refresh, probe, and archive records.")
        activity = pd.DataFrame(workspace["activity"]["records"][:12])
        if activity.empty:
            st.info("No activity records are available yet.")
        else:
            activity["timestamp"] = activity["timestamp"].map(format_timestamp)
            st.dataframe(activity[["timestamp", "type", "status", "title", "message"]], hide_index=True, width="stretch")


def render_shortlist_page(workspace: dict[str, Any]) -> None:
    render_section_header("Shortlist", "Filter and sort the current candidates to focus on the best names to track.")
    run_status = workspace["run_status"]["data"]
    render_detail_panel(
        "Confidence",
        [
            ("Data Mode", str(run_status.get("data_mode") or "mock").upper()),
            ("Fallback Used", "Yes" if run_status.get("used_fallback_data") else "No"),
            ("Latest Run", format_timestamp(run_status.get("generated_at"))),
            ("Alpaca", _provider_label(run_status.get("alpaca_status"))),
            ("ML", str(run_status.get("ml_status") or "fallback").title()),
        ],
    )
    _render_candidate_workspace(workspace, limit=None)


def _render_candidate_workspace(workspace: dict[str, Any], limit: int | None) -> None:
    latest_df = workspace["latest"]["data"]
    filtered = apply_shortlist_filters(latest_df)
    if filtered.empty:
        st.info("No candidates currently match the active filters, or the latest run has not produced a usable shortlist yet.")
        return
    sort_by = st.session_state.get("sort_by", "total_score")
    filtered = filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)
    working_df = filtered.head(limit).reset_index(drop=True) if limit else filtered

    table = _display_shortlist_table(working_df)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Last Price": st.column_config.NumberColumn("Last Price", format="%.2f"),
            "Gap %": st.column_config.NumberColumn("Gap %", format="%.2f"),
            "Relative Volume": st.column_config.NumberColumn("Relative Volume", format="%.2f"),
            "Spread %": st.column_config.NumberColumn("Spread %", format="%.2f"),
            "News Score": st.column_config.NumberColumn("News Score", format="%.2f"),
            "Setup Score": st.column_config.NumberColumn("Setup Score", format="%.2f"),
            "Liquidity Score": st.column_config.NumberColumn("Liquidity Score", format="%.2f"),
            "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.2f"),
            "Total Score": st.column_config.NumberColumn("Total Score", format="%.2f"),
        },
    )

    symbols = working_df["symbol"].astype(str).tolist()
    if st.session_state["selected_symbol"] not in symbols:
        st.session_state["selected_symbol"] = symbols[0]
    st.selectbox("Inspect symbol", options=symbols, key="selected_symbol")
    row = working_df[working_df["symbol"] == st.session_state["selected_symbol"]].iloc[0].to_dict()
    symbol_context = (workspace["context"]["data"].get("symbols") or {}).get(str(row.get("symbol") or ""), {})
    render_candidate_detail(row, symbol_context)


def render_history_page(workspace: dict[str, Any]) -> None:
    render_section_header("History", "Preview archived shortlist files directly in the app.")
    entries = workspace["history"]["entries"]
    if not entries:
        st.info("No history files are available yet.")
        return
    history_index = pd.DataFrame(entries)
    st.dataframe(history_index[["file_name", "generated_at", "row_count", "top_ticker", "top_score"]], hide_index=True, width="stretch")
    selected = st.selectbox("Preview archive", options=[entry["file_name"] for entry in entries])
    preview = load_history_file(workspace["paths"]["history_dir"], selected)
    if preview["data"].empty:
        st.info("The selected archive is empty.")
    else:
        st.dataframe(_display_shortlist_table(preview["data"]), hide_index=True, width="stretch")


def render_activity_page(workspace: dict[str, Any]) -> None:
    render_section_header("Activity", "A quick view of what the system has been doing recently.")
    records = pd.DataFrame(workspace["activity"]["records"])
    if records.empty:
        st.info("No activity records are available yet.")
        return
    type_filter = st.multiselect("Filter type", options=sorted(records["type"].unique().tolist()))
    status_filter = st.multiselect("Filter status", options=sorted(records["status"].unique().tolist()))
    if type_filter:
        records = records[records["type"].isin(type_filter)]
    if status_filter:
        records = records[records["status"].isin(status_filter)]
    records["timestamp"] = records["timestamp"].map(format_timestamp)
    st.dataframe(records[["timestamp", "type", "status", "title", "message", "related_file"]], hide_index=True, width="stretch")


def render_files_page(workspace: dict[str, Any]) -> None:
    render_section_header("Files", "Safe index of the current shortlist artifacts, status JSON, context snapshot, and sqlite store.")
    run_status = workspace["run_status"]["data"]
    latest_history = Path(run_status.get("history_file")) if run_status.get("history_file") else None
    if latest_history is None and workspace["history"]["entries"]:
        latest_history = Path(workspace["history"]["entries"][0]["path"])
    targets = [
        ("Latest shortlist", workspace["paths"]["latest_shortlist"]),
        ("Latest context", workspace["paths"]["latest_context"]),
        ("Latest history file", latest_history or (workspace["paths"]["history_dir"] / "shortlist_missing.csv")),
        ("Run status", workspace["paths"]["run_status"]),
        ("Activity log", workspace["paths"]["activity_log"]),
        ("SQLite DB", workspace["db_path"]),
        ("HTML report", Path(run_status.get("report_file") or workspace["paths"]["legacy_report"])),
    ]
    for title, path in targets:
        with st.expander(title, expanded=title in {"Latest shortlist", "Run status"}):
            if path.suffix.lower() == ".db":
                exists = path.exists()
                render_detail_panel(title, [
                    ("Path", shorten_path(path)),
                    ("Exists", "Yes" if exists else "No"),
                    ("Size", f"{path.stat().st_size} bytes" if exists else "Unavailable"),
                ])
                continue
            summary = get_file_summary(path)
            render_detail_panel(title, [
                ("Path", shorten_path(summary["path"])),
                ("Exists", "Yes" if summary["exists"] else "No"),
                ("Modified", summary["modified_at"] or "Unavailable"),
                ("Rows", str(summary["row_count"]) if summary["row_count"] is not None else "Unavailable"),
            ])
            if summary["preview_df"] is not None:
                st.dataframe(summary["preview_df"].head(5), hide_index=True, width="stretch")
            elif summary["preview_json"] is not None:
                st.json(_safe_json(summary["preview_json"]))
            elif summary["load_error"]:
                st.error(summary["load_error"])


def apply_shortlist_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    filtered = df.copy()
    filtered = filtered[pd.to_numeric(filtered["last_price"], errors="coerce").fillna(0.0) >= float(st.session_state["min_price"])]
    filtered = filtered[pd.to_numeric(filtered["last_price"], errors="coerce").fillna(0.0) <= float(st.session_state["max_price"])]
    filtered = filtered[pd.to_numeric(filtered["relative_volume"], errors="coerce").fillna(0.0) >= float(st.session_state["min_relative_volume"])]
    filtered = filtered[pd.to_numeric(filtered["spread_pct"], errors="coerce").fillna(0.0) <= float(st.session_state["max_spread_pct"])]
    filtered = filtered[pd.to_numeric(filtered["total_score"], errors="coerce").fillna(0.0) >= float(st.session_state["minimum_total_score"])]
    if st.session_state["show_tradeable_only"]:
        filtered = filtered[filtered["tradeable"] == True]
    if st.session_state["catalyst_filter"] != "all":
        filtered = filtered[filtered["catalyst_type"] == st.session_state["catalyst_filter"]]
    if st.session_state["setup_filter"] in {"Breakout Watch", "Pullback Watch"}:
        filtered = filtered[filtered["status_tag"] == st.session_state["setup_filter"]]
    return filtered.reset_index(drop=True)


def _display_shortlist_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    if table.empty:
        return table
    table["catalyst_short"] = table["catalyst"].astype(str).str.slice(0, 80)
    table["risk_short"] = table["risk_note"].astype(str).str.slice(0, 60)
    table = preview_frame(table, [
        "rank",
        "symbol",
        "last_price",
        "gap_pct",
        "volume_today",
        "relative_volume",
        "spread_pct",
        "catalyst_short",
        "news_score",
        "setup_score",
        "liquidity_score",
        "risk_score",
        "ml_score",
        "breakout_probability",
        "failure_probability",
        "total_score",
        "status_tag",
        "tradeable",
        "last_update_time",
    ])
    table = table.rename(columns={
        "rank": "Rank",
        "symbol": "Symbol",
        "last_price": "Last Price",
        "gap_pct": "Gap %",
        "volume_today": "Volume",
        "relative_volume": "Relative Volume",
        "spread_pct": "Spread %",
        "catalyst_short": "Catalyst",
        "news_score": "News Score",
        "setup_score": "Setup Score",
        "liquidity_score": "Liquidity Score",
        "risk_score": "Risk Score",
        "ml_score": "ML Score",
        "breakout_probability": "Breakout Prob",
        "failure_probability": "Failure Prob",
        "total_score": "Total Score",
        "status_tag": "Status Tag",
        "tradeable": "Tradeable",
        "last_update_time": "Last Update",
    })
    table["Last Update"] = table["Last Update"].map(format_timestamp)
    for column in ["Breakout Prob", "Failure Prob"]:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0).map(lambda value: f"{value * 100:.1f}%")
    return table


def run_engine() -> dict[str, Any]:
    try:
        append_activity_event(
            PROJECT_ROOT / "data" / "activity_log.json",
            event_type="refresh",
            title="Dashboard refresh triggered",
            message="Streamlit requested a new shortlist run.",
            status="info",
            related_file=str(PROJECT_ROOT / "main.py"),
        )
    except Exception:
        pass
    try:
        result = subprocess.run([sys.executable, "main.py"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180, check=False)
        output = "\n".join([line for line in [result.stdout.strip(), result.stderr.strip()] if line])
        try:
            append_activity_event(
                PROJECT_ROOT / "data" / "activity_log.json",
                event_type="refresh",
                title="Dashboard refresh completed" if result.returncode == 0 else "Dashboard refresh failed",
                message=output or "Refresh process exited without output.",
                status="success" if result.returncode == 0 else "failed",
                related_file=str(PROJECT_ROOT / "main.py"),
            )
        except Exception:
            pass
        return {"success": result.returncode == 0, "output": output}
    except Exception as exc:
        try:
            append_activity_event(
                PROJECT_ROOT / "data" / "activity_log.json",
                event_type="refresh",
                title="Dashboard refresh failed",
                message=str(exc),
                status="failed",
                related_file=str(PROJECT_ROOT / "main.py"),
            )
        except Exception:
            pass
        return {"success": False, "output": str(exc)}


def _provider_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text == "connected":
        return "Connected"
    if text == "failed":
        return "Failed"
    if text == "skipped":
        return "Skipped"
    return "Unavailable"


def _safe_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        safe = {}
        for key, value in payload.items():
            key_text = str(key).lower()
            if any(token in key_text for token in ["key", "secret", "token", "password", "auth"]):
                safe[key] = "[REDACTED]"
            else:
                safe[key] = _safe_json(value)
        return safe
    if isinstance(payload, list):
        return [_safe_json(item) for item in payload]
    return payload


render_page()
