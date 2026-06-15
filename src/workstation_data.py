"""File-backed helpers for the local shortlist workstation."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SHORTLIST_DEFAULTS = {
    "rank": 0,
    "list_type": "unassigned",
    "symbol": "",
    "ticker": "",
    "last_price": 0.0,
    "current_price": 0.0,
    "bid_price": 0.0,
    "ask_price": 0.0,
    "gap_pct": 0.0,
    "move_percent": 0.0,
    "volume_today": 0.0,
    "volume": 0.0,
    "relative_volume": 0.0,
    "spread_pct": 0.0,
    "dollar_volume": 0.0,
    "trade_count_recent": 0.0,
    "catalyst": "No catalyst available",
    "catalyst_type": "none",
    "sec_filing_type": "",
    "liquidity_flag": "unknown",
    "setup_type": "unknown",
    "risk_note": "Manual review required",
    "score": 0.0,
    "total_score": 0.0,
    "ml_score": 0.0,
    "breakout_probability": 0.0,
    "failure_probability": 0.0,
    "news_score": 0.0,
    "liquidity_score": 0.0,
    "setup_score": 0.0,
    "risk_score": 0.0,
    "status_tag": "Ignore",
    "confidence": "low",
    "last_update_time": "",
    "quote_time": "",
    "selection_reason": "",
    "action_note": "",
    "tradeable": False,
    "tradeable_reason": "",
    "not_tradeable_reason": "",
    "base_low": 0.0,
    "base_high": 0.0,
    "predicted_upper_band": 0.0,
    "predicted_lower_band": 0.0,
    "breakout_low": 0.0,
    "breakout_high": 0.0,
    "pullback_low": 0.0,
    "pullback_high": 0.0,
    "invalidation": 0.0,
    "vwap": 0.0,
    "news_count_30m": 0.0,
    "freshness_minutes": 0.0,
    "headline_strength": 0.0,
    "source_count": 0.0,
    "ret_1m": 0.0,
    "ret_5m": 0.0,
    "ret_15m": 0.0,
    "volatility_1m": 0.0,
    "range_5m": 0.0,
    "range_15m": 0.0,
    "distance_to_vwap_pct": 0.0,
    "distance_to_intraday_high_pct": 0.0,
    "distance_to_premarket_high_pct": 0.0,
    "breakout_ready_score": 0.0,
    "volatility_5m": 0.0,
    "volatility_15m": 0.0,
    "overextension_score": 0.0,
    "wickiness_score": 0.0,
    "halt_risk_proxy": 0.0,
    "index_regime": "unknown",
    "market_trend_strength": 0.0,
    "sector_strength": 0.0,
    "breadth_proxy": 0.0,
    "vwap_regime_flag": 0.0,
    "setup_bias": "neutral",
}
RUN_STATUS_DEFAULTS = {
    "success": None,
    "run_mode": "unknown",
    "generated_at": None,
    "api_probe_time": None,
    "row_count": 0,
    "source_file": "",
    "latest_file": "",
    "history_file": "",
    "report_file": "",
    "activity_file": "",
    "context_file": "",
    "db_file": "",
    "alpaca_status": "unavailable",
    "benzinga_status": "unavailable",
    "news_status": "unavailable",
    "sec_status": "unavailable",
    "alpaca_message": "No probe recorded.",
    "benzinga_message": "No probe recorded.",
    "news_message": "No probe recorded.",
    "sec_message": "No probe recorded.",
    "data_mode": "mock",
    "used_fallback_data": False,
    "candidate_count": 0,
    "news_items_ingested": 0,
    "ml_status": "fallback",
    "ml_message": "No model run recorded.",
    "error": None,
}
SETUP_LABELS = {
    "event_driven": "Event Driven",
    "trend_continuation": "Trend Continuation",
    "relative_strength": "Relative Strength",
    "news_momentum": "News Momentum",
    "reversal": "Reversal",
    "watchlist": "Watchlist",
    "speculative": "Speculative",
    "unknown": "Unknown",
}
LIST_LABELS = {
    "all": "All Lists",
    "short_term": "Short-Term",
    "mid_term": "Mid-Term",
    "unassigned": "Unassigned",
}
HIGH_SIGNAL_TERMS = {
    "earnings",
    "guidance",
    "contract",
    "fda",
    "deal",
    "delivery",
    "upgrade",
    "downgrade",
    "partnership",
    "acquisition",
    "merger",
    "approval",
    "launch",
    "analyst",
}
SETUP_BASE_SCORES = {
    "event_driven": 9.0,
    "trend_continuation": 8.5,
    "relative_strength": 8.0,
    "news_momentum": 7.5,
    "reversal": 6.5,
    "watchlist": 4.5,
    "speculative": 2.5,
    "unknown": 3.0,
}
OBSCURE_TICKERS = {"SOUN", "TOP", "HKD", "GNS"}


def get_data_paths(project_root: Path) -> dict[str, Path]:
    """Return the canonical file paths used by the workstation."""
    data_dir = project_root / "data"
    history_dir = data_dir / "history"
    return {
        "data_dir": data_dir,
        "history_dir": history_dir,
        "latest_shortlist": data_dir / "shortlist_latest.csv",
        "latest_context": data_dir / "latest_context.json",
        "run_status": data_dir / "run_status.json",
        "activity_log": data_dir / "activity_log.json",
        "db_file": data_dir / "shortlist.db",
        "sec_cache": data_dir / "sec_company_tickers.json",
        "legacy_shortlist": project_root / "outputs" / "daily_shortlist.csv",
        "legacy_report": project_root / "outputs" / "daily_shortlist_report.html",
    }


def ensure_data_directories(project_root: Path) -> dict[str, Path]:
    """Ensure the local data folders exist and return their paths."""
    paths = get_data_paths(project_root)
    paths["data_dir"].mkdir(parents=True, exist_ok=True)
    paths["history_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def make_history_path(history_dir: Path, generated_at: datetime) -> Path:
    """Return the dated history file path for a given run timestamp."""
    return history_dir / f"shortlist_{generated_at.date().isoformat()}.csv"


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically so partial writes do not corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically using a compact, readable format."""
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=True))


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe atomically to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f"{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def append_activity_event(
    activity_file: Path,
    event_type: str,
    title: str,
    message: str,
    status: str,
    related_file: str | None = None,
) -> None:
    """Append a simple event record to the activity log."""
    bundle = load_activity_log(activity_file)
    records = bundle["records"]
    records.append(
        {
            "timestamp": current_timestamp(),
            "type": str(event_type),
            "title": str(title),
            "message": str(message),
            "status": str(status),
            "related_file": str(related_file or ""),
        }
    )
    write_json_atomic(activity_file, records)



def write_run_status(status_file: Path, payload: dict[str, Any]) -> None:
    """Write the run status file with stable keys."""
    data = RUN_STATUS_DEFAULTS.copy()
    data.update(payload)
    write_json_atomic(status_file, data)


def write_context_snapshot(context_file: Path, payload: dict[str, Any]) -> None:
    """Write the latest shortlist context snapshot for the dashboard detail panel."""
    write_json_atomic(context_file, payload)



def load_run_status(status_file: Path) -> dict[str, Any]:
    """Load run_status.json safely and return a stable structure."""
    bundle = {
        "data": RUN_STATUS_DEFAULTS.copy(),
        "exists": status_file.exists(),
        "messages": [],
        "load_error": None,
        "path": status_file,
        "modified_at": _safe_mtime(status_file),
    }
    if not status_file.exists():
        bundle["messages"].append("run_status.json is not available yet.")
        return bundle

    try:
        payload = json.loads(status_file.read_text(encoding="utf-8"))
    except Exception as exc:
        bundle["load_error"] = f"Could not read run_status.json: {exc}"
        return bundle

    if not isinstance(payload, dict):
        bundle["load_error"] = "run_status.json is malformed and not an object."
        return bundle

    data = RUN_STATUS_DEFAULTS.copy()
    data.update(payload)
    data["used_fallback_data"] = _coerce_status_bool(data.get("used_fallback_data"))
    data["data_mode"] = _normalize_data_mode(data.get("data_mode"), data.get("run_mode"))
    data["alpaca_status"] = _normalize_provider_status(data.get("alpaca_status"))
    data["benzinga_status"] = _normalize_provider_status(data.get("benzinga_status"))
    data["news_status"] = _normalize_provider_status(data.get("news_status"))
    data["sec_status"] = _normalize_provider_status(data.get("sec_status"))
    data["alpaca_message"] = str(data.get("alpaca_message") or "No probe recorded.")
    data["benzinga_message"] = str(data.get("benzinga_message") or "No probe recorded.")
    data["news_message"] = str(data.get("news_message") or "No probe recorded.")
    data["sec_message"] = str(data.get("sec_message") or "No probe recorded.")
    data["ml_status"] = str(data.get("ml_status") or "fallback")
    data["ml_message"] = str(data.get("ml_message") or "No model run recorded.")
    if not data.get("api_probe_time"):
        data["api_probe_time"] = data.get("generated_at")
    bundle["data"] = data
    return bundle



def load_activity_log(activity_file: Path) -> dict[str, Any]:
    """Load the activity log safely and return stable records."""
    bundle = {
        "records": [],
        "exists": activity_file.exists(),
        "messages": [],
        "load_error": None,
        "path": activity_file,
        "modified_at": _safe_mtime(activity_file),
    }
    if not activity_file.exists():
        bundle["messages"].append("activity_log.json is not available yet.")
        return bundle

    try:
        payload = json.loads(activity_file.read_text(encoding="utf-8"))
    except Exception as exc:
        bundle["load_error"] = f"Could not read activity_log.json: {exc}"
        return bundle

    if not isinstance(payload, list):
        bundle["load_error"] = "activity_log.json is malformed and not a list."
        return bundle

    records = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "timestamp": str(item.get("timestamp") or ""),
                "type": str(item.get("type") or "info"),
                "title": str(item.get("title") or "Untitled event"),
                "message": str(item.get("message") or ""),
                "status": str(item.get("status") or "info"),
                "related_file": str(item.get("related_file") or ""),
            }
        )

    records.sort(key=lambda record: record.get("timestamp", ""), reverse=True)
    bundle["records"] = records
    return bundle


def load_context_snapshot(context_file: Path) -> dict[str, Any]:
    """Load latest_context.json safely for the dashboard detail panel."""
    bundle = {
        "data": {"generated_at": None, "symbols": {}, "news_items": 0, "candidate_count": 0},
        "exists": context_file.exists(),
        "messages": [],
        "load_error": None,
        "path": context_file,
        "modified_at": _safe_mtime(context_file),
    }
    if not context_file.exists():
        bundle["messages"].append("latest_context.json is not available yet.")
        return bundle
    try:
        payload = json.loads(context_file.read_text(encoding="utf-8"))
    except Exception as exc:
        bundle["load_error"] = f"Could not read latest_context.json: {exc}"
        return bundle
    if not isinstance(payload, dict):
        bundle["load_error"] = "latest_context.json is malformed and not an object."
        return bundle
    payload.setdefault("generated_at", None)
    payload.setdefault("symbols", {})
    payload.setdefault("news_items", 0)
    payload.setdefault("candidate_count", 0)
    bundle["data"] = payload
    return bundle



def load_latest_shortlist(project_root: Path, preferred_path: str | Path | None = None) -> dict[str, Any]:
    """Load the latest shortlist, falling back to the legacy output if needed."""
    paths = get_data_paths(project_root)
    candidates: list[Path] = []
    if preferred_path:
        candidates.append(Path(preferred_path))
    candidates.extend([paths["latest_shortlist"], paths["legacy_shortlist"]])

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else project_root / candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        bundle = load_shortlist_file(resolved)
        if bundle["exists"] and not bundle["load_error"]:
            if resolved != paths["latest_shortlist"]:
                bundle["messages"].append(f"Using fallback shortlist file: {resolved}")
            return bundle

    return load_shortlist_file(paths["latest_shortlist"])



def load_shortlist_file(csv_path: Path) -> dict[str, Any]:
    """Load and normalize a shortlist CSV safely."""
    bundle = {
        "data": empty_shortlist_frame(),
        "exists": csv_path.exists(),
        "messages": [],
        "load_error": None,
        "row_count": 0,
        "path": csv_path,
        "modified_at": _safe_mtime(csv_path),
    }
    if not csv_path.exists():
        bundle["messages"].append(f"Shortlist file is not available: {csv_path}")
        return bundle

    try:
        raw_df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        bundle["messages"].append(f"Shortlist file is empty: {csv_path}")
        return bundle
    except Exception as exc:
        bundle["load_error"] = f"Could not read shortlist CSV: {exc}"
        return bundle

    normalized, messages = normalize_shortlist(raw_df)
    bundle["data"] = normalized
    bundle["messages"].extend(messages)
    bundle["row_count"] = len(normalized)
    return bundle



def load_history_index(history_dir: Path) -> dict[str, Any]:
    """Return a lightweight index of archived shortlist files."""
    bundle = {
        "entries": [],
        "exists": history_dir.exists(),
        "messages": [],
        "load_error": None,
        "path": history_dir,
    }
    if not history_dir.exists():
        bundle["messages"].append("History directory does not exist yet.")
        return bundle

    files = sorted(history_dir.glob("shortlist_*.csv"), reverse=True)
    if not files:
        bundle["messages"].append("No archived shortlist files found yet.")
        return bundle

    entries = []
    for csv_path in files:
        shortlist_bundle = load_shortlist_file(csv_path)
        df = shortlist_bundle["data"]
        top_ticker = ""
        top_score = 0.0
        if not df.empty:
            score_column = "total_score" if "total_score" in df.columns else "score"
            top_row = df.sort_values([score_column, "news_score", "move_percent"], ascending=False).iloc[0]
            top_ticker = str(top_row.get("ticker") or "")
            top_score = float(top_row.get(score_column) or top_row.get("score") or 0.0)

        entries.append(
            {
                "file_name": csv_path.name,
                "path": csv_path,
                "generated_at": _format_timestamp(_safe_mtime(csv_path)),
                "row_count": len(df),
                "top_ticker": top_ticker,
                "top_score": round(top_score, 2),
                "load_error": shortlist_bundle["load_error"],
            }
        )

    bundle["entries"] = entries
    return bundle



def load_history_file(history_dir: Path, date_or_path: str | Path | None) -> dict[str, Any]:
    """Load one archived shortlist file by file name, date label, or path."""
    if not date_or_path:
        return load_shortlist_file(history_dir / "shortlist_missing.csv")

    if isinstance(date_or_path, Path):
        return load_shortlist_file(date_or_path)

    text = str(date_or_path).strip()
    candidate = Path(text)
    if candidate.is_absolute() or candidate.suffix == ".csv":
        if not candidate.is_absolute():
            candidate = history_dir / candidate.name
        return load_shortlist_file(candidate)

    return load_shortlist_file(history_dir / f"shortlist_{text}.csv")



def get_file_summary(path: Path) -> dict[str, Any]:
    """Return a safe summary and preview for a file used by the workstation."""
    summary = {
        "path": path,
        "exists": path.exists(),
        "modified_at": _format_timestamp(_safe_mtime(path)),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "row_count": None,
        "preview_df": None,
        "preview_json": None,
        "load_error": None,
        "kind": path.suffix.lower().lstrip("."),
    }
    if not path.exists():
        return summary

    if path.suffix.lower() == ".csv":
        bundle = load_shortlist_file(path)
        summary["row_count"] = bundle["row_count"]
        summary["preview_df"] = bundle["data"].head(5)
        summary["load_error"] = bundle["load_error"]
        return summary

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            summary["load_error"] = f"Could not read JSON: {exc}"
            return summary

        if isinstance(payload, list):
            summary["row_count"] = len(payload)
            summary["preview_json"] = payload[:5]
        elif isinstance(payload, dict):
            summary["row_count"] = len(payload)
            if isinstance(payload.get("symbols"), dict) and len(payload["symbols"]) > 2:
                preview_symbols = dict(list(payload["symbols"].items())[:2])
                preview_payload = payload.copy()
                preview_payload["symbols"] = preview_symbols
                summary["preview_json"] = preview_payload
            else:
                summary["preview_json"] = payload
        else:
            summary["preview_json"] = payload
        return summary

    try:
        summary["preview_json"] = path.read_text(encoding="utf-8")[:600]
    except Exception as exc:
        summary["load_error"] = f"Could not read file: {exc}"
    return summary



def normalize_shortlist(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Normalize shortlist data to a stable schema for the app."""
    messages: list[str] = []
    if df.empty:
        messages.append("Shortlist file contained no rows.")
        return empty_shortlist_frame(), messages

    normalized = df.copy()
    missing_columns = [column for column in SHORTLIST_DEFAULTS if column not in normalized.columns]
    for column in missing_columns:
        normalized[column] = SHORTLIST_DEFAULTS[column]
    if missing_columns:
        messages.append(
            "Missing shortlist columns were backfilled: " + ", ".join(sorted(missing_columns))
        )

    normalized["symbol"] = normalized["symbol"].map(lambda value: str(value or "").upper().strip())
    normalized["ticker"] = normalized["ticker"].map(lambda value: str(value or "").upper().strip())
    normalized["ticker"] = normalized["ticker"].where(
        normalized["ticker"].astype(str).str.strip() != "",
        normalized["symbol"],
    )
    normalized["symbol"] = normalized["symbol"].where(
        normalized["symbol"].astype(str).str.strip() != "",
        normalized["ticker"],
    )
    normalized = normalized[normalized["ticker"] != ""].copy()

    normalized["list_type"] = normalized["list_type"].map(_normalize_list_type)
    normalized["setup_type"] = normalized["setup_type"].map(_normalize_setup_type)
    normalized["liquidity_flag"] = normalized["liquidity_flag"].map(_normalize_liquidity_flag)
    normalized["catalyst"] = _clean_text_series(normalized["catalyst"], SHORTLIST_DEFAULTS["catalyst"])
    normalized["risk_note"] = _clean_text_series(normalized["risk_note"], SHORTLIST_DEFAULTS["risk_note"])
    normalized["tradeable_reason"] = _clean_text_series(normalized["tradeable_reason"], "")
    normalized["not_tradeable_reason"] = _clean_text_series(normalized["not_tradeable_reason"], "")

    numeric_columns = [
        "rank",
        "last_price",
        "current_price",
        "bid_price",
        "ask_price",
        "gap_pct",
        "move_percent",
        "volume_today",
        "volume",
        "relative_volume",
        "spread_pct",
        "dollar_volume",
        "trade_count_recent",
        "score",
        "total_score",
        "ml_score",
        "breakout_probability",
        "failure_probability",
        "news_score",
        "liquidity_score",
        "setup_score",
        "risk_score",
        "base_low",
        "base_high",
        "predicted_upper_band",
        "predicted_lower_band",
        "breakout_low",
        "breakout_high",
        "pullback_low",
        "pullback_high",
        "invalidation",
        "vwap",
        "news_count_30m",
        "freshness_minutes",
        "headline_strength",
        "source_count",
        "ret_1m",
        "ret_5m",
        "ret_15m",
        "volatility_1m",
        "range_5m",
        "range_15m",
        "distance_to_vwap_pct",
        "distance_to_intraday_high_pct",
        "distance_to_premarket_high_pct",
        "breakout_ready_score",
        "volatility_5m",
        "volatility_15m",
        "overextension_score",
        "wickiness_score",
        "halt_risk_proxy",
        "market_trend_strength",
        "sector_strength",
        "breadth_proxy",
        "vwap_regime_flag",
    ]
    for column in numeric_columns:
        normalized[column] = _coerce_numeric_series(normalized[column])

    normalized["tradeable"] = normalized["tradeable"].map(_coerce_bool)

    fallback_news = normalized.apply(_fallback_news_score, axis=1)
    normalized["news_score"] = normalized["news_score"].where(normalized["news_score"].notna(), fallback_news)

    fallback_liquidity = normalized.apply(_fallback_liquidity_score, axis=1)
    normalized["liquidity_score"] = normalized["liquidity_score"].where(
        normalized["liquidity_score"].notna(), fallback_liquidity
    )

    fallback_setup = normalized.apply(_fallback_setup_score, axis=1)
    normalized["setup_score"] = normalized["setup_score"].where(normalized["setup_score"].notna(), fallback_setup)

    fallback_tradeable = normalized.apply(_fallback_tradeable_flag, axis=1)
    normalized["tradeable"] = normalized["tradeable"].where(normalized["tradeable"].notna(), fallback_tradeable)

    fallback_score = normalized.apply(_fallback_total_score, axis=1)
    normalized["score"] = normalized["score"].where(normalized["score"].notna(), normalized["total_score"])
    normalized["score"] = normalized["score"].where(normalized["score"].notna(), fallback_score)
    normalized["total_score"] = normalized["total_score"].where(normalized["total_score"].notna(), normalized["score"])

    normalized[["current_price", "move_percent", "score", "total_score", "ml_score", "breakout_probability", "failure_probability", "news_score", "liquidity_score", "setup_score", "risk_score"]] = (
        normalized[["current_price", "move_percent", "score", "total_score", "ml_score", "breakout_probability", "failure_probability", "news_score", "liquidity_score", "setup_score", "risk_score"]]
        .fillna(0.0)
        .astype(float)
    )
    normalized["tradeable"] = normalized["tradeable"].fillna(False).astype(bool)

    explanation_frame = normalized.apply(_tradeability_explanations, axis=1, result_type="expand")
    explanation_frame.columns = ["fallback_tradeable_reason", "fallback_not_tradeable_reason"]

    normalized["tradeable_reason"] = normalized["tradeable_reason"].where(
        normalized["tradeable_reason"].astype(str).str.strip() != "",
        explanation_frame["fallback_tradeable_reason"],
    )
    normalized["not_tradeable_reason"] = normalized["not_tradeable_reason"].where(
        normalized["not_tradeable_reason"].astype(str).str.strip() != "",
        explanation_frame["fallback_not_tradeable_reason"],
    )

    normalized["setup_display"] = normalized["setup_type"].map(
        lambda value: SETUP_LABELS.get(value, value.replace("_", " ").title())
    )
    normalized["list_display"] = normalized["list_type"].map(
        lambda value: LIST_LABELS.get(value, value.replace("_", " ").title())
    )
    normalized["liquidity_display"] = normalized["liquidity_flag"].map(
        lambda value: value.replace("_", " ").title()
    )

    return normalized.reset_index(drop=True), messages



def empty_shortlist_frame() -> pd.DataFrame:
    """Return an empty dataframe with the normalized shortlist columns."""
    columns = list(SHORTLIST_DEFAULTS) + ["setup_display", "list_display", "liquidity_display"]
    return pd.DataFrame(columns=columns)



def current_timestamp() -> str:
    """Return a stable timestamp string for logs and status files."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")



def _clean_text_series(series: pd.Series, default: str) -> pd.Series:
    """Normalize text columns and replace obvious null markers."""
    return (
        series.fillna(default)
        .astype(str)
        .replace({"nan": default, "None": default, "<NA>": default})
        .map(lambda value: value.strip())
    )



def _normalize_list_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"short_term", "mid_term"}:
        return text
    return "unassigned"



def _normalize_setup_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or "unknown"



def _normalize_liquidity_flag(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or "unknown"


def _normalize_provider_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"connected", "failed", "unavailable", "skipped"}:
        return text
    return "unavailable"


def _normalize_data_mode(value: Any, run_mode: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"live", "mixed", "mock"}:
        return text
    return "mock" if str(run_mode or "").strip().lower() == "mock" else "mixed"


def _coerce_status_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return bool(value)



def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")



def _coerce_bool(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "tradeable"}:
        return True
    if text in {"false", "0", "no", "n", "review"}:
        return False
    return pd.NA



def _fallback_news_score(row: pd.Series) -> float:
    catalyst = str(row.get("catalyst") or "").lower()
    if not catalyst or catalyst == SHORTLIST_DEFAULTS["catalyst"].lower():
        return 1.0
    score = 2.0 + min(sum(term in catalyst for term in HIGH_SIGNAL_TERMS) * 1.2, 4.0)
    if row.get("setup_type") in {"event_driven", "news_momentum"}:
        score += 1.5
    if abs(_safe_float(row.get("move_percent"))) >= 4:
        score += 1.0
    return round(min(score, 10.0), 2)



def _fallback_liquidity_score(row: pd.Series) -> float:
    score = 0.0
    if row.get("liquidity_flag") == "liquid":
        score += 4.0
    price = _safe_float(row.get("current_price"))
    if 10 <= price <= 250:
        score += 3.0
    elif 5 <= price <= 500:
        score += 2.0
    move_percent = abs(_safe_float(row.get("move_percent")))
    if move_percent >= 5:
        score += 2.0
    elif move_percent >= 2:
        score += 1.0
    if row.get("tradeable") is True:
        score += 1.0
    return round(min(score, 10.0), 2)



def _fallback_setup_score(row: pd.Series) -> float:
    setup_type = _normalize_setup_type(row.get("setup_type"))
    score = SETUP_BASE_SCORES.get(setup_type, SETUP_BASE_SCORES["unknown"])
    if abs(_safe_float(row.get("move_percent"))) >= 4 and setup_type not in {"unknown", "speculative"}:
        score += 0.5
    return round(min(score, 10.0), 2)



def _fallback_tradeable_flag(row: pd.Series) -> bool:
    ticker = str(row.get("ticker") or "").upper().strip()
    price = _safe_float(row.get("current_price"))
    setup_type = _normalize_setup_type(row.get("setup_type"))
    liquidity_flag = row.get("liquidity_flag")
    valid_symbol = ticker.isalpha() and 1 <= len(ticker) <= 5 and ticker not in OBSCURE_TICKERS
    valid_setup = setup_type not in {"unknown", "watchlist", "speculative"}
    valid_price = 7 <= price <= 500
    return bool(liquidity_flag == "liquid" and valid_symbol and valid_setup and valid_price)



def _fallback_total_score(row: pd.Series) -> float:
    move_component = min(abs(_safe_float(row.get("move_percent"))), 10.0)
    score = (
        0.40 * _safe_float(row.get("news_score"))
        + 0.35 * _safe_float(row.get("liquidity_score"))
        + 0.25 * _safe_float(row.get("setup_score"))
        + 0.10 * move_component
    )
    return round(min(score, 10.0), 2)



def _tradeability_explanations(row: pd.Series) -> tuple[str, str]:
    positives: list[str] = []
    negatives: list[str] = []

    if row.get("liquidity_flag") == "liquid":
        positives.append("liquidity ok")
    else:
        negatives.append("low liquidity")

    price = _safe_float(row.get("current_price"))
    if 7 <= price <= 500:
        positives.append("price in range")
    else:
        negatives.append("price outside preferred range")

    setup_type = _normalize_setup_type(row.get("setup_type"))
    if setup_type not in {"unknown", "watchlist", "speculative"}:
        positives.append("setup acceptable")
    else:
        negatives.append("weak setup")

    ticker = str(row.get("ticker") or "").upper().strip()
    if ticker.isalpha() and 1 <= len(ticker) <= 5 and ticker not in OBSCURE_TICKERS:
        positives.append("ticker quality acceptable")
    else:
        negatives.append("suspicious ticker quality")

    if abs(_safe_float(row.get("move_percent"))) >= 2:
        positives.append("meaningful price move")

    if bool(row.get("tradeable")):
        return ", ".join(positives) or "passes basic tradeability checks", "no blocking issues found"
    return ", ".join(positives) or "limited positives", ", ".join(negatives) or "failed tradeability checks"



def _safe_mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None



def _format_timestamp(value: datetime | None) -> str:
    if not value:
        return "Unavailable"
    return value.strftime("%Y-%m-%d %H:%M:%S")



def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
