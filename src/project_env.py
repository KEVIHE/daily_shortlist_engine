"""Shared project-local environment loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


DEFAULTS = {
    "ALPACA_API_KEY": "",
    "ALPACA_API_SECRET": "",
    "ALPACA_BASE_URL": "https://data.alpaca.markets",
    "BENZINGA_API_KEY": "",
    "BENZINGA_BASE_URL": "https://api.benzinga.com/api/v2",
    "MOCK_MODE": "true",
    "MIN_PRICE": "5",
    "MAX_PRICE": "500",
    "MIN_DOLLAR_VOLUME": "5000000",
    "MIN_ABS_MOVE_PERCENT": "2",
    "MIN_RELATIVE_VOLUME": "1.2",
    "MAX_SPREAD_PCT": "1.0",
    "MINIMUM_TOTAL_SCORE": "4.0",
    "MOVERS_LIMIT": "40",
    "SHORTLIST_LIMIT": "20",
    "ANALYSIS_HISTORY_MINUTES": "90",
    "DAILY_LOOKBACK_DAYS": "10",
    "ALPACA_NEWS_LIMIT": "50",
    "SEC_FEED_LIMIT": "100",
    "SHORTLIST_DB": "shortlist.db",
    "SEC_TICKER_CACHE": "sec_company_tickers.json",
    "SEC_USER_AGENT": "DailyShortlistEngine/1.0 research@local",
    "WEIGHT_CATALYST": "0.25",
    "WEIGHT_SETUP": "0.25",
    "WEIGHT_LIQUIDITY": "0.20",
    "WEIGHT_RISK": "0.15",
    "WEIGHT_ML": "0.15",
    "OUTPUT_CSV": "outputs/daily_shortlist.csv",
    "OUTPUT_HTML": "outputs/daily_shortlist_report.html",
}
ALIASES = {
    "ALPACA_API_KEY": ["APCA_API_KEY_ID"],
    "ALPACA_API_SECRET": ["APCA_API_SECRET_KEY"],
}


def load_project_env(project_root: Path) -> dict[str, Any]:
    """Load the project-local .env file with predictable precedence.

    Precedence:
    1. Values present in project_root/.env
    2. Existing process environment variables
    3. Built-in defaults
    """
    env_path = project_root / ".env"
    file_values = dotenv_values(env_path) if env_path.exists() else {}

    def resolve(name: str) -> str:
        value = _resolve_from_mapping(file_values, name)
        if value != "":
            return value
        env_value = os.getenv(name)
        if _normalize_value(env_value) != "":
            return _normalize_value(env_value)
        for alias in ALIASES.get(name, []):
            alias_env = os.getenv(alias)
            if _normalize_value(alias_env) != "":
                return _normalize_value(alias_env)
        return str(DEFAULTS.get(name, ""))

    return {
        "env_path": env_path,
        "env_exists": env_path.exists(),
        "mock_mode_raw": resolve("MOCK_MODE"),
        "mock_mode": resolve("MOCK_MODE").strip().lower() == "true",
        "alpaca_api_key": resolve("ALPACA_API_KEY"),
        "alpaca_api_secret": resolve("ALPACA_API_SECRET"),
        "alpaca_base_url": resolve("ALPACA_BASE_URL"),
        "benzinga_api_key": resolve("BENZINGA_API_KEY"),
        "benzinga_base_url": resolve("BENZINGA_BASE_URL"),
        "min_price": float(resolve("MIN_PRICE")),
        "max_price": float(resolve("MAX_PRICE")),
        "min_dollar_volume": float(resolve("MIN_DOLLAR_VOLUME")),
        "min_abs_move_percent": float(resolve("MIN_ABS_MOVE_PERCENT")),
        "min_relative_volume": float(resolve("MIN_RELATIVE_VOLUME")),
        "max_spread_pct": float(resolve("MAX_SPREAD_PCT")),
        "minimum_total_score": float(resolve("MINIMUM_TOTAL_SCORE")),
        "movers_limit": int(resolve("MOVERS_LIMIT")),
        "shortlist_limit": int(resolve("SHORTLIST_LIMIT")),
        "analysis_history_minutes": int(resolve("ANALYSIS_HISTORY_MINUTES")),
        "daily_lookback_days": int(resolve("DAILY_LOOKBACK_DAYS")),
        "alpaca_news_limit": int(resolve("ALPACA_NEWS_LIMIT")),
        "sec_feed_limit": int(resolve("SEC_FEED_LIMIT")),
        "shortlist_db": resolve("SHORTLIST_DB"),
        "sec_ticker_cache": resolve("SEC_TICKER_CACHE"),
        "sec_user_agent": resolve("SEC_USER_AGENT"),
        "weight_catalyst": float(resolve("WEIGHT_CATALYST")),
        "weight_setup": float(resolve("WEIGHT_SETUP")),
        "weight_liquidity": float(resolve("WEIGHT_LIQUIDITY")),
        "weight_risk": float(resolve("WEIGHT_RISK")),
        "weight_ml": float(resolve("WEIGHT_ML")),
        "output_csv": project_root / resolve("OUTPUT_CSV"),
        "output_html": project_root / resolve("OUTPUT_HTML"),
    }


def _normalize_value(value: Any) -> str:
    """Normalize raw env values and treat template placeholders as unset."""
    text = "" if value is None else str(value).strip()
    if text.startswith("PASTE_YOUR_") and text.endswith("_HERE"):
        return ""
    return text


def _resolve_from_mapping(mapping: dict[str, Any], name: str) -> str:
    """Resolve one value from a dotenv mapping with alias fallback."""
    primary = _normalize_value(mapping.get(name))
    if primary != "":
        return primary
    for alias in ALIASES.get(name, []):
        alias_value = _normalize_value(mapping.get(alias))
        if alias_value != "":
            return alias_value
    return ""
