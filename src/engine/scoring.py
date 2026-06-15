"""Explainable scoring model for intraday shortlist candidates."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.settings import EngineSettings
from src.engine.labels import build_action_note, build_risk_text, build_selection_reason, determine_setup_bias, determine_status_tag


def score_candidates(feature_df: pd.DataFrame, settings: EngineSettings) -> pd.DataFrame:
    """Score the candidate universe using explainable weighted rules."""
    if feature_df.empty:
        return feature_df.copy()

    scored = feature_df.copy()
    scored["news_score"] = scored.apply(_score_news, axis=1)
    scored["catalyst_score"] = scored["news_score"]
    scored["liquidity_score"] = scored.apply(_score_liquidity, axis=1)
    scored["setup_score"] = scored.apply(_score_setup, axis=1)
    scored["risk_score"] = scored.apply(_score_risk, axis=1)
    if "ml_score" not in scored.columns:
        scored["ml_score"] = 0.0
    scored["ml_score"] = pd.to_numeric(scored["ml_score"], errors="coerce").fillna(0.0)
    scored["total_score"] = scored.apply(lambda row: _total_score(row, settings), axis=1)
    scored["eligible"] = scored.apply(lambda row: _eligible(row, settings), axis=1)
    scored["status_tag"] = scored.apply(
        lambda row: determine_status_tag(row.to_dict(), settings.minimum_total_score, settings.max_spread_pct),
        axis=1,
    )
    scored["setup_bias"] = scored.apply(lambda row: determine_setup_bias(row.to_dict()), axis=1)
    scored["tradeable"] = scored.apply(lambda row: _is_tradeable(row, settings), axis=1)
    scored["tradeable_reason"] = scored.apply(lambda row: _tradeable_reason(row), axis=1)
    scored["not_tradeable_reason"] = scored.apply(lambda row: _not_tradeable_reason(row, settings), axis=1)
    scored["selection_reason"] = scored.apply(lambda row: build_selection_reason(row.to_dict()), axis=1)
    scored["risk_note"] = scored.apply(lambda row: build_risk_text(row.to_dict()), axis=1)
    scored["action_note"] = scored.apply(lambda row: build_action_note(row.to_dict()), axis=1)
    scored["list_type"] = scored.apply(_assign_list_type, axis=1)
    return scored


def _score_news(row: pd.Series) -> float:
    score = min(float(row.get("headline_strength", 0.0)), 6.5)
    freshness = float(row.get("freshness_minutes", 9999.0))
    if freshness <= 10:
        score += 2.5
    elif freshness <= 30:
        score += 1.75
    elif freshness <= 120:
        score += 0.75
    score += min(float(row.get("source_count", 0.0)) * 0.5, 1.5)
    score += min(float(row.get("news_count_30m", 0.0)) * 0.35, 1.0)
    if str(row.get("catalyst_type") or "") in {"m_and_a", "earnings", "contract", "sec_8k", "sec_13d"}:
        score += 0.75
    if str(row.get("catalyst_type") or "") == "price_action":
        score += 0.75
    return round(min(score, 10.0), 2)


def _score_liquidity(row: pd.Series) -> float:
    score = 0.0
    dollar_volume = float(row.get("dollar_volume", 0.0))
    relative_volume = float(row.get("relative_volume", 0.0))
    spread_pct = float(row.get("spread_pct", 0.0))
    price = float(row.get("current_price", 0.0))

    if dollar_volume >= 100_000_000:
        score += 4.0
    elif dollar_volume >= 30_000_000:
        score += 3.0
    elif dollar_volume >= 10_000_000:
        score += 2.0
    elif dollar_volume >= 5_000_000:
        score += 1.0

    if relative_volume >= 5.0:
        score += 3.0
    elif relative_volume >= 2.0:
        score += 2.0
    elif relative_volume >= 1.2:
        score += 1.0

    trades = float(row.get("trade_count_recent", 0.0))
    if trades >= 5000:
        score += 1.0
    elif trades >= 500:
        score += 0.5

    if 0 < spread_pct <= 0.15:
        score += 2.0
    elif spread_pct <= 0.40:
        score += 1.5
    elif spread_pct <= 0.80:
        score += 0.75

    if 5 <= price <= 150:
        score += 1.0
    return round(min(score, 10.0), 2)


def _score_setup(row: pd.Series) -> float:
    score = min(float(row.get("breakout_ready_score", 0.0)), 5.0)
    if abs(float(row.get("distance_to_vwap_pct", 0.0))) <= 1.0:
        score += 2.0
    if float(row.get("ret_15m", 0.0)) > 0:
        score += 1.25
    if str(row.get("catalyst_type") or "") in {"sec_10q", "sec_10k", "earnings", "contract"}:
        score += 1.0
    if float(row.get("range_15m", 0.0)) > 0:
        score += 0.75
    return round(min(score, 10.0), 2)


def _score_risk(row: pd.Series) -> float:
    score = 0.0
    score += min(float(row.get("overextension_score", 0.0)) * 0.7, 3.5)
    score += min(float(row.get("volatility_1m", 0.0)) * 0.25, 1.0)
    score += min(float(row.get("volatility_5m", 0.0)) * 0.4, 2.0)
    score += min(float(row.get("volatility_15m", 0.0)) * 0.2, 1.5)
    score += min(float(row.get("wickiness_score", 0.0)) * 0.3, 1.5)
    score += min(float(row.get("halt_risk_proxy", 0.0)) * 0.5, 2.0)
    return round(min(score, 10.0), 2)


def _total_score(row: pd.Series, settings: EngineSettings) -> float:
    score = (
        settings.weights.catalyst * float(row.get("catalyst_score", 0.0))
        + settings.weights.setup * float(row.get("setup_score", 0.0))
        + settings.weights.liquidity * float(row.get("liquidity_score", 0.0))
        - settings.weights.risk * float(row.get("risk_score", 0.0))
        + settings.weights.ml * float(row.get("ml_score", 0.0))
    )
    return round(max(min(score, 10.0), 0.0), 2)


def _eligible(row: pd.Series, settings: EngineSettings) -> bool:
    if not _has_quality_symbol(str(row.get("symbol") or row.get("ticker") or "")):
        return False
    if float(row.get("current_price", 0.0)) < settings.min_price:
        return False
    if float(row.get("current_price", 0.0)) > settings.max_price:
        return False
    if float(row.get("spread_pct", 0.0)) > settings.max_spread_pct * 2.0:
        return False
    if float(row.get("dollar_volume", 0.0)) < settings.min_dollar_volume * 0.5:
        return False
    if float(row.get("relative_volume", 0.0)) < 0.7 and str(row.get("catalyst_type") or "none") == "none":
        return False
    return True


def _is_tradeable(row: pd.Series, settings: EngineSettings) -> bool:
    return bool(
        row.get("eligible")
        and float(row.get("total_score", 0.0)) >= settings.minimum_total_score
        and float(row.get("spread_pct", 0.0)) <= settings.max_spread_pct
        and float(row.get("dollar_volume", 0.0)) >= settings.min_dollar_volume
        and float(row.get("relative_volume", 0.0)) >= settings.min_relative_volume
        and str(row.get("status_tag") or "") not in {"Extended", "Risky", "Ignore"}
    )


def _tradeable_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if float(row.get("relative_volume", 0.0)) >= 1.2:
        reasons.append("relative volume is strong enough")
    if float(row.get("spread_pct", 0.0)) <= 0.4:
        reasons.append("spread is manageable")
    if float(row.get("dollar_volume", 0.0)) >= 5_000_000:
        reasons.append("dollar volume is healthy")
    if str(row.get("status_tag") or "") == "Breakout Watch":
        reasons.append("near the breakout zone")
    if str(row.get("status_tag") or "") == "Pullback Watch":
        reasons.append("pullback structure is clear")
    return ", ".join(reasons) or "needs more confirmation"


def _not_tradeable_reason(row: pd.Series, settings: EngineSettings) -> str:
    reasons: list[str] = []
    if float(row.get("spread_pct", 0.0)) > settings.max_spread_pct:
        reasons.append("spread is too wide")
    if float(row.get("dollar_volume", 0.0)) < settings.min_dollar_volume:
        reasons.append("dollar volume is too low")
    if float(row.get("relative_volume", 0.0)) < settings.min_relative_volume:
        reasons.append("relative volume is too weak")
    if str(row.get("status_tag") or "") == "Extended":
        reasons.append("already extended")
    if str(row.get("status_tag") or "") == "Risky":
        reasons.append("risk weighting is too high")
    if str(row.get("status_tag") or "") == "Ignore":
        reasons.append("no executable structure has formed")
    return ", ".join(reasons) or "no major blocking factor"


def _assign_list_type(row: pd.Series) -> str:
    if str(row.get("catalyst_type") or "") in {"sec_10q", "sec_10k", "sec_13d", "sec_13g"} and float(row.get("total_score", 0.0)) >= 4.5:
        return "mid_term"
    if str(row.get("status_tag") or "") in {"Breakout Watch", "Pullback Watch"}:
        return "short_term"
    return "mid_term"


def _has_quality_symbol(symbol: str) -> bool:
    cleaned = str(symbol or "").strip().upper()
    if not cleaned:
        return False
    if "." in cleaned or "/" in cleaned or "-" in cleaned:
        return False
    if cleaned.endswith("W") and len(cleaned) > 4:
        return False
    return cleaned.isalpha() and 1 <= len(cleaned) <= 5
