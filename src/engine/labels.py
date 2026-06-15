"""Labeling helpers for shortlist status tags and action notes."""

from __future__ import annotations

from typing import Any


def determine_status_tag(row: dict[str, Any], minimum_total_score: float, max_spread_pct: float) -> str:
    """Map feature state to an actionable shortlist tag."""
    if row.get("spread_pct", 0.0) > max_spread_pct * 1.4 or row.get("dollar_volume", 0.0) < 1_000_000:
        return "Risky"
    if row.get("overextension_score", 0.0) >= 7.0 or row.get("distance_to_vwap_pct", 0.0) >= 4.0:
        return "Extended"
    if row.get("total_score", 0.0) < minimum_total_score:
        return "Ignore"
    if row.get("breakout_ready_score", 0.0) >= 5.5 and row.get("distance_to_intraday_high_pct", 0.0) >= -0.8:
        return "Breakout Watch"
    if abs(row.get("distance_to_vwap_pct", 0.0)) <= 1.0 and row.get("ret_5m", 0.0) <= 1.5:
        return "Pullback Watch"
    return "Risky"


def determine_setup_bias(row: dict[str, Any]) -> str:
    """Explain whether the current structure looks more like breakout or pullback."""
    if row.get("status_tag") == "Breakout Watch":
        return "breakout"
    if row.get("status_tag") == "Pullback Watch":
        return "pullback"
    if row.get("status_tag") == "Extended":
        return "extended"
    return "neutral"


def build_action_note(row: dict[str, Any]) -> str:
    """Return a short execution-oriented note without issuing a buy order."""
    tag = str(row.get("status_tag") or "Ignore")
    if tag == "Breakout Watch":
        return "Only suitable after breakout confirmation."
    if tag == "Pullback Watch":
        return "Prefer to wait for a pullback instead of chasing."
    if tag == "Extended":
        return "Momentum is still strong, but the move is already extended."
    if tag == "Risky":
        return "Catalyst or price action is noisy, so stay cautious."
    return "Not suitable for active trading right now."


def build_selection_reason(row: dict[str, Any]) -> str:
    """Summarize why the symbol survived into the shortlist."""
    reasons: list[str] = []
    if row.get("news_score", 0.0) >= 6.0:
        reasons.append("fresh catalyst")
    if row.get("relative_volume", 0.0) >= 2.0:
        reasons.append("volume expansion")
    if row.get("breakout_ready_score", 0.0) >= 5.0:
        reasons.append("structure is near breakout")
    if abs(row.get("distance_to_vwap_pct", 0.0)) <= 1.0:
        reasons.append("price is near VWAP")
    if row.get("spread_pct", 0.0) <= 0.4:
        reasons.append("spread is manageable")
    return ", ".join(reasons) or "made the candidate pool but still needs more confirmation"


def build_risk_text(row: dict[str, Any]) -> str:
    """Summarize the main risk factors behind the candidate."""
    risks: list[str] = []
    if row.get("spread_pct", 0.0) > 0.7:
        risks.append("spread is wide")
    if row.get("overextension_score", 0.0) >= 6.0:
        risks.append("too far from VWAP")
    if row.get("headline_strength", 0.0) <= 2.5:
        risks.append("catalyst explanation is weak")
    if row.get("halt_risk_proxy", 0.0) >= 5.0:
        risks.append("price action is too abrupt")
    if row.get("relative_volume", 0.0) < 1.0:
        risks.append("volume confirmation is weak")
    return ", ".join(risks) or "main risks look manageable, but the name still needs manual monitoring"
