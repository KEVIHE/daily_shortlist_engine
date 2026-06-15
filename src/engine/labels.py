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
        return "只适合突破确认后跟随"
    if tag == "Pullback Watch":
        return "优先等回踩，不追高"
    if tag == "Extended":
        return "虽然强，但已过度延伸"
    if tag == "Risky":
        return "催化或走势存在噪音，谨慎"
    return "暂不纳入主动交易"


def build_selection_reason(row: dict[str, Any]) -> str:
    """Summarize why the symbol survived into the shortlist."""
    reasons: list[str] = []
    if row.get("news_score", 0.0) >= 6.0:
        reasons.append("催化新鲜")
    if row.get("relative_volume", 0.0) >= 2.0:
        reasons.append("量能放大")
    if row.get("breakout_ready_score", 0.0) >= 5.0:
        reasons.append("结构接近突破")
    if abs(row.get("distance_to_vwap_pct", 0.0)) <= 1.0:
        reasons.append("位置接近VWAP")
    if row.get("spread_pct", 0.0) <= 0.4:
        reasons.append("点差可控")
    return "、".join(reasons) or "进入候选池但需要更多确认"


def build_risk_text(row: dict[str, Any]) -> str:
    """Summarize the main risk factors behind the candidate."""
    risks: list[str] = []
    if row.get("spread_pct", 0.0) > 0.7:
        risks.append("点差偏宽")
    if row.get("overextension_score", 0.0) >= 6.0:
        risks.append("离VWAP过远")
    if row.get("headline_strength", 0.0) <= 2.5:
        risks.append("催化解释弱")
    if row.get("halt_risk_proxy", 0.0) >= 5.0:
        risks.append("波动过急")
    if row.get("relative_volume", 0.0) < 1.0:
        risks.append("量能确认不足")
    return "、".join(risks) or "主要风险可控，但仍需手动盯盘"
