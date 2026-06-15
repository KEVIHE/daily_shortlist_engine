"""Scoring logic for short-term and mid-term candidate lists."""

from __future__ import annotations

from typing import Any

import pandas as pd


OBSCURE_TICKERS = {"SOUN", "TOP", "HKD", "GNS"}
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
OUTPUT_COLUMNS = [
    "list_type",
    "ticker",
    "current_price",
    "move_percent",
    "catalyst",
    "liquidity_flag",
    "setup_type",
    "risk_note",
    "score",
    "news_score",
    "liquidity_score",
    "setup_score",
    "tradeable",
    "tradeable_reason",
    "not_tradeable_reason",
]


def build_shortlists(df: pd.DataFrame) -> pd.DataFrame:
    """Build short-term and mid-term ranked lists from filtered movers."""
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    base = df.copy()
    base["setup_type"] = base["setup_type"].map(_normalize_setup_type)
    base["news_score"] = base.apply(_score_news, axis=1)
    base["liquidity_score"] = base.apply(_score_liquidity, axis=1)
    base["setup_score"] = base.apply(_score_setup, axis=1)
    base["tradeable"] = base.apply(_is_tradeable, axis=1)
    explanation_frame = base.apply(_tradeability_reasons, axis=1, result_type="expand")
    explanation_frame.columns = ["tradeable_reason", "not_tradeable_reason"]
    base[["tradeable_reason", "not_tradeable_reason"]] = explanation_frame
    base["short_term_score"] = base.apply(_score_short_term, axis=1)
    base["mid_term_score"] = base.apply(_score_mid_term, axis=1)

    short_term = (
        base.sort_values(["short_term_score", "news_score", "move_percent"], ascending=False)
        .head(5)
        .assign(list_type="short_term", score=lambda frame: frame["short_term_score"])
    )
    mid_term = (
        base.sort_values(["mid_term_score", "setup_score", "move_percent"], ascending=False)
        .head(5)
        .assign(list_type="mid_term", score=lambda frame: frame["mid_term_score"])
    )

    output = pd.concat([short_term, mid_term], ignore_index=True)
    return output[OUTPUT_COLUMNS]


def _score_short_term(row: pd.Series) -> float:
    """Score same-day setups using movement, session timing, and catalysts."""
    move_score = min(abs(_as_float(row.get("move_percent"))), 10.0)
    score = (
        0.35 * _as_float(row.get("news_score"))
        + 0.30 * _as_float(row.get("liquidity_score"))
        + 0.20 * _as_float(row.get("setup_score"))
        + 0.15 * move_score
    )
    if str(row.get("session", "")).lower() == "premarket":
        score += 0.75
    if str(row.get("mover_group", "")).lower() == "most_actives":
        score += 0.35
    return round(min(score, 10.0), 2)


def _score_mid_term(row: pd.Series) -> float:
    """Score event-driven or continuation setups for a multi-day watchlist."""
    move_score = min(abs(_as_float(row.get("move_percent"))), 10.0)
    score = (
        0.25 * _as_float(row.get("news_score"))
        + 0.35 * _as_float(row.get("liquidity_score"))
        + 0.30 * _as_float(row.get("setup_score"))
        + 0.10 * move_score
    )
    if row.get("setup_type") in {"event_driven", "trend_continuation", "relative_strength"}:
        score += 0.75
    if _as_float(row.get("current_price")) >= 20:
        score += 0.25
    return round(min(score, 10.0), 2)


def _score_news(row: pd.Series) -> float:
    """Score the clarity and strength of the catalyst."""
    catalyst = str(row.get("catalyst") or "").strip().lower()
    if not catalyst or catalyst == "no fresh catalyst found":
        return 1.0

    score = 2.0
    score += min(sum(term in catalyst for term in HIGH_SIGNAL_TERMS) * 1.2, 4.0)
    if row.get("setup_type") in {"event_driven", "news_momentum"}:
        score += 1.5
    if str(row.get("session", "")).lower() == "premarket":
        score += 1.0
    if abs(_as_float(row.get("move_percent"))) >= 4:
        score += 1.0
    return round(min(score, 10.0), 2)


def _score_liquidity(row: pd.Series) -> float:
    """Score the quality of liquidity and tradeability."""
    score = 0.0
    dollar_volume = _as_float(row.get("dollar_volume"))
    if dollar_volume <= 0:
        dollar_volume = _as_float(row.get("volume")) * max(_as_float(row.get("current_price")), 0)

    if str(row.get("liquidity_flag", "")).lower() == "liquid":
        score += 2.0

    if dollar_volume >= 2_000_000_000:
        score += 4.0
    elif dollar_volume >= 1_000_000_000:
        score += 3.5
    elif dollar_volume >= 500_000_000:
        score += 2.5
    elif dollar_volume >= 100_000_000:
        score += 1.5
    elif dollar_volume >= 25_000_000:
        score += 0.75

    volume = _as_float(row.get("volume"))
    if volume >= 50_000_000:
        score += 2.0
    elif volume >= 20_000_000:
        score += 1.5
    elif volume >= 5_000_000:
        score += 1.0

    price = _as_float(row.get("current_price"))
    if 10 <= price <= 250:
        score += 2.0
    elif 5 <= price <= 500:
        score += 1.0

    return round(min(score, 10.0), 2)


def _score_setup(row: pd.Series) -> float:
    """Score the attractiveness of the setup classification."""
    setup_type = _normalize_setup_type(row.get("setup_type"))
    score = SETUP_BASE_SCORES.get(setup_type, SETUP_BASE_SCORES["unknown"])
    if abs(_as_float(row.get("move_percent"))) >= 4 and setup_type not in {"speculative", "unknown"}:
        score += 0.5
    return round(min(score, 10.0), 2)


def _is_tradeable(row: pd.Series) -> bool:
    """Flag ideas that look tradable using simple, conservative rules."""
    ticker = str(row.get("ticker") or "").upper().strip()
    price = _as_float(row.get("current_price"))
    setup_type = _normalize_setup_type(row.get("setup_type"))
    liquidity_flag = str(row.get("liquidity_flag") or "unknown").lower()

    is_reasonable_symbol = ticker.isalpha() and 1 <= len(ticker) <= 5 and ticker not in OBSCURE_TICKERS
    has_valid_setup = setup_type not in {"", "unknown", "watchlist", "speculative"}
    has_reasonable_price = 7 <= price <= 500
    return bool(liquidity_flag == "liquid" and is_reasonable_symbol and has_valid_setup and has_reasonable_price)


def _tradeability_reasons(row: pd.Series) -> tuple[str, str]:
    """Explain the tradeable flag using simple, transparent rules."""
    positives: list[str] = []
    negatives: list[str] = []

    if str(row.get("liquidity_flag") or "").lower() == "liquid":
        positives.append("liquidity ok")
    else:
        negatives.append("low liquidity")

    price = _as_float(row.get("current_price"))
    if 7 <= price <= 500:
        positives.append("price in range")
    else:
        negatives.append("price outside preferred range")

    setup_type = _normalize_setup_type(row.get("setup_type"))
    if setup_type not in {"", "unknown", "watchlist", "speculative"}:
        positives.append("setup acceptable")
    else:
        negatives.append("weak setup")

    ticker = str(row.get("ticker") or "").upper().strip()
    if ticker.isalpha() and 1 <= len(ticker) <= 5 and ticker not in OBSCURE_TICKERS:
        positives.append("ticker quality acceptable")
    else:
        negatives.append("suspicious ticker quality")

    if abs(_as_float(row.get("move_percent"))) >= 2:
        positives.append("meaningful price move")

    if bool(row.get("tradeable")):
        return ", ".join(positives) or "passes basic tradeability checks", "no blocking issues found"
    return ", ".join(positives) or "limited positives", ", ".join(negatives) or "failed tradeability checks"


def _normalize_setup_type(value: Any) -> str:
    """Normalize setup labels to consistent snake_case values."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text or "unknown"


def _as_float(value: Any) -> float:
    """Return a best-effort float value."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
