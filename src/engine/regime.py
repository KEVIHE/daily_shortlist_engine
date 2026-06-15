"""Market regime helpers shared across rule and ML layers."""

from __future__ import annotations

from typing import Any

import pandas as pd

REGIME_SYMBOLS = ["SPY", "QQQ", "IWM", "SMH"]


def build_market_regime(market_context: dict[str, Any]) -> dict[str, Any]:
    """Derive simple market-environment features from broad index proxies."""
    intraday_map = market_context.get("intraday_bars", {}) or {}
    snapshots = market_context.get("snapshots", {}) or {}
    latest_bars = market_context.get("latest_bars", {}) or {}
    latest_quotes = market_context.get("latest_quotes", {}) or {}

    rows: list[dict[str, float]] = []
    for symbol in REGIME_SYMBOLS:
        frame = _bars_to_frame(intraday_map.get(symbol, []))
        snapshot = snapshots.get(symbol, {}) if isinstance(snapshots, dict) else {}
        latest_bar = latest_bars.get(symbol, {}) if isinstance(latest_bars, dict) else {}
        latest_quote = latest_quotes.get(symbol, {}) if isinstance(latest_quotes, dict) else {}
        price = _first_positive(
            ((snapshot.get("latestTrade") or {}).get("p") if isinstance(snapshot, dict) else 0),
            latest_bar.get("c") if isinstance(latest_bar, dict) else 0,
            frame["c"].iloc[-1] if not frame.empty else 0,
        )
        prev = _first_positive(
            ((snapshot.get("prevDailyBar") or {}).get("c") if isinstance(snapshot, dict) else 0),
            frame["c"].iloc[0] if len(frame) >= 2 else 0,
        )
        vwap = _session_vwap(frame, (snapshot.get("dailyBar") or {}) if isinstance(snapshot, dict) else {})
        ret_15m = _return_over_bars(frame, price, 15)
        spread = _spread_pct(latest_quote.get("bp") if isinstance(latest_quote, dict) else 0, latest_quote.get("ap") if isinstance(latest_quote, dict) else 0, price)
        rows.append(
            {
                "symbol": symbol,
                "price": price,
                "prev": prev,
                "vwap": vwap,
                "ret_15m": ret_15m,
                "above_vwap": 1.0 if price > 0 and vwap > 0 and price >= vwap else 0.0,
                "spread_pct": spread,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "index_regime": "unknown",
            "market_trend_strength": 0.0,
            "sector_strength": 0.0,
            "breadth_proxy": 0.0,
            "vwap_regime_flag": 0,
        }

    market_trend_strength = float(frame["ret_15m"].mean()) if not frame.empty else 0.0
    sector_strength = float(frame.loc[frame["symbol"] == "SMH", "ret_15m"].mean()) if (frame["symbol"] == "SMH").any() else market_trend_strength
    breadth_proxy = float(frame["above_vwap"].mean()) if not frame.empty else 0.0
    vwap_regime_flag = int(breadth_proxy >= 0.5)
    if market_trend_strength >= 0.5:
        index_regime = "risk_on"
    elif market_trend_strength <= -0.5:
        index_regime = "risk_off"
    else:
        index_regime = "balanced"
    return {
        "index_regime": index_regime,
        "market_trend_strength": round(market_trend_strength, 3),
        "sector_strength": round(sector_strength, 3),
        "breadth_proxy": round(breadth_proxy, 3),
        "vwap_regime_flag": vwap_regime_flag,
    }


def _bars_to_frame(bars: Any) -> pd.DataFrame:
    if not isinstance(bars, list) or not bars:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v", "vw", "n"])
    frame = pd.DataFrame(bars)
    for column in ["o", "h", "l", "c", "v", "vw", "n"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if "t" in frame.columns:
        frame["t"] = pd.to_datetime(frame["t"], errors="coerce", utc=True)
    else:
        frame["t"] = pd.NaT
    return frame.sort_values("t").reset_index(drop=True)


def _first_positive(*values: Any) -> float:
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if numeric > 0:
            return numeric
    return 0.0


def _session_vwap(intraday_df: pd.DataFrame, daily_bar_snapshot: dict[str, Any]) -> float:
    if not intraday_df.empty and intraday_df["v"].sum() > 0:
        weighted = (intraday_df["c"] * intraday_df["v"]).sum()
        return float(weighted / intraday_df["v"].sum())
    return _first_positive(daily_bar_snapshot.get("vw"), daily_bar_snapshot.get("c"))


def _return_over_bars(frame: pd.DataFrame, current_price: float, lookback: int) -> float:
    if frame.empty or len(frame) <= lookback:
        return 0.0
    reference = _first_positive(frame["c"].iloc[-lookback - 1])
    if current_price <= 0 or reference <= 0:
        return 0.0
    return ((current_price / reference) - 1.0) * 100.0


def _spread_pct(bid: Any, ask: Any, fallback_price: float) -> float:
    bid_f = _first_positive(bid)
    ask_f = _first_positive(ask)
    mid = ((bid_f + ask_f) / 2.0) if bid_f > 0 and ask_f > 0 else fallback_price
    if mid <= 0 or bid_f <= 0 or ask_f <= 0:
        return 0.0
    return ((ask_f - bid_f) / mid) * 100.0
