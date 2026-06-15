"""Feature engineering for intraday shortlist candidates."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.engine.regime import build_market_regime


def build_feature_frame(
    seed_symbols: list[str],
    movers_df: pd.DataFrame,
    market_context: dict[str, Any],
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a feature dataframe from Alpaca market data and catalyst events."""
    movers_map = {str(row["ticker"]).upper(): row.to_dict() for _, row in movers_df.iterrows()} if not movers_df.empty else {}
    snapshots = market_context.get("snapshots", {}) or {}
    latest_bars = market_context.get("latest_bars", {}) or {}
    latest_quotes = market_context.get("latest_quotes", {}) or {}
    intraday_bars = market_context.get("intraday_bars", {}) or {}
    daily_bars = market_context.get("daily_bars", {}) or {}
    news_grouped = _group_news(news_df)
    regime = build_market_regime(market_context)

    rows: list[dict[str, Any]] = []
    for symbol in _clean_symbols(seed_symbols):
        snapshot = snapshots.get(symbol, {}) if isinstance(snapshots, dict) else {}
        latest_bar = latest_bars.get(symbol, {}) if isinstance(latest_bars, dict) else {}
        latest_quote = latest_quotes.get(symbol, {}) if isinstance(latest_quotes, dict) else {}
        intraday_df = _bars_to_frame(intraday_bars.get(symbol, []))
        daily_df = _bars_to_frame(daily_bars.get(symbol, []))
        mover = movers_map.get(symbol, {})
        symbol_news = news_grouped.get(symbol, pd.DataFrame())

        latest_trade = snapshot.get("latestTrade", {}) if isinstance(snapshot, dict) else {}
        latest_quote_snapshot = snapshot.get("latestQuote", {}) if isinstance(snapshot, dict) else {}
        minute_bar_snapshot = snapshot.get("minuteBar", {}) if isinstance(snapshot, dict) else {}
        daily_bar_snapshot = snapshot.get("dailyBar", {}) if isinstance(snapshot, dict) else {}
        prev_daily_bar = snapshot.get("prevDailyBar", {}) if isinstance(snapshot, dict) else {}

        current_price = _first_positive(
            latest_trade.get("p"),
            latest_bar.get("c"),
            minute_bar_snapshot.get("c"),
            mover.get("current_price"),
            daily_bar_snapshot.get("c"),
        )
        bid_price = _first_positive(latest_quote.get("bp"), latest_quote_snapshot.get("bp"))
        ask_price = _first_positive(latest_quote.get("ap"), latest_quote_snapshot.get("ap"))
        spread_pct = _spread_pct(bid_price, ask_price, current_price)
        prev_close = _first_positive(prev_daily_bar.get("c"), _series_last(daily_df, "c", offset=2))
        gap_pct = _pct_change(current_price, prev_close)

        volume_today = _first_positive(daily_bar_snapshot.get("v"), intraday_df["v"].sum() if not intraday_df.empty else 0)
        avg_daily_volume = _average_volume(daily_df)
        relative_volume = volume_today / avg_daily_volume if avg_daily_volume > 0 else 0.0
        dollar_volume = volume_today * current_price

        vwap = _session_vwap(intraday_df, daily_bar_snapshot)
        ret_1m = _return_over_bars(intraday_df, current_price, 1)
        ret_5m = _return_over_bars(intraday_df, current_price, 5)
        ret_15m = _return_over_bars(intraday_df, current_price, 15)
        range_5m = _range_over_bars(intraday_df, 5)
        range_15m = _range_over_bars(intraday_df, 15)
        atr_5m = _atr_over_bars(intraday_df, 5)
        volatility_1m = _volatility_over_bars(intraday_df, 1)
        volatility_5m = _volatility_over_bars(intraday_df, 5)
        volatility_15m = _volatility_over_bars(intraday_df, 15)
        trade_count_recent = _trade_count_recent(intraday_df, 5)
        intraday_high = _max_value(intraday_df, "h", fallback=daily_bar_snapshot.get("h"))
        intraday_low = _min_value(intraday_df, "l", fallback=daily_bar_snapshot.get("l"))
        premarket_high, premarket_low = _premarket_range(intraday_df)
        recent_support = _recent_support(intraday_df)
        recent_resistance = _recent_resistance(intraday_df)
        distance_to_vwap_pct = _distance_pct(current_price, vwap)
        distance_to_intraday_high_pct = _distance_pct(current_price, intraday_high)
        distance_to_premarket_high_pct = _distance_pct(current_price, premarket_high)
        breakout_ready_score = _breakout_ready_score(current_price, vwap, recent_resistance, relative_volume, spread_pct)
        overextension_score = _overextension_score(distance_to_vwap_pct, gap_pct, distance_to_intraday_high_pct)
        wickiness_score = _wickiness_score(intraday_df)
        halt_risk_proxy = _halt_risk_proxy(current_price, spread_pct, volatility_5m, gap_pct)

        latest_news = _select_latest_news(symbol_news)
        freshness_minutes = float(latest_news.get("freshness_minutes", 9999.0))
        headline_strength = float(latest_news.get("headline_strength", 0.0))
        catalyst_type = str(latest_news.get("catalyst_type") or "none")
        sec_filing_type = str(latest_news.get("form_type") or "")
        catalyst = str(latest_news.get("headline") or "No fresh catalyst found")
        if catalyst_type == "none" and (abs(gap_pct) >= 3.0 or abs(_as_float(mover.get("move_percent", 0.0))) >= 3.0):
            catalyst_type = "price_action"
            headline_strength = max(headline_strength, 2.5 + min(relative_volume * 0.4, 2.0))
            catalyst = "No fresh headline; price action and tape strength are driving attention"
        source_count = int(symbol_news["source"].nunique()) if not symbol_news.empty and "source" in symbol_news.columns else 0
        news_count_30m = int((symbol_news["freshness_minutes"] <= 30).sum()) if not symbol_news.empty and "freshness_minutes" in symbol_news.columns else 0
        last_update_time = _latest_timestamp(latest_trade, latest_bar, minute_bar_snapshot, intraday_df)
        data_completeness = _data_completeness(snapshot, latest_bar, latest_quote, intraday_df, daily_df)

        rows.append(
            {
                "symbol": symbol,
                "ticker": symbol,
                "current_price": round(current_price, 4),
                "last_price": round(current_price, 4),
                "last_update_time": last_update_time,
                "quote_time": _latest_timestamp(latest_quote, latest_quote_snapshot),
                "bid_price": round(bid_price, 4),
                "ask_price": round(ask_price, 4),
                "gap_pct": round(gap_pct, 2),
                "move_percent": round(_as_float(mover.get("move_percent", gap_pct)), 2),
                "volume_today": round(volume_today, 2),
                "volume": round(volume_today, 2),
                "relative_volume": round(relative_volume, 2),
                "dollar_volume": round(dollar_volume, 2),
                "bid_ask_spread_pct": round(spread_pct, 4),
                "spread_pct": round(spread_pct, 4),
                "trade_count_recent": round(trade_count_recent, 2),
                "ret_1m": round(ret_1m, 2),
                "ret_5m": round(ret_5m, 2),
                "ret_15m": round(ret_15m, 2),
                "range_5m": round(range_5m, 4),
                "range_15m": round(range_15m, 4),
                "vwap": round(vwap, 4),
                "distance_to_vwap_pct": round(distance_to_vwap_pct, 2),
                "distance_to_intraday_high_pct": round(distance_to_intraday_high_pct, 2),
                "distance_to_premarket_high_pct": round(distance_to_premarket_high_pct, 2),
                "intraday_high": round(intraday_high, 4),
                "intraday_low": round(intraday_low, 4),
                "premarket_high": round(premarket_high, 4),
                "premarket_low": round(premarket_low, 4),
                "recent_support": round(recent_support, 4),
                "recent_resistance": round(recent_resistance, 4),
                "atr_5m": round(atr_5m, 4),
                "recent_15m_range": round(range_15m, 4),
                "breakout_ready_score": round(breakout_ready_score, 2),
                "news_count_30m": news_count_30m,
                "freshness_minutes": round(freshness_minutes, 2),
                "headline_strength": round(headline_strength, 2),
                "catalyst_type": catalyst_type,
                "sec_filing_type": sec_filing_type,
                "source_count": source_count,
                "catalyst": catalyst,
                "volatility_1m": round(volatility_1m, 2),
                "volatility_5m": round(volatility_5m, 2),
                "volatility_15m": round(volatility_15m, 2),
                "overextension_score": round(overextension_score, 2),
                "wickiness_score": round(wickiness_score, 2),
                "halt_risk_proxy": round(halt_risk_proxy, 2),
                "data_completeness": round(data_completeness, 2),
                "index_regime": regime["index_regime"],
                "market_trend_strength": regime["market_trend_strength"],
                "sector_strength": regime["sector_strength"],
                "breadth_proxy": regime["breadth_proxy"],
                "vwap_regime_flag": regime["vwap_regime_flag"],
                "session": str(mover.get("session") or "regular"),
                "mover_group": str(mover.get("mover_group") or "watchlist"),
            }
        )
    return pd.DataFrame(rows)


def _group_news(news_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if news_df.empty or "symbol" not in news_df.columns:
        return {}
    grouped: dict[str, pd.DataFrame] = {}
    for symbol, frame in news_df.groupby(news_df["symbol"].astype(str).str.upper()):
        grouped[str(symbol)] = frame.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return grouped


def _select_latest_news(symbol_news: pd.DataFrame) -> dict[str, Any]:
    if symbol_news.empty:
        return {}
    ranked = symbol_news.sort_values(["headline_strength", "freshness_minutes"], ascending=[False, True])
    return ranked.iloc[0].to_dict()


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


def _clean_symbols(symbols: list[str]) -> list[str]:
    cleaned: list[str] = []
    for symbol in symbols:
        text = str(symbol or "").strip().upper()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first_positive(*values: Any) -> float:
    for value in values:
        numeric = _as_float(value)
        if numeric > 0:
            return numeric
    return 0.0


def _pct_change(current: float, reference: float) -> float:
    if current <= 0 or reference <= 0:
        return 0.0
    return ((current / reference) - 1.0) * 100.0


def _spread_pct(bid: float, ask: float, fallback_price: float) -> float:
    mid = ((bid + ask) / 2.0) if bid > 0 and ask > 0 else fallback_price
    if mid <= 0 or bid <= 0 or ask <= 0:
        return 0.0
    return ((ask - bid) / mid) * 100.0


def _average_volume(daily_df: pd.DataFrame) -> float:
    if daily_df.empty:
        return 0.0
    volumes = daily_df["v"].tail(6)
    if len(volumes) > 1:
        volumes = volumes.iloc[:-1]
    return float(volumes.mean()) if not volumes.empty else 0.0


def _session_vwap(intraday_df: pd.DataFrame, daily_bar_snapshot: dict[str, Any]) -> float:
    if not intraday_df.empty and intraday_df["v"].sum() > 0:
        weighted = (intraday_df["c"] * intraday_df["v"]).sum()
        return float(weighted / intraday_df["v"].sum())
    return _first_positive(daily_bar_snapshot.get("vw"), daily_bar_snapshot.get("c"))


def _return_over_bars(frame: pd.DataFrame, current_price: float, lookback: int) -> float:
    if frame.empty or len(frame) <= lookback:
        return 0.0
    reference = _as_float(frame["c"].iloc[-lookback - 1])
    return _pct_change(current_price, reference)


def _range_over_bars(frame: pd.DataFrame, lookback: int) -> float:
    if frame.empty:
        return 0.0
    tail = frame.tail(max(lookback, 1))
    return float(tail["h"].max() - tail["l"].min())


def _atr_over_bars(frame: pd.DataFrame, lookback: int) -> float:
    if frame.empty:
        return 0.0
    tail = frame.tail(max(lookback, 1)).copy()
    tail["prev_close"] = tail["c"].shift(1).fillna(tail["o"])
    tail["tr"] = (tail[["h", "prev_close"]].max(axis=1) - tail[["l", "prev_close"]].min(axis=1)).abs()
    return float(tail["tr"].mean()) if not tail.empty else 0.0


def _volatility_over_bars(frame: pd.DataFrame, lookback: int) -> float:
    if frame.empty or len(frame) <= 2:
        return 0.0
    tail = frame.tail(max(lookback + 1, 3)).copy()
    tail["ret"] = tail["c"].pct_change()
    return float(tail["ret"].std(ddof=0) * 100.0) if tail["ret"].notna().any() else 0.0


def _trade_count_recent(frame: pd.DataFrame, lookback: int) -> float:
    if frame.empty or "n" not in frame.columns:
        return 0.0
    tail = frame.tail(max(lookback, 1))
    return float(pd.to_numeric(tail["n"], errors="coerce").fillna(0.0).sum())


def _max_value(frame: pd.DataFrame, column: str, fallback: Any = 0.0) -> float:
    if frame.empty or column not in frame.columns:
        return _as_float(fallback)
    return _first_positive(frame[column].max(), fallback)


def _min_value(frame: pd.DataFrame, column: str, fallback: Any = 0.0) -> float:
    if frame.empty or column not in frame.columns:
        return _as_float(fallback)
    numeric = pd.to_numeric(frame[column], errors="coerce").replace(0, pd.NA).dropna()
    return float(numeric.min()) if not numeric.empty else _as_float(fallback)


def _premarket_range(frame: pd.DataFrame) -> tuple[float, float]:
    if frame.empty or frame["t"].isna().all():
        return 0.0, 0.0
    eastern = frame.copy()
    eastern["et"] = eastern["t"].dt.tz_convert("America/New_York")
    premarket = eastern[(eastern["et"].dt.hour < 9) | ((eastern["et"].dt.hour == 9) & (eastern["et"].dt.minute < 30))]
    if premarket.empty:
        return 0.0, 0.0
    return float(premarket["h"].max()), float(premarket["l"].min())


def _recent_support(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    tail = frame.tail(15)
    return float(tail["l"].min())


def _recent_resistance(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    tail = frame.tail(15)
    return float(tail["h"].max())


def _distance_pct(current: float, reference: float) -> float:
    if current <= 0 or reference <= 0:
        return 0.0
    return ((current / reference) - 1.0) * 100.0


def _breakout_ready_score(current: float, vwap: float, resistance: float, relative_volume: float, spread_pct: float) -> float:
    score = 0.0
    if current > 0 and vwap > 0 and current >= vwap:
        score += 2.0
    if resistance > 0:
        distance = abs(_distance_pct(current, resistance))
        if distance <= 0.4:
            score += 3.5
        elif distance <= 1.0:
            score += 2.0
    if relative_volume >= 3.0:
        score += 2.5
    elif relative_volume >= 1.5:
        score += 1.5
    if 0 < spread_pct <= 0.4:
        score += 2.0
    elif spread_pct <= 1.0:
        score += 1.0
    return min(score, 10.0)


def _overextension_score(distance_to_vwap_pct: float, gap_pct: float, distance_to_intraday_high_pct: float) -> float:
    score = max(distance_to_vwap_pct - 1.0, 0.0) * 1.5
    score += max(gap_pct - 12.0, 0.0) * 0.15
    if distance_to_intraday_high_pct > -0.2:
        score += 1.5
    return min(score, 10.0)


def _wickiness_score(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    tail = frame.tail(5).copy()
    ranges = (tail["h"] - tail["l"]).replace(0, pd.NA)
    body = (tail["c"] - tail["o"]).abs()
    wick_ratio = ((ranges - body) / ranges).clip(lower=0).fillna(0)
    return float(wick_ratio.mean() * 10.0)


def _halt_risk_proxy(current_price: float, spread_pct: float, volatility_5m: float, gap_pct: float) -> float:
    score = 0.0
    if current_price < 5:
        score += 3.0
    score += min(spread_pct * 2.0, 3.0)
    score += min(volatility_5m * 0.6, 2.5)
    score += min(max(abs(gap_pct) - 15.0, 0.0) * 0.12, 2.5)
    return min(score, 10.0)


def _latest_timestamp(*candidates: Any) -> str:
    values: list[pd.Timestamp] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ["t", "timestamp"]:
                if key in candidate and candidate.get(key):
                    parsed = pd.to_datetime(candidate.get(key), errors="coerce", utc=True)
                    if pd.notna(parsed):
                        values.append(parsed)
        elif isinstance(candidate, pd.DataFrame) and not candidate.empty and "t" in candidate.columns:
            parsed = pd.to_datetime(candidate["t"], errors="coerce", utc=True).dropna()
            if not parsed.empty:
                values.append(parsed.max())
    if not values:
        return ""
    return max(values).isoformat()


def _data_completeness(snapshot: dict[str, Any], latest_bar: dict[str, Any], latest_quote: dict[str, Any], intraday_df: pd.DataFrame, daily_df: pd.DataFrame) -> float:
    checks = [
        bool(snapshot),
        bool(latest_bar),
        bool(latest_quote),
        not intraday_df.empty,
        not daily_df.empty,
    ]
    return sum(checks) / len(checks)


def _series_last(frame: pd.DataFrame, column: str, offset: int = 1) -> float:
    if frame.empty or column not in frame.columns or len(frame) < offset:
        return 0.0
    return _as_float(frame[column].iloc[-offset])
