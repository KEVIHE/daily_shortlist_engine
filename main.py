"""Entry point for the intraday shortlist workstation pipeline."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.alpaca_client import AlpacaClient
from src.config.settings import EngineSettings, load_settings
from src.data.alpaca_market import AlpacaMarketDataClient
from src.data.alpaca_news import AlpacaNewsClient
from src.data.sec_rss import SecRssClient
from src.engine.features import build_feature_frame
from src.engine.filters import build_shortlist
from src.engine.range_model import apply_range_model
from src.engine.scoring import score_candidates
from src.models.model_service import run_ml_pipeline
from src.report import build_outputs
from src.storage.db import backfill_range_outcomes, initialize_database, load_news_count_today, record_candidate_snapshots, record_news_events
from src.workstation_data import (
    append_activity_event,
    ensure_data_directories,
    make_history_path,
    write_context_snapshot,
    write_csv_atomic,
    write_run_status,
)


class IBKRClient:
    """Reserved interface for future IBKR integration."""

    def __init__(self) -> None:
        self.enabled = False


def run() -> int:
    """Run the upgraded shortlist pipeline and persist workstation outputs."""
    project_root = Path(__file__).resolve().parent
    data_paths = ensure_data_directories(project_root)
    settings = load_settings(project_root)
    generated_at = datetime.now().astimezone()
    latest_file = data_paths["latest_shortlist"]
    history_file = make_history_path(data_paths["history_dir"], generated_at)
    source_csv = settings.output_csv
    report_html = settings.output_html
    run_mode = "mock" if settings.mock_mode else "live"
    api_probe_time = generated_at.isoformat(timespec="seconds")
    row_count = 0
    used_fallback_data = False
    data_mode = "mock"
    alpaca_status = "skipped" if settings.mock_mode else "unavailable"
    alpaca_message = "mock mode enabled" if settings.mock_mode else "credentials missing"
    news_status = "skipped" if settings.mock_mode else "unavailable"
    news_message = "mock mode enabled" if settings.mock_mode else "no news probe yet"
    sec_status = "skipped" if settings.mock_mode else "unavailable"
    sec_message = "mock mode enabled" if settings.mock_mode else "no SEC probe yet"
    benzinga_status = "skipped"
    benzinga_message = "disabled in this pipeline"
    live_sources_used: set[str] = set()
    news_items_ingested = 0
    candidate_count = 0
    ml_status = "fallback"
    ml_message = "No model run recorded."
    stage = "startup"

    _safe_append_activity(
        data_paths["activity_log"],
        event_type="run",
        title="Shortlist run started",
        message=f"main.py started a {run_mode} shortlist run.",
        status="info",
        related_file=str(project_root / "main.py"),
    )

    try:
        initialize_database(settings.db_path)
        market_client = AlpacaMarketDataClient(
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            base_url=settings.alpaca_base_url,
            mock_mode=settings.mock_mode,
        )
        news_client = AlpacaNewsClient(
            api_key=settings.alpaca_api_key,
            api_secret=settings.alpaca_api_secret,
            base_url=settings.alpaca_base_url,
            mock_mode=settings.mock_mode,
        )
        sec_client = SecRssClient(user_agent=settings.sec_user_agent, cache_path=settings.sec_cache_path)
        _ibkr = IBKRClient()

        stage = "api_probe"
        api_probe_time = datetime.now().astimezone().isoformat(timespec="seconds")
        market_probe = market_client.probe()
        alpaca_status = str(market_probe.get("status") or "unavailable")
        alpaca_message = str(market_probe.get("message") or "probe unavailable")
        _log_provider_probe(data_paths["activity_log"], "Alpaca", market_probe)

        if not settings.mock_mode:
            news_probe = news_client.probe()
            sec_probe = sec_client.probe()
        else:
            news_probe = {"status": "skipped", "message": "mock mode enabled"}
            sec_probe = {"status": "skipped", "message": "mock mode enabled"}
        _log_provider_probe(data_paths["activity_log"], "Alpaca News", news_probe)
        _log_provider_probe(data_paths["activity_log"], "SEC RSS", sec_probe)
        news_status, news_message = _combine_news_status(news_probe, sec_probe)
        sec_status = str(sec_probe.get("status") or "unavailable")
        sec_message = str(sec_probe.get("message") or "probe unavailable")

        stage = "market_fetch"
        movers_df, alpaca_status, alpaca_message, market_live, market_fallback = _load_market_movers(settings, market_client, market_probe)
        used_fallback_data = used_fallback_data or market_fallback
        if market_live:
            live_sources_used.add("alpaca")

        stage = "news_fetch"
        general_news_df = _load_alpaca_news(settings, news_client)
        sec_filings_df = _load_sec_filings(settings, sec_client)
        combined_news = _combine_news_frames(general_news_df, sec_filings_df)
        news_items_ingested = len(combined_news)
        inserted_news = record_news_events(settings.db_path, combined_news)
        _safe_append_activity(
            data_paths["activity_log"],
            event_type="news",
            title="News ingestion completed",
            message=f"Collected {news_items_ingested} items and inserted {inserted_news} new records.",
            status="success" if news_items_ingested else "info",
            related_file=str(settings.db_path),
        )

        news_live = bool(not general_news_df.empty or not sec_filings_df.empty)
        if news_live:
            live_sources_used.add("news")
        elif not settings.mock_mode:
            used_fallback_data = True

        stage = "candidate_build"
        seed_symbols = _build_seed_symbols(movers_df, combined_news, settings.movers_limit)
        market_context = _build_market_context(settings, market_client, movers_df, seed_symbols)
        feature_df = build_feature_frame(seed_symbols, movers_df, market_context, combined_news)
        feature_df, ml_result = run_ml_pipeline(project_root, settings.db_path, feature_df)
        ml_status = str(ml_result.get("status") or "fallback")
        ml_message = str(ml_result.get("message") or "No model run recorded.")
        scored_df = score_candidates(feature_df, settings)
        ranged_df = apply_range_model(scored_df)
        shortlist_df = build_shortlist(ranged_df, settings)
        shortlist_df = _finalize_shortlist(shortlist_df)
        row_count = len(shortlist_df)
        candidate_count = row_count

        stage = "persistence"
        record_candidate_snapshots(settings.db_path, shortlist_df, generated_at)
        _safe_append_activity(
            data_paths["activity_log"],
            event_type="run",
            title="Candidate snapshots recorded",
            message=f"Stored {row_count} candidate rows in sqlite.",
            status="success",
            related_file=str(settings.db_path),
        )
        if alpaca_status == "connected":
            try:
                outcomes_backfilled = backfill_range_outcomes(settings.db_path, market_client, datetime.now().astimezone())
                _safe_append_activity(
                    data_paths["activity_log"],
                    event_type="review",
                    title="Replay backfill completed",
                    message=f"Updated {outcomes_backfilled} range outcome rows.",
                    status="success" if outcomes_backfilled else "info",
                    related_file=str(settings.db_path),
                )
            except Exception as exc:
                _safe_append_activity(
                    data_paths["activity_log"],
                    event_type="review",
                    title="Replay backfill failed",
                    message=f"Replay outcome update skipped: {_humanize_error_text(exc)}",
                    status="failed",
                    related_file=str(settings.db_path),
                )
        else:
            _safe_append_activity(
                data_paths["activity_log"],
                event_type="review",
                title="Replay backfill skipped",
                message="Skipped because Alpaca live market data was not connected for this run.",
                status="info",
                related_file=str(settings.db_path),
            )

        data_mode = _determine_data_mode(live_sources_used, used_fallback_data)
        _log_data_mode(data_paths["activity_log"], data_mode, used_fallback_data, live_sources_used)

        stage = "report_generation"
        csv_path, html_path = build_outputs(
            shortlist_df,
            output_csv=source_csv,
            output_html=report_html,
            mock_mode=settings.mock_mode,
            data_mode=data_mode,
        )
        _safe_append_activity(
            data_paths["activity_log"],
            event_type="report",
            title="Report generated",
            message=f"Saved HTML report to {html_path}.",
            status="success",
            related_file=str(html_path),
        )

        stage = "latest_output"
        write_csv_atomic(shortlist_df, latest_file)
        _safe_append_activity(
            data_paths["activity_log"],
            event_type="run",
            title="Latest shortlist generated",
            message=f"Updated latest shortlist at {latest_file}.",
            status="success",
            related_file=str(latest_file),
        )

        stage = "context_output"
        context_payload = _build_detail_context(shortlist_df, market_context, combined_news, generated_at, data_mode)
        write_context_snapshot(data_paths["latest_context"], context_payload)
        _safe_append_activity(
            data_paths["activity_log"],
            event_type="run",
            title="Latest context generated",
            message=f"Updated detail context at {data_paths['latest_context']}.",
            status="success",
            related_file=str(data_paths["latest_context"]),
        )

        stage = "history_archive"
        write_csv_atomic(shortlist_df, history_file)
        _safe_append_activity(
            data_paths["activity_log"],
            event_type="archive",
            title="History shortlist archived",
            message=f"Archived shortlist to {history_file}.",
            status="success",
            related_file=str(history_file),
        )

        status_payload = {
            "success": True,
            "run_mode": run_mode,
            "generated_at": generated_at.isoformat(timespec="seconds"),
            "api_probe_time": api_probe_time,
            "row_count": row_count,
            "candidate_count": candidate_count,
            "news_items_ingested": news_items_ingested,
            "ml_status": ml_status,
            "ml_message": ml_message,
            "source_file": str(csv_path),
            "latest_file": str(latest_file),
            "history_file": str(history_file),
            "report_file": str(html_path),
            "activity_file": str(data_paths["activity_log"]),
            "context_file": str(data_paths["latest_context"]),
            "db_file": str(settings.db_path),
            "alpaca_status": alpaca_status,
            "alpaca_message": alpaca_message,
            "news_status": news_status,
            "news_message": news_message,
            "sec_status": sec_status,
            "sec_message": sec_message,
            "benzinga_status": benzinga_status,
            "benzinga_message": benzinga_message,
            "data_mode": data_mode,
            "used_fallback_data": used_fallback_data,
            "error": None,
        }
        _safe_write_status(data_paths["run_status"], status_payload)
        _safe_append_activity(
            data_paths["activity_log"],
            event_type="run",
            title="Shortlist run succeeded",
            message=f"Run completed successfully with {row_count} candidates and {news_items_ingested} news items.",
            status="success",
            related_file=str(latest_file),
        )

        print(f"Saved CSV: {csv_path}")
        print(f"Saved HTML: {html_path}")
        print(f"Saved latest shortlist: {latest_file}")
        print(f"Saved history archive: {history_file}")
        print(f"Saved run status: {data_paths['run_status']}")
        print(f"Saved sqlite DB: {settings.db_path}")
        print(f"Rows written: {row_count}")
        print(f"News items today: {load_news_count_today(settings.db_path)}")
        return 0
    except Exception as exc:
        error_message = f"{exc.__class__.__name__}: {_safe_error_text(exc)}"
        _safe_write_status(
            data_paths["run_status"],
            {
                "success": False,
                "run_mode": run_mode,
                "generated_at": generated_at.isoformat(timespec="seconds"),
                "api_probe_time": api_probe_time,
                "row_count": row_count,
                "candidate_count": candidate_count,
                "news_items_ingested": news_items_ingested,
                "ml_status": ml_status,
                "ml_message": ml_message,
                "source_file": str(source_csv),
                "latest_file": str(latest_file),
                "history_file": str(history_file),
                "report_file": str(report_html),
                "activity_file": str(data_paths["activity_log"]),
                "context_file": str(data_paths["latest_context"]),
                "db_file": str(settings.db_path),
                "alpaca_status": alpaca_status,
                "alpaca_message": alpaca_message,
                "news_status": news_status,
                "news_message": news_message,
                "sec_status": sec_status,
                "sec_message": sec_message,
                "benzinga_status": benzinga_status,
                "benzinga_message": benzinga_message,
                "data_mode": data_mode,
                "used_fallback_data": used_fallback_data,
                "error": f"{stage}: {error_message}",
            },
        )
        _safe_append_activity(
            data_paths["activity_log"],
            event_type="run",
            title="Shortlist run failed",
            message=f"{stage}: {error_message}",
            status="failed",
            related_file=str(project_root / "main.py"),
        )
        print(f"Runtime error: {stage}: {error_message}")
        return 1


def _load_market_movers(
    settings: EngineSettings,
    market_client: AlpacaMarketDataClient,
    probe_result: dict[str, str],
) -> tuple[pd.DataFrame, str, str, bool, bool]:
    """Fetch market movers with fallback to the existing mock sample."""
    if settings.mock_mode:
        return market_client.fetch_movers(), "skipped", "mock mode enabled", False, False
    if probe_result.get("status") != "connected":
        fallback = AlpacaClient("", "", settings.alpaca_base_url, mock_mode=True).get_movers()
        return fallback, str(probe_result.get("status") or "failed"), str(probe_result.get("message") or "probe failed"), False, True
    try:
        movers = market_client.fetch_movers()
        if movers.empty:
            fallback = AlpacaClient("", "", settings.alpaca_base_url, mock_mode=True).get_movers()
            return fallback, "failed", "live movers returned no rows; using fallback sample", False, True
        return movers, "connected", "probe succeeded", True, False
    except Exception as exc:
        fallback = AlpacaClient("", "", settings.alpaca_base_url, mock_mode=True).get_movers()
        return fallback, "failed", f"live movers fetch failed: {_humanize_error_text(exc)}", False, True


def _load_alpaca_news(settings: EngineSettings, news_client: AlpacaNewsClient) -> pd.DataFrame:
    """Fetch general Alpaca news to seed candidates and catalysts."""
    if settings.mock_mode:
        return _mock_news_frame([])
    try:
        return news_client.fetch_news([], settings.alpaca_news_limit)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "symbol", "headline", "source", "url", "catalyst_type", "headline_strength", "freshness_minutes", "raw_json"])


def _load_sec_filings(settings: EngineSettings, sec_client: SecRssClient) -> pd.DataFrame:
    """Fetch recent SEC filing catalysts."""
    if settings.mock_mode:
        return pd.DataFrame(columns=["timestamp", "symbol", "headline", "source", "url", "catalyst_type", "headline_strength", "freshness_minutes", "form_type", "raw_json"])
    try:
        return sec_client.fetch_filings([], settings.sec_feed_limit)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "symbol", "headline", "source", "url", "catalyst_type", "headline_strength", "freshness_minutes", "form_type", "raw_json"])


def _combine_news_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    valid_frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["timestamp", "symbol", "headline", "source", "url", "catalyst_type", "headline_strength", "freshness_minutes", "raw_json"])
    combined = pd.concat(valid_frames, ignore_index=True)
    if "symbol" in combined.columns:
        combined["symbol"] = combined["symbol"].astype(str).str.upper().str.strip()
        combined = combined[combined["symbol"] != ""]
    return combined.drop_duplicates(subset=["symbol", "headline", "source"], keep="first")


def _build_seed_symbols(movers_df: pd.DataFrame, news_df: pd.DataFrame, limit: int) -> list[str]:
    symbols: list[str] = []
    if not movers_df.empty and "ticker" in movers_df.columns:
        for symbol in movers_df["ticker"].astype(str).str.upper().tolist():
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    if not news_df.empty and "symbol" in news_df.columns:
        ranked_news = news_df.sort_values(["headline_strength", "freshness_minutes"], ascending=[False, True])
        for symbol in ranked_news["symbol"].astype(str).str.upper().tolist():
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    for symbol in _core_watchlist_symbols():
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols[:limit]


def _build_market_context(
    settings: EngineSettings,
    market_client: AlpacaMarketDataClient,
    movers_df: pd.DataFrame,
    seed_symbols: list[str],
) -> dict[str, Any]:
    """Fetch or synthesize market context for the candidate seed universe."""
    context_symbols = list(dict.fromkeys(seed_symbols + ["SPY", "QQQ", "IWM", "SMH"]))
    if settings.mock_mode:
        return _mock_market_context(movers_df)
    try:
        context = market_client.fetch_market_context(context_symbols, settings.history_minutes, settings.daily_lookback_days)
        return _blend_market_context_with_movers(context, movers_df)
    except Exception:
        return _mock_market_context(movers_df)


def _mock_market_context(movers_df: pd.DataFrame) -> dict[str, Any]:
    """Build a minimal market context from the mock/live movers frame when richer data is missing."""
    snapshots: dict[str, dict[str, Any]] = {}
    latest_bars: dict[str, dict[str, Any]] = {}
    latest_quotes: dict[str, dict[str, Any]] = {}
    intraday_bars: dict[str, list[dict[str, Any]]] = {}
    daily_bars: dict[str, list[dict[str, Any]]] = {}
    for _, row in movers_df.iterrows():
        symbol = str(row.get("ticker") or "").upper()
        price = float(row.get("current_price") or 0.0)
        volume = float(row.get("volume") or 0.0)
        move_pct = float(row.get("move_percent") or 0.0)
        prev_close = price / (1 + move_pct / 100.0) if price > 0 and move_pct else price
        snapshots[symbol] = {
            "latestTrade": {"p": price},
            "latestQuote": {"bp": price * 0.999, "ap": price * 1.001},
            "minuteBar": {"c": price, "h": price * 1.002, "l": price * 0.998, "v": max(volume * 0.02, 1000)},
            "dailyBar": {"c": price, "h": price * 1.01, "l": price * 0.99, "v": volume, "vw": price},
            "prevDailyBar": {"c": prev_close, "v": max(volume * 0.8, 1000)},
        }
        latest_bars[symbol] = {"c": price, "h": price * 1.002, "l": price * 0.998, "o": prev_close or price, "v": max(volume * 0.02, 1000), "t": datetime.now().astimezone().isoformat(timespec="seconds")}
        latest_quotes[symbol] = {"bp": price * 0.999, "ap": price * 1.001}
        intraday_bars[symbol] = [
            {
                "t": datetime.now().astimezone().isoformat(timespec="seconds"),
                "o": prev_close or price,
                "h": price * 1.005,
                "l": price * 0.995,
                "c": price,
                "v": max(volume * 0.05, 5000),
                "vw": price,
            }
        ]
        daily_bars[symbol] = [
            {"t": datetime.now().astimezone().isoformat(timespec="seconds"), "o": prev_close or price, "h": price * 1.01, "l": price * 0.99, "c": price, "v": max(volume, 10000), "vw": price}
        ]
    return {
        "snapshots": snapshots,
        "latest_bars": latest_bars,
        "latest_quotes": latest_quotes,
        "intraday_bars": intraday_bars,
        "daily_bars": daily_bars,
    }


def _blend_market_context_with_movers(context: dict[str, Any], movers_df: pd.DataFrame) -> dict[str, Any]:
    """Fill obvious gaps in live market context with mover fields."""
    blended = {key: value.copy() if isinstance(value, dict) else value for key, value in context.items()}
    for _, row in movers_df.iterrows():
        symbol = str(row.get("ticker") or "").upper()
        if not symbol:
            continue
        blended.setdefault("snapshots", {}).setdefault(symbol, {})
        blended.setdefault("latest_bars", {}).setdefault(symbol, {})
        blended.setdefault("latest_quotes", {}).setdefault(symbol, {})
        snapshot = blended["snapshots"][symbol]
        snapshot.setdefault("latestTrade", {"p": float(row.get("current_price") or 0.0)})
        snapshot.setdefault("dailyBar", {"c": float(row.get("current_price") or 0.0), "v": float(row.get("volume") or 0.0), "vw": float(row.get("current_price") or 0.0)})
        snapshot.setdefault("prevDailyBar", {"c": float(row.get("current_price") or 0.0) / (1 + float(row.get("move_percent") or 0.0) / 100.0) if float(row.get("current_price") or 0.0) > 0 else 0.0})
        blended["latest_bars"][symbol].setdefault("c", float(row.get("current_price") or 0.0))
        blended["latest_bars"][symbol].setdefault("v", float(row.get("volume") or 0.0))
        if symbol not in blended.get("intraday_bars", {}):
            blended.setdefault("intraday_bars", {})[symbol] = []
        if symbol not in blended.get("daily_bars", {}):
            blended.setdefault("daily_bars", {})[symbol] = []
    return blended


def _mock_news_frame(symbols: list[str]) -> pd.DataFrame:
    records = []
    for symbol in symbols:
        records.append(
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "symbol": symbol,
                "headline": f"{symbol} remains on watch after a strong mover screen reading",
                "source": "Mock News",
                "url": "",
                "catalyst_type": "general",
                "headline_strength": 3.0,
                "freshness_minutes": 30.0,
                "raw_json": "{}",
            }
        )
    return pd.DataFrame(records)


def _core_watchlist_symbols() -> list[str]:
    """Return a compact liquid watchlist used when movers are low quality."""
    return [
        "AAPL",
        "NVDA",
        "MSFT",
        "AMZN",
        "META",
        "TSLA",
        "AMD",
        "AVGO",
        "PLTR",
        "COIN",
        "MSTR",
        "SMCI",
        "HOOD",
        "NFLX",
        "GOOGL",
        "MU",
        "ARM",
        "RDDT",
        "SOFI",
        "UBER",
    ]


def _finalize_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize final shortlist columns for CSV, HTML, and Streamlit."""
    if df.empty:
        return pd.DataFrame(columns=[
            "rank", "list_type", "symbol", "ticker", "last_price", "current_price", "gap_pct", "move_percent",
            "volume_today", "relative_volume", "spread_pct", "bid_price", "ask_price", "quote_time",
            "catalyst", "catalyst_type", "news_score", "setup_score",
            "liquidity_score", "risk_score", "ml_score", "breakout_probability", "failure_probability",
            "predicted_upper_band", "predicted_lower_band", "total_score", "status_tag", "tradeable", "tradeable_reason",
            "not_tradeable_reason", "selection_reason", "risk_note", "action_note", "base_low", "base_high",
            "breakout_low", "breakout_high", "pullback_low", "pullback_high", "invalidation", "confidence",
            "last_update_time", "vwap", "news_count_30m", "freshness_minutes", "headline_strength", "source_count",
            "ret_1m", "ret_5m", "ret_15m", "range_5m", "range_15m", "distance_to_vwap_pct", "distance_to_intraday_high_pct",
            "distance_to_premarket_high_pct", "breakout_ready_score", "trade_count_recent", "volatility_1m", "volatility_5m", "volatility_15m",
            "overextension_score", "wickiness_score", "halt_risk_proxy", "dollar_volume", "volume", "setup_bias",
            "index_regime", "market_trend_strength", "sector_strength", "breadth_proxy", "vwap_regime_flag", "sec_filing_type"
        ])
    finalized = df.copy()
    finalized["ticker"] = finalized.get("ticker", finalized.get("symbol", ""))
    finalized["current_price"] = finalized.get("current_price", finalized.get("last_price", 0.0))
    finalized["bid_price"] = finalized.get("bid_price", 0.0)
    finalized["ask_price"] = finalized.get("ask_price", 0.0)
    finalized["quote_time"] = finalized.get("quote_time", "")
    finalized["ml_score"] = finalized.get("ml_score", 0.0)
    finalized["breakout_probability"] = finalized.get("breakout_probability", 0.0)
    finalized["failure_probability"] = finalized.get("failure_probability", 0.0)
    finalized["predicted_upper_band"] = finalized.get("predicted_upper_band", finalized.get("base_high", 0.0))
    finalized["predicted_lower_band"] = finalized.get("predicted_lower_band", finalized.get("base_low", 0.0))
    finalized["score"] = finalized.get("total_score", 0.0)
    finalized["liquidity_flag"] = finalized.apply(lambda row: "liquid" if bool(row.get("tradeable")) else "review", axis=1)
    finalized["setup_type"] = finalized.get("setup_bias", "neutral")
    finalized["risk_note"] = finalized.get("risk_note", "Manual review required")
    return finalized


def _build_detail_context(
    shortlist_df: pd.DataFrame,
    market_context: dict[str, Any],
    news_df: pd.DataFrame,
    generated_at: datetime,
    data_mode: str,
) -> dict[str, Any]:
    """Persist lightweight symbol context for the dashboard detail panel."""
    payload: dict[str, Any] = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "data_mode": data_mode,
        "candidate_count": len(shortlist_df),
        "news_items": len(news_df),
        "symbols": {},
    }
    if shortlist_df.empty:
        return payload

    news_by_symbol: dict[str, list[dict[str, Any]]] = {}
    if not news_df.empty and "symbol" in news_df.columns:
        cleaned_news = news_df.copy()
        cleaned_news["symbol"] = cleaned_news["symbol"].astype(str).str.upper().str.strip()
        cleaned_news = cleaned_news.sort_values(["freshness_minutes", "headline_strength"], ascending=[True, False])
        for symbol, frame in cleaned_news.groupby("symbol"):
            news_by_symbol[str(symbol)] = frame.head(8).to_dict("records")

    snapshots = market_context.get("snapshots", {}) or {}
    latest_bars = market_context.get("latest_bars", {}) or {}
    latest_quotes = market_context.get("latest_quotes", {}) or {}
    intraday_bars = market_context.get("intraday_bars", {}) or {}
    daily_bars = market_context.get("daily_bars", {}) or {}

    for _, row in shortlist_df.iterrows():
        symbol = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
        if not symbol:
            continue
        payload["symbols"][symbol] = {
            "snapshot": snapshots.get(symbol, {}),
            "latest_bar": latest_bars.get(symbol, {}),
            "latest_quote": latest_quotes.get(symbol, {}),
            "intraday_bars": (intraday_bars.get(symbol, []) or [])[-60:],
            "daily_bars": (daily_bars.get(symbol, []) or [])[-20:],
            "news": news_by_symbol.get(symbol, []),
            "summary": {
                "status_tag": row.get("status_tag"),
                "tradeable": bool(row.get("tradeable")),
                "total_score": float(row.get("total_score") or 0.0),
                "catalyst_type": row.get("catalyst_type"),
            },
        }
    return payload


def _combine_news_status(news_probe: dict[str, str], sec_probe: dict[str, str]) -> tuple[str, str]:
    statuses = [str(news_probe.get("status") or "unavailable"), str(sec_probe.get("status") or "unavailable")]
    if "connected" in statuses:
        status = "connected"
    elif all(item == "skipped" for item in statuses):
        status = "skipped"
    elif any(item == "failed" for item in statuses):
        status = "failed"
    else:
        status = "unavailable"
    message = f"alpaca news: {news_probe.get('message', '')}; sec rss: {sec_probe.get('message', '')}"
    return status, message


def _determine_data_mode(live_sources_used: set[str], used_fallback_data: bool) -> str:
    if {"alpaca", "news"}.issubset(live_sources_used) and not used_fallback_data:
        return "live"
    if not live_sources_used:
        return "mock"
    return "mixed"


def _log_provider_probe(activity_file: Path, provider: str, probe_result: dict[str, str]) -> None:
    status = str(probe_result.get("status") or "unavailable")
    if status == "connected":
        activity_status = "success"
        title = f"{provider} probe succeeded"
    elif status == "failed":
        activity_status = "failed"
        title = f"{provider} probe failed"
    elif status == "skipped":
        activity_status = "info"
        title = f"{provider} probe skipped"
    else:
        activity_status = "info"
        title = f"{provider} probe unavailable"
    _safe_append_activity(
        activity_file,
        event_type="probe",
        title=title,
        message=str(probe_result.get("message") or ""),
        status=activity_status,
        related_file="",
    )


def _log_data_mode(activity_file: Path, data_mode: str, used_fallback_data: bool, live_sources_used: set[str]) -> None:
    if data_mode == "live":
        title = "Run used live data"
        message = "Market data and news catalysts were supplied by live providers."
    elif data_mode == "mixed":
        title = "Run used mixed data"
        message = "At least one live source succeeded and fallback or empty-source handling was also used."
    else:
        title = "Run used mock-only data"
        message = "No live source contributed to the final shortlist."
    _safe_append_activity(
        activity_file,
        event_type="run",
        title=title,
        message=message if used_fallback_data or data_mode != "live" else f"Live sources used: {', '.join(sorted(live_sources_used))}",
        status="success" if data_mode == "live" else "info",
        related_file="",
    )


def _safe_error_text(exc: Exception) -> str:
    text = str(exc)
    text = re.sub(r"(?i)(token|api[_-]?key|secret|password)=([^&\s]+)", r"\1=[REDACTED]", text)
    text = re.sub(r"(?i)(bearer\s+)[a-z0-9._-]+", r"\1[REDACTED]", text)
    return text


def _humanize_error_text(exc: Exception) -> str:
    text = _safe_error_text(exc).lower()
    if "401" in text or "unauthorized" in text:
        return "401 unauthorized"
    if "403" in text or "forbidden" in text:
        return "403 forbidden"
    if "timeout" in text:
        return "timeout"
    if "failed to resolve" in text or "nameresolutionerror" in text or "connectionerror" in text:
        return "connection error"
    if "ssl" in text:
        return "ssl error"
    if "404" in text or "not found" in text:
        return "404 not found"
    return _safe_error_text(exc)[:120]


def _safe_append_activity(activity_file: Path, event_type: str, title: str, message: str, status: str, related_file: str) -> None:
    try:
        append_activity_event(
            activity_file,
            event_type=event_type,
            title=title,
            message=message,
            status=status,
            related_file=related_file,
        )
    except Exception:
        return


def _safe_write_status(status_file: Path, payload: dict[str, Any]) -> None:
    try:
        write_run_status(status_file, payload)
    except Exception:
        return


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except ImportError as exc:
        print(
            "Dependency error: missing required package. "
            "Install requirements.txt before running. "
            f"Details: {exc}"
        )
        raise SystemExit(1)
    except Exception as exc:
        print(f"Runtime error: {exc}")
        raise SystemExit(1)
