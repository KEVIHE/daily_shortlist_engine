"""Alpaca market data helpers for intraday shortlist analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from src.alpaca_client import AlpacaClient


@dataclass
class AlpacaMarketDataClient:
    """Wrapper around Alpaca market data endpoints used by the workstation."""

    api_key: str
    api_secret: str
    base_url: str
    mock_mode: bool = True

    def __post_init__(self) -> None:
        self._auth_client = AlpacaClient(
            api_key=self.api_key,
            api_secret=self.api_secret,
            base_url=self.base_url,
            mock_mode=self.mock_mode,
        )

    def probe(self) -> dict[str, str]:
        """Probe the movers endpoint through the trading header auth flow."""
        return self._auth_client.probe()

    def fetch_movers(self) -> pd.DataFrame:
        """Fetch top movers using the shared Alpaca market client."""
        return self._auth_client.get_movers()

    def fetch_snapshots(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch latest trade/quote/bar snapshot data for the supplied symbols."""
        payload = self._request_multi_symbol_json("/v2/stocks/snapshots", symbols)
        return _extract_symbol_mapping(payload, "snapshots")

    def fetch_latest_bars(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch latest minute bars for the supplied symbols."""
        payload = self._request_multi_symbol_json("/v2/stocks/bars/latest", symbols)
        return _extract_symbol_mapping(payload, "bars")

    def fetch_latest_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch latest quotes for the supplied symbols."""
        payload = self._request_multi_symbol_json("/v2/stocks/quotes/latest", symbols)
        return _extract_symbol_mapping(payload, "quotes")

    def fetch_intraday_bars(self, symbols: list[str], history_minutes: int) -> dict[str, list[dict[str, Any]]]:
        """Fetch recent intraday minute bars for feature engineering."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=history_minutes)
        return self.fetch_historical_bars(symbols, "1Min", start, end, limit=max(history_minutes, 30))

    def fetch_daily_bars(self, symbols: list[str], lookback_days: int) -> dict[str, list[dict[str, Any]]]:
        """Fetch recent daily bars to estimate average volume and trend context."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(lookback_days * 3, 14))
        return self.fetch_historical_bars(symbols, "1Day", start, end, limit=max(lookback_days, 5))

    def fetch_historical_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch historical bars and return a mapping keyed by symbol."""
        request_symbols = _clean_symbols(symbols)
        if not request_symbols:
            return {}

        results: dict[str, list[dict[str, Any]]] = {}
        for chunk in _chunked(request_symbols, 100):
            payload = self._request_json(
                "/v2/stocks/bars",
                params={
                    "symbols": ",".join(chunk),
                    "timeframe": timeframe,
                    "start": _iso_utc(start),
                    "end": _iso_utc(end),
                    "adjustment": "raw",
                    "sort": "asc",
                    "limit": limit,
                },
                timeout=20,
            )
            symbol_map = _extract_symbol_mapping(payload, "bars")
            for symbol, bars in symbol_map.items():
                results.setdefault(symbol, []).extend(bars if isinstance(bars, list) else [])
        return results

    def fetch_market_context(self, symbols: list[str], history_minutes: int, daily_lookback_days: int) -> dict[str, Any]:
        """Fetch the combined market context used by the analysis engine."""
        cleaned = _clean_symbols(symbols)
        if not cleaned:
            return {
                "snapshots": {},
                "latest_bars": {},
                "latest_quotes": {},
                "intraday_bars": {},
                "daily_bars": {},
            }
        return {
            "snapshots": self.fetch_snapshots(cleaned),
            "latest_bars": self.fetch_latest_bars(cleaned),
            "latest_quotes": self.fetch_latest_quotes(cleaned),
            "intraday_bars": self.fetch_intraday_bars(cleaned, history_minutes),
            "daily_bars": self.fetch_daily_bars(cleaned, daily_lookback_days),
        }

    def stream_config(self) -> dict[str, str]:
        """Reserve websocket endpoints for V2 streaming without enabling them yet."""
        return {
            "stocks": "wss://stream.data.alpaca.markets/v2/sip",
            "news": "wss://stream.data.alpaca.markets/v1beta1/news",
        }

    def _request_multi_symbol_json(self, endpoint: str, symbols: list[str]) -> dict[str, Any]:
        cleaned = _clean_symbols(symbols)
        if not cleaned:
            return {}
        merged: dict[str, Any] = {}
        for chunk in _chunked(cleaned, 100):
            payload = self._request_json(endpoint, {"symbols": ",".join(chunk)}, timeout=15)
            symbol_map = _extract_symbol_mapping(payload)
            if isinstance(symbol_map, dict):
                merged.update(symbol_map)
        return merged

    def _request_json(self, endpoint: str, params: dict[str, Any], timeout: int) -> dict[str, Any]:
        if self.mock_mode:
            return {}
        response = requests.get(
            f"{self.base_url.rstrip('/')}{endpoint}",
            headers=self._auth_client.headers(),
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json() or {}
        return payload if isinstance(payload, dict) else {"data": payload}


def _clean_symbols(symbols: list[str]) -> list[str]:
    cleaned: list[str] = []
    for symbol in symbols:
        text = str(symbol or "").strip().upper()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _extract_symbol_mapping(payload: dict[str, Any], preferred_key: str | None = None) -> dict[str, Any]:
    if preferred_key and isinstance(payload.get(preferred_key), dict):
        return payload[preferred_key]
    for key in (preferred_key, "snapshots", "bars", "quotes"):
        if key and isinstance(payload.get(key), dict):
            return payload[key]
    if all(isinstance(key, str) for key in payload.keys()):
        return payload
    return {}


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
