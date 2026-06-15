"""Alpaca market-data auth and movers helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse

import pandas as pd
import requests


@dataclass
class AlpacaClient:
    """Fetch Alpaca stock movers using Trading API header auth."""

    api_key: str
    api_secret: str
    base_url: str
    mock_mode: bool = True

    def probe(self) -> dict[str, str]:
        """Run a lightweight connectivity check for the movers endpoint."""
        if self.mock_mode:
            return {"status": "skipped", "message": "mock mode enabled"}

        missing_fields = _missing_alpaca_fields(self.api_key, self.api_secret, self.base_url)
        if missing_fields:
            return {
                "status": "unavailable",
                "message": "credentials missing: " + ", ".join(missing_fields),
            }

        route_issue = _validate_market_data_route(self.api_key, self.base_url)
        if route_issue:
            return {"status": "failed", "message": route_issue}

        try:
            response = requests.get(
                f"{self.base_url.rstrip('/')}/v1beta1/screener/stocks/movers",
                headers=self.headers(),
                timeout=8,
            )
            response.raise_for_status()
            return {"status": "connected", "message": "probe succeeded"}
        except Exception as exc:
            return {"status": "failed", "message": f"probe failed: {_humanize_error_text(exc)}"}

    def headers(self) -> dict[str, str]:
        """Return Trading API style market-data headers."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "accept": "application/json",
        }

    def get_movers(self) -> pd.DataFrame:
        """Return a normalized movers dataframe."""
        if self.mock_mode:
            return self._mock_movers()

        route_issue = _validate_market_data_route(self.api_key, self.base_url)
        if route_issue:
            raise ValueError(route_issue)

        response = requests.get(
            f"{self.base_url.rstrip('/')}/v1beta1/screener/stocks/movers",
            headers=self.headers(),
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json() or {}

        rows: list[dict[str, object]] = []
        for group_name in ("gainers", "losers", "most_actives"):
            for item in payload.get(group_name, []) or []:
                rows.append(
                    {
                        "ticker": item.get("symbol", ""),
                        "current_price": float(item.get("price") or 0),
                        "move_percent": float(item.get("percent_change") or item.get("change_percent") or 0),
                        "volume": float(item.get("volume") or 0),
                        "dollar_volume": float(item.get("dollar_volume") or 0),
                        "session": item.get("session") or "regular",
                        "mover_group": group_name,
                    }
                )

        movers = pd.DataFrame(rows).drop_duplicates(subset=["ticker"], keep="first")
        if movers.empty:
            return movers

        snapshots = self._fetch_snapshots(movers["ticker"].tolist())
        if not snapshots:
            return movers

        merged_rows = [
            self._merge_snapshot_fields(row, snapshots.get(str(row["ticker"]), {}))
            for _, row in movers.iterrows()
        ]
        return pd.DataFrame(merged_rows)

    def _mock_movers(self) -> pd.DataFrame:
        """Return mock movers for offline runs."""
        rows = [
            {
                "ticker": "NVDA",
                "current_price": 121.4,
                "move_percent": 5.8,
                "volume": 58000000,
                "dollar_volume": 7041200000,
                "session": "premarket",
                "mover_group": "gainers",
            },
            {
                "ticker": "SMCI",
                "current_price": 48.1,
                "move_percent": 7.2,
                "volume": 32000000,
                "dollar_volume": 1539200000,
                "session": "regular",
                "mover_group": "gainers",
            },
            {
                "ticker": "PLTR",
                "current_price": 31.2,
                "move_percent": 3.4,
                "volume": 41000000,
                "dollar_volume": 1279200000,
                "session": "regular",
                "mover_group": "most_actives",
            },
            {
                "ticker": "TSLA",
                "current_price": 179.9,
                "move_percent": -4.1,
                "volume": 69000000,
                "dollar_volume": 12413100000,
                "session": "premarket",
                "mover_group": "losers",
            },
            {
                "ticker": "SOUN",
                "current_price": 4.2,
                "move_percent": 12.5,
                "volume": 24000000,
                "dollar_volume": 100800000,
                "session": "regular",
                "mover_group": "gainers",
            },
        ]
        return pd.DataFrame(rows)

    def _fetch_snapshots(self, tickers: list[str]) -> dict[str, dict[str, object]]:
        """Fetch lightweight snapshots so live movers retain price and liquidity context."""
        symbols = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
        if not symbols:
            return {}

        response = requests.get(
            f"{self.base_url.rstrip('/')}/v2/stocks/snapshots",
            headers=self.headers(),
            params={"symbols": ",".join(symbols)},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json() or {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _merge_snapshot_fields(row: pd.Series, snapshot: dict[str, object]) -> dict[str, object]:
        """Merge one snapshot payload into the movers row."""
        merged = row.to_dict()
        daily_bar = snapshot.get("dailyBar") if isinstance(snapshot, dict) else {}
        latest_trade = snapshot.get("latestTrade") if isinstance(snapshot, dict) else {}
        if isinstance(daily_bar, dict):
            volume = float(daily_bar.get("v") or 0)
            volume_weighted_price = float(daily_bar.get("vw") or merged.get("current_price") or 0)
            if volume > 0:
                merged["volume"] = volume
                merged["dollar_volume"] = volume * max(volume_weighted_price, 0)
        if isinstance(latest_trade, dict):
            latest_price = float(latest_trade.get("p") or 0)
            if latest_price > 0:
                merged["current_price"] = latest_price
        return merged


def _safe_error_text(exc: Exception) -> str:
    """Remove secret-like fragments from exception text."""
    text = str(exc)
    text = re.sub(r"(?i)(token|api[_-]?key|secret|password)=([^&\\s]+)", r"\1=[REDACTED]", text)
    text = re.sub(r"(?i)(bearer\s+)[a-z0-9._-]+", r"\1[REDACTED]", text)
    return text


def _missing_alpaca_fields(api_key: str, api_secret: str, base_url: str) -> list[str]:
    """Return missing required Alpaca configuration names."""
    missing = []
    if not api_key:
        missing.append("ALPACA_API_KEY")
    if not api_secret:
        missing.append("ALPACA_API_SECRET")
    if not str(base_url).strip():
        missing.append("ALPACA_BASE_URL")
    return missing


def _validate_market_data_route(api_key: str, base_url: str) -> str | None:
    """Reject broker-specific credentials or hosts in this project mode."""
    hostname = urlparse(str(base_url).strip()).netloc.lower()
    if any(token in hostname for token in ["broker", "authx", "paper-api"]):
        return "unsupported base URL for market data; use https://data.alpaca.markets"
    if str(api_key or "").strip().upper().startswith("C"):
        return "broker-style credential detected; paste Trading API key/secret into .env"
    return None


def _humanize_error_text(exc: Exception) -> str:
    """Return a short readable failure reason."""
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
