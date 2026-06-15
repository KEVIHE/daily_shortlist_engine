"""Alpaca news ingestion for catalyst-aware shortlist scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from src.alpaca_client import AlpacaClient, _humanize_error_text


CATALYST_KEYWORDS = {
    "m_and_a": ["acquire", "acquisition", "merger", "buyout", "take private"],
    "earnings": ["earnings", "guidance", "revenue", "eps", "quarter"],
    "analyst": ["upgrade", "downgrade", "price target", "analyst"],
    "contract": ["contract", "award", "partnership", "deal", "agreement"],
    "product": ["launch", "approval", "fda", "trial", "phase", "product"],
    "filing": ["8-k", "10-q", "10-k", "s-3", "13d", "13g"],
    "macro": ["fed", "tariff", "inflation", "rate", "macro"],
}


@dataclass
class AlpacaNewsClient:
    """Fetch and normalize Alpaca news data for shortlist candidates."""

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
        """Probe the Alpaca news endpoint."""
        if self.mock_mode:
            return {"status": "skipped", "message": "mock mode enabled"}
        market_probe = self._auth_client.probe()
        if market_probe["status"] != "connected":
            return market_probe
        try:
            response = requests.get(
                f"{self.base_url.rstrip('/')}/v1beta1/news",
                headers=self._auth_client.headers(),
                params={"symbols": "AAPL", "limit": 1, "sort": "desc"},
                timeout=10,
            )
            response.raise_for_status()
            return {"status": "connected", "message": "probe succeeded"}
        except Exception as exc:
            return {"status": "failed", "message": f"probe failed: {_humanize_error_text(exc)}"}

    def fetch_news(self, symbols: list[str], limit: int) -> pd.DataFrame:
        """Fetch Alpaca news and normalize it into a symbol-level dataframe."""
        if self.mock_mode:
            return pd.DataFrame(columns=_news_columns())

        cleaned_symbols = _clean_symbols(symbols)
        records: list[dict[str, Any]] = []
        symbol_chunks = _chunked(cleaned_symbols, 50) if cleaned_symbols else [[]]

        for chunk in symbol_chunks:
            params: dict[str, Any] = {"limit": limit, "sort": "desc", "include_content": "false"}
            if chunk:
                params["symbols"] = ",".join(chunk)
            response = requests.get(
                f"{self.base_url.rstrip('/')}/v1beta1/news",
                headers=self._auth_client.headers(),
                params=params,
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json() or []
            articles = payload.get("news", []) if isinstance(payload, dict) else payload
            for article in articles or []:
                article_symbols = article.get("symbols") or chunk
                for symbol in article_symbols:
                    text = str(symbol or "").strip().upper()
                    if not text:
                        continue
                    headline = str(article.get("headline") or article.get("summary") or article.get("title") or "").strip()
                    published_at = _parse_timestamp(article.get("created_at") or article.get("updated_at") or article.get("timestamp"))
                    records.append(
                        {
                            "timestamp": published_at.isoformat(timespec="seconds") if published_at else "",
                            "symbol": text,
                            "headline": headline or "No headline",
                            "source": str(article.get("source") or "Alpaca News"),
                            "url": str(article.get("url") or ""),
                            "catalyst_type": classify_catalyst_type(headline),
                            "headline_strength": headline_strength(headline),
                            "freshness_minutes": _freshness_minutes(published_at),
                            "raw_json": json.dumps(article, ensure_ascii=True),
                        }
                    )
        if not records:
            return pd.DataFrame(columns=_news_columns())
        frame = pd.DataFrame(records).drop_duplicates(subset=["symbol", "headline", "source"], keep="first")
        return frame.sort_values(["symbol", "timestamp"], ascending=[True, False]).reset_index(drop=True)

    def stream_config(self) -> dict[str, str]:
        """Reserve the real-time news websocket endpoint for V2."""
        return {"news": "wss://stream.data.alpaca.markets/v1beta1/news"}


def classify_catalyst_type(headline: str) -> str:
    text = str(headline or "").lower()
    for label, keywords in CATALYST_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return label
    return "general"


def headline_strength(headline: str) -> float:
    text = str(headline or "").lower()
    score = 1.5
    for keywords in CATALYST_KEYWORDS.values():
        score += sum(keyword in text for keyword in keywords) * 1.2
    if any(term in text for term in ["surge", "jump", "soar", "beats", "wins", "approval"]):
        score += 1.0
    return round(min(score, 10.0), 2)


def _freshness_minutes(published_at: datetime | None) -> float:
    if not published_at:
        return 9999.0
    return round(max((datetime.now(timezone.utc) - published_at).total_seconds() / 60.0, 0.0), 2)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _clean_symbols(symbols: list[str]) -> list[str]:
    cleaned: list[str] = []
    for symbol in symbols:
        text = str(symbol or "").strip().upper()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _news_columns() -> list[str]:
    return [
        "timestamp",
        "symbol",
        "headline",
        "source",
        "url",
        "catalyst_type",
        "headline_strength",
        "freshness_minutes",
        "raw_json",
    ]
