"""Benzinga client for news and catalyst enrichment."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
import requests


@dataclass
class BenzingaClient:
    """Attach catalyst context from Benzinga or a mock dataset."""

    api_key: str
    base_url: str
    mock_mode: bool = True

    def probe(self) -> dict[str, str]:
        """Run a lightweight connectivity check for the Benzinga news endpoint."""
        if self.mock_mode:
            return {"status": "skipped", "message": "mock mode enabled"}
        if not self.api_key:
            return {"status": "unavailable", "message": "credentials missing: BENZINGA_API_KEY"}

        params = {
            "token": self.api_key,
            "tickers": "SPY",
            "pageSize": 1,
            "displayOutput": "headline",
        }
        url = f"{self.base_url.rstrip('/')}/news"
        try:
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
            return {"status": "connected", "message": "probe succeeded"}
        except Exception as exc:
            return {"status": "failed", "message": f"probe failed: {_humanize_error_text(exc)}"}

    def enrich_with_catalysts(self, movers_df: pd.DataFrame) -> pd.DataFrame:
        """Return movers with catalyst and setup metadata."""
        if movers_df.empty:
            return movers_df.copy()

        if self.mock_mode:
            catalysts = self._mock_catalysts()
        else:
            catalysts = self._fetch_catalysts(movers_df["ticker"].tolist())

        merged = movers_df.merge(catalysts, on="ticker", how="left")
        merged["catalyst"] = merged["catalyst"].fillna("No fresh catalyst found")
        merged["setup_type"] = merged["setup_type"].fillna("watchlist")
        merged["risk_note"] = merged["risk_note"].fillna("Needs manual review")
        return merged

    def _fetch_catalysts(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch the latest Benzinga news headlines for the supplied tickers."""
        rows: list[dict[str, str]] = []
        for ticker in tickers:
            params = {
                "token": self.api_key,
                "tickers": ticker,
                "pageSize": 1,
                "displayOutput": "full",
            }
            url = f"{self.base_url.rstrip('/')}/news"
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            articles = response.json() or []
            article = articles[0] if articles else {}
            headline = article.get("title") or article.get("headline") or "No fresh catalyst found"
            rows.append(
                {
                    "ticker": ticker,
                    "catalyst": headline,
                    "setup_type": self._infer_setup_type(headline),
                    "risk_note": self._build_risk_note(headline),
                }
            )
        return pd.DataFrame(rows)

    def _mock_catalysts(self) -> pd.DataFrame:
        """Return mock catalyst data for offline runs."""
        rows = [
            {
                "ticker": "NVDA",
                "catalyst": "AI demand commentary and strong semiconductor sympathy bid",
                "setup_type": "news_momentum",
                "risk_note": "Extended after gap; manage chase risk",
            },
            {
                "ticker": "SMCI",
                "catalyst": "Server infrastructure momentum and follow-through after analyst note",
                "setup_type": "trend_continuation",
                "risk_note": "Volatile name with sharp intraday reversals",
            },
            {
                "ticker": "PLTR",
                "catalyst": "Government contract chatter supporting relative strength",
                "setup_type": "relative_strength",
                "risk_note": "Can fade if sector momentum cools",
            },
            {
                "ticker": "TSLA",
                "catalyst": "Delivery and margin concerns driving downside pressure",
                "setup_type": "event_driven",
                "risk_note": "Headline sensitivity remains high",
            },
            {
                "ticker": "SOUN",
                "catalyst": "Retail-driven move without durable institutional confirmation",
                "setup_type": "speculative",
                "risk_note": "Thin quality profile despite strong percentage move",
            },
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _infer_setup_type(headline: str) -> str:
        """Infer a simple setup category from a news headline."""
        text = headline.lower()
        if any(term in text for term in ["earnings", "fda", "contract", "guidance", "delivery"]):
            return "event_driven"
        if any(term in text for term in ["analyst", "upgrade", "downgrade"]):
            return "trend_continuation"
        return "news_momentum"

    @staticmethod
    def _build_risk_note(headline: str) -> str:
        """Build a lightweight risk note from the catalyst text."""
        if "downgrade" in headline.lower() or "margin" in headline.lower():
            return "Negative catalyst may continue to pressure price"
        return "Needs validation against broader market tape"


def _safe_error_text(exc: Exception) -> str:
    """Remove secret-like fragments from exception text."""
    text = str(exc)
    text = re.sub(r"(?i)(token|api[_-]?key|secret|password)=([^&\\s]+)", r"\1=[REDACTED]", text)
    text = re.sub(r"(?i)(bearer\s+)[a-z0-9._-]+", r"\1[REDACTED]", text)
    return text


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
