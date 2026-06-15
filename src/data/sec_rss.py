"""SEC RSS ingestion for filing catalysts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
import requests

SEC_CURRENT_FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&company=&owner=exclude&count={count}&output=atom"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FORM_STRENGTH = {
    "8-K": ("sec_8k", 8.0),
    "10-Q": ("sec_10q", 6.5),
    "10-K": ("sec_10k", 7.0),
    "S-3": ("sec_s3", 5.5),
    "13D": ("sec_13d", 8.0),
    "13G": ("sec_13g", 6.0),
}
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class SecRssClient:
    """Fetch SEC filing feeds and normalize them into catalyst records."""

    user_agent: str
    cache_path: Path

    def probe(self) -> dict[str, str]:
        """Probe the SEC RSS endpoint."""
        if _is_placeholder_user_agent(self.user_agent):
            return {"status": "unavailable", "message": "set SEC_USER_AGENT to a real name/email for SEC requests"}
        try:
            response = requests.get(
                SEC_CURRENT_FEED.format(count=1),
                headers=self._headers(),
                timeout=12,
            )
            response.raise_for_status()
            return {"status": "connected", "message": "probe succeeded"}
        except Exception as exc:
            return {"status": "failed", "message": f"probe failed: {_humanize_sec_error(exc)}"}

    def fetch_filings(self, watched_symbols: list[str], limit: int) -> pd.DataFrame:
        """Fetch recent SEC filings and map them to tickers when possible."""
        if _is_placeholder_user_agent(self.user_agent):
            return pd.DataFrame(columns=_sec_columns())
        mapping = self._load_ticker_mapping()
        response = requests.get(
            SEC_CURRENT_FEED.format(count=limit),
            headers=self._headers(),
            timeout=20,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        watched = {str(symbol or "").strip().upper() for symbol in watched_symbols if str(symbol or "").strip()}
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        records: list[dict[str, Any]] = []

        for entry in root.findall("atom:entry", ATOM_NS):
            title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
            updated_text = (entry.findtext("atom:updated", default="", namespaces=ATOM_NS) or "").strip()
            link_el = entry.find("atom:link", ATOM_NS)
            link = link_el.attrib.get("href", "") if link_el is not None else ""
            category_el = entry.find("atom:category", ATOM_NS)
            form_type = str(category_el.attrib.get("term", "") if category_el is not None else "").strip().upper()
            published_at = _parse_timestamp(updated_text)
            if published_at and published_at < cutoff:
                continue

            cik = _extract_cik(link, title)
            mapping_item = mapping.get(cik, {}) if cik else {}
            symbol = str(mapping_item.get("ticker") or "").strip().upper()
            if watched and symbol and symbol not in watched:
                continue
            if not symbol and watched:
                continue

            catalyst_type, strength = FORM_STRENGTH.get(form_type, ("sec_other", 4.0))
            headline = title or summary or f"SEC filing {form_type}"
            records.append(
                {
                    "timestamp": published_at.isoformat(timespec="seconds") if published_at else "",
                    "symbol": symbol,
                    "headline": headline,
                    "source": "SEC RSS",
                    "url": link,
                    "catalyst_type": catalyst_type,
                    "headline_strength": strength,
                    "freshness_minutes": _freshness_minutes(published_at),
                    "form_type": form_type,
                    "raw_json": json.dumps({"title": title, "summary": summary, "updated": updated_text, "link": link}, ensure_ascii=True),
                }
            )
        if not records:
            return pd.DataFrame(columns=_sec_columns())
        frame = pd.DataFrame(records)
        return frame.drop_duplicates(subset=["symbol", "headline", "url"], keep="first").reset_index(drop=True)

    def _load_ticker_mapping(self) -> dict[str, dict[str, str]]:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(self.cache_path.stat().st_mtime, timezone.utc)
            if age < timedelta(days=1):
                try:
                    payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                    return _normalize_mapping(payload)
                except Exception:
                    pass
        response = requests.get(SEC_TICKER_MAP_URL, headers=self._headers(), timeout=20)
        response.raise_for_status()
        payload = response.json() or {}
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        return _normalize_mapping(payload)

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }


def _is_placeholder_user_agent(user_agent: str) -> bool:
    text = str(user_agent or "").strip().lower()
    if not text:
        return True
    return text.endswith("research@local") or text.startswith("dailyshortlistengine/1.0 research@local")


def _humanize_sec_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "403" in text or "forbidden" in text:
        return "403 forbidden"
    if "401" in text or "unauthorized" in text:
        return "401 unauthorized"
    if "timeout" in text:
        return "timeout"
    return exc.__class__.__name__


def _normalize_mapping(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    if isinstance(payload, dict):
        values = payload.values()
    else:
        values = payload
    for item in values:
        if not isinstance(item, dict):
            continue
        cik = str(item.get("cik_str") or item.get("cik") or "").strip()
        ticker = str(item.get("ticker") or "").strip().upper()
        title = str(item.get("title") or "")
        if cik and ticker:
            mapping[cik.lstrip("0")] = {"ticker": ticker, "title": title}
    return mapping


def _extract_cik(link: str, title: str) -> str | None:
    for text in [link, title]:
        match = re.search(r"/data/(\d+)/", text)
        if match:
            return match.group(1).lstrip("0")
        match = re.search(r"\((\d{6,10})\)", text)
        if match:
            return match.group(1).lstrip("0")
    return None


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _freshness_minutes(published_at: datetime | None) -> float:
    if not published_at:
        return 9999.0
    return round(max((datetime.now(timezone.utc) - published_at).total_seconds() / 60.0, 0.0), 2)


def _sec_columns() -> list[str]:
    return [
        "timestamp",
        "symbol",
        "headline",
        "source",
        "url",
        "catalyst_type",
        "headline_strength",
        "freshness_minutes",
        "form_type",
        "raw_json",
    ]
