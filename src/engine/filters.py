"""Shortlist filtering and ranking for the intraday analysis layer."""

from __future__ import annotations

import pandas as pd

from src.config.settings import EngineSettings


def build_shortlist(scored_df: pd.DataFrame, settings: EngineSettings) -> pd.DataFrame:
    """Return the ranked shortlist after hard filters and status downgrades."""
    if scored_df.empty:
        return scored_df.copy()

    shortlist = scored_df.copy()
    shortlist = shortlist[shortlist["eligible"]].copy()
    if shortlist.empty:
        shortlist = scored_df[scored_df.apply(_is_watch_quality, axis=1)].copy()
        if shortlist.empty:
            shortlist = scored_df[pd.to_numeric(scored_df.get("current_price"), errors="coerce").fillna(0.0) > 0].copy()
        if shortlist.empty:
            return shortlist
        shortlist["tradeable"] = False
        shortlist["not_tradeable_reason"] = shortlist["not_tradeable_reason"].where(
            shortlist["not_tradeable_reason"].astype(str).str.strip() != "",
            "Did not meet the hard-filter thresholds, so it is being kept as watchlist-only.",
        )

    shortlist = shortlist.sort_values(
        ["tradeable", "total_score", "news_score", "setup_score", "relative_volume"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    preferred = shortlist[shortlist["status_tag"] != "Ignore"].copy()
    if preferred.empty:
        preferred = shortlist.head(min(settings.shortlist_limit, 8)).copy()
        preferred["tradeable"] = False
        preferred["status_tag"] = preferred["status_tag"].replace({"Ignore": "Risky"})
        preferred["action_note"] = preferred["action_note"].where(
            preferred["action_note"].astype(str).str.strip() != "",
            "Better kept as a watch candidate for now rather than actively chased.",
        )
    shortlist = preferred

    shortlist = shortlist.head(settings.shortlist_limit).reset_index(drop=True)
    shortlist.insert(0, "rank", range(1, len(shortlist) + 1))
    return shortlist


def _is_watch_quality(row: pd.Series) -> bool:
    """Keep non-tradeable fallback rows readable enough for the dashboard."""
    symbol = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
    if not symbol or "." in symbol or "/" in symbol or "-" in symbol:
        return False
    if symbol.endswith("W") and len(symbol) > 4:
        return False
    if not symbol.isalpha() or len(symbol) > 5:
        return False
    if float(row.get("current_price") or 0.0) <= 1.0:
        return False
    if float(row.get("dollar_volume") or 0.0) < 250_000:
        return False
    return True
