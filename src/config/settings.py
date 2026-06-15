"""Configuration for the intraday shortlist workstation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.project_env import load_project_env


@dataclass(frozen=True)
class ScoreWeights:
    catalyst: float = 0.25
    setup: float = 0.25
    liquidity: float = 0.20
    risk: float = 0.15
    ml: float = 0.15


@dataclass(frozen=True)
class EngineSettings:
    project_root: Path
    data_dir: Path
    history_dir: Path
    db_path: Path
    sec_cache_path: Path
    output_csv: Path
    output_html: Path
    mock_mode: bool
    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_base_url: str
    sec_user_agent: str
    min_price: float
    max_price: float
    min_dollar_volume: float
    min_abs_move_percent: float
    min_relative_volume: float
    max_spread_pct: float
    minimum_total_score: float
    movers_limit: int
    shortlist_limit: int
    history_minutes: int
    daily_lookback_days: int
    sec_feed_limit: int
    alpaca_news_limit: int
    weights: ScoreWeights


def load_settings(project_root: Path) -> EngineSettings:
    """Load runtime settings with safe defaults for the analysis layer."""
    base = load_project_env(project_root)
    data_dir = project_root / "data"
    return EngineSettings(
        project_root=project_root,
        data_dir=data_dir,
        history_dir=data_dir / "history",
        db_path=data_dir / str(base["shortlist_db"]),
        sec_cache_path=data_dir / str(base["sec_ticker_cache"]),
        output_csv=Path(base["output_csv"]),
        output_html=Path(base["output_html"]),
        mock_mode=bool(base["mock_mode"]),
        alpaca_api_key=str(base["alpaca_api_key"]),
        alpaca_api_secret=str(base["alpaca_api_secret"]),
        alpaca_base_url=str(base["alpaca_base_url"]),
        sec_user_agent=str(base["sec_user_agent"]),
        min_price=float(base["min_price"]),
        max_price=float(base["max_price"]),
        min_dollar_volume=float(base["min_dollar_volume"]),
        min_abs_move_percent=float(base["min_abs_move_percent"]),
        min_relative_volume=float(base["min_relative_volume"]),
        max_spread_pct=float(base["max_spread_pct"]),
        minimum_total_score=float(base["minimum_total_score"]),
        movers_limit=int(base["movers_limit"]),
        shortlist_limit=int(base["shortlist_limit"]),
        history_minutes=int(base["analysis_history_minutes"]),
        daily_lookback_days=int(base["daily_lookback_days"]),
        sec_feed_limit=int(base["sec_feed_limit"]),
        alpaca_news_limit=int(base["alpaca_news_limit"]),
        weights=ScoreWeights(
            catalyst=float(base.get("weight_catalyst", 0.25) or 0.25),
            setup=float(base.get("weight_setup", 0.25) or 0.25),
            liquidity=float(base.get("weight_liquidity", 0.20) or 0.20),
            risk=float(base.get("weight_risk", 0.15) or 0.15),
            ml=float(base.get("weight_ml", 0.15) or 0.15),
        ),
    )
