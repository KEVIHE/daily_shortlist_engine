"""CSV and HTML report generation for the shortlist workstation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template


REPORT_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Daily Shortlist Workstation Report</title>
  <style>
    body { font-family: "Avenir Next", "Segoe UI", sans-serif; margin: 0; padding: 28px; background: #f3f5f8; color: #132033; }
    .wrap { max-width: 1280px; margin: 0 auto; }
    .hero, .panel { background: #fff; border: 1px solid #dbe3ea; border-radius: 16px; padding: 20px; margin-bottom: 18px; }
    .hero h1 { margin: 0 0 8px; font-size: 30px; }
    .meta { color: #5d6b7d; font-size: 14px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid #e6edf3; vertical-align: top; }
    th { text-transform: uppercase; letter-spacing: .08em; font-size: 12px; color: #31506d; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .metric { background: #f8fbfd; border: 1px solid #e6edf3; border-radius: 12px; padding: 14px; }
    .metric-label { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: #5d6b7d; }
    .metric-value { font-size: 22px; font-weight: 800; margin-top: 6px; }
    @media (max-width: 900px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Daily Shortlist Workstation</h1>
      <div class="meta">Generated {{ generated_at }} | Data mode {{ data_mode|upper }} | Rows {{ row_count }}</div>
    </section>
    <section class="panel">
      <div class="grid">
        {% for label, value in metrics %}
        <div class="metric">
          <div class="metric-label">{{ label }}</div>
          <div class="metric-value">{{ value }}</div>
        </div>
        {% endfor %}
      </div>
    </section>
    <section class="panel">
      <h2>Top Candidates</h2>
      {{ shortlist_table | safe }}
    </section>
  </div>
</body>
</html>
"""
)


def build_outputs(
    df: pd.DataFrame,
    output_csv: Path,
    output_html: Path,
    mock_mode: bool,
    data_mode: str | None = None,
) -> tuple[Path, Path]:
    """Write the latest shortlist CSV and a compact HTML report."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()
    export_df.to_csv(output_csv, index=False)

    display_df = _display_frame(export_df)
    html = REPORT_TEMPLATE.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        data_mode=(data_mode or ("mock" if mock_mode else "live")).lower(),
        row_count=len(export_df),
        metrics=[
            ("Tradeable", int(export_df["tradeable"].sum()) if "tradeable" in export_df.columns else 0),
            ("Avg Score", f"{export_df['total_score'].mean():.2f}" if not export_df.empty and "total_score" in export_df.columns else "0.00"),
            ("Avg ML", f"{export_df['ml_score'].mean():.2f}" if not export_df.empty and "ml_score" in export_df.columns else "0.00"),
            ("Breakout Watch", int((export_df.get("status_tag", pd.Series(dtype='object')) == "Breakout Watch").sum()) if not export_df.empty else 0),
            ("Pullback Watch", int((export_df.get("status_tag", pd.Series(dtype='object')) == "Pullback Watch").sum()) if not export_df.empty else 0),
        ],
        shortlist_table=display_df.to_html(index=False, border=0, escape=False),
    )
    output_html.write_text(html, encoding="utf-8")
    return output_csv, output_html


def _display_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank",
        "symbol",
        "last_price",
        "gap_pct",
        "volume_today",
        "relative_volume",
        "spread_pct",
        "catalyst",
        "news_score",
        "setup_score",
        "liquidity_score",
        "risk_score",
        "ml_score",
        "breakout_probability",
        "failure_probability",
        "total_score",
        "status_tag",
        "confidence",
        "action_note",
    ]
    display = df.copy()
    for column in columns:
        if column not in display.columns:
            display[column] = ""
    for column in ["last_price", "gap_pct", "relative_volume", "spread_pct", "news_score", "setup_score", "liquidity_score", "risk_score", "ml_score", "total_score"]:
        display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0.0).map(lambda value: f"{value:.2f}")
    for column in ["breakout_probability", "failure_probability"]:
        display[column] = pd.to_numeric(display[column], errors="coerce").fillna(0.0).map(lambda value: f"{value * 100:.1f}%")
    display["volume_today"] = pd.to_numeric(display["volume_today"], errors="coerce").fillna(0.0).map(lambda value: f"{value:,.0f}")
    return display[columns]
