# Daily Shortlist Workstation

A local intraday stock shortlist and analysis workstation for US equities.

This project is not an auto-trading bot and it does not place orders. The goal is to combine market data, news, and simple explainable rules into a daily shortlist so I can decide which names deserve attention before or during the trading session. Final execution is still manual.

## Why I built this 

I became interested in the way short-term volatility shows up around news, unusual volume, and repeated market attention. In practice, the names that move the most are often the ones that start appearing in movers lists, news feeds, and recurring keywords before the rest of the market fully reacts.

I built this project as a personal research tool to make that process more structured:

- pull market data from Alpaca
- pull catalyst information from Alpaca News and SEC RSS
- turn those inputs into features and scores
- generate a shortlist with structure tags and price zones
- save each run for later review and validation

## What the project does today

### 1. Builds a daily shortlist

The system creates a seed universe from a few practical sources:

- Alpaca movers
- symbols that appear in recent Alpaca News or SEC RSS events
- a small fallback watchlist so the system is still usable when live data is thin

It then enriches those symbols with market context such as:

- snapshots
- latest quotes
- latest bars
- intraday bars
- daily bars

### 2. Explains why a stock made the list

This is not just a ranking table. Each candidate gets explainable fields such as:

- `news_score`
- `setup_score`
- `liquidity_score`
- `risk_score`
- `total_score`
- `status_tag`
- `tradeable`
- `tradeable_reason`
- `not_tradeable_reason`
- `selection_reason`
- `risk_note`
- `action_note`

So the output is not only "what ranked highest" but also:

- why it entered the candidate pool
- whether the structure is closer to breakout or pullback
- whether it is currently tradeable or only watchable
- what the main risk is

### 3. Produces practical intraday price zones

Instead of giving hard buy targets or black-box predictions, the system generates price zones that are more useful for manual observation:

- `Base Range`
- `Breakout Zone`
- `Pullback Zone`
- `Invalidation`

These are based on current price, VWAP, recent volatility, and nearby support/resistance. The goal is to answer questions like:

- is the stock still in a reasonable area to watch, or already too extended
- is it better treated as a breakout setup or a pullback setup
- where does the trade idea stop making sense

### 4. Shows everything in a local Streamlit workstation

The project includes a local Streamlit app with these main pages:

- `Dashboard`
- `Shortlist`
- `Replay`
- `History`
- `Activity`
- `Files`

In day-to-day use, the most important parts are:

- the status bar: live vs mock mode, provider health, recent refresh
- the shortlist table: which names deserve attention first
- the detail panel: price zones, quotes, bars, news, and risk notes
- the replay page: basic review statistics and a place for later validation work

### 5. Stores outputs for later review

Each run writes:

- the latest shortlist CSV
- an HTML report
- `run_status.json`
- `activity_log.json`
- `latest_context.json`
- `shortlist.db`

SQLite is used to keep:

- candidate snapshots
- news events
- range outcomes
- model run metadata

That makes it possible to review prior outputs and continue building replay and validation logic over time.

## End-to-end pipeline

This is the core flow from raw data to final shortlist:

1. Load `.env` and runtime settings
2. Probe Alpaca, News, and SEC RSS status
3. Pull movers as the first candidate source
4. Pull Alpaca News and SEC RSS events as catalyst sources
5. Merge these into a seed symbol list
6. Fetch snapshots, quotes, and bars for those symbols
7. Build features
8. Apply explainable rule-based scoring
9. Apply the range model
10. Apply hard filters and tradeability checks
11. Build the final shortlist
12. Write CSV, HTML, SQLite, and local status files
13. Display the result in Streamlit

In simple terms, the system does not try to predict every stock in the market. It first finds the stocks worth watching, then decides which ones are worth trading.

## Scoring logic

The main score is still rule-based and explainable. The current scoring logic mainly asks four things:

- is the catalyst fresh enough
- is the liquidity good enough
- is the structure clean enough
- is the risk too high

The total score is built from:

- catalyst / news
- setup
- liquidity
- risk
- ml, if the ML layer is available

A name is downgraded or filtered out when it shows one or more of these problems:

- spread is too wide
- dollar volume is too low
- relative volume is too weak
- there is no clear catalyst and the move looks random
- the stock is already too extended to chase

## Directory structure

```text
daily_shortlist_engine/
├── app.py
├── main.py
├── requirements.txt
├── .env.example
├── data/
├── outputs/
└── src/
    ├── alpaca_client.py
    ├── project_env.py
    ├── config/
    │   └── settings.py
    ├── data/
    │   ├── alpaca_market.py
    │   ├── alpaca_news.py
    │   ├── sec_rss.py
    │   └── dataset_builder.py
    ├── engine/
    │   ├── features.py
    │   ├── scoring.py
    │   ├── filters.py
    │   ├── labels.py
    │   ├── range_model.py
    │   ├── regime.py
    │   ├── ml_features.py
    │   └── ml_targets.py
    ├── models/
    │   ├── lightgbm_ranker.py
    │   ├── lightgbm_classifier.py
    │   ├── lightgbm_regressor.py
    │   ├── model_registry.py
    │   └── model_service.py
    ├── storage/
    │   ├── db.py
    │   └── models.py
    ├── ui/
    │   ├── dashboard.py
    │   ├── detail_panel.py
    │   ├── ml_panel.py
    │   └── replay_page.py
    └── backtest/
        ├── walkforward.py
        ├── evaluation.py
        └── strategy_templates.py
```

## Tech stack

- Python
- Streamlit
- pandas
- requests
- SQLite
- Jinja2
- scikit-learn
- LightGBM

## Running locally

### 1. Create and configure the environment file

```bash
cp .env.example .env
```

At minimum, fill in:

```env
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_BASE_URL=https://data.alpaca.markets
SEC_USER_AGENT=Your Name your@email.com
MOCK_MODE=false
```

If `SEC_USER_AGENT` is still a placeholder, SEC RSS will remain unavailable by design.

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, `lightgbm` may require `libomp` if import fails after installation.

### 3. Run the data pipeline

```bash
python main.py
```

### 4. Start the local dashboard

```bash
streamlit run app.py --server.port 8503
```

Then open:

- [http://localhost:8503](http://localhost:8503)

## Current state

At this stage, the project can already:

- connect to Alpaca Market Data API
- fetch movers, snapshots, latest quotes, latest bars, and historical bars
- ingest Alpaca News
- ingest SEC RSS when `SEC_USER_AGENT` is valid
- generate a shortlist
- compute rule-based scores and status tags
- compute Base / Breakout / Pullback / Invalidation zones
- save results into SQLite for later replay work
- display run status, shortlist tables, and symbol detail views in the local dashboard

## What is still in progress

I see this project as an evolving research tool rather than a finished product. The main areas still being improved are:

- stronger event-theme detection
- better industry / theme mapping
- deeper replay statistics
- more stable LightGBM training and validation once enough sample data exists
- websocket-driven real-time updates
- more mature strategy template research

So the rule-based layer and local workstation are already usable, while the deeper model and research layers are still being refined.

## What this project does not do

To avoid confusion, this project does not:

- place orders automatically
- promise profits
- output "guaranteed" trade ideas
- throw every stock into the shortlist
- present a black-box model as if it were a magical predictor

The role of the project is very specific:

It is a local intraday analysis workstation meant to improve observation speed and decision quality, not to replace trading judgment itself.

## Notes

If I keep pushing the project forward, the two highest-priority areas are:

1. making the shortlist more executable and less watch-only
2. turning the LightGBM layer into something properly trained, validated, and replay-tested once enough historical samples accumulate

That is also why I wanted the project on GitHub in the first place: it is no longer a loose collection of scripts, but a structured personal project with room to keep growing.
