# ESIS-QUANT

A one-person quantitative research system for the **Egyptian Exchange (EGX)**. It
collects prices, financial statements and Arabic/English news, then tests trading
ideas against that data as strictly as it can. It runs a daily routine for a small
real account, and gives Claude a research desk through its own MCP server, written
from the protocol spec.

> **Not investment advice.** This is a personal research project. The signals it
> prints come from fixed, mechanical rules and are recorded so they can be scored.
> **No stock-picking method tested here has beaten the EGX30 index or cash out of
> sample** (details below). Nothing in this repository places orders.

---

## What it does

| Part | What it is | Where |
|---|---|---|
| **Data** | Daily EGX prices adjusted for splits, bonus and rights issues; the real EGX30 index; quarterly and annual statements dated by when they became public; Egyptian holidays; T-bill rates | `data/` |
| **News** | Arabic and English news collected, linked to companies, with headline figures extracted (profit, sales, dividends) | `data/news_collector.py`, `data/disclosure_reader.py` |
| **Backtesting** | Point-in-time engine using measured broker fees (EGP 4.95 + 0.162%), stamp duty, slippage and T+2 settlement | `backtest/` |
| **Research lab** | Pre-registered hypothesis tests with permutation tests, deflated Sharpe and walk-forward splits | `research/`, `analysis/` |
| **Daily operations** | Morning brief, order ticket, book reconciliation, kill switch, nightly verified backup | `ops/` |
| **Agents** | Small single-purpose agents, each held to its own database tables by a SQLite authorizer: sentinel, ledger clerk, librarian, challenger, allocator, chief of staff | `ops/agent_guard.py` and `ops/*` |
| **Watchlist** | Stocks with a *fresh, dated* trigger (earnings, dividend, price, unusual attention). Entries expire after 7 days unless renewed. | `ops/watchlist.py` |
| **Rules** | A careful long-term rule (9 quality and valuation gates, sized to the account's name and sector limits) and a separate high-risk rule (stop −20%, target +40%, 60-session limit). Both are versioned and scored. | `ops/system_picks.py`, `ops/high_risk_picks.py` |
| **4-week ranges** | Calibrated price *ranges*, not point predictions | `ops/forecast.py` |
| **MCP server** | JSON-RPC 2.0 over stdio, written without an SDK. Read-only database, with no trading tools. | `esis_mcp/` |

## What the research found

The negative results are the point of the project:

- **Stock selection:** five selection approaches failed pre-registered tests: cross-sectional factors, rank persistence, momentum (*significantly negative*, −20.9%/yr, p = 0.010), news events and post-earnings drift.
- **Real terms:** a nominal Sharpe of 1.93 shrinks to about 0.5 in real US-dollar terms, a figure that can't be told apart from luck.
- **Regime timing:** it reduced drawdowns in one universe, but flipped sign when the universe changed.
- **Costs:** on a small account the flat broker fee dominates. A 1,000 EGP book split across 24 names costs about 10% per round trip.
- **Risk:** riskier EGX stocks paid more only up to a point. The most volatile fifth earned less than the middle fifth (2016–2026).
- **Forecasts:** 4-week ranges are well calibrated (79.8% of outcomes inside the 80% band, over 5,464 historical forecasts). Direction is a coin flip (51%), so the centre of each range is today's price.

## The safeguards

- **No lookahead:** every fundamental is used from its `available_from` date, and news from its publication date.
- **Fixed rules:** rules are frozen and versioned. A change starts a new version, and its scorecard starts from zero.
- **Recorded predictions:** every watchlist entry, signal and forecast is stored with the price and EGX30 level of that day, then scored against the EGX30, cash and the average stock at 20, 60 and 250 sessions.
- **Enforced write limits:** agents and the MCP server can't write outside their own tables. SQLite refuses the write; nothing depends on the code behaving.

## Running it

Requirements: Windows, Python 3.11.

```bash
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pytest -q tests
```

The database (`esis_quant.db`) is **not** in the repository: it holds a private
account. The collectors in `data/` rebuild the market data.

Common entry points:

```bash
py -3.11 -m ops.chief_of_staff          # morning brief
py -3.11 -m ops.watchlist               # refresh and show the watchlist
py -3.11 -m ops.system_picks            # careful rule: BUY / TRIM / SELL / HOLD
py -3.11 -m ops.high_risk_picks         # high-risk rule
py -3.11 -m ops.forecast COMI EGAL      # 4-week ranges
py -3.11 -m ops.forecast --calibrate    # check the ranges against history
```

### MCP server for Claude

Copy `.mcp.json.example` to `.mcp.json` and fill in your Python path and the
repository path. Claude Code will then offer the `esis-quant` server. It provides
the following:

- **Tools:** `watchlist`, `stock_profile`, `stock_news`, `fundamentals_history`, `screen`, `price_forecast`, `system_signals`, `high_risk_signals`, `position_size_arithmetic`, the two scorecards, `market` and `my_account`.
- **Write tools**, limited to the watchlist tables: `refresh_watchlist`, `watch_add` and `research_note`.
- **Prompts:** `due_diligence` and `watchlist_review`.

News text is passed to the model labelled as untrusted data.

## Layout

```
data/        collectors, point-in-time fundamentals, corporate actions, calendar
backtest/    engine, fees, fill model
analysis/    event studies, diagnostics
research/    hypothesis lab
models/      regime model, sentiment and TFT experiments (mostly retired)
ops/         daily routine, agents, watchlist, rules, forecasts
esis_mcp/    the MCP server (protocol.py, server.py, tools.py)
risk/ portfolio/ visualization/ alerts/
reports/     audits and the final research report
tests/       700+ tests
```

See `OPERATIONS.md` for the daily runbook and failover rules, and
`reports/ESIS_QUANT_final_report.md` for the full research write-up.
