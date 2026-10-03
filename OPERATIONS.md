# ESIS-QUANT — Operations Runbook

Execution, reconciliation and monitoring. Strategy logic is **out of scope
here** — this document covers only the layer that fails first in live trading.

The governing rule, learned the hard way: **on 2026-07-23 the daily job died and
stayed dead for three days without anyone noticing.** Every procedure below
exists so that a failure is loud, not silent. *Absence of news is not good news.*

---

## Daily sequence

| When | Step | Command | On failure |
|---|---|---|---|
| 17:00 (after close) | Fetch data, generate signal + ticket | `py -3.11 -m ops.daily_run` | Records to `job_runs`, pops a Windows alert, **exits non-zero** |
| 10:00 (next open) | Review + execute | `daily.ps1` (scheduled popup) | Shows a red health banner if the pipeline is unhealthy |
| After trading | Record fills | `py -3.11 -m ops.order_ticket apply` | Fill not recorded ⇒ book silently wrong |
| After fills | Reconcile against broker | `py -3.11 -m ops.reconcile totals <stocks> <cash>` | Loud alert on mismatch |

Health at any time: `py -3.11 -m ops.health` · Execution quality:
`py -3.11 -m ops.execution_quality`

### Settlement (T+2)

EGX settles equities T+2. Selling credits the broker balance immediately, but
those funds are **not spendable** until settlement. The ticket sizes buys
against `available_cash()` (settled only) and prints the unsettled amount in its
header when there is any. If Thndr does permit reinvesting unsettled proceeds,
set `config.ALLOW_UNSETTLED_REINVESTMENT = True` — **only after confirming it**.

### Idle cash

When the overlay de-risks, freed capital earns ~0% at the broker while Egyptian
short rates run 8–27%. The daily report now computes what that is costing and
whether it clears real product minimums:

| idle cash | route | reminder |
|---|---|---|
| < 5,000 EGP | none worth the friction | **silent** |
| 5,000–25,000 | money market fund (daily liquidity) | fires |
| ≥ 25,000 | direct T-bill (min subscription) *or* fund | fires |

Thresholds apply to **idle cash, not book size**, and nothing fires until the
cash has sat for `IDLE_CASH_MIN_DAYS` (14) — a brief de-risk is not a reason to
go buy a 3-month bill. At the current 1,000 EGP book it is correctly silent.

### Kill-switch (model-independent backstop)

Reads the forward paper record's realized losses, not the regime model — the
regime model is the thing that might be wrong. If a daily or 5-session loss
limit breaks (`config.KILL_SWITCH_MAX_*`), today's target book is forced FLAT
regardless of the regime signal, and a loud alert fires. It can only ever
*reduce* exposure. With fewer than 5 forward sessions it fails OPEN and reports
itself as inconclusive rather than passing silently.

---

## Failover policy — explicit defaults

### 1. Data feed down or rate-limited on ticket day
**Default: NO TRADE, ALERT.** Never proceed on stale data.

`data/quality.assert_data_fit_to_trade()` runs *before* any book is built and
raises `DataQualityError` if the DB is stale, the universe is degenerate, or
**macro is stale**. `ops/trade_list.generate_trade_list()` calls it first, so a
failed gate means no ticket exists — the correct outcome.

> **Macro is checked separately and deliberately.** On 2026-07-30 prices were
> perfectly current while `usd_egp` had stopped on 2026-07-07. Because the regime
> observation matrix depends on FX, the live exposure signal was silently
> truncated by 23 days and merely forward-filled. Every price-based check passed.
> `MAX_MACRO_STALE_DAYS = 7` now closes this.

### 2. Ticket not executed by the close
**Default: the ticket EXPIRES. It is never carried forward silently.**

Tickets are named by *generation date* and only the newest is kept
(`KEEP_TICKETS = 1`), so a stale ticket cannot be mistaken for today's. Prices in
an unexecuted ticket are from a session that has since closed; acting on it the
next day means trading on a price that no longer exists. Regenerate instead:

```bash
py -3.11 -m ops.order_ticket write
```

### 3. Order looks unfillable
The ticket carries a `flags` column. **Do not chase a flagged line.**

| Flag | Meaning | Action |
|---|---|---|
| `SIZE>20%ADV` | Order is large vs the name's daily volume | Split across days, or skip |
| `MOVED±x%` | Name already moved sharply today; may be at its price band | Verify it is quoting before sending |
| `NO-VOLUME/HALT?` | No traded volume in recent sessions | Treat as untradable |
| `ADV?` / `MKT?` | Check could not be computed | Treat as unknown, **not** as safe |

Also honour `max_price` — if the live price exceeds it, **skip the line**. The
ticket is priced off the previous close; a name that gapped is no longer the
trade the optimizer sized.

### 4. Book does not match the broker
**Stop. Do not act on further suggestions until resolved.** Every subsequent
suggestion is computed from the internal book, so if that book is wrong, so is
everything downstream.

### 5. PC off at a scheduled time
Both tasks have `StartWhenAvailable = true` and run at next power-on. The daily
report flags any unanswered suggestions; clear them with
`py -3.11 -m ops.portfolio_tracker catchup` (logged as `MISSED`, not as a trade).

---

## Known limitations — do not mistake these for handled

| Limitation | Status |
|---|---|
| **Does this beat holding T-bills, in real terms? — MARGINAL AND NOT ROBUST.** Strategy real-USD CAGR ≈ **0.0%** vs T-bills ≈ **−2.2%**, a ~2pp edge that *reverses* if bills yielded >19%; both lost purchasing power, and the strategy took a **−52.8%** real drawdown to earn ~zero. See the report addendum | **Answered** |
| **No broker API.** Verified 2026-08-04: Thndr is a *consumer* of Alpaca's Broker API (B2B); no customer-facing trading or quote API exists. Execution stays manual | Permanent (verified) |
| **Live market data — NOT PURSUED, verified 2026-08-04.** Every option was priced/checked: **ICE** real-time EGX = institutional pricing; **Twelve Data** EGX (XCAI) is **EOD only** and needs the **$229/mo Pro tier** (free tier covers 3 exchanges, EGX not among them) — i.e. $2,748/yr for data yfinance gives free; **EODHD** = EOD (no gain); **Mansa Markets** advertises "live" but its own page says *30-min refresh*; **Stockastic** unverified. **No retail-viable real-time EGX feed exists at this book size.** The strategy decides once daily at the close, so live data would change no decision anyway — the only use is the execution-time price check, which `max_price` + `flags` + the Thndr app already cover | Not pursued (verified) |
| **⛔ EGXAPI — DO NOT USE.** Offers "free forever" real-time EGX data *and* commission-free order routing — a combination inconsistent with the cost of licensing real-time exchange data. Site returns 403; markets itself as "FRA-**aligned**" (not FRA-*licensed*); no independent reviews found. **Precedent: EGXPILOT, a similarly EGX-branded platform, was added to Egypt's FRA blacklist in 2025** as an unregistered entity offering financial services. FRA Decision No. 87 of 2026 created a formal three-tier blacklist for exactly this category. Verify any broker against the [FRA public register](https://fra.gov.eg/) before sending money or credentials | Avoid |
| ~~T+2 settlement not modelled~~ — **FIXED.** Sale proceeds are tracked in `pending_settlements` and excluded from spendable cash until T+2. See `portfolio_tracker.available_cash()`. Broker reinvestment policy still unverified — `config.ALLOW_UNSETTLED_REINVESTMENT=False` is the conservative default | Closed |
| ~~EGX price band not authoritative~~ — **CLOSED, VERIFIED.** Regular band **±20%** from previous close; a **10%** move triggers a ~10-min suspension; auction matching ±10% of MVWAP. Encoded as `PRICE_LIMIT_HARD`/`PRICE_SUSPEND_TRIGGER`/`PRICE_LIMIT_WARN` with the source cited in `ops/order_ticket.py`. Source: [EFG Hermes EGX specs](https://tg.efghermes.com/Market/GetById/1103) | Closed |
| ~~Minimum lot sizes unverified~~ — **CLOSED, NOT APPLICABLE.** EGX has **no board lots**; whole-share rounding in the integer allocator is already correct and complete. Same source | Closed |
| **Slippage still modelled, not measured** — **0 of 20 fills** recorded. Cannot be closed until real fills accumulate; progress is visible via `py -3.11 -m ops.execution_quality` | Open (blocked on data) |
| **`data/microstructure.py` still dead code** — kept, not wired. The pre-trade check already uses ADV *and* the fee schedule; Amihud measures the market's price response to the market's own volume, which the Phase-3 tier study showed explains only 4–9% of variance. Adding it would add noise, not information. **Delete if still unused after live fills exist** | Not pursued (reasoned) |
| ~~Kill-switch backtest-only~~ — **FIXED.** `risk.kill_switch.live_state()` reads the forward paper record and forces exposure to 0 in `ops/trade_list.py` when a hard loss limit breaks, independently of the regime model | Closed |

---

## What the system will not do

It will not place an order, size a position for you, or tell you what to buy. It
prints a mechanical target and measures what you actually did.

**The go-live gate governs regardless of anything in this document:**
`py -3.11 -m ops.go_live_gate`
