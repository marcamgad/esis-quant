"""
Daily report: one screen that (1) shows the system's current suggestion,
(2) records where YOU stand on it (accepted / declined / own / not-yet-decided),
and (3) gets smarter over time by MEASURING how your past choices actually
turned out -- not by giving advice.

The "smart" part is deliberately mechanical and honest. Every day this report
runs it snapshots your book value next to the system's paper book (into
user_daily_snapshot). With both curves stored, it can later state facts like
"on the 3 days you DECLINED, the system book then rose +2.1% -- following it
would have earned more" or "your own picks are +1.4% vs the system's +0.3%".
That is feedback from outcomes, which is the only legitimate way to learn here.
It never tells you what to buy; it tells you what happened.

Nothing here places orders or fetches data. It reads the last logged trade
list, your holdings, your decisions, and the forward paper record.

CLI:
  py -3.11 -m ops.daily_report            # today's full report (+ snapshot)
  py -3.11 -m ops.daily_report --no-write  # print only, don't snapshot
"""

import logging

import pandas as pd

from data.database import get_connection
from ops import portfolio_tracker as pt
from ops.trade_list import latest_trade_list

logger = logging.getLogger("esis_quant.ops.daily_report")

WINDOW = 66   # sessions to horizon a decision's outcome over (~quarter)


# --------------------------------------------------------------------------
# snapshot: your book value + the system book value, once per day
# --------------------------------------------------------------------------
def _system_nav_on(as_of: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT nav FROM paper_trading_nav WHERE date <= ? ORDER BY date DESC LIMIT 1",
            (str(as_of),)).fetchone()
    return float(row["nav"]) if row else None


def snapshot_today(as_of: str, regime: str, exposure: float, decision: str | None) -> dict:
    """Record your book value (and the system's) for `as_of`. Idempotent: one
    row per date, overwritten if the report is re-run the same day."""
    pos, cash = pt.holdings()
    user_value = float(pos["value"].sum() if not pos.empty else 0.0) + float(cash)
    n_pos = int(len(pos))
    sys_nav = _system_nav_on(as_of)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO user_daily_snapshot "
            "(date, user_value, cash, n_positions, decision, regime, exposure, system_nav) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(date) DO UPDATE SET user_value=excluded.user_value, cash=excluded.cash, "
            "n_positions=excluded.n_positions, decision=excluded.decision, regime=excluded.regime, "
            "exposure=excluded.exposure, system_nav=excluded.system_nav",
            (str(as_of), user_value, float(cash), n_pos, decision, regime, float(exposure), sys_nav))
    return {"user_value": user_value, "cash": cash, "n_positions": n_pos, "system_nav": sys_nav}


def _snapshots() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM user_daily_snapshot ORDER BY date", conn, parse_dates=["date"])


# --------------------------------------------------------------------------
# smart feedback: how did each past decision actually turn out?
# --------------------------------------------------------------------------
def _paper_nav() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT date, nav FROM paper_trading_nav ORDER BY date",
                           conn, parse_dates=["date"]).set_index("date")


def decision_scorecard() -> pd.DataFrame:
    """For each decision you logged, the system book's forward return from the
    decision date to `WINDOW` sessions later (or the latest data). This answers,
    mechanically: was declining / going your own way the right call so far?
      - DECLINED : forward system return = what you sat out of.
      - OWN      : forward system return = the benchmark your own picks beat/lost to.
      - ACCEPTED : you were IN the system book, so this is simply what you earned.
    Positive 'system_fwd' after a DECLINE means the system would have made money
    (you missed it); negative means declining protected you.
    """
    dh = pt.decision_history()
    nav = _paper_nav()
    rows = []
    for r in dh.itertuples():
        d0 = pd.Timestamp(r.as_of_date) if r.as_of_date else pd.Timestamp(str(r.decided_at)[:10])
        fwd = float("nan")
        if not nav.empty:
            after = nav[nav.index >= d0]
            if len(after) >= 2:
                horizon = after.iloc[: WINDOW + 1]
                fwd = float(horizon["nav"].iloc[-1] / horizon["nav"].iloc[0] - 1)
        rows.append({"as_of": str(d0.date()), "decision": r.decision,
                     "system_fwd": fwd, "note": r.note})
    return pd.DataFrame(rows)


def learning_summary() -> dict:
    """Aggregate the scorecard into the few honest facts worth stating."""
    sc = decision_scorecard()
    snaps = _snapshots()
    out = {"n_decisions": int(len(sc)), "counts": {}, "avg_fwd_after_decline": None,
           "your_return": None, "system_return": None}
    if not sc.empty:
        out["counts"] = sc["decision"].value_counts().to_dict()
        dec = sc[(sc["decision"] == "DECLINED") & sc["system_fwd"].notna()]
        if not dec.empty:
            out["avg_fwd_after_decline"] = float(dec["system_fwd"].mean())
    # your book vs system book, both rebased to their first snapshot
    valid = snaps[snaps["system_nav"].notna() & (snaps["user_value"] > 0)]
    if len(valid) >= 2:
        out["your_return"] = float(valid["user_value"].iloc[-1] / valid["user_value"].iloc[0] - 1)
        out["system_return"] = float(valid["system_nav"].iloc[-1] / valid["system_nav"].iloc[0] - 1)
    return out


# --------------------------------------------------------------------------
# cost + capacity reality of the current book (Tier-2 #7/#8, surfaced daily)
# --------------------------------------------------------------------------
# Standing real-terms finding from the cost-realistic backtest sweep. These are
# BACKTEST facts (in-sample), not forward evidence -- kept as fixed context so
# every daily report reminds you the nominal number is mostly currency melt.
# (nominal CAGR -> real-USD CAGR at each AUM; real Sharpe 95% CI includes 0.)
_REAL_TERMS_NOTE = [
    "nominal backtest CAGR 29.7% is mostly EGP devaluation (USD/EGP 7.8->48.8).",
    "in REAL (USD) terms: +9.1% at tiny size, ~0% at 12.5M cap, NEGATIVE above it.",
    "real-USD Sharpe ~0.50-0.62 but FAILS the deflated-Sharpe test even at 1 trial.",
    "DEVALUATION JUMP RISK: 2 days (2016-11-06 -39%, 2024-03-07 -37%) cost 552pts",
    "of real return -- and the regime overlay was 100% INVESTED through both.",
    "The BEAR signal reads EGX prices; a devaluation hits the CURRENCY. No defense.",
]


def reality_check(weights: pd.Series, book_egp: float) -> dict:
    """Live capacity + liquidity-aware round-trip cost for the current book,
    sized to the user's own capital (or a 1M EGP reference in practice mode)."""
    from analysis.capacity import book_capacity
    from backtest.costs import book_roundtrip_cost, average_daily_value
    w = weights[weights > 0]
    if w.empty:
        return {}
    adv = average_daily_value(list(w.index))
    aum = book_egp if book_egp and book_egp > 0 else 1_000_000.0
    cap = book_capacity(w, participation_cap=0.10, spread_days=3, adv=adv)
    cost = book_roundtrip_cost(w, capital_egp=aum, adv=adv)
    return {"aum": aum, "is_reference": not (book_egp and book_egp > 0),
            "capacity": cap, "roundtrip_cost": cost["roundtrip_cost"],
            "flagged": cost["flagged_high_impact"], "flat_ref": cost["flat_cost_reference"],
            "n_orders": cost.get("n_orders", 0), "total_egp": cost.get("total_egp", 0.0),
            "flat_fee_share": cost.get("flat_fee_share", 0.0)}


def regime_health() -> dict:
    """Part 4 -- cheap DAILY sanity check on the regime model between annual
    refits. Not a retrain, not a new model: two read-only cross-checks.

    (1) CONFIDENCE DRIFT. The ensemble's P(BEAR) is a consensus across
        independent refits. Historically the fits agree strongly (P near 0 or 1);
        a drift toward 0.5 means they are disagreeing more, which is the earliest
        cheap signal that the model no longer fits the data it is being fed.
    (2) VOLATILITY CROSS-CHECK. Compare realized market vol now against the
        vol typical of the regime the model currently believes it is in. A
        sustained mismatch (model says calm, market is not) is a red flag that
        does not require retraining anything to notice.
    """
    from models.regime_ensemble import load_ensemble
    from backtest.engine import load_price_returns
    out = {}
    p = load_ensemble()
    if not p.empty:
        recent, hist = p.tail(20), p
        # decisiveness = distance from 0.5; 0.5 means the fits cannot agree
        out["p_bear_now"] = float(recent.iloc[-1])
        out["decisiveness_recent"] = float((recent - 0.5).abs().mean() * 2)
        out["decisiveness_history"] = float((hist - 0.5).abs().mean() * 2)
        out["confidence_drift"] = out["decisiveness_recent"] - out["decisiveness_history"]
    try:
        r = load_price_returns().mean(axis=1).dropna()
        out["vol_20d"] = float(r.tail(20).std() * (252 ** 0.5))
        out["vol_1y"] = float(r.tail(252).std() * (252 ** 0.5))
        out["vol_ratio"] = out["vol_20d"] / out["vol_1y"] if out["vol_1y"] > 0 else float("nan")
    except Exception:  # noqa: BLE001
        pass
    return out


def format_regime_health(h: dict) -> str:
    if not h:
        return "    (regime health unavailable)"
    lines = []
    if "p_bear_now" in h:
        drift = h["confidence_drift"]
        tag = ("fits AGREEING LESS than usual — watch" if drift < -0.15
               else "consensus normal")
        lines.append(f"    ensemble P(BEAR) now {h['p_bear_now']:.2f}; consensus strength "
                     f"{h['decisiveness_recent']:.2f} vs {h['decisiveness_history']:.2f} historical "
                     f"({tag})")
    if "vol_ratio" in h:
        v = h["vol_ratio"]
        tag = "market far more volatile than its year" if v > 1.6 else               "market unusually calm" if v < 0.6 else "in line with the year"
        lines.append(f"    realized vol 20d {h['vol_20d']*100:.1f}% vs 1y {h['vol_1y']*100:.1f}% "
                     f"(ratio {v:.2f} — {tag})")
    return "\n".join(lines) if lines else "    (regime health unavailable)"


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def daily_report(write: bool = True) -> str:
    tl = latest_trade_list()
    # Bias disclosure goes in the HEADER, not a docstring: a caveat that lives
    # next to the code instead of next to the number is not a disclosure.
    from data.universe import survivorship_disclosure
    lines = ["=" * 68, "ESIS-QUANT — DAILY REPORT", "=" * 68,
             survivorship_disclosure(short=True), ""]

    # [0] pipeline health FIRST -- a stale suggestion from a dead job looks
    # exactly like a fresh one, so this must be the first thing read.
    try:
        from ops.health import health_status, format_health
        h = health_status()
        if not h["ok"]:
            lines.append("")
            lines.append(format_health(h))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"\n  (health check unavailable: {exc})")

    if tl.empty:
        lines.append("\nNo suggestion logged yet. Run the daily job (ops.daily_run) first.")
        return "\n".join(lines)

    as_of = str(tl["as_of_date"].iloc[0].date())
    regime = str(tl["regime"].iloc[0])
    exposure = float(tl["exposure"].iloc[0])
    run_id = str(tl["run_id"].iloc[0])

    # ---- did the call change since the previous run? ----------------------
    with get_connection() as conn:
        prev = pd.read_sql(
            "SELECT DISTINCT as_of_date, regime, exposure FROM trade_list_log "
            "WHERE run_id != ? ORDER BY as_of_date DESC LIMIT 1", conn, params=(run_id,))
    changed = ""
    if not prev.empty:
        pr, pe = str(prev["regime"].iloc[0]), float(prev["exposure"].iloc[0])
        if pr != regime or abs(pe - exposure) > 1e-6:
            changed = f"  (CHANGED from {pr} / {pe*100:.0f}% invested)"
        else:
            changed = "  (unchanged from last run)"

    # ---- 1) the suggestion ------------------------------------------------
    lines.append(f"\n[1] SYSTEM SUGGESTION  as-of {as_of}   run {run_id}")
    lines.append(f"    regime = {regime}   exposure = {exposure*100:.0f}% invested{changed}")
    hold = tl[tl["ticker"] != "__CASH__"].sort_values("weight", ascending=False)
    if hold.empty or exposure <= 1e-6:
        lines.append("    book: FULLY IN CASH — the system's de-risk (BEAR) signal is ON.")
    else:
        # EVERY name, not a head(6) teaser. A truncated list invites the reader
        # to assume the tail is negligible; at 27 names the tail is a third of
        # the book. `cap` marks which names the ticket would actually trade --
        # the rest are target weights the name cap deliberately does not buy.
        cap_keep = set()
        try:
            from ops.order_ticket import cap_target_names
            w_all = hold.set_index("ticker")["weight"]
            capped, _ = cap_target_names(w_all / w_all.sum(), capital_egp=pt.holdings()[1]
                                         + float(pt.holdings()[0]["value"].sum()
                                                 if not pt.holdings()[0].empty else 0.0))
            cap_keep = set(capped.index) if capped is not None else set(w_all.index)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not mark capped names: %s", exc)
        lines.append(f"    {len(hold)} names in the target book "
                     f"({len(cap_keep) or len(hold)} of them the ticket would trade):")
        # The names the ticket would TRADE are listed in full; the rest are the
        # long tail the name cap excludes on cost grounds and are listed
        # compactly. At 69 target names a full table every morning is unreadable,
        # and an unread list conveys nothing -- but nothing is hidden.
        lines.append(f"       {'#':>3} {'ticker':8s}{'weight':>8s}   in ticket?")
        tail = []
        for i, r in enumerate(hold.itertuples(), 1):
            in_ticket = (not cap_keep) or r.ticker in cap_keep
            if in_ticket:
                lines.append(f"       {i:>3} {r.ticker:8s}{r.weight*100:>7.2f}%   yes")
            else:
                tail.append(f"{r.ticker} {r.weight*100:.2f}%")
        if tail:
            lines.append(f"       + {len(tail)} more the name cap excludes on cost "
                         f"(total weight {sum(float(x.split()[1][:-1]) for x in tail):.1f}%):")
            for j in range(0, len(tail), 6):
                lines.append("         " + ", ".join(tail[j:j + 6]))

    # ---- 2) where you stand vs the suggestion -----------------------------
    rec = pt.reconcile()
    pos, cash = pt.holdings()
    pv = rec["portfolio_value"]
    lines.append(f"\n[2] YOUR BOOK  {pv:,.0f} EGP  (cash {cash:,.0f}, {len(pos)} positions)")
    moves = [a for a in rec["actions"] if a["action"] != "HOLD"]
    if pv <= 0:
        lines.append("    no capital set yet — 'cash <amount>' to start your practice book.")
    elif not moves:
        lines.append("    you are ALIGNED with the system's target — nothing to do.")
    else:
        # Split by whether the gap is even one whole share at this book size.
        # Rounding 0.53 shares up to "~1 sh" (the old display) overstated what is
        # actionable and disagreed with the order ticket, which correctly FLOORS
        # -- buying a whole share to close a half-share gap overshoots the target.
        actionable = [a for a in moves if a["shares"] == a["shares"] and int(a["shares"]) >= 1]
        sub_share = [a for a in moves if a["shares"] == a["shares"] and int(a["shares"]) < 1]
        if actionable:
            # ALL moves, with a running total. Truncating at 8 hid how much
            # capital the listed moves actually commit, which is the one number
            # a reader needs before deciding to act on any of them.
            lines.append(f"    {len(actionable)} actionable move(s) (>= 1 whole share):")
            running = 0.0
            for a in actionable:
                egp = int(a["shares"]) * a["price"]
                if str(a["action"]).upper() in ("BUY", "ADD"):
                    running += egp
                lines.append(f"       {a['action']:5s} {a['ticker']:6s}  {a['current_w']*100:4.1f}% -> "
                             f"{a['target_w']*100:4.1f}%  ({int(a['shares'])} sh @ {a['price']:.2f}"
                             f" = {egp:,.0f} EGP)")
            lines.append(f"       {'':5s} {'TOTAL':6s}  buys/adds commit {running:,.0f} EGP "
                         f"({running/pv*100:.0f}% of the book)" if pv > 0 else "")
        else:
            lines.append("    NOTHING ACTIONABLE today — every remaining gap is under one whole share")
            lines.append("    at your book size. You are as close to the target as integers allow.")
        if sub_share:
            names = ", ".join(a["ticker"] for a in sub_share[:10])
            lines.append(f"    ({len(sub_share)} name(s) under 1 share, skipped: {names})")

    # ---- 3) what you decided about THIS suggestion ------------------------
    dh = pt.decision_history()
    today_dec = dh[dh["as_of_date"].astype(str) == as_of]
    decision = str(today_dec["decision"].iloc[0]) if not today_dec.empty else None
    # ---- 2b) what the book you actually hold can do to you ----------------
    # The sections above describe the TARGET and the gap to it. This describes
    # the book as held: where risk is concentrated (which is not where weight
    # is), how many independent bets it really contains, and the largest
    # single-name exposure in EGP. It recommends nothing -- the system has no
    # measured return edge, so a recommendation here would dress a null result
    # as guidance.
    holds = rec.get("deliberate_holds") or {}
    if holds:
        lines.append(f"    HELD BY YOUR DECISION (not re-proposed): {', '.join(sorted(holds))}")
        lines.append("      the model would exit these; you chose otherwise. They still count in")
        lines.append("      the risk diagnostic below, and the kill switch still applies to them.")

    lines.append("\n[2b] RISK DIAGNOSTIC  (what you hold, not what to do about it)")
    try:
        from analysis.portfolio_diagnostic import run as _diag, format_compact
        lines.append(format_compact(_diag()))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"    unavailable: {exc}")

    try:
        from ops.yardstick import format_section as _yardstick
        lines.append(_yardstick())
    except Exception as exc:  # noqa: BLE001
        lines.append(f"\n[2c] VS DOING NOTHING  unavailable: {exc}")

    lines.append("\n[3] YOUR DECISION on this suggestion")
    if decision:
        note = str(today_dec["note"].iloc[0])
        lines.append(f"    {decision}  —  {note}")
        if decision == "ACCEPTED" and moves:
            lines.append("    NOTE: you accepted but your book is not aligned — re-run 'accept' to rebalance.")
    else:
        lines.append("    NOT YET DECIDED. Tell the system what you did:")
        lines.append("       py -3.11 -m ops.portfolio_tracker accept     # take the system's book")
        lines.append("       py -3.11 -m ops.portfolio_tracker decline    # skip this one")
        lines.append('       py -3.11 -m ops.portfolio_tracker own "..."  # you traded your own view')

    # missed/forgotten earlier days -> nothing was lost, just flag the gaps
    pend = pt.pending_decisions()
    pend = pend[pend["as_of_date"].astype(str) != as_of]      # exclude today's (handled above)
    if not pend.empty:
        days = ", ".join(str(d.date()) for d in pend["as_of_date"].head(5))
        lines.append(f"    NOTE: {len(pend)} earlier suggestion(s) never answered ({days}"
                     + (" ..." if len(pend) > 5 else "") + ").")
        lines.append("       nothing was lost — run 'py -3.11 -m ops.portfolio_tracker catchup' to clear them.")

    # ---- 4) smart feedback: how past choices actually turned out ----------
    lines.append("\n[4] WHAT WE'VE LEARNED SO FAR  (measured, not advice)")
    summ = learning_summary()
    if summ["n_decisions"] == 0:
        lines.append("    no decisions logged yet — this section fills in as you make calls.")
    else:
        lines.append(f"    decisions so far: {summ['counts']}")
        if summ["avg_fwd_after_decline"] is not None:
            v = summ["avg_fwd_after_decline"] * 100
            verdict = "you missed gains" if v > 0 else "sitting out protected you"
            lines.append(f"    after your DECLINEs, the system book then moved {v:+.2f}% on avg — {verdict}.")
        if summ["your_return"] is not None:
            yr, sr = summ["your_return"] * 100, summ["system_return"] * 100
            edge = yr - sr
            who = "you are AHEAD of" if edge > 0 else "you are BEHIND" if edge < 0 else "you match"
            lines.append(f"    since tracking began: your book {yr:+.2f}%  vs system book {sr:+.2f}%  "
                         f"({who} the system by {abs(edge):.2f} pts).")
        else:
            lines.append("    (need a few days of snapshots before your-book-vs-system is meaningful.)")

    # ---- 4b) regime model health (Part 4) ---------------------------------
    # ---- 4a) the whole investable universe, not just what was selected ----
    # The target book shows what the optimizer CHOSE. This shows what it chose
    # FROM, and why each excluded name was excluded -- without it, a name that
    # silently drops out of the universe (delisting, liquidity collapse, a data
    # gap) is invisible until it stops appearing in the target.
    try:
        from data.universe import load_tradable, live_universe_tickers
        uni = load_tradable()
        declared = sorted(live_universe_tickers())
        target_names = set(hold["ticker"]) if not hold.empty else set()
        if not uni.empty:
            # Anchor to the SIGNAL's as-of session, not max(date). The last row
            # in the table can be a near-empty session where one name printed --
            # reading tradability there reported "1 of 51 tradable", which is a
            # phantom-date artifact, not a universe collapse. data/quality.py
            # already guards this for its own checks; this section must too.
            asof_ts = pd.Timestamp(as_of)
            avail = uni.loc[uni["date"] <= asof_ts, "date"]
            last_day = avail.max() if len(avail) else uni["date"].max()
            today_uni = uni[uni["date"] == last_day].set_index("ticker")["is_tradable"]
            prices = pt._latest_prices()
            # A delisted name still has a "latest price" -- its final print. Show
            # when that print was, so a dead name is obvious rather than looking
            # like a live quote.
            with get_connection() as _c:
                last_seen = pd.read_sql(
                    "SELECT ticker, MAX(date) AS d FROM daily_ohlcv GROUP BY ticker",
                    _c).set_index("ticker")["d"]
            lines.append(f"\n[4a] INVESTABLE UNIVERSE  ({len(declared)} declared names, "
                         f"tradability as of {last_day.date()})")
            lines.append(f"       {'ticker':8s}{'price':>10s}  {'last print':12s}"
                         f"{'tradable':10s}{'in target':10s}")
            n_trad, n_stale = 0, 0
            for t in declared:
                is_t = int(today_uni.get(t, 0)) == 1
                n_trad += is_t
                px = float(prices.get(t, float("nan")))
                px_s = f"{px:,.2f}" if px == px else "n/a"
                seen = str(last_seen.get(t, ""))[:10] or "never"
                stale = seen != "never" and pd.Timestamp(seen) < asof_ts - pd.Timedelta(days=30)
                n_stale += stale
                lines.append(f"       {t:8s}{px_s:>10s}  {seen:12s}"
                             f"{'yes' if is_t else 'NO':10s}"
                             f"{'yes' if t in target_names else '-':10s}"
                             + ("   DELISTED/DORMANT" if stale else ""))
            lines.append(f"       {n_trad} of {len(declared)} tradable on {last_day.date()}; "
                         f"{len(target_names)} selected into the target book; "
                         f"{n_stale} delisted/dormant.")
            lines.append("       'NO' means the liquidity/staleness filter excluded it that day.")
            lines.append("       DELISTED/DORMANT = no print for 30+ sessions; the price shown is")
            lines.append("       that name's FINAL print, not a live quote. It cannot be traded.")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"\n[4a] INVESTABLE UNIVERSE  (unavailable: {exc})")

    lines.append("\n[4b] REGIME MODEL HEALTH  (cheap daily check, not a retrain)")
    try:
        lines.append(format_regime_health(regime_health()))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"    (unavailable: {exc})")

    # ---- 4c) idle-cash yield reminder -------------------------------------
    # Silent below the product minimums by design -- see ops/idle_cash.py.
    try:
        from ops.idle_cash import idle_cash_state, format_note
        _note = format_note(idle_cash_state())
        if _note:
            lines.append("\n[4c] IDLE CASH  (freed capital earns ~0% at the broker)")
            lines.append(_note)
    except Exception as exc:  # noqa: BLE001
        logger.warning("idle-cash note unavailable: %s", exc)

    # ---- 5) cost + capacity reality of THIS book --------------------------
    # The model kill switch watches the STRATEGY. This watches the ACCOUNT --
    # they diverge whenever the book is traded differently from the model, which
    # is exactly the situation real money is in.
    try:
        from risk.kill_switch import book_state, format_book_state
        lines.append("\n[4d] BOOK GUARD  (drawdown on YOUR account, not the model's)")
        lines.append(format_book_state(book_state()))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"\n[4d] BOOK GUARD  (unavailable: {exc})")
    lines.append("\n[5] COST & CAPACITY REALITY  (why size matters)")
    hold_w = hold.set_index("ticker")["weight"] if not hold.empty else pd.Series(dtype=float)
    if hold_w.empty or exposure <= 1e-6:
        lines.append("    book is in cash — no capacity/cost exposure right now.")
    else:
        try:
            # Price what you would ACTUALLY trade. The ticket is capped
            # (ops/order_ticket.cap_target_names), so costing the full 27-name
            # target overstated the round trip by roughly 2x. Costing a book
            # that never gets traded is not a conservative assumption, it is
            # simply a wrong one.
            priced_w = hold_w / hold_w.sum()
            cap_info = None
            try:
                from ops.order_ticket import cap_target_names
                capped_w, cap_info = cap_target_names(priced_w, capital_egp=pv)
                if capped_w is not None and len(capped_w):
                    priced_w = capped_w
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not apply the name cap to the cost estimate: %s", exc)
            rc = reality_check(priced_w, pv)
            cap = rc["capacity"]
            sized = "a 1,000,000 EGP reference book" if rc["is_reference"] else f"your {rc['aum']:,.0f} EGP book"
            lines.append(f"    capacity ~ {cap['capacity_egp']:,.0f} EGP before your own trades move "
                         f"{cap['binding_ticker']} (10% of ADV / 3 days).")
            if cap_info:
                lines.append(f"    (costed on the {cap_info['kept']} names the ticket would "
                             f"actually trade, not all {len(hold_w)} target names)")
            lines.append(f"    round-trip cost for {sized}: {rc['roundtrip_cost']*100:.2f}%  "
                         f"(the flat backtest assumed {rc['flat_ref']*100:.2f}%).")
            if rc.get("n_orders"):
                lines.append(f"      -> {rc['n_orders']} orders per side, {rc['total_egp']:,.0f} EGP "
                             f"round trip, of which {rc['flat_fee_share']*100:.0f}% is the FLAT "
                             f"per-order fee.")
                if rc["flat_fee_share"] > 0.5:
                    lines.append("      The flat fee does not shrink with order size — only with the "
                                 "NUMBER of orders. Holding fewer names is the lever here.")
            if rc["flagged"]:
                lines.append(f"    HIGH-IMPACT names at this size: {', '.join(rc['flagged'])}.")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"    (could not compute capacity/cost: {exc})")
    for note in _REAL_TERMS_NOTE:
        lines.append(f"    * {note}")

    # ---- 6) profit estimate (honest ranges, not a promise) ----------------
    lines.append("\n[6] PROFIT ESTIMATE  (a range, because a single number would be a lie)")
    try:
        from analysis.profit_estimate import estimate, format_estimate
        lines.append(format_estimate(estimate(pv)))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"    (estimate unavailable: {exc})")

    # ---- 7) news brief (context for you, NOT a signal) --------------------
    lines.append("\n[7] NEWS ON YOUR HOLDINGS  (read it; the system does NOT trade on it)")
    try:
        from ops.news_brief import format_news_brief
        book_tickers = list(hold_w.index) if not hold_w.empty else list(hold["ticker"])
        lines.append(format_news_brief(book_tickers))
    except Exception as exc:  # noqa: BLE001
        lines.append(f"    (news unavailable: {exc})")

    # ---- snapshot for tomorrow's learning ---------------------------------
    if write:
        snap = snapshot_today(as_of, regime, exposure, decision)
        lines.append(f"\n    [snapshot saved: book {snap['user_value']:,.0f} EGP, "
                     f"system nav {snap['system_nav']}]" if snap["system_nav"] else
                     f"\n    [snapshot saved: book {snap['user_value']:,.0f} EGP]")

    lines.append("\n" + "=" * 68)
    # This used to read "Paper only — no real orders". That was true for the
    # simulator and became FALSE the moment real capital went in, while
    # remaining the last line the reader sees. It is now DERIVED from the gate
    # rather than asserted, so it cannot go stale again.
    try:
        from ops.go_live_gate import evaluate_go_live
        ready = bool(evaluate_go_live()["go_live"])
    except Exception:  # noqa: BLE001
        ready = False
    if not ready:
        lines.append("GO-LIVE GATE NOT MET. Real capital here is deployed AGAINST the")
        lines.append("system's own pre-committed criteria — deliberately, not by oversight.")
    lines.append("Measured: alpha not distinguishable from zero, does not beat T-bills in")
    lines.append("real terms, regime signal not significant and does not generalise.")
    lines.append("=" * 68)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    # Arabic headlines need UTF-8; the Windows console defaults to cp1252 and
    # would crash on them. Reconfigure stdout so the news section prints.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    logging.basicConfig(level=logging.ERROR)
    write = "--no-write" not in sys.argv
    print(daily_report(write=write))
