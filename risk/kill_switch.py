"""
Hard kill-switch (audit Tier-2 #10).

WHY. The regime overlay is the entire drawdown defense, and it can be WRONG --
the India cross-market test showed the regime model cashing out at the wrong
times and missing the real drawdown. A single model carrying all risk control
is a single point of failure. The kill-switch is a model-INDEPENDENT backstop:
it does not care what the regime model believes; if realized loss breaches a
hard limit, it flattens the book and stays flat for a cooldown.

RULES (from config, no look-ahead -- each day uses only losses through that day):
  * daily loss  <= KILL_SWITCH_MAX_DAILY_LOSS   (e.g. -7% in one session), or
  * rolling 5-session loss <= KILL_SWITCH_MAX_WEEKLY_LOSS (e.g. -12%)
  -> trip: exposure forced to 0 for KILL_SWITCH_COOLDOWN_DAYS sessions, then
     released (the regime overlay resumes control).

This is intentionally crude and rare. It is not meant to improve Sharpe; it is
meant to bound the worst case when the model is simply wrong.
"""

import numpy as np
import pandas as pd

import config


def check_kill_switch(returns: pd.Series,
                      max_daily: float = None, max_weekly: float = None,
                      cooldown: int = None, weekly_window: int = 5) -> pd.Series:
    """Return a per-date exposure multiplier in {0,1}: 0 while the kill-switch
    is tripped/cooling-down, 1 otherwise. Multiply this into any other exposure.

    Causal: the decision for day t uses realized returns THROUGH t only, and the
    multiplier applies from the day the breach is observed onward.
    """
    max_daily = config.KILL_SWITCH_MAX_DAILY_LOSS if max_daily is None else max_daily
    max_weekly = config.KILL_SWITCH_MAX_WEEKLY_LOSS if max_weekly is None else max_weekly
    cooldown = config.KILL_SWITCH_COOLDOWN_DAYS if cooldown is None else cooldown

    r = returns.dropna()
    mult = np.ones(len(r))
    vals = r.to_numpy()
    cool = 0
    for t in range(len(r)):
        if cool > 0:
            mult[t] = 0.0
            cool -= 1
            continue
        daily = vals[t]
        wk_start = max(0, t - weekly_window + 1)
        weekly = float(np.prod(1 + vals[wk_start:t + 1]) - 1)
        if daily <= max_daily or weekly <= max_weekly:
            mult[t] = 0.0            # flatten from the breach day
            cool = cooldown          # and stay flat for the cooldown
    return pd.Series(mult, index=r.index, name="kill_switch")


def kill_switch_events(returns: pd.Series, **kw) -> pd.DataFrame:
    """The dates on which the kill-switch was active (for the audit trail)."""
    m = check_kill_switch(returns, **kw)
    tripped = m[m == 0.0]
    return pd.DataFrame({"date": tripped.index, "flat": True})


def latest_state(returns: pd.Series, **kw) -> dict:
    """Is the kill-switch tripped as of the last observation? For the live path
    to consult before emitting a trade list."""
    m = check_kill_switch(returns, **kw)
    if m.empty:
        return {"tripped": False, "as_of": None}
    return {"tripped": bool(m.iloc[-1] == 0.0), "as_of": str(m.index[-1].date()),
            "recent_flat_days": int((m.tail(kw.get("cooldown", config.KILL_SWITCH_COOLDOWN_DAYS) + 1) == 0).sum())}


# --------------------------------------------------------------------------
# LIVE wiring -- this was backtest-only until 2026-08
# --------------------------------------------------------------------------
def live_returns() -> "pd.Series":
    """The strategy's own realized daily returns, from the FORWARD paper record.

    Deliberately the paper NAV and not the user's book: the kill-switch exists
    to catch the STRATEGY being wrong, and the paper series is the strategy's
    uninterrupted track. A user book with sporadic manual trades would confound
    "the model is failing" with "the human traded differently today".
    """
    import pandas as pd
    from data.database import get_connection
    with get_connection() as conn:
        df = pd.read_sql("SELECT date, daily_return FROM paper_trading_nav ORDER BY date",
                         conn, parse_dates=["date"])
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["daily_return"].dropna()


def live_state() -> dict:
    """Is the kill-switch tripped RIGHT NOW? Consulted by the live trade list.

    Returns exposure_multiplier in {0, 1} plus enough context to explain the
    decision. Fails OPEN (multiplier 1) when there is too little history to
    judge -- a brand-new record must not be read as "flat", but the caller is
    told the check was inconclusive rather than passed.
    """
    r = live_returns()
    if len(r) < 5:
        return {"tripped": False, "multiplier": 1.0, "n_days": int(len(r)),
                "conclusive": False,
                "reason": f"only {len(r)} forward sessions -- too few to evaluate"}
    mult = check_kill_switch(r)
    tripped = bool(mult.iloc[-1] == 0.0)
    last = r.iloc[-1]
    wk = float((1 + r.tail(5)).prod() - 1)
    return {"tripped": tripped, "multiplier": 0.0 if tripped else 1.0,
            "n_days": int(len(r)), "conclusive": True,
            "last_daily": float(last), "last_weekly": wk,
            "reason": (f"daily {last*100:+.2f}% / 5-session {wk*100:+.2f}% breached "
                       f"({config.KILL_SWITCH_MAX_DAILY_LOSS*100:.0f}% / "
                       f"{config.KILL_SWITCH_MAX_WEEKLY_LOSS*100:.0f}%)")
            if tripped else "within limits"}


# --- guard on YOUR ACTUAL BOOK, not the strategy's paper track ---------------
#
# WHY THIS IS SEPARATE FROM live_returns(). The kill switch above deliberately
# reads the strategy's uninterrupted paper NAV, so that "the model is failing"
# is never confused with "the human traded differently today". That is the right
# design for judging the MODEL.
#
# It is the wrong design for protecting an ACCOUNT. The live book deviates from
# the model by choice -- names skipped, cash held deliberately -- so the paper
# series can be perfectly healthy while the real account is down badly, and
# nothing would fire. Going to real money makes that gap material rather than
# theoretical.
#
# So: two independent guards watching two different things. This one asks only
# "is the money falling", and it cannot be satisfied by the model being fine.
#
# IT DOES NOT SELL. Nothing in this system executes, and an automatic
# liquidation recommendation on a book whose deviations are deliberate would be
# presumptuous. It escalates: warn -> stop adding risk -> recommend flattening,
# and says which and why.

BOOK_WARN_DRAWDOWN = -0.10        # from peak: say something
BOOK_BLOCK_DRAWDOWN = -0.15       # from peak: no new buys
BOOK_FLATTEN_DRAWDOWN = -0.25     # from peak: recommend going flat (= GO_LIVE_MAX_DRAWDOWN)


def book_nav() -> "pd.Series":
    """Total value of YOUR account (positions + cash) per session."""
    import pandas as pd
    from data.database import get_connection
    with get_connection() as conn:
        df = pd.read_sql("SELECT date, user_value FROM user_daily_snapshot "
                         "WHERE user_value IS NOT NULL ORDER BY date",
                         conn, parse_dates=["date"])
    if df.empty:
        return pd.Series(dtype=float)
    # user_value is ALREADY positions + cash (see the ops.daily_report snapshot).
    # Adding `cash` again double-counted it -- caught showing a 13,119 peak on a
    # 9,963 book, which would have made this guard fire far too late.
    return df["user_value"].astype(float).set_axis(df["date"]).dropna()


def cash_flows() -> "pd.Series":
    """Deposits (+) and withdrawals (-) by date, EGP. Empty if none recorded."""
    import pandas as pd
    from data.database import get_connection
    with get_connection() as conn:
        ensure_cash_flows_table(conn)
        df = pd.read_sql("SELECT date, amount FROM cash_flows", conn, parse_dates=["date"])
    return df.groupby("date")["amount"].sum() if not df.empty else pd.Series(dtype=float)


def ensure_cash_flows_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS cash_flows (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT NOT NULL, amount REAL NOT NULL,
        note TEXT, recorded_at TEXT)""")


def book_twr(nav=None, flows=None) -> "pd.Series":
    """Time-weighted growth index of the account (starts at 1.0).

    WHY. A deposit raises the account value without anyone earning anything, and
    a withdrawal lowers it without any loss. The guard used raw value, so a
    deposit lifted the peak and a later withdrawal would read as a drawdown --
    the +3,242 jump between 23 Aug and 3 Sep was almost certainly money in, not
    profit. Each period's return here is (V_t - flows in (t-1, t]) / V_{t-1} - 1,
    so only market moves change the index. Unrecorded flows still distort it;
    record them with `portfolio_tracker deposit/withdraw`.
    """
    import pandas as pd
    nav = book_nav() if nav is None else nav
    flows = cash_flows() if flows is None else flows
    if len(nav) < 2:
        return pd.Series(1.0, index=nav.index)
    rets = []
    for prev_d, d in zip(nav.index[:-1], nav.index[1:]):
        f = float(flows[(flows.index > prev_d) & (flows.index <= d)].sum()) if len(flows) else 0.0
        v0, v1 = float(nav.loc[prev_d]), float(nav.loc[d])
        rets.append((v1 - f) / v0 - 1.0 if v0 > 0 else 0.0)
    return pd.Series([1.0] + list(pd.Series(rets).add(1).cumprod()), index=nav.index)


def book_state() -> dict:
    """Drawdown-from-peak on the real account, with an escalating verdict.

    Reports `conclusive: False` on a short history rather than pretending a
    two-day record proves anything -- but unlike the model kill switch, it still
    reports the drawdown, because a real 20% loss matters on day three just as
    much as on day three hundred.

    The drawdown is measured on the TIME-WEIGHTED index (book_twr), so deposits
    and withdrawals recorded in cash_flows do not move it.
    """
    nav = book_nav()
    if len(nav) < 2:
        return {"level": "none", "conclusive": False, "n_days": int(len(nav)),
                "drawdown": float("nan"), "peak": float("nan"), "now": float("nan"),
                "block_new_risk": False,
                "reason": "not enough book history yet to compute a drawdown"}
    twr = book_twr(nav)
    peak = float(nav.loc[twr.idxmax()])
    now = float(nav.iloc[-1])
    dd = float(twr.iloc[-1] / twr.cummax().iloc[-1] - 1.0)
    if dd <= BOOK_FLATTEN_DRAWDOWN:
        level, reason = "flatten", (f"account is {dd*100:.1f}% below its peak "
                                    f"({peak:,.0f} -> {now:,.0f}) — at or past the "
                                    f"{BOOK_FLATTEN_DRAWDOWN*100:.0f}% level you "
                                    f"pre-committed to as intolerable")
    elif dd <= BOOK_BLOCK_DRAWDOWN:
        level, reason = "block", (f"account is {dd*100:.1f}% below its peak — "
                                  "not adding new risk until it recovers")
    elif dd <= BOOK_WARN_DRAWDOWN:
        level, reason = "warn", f"account is {dd*100:.1f}% below its peak"
    else:
        level, reason = "none", ""
    return {"level": level, "conclusive": len(nav) >= 5, "n_days": int(len(nav)),
            "drawdown": dd, "peak": peak, "now": now,
            "block_new_risk": level in ("block", "flatten"), "reason": reason}


def format_book_state(s: dict) -> str:
    if s["level"] == "none":
        return (f"  book guard: no drawdown alert "
                f"({s['n_days']} session(s) tracked)" if s["n_days"] >= 2 else
                "  book guard: not enough history yet")
    return f"  *** BOOK GUARD [{s['level'].upper()}]: {s['reason']}"
