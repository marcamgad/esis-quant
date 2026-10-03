"""
Day-trade experiment tracker: buy in the morning, flat by the close, measured
NET of fees.

WHY THIS MODULE EXISTS. The plan is to enter each morning and sell everything
before the close, targeting ~0.3-0.4% per day. Whether that works is an
empirical question, and it is being tested in a SIMULATOR, so the correct
response is to measure it properly rather than argue about it. The one thing
that must not happen is measuring it wrong -- and the way this specific strategy
gets measured wrong is by looking at the GROSS move and forgetting that a full
round trip is paid every single day.

THE ARITHMETIC THAT DOMINATES EVERYTHING. A round trip costs `round_trip_cost`
(commission both sides + regulatory fees + the spread you cross). Repeated daily
it compounds: with a 0.59% round trip, 252 sessions of perfectly flat picks
leave you at 0.9941^252 = 22.5% of starting capital. So the daily gross move
must clear the fee before a single pound of profit exists. `breakeven_move()`
reports exactly what that hurdle is, and `required_gross()` what is needed to
hit a target.

FEES ARE A GUESS UNTIL YOU CALIBRATE THEM. config.THNDR_COMMISSION_PCT is the
brokerage commission only. Egyptian trades also carry exchange/clearing (MCDR)
fees and stamp duty, and you additionally pay the bid-ask spread, which is not a
"fee" but costs exactly like one. Rather than assert numbers I cannot verify,
this module takes them as parameters and -- more usefully -- can INFER your true
all-in cost from your own simulator results (`implied_cost_from_log`): if you
log the buy and sell totals, the difference between the gross move and your
realized net IS the cost, measured rather than assumed.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.day_trade")

# Brokerage commission per side (config) plus a placeholder for the EGX/MCDR
# fees and stamp duty that a real Egyptian ticket also carries. VERIFY THESE
# AGAINST A REAL THNDR CONTRACT NOTE -- they are deliberately conservative
# stand-ins, not researched values, and implied_cost_from_log() will replace
# them with your measured reality once you have a few days logged.
OTHER_FEES_PER_SIDE = 0.0005      # exchange + clearing + stamp, placeholder
DEFAULT_SPREAD_COST = 0.0010      # half-spread crossed on each side, placeholder
TRADING_DAYS = 252


def round_trip_cost(commission_per_side: float = None, other_fees_per_side: float = None,
                    spread_per_side: float = None) -> dict:
    """All-in cost of buying and selling once, as a fraction of notional."""
    from backtest.fees import book_pct_per_side
    c = (book_pct_per_side(config.BACKTEST_DEFAULT_AUM_EGP)
         if commission_per_side is None else commission_per_side)
    o = OTHER_FEES_PER_SIDE if other_fees_per_side is None else other_fees_per_side
    s = DEFAULT_SPREAD_COST if spread_per_side is None else spread_per_side
    per_side = c + o + s
    return {"commission_per_side": c, "other_fees_per_side": o, "spread_per_side": s,
            "per_side": per_side, "round_trip": 2 * per_side}


def breakeven_move(cost: float = None) -> float:
    """The gross intraday move that leaves you exactly flat after fees."""
    return round_trip_cost()["round_trip"] if cost is None else cost


def required_gross(target_net_daily: float, cost: float = None) -> float:
    """Gross daily move needed to net `target_net_daily` after a round trip."""
    c = breakeven_move(cost)
    return target_net_daily + c


def annualize(net_daily: float, days: int = TRADING_DAYS) -> float:
    return (1.0 + net_daily) ** days - 1.0


def target_table(targets=(0.001, 0.002, 0.003, 0.004, 0.005), cost: float = None) -> pd.DataFrame:
    """For each net daily target: the gross move required and the annual result."""
    c = breakeven_move(cost)
    rows = []
    for t in targets:
        rows.append({"net_daily": t, "gross_needed": t + c,
                     "annual_net": annualize(t), "cost_drag_annual": annualize(-c)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# logging real simulator results
# --------------------------------------------------------------------------
def _ensure_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS day_trade_log (
                date DATE PRIMARY KEY,
                buy_total REAL,       -- EGP deployed at entry
                sell_total REAL,      -- EGP received at exit
                gross_pct REAL,       -- (sell/buy - 1) BEFORE fees, as reported
                net_egp REAL,         -- sell_total - buy_total (what actually changed)
                net_pct REAL,
                note TEXT
            )""")


def log_day(buy_total: float, sell_total: float, date: str = None, note: str = "") -> dict:
    """Record one full day: what you put in and what came back out.

    net_pct is measured, not modelled -- sell_total already reflects every fee
    the simulator charged, so this is the honest number.
    """
    _ensure_table()
    date = date or pd.Timestamp.now().strftime("%Y-%m-%d")
    buy_total, sell_total = float(buy_total), float(sell_total)
    net_egp = sell_total - buy_total
    net_pct = (sell_total / buy_total - 1.0) if buy_total > 0 else 0.0
    with get_connection() as conn:
        conn.execute("INSERT INTO day_trade_log (date, buy_total, sell_total, gross_pct, "
                     "net_egp, net_pct, note) VALUES (?, ?, ?, NULL, ?, ?, ?) "
                     "ON CONFLICT(date) DO UPDATE SET buy_total=excluded.buy_total, "
                     "sell_total=excluded.sell_total, net_egp=excluded.net_egp, "
                     "net_pct=excluded.net_pct, note=excluded.note",
                     (date, buy_total, sell_total, net_egp, net_pct, note or None))
    logger.info("log_day %s: %.2f -> %.2f (%.3f%%)", date, buy_total, sell_total, net_pct * 100)
    return {"date": date, "buy_total": buy_total, "sell_total": sell_total,
            "net_egp": net_egp, "net_pct": net_pct}


def load_log() -> pd.DataFrame:
    _ensure_table()
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM day_trade_log ORDER BY date", conn, parse_dates=["date"])


def implied_cost_from_log(assumed_gross: float = 0.0) -> dict:
    """Infer your true all-in round-trip cost from logged days.

    If you assume the average GROSS intraday move across many days is roughly
    `assumed_gross` (0 is the honest prior -- intraday direction is close to a
    coin flip), then the average NET you realize is that gross minus the cost.
    So cost ~ assumed_gross - mean(net). Crude, but it is measured from your own
    fills instead of guessed from a fee schedule.
    """
    df = load_log()
    if df.empty:
        return {"n_days": 0}
    mean_net = float(df["net_pct"].mean())
    return {"n_days": int(len(df)), "mean_net_pct": mean_net,
            "implied_round_trip": float(assumed_gross - mean_net),
            "modelled_round_trip": round_trip_cost()["round_trip"]}


def summarize() -> dict:
    """How the experiment is actually going, net of everything."""
    df = load_log()
    if df.empty:
        return {"n_days": 0}
    net = df["net_pct"]
    cum = float((1 + net).prod() - 1)
    wins = int((net > 0).sum())
    out = {"n_days": int(len(df)), "cum_net_pct": cum, "mean_net_pct": float(net.mean()),
           "median_net_pct": float(net.median()), "std_net_pct": float(net.std(ddof=1)) if len(net) > 1 else 0.0,
           "win_rate": wins / len(net), "best": float(net.max()), "worst": float(net.min()),
           "total_egp": float(df["net_egp"].sum()),
           "projected_annual": annualize(float(net.mean())) if len(net) else float("nan")}
    return out


def format_summary(s: dict, target_net_daily: float = 0.0035) -> str:
    if not s or s.get("n_days", 0) == 0:
        return ("  no day-trade days logged yet.\n"
                "  log one with:  py -3.11 -m analysis.day_trade log <buy_total> <sell_total>")
    c = round_trip_cost()
    lines = [f"DAY-TRADE EXPERIMENT — {s['n_days']} session(s), all figures NET of fees",
             f"  cumulative      : {s['cum_net_pct']*100:+.2f}%   ({s['total_egp']:+,.2f} EGP)",
             f"  per day         : mean {s['mean_net_pct']*100:+.3f}%  median {s['median_net_pct']*100:+.3f}%"
             f"  std {s['std_net_pct']*100:.3f}%",
             f"  win rate        : {s['win_rate']*100:.0f}%   best {s['best']*100:+.2f}%  worst {s['worst']*100:+.2f}%",
             f"  at this pace    : {s['projected_annual']*100:+.1f}% per year",
             f"  target          : {target_net_daily*100:.2f}%/day net -> "
             f"{annualize(target_net_daily)*100:+.0f}%/year",
             f"  modelled hurdle : every day must gross > {c['round_trip']*100:.2f}% just to break even"]
    if s["n_days"] < 20:
        lines.append(f"  NOTE: {s['n_days']} day(s) proves nothing -- daily results are dominated by noise.")
    return "\n".join(lines)


def _main(argv=None):
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "status"
    if cmd == "log" and len(argv) >= 3:
        r = log_day(float(argv[1]), float(argv[2]),
                    note=" ".join(argv[3:]) if len(argv) > 3 else "")
        print(f"logged {r['date']}: {r['buy_total']:,.2f} -> {r['sell_total']:,.2f} "
              f"= {r['net_egp']:+,.2f} EGP ({r['net_pct']*100:+.3f}% net)")
        print()
        print(format_summary(summarize()))
    elif cmd == "hurdle":
        c = round_trip_cost()
        print(f"MODELLED ROUND-TRIP COST: {c['round_trip']*100:.2f}%")
        print(f"   commission {c['commission_per_side']*100:.3f}% + other fees "
              f"{c['other_fees_per_side']*100:.3f}% + spread {c['spread_per_side']*100:.3f}% per side")
        print("   (placeholders -- verify against a real Thndr contract note)")
        print()
        t = target_table()
        print(f"  {'net/day':>9}{'gross needed':>14}{'-> per year':>14}")
        for r in t.itertuples():
            print(f"  {r.net_daily*100:>8.2f}%{r.gross_needed*100:>13.2f}%{r.annual_net*100:>13.0f}%")
        print(f"\n  doing nothing but paying fees for a year: {t['cost_drag_annual'].iloc[0]*100:+.1f}%")
    elif cmd == "ticket":
        cap = float(argv[1]) if len(argv) > 1 else None
        o = write_day_ticket(cap)
        print(f"wrote {o['path']}")
        print(f"  capital {o['capital']:,.2f} EGP  ({o['capital_source']})")
        if o.get("cum_pct") is not None:
            print(f"  cumulative since day 1: {o['cum_pct']*100:+.2f}%")
        print(f"  signal as-of {o['as_of']}  |  {o['n']} names  |  "
              f"{o['spent']:,.0f} EGP deployed, {o['left']:,.0f} left")
        print(f"  break-even today: must gross > {o['hurdle']*100:.2f}%")
        import csv as _csv
        with open(o["path"], encoding="utf-8-sig") as f:
            for row in _csv.reader(l for l in f if not l.lstrip().startswith("#")):
                if row and row[0] == "BUY":
                    print(f"     BUY {row[1]:6s} {row[5]:>4} sh @ {row[3]:>8} = {row[6]:>6} EGP")
    elif cmd == "implied":
        print(implied_cost_from_log())
    else:
        print(format_summary(summarize()))




# --------------------------------------------------------------------------
# daily buy ticket: deploy a fixed amount of cash each morning
# --------------------------------------------------------------------------
DAYTRADE_DIR = None          # defaults to <project>/orders
MIN_ADV_EGP = 20_000_000     # only names deep enough to round-trip cheaply


def current_capital(default: float = 1000.0) -> dict:
    """How much to deploy tomorrow: whatever yesterday's sell brought back.

    The strategy compounds -- each morning you put in everything you have, so
    the base is the LAST LOGGED sell_total, not a fixed number. Falls back to
    `default` before the first day is logged.
    """
    df = load_log()
    if df.empty:
        return {"capital": float(default), "source": "default (no days logged yet)",
                "as_of": None}
    last = df.iloc[-1]
    return {"capital": float(last["sell_total"]), "source": "last logged sell_total",
            "as_of": str(pd.Timestamp(last["date"]).date()),
            "start": float(df["buy_total"].iloc[0]),
            "cum_pct": float(last["sell_total"] / df["buy_total"].iloc[0] - 1)}


def write_day_ticket(capital: float = None, max_names: int = 8,
                     min_adv: float = MIN_ADV_EGP, path: str = None) -> dict:
    """Buy list for deploying `capital` from all cash this morning.

    Weights come from the system's CURRENT target book -- the only allocation
    the system actually produces. They are then restricted to the most LIQUID
    names, because this book is round-tripped the same day and the fee+spread
    hurdle is what decides the outcome; a thin name's spread alone can exceed
    the whole daily target. Shares are whole and never overspend the capital.

    This sizes a book. It does NOT predict which names rise today -- the system
    has no intraday signal, and nothing here should be read as one.
    """
    import os
    import config as _cfg
    from ops.trade_list import latest_trade_list
    from ops import portfolio_tracker as _pt
    from backtest.costs import average_daily_value
    from portfolio.integer_allocator import greedy_integer_allocation

    cap_info = current_capital()
    if capital is None:
        capital = cap_info["capital"]

    tl = latest_trade_list()
    if tl.empty:
        raise RuntimeError("no suggestion logged yet -- run the daily job first")
    as_of = str(tl["as_of_date"].iloc[0].date())
    hold = tl[tl["ticker"] != "__CASH__"]
    if hold.empty:
        raise RuntimeError("system is de-risked (cash target) -- no buy list today")

    w = hold.set_index("ticker")["weight"].astype(float)
    prices = _pt._latest_prices()
    adv = average_daily_value(list(w.index))

    liquid = [t for t in w.index if float(adv.get(t, 0) or 0) >= min_adv]
    liquid = sorted(liquid, key=lambda t: -float(adv.get(t, 0)))[:max_names]
    if not liquid:
        liquid = sorted(w.index, key=lambda t: -float(adv.get(t, 0) or 0))[:max_names]
    tgt = w.reindex(liquid).fillna(0.0)
    tgt = tgt / tgt.sum()

    alloc = greedy_integer_allocation(tgt, prices, float(capital))
    shares = {t: int(n) for t, n in alloc["orders"].items()}
    left = float(capital) - sum(shares.get(t, 0) * float(prices[t]) for t in shares)

    # DEPLOY-ALL pass. The tracking-error allocator stops once another share
    # would push a name past its target weight -- correct when you are holding a
    # book, wrong here: this capital is round-tripped today, so cash left behind
    # is simply capital not working. Keep buying the affordable name that is
    # furthest BELOW its target weight until nothing else fits.
    px = {t: float(prices[t]) for t in tgt.index
          if float(prices.get(t, float("nan"))) == float(prices.get(t, float("nan")))
          and float(prices.get(t, 0)) > 0}
    for _ in range(10_000):
        affordable = [t for t, p_ in px.items() if p_ <= left]
        if not affordable:
            break
        total = sum(shares.get(t, 0) * px[t] for t in px) + left
        deficit = {t: float(tgt[t]) - (shares.get(t, 0) * px[t] / total if total > 0 else 0.0)
                   for t in affordable}
        pick = max(affordable, key=lambda t: deficit[t])
        shares[pick] = shares.get(pick, 0) + 1
        left -= px[pick]

    rows, spent = [], 0.0
    for t, n in sorted(shares.items(), key=lambda kv: -kv[1] * float(prices[kv[0]])):
        p = float(prices.get(t, float("nan")))
        if not (p == p and p > 0):
            continue
        sh = int(n)
        egp = sh * p
        spent += egp
        rows.append(["BUY", t, round(float(tgt.get(t, 0.0)) * 100, 1), round(p, 2),
                     round(p * 1.03, 2), sh, round(egp, 0),
                     f"{float(adv.get(t,0)):,.0f}", "", ""])
    rows.sort(key=lambda r: -r[6])

    base = DAYTRADE_DIR or os.path.join(_cfg.BASE_DIR, "orders")
    os.makedirs(base, exist_ok=True)
    day = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = path or os.path.join(base, f"daytrade_{day}.csv")
    hurdle = round_trip_cost()["round_trip"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        f.write(f"# ESIS-QUANT DAY-TRADE buy list  {day}  (signal as-of {as_of}, "
                f"regime {tl['regime'].iloc[0]})\n")
        f.write(f"# deploy {capital:,.0f} EGP this morning, sell all before the close.\n")
        f.write(f"# must gross > {hurdle*100:.2f}% today just to break even after fees.\n")
        f.write("# sizing only - the system has NO intraday forecast.\n")
        import csv as _csv
        w_ = _csv.writer(f)
        w_.writerow(["action", "ticker", "target_pct", "ref_price", "max_price",
                     "shares", "egp", "adv_egp", "FILLED_shares", "FILLED_price"])
        w_.writerows(rows)
    return {"path": path, "as_of": as_of, "n": len(rows), "spent": spent,
            "left": float(capital) - spent, "hurdle": hurdle,
            "capital": float(capital), "capital_source": cap_info["source"],
            "cum_pct": cap_info.get("cum_pct")}


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    _main()
