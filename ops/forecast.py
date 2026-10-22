"""
4-week (20-session) price forecasts as RANGES with probabilities, checked
against history before they are shown.

WHY A RANGE, NOT ONE NUMBER. Nothing in this system predicts the DIRECTION of
an EGX stock over a month better than chance (selection, momentum, news events,
earnings drift: all failed out of sample). What can be forecast honestly is the
SIZE of the likely move: volatility clusters, so a stock that has been swinging
will keep swinging. The forecast therefore says "where the price will probably
be", with stated odds, and the centre carries only the market's long-run drift.

MODEL (fixed; v1-2026-10-17):
  sigma_now   EWMA daily volatility (lambda 0.94) of the stock's adjusted returns,
              floored at 60% of its own 1-year volatility (a quiet week is not
              a quiet month)
  shape       the stock's own past 20-session returns, each divided by the
              volatility known at its start -> fat tails and skew come from its
              own history (needs >= 1 year; else a market-wide shape)
  centre      0: the middle of the range is today's price (direction is not
              forecastable here; a market-drift centre made the odds worse).
              Beta (weekly returns vs the real EGX30, 2 years) is reported.
  quantiles   P10 / P25 / P50 / P75 / P90 = price x (1 + centre + shape_q x sigma_now x sqrt(20))
  news        dated events shown as context: results likely inside the window
              (from the company's own reporting lag), recent triggers. They do
              NOT move the centre -- news-event tests here were null.

CALIBRATION. backtest_calibration() rebuilds forecasts at past month-ends using
only data known then and counts how often the real price landed inside the
P10-P90 band (should be ~80%) and P25-P75 (~50%). Every live forecast is saved
in price_forecasts and score() checks them when they mature.

    py -3.11 -m ops.forecast COMI EGAL          # forecasts
    py -3.11 -m ops.forecast --calibrate        # history check
    py -3.11 -m ops.forecast --score            # matured live forecasts
"""

import json
import sys

import numpy as np
import pandas as pd

from data.database import get_connection

MODEL = "v1-2026-10-17"
H = 20
LAMBDA = 0.94
VOL_FLOOR = 0.6
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
MARKET_DRIFT_YEAR = 0.20          # EGX30 price CAGR 2016-2026 was ~20.6% (data/market.py)


_PANEL = {}


def _panel():
    if "r" not in _PANEL:
        from backtest.engine import load_price_returns
        from data import market
        r = load_price_returns().clip(-0.5, 1.0)
        _PANEL["r"] = r
        _PANEL["egx"] = market.returns(r.index)
    return _PANEL["r"], _PANEL["egx"]


def _ewma_vol(x: pd.Series) -> float:
    x = x.dropna()
    w = LAMBDA ** np.arange(len(x))[::-1]
    return float(np.sqrt(np.sum(w * x.values ** 2) / np.sum(w)))


def _shape(s: pd.Series) -> np.ndarray:
    """Past 20-session returns divided by the volatility known when each began."""
    s = s.dropna()
    if len(s) < 250 + H:
        return np.array([])
    vol = s.rolling(60).std()
    out = []
    for i in range(60, len(s) - H, 5):
        v = vol.iloc[i - 1]
        if v and np.isfinite(v) and v > 0:
            out.append((np.prod(1 + s.iloc[i:i + H].values) - 1) / (v * np.sqrt(H)))
    return np.array(out)


def _market_shape(r: pd.DataFrame) -> np.ndarray:
    if "mshape" not in _PANEL:
        sample = [c for c in r.columns if r[c].notna().sum() > 1000][:60]
        _PANEL["mshape"] = np.concatenate([_shape(r[c]) for c in sample]) if sample else np.array([0.0])
    return _PANEL["mshape"]


def forecast(ticker: str, as_of=None, record_it: bool = False) -> dict:
    r, egx = _panel()
    d = pd.Timestamp(as_of) if as_of is not None else r.index[-1]
    if ticker not in r:
        return {}
    s = r[ticker][r.index <= d].dropna()
    if len(s) < 120:
        return {"ticker": ticker, "error": "less than 120 sessions of history"}
    with get_connection() as conn:
        row = conn.execute("SELECT date, close FROM daily_ohlcv WHERE ticker = ? AND date <= ? AND close > 0 "
                           "ORDER BY date DESC LIMIT 1", (ticker, str(d.date()))).fetchone()
    price, price_date = float(row[1]), row[0]
    yr = s.tail(250)
    sig = max(_ewma_vol(s.tail(250)), VOL_FLOOR * float(yr.std()))
    # beta on WEEKLY returns: Mubasher's EGX30 file skips some days, and a missing
    # day pairs the stock's move with a 0 for the index (then a double move the
    # next day), which drags a daily beta towards zero (AMOC read 0.04).
    two = s.tail(500)
    sw = (1 + two.fillna(0)).resample("W").prod() - 1
    mw = (1 + egx.reindex(two.index).fillna(0)).resample("W").prod() - 1
    ok = (sw != 0) & (mw != 0)
    beta = float(np.cov(sw[ok], mw[ok])[0, 1] / np.var(mw[ok])) if ok.sum() > 30 and np.var(mw[ok]) > 0 else 1.0
    beta = float(np.clip(beta, 0.0, 2.5))
    # NEUTRAL centre: the middle of the range is today's price. A beta x market-drift
    # centre (20%/yr) was tried first and made the up/down odds WORSE than a coin
    # (Brier 0.277 vs 0.25): only 48.6% of past 4-week moves were up. Direction is
    # not forecastable here, so the forecast does not pretend to know it.
    centre = 0.0
    shp = _shape(s)
    own = len(shp) >= 40
    if not own:
        shp = _market_shape(r)
    shp = shp - np.median(shp)                     # the centre is set by drift, not by the past sample's luck
    z = np.quantile(shp, QS)
    rets = centre + z * sig * np.sqrt(H)
    rets = np.maximum(rets, -0.95)
    probs_up = float(np.mean(centre + shp * sig * np.sqrt(H) > 0))
    p_up10 = float(np.mean(centre + shp * sig * np.sqrt(H) >= 0.10))
    p_dn10 = float(np.mean(centre + shp * sig * np.sqrt(H) <= -0.10))
    out = {"ticker": ticker, "model": MODEL, "as_of": str(d.date()), "price": price, "price_date": price_date,
           "horizon_sessions": H, "daily_vol": sig, "month_vol": sig * np.sqrt(H), "beta": beta, "centre": centre,
           "quantiles": {f"p{int(q * 100)}": float(price * (1 + x)) for q, x in zip(QS, rets)},
           "p_up": probs_up, "p_up10": p_up10, "p_down10": p_dn10, "shape": "own history" if own else "market-wide",
           "events": _events(ticker, d)}
    if record_it:
        _record(out)
    return out


def _events(ticker: str, d: pd.Timestamp) -> list:
    """Dated context that may move the price inside the window. Not used in the numbers."""
    ev = []
    with get_connection() as conn:
        q = conn.execute("SELECT MAX(period_end), MAX(available_from) FROM fundamentals WHERE ticker = ? "
                         "AND frequency = 'quarterly' AND line_item = 'Net Income' AND available_from <= ?",
                         (ticker, str(d.date()))).fetchone()
        try:
            t = conn.execute("SELECT grp, detail, event_date FROM watchlist_triggers WHERE ticker = ? AND "
                             "event_date > ? AND event_date <= ? ORDER BY event_date DESC LIMIT 3",
                             (ticker, str((d - pd.Timedelta(days=10)).date()), str(d.date()))).fetchall()
        except Exception:  # noqa: BLE001
            t = []
    if q and q[0]:
        nxt = pd.Timestamp(q[0]) + pd.DateOffset(months=3)       # the next quarter's end
        press = nxt + pd.Timedelta(days=51)                      # median press lag measured on EGX
        end = d + pd.Timedelta(days=28)
        if d <= press <= end:
            ev.append(f"next results (quarter ending {nxt.date()}) likely in the press around {press.date()}, "
                      f"inside the 4 weeks: expect a bigger move than usual either way")
    for g, detail, day in t:
        ev.append(f"{day} [{g}] {detail[:120]}")
    return ev


def _record(f: dict) -> None:
    import sqlite3
    import config
    from ops import agent_guard as ag
    con = ag.guarded(sqlite3.connect(config.DB_PATH, timeout=30), {"price_forecasts"}, "forecast")
    try:
        ag.run(con, "CREATE TABLE IF NOT EXISTS price_forecasts (ticker TEXT, made_on TEXT, model TEXT, price REAL, "
                    "p10 REAL, p25 REAL, p50 REAL, p75 REAL, p90 REAL, p_up REAL, horizon INTEGER, "
                    "PRIMARY KEY (ticker, made_on, model))", (), "forecast")
        q = f["quantiles"]
        ag.run(con, "INSERT OR REPLACE INTO price_forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
               (f["ticker"], f["as_of"], f["model"], f["price"], q["p10"], q["p25"], q["p50"], q["p75"], q["p90"],
                f["p_up"], H), "forecast")
        con.commit()
    finally:
        con.close()


def backtest_calibration(tickers=None, start="2019-01-01", step_months=1, max_names=60) -> dict:
    """Forecasts rebuilt at past month-ends with data known then, against what happened."""
    r, _ = _panel()
    if tickers is None:
        tickers = [c for c in r.columns if r[c].notna().sum() > 800][:max_names]
    ends = r[r.index >= start].resample(f"{step_months}ME").last().index
    rows = []
    for e in ends:
        d = r.index[r.index <= e][-1]
        i = r.index.get_loc(d)
        if i + H >= len(r.index):
            break
        for t in tickers:
            f = forecast(t, d)
            if not f or "error" in f:
                continue
            fut = r[t].iloc[i + 1:i + 1 + H]
            if fut.notna().sum() < H * 0.6:
                continue
            realised = f["price"] * float(np.prod(1 + fut.fillna(0)))
            q = f["quantiles"]
            rows.append({"inside_80": q["p10"] <= realised <= q["p90"], "inside_50": q["p25"] <= realised <= q["p75"],
                         "up": realised > f["price"], "p_up": f["p_up"],
                         "dir_hit": (realised > f["price"]) == (f["p_up"] > 0.5)})
    d = pd.DataFrame(rows)
    if d.empty:
        return {"n": 0}
    return {"n": len(d), "inside_80_band": float(d["inside_80"].mean()), "inside_50_band": float(d["inside_50"].mean()),
            "share_up": float(d["up"].mean()), "direction_hit": float(d["dir_hit"].mean()),
            "brier_p_up": float(((d["p_up"] - d["up"].astype(float)) ** 2).mean())}


def score() -> dict:
    """Matured live forecasts: did the price land in the bands?"""
    r, _ = _panel()
    with get_connection() as conn:
        try:
            f = pd.read_sql("SELECT * FROM price_forecasts", conn, parse_dates=["made_on"])
        except Exception:  # noqa: BLE001
            return {"n": 0}
    rows = []
    for x in f.itertuples():
        i0 = r.index.searchsorted(x.made_on, side="right")
        if i0 + H > len(r.index) or x.ticker not in r:
            continue
        realised = x.price * float(np.prod(1 + r[x.ticker].iloc[i0:i0 + H].fillna(0)))
        rows.append({"inside_80": x.p10 <= realised <= x.p90, "inside_50": x.p25 <= realised <= x.p75})
    if not rows:
        return {"n": 0, "pending": len(f)}
    d = pd.DataFrame(rows)
    return {"n": len(d), "inside_80_band": float(d["inside_80"].mean()), "inside_50_band": float(d["inside_50"].mean())}


def format_forecast(f: dict) -> str:
    if not f:
        return "no data"
    if "error" in f:
        return f"{f['ticker']}: {f['error']}"
    q = f["quantiles"]
    L = [f"{f['ticker']} — 4-week range from {f['price']:,.2f} ({f['price_date']}), model {f['model']}",
         f"   80% band: {q['p10']:,.2f} to {q['p90']:,.2f}  ({q['p10'] / f['price'] - 1:+.0%} / {q['p90'] / f['price'] - 1:+.0%})",
         f"   50% band: {q['p25']:,.2f} to {q['p75']:,.2f}   middle {q['p50']:,.2f}",
         f"   odds: up {f['p_up']:.0%} | up 10%+ {f['p_up10']:.0%} | down 10%+ {f['p_down10']:.0%}",
         f"   why this wide: volatility {f['month_vol']:.1%} a month, beta {f['beta']:.2f} to the EGX30, "
         f"shape from {f['shape']}"]
    for e in f["events"]:
        L.append(f"   news/context: {e}")
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    import logging
    logging.disable(logging.WARNING)
    if "--calibrate" in argv:
        print(json.dumps(backtest_calibration(), indent=1))
    elif "--score" in argv:
        print(score())
    else:
        for t in argv:
            print(format_forecast(forecast(t.upper(), record_it=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
