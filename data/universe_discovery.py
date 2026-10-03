"""
Universe discovery: which EGX tickers actually have usable Yahoo history?

WHY. The system trades 27 names. That number is the binding constraint on the
whole project: with 27 stocks you cannot build meaningful factor portfolios
(a quintile sort gives ~5 names per bucket -- noise, not a factor), every
apparent edge is hard to separate from luck, and the min-variance optimizer has
almost nothing to choose between. Expanding the universe does not create alpha
by itself; it creates the STATISTICAL POWER needed to tell whether any alpha is
there, and gives the optimizer more names to diversify across.

WHAT THIS DOES. Takes a candidate list of EGX tickers, asks Yahoo for each one's
history, and classifies it honestly:
  * OK        -- enough history and enough recent activity to be tradable
  * THIN      -- resolves, but too illiquid / too many non-trading days to trust
  * SHORT     -- resolves, but not enough history for a 180-day lookback + warmup
  * NO_DATA   -- Yahoo has nothing under this symbol

Nothing is added to the live universe automatically. This produces a REPORT;
promoting names is a separate, deliberate step, because a thin name that passes
a coverage check can still be untradable at size (see backtest/costs.py) and
because widening the universe changes every historical number in the project.

Survivorship bias warning: this checks symbols that exist TODAY. Names that
delisted or collapsed over 2015-2026 will not appear, so a universe built this
way remains survivorship-biased -- it is wider, not unbiased.
"""

import logging
import time

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("esis_quant.universe_discovery")

MIN_HISTORY_DAYS = 400        # need 180-day lookback + warm-up + margin
MIN_NONZERO_FRAC = 0.60       # fraction of days that must show actual trading
MIN_RECENT_DAYS = 20          # must have printed recently (still listed)

# Candidate EGX symbols beyond the current seed list. These are BEST-EFFORT
# guesses at Yahoo's ".CA" symbology for EGX-listed companies -- every one is
# verified against real data by check_ticker() before it means anything.
CANDIDATE_TICKERS = [
    # banks / financials
    "CIEB", "HDBK", "EXPA", "SAUD", "FAIT", "BTFH", "PRMH", "AIVC", "CANA",
    # real estate / construction
    "HELI", "OCDI", "EMFD", "AMER", "PORT", "UEGC", "MENA", "ODHN", "ZMID",
    "ARCC", "SUEZ", "TORA", "MCQE", "SCEM", "LCSW", "RUBX", "ACGC",
    # industrials / materials
    "ETRS", "IRON", "KABO", "ATQA", "DSCW", "SPMD", "NCCW", "GGCC", "ECAP",
    "ALCN", "MOIL", "SPIN", "KZPC", "ISMQ", "OIH", "MTIE",
    # consumer / food / pharma
    "DOMT", "PHAR", "EIUD", "TALM", "MPCI", "AJWA", "OCPH", "MICH", "NEDA",
    # autos / transport / tech
    "AUTO", "GBCO", "RAYA", "CCAP", "ADPC", "SCFM", "ELEC", "OFH", "SKPC",
    # previously tried and failed -- re-checked so the record stays current
    "ESRS", "MNHD", "PALTIN",
]


def check_ticker(ticker: str, suffix: str = None, period: str = "10y") -> dict:
    """Fetch one symbol and classify its usability. Never raises."""
    import yfinance as yf
    suffix = config.YFINANCE_SUFFIX if suffix is None else suffix
    sym = f"{ticker}{suffix}"
    out = {"ticker": ticker, "symbol": sym, "status": "NO_DATA", "n_days": 0,
           "first": None, "last": None, "nonzero_frac": 0.0, "median_value_egp": 0.0,
           "days_since_last": None}
    try:
        h = yf.Ticker(sym).history(period=period, auto_adjust=False)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "ERROR"
        out["note"] = str(exc)[:80]
        return out
    if h is None or h.empty:
        return out

    h = h.dropna(subset=["Close"])
    out["n_days"] = int(len(h))
    if len(h) == 0:
        return out
    out["first"] = str(h.index[0].date())
    out["last"] = str(h.index[-1].date())
    out["days_since_last"] = int((pd.Timestamp.now(tz=h.index.tz) - h.index[-1]).days)

    vol = h["Volume"].fillna(0.0)
    out["nonzero_frac"] = float((vol > 0).mean())
    out["median_value_egp"] = float((h["Close"] * vol).replace(0, np.nan).median() or 0.0)

    if out["n_days"] < MIN_HISTORY_DAYS:
        out["status"] = "SHORT"
    elif out["days_since_last"] is not None and out["days_since_last"] > 30:
        out["status"] = "STALE"          # stopped printing -> delisted/suspended
    elif out["nonzero_frac"] < MIN_NONZERO_FRAC:
        out["status"] = "THIN"
    else:
        out["status"] = "OK"
    return out


def discover(tickers=None, delay: float = 0.4, include_current: bool = False) -> pd.DataFrame:
    """Check a list of candidates. Returns one row per ticker, best first."""
    current = {t["ticker"] for t in config.EGX_SEED_TICKERS}
    cands = list(tickers if tickers is not None else CANDIDATE_TICKERS)
    if include_current:
        cands = sorted(set(cands) | current)
    else:
        cands = [t for t in dict.fromkeys(cands) if t not in current]

    rows = []
    for i, t in enumerate(cands, 1):
        r = check_ticker(t)
        rows.append(r)
        logger.info("discover %d/%d %s -> %s (%d days)", i, len(cands), t,
                    r["status"], r["n_days"])
        time.sleep(delay)          # be polite to a free, rate-limited source
    df = pd.DataFrame(rows)
    order = {"OK": 0, "THIN": 1, "SHORT": 2, "STALE": 3, "NO_DATA": 4, "ERROR": 5}
    return df.sort_values(["status", "median_value_egp"],
                          key=lambda s: s.map(order) if s.name == "status" else -s,
                          ).reset_index(drop=True)


def format_discovery(df: pd.DataFrame) -> str:
    if df.empty:
        return "universe discovery: nothing checked."
    counts = df["status"].value_counts().to_dict()
    lines = [f"UNIVERSE DISCOVERY — {len(df)} candidate(s) checked",
             f"  {counts}",
             f"  {'ticker':8}{'status':9}{'days':>6}{'trade%':>8}{'med value EGP':>15}  span"]
    for r in df.itertuples():
        if r.status in ("NO_DATA", "ERROR"):
            continue
        lines.append(f"  {r.ticker:8}{r.status:9}{r.n_days:>6}{r.nonzero_frac*100:>7.0f}%"
                     f"{r.median_value_egp:>15,.0f}  {r.first} -> {r.last}")
    ok = df[df["status"] == "OK"]
    lines.append(f"  USABLE NOW: {len(ok)} -> {', '.join(ok['ticker'].tolist()) if len(ok) else 'none'}")
    lines.append("  NOTE: survivorship-biased (only symbols that exist today were checked).")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = discover()
    df.to_csv("universe_discovery.csv", index=False)
    print()
    print(format_discovery(df))
    print("\nfull results written to universe_discovery.csv")
