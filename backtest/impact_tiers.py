"""
Phase 3 -- liquidity-TIERED market-impact model, fitted from observable data.

THE PROBLEM. backtest/costs.py applies ONE impact law to every name:
slippage = base + COST_IMPACT_COEF * sqrt(participation), with COST_IMPACT_COEF
a literature default, identical for the most and least liquid ticker in the
universe. On EGX that is wrong in a specific direction: daily value traded spans
roughly three orders of magnitude here (CCAP ~39M EGP vs SCFM ~37k), so a single
coefficient systematically UNDER-costs the illiquid tail -- and the illiquid tail
is exactly where a min-variance optimizer likes to hide, because thin names look
low-covariance. That inflates backtested performance on the names least likely
to survive contact with a real order book.

WHAT WE CAN AND CANNOT FIT. The build plan asks for k and alpha estimated from
"realized spread/impact data". We have NO broker fill data -- Phase 0.2 is
unbuilt, so realized slippage per order does not exist. Fitting from our own
fills is therefore impossible today, and inventing it would be fabrication.

What DOES exist is market-observable impact: the AMIHUD illiquidity ratio,

    amihud_t = |return_t| / value_traded_t

which measures how far price moves per EGP transacted. It is the standard
observable proxy for impact and is computable from daily_ohlcv alone. Fitting a
power law to the relationship between traded value and absolute return gives a
per-tier impact curve grounded in real market behaviour rather than in a
literature constant.

THE MODEL. Within each liquidity tier, regress in logs:

    log|r_t| = log k + alpha * log( value_traded_t / ADV )  + e

so impact(participation) = k * participation^alpha, with k and alpha estimated
per tier. Estimation is ordinary least squares on the log-log relation, done
from scratch with numpy.

HONEST LIMITS -- these must travel with any number this module produces:
  * Amihud measures the impact of the WHOLE market's volume, not of one
    participant's order. It is a proxy, and it is generally a CONSERVATIVE one
    for small orders and an optimistic one for large ones.
  * Daily data cannot separate impact from information; a big move on big
    volume may be news, not price pressure.
  * Fitted alpha is therefore a description of the observed price/volume
    relation, not a causal execution cost.
  * The right replacement is realized fill data. Until Phase 0.2 exists, this
    is the best evidence available, and it is strictly better than one constant
    for a universe spanning 1000x in liquidity.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.impact_tiers")

N_TIERS = 4                 # 47 names -> ~12 per tier; more tiers would be noise
MIN_OBS_PER_TIER = 200
ADV_LOOKBACK = 60
DEFAULT_ALPHA = 0.5         # sqrt law, the fallback when a tier cannot be fitted


def _load_panel() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql("SELECT ticker, date, adjusted_close, value_traded_egp "
                         "FROM daily_ohlcv WHERE value_traded_egp > 0 AND adjusted_close > 0 "
                         "ORDER BY ticker, date", conn, parse_dates=["date"])
    if df.empty:
        return df
    df["ret"] = df.groupby("ticker")["adjusted_close"].pct_change()
    df["adv"] = (df.groupby("ticker")["value_traded_egp"]
                   .transform(lambda s: s.rolling(ADV_LOOKBACK, min_periods=20).mean()))
    df["participation"] = df["value_traded_egp"] / df["adv"]
    return df.dropna(subset=["ret", "adv", "participation"])


def assign_tiers(n_tiers: int = N_TIERS) -> pd.Series:
    """Bucket tickers into liquidity tiers by median daily value traded.
    Tier 0 = most liquid."""
    with get_connection() as conn:
        med = pd.read_sql("SELECT ticker, value_traded_egp FROM daily_ohlcv "
                          "WHERE value_traded_egp > 0", conn).groupby("ticker")["value_traded_egp"].median()
    if med.empty:
        return pd.Series(dtype=int)
    ranks = med.rank(ascending=False, method="first")
    tiers = np.ceil(ranks / (len(med) / n_tiers)).astype(int) - 1
    return tiers.clip(0, n_tiers - 1).rename("tier")


def fit_tier_impact(n_tiers: int = N_TIERS) -> pd.DataFrame:
    """Estimate k and alpha per tier from the log-log |return| vs participation
    relation. Tiers with too few observations fall back to the sqrt law."""
    panel = _load_panel()
    if panel.empty:
        return pd.DataFrame()
    tiers = assign_tiers(n_tiers)
    panel["tier"] = panel["ticker"].map(tiers)
    rows = []
    for tier, g in panel.groupby("tier"):
        g = g[(g["participation"] > 0) & (g["ret"].abs() > 0)]
        if len(g) < MIN_OBS_PER_TIER:
            rows.append({"tier": int(tier), "n_obs": len(g), "k": np.nan,
                         "alpha": DEFAULT_ALPHA, "fitted": False,
                         "median_adv": float(g["adv"].median()) if len(g) else np.nan,
                         "n_tickers": int(g["ticker"].nunique())})
            continue
        x = np.log(g["participation"].to_numpy(float))
        y = np.log(g["ret"].abs().to_numpy(float))
        X = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else np.nan
        rows.append({"tier": int(tier), "n_obs": int(len(g)), "k": float(np.exp(coef[0])),
                     "alpha": float(coef[1]), "fitted": True, "r2": float(r2),
                     "median_adv": float(g["adv"].median()),
                     "n_tickers": int(g["ticker"].nunique())})
    return pd.DataFrame(rows).sort_values("tier").reset_index(drop=True)


def tiered_impact(participation: float, tier: int, fits: pd.DataFrame) -> float:
    """Impact as a fraction of notional for a given participation rate."""
    row = fits[fits["tier"] == tier]
    if row.empty or not bool(row["fitted"].iloc[0]):
        return config.COST_IMPACT_COEF * np.sqrt(max(participation, 0.0))
    k = float(row["k"].iloc[0])
    a = float(row["alpha"].iloc[0])
    return float(k * max(participation, 1e-12) ** a)


def compare_to_flat(fits: pd.DataFrame, participations=(0.01, 0.05, 0.20)) -> pd.DataFrame:
    """Tiered vs the single-coefficient model, at representative order sizes.
    Positive `diff` = the flat model was UNDER-costing that tier."""
    rows = []
    for tier in sorted(fits["tier"].unique()):
        for p in participations:
            t_imp = tiered_impact(p, tier, fits)
            f_imp = config.COST_IMPACT_COEF * np.sqrt(p)
            rows.append({"tier": int(tier), "participation": p, "tiered": t_imp,
                         "flat": f_imp, "diff": t_imp - f_imp,
                         "ratio": (t_imp / f_imp) if f_imp > 0 else np.nan})
    return pd.DataFrame(rows)


def format_fits(fits: pd.DataFrame) -> str:
    if fits.empty:
        return "impact tiers: no data."
    lines = ["LIQUIDITY-TIERED IMPACT (fitted from Amihud-style log-log price/volume)",
             f"  {'tier':>5}{'names':>7}{'obs':>8}{'median ADV':>14}{'k':>9}{'alpha':>8}{'R2':>7}"]
    for r in fits.itertuples():
        k = f"{r.k:.4f}" if r.fitted else "   n/a"
        r2 = f"{getattr(r, 'r2', float('nan')):.3f}" if r.fitted else "  n/a"
        lines.append(f"  {r.tier:>5}{r.n_tickers:>7}{r.n_obs:>8}{r.median_adv:>14,.0f}"
                     f"{k:>9}{r.alpha:>8.3f}{r2:>7}")
    lines.append("  tier 0 = most liquid.  alpha < 1 means concave (sqrt-like) impact.")
    lines.append("  PROXY WARNING: Amihud measures the market's volume, not your order.")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    f = fit_tier_impact()
    print(format_fits(f))
    print()
    c = compare_to_flat(f)
    print("TIERED vs FLAT impact (ratio > 1 = flat model was under-costing):")
    print(f"  {'tier':>5}{'partic':>9}{'tiered':>10}{'flat':>10}{'ratio':>8}")
    for r in c.itertuples():
        print(f"  {r.tier:>5}{r.participation:>9.2f}{r.tiered:>10.4f}{r.flat:>10.4f}{r.ratio:>8.2f}")
