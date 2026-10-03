"""
Part 3 -- the business case: does this beat doing nothing?

THE QUESTION THAT HAS BEEN OPEN SINCE THE FIRST REPORT. Every performance number
this project has produced answers "did the strategy make money in EGP?" That is
the wrong question in an economy where the currency lost most of its value over
the sample and where a completely passive alternative -- Egyptian T-bills --
paid a high nominal yield for zero work and zero equity risk. The only question
that matters to someone deciding whether to run this at all is:

    In REAL terms, net of costs, after accounting for the risk actually taken,
    does this beat simply holding T-bills?

So the ordering here is deliberately reversed from every previous report: REAL
(USD-deflated) figures are the headline, nominal EGP is the footnote.

WHY USD-DEFLATED. Egyptian CPI would be the textbook deflator, but there is no
clean daily CPI series in this database. USD/EGP is available daily and, for
EGP, the devaluation IS the dominant real-terms driver -- the currency move and
the inflation that drove it are two faces of one event. It is a first-order
correction and a conservative one. It is NOT a precise CPI deflation and is
never described as one.

THE BENCHMARK. Egyptian T-bill yields are not in this database and are not
freely available as a daily series. data/rates.py already reconstructs the CBE
policy rate from published MPC decisions; T-bill yields historically track it
closely (usually a little above). That reconstruction is APPROXIMATE and
unverified, so the comparison is run across a SWEEP of assumed yields as well as
the reconstruction -- if the verdict is the same at every plausible yield, it
does not depend on the approximation being exactly right.

RISK-ADJUSTMENT. Comparing raw returns to T-bills would flatter the strategy,
which takes equity risk while T-bills take (almost) none. So the comparison
reports excess return over the bill AND drawdown, because a strategy that beats
bills by two points while suffering a 30% drawdown has not obviously won.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.business_case")

TRADING_DAYS = 252
# Egyptian T-bills price off the CBE corridor; auction yields have historically
# sat modestly ABOVE the overnight deposit rate. This premium is an assumption,
# not a measurement, and is swept over below.
TBILL_OVER_POLICY = 0.01
YIELD_SWEEP = (0.08, 0.12, 0.16, 0.20, 0.24)


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    nav = (1 + r).cumprod()
    sd = r.std(ddof=1)
    return {"cagr": float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1),
            "vol": float(sd * np.sqrt(TRADING_DAYS)),
            "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "max_dd": float((nav / nav.cummax() - 1).min()),
            "total": float((1 + r).prod() - 1), "n_days": int(len(r))}


def tbill_daily(index: pd.DatetimeIndex, flat_yield: float = None,
                premium: float = TBILL_OVER_POLICY) -> pd.Series:
    """Daily return of rolling T-bills. Uses the reconstructed CBE policy rate
    plus a premium unless a flat yield is supplied."""
    from data.rates import policy_rate_series
    if flat_yield is not None:
        ann = pd.Series(float(flat_yield), index=index)
    else:
        ann = policy_rate_series(index) + premium
    return (1.0 + ann) ** (1.0 / TRADING_DAYS) - 1.0


def compare(strategy_returns: pd.Series, flat_yield: float = None) -> dict:
    """Strategy vs T-bills, in BOTH nominal EGP and real USD terms.

    The T-bill leg is deflated too -- an EGP bill earning 20% while the pound
    falls 20% has earned nothing real, and comparing a real strategy return to a
    nominal bill return would be the single most misleading thing this module
    could do.
    """
    from analysis import real_returns as rr
    s = strategy_returns.dropna()
    if s.empty:
        return {}
    bill = tbill_daily(s.index, flat_yield=flat_yield)
    s_usd = rr.to_hard_currency(s)
    b_usd = rr.to_hard_currency(bill.reindex(s.index).fillna(0.0))

    out = {"strategy_nominal": _stats(s), "tbill_nominal": _stats(bill),
           "strategy_real": _stats(s_usd), "tbill_real": _stats(b_usd),
           "mean_yield": float(((1 + bill) ** TRADING_DAYS - 1).mean())}
    # excess of strategy over the bill, computed on the DIFFERENCE series so the
    # comparison is like-for-like day by day rather than a difference of CAGRs
    ex_nom = (s - bill.reindex(s.index).fillna(0.0)).dropna()
    ex_real = (s_usd - b_usd.reindex(s_usd.index).fillna(0.0)).dropna()
    out["excess_nominal"] = _stats(ex_nom)
    out["excess_real"] = _stats(ex_real)
    out["beats_tbill_real"] = bool(out["strategy_real"]["cagr"] > out["tbill_real"]["cagr"])
    return out


def yield_sweep(strategy_returns: pd.Series, yields=YIELD_SWEEP) -> pd.DataFrame:
    """Does the verdict survive every plausible T-bill yield? If it flips inside
    the sweep, the answer depends on an assumption we cannot verify."""
    rows = []
    for y in yields:
        c = compare(strategy_returns, flat_yield=y)
        if not c:
            continue
        rows.append({"assumed_yield": y,
                     "strategy_real_cagr": c["strategy_real"]["cagr"],
                     "tbill_real_cagr": c["tbill_real"]["cagr"],
                     "excess_real_cagr": c["strategy_real"]["cagr"] - c["tbill_real"]["cagr"],
                     "beats": c["beats_tbill_real"]})
    return pd.DataFrame(rows)


def format_comparison(c: dict, sweep: pd.DataFrame = None) -> str:
    if not c:
        return "business case: no returns supplied."
    sr, tr = c["strategy_real"], c["tbill_real"]
    sn, tn = c["strategy_nominal"], c["tbill_nominal"]
    lines = [
        "BUSINESS CASE — does this beat holding T-bills?",
        "  REAL (USD-deflated) is the headline. Nominal EGP is the footnote.",
        "",
        f"  {'':22}{'CAGR':>9}{'vol':>9}{'Sharpe':>9}{'MaxDD':>9}",
        f"  {'STRATEGY (real USD)':22}{sr['cagr']*100:>8.1f}%{sr['vol']*100:>8.1f}%"
        f"{sr['sharpe']:>9.2f}{sr['max_dd']*100:>8.1f}%",
        f"  {'T-BILLS (real USD)':22}{tr['cagr']*100:>8.1f}%{tr['vol']*100:>8.1f}%"
        f"{tr['sharpe']:>9.2f}{tr['max_dd']*100:>8.1f}%",
        f"  {'-> excess (real)':22}{(sr['cagr']-tr['cagr'])*100:>8.1f}%",
        "",
        f"  {'strategy (nominal EGP)':22}{sn['cagr']*100:>8.1f}%{sn['vol']*100:>8.1f}%"
        f"{sn['sharpe']:>9.2f}{sn['max_dd']*100:>8.1f}%",
        f"  {'T-bills (nominal EGP)':22}{tn['cagr']*100:>8.1f}%{tn['vol']*100:>8.1f}%"
        f"{tn['sharpe']:>9.2f}{tn['max_dd']*100:>8.1f}%",
        f"  (mean assumed bill yield {c['mean_yield']*100:.1f}%)",
    ]
    if sweep is not None and not sweep.empty:
        lines += ["", "  ROBUSTNESS — does the verdict depend on the assumed yield?",
                  f"  {'yield':>7}{'strategy real':>15}{'tbill real':>12}{'excess':>9}{'beats?':>8}"]
        for r in sweep.itertuples():
            lines.append(f"  {r.assumed_yield*100:>6.0f}%{r.strategy_real_cagr*100:>14.1f}%"
                         f"{r.tbill_real_cagr*100:>11.1f}%{r.excess_real_cagr*100:>8.1f}%"
                         f"{('YES' if r.beats else 'NO'):>8}")
        agree = sweep["beats"].all() or (~sweep["beats"]).all()
        lines.append(f"  -> verdict is {'STABLE' if agree else 'SENSITIVE'} across the sweep")
    lines += ["",
              "  T-bill yields are approximated from the RECONSTRUCTED CBE policy rate",
              "  (data/rates.py) plus a premium -- not a fetched auction series.",
              "  Real terms use USD deflation, not CPI. Both are stated approximations."]
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from portfolio.system import run_full_system
    res = run_full_system()
    strat = res["returns"]["full_system"]
    c = compare(strat)
    print(format_comparison(c, yield_sweep(strat)))
