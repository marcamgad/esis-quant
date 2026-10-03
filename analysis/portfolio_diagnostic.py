"""
Portfolio diagnostic — what you own and what it can do to you.

WHAT THIS IS AND IS NOT. It is the analytical half of what a professional
adviser does: measure the book, find the concentrations, price the scenarios,
show what the deviations cost. It is NOT advice. Nothing here recommends buying
or selling anything, and it deliberately produces no "score" or verdict that
could be read as one.

The distinction matters here more than usual. Five audits established this
system has no measured return edge (alpha -2.46%, t = -0.61; loses to T-bills in
real terms). A module that produced recommendations on top of that would be
dressing a null result as guidance. Diagnosis does not depend on having an edge:
"OIH is 35% of your book and contributes 61% of its variance" is true and useful
whether or not anyone can predict OIH.

WHAT IT MEASURES

  1. CONCENTRATION. Weight is the obvious view and the misleading one. What
     matters is RISK contribution: a 35% weight in a volatile, uncorrelated name
     can be 60%+ of total variance. Reported as marginal and total contribution,
     plus the effective number of bets (1/HHI on risk).

  2. SINGLE-NAME SHOCK. What a -30% or -50% move in each holding does to the
     whole book, in EGP. This is the number people actually think in, and it is
     never shown by a weights table.

  3. SCENARIOS on YOUR book, not the backtest. A broad market fall, a currency
     devaluation (using the two real EGX devaluation days), and the worst
     historical 21-day stretch this book's constituents actually lived through.

  4. CORRELATION STRUCTURE. Eight holdings can be three bets. Reported as the
     average pairwise correlation and the variance explained by the first
     principal component -- if PC1 explains 70%+, the book is one bet wearing
     eight names.

  5. DEVIATION COST. The book you hold vs the book the model targets, priced in
     volatility. This is the only place the model's opinion appears, and it is
     expressed as a cost, not a recommendation.

EVERY NUMBER IS BACKWARD-LOOKING. Covariance is estimated from trailing returns;
a book that has not been stressed will look safe. That is a property of the
estimator, not of the book, and is stated in the output rather than left for the
reader to infer.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.portfolio_diagnostic")

TRADING_DAYS = 252
LOOKBACK = 180

# Above this share of book value being unmeasurable, the caveat is promoted from
# a footnote to the headline. Set at 10% because that is roughly the point where
# the portfolio-level volatility stops being a description of the portfolio: a
# 19.2% unmeasured slice (CRST, 2026-09-04) can move the true figure by more
# than the difference between any two allocations the system would propose.
UNMEASURED_ALARM = 0.10


def _unpriced_weight(unpriced, priced_value: float) -> float:
    """Share of TOTAL book value that carries no usable price history.

    Deliberately measured against priced + unpriced value rather than against
    the priced book alone: the question being answered is "how much of my money
    is unmeasured", not "how much of the measured book is unmeasured".
    """
    if not unpriced:
        return 0.0
    from ops.portfolio_tracker import holdings
    try:
        pos, _ = holdings()
    except Exception:  # noqa: BLE001
        return float("nan")
    miss = float(pos.loc[pos["ticker"].isin(unpriced), "value"].sum())
    total = priced_value + miss
    return miss / total if total > 0 else 0.0
SHOCKS = (-0.30, -0.50)
# The two real EGP devaluation sessions in the sample. Used as scenario inputs
# rather than a hypothetical percentage, because they actually happened.
DEVALUATION_DAYS = ("2016-11-06", "2024-03-07")


def _book(as_of=None):
    """Current holdings with weights, plus the return panel they live in."""
    from ops import portfolio_tracker as pt
    from backtest.engine import load_price_returns
    pos, cash = pt.holdings()
    if pos.empty:
        return None, None, 0.0, 0.0
    rets = load_price_returns()
    if as_of is not None:
        rets = rets.loc[:pd.Timestamp(as_of)]
    held = pos.set_index("ticker")
    # a holding with no price feed cannot be measured; excluded from the risk
    # maths and reported separately rather than silently dropped
    priced = [t for t in held.index if t in rets.columns and rets[t].tail(LOOKBACK).notna().sum() > LOOKBACK // 2]
    unpriced = [t for t in held.index if t not in priced]
    val = held.loc[priced, "value"].astype(float)
    return val, rets[priced].tail(LOOKBACK).fillna(0.0), float(cash), unpriced


def risk_contribution(values: pd.Series, win: pd.DataFrame) -> pd.DataFrame:
    """Weight vs RISK contribution. The gap between them is the point."""
    w = values / values.sum()
    cov = win[w.index].cov() * TRADING_DAYS
    port_var = float(w.values @ cov.values @ w.values)
    port_vol = np.sqrt(port_var)
    marginal = cov.values @ w.values                 # d(var)/d(w_i)
    contrib = w.values * marginal / port_var          # fraction of total variance
    return pd.DataFrame({
        "value_egp": values, "weight": w,
        "own_vol": np.sqrt(np.diag(cov.values)),
        "risk_contrib": contrib,
        "risk_vs_weight": contrib / w.values,
    }).sort_values("risk_contrib", ascending=False), port_vol


def single_name_shocks(values: pd.Series, shocks=SHOCKS) -> pd.DataFrame:
    """What each holding falling by X does to the whole book, in EGP."""
    total = float(values.sum())
    rows = []
    for t, v in values.items():
        row = {"ticker": t, "value_egp": float(v)}
        for s in shocks:
            row[f"book_{int(s*100)}%"] = float(v * s)
            row[f"book_{int(s*100)}%_pct"] = float(v * s / total)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"book_{int(shocks[0]*100)}%")


def scenarios(values: pd.Series, win: pd.DataFrame) -> dict:
    """Market fall, real devaluation days, and the worst 21-day stretch these
    specific names actually lived through."""
    from backtest.engine import load_price_returns
    w = values / values.sum()
    total = float(values.sum())
    full = load_price_returns()
    out = {}

    # broad market fall, scaled by each name's beta to the equal-weight market
    mkt = full.mean(axis=1).tail(LOOKBACK * 3)
    betas = {}
    for t in w.index:
        r = full[t].reindex(mkt.index)
        d = pd.concat([r, mkt], axis=1).dropna()
        betas[t] = float(np.cov(d.iloc[:, 0], d.iloc[:, 1])[0, 1] / np.var(d.iloc[:, 1])) if len(d) > 30 else 1.0
    beta_book = float(sum(w[t] * betas[t] for t in w.index))
    for m in (-0.10, -0.20, -0.30):
        out[f"market {int(m*100)}%"] = {"pct": m * beta_book, "egp": m * beta_book * total}

    # the two REAL devaluation sessions
    for d in DEVALUATION_DAYS:
        ts = pd.Timestamp(d)
        if ts in full.index:
            day = full.loc[ts]
            hit = float(sum(w[t] * day.get(t, day.median()) for t in w.index))
            out[f"devaluation {d}"] = {"pct": hit, "egp": hit * total}

    # worst realised 21-day stretch for THIS combination of names
    hist = full[list(w.index)].dropna(how="all").fillna(0.0)
    port = (hist * w).sum(axis=1)
    roll = (1 + port).rolling(21).apply(np.prod, raw=True) - 1
    if roll.notna().any():
        out["worst realised 21d"] = {"pct": float(roll.min()), "egp": float(roll.min()) * total,
                                     "when": str(roll.idxmin().date())}
    out["_beta"] = beta_book
    return out


def correlation_structure(values: pd.Series, win: pd.DataFrame) -> dict:
    """Are these N bets, or one bet wearing N names?"""
    w = values / values.sum()
    c = win[w.index].corr()
    off = c.values[np.triu_indices_from(c.values, k=1)]
    cov = win[w.index].cov()
    eig = np.linalg.eigvalsh(cov.values)[::-1]
    pc1 = float(eig[0] / eig.sum()) if eig.sum() > 0 else np.nan
    contrib, _ = risk_contribution(values, win)
    hhi = float((contrib["risk_contrib"] ** 2).sum())
    return {"avg_pairwise_corr": float(np.mean(off)), "max_pairwise_corr": float(np.max(off)),
            "pc1_share": pc1, "effective_bets": (1.0 / hhi) if hhi > 0 else np.nan,
            "n_holdings": int(len(w))}


def deviation_cost(values: pd.Series, win: pd.DataFrame) -> dict:
    """What holding YOUR book instead of the model's costs, in volatility.

    Stated as a cost, never as an instruction. The model has no measured return
    edge, so this compares RISK only -- claiming a return difference would be
    claiming an edge the audits say is not there.
    """
    try:
        from ops.trade_list import latest_trade_list
        tl = latest_trade_list()
        tgt = tl[tl["ticker"] != "__CASH__"].set_index("ticker")["weight"]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)}
    common = [t for t in tgt.index if t in win.columns]
    if len(common) < 3:
        return {"ok": False, "reason": "target names not priced in the window"}
    cov = win.cov() * TRADING_DAYS

    def vol(wts):
        wts = wts / wts.sum()
        idx = [t for t in wts.index if t in cov.index]
        wts = wts[idx]
        return float(np.sqrt(wts.values @ cov.loc[idx, idx].values @ wts.values))

    yours = values / values.sum()
    model = tgt[common]
    overlap = sorted(set(yours.index) & set(model.index))
    return {"ok": True, "your_vol": vol(yours), "model_vol": vol(model),
            "delta": vol(yours) - vol(model), "overlap": overlap,
            "n_overlap": len(overlap), "n_yours": len(yours), "n_model": len(model)}


def run(as_of=None) -> dict:
    values, win, cash, unpriced = _book(as_of)
    if values is None or values.empty:
        return {"ok": False, "reason": "no priced holdings"}
    contrib, port_vol = risk_contribution(values, win)
    # What share of the book the numbers below do NOT cover. Reported as a
    # number rather than left implicit in a list of tickers, because "CRST is
    # excluded" and "a fifth of your money is unmeasured" are the same fact told
    # at two very different volumes.
    unpriced_weight = _unpriced_weight(unpriced, float(values.sum()))
    return {"ok": True, "cash": cash, "unpriced": unpriced,
            "unpriced_weight": unpriced_weight,
            "total": float(values.sum()) + cash, "invested": float(values.sum()),
            "port_vol": port_vol, "contrib": contrib,
            "shocks": single_name_shocks(values),
            "scenarios": scenarios(values, win),
            "structure": correlation_structure(values, win),
            "deviation": deviation_cost(values, win)}


def format_report(d: dict) -> str:
    if not d.get("ok"):
        return f"PORTFOLIO DIAGNOSTIC — {d.get('reason')}"
    L = ["=" * 70, "PORTFOLIO DIAGNOSTIC — measurement only, nothing here is advice", "=" * 70,
         f"  invested {d['invested']:,.0f} EGP + cash {d['cash']:,.0f} = {d['total']:,.0f}",
         f"  annualised volatility of this book: {d['port_vol']*100:.1f}%",
         f"  one ordinary day: +/-{d['port_vol']/np.sqrt(TRADING_DAYS)*100:.2f}%"]
    if d["unpriced"]:
        # A footnote is the wrong weight of warning once the unmeasured slice is
        # large. On 2026-09-04 CRST was 19.2% of the book with 5 usable returns in
        # 180 sessions, and every figure in this report silently described the
        # other 81% under a two-line note that read like housekeeping. Past the
        # threshold the caveat is promoted above the numbers, because at that
        # point "book volatility 28.5%" is not a fact about the book.
        unp = d.get("unpriced_weight")
        if unp is not None and unp >= UNMEASURED_ALARM:
            L = ["=" * 70,
                 "PORTFOLIO DIAGNOSTIC — *** INCOMPLETE: SEE WARNING FIRST ***",
                 "=" * 70,
                 f"  !! {unp*100:.0f}% OF YOUR BOOK CANNOT BE MEASURED: {', '.join(d['unpriced'])}",
                 "     These holdings have too little price history to carry a volatility, a",
                 "     correlation, or a risk share. They are NOT low-risk -- they are unknown.",
                 f"     Every number below describes only the remaining {(1-unp)*100:.0f}% of the book,",
                 "     so the portfolio-level figures UNDERSTATE what you are actually exposed to.",
                 ""] + L[3:]
        else:
            L.append(f"  !! NOT MEASURABLE (no price feed): {', '.join(d['unpriced'])} — excluded from every")
            L.append("     number below, so the book is larger than what is analysed here.")

    L += ["", "1. CONCENTRATION — weight is not risk",
          f"     {'ticker':8}{'EGP':>9}{'weight':>9}{'own vol':>9}{'risk share':>12}{'risk/weight':>13}"]
    for t, r in d["contrib"].iterrows():
        flag = "  <-- risk >> weight" if r["risk_vs_weight"] > 1.3 else ""
        L.append(f"     {t:8}{r['value_egp']:>9,.0f}{r['weight']*100:>8.1f}%{r['own_vol']*100:>8.1f}%"
                 f"{r['risk_contrib']*100:>11.1f}%{r['risk_vs_weight']:>12.2f}x{flag}")
    s = d["structure"]
    L += ["", f"     {s['n_holdings']} holdings, but effectively {s['effective_bets']:.1f} independent bets",
          f"     average pairwise correlation {s['avg_pairwise_corr']:.2f} (max {s['max_pairwise_corr']:.2f}); "
          f"first component explains {s['pc1_share']*100:.0f}% of variance"]

    L += ["", "2. SINGLE-NAME SHOCK — what one holding falling does to the whole book"]
    sh = d["shocks"]
    L.append(f"     {'ticker':8}{'if -30%':>12}{'of book':>10}{'if -50%':>12}{'of book':>10}")
    for r in sh.itertuples():
        L.append(f"     {r.ticker:8}{getattr(r, '_3'):>11,.0f}{getattr(r, '_4')*100:>9.1f}%"
                 f"{getattr(r, '_5'):>11,.0f}{getattr(r, '_6')*100:>9.1f}%")

    L += ["", "3. SCENARIOS on this book (beta to the EGX equal-weight: "
          f"{d['scenarios'].get('_beta', float('nan')):.2f})"]
    for k, v in d["scenarios"].items():
        if k.startswith("_"):
            continue
        when = f"   ({v['when']})" if "when" in v else ""
        L.append(f"     {k:24} {v['pct']*100:>7.1f}%   {v['egp']:>9,.0f} EGP{when}")

    dev = d["deviation"]
    L += ["", "4. DEVIATION FROM THE MODEL — priced in RISK, not return"]
    if dev.get("ok"):
        L.append(f"     your book vol {dev['your_vol']*100:.1f}%  vs  model book vol {dev['model_vol']*100:.1f}%"
                 f"   -> {dev['delta']*100:+.1f}pp")
        L.append(f"     you hold {dev['n_yours']} names, the model targets {dev['n_model']}, "
                 f"{dev['n_overlap']} in common ({', '.join(dev['overlap']) or 'none'})")
        L.append("     Return is NOT compared: the model has no measured return edge, so a")
        L.append("     return comparison would assert something the evidence does not support.")
    else:
        L.append(f"     unavailable: {dev.get('reason')}")

    L += ["", "CAVEATS", "  Covariance is estimated from trailing returns, so a book that has not",
          "  been stressed looks safe. Correlations rise in crises, which means the",
          "  scenarios above are optimistic exactly when it matters most.",
          "  Nothing here is a recommendation to buy, sell or hold.", "=" * 70]
    return "\n".join(L)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    print(format_report(run()))


def format_compact(d: dict, max_lines: int = 3) -> str:
    """A few lines for the daily report.

    The full report is a page; nobody reads a page every morning. This keeps
    only what changes a decision: where risk is concentrated (as distinct from
    where weight is), how many real bets the book contains, and the largest
    single-name exposure in EGP. Anything that is fine is stated in one line
    rather than itemised -- a diagnostic that prints the same wall of text daily
    trains the reader to skip it.
    """
    if not d.get("ok"):
        return f"    unavailable: {d.get('reason')}"
    L = []
    day = d["port_vol"] / np.sqrt(TRADING_DAYS)
    s = d["structure"]
    L.append(f"    book volatility {d['port_vol']*100:.1f}%/yr — an ordinary day is "
             f"+/-{day*100:.2f}% ({day*d['invested']:,.0f} EGP)")
    L.append(f"    {s['n_holdings']} holdings = {s['effective_bets']:.1f} effective bets "
             f"(avg pairwise correlation {s['avg_pairwise_corr']:.2f})")

    c = d["contrib"]
    heavy = c[c["risk_vs_weight"] > 1.3]
    if not heavy.empty:
        for t, r in heavy.head(max_lines).iterrows():
            L.append(f"    {t}: {r['weight']*100:.0f}% of the money but "
                     f"{r['risk_contrib']*100:.0f}% of the RISK ({r['risk_vs_weight']:.2f}x)")
    else:
        L.append("    no holding carries materially more risk than its weight.")

    top = d["shocks"].iloc[0]
    L.append(f"    worst single-name shock: {top['ticker']} -50% costs "
             f"{abs(top[[c for c in d['shocks'].columns if c.endswith('-50%')][0]]):,.0f} EGP "
             f"({abs(top[[c for c in d['shocks'].columns if c.endswith('-50%_pct')][0]])*100:.1f}% of the book)")

    w21 = d["scenarios"].get("worst realised 21d")
    if w21:
        L.append(f"    worst 21d these names actually had: {w21['pct']*100:.1f}% "
                 f"({w21['egp']:,.0f} EGP, {w21['when']})")
    if d["unpriced"]:
        L.append(f"    NOT MEASURED (no price feed): {', '.join(d['unpriced'])}")
    L.append("    measurement only — nothing here is a recommendation.")
    return "\n".join(L)
