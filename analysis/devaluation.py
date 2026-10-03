"""
Devaluation early-warning research  --  RESULT: NULL. Not used in the live path.

WHY THIS WAS ATTEMPTED. Two days destroyed half of this system's real return:
2016-11-06 (-39.2%) and 2024-03-07 (-36.5%), both EGP devaluations. The regime
overlay -- the system's only measured edge -- was 100% INVESTED through both,
because it reads EGX prices and a devaluation is a currency event (EGX often
RISES in EGP while USD wealth collapses). So the single highest-value question
in the project was: can a devaluation be seen coming?

WHAT WAS TESTED (four pre-specified hypotheses, all reported, none hidden):
  H1  flat peg + long duration  -- a managed peg looks calm right before it
      breaks, so near-zero FX vol plus a long time since the last jump should
      mean accumulating tension.
  H2  EGX in USD momentum       -- foreign/dollar demand for real assets.
  H3  EGX in EGP momentum       -- locals hedging inflation by buying equities.
  H4  EGX volatility            -- generalised market stress.

WHAT WAS FOUND, against a base rate of P(devaluation within 60 days) = 8.6%
over 6 distinct episodes:
  H1  P = 3.8%   lift 0.44x   recall 2/6   -> WORSE than random. Refuted.
  H2  P = 8.9%   lift 1.03x   recall 1/6   -> no information.
  H3  P = 15.5%  lift 1.80x   recall 2/6   -> looked promising; see below.
  H4  P = 8.1%   lift 0.94x   recall 2/6   -> no information.

H3 was then permutation-tested by re-drawing random event dates 5,000 times:
the null distribution's 90th percentile is 1.77x, so an observed 1.80x is
ordinary. p = 0.096 alone, and p = 0.332 after correcting for the four
hypotheses tried. **Not significant. Consistent with chance.**

CONCLUSION, STATED PLAINLY. With this data there is no usable devaluation
early-warning signal. Six events over eleven years is far too few to fit a
predictor to: any pattern found by searching harder would be curve-fitting, and
this project has already been burned by exactly that (the circuit breaker that
turned +824% into -90%). The honest implication is that devaluation risk here
cannot be TIMED -- it can only be MANAGED structurally (size, or hold assets
outside EGP), and the system should not pretend otherwise.

WHAT WOULD CHANGE THE ANSWER: actual parallel-market ("black market") EGP
quotes, Egypt sovereign CDS spreads, or reserve/import-cover data -- none of
which are in this database and none available free. Those are the series that
practitioners actually watch, and their absence is the real constraint, not the
choice of statistic.

Nothing here is imported by the live path. It exists so the null result is
reproducible rather than a claim in a chat log.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.devaluation")

JUMP_THRESHOLD = 0.05        # a one-day official-rate move this large = devaluation
EPISODE_GAP_DAYS = 30        # jumps within this window are the same episode
HORIZON_DAYS = 60            # "did a devaluation follow within N days"


def load_fx() -> pd.Series:
    with get_connection() as conn:
        return pd.read_sql("SELECT date, usd_egp FROM macro_daily WHERE usd_egp IS NOT NULL "
                           "ORDER BY date", conn, parse_dates=["date"]).set_index("date")["usd_egp"]


def devaluation_episodes(fx: pd.Series = None, threshold: float = JUMP_THRESHOLD,
                         gap_days: int = EPISODE_GAP_DAYS) -> list:
    """Distinct devaluation starts -- a big jump with no jump in the prior
    `gap_days`, so a multi-day cascade counts once, not five times."""
    fx = load_fx() if fx is None else fx
    jump = fx.pct_change() > threshold
    return [d for d in jump[jump].index
            if not jump.loc[:d].iloc[:-1].tail(gap_days).any()]


def forward_window(index: pd.DatetimeIndex, episodes, horizon: int = HORIZON_DAYS) -> pd.Series:
    """True on days that are within `horizon` days BEFORE a devaluation."""
    fwd = pd.Series(False, index=index)
    for d in episodes:
        fwd.loc[(fwd.index >= d - pd.Timedelta(days=horizon)) & (fwd.index < d)] = True
    return fwd


def evaluate_signal(signal: pd.Series, episodes=None, index: pd.DatetimeIndex = None,
                    horizon: int = HORIZON_DAYS) -> dict:
    """Precision, lift over base rate, and recall for a boolean warning signal."""
    fx = load_fx()
    index = fx.index if index is None else index
    episodes = devaluation_episodes(fx) if episodes is None else episodes
    sig = signal.reindex(index).fillna(False).astype(bool)
    fwd = forward_window(index, episodes, horizon)
    base = float(fwd.mean())
    precision = float(fwd[sig].mean()) if sig.any() else float("nan")
    recall = sum(1 for d in episodes if sig.loc[:d].iloc[-2:-1].any())
    return {"base_rate": base, "precision": precision,
            "lift": (precision / base) if base > 0 else float("nan"),
            "recall": recall, "n_episodes": len(episodes), "n_flagged": int(sig.sum())}


def permutation_pvalue(signal: pd.Series, n_perm: int = 5000, seed: int = 0,
                       n_trials: int = 1, horizon: int = HORIZON_DAYS) -> dict:
    """How often does a RANDOM set of event dates produce a lift this large?

    This is the test that matters with six events: almost any statistic can look
    predictive against so few, so the observed lift is compared against a null
    built by re-drawing the event dates. `n_trials` applies a family-wise
    correction for how many hypotheses were tried before reporting this one --
    without it, searching guarantees a 'discovery'.
    """
    fx = load_fx()
    episodes = devaluation_episodes(fx)
    sig = signal.reindex(fx.index).fillna(False).astype(bool)

    def lift_for(events):
        fwd = forward_window(fx.index, events, horizon)
        b = float(fwd.mean())
        return (float(fwd[sig].mean()) / b) if b > 0 else np.nan

    observed = lift_for(episodes)
    rng = np.random.default_rng(seed)
    pool = fx.index[horizon:-horizon]
    null = []
    for _ in range(n_perm):
        fake = pd.DatetimeIndex(rng.choice(pool, size=len(episodes), replace=False))
        l = lift_for(fake)
        if l == l:
            null.append(l)
    null = np.asarray(null)
    p = float((null >= observed).mean())
    return {"observed_lift": float(observed), "p_value": p,
            "p_value_family": float(1 - (1 - p) ** max(int(n_trials), 1)),
            "null_median": float(np.median(null)), "null_p90": float(np.percentile(null, 90)),
            "significant": bool((1 - (1 - p) ** max(int(n_trials), 1)) < 0.05)}


def candidate_signals() -> dict:
    """The four hypotheses that were tested, rebuilt from the database."""
    fx = load_fx()
    with get_connection() as conn:
        px = pd.read_sql("SELECT date, AVG(adjusted_close) egx FROM daily_ohlcv "
                         "GROUP BY date ORDER BY date", conn,
                         parse_dates=["date"]).set_index("date")["egx"]
    egx = px.reindex(fx.index).ffill()
    r = fx.pct_change()
    vol = r.rolling(60).std() * np.sqrt(252)

    episodes = devaluation_episodes(fx)
    days_since = pd.Series(index=fx.index, dtype=float)
    last = None
    for d in fx.index:
        if d in episodes:
            last = d
        days_since.loc[d] = (d - last).days if last else (d - fx.index[0]).days

    egx_vol = egx.pct_change().rolling(20).std() * np.sqrt(252)
    return {
        "H1_flat_peg_long_duration": (vol < 0.044) & (days_since > 250),
        "H2_egx_usd_momentum": (egx / fx).pct_change(60) > (egx / fx).pct_change(60).quantile(0.80),
        "H3_egx_egp_momentum": egx.pct_change(60) > egx.pct_change(60).quantile(0.80),
        "H4_egx_volatility": egx_vol > egx_vol.quantile(0.80),
    }


def run_study() -> pd.DataFrame:
    """Reproduce the full null result: every hypothesis, reported."""
    sigs = candidate_signals()
    rows = []
    for name, s in sigs.items():
        e = evaluate_signal(s)
        rows.append({"hypothesis": name, **e})
    return pd.DataFrame(rows)


def format_study(df: pd.DataFrame) -> str:
    lines = ["DEVALUATION EARLY-WARNING STUDY — RESULT: NULL",
             f"  base rate P(devaluation within {HORIZON_DAYS}d) = {df['base_rate'].iloc[0]*100:.1f}%"
             f"   ({int(df['n_episodes'].iloc[0])} episodes)",
             f"  {'hypothesis':32}{'precision':>11}{'lift':>7}{'recall':>8}"]
    for r in df.itertuples():
        lines.append(f"  {r.hypothesis:32}{r.precision*100:>10.1f}%{r.lift:>7.2f}x"
                     f"{r.recall:>5}/{int(r.n_episodes)}")
    lines.append("  Best candidate (H3) permutation-tested: p=0.096 alone, p=0.332 after")
    lines.append("  correcting for 4 trials -> NOT significant. No usable signal.")
    lines.append("  Devaluation risk here cannot be timed; it must be managed structurally.")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore")
    print(format_study(run_study()))
