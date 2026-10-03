"""
Deflated Sharpe Ratio -- multiple-testing correction (Bailey & Lopez de Prado).

WHY. This project just reported a real-USD Sharpe of 0.62 with a bootstrap CI
of [+0.07, +1.48] -- "significant", but only barely. That CI answers "could ONE
strategy with this track record have arisen by chance?" It does NOT answer the
question that actually matters here: **we tried many variants before reporting
this one.** Optimizers (min-var / equal-weight / risk-parity / CVaR / ensemble),
overlay settings (bear_scale, breaker on/off, vol-target, Kelly), six cash rates
plus a reconstructed schedule, five AUM levels, three markets. Pick the best of
N noisy backtests and the winner's Sharpe is biased upward even if every single
strategy is worthless. That is exactly how this codebase previously mistook a
circuit breaker for edge (+824% -> -90%).

THE MATH, derived rather than imported.

1. Probabilistic Sharpe Ratio. The estimator SR_hat has sampling variance that
   depends on the return distribution's skew and kurtosis (fat tails make a
   Sharpe estimate less trustworthy):

       Var(SR_hat) ~ [1 - g3*SR + ((g4-1)/4)*SR^2] / (n-1)

   where g3 = skew, g4 = kurtosis (NOT excess; g4 = 3 for a normal, which
   recovers the familiar (1 + SR^2/2)/(n-1)). Hence

       PSR(SR*) = Phi[ (SR_hat - SR*) * sqrt(n-1)
                       / sqrt(1 - g3*SR_hat + ((g4-1)/4)*SR_hat^2) ]

   -- the probability the TRUE Sharpe exceeds the benchmark SR*.

2. Deflating the benchmark for N trials. Under the null that all N trials have
   zero true Sharpe, the EXPECTED MAXIMUM of N draws is not 0 -- it grows with
   N. Using the Gumbel approximation for the maximum of iid normals:

       SR0 = sigma_SR * [ (1-gamma)*Phi^-1(1 - 1/N) + gamma*Phi^-1(1 - 1/(N*e)) ]

   with gamma = Euler-Mascheroni (0.5772) and sigma_SR the dispersion of Sharpe
   ratios ACROSS the trials actually run. Then

       DSR = PSR(SR0)

   i.e. "having tried N things, how confident am I that this one is real?"
   DSR > 0.95 is the usual bar.

ALL SHARPES HERE ARE PER-PERIOD (daily), not annualized -- the formulas require
SR and n to be in the same frequency. Annualized figures are shown only for
readability.

HONESTY. sigma_SR and N are judgement calls, so the headline output is a SWEEP
over N (as with the cash-rate sweep): report the trial count at which the result
stops surviving, rather than one flattering number. NOTE: this implements the
deflated-Sharpe half of the standard gate. Combinatorially-symmetric
cross-validation (PBO) is NOT implemented -- do not claim it is.
"""

import logging

import numpy as np
import pandas as pd

from risk.metrics import norm_ppf, skewness, excess_kurtosis

logger = logging.getLogger("esis_quant.deflated_sharpe")

EULER_GAMMA = 0.5772156649015329
TRIAL_SWEEP = [1, 5, 10, 25, 50, 100, 250]


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the Abramowitz-Stegun erf approximation
    (|error| < 1.5e-7). From scratch -- no scipy, per the project's house rule."""
    z = float(x) / np.sqrt(2.0)
    sign = 1.0 if z >= 0 else -1.0
    z = abs(z)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1.0 + sign * y)


def _daily_sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe(returns, sr_benchmark: float = 0.0) -> dict:
    """P(true Sharpe > sr_benchmark), adjusting for skew/kurtosis.
    `sr_benchmark` is a PER-PERIOD Sharpe."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(r)
    if n < 30:
        raise ValueError("need >= 30 observations for PSR")
    sr = _daily_sharpe(r)
    g3 = skewness(r)
    g4 = excess_kurtosis(r) + 3.0          # convert excess -> raw kurtosis
    denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    denom = max(denom, 1e-12)
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return {"psr": float(norm_cdf(z)), "sharpe_daily": sr,
            "sharpe_annual": sr * np.sqrt(252), "skew": float(g3),
            "kurtosis": float(g4), "n_obs": int(n), "z": float(z)}


def expected_max_sharpe(n_trials: int, sigma_sr: float) -> float:
    """E[max of N iid zero-mean Sharpes] -- the benchmark a winner must beat."""
    n_trials = max(int(n_trials), 1)
    if n_trials == 1:
        return 0.0
    a = norm_ppf(1.0 - 1.0 / n_trials)
    b = norm_ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sigma_sr * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(returns, n_trials: int, sigma_sr: float = None,
                    trial_sharpes=None) -> dict:
    """DSR = PSR(E[max Sharpe under the null across n_trials]).
    `sigma_sr` is the PER-PERIOD dispersion of Sharpes across trials; if
    `trial_sharpes` (annualized) is given it is estimated from them."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if sigma_sr is None:
        if trial_sharpes is not None and len(trial_sharpes) > 1:
            sigma_sr = float(np.std(np.asarray(trial_sharpes, dtype=float)
                                    / np.sqrt(252), ddof=1))
        else:
            # fall back to the theoretical sampling sd of a zero-Sharpe estimate
            sigma_sr = 1.0 / np.sqrt(len(r) - 1)
    sr0 = expected_max_sharpe(n_trials, sigma_sr)
    out = probabilistic_sharpe(r, sr_benchmark=sr0)
    out.update({"n_trials": int(n_trials), "sigma_sr": float(sigma_sr),
                "sr0_daily": sr0, "sr0_annual": sr0 * np.sqrt(252),
                "dsr": out.pop("psr"), "survives": None})
    out["survives"] = bool(out["dsr"] > 0.95)
    return out


def sweep_trials(returns, trials=None, sigma_sr: float = None,
                 trial_sharpes=None) -> pd.DataFrame:
    """DSR across a range of assumed trial counts -- the honest headline."""
    trials = TRIAL_SWEEP if trials is None else trials
    rows = []
    for n in trials:
        d = deflated_sharpe(returns, n, sigma_sr=sigma_sr, trial_sharpes=trial_sharpes)
        rows.append({"n_trials": n, "sr0_annual": d["sr0_annual"],
                     "dsr": d["dsr"], "survives": d["survives"]})
    return pd.DataFrame(rows)


def format_sweep(df: pd.DataFrame, label: str, sharpe_annual: float) -> str:
    lines = [f"DEFLATED SHARPE -- {label}  (observed annualized Sharpe {sharpe_annual:.2f})",
             f"  {'trials':>7}{'hurdle SR':>12}{'DSR':>9}   verdict"]
    for r in df.itertuples():
        verdict = "survives" if r.survives else "FAILS (could be selection luck)"
        lines.append(f"  {r.n_trials:>7}{r.sr0_annual:>12.2f}{r.dsr:>9.3f}   {verdict}")
    lines.append("  bar: DSR > 0.95. 'hurdle SR' = Sharpe the best of N worthless")
    lines.append("  strategies would show by chance alone.")
    return "\n".join(lines)
