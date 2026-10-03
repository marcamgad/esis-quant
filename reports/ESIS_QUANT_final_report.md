# ESIS-QUANT — Final Performance Report

_Generated 2026-07-12 from the live database. All figures are
walk-forward, point-in-time, and net of the 0.55% round-trip transaction cost
(including the regime overlay's cash-in/cash-out trades)._

## What the system is
A **risk-managed passive core** for the Egyptian Exchange: a minimum-variance
book (from-scratch Ledoit-Wolf covariance + projected-gradient optimizer) with
a **walk-forward BEAR-regime de-risking overlay** (from-scratch Gaussian HMM,
filtered/causal inference, annual refit). It is **not** a stock-picking system.

## Headline results

| Metric | Full system | Min-var core | Equal-weight (market) |
|---|---|---|---|
| Annualized return | **+24.5%** | +23.9% | +25.2% |
| Sharpe | **1.70** | 1.40 | 1.24 |
| Sortino | **1.50** | 1.27 | 1.05 |
| Max drawdown | **-24.9%** | -42.8% | -57.1% |
| Calmar | **0.98** | 0.56 | 0.44 |
| Max-DD duration (days) | **61.00** | 487.00 | 481.00 |
| Cornish-Fisher VaR 95% | **1.04%** | 1.54% | 1.99% |

**Bottom line:** the full system delivers **Sharpe 1.70 / max drawdown
-25%** versus the market's **1.24 / -57%**,
and time-underwater collapses from **481 days to 61**.
Roughly the market's return at less than half the drawdown.

## Where the edge comes from (Brinson-Hood-Beebower + timing, reconciles exactly)
- **Stock selection adds NO alpha — it costs return.** Min-var core vs equal-weight active
  return = allocation +6.2% + selection -18.4%
  + interaction -5.8%. The core buys lower risk by giving up return
  (the low-volatility trade-off).
- **The regime-timing overlay is the entire value-add: +46% cumulative**,
  and it cuts drawdown at the same time.

## Live state (as of 2026-07-06)
- Regime: **RECOVERY** · Book exposure: **100%** ·
  Circuit breaker: **OK** ·
  Current drawdown: **0.0%**

## Honest caveats (read these)
1. **Returns are nominal EGP.** The pound devalued ~5x over the sample, inflating return
   *levels*; the Sharpe/drawdown *comparisons* remain valid.
2. **No selection alpha.** Confirmed by attribution — do not treat this as a stock picker.
3. **The drawdown circuit breaker whipsaws** on this data (it was found to turn a +824% path
   into -90% via ~1,185 costly round trips) and is **disabled by default**; the regime overlay
   alone controls drawdown better.
4. **Single market, ~9 years, one regime history.** Out-of-sample / other-market robustness is
   untested. This is an in-sample-of-history walk-forward, not a live track record.
5. Sentiment (Layer 4) and the TFT (Layer 6) do **not** contribute to this result — measured to
   be at/near random and excluded; kept for research, not trading.

_Not investment advice._

---

# ADDENDUM (2026-08-04) — The Business Case: Does This Beat Doing Nothing?

This question has been open since the first report and is the only one that
matters to a decision to run the system at all. Everything above answers *"did
it make money in EGP?"* — the wrong question in an economy whose currency lost
most of its value over the sample, and where Egyptian T-bills paid a high
nominal yield for zero work and no equity risk.

**Real (USD-deflated) figures are the headline below. Nominal EGP is the
footnote.** That ordering is the reverse of the rest of this report, deliberately.

## The comparison

| | CAGR | vol | Sharpe | MaxDD |
|---|---|---|---|---|
| **Strategy (real USD)** | **−0.0%** | 24.9% | 0.14 | **−52.8%** |
| **T-bills (real USD)** | **−2.2%** | 21.0% | 0.03 | −57.6% |
| *→ excess (real)* | *+2.2%* | | | |
| Strategy (nominal EGP) | 19.3% | 14.9% | 1.26 | −45.3% |
| T-bills (nominal EGP) | 16.7% | 0.3% | 49.16 | 0.0% |

*(mean assumed bill yield 16.9%)*

## Does the verdict survive the assumption?

T-bill yields are approximated from the **reconstructed** CBE policy rate
(`data/rates.py`) plus a premium — not a fetched auction series. So the
comparison was re-run across plausible yields:

| assumed yield | strategy real | T-bill real | excess | beats? |
|---|---|---|---|---|
| 8% | −0.0% | −9.5% | +9.5% | YES |
| 12% | −0.0% | −6.1% | +6.1% | YES |
| 16% | −0.0% | −2.8% | +2.8% | YES |
| **20%** | −0.0% | +0.6% | **−0.6%** | **NO** |
| **24%** | −0.0% | +3.9% | **−3.9%** | **NO** |

**The verdict is SENSITIVE.** The strategy beats bills only if they yielded 16%
or less over the period. Above ~19% it loses. Egyptian rates spent meaningful
stretches above 20%, so this is not a comfortable margin — it is a coin-flip
that depends on a number we have not verified.

## What this actually says

1. **The strategy's real return is approximately zero.** Over eleven years it
   produced ~0% CAGR in hard-currency terms. The nominal 19.3% is almost
   entirely currency illusion.
2. **It took a −52.8% real drawdown to earn that zero.** T-bills reached −57.6%
   — but that drawdown is *pure currency*, suffered while sitting in the safest
   instrument available. The strategy took full equity risk on top and did not
   escape it.
3. **Both alternatives lost real money.** This is the finding that reframes the
   project: the dominant problem is not strategy selection, it is **EGP
   exposure itself**. No allocation among Egyptian equities fixes a currency
   that fell ~6.25× against the dollar.
4. Consistent with the earlier deflated-Sharpe result (fails at N=1), the
   devaluation-timing null, and the regime-instability finding — every
   independent line of evidence points the same way.

## Answer

**Does this beat holding T-bills, in real terms, net of costs, after accounting
for the risk taken? — MARGINAL AT BEST, AND NOT ROBUST.** It wins by ~2pp under
mid-range yield assumptions, loses under high ones, and earns roughly zero real
return either way while carrying a ~53% real drawdown. Neither option preserved
purchasing power.

*Reproduce: `py -3.11 -m analysis.business_case`*
