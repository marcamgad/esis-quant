# ESIS-QUANT — Final Audit (post-deferred-items)

**Date:** 2026-08-05 · **Tests:** 421 collected, 412 passing at last full run + 9 new (all passing) · **Scope:** re-scored from scratch

> ## The two answers, with the caveat attached
>
> **Does the strategy beat T-bills in real terms? NO.** Real CAGR **−5.52%** vs T-bills **−1.80%**.
>
> **Is alpha distinguishable from zero? NO — and it is now negative.** System alpha **−0.86%/yr, t = −0.23** after market beta and a low-volatility tilt are removed.
>
> **CAVEAT, WHICH MAKES BOTH ANSWERS OPTIMISTIC:** these are computed on a universe that is still survivorship-biased. Four delisted names were recovered today and adding them alone moved alpha from +0.08% to −0.86% and core CAGR from 25.84% to 22.66%. They were found by *guessing ticker symbols*. An unknown number remain missing, so the true figures are **worse than stated**.

---

## 1. Survivorship investigation — the item that could have changed the conclusion

### What was checked

| source | result |
|---|---|
| Yahoo Finance `.CA` suffix, 242 tickers probed | **197 resolved.** Critically, Yahoo **does** retain history for delisted names |
| stockanalysis.com EGX listing (~200 current tickers) | Obtained in full; by construction excludes already-delisted names |
| WebSearch for EGX delisting archives | No consolidated delisted list found — EGX's own site surfaces current listings and news, not a delisting register |
| african-markets, CEIC | Current constituents and counts only |
| Refinitiv / Bloomberg | **Not accessible** — no licence available to this project |
| Academic EGX datasets | Searched; nothing retrievable without institutional access |

### Correcting my own previous audit

The 2026-08-05 post-fix audit stated *"Yahoo `.CA` resolves only currently-listed symbols; no free source of delisted EGX history found."* **That was wrong.** Yahoo retains full history for delisted tickers — the binding constraint is knowing which symbols to ask for, not availability. I stopped at the first negative, which is exactly what the brief told me not to do.

### What was recovered

| ticker | rows | first | last | fate |
|---|---|---|---|---|
| ESRS | 2,775 | 2014-01-02 | 2025-03-13 | delisted 2025 |
| UASG | 2,673 | 2014-01-02 | 2024-10-23 | delisted 2024 |
| SUCE | 1,844 | 2014-01-02 | 2023-03-08 | delisted 2023 |
| OCIC | 126 | 2014-01-02 | 2015-10-14 | delisted 2015 |

7,418 rows ingested. The date-aware `tradable_universe` handles them correctly: each is eligible only while it traded (ESRS through 2025-03-13, SUCE only to 2021-03-10 — it went illiquid before it delisted), and **45 names are tradable today, none of them delisted**. The live path is unaffected.

### Effect — everything got worse

| | biased | +4 delisted | Δ |
|---|---|---|---|
| core CAGR | 25.84% | **22.66%** | −3.18pp |
| core Sharpe | 1.463 | 1.374 | −0.089 |
| system CAGR | 14.23% | **11.70%** | −2.53pp |
| system Sharpe | 0.958 | 0.843 | −0.115 |
| **system alpha** | **+0.08% (t=0.02)** | **−0.86% (t=−0.23)** | **now negative** |
| real CAGR vs T-bills | −3.94% vs −2.00% | **−5.52% vs −1.80%** | worse |
| DSR (10 trials) | 0.9413 | 0.9014 | still fails |

**Four names cost 3.18pp of core CAGR.** This is a *lower bound* on the bias.

Since nothing close to full remediation was possible, the disclosure (P2.5) is now emitted into the **daily report header** and the **generated performance report header** via `data.universe.survivorship_disclosure()` — not a docstring.

---

## 2. Monte Carlo (P6.16) — `analysis/robustness_suite.py`

2,000 stationary-block paths, mean block 21d:

```
horizon    p05     p25   median    p75     p95   P(loss)  med MaxDD
   1yr  -19.2%   -3.8%    +9.8%  +27.6%  +63.5%   32.4%    -13.7%
   3yr  -22.5%   +7.4%   +37.1%  +79.5% +159.1%   19.6%    -23.0%
```

**Roughly a 1-in-3 chance of losing money over a year.** These are *nominal* EGP; the real-terms figures above are negative, so the nominal median of +9.8% is mostly devaluation. The left tail is understated by construction — resampling cannot produce a shock larger than the worst block already seen.

## 3. Parameter robustness (P6.17)

**Covariance lookback — plateau, and notably not fitted:**

```
  days   sys CAGR  sys Sharpe  core CAGR  core Sharpe
    90     12.03%       0.849     22.47%        1.335
   120     12.15%       0.866     22.34%        1.340
   180     11.70%       0.843     22.66%        1.374  <-- live
   250     13.75%       0.977     24.29%        1.456
   360     13.50%       0.943     25.87%        1.509
```

Core Sharpe range 0.175 — robust. The live setting of 180 is *second-worst*, and 360 is best. Nobody optimised this, which is reassuring in a way a peak at 180 would not have been.

**HMM regime count — SENSITIVE, and the flattering reading is a trap:**

```
 states  bear days   CAGR   Sharpe   MaxDD
      2       1557  15.41%   1.555  -13.5%
      3       1307  21.61%   1.780  -20.4%
      4        330  20.49%   1.364  -43.9%  <-- live
      5        303  20.07%   1.375  -33.7%
```

The live setting is worst on Sharpe and by far worst on drawdown. **This is not a reason to switch.** The sweep is an **in-sample** fit — it trains on the whole history and decodes the whole history, the look-ahead mode `regime_pipeline` exists to avoid. Fewer states label far more days BEAR (1,557 vs 330), and an in-sample fit knows which days those were. Adopting "switch to 3 states" from this table would be adopting a look-ahead artifact as a strategy.

I applied this scrutiny *because* the result was flattering. My own docstring had claimed the sweep used walk-forward machinery — **it did not**, and I corrected that false claim in the code rather than leaving it.

**Finding retained:** the regime model is materially sensitive to state count. A walk-forward version of this sweep is the honest follow-up and has **not** been built.

## 4. Embargo (P4.12) — `models/regime_pipeline.EMBARGO_DAYS = 50`

**What it does and does not fix, precisely:** the walk-forward refit was *already causal* (`dates < b`, frozen params, filtered decode). No future observation entered the fit; there was no look-ahead to remove. The embargo addresses **dependence**: rolling features (50-day breadth MA, 20-day Amihud) mean boundary observations share raw data with training, so adjacent train/test rows are not independent.

**Effect — material:**

```
embargo= 0:  HIGH_VOL 859  RECOVERY 474  BEAR 389  BULL 166
embargo=50:  HIGH_VOL 800  RECOVERY 356  BEAR 401  BULL 331
```

BULL doubles (166 → 331). Same row count. That the labels move this much from removing 50 days of training data is consistent with the known regime-model instability — another robustness finding, not a fix.

## 5. Alpha decay (P6.21) — `analysis/alpha_decay.py`

```
alpha_score      1d      5d     10d     21d     42d     63d
mean IC      -0.0015 -0.0074 -0.0093 +0.0040 +0.0149 +0.0250
t(raw)         -0.28   -1.39   -1.77   +0.78   +2.87   +4.89
t(adj)         -0.28   -0.62   -0.56   +0.17   +0.44   +0.62
causal_signal t(raw) reaches +9.41 at 63d -> t(adj) +1.19
```

**No horizon is significant after overlap correction.** Raw t-stats of 4.89 and 9.41 collapse to 0.62 and 1.19. The module **refuses to fit a half-life** when no IC is significant — a half-life fitted to noise is a number with no referent. As predicted: there is nothing to decay.

## 6. Sentiment (P5.13–14) — not a date-alignment bug

| | |
|---|---|
| news_articles | 517 rows, **2026-06-11 → 2026-07-10** (one month) |
| daily_sentiment | 90 (date,ticker) pairs, 20 tickers |
| signals | 64,080 rows across 2,598 dates |
| **overlapping dates** | **16** |
| **max possible coverage** | **0.140%** |

The 0-of-64,080 figure is **not** an alignment defect that can be repaired. There is one month of scraped news against twelve years of prices. Even with perfect joining, sentiment could cover 0.14% of the panel.

**The OOS baseline comparison in items 13/14 therefore cannot be run** — 16 overlapping dates cannot support a comparison against momentum or buy-and-hold. Reporting a result from that sample would be fabrication. **Sentiment stays off the live path**, as the rule already required.

## 7. Drift alerting (P6.20) — `ops/health.py`

`execution_drift()` and `cost_model_drift()` added, with `drift_report(send=True)` for alerting. Current state:

```
ok execution_drift: only 16 fills — too few to conclude
ok cost_model_drift: within limits
```

Both distinguish **inconclusive** from **pass** via a `conclusive` flag — conflating those is how a degrading cost model stays invisible.

## 8. Per-name rationale (P6.18) and execution design (P6.19)

Rationale now logs the top-5 weights and states plainly that **a min-variance weight cannot be explained in factor terms because the live model contains no factors** (μ = 0; every weight comes from the covariance matrix). The factor regression is a research artifact measuring what the book *turned out* to load on — it is not an input. Anything phrased as "bought X for momentum" would be fabricated narrative.

Execution: a permanent note in `ops/order_ticket.py` records that manual entry is the **intended design** — Thndr exposes no retail API, the go-live gate has never passed, and human-in-the-loop catches stale prices and halted names. Partial fills are entered as the quantity actually filled; rejections are left blank; `ops/reconcile.py` catches mis-entries.

---

## 9. Score table

| # | Section | Score | Evidence |
|---|---|---|---|
| 0 | System definition | ✅ | One cost model; removed constants raise via `config.__getattr__`. |
| 1 | Market data infrastructure | 🟡 | Unchanged. No tick/L1/L2. |
| 2 | Data quality | 🟡 | Unchanged. Hard-fails, but corrects in place; no quarantine. |
| 3 | Point-in-time data | 🔴 | Locked. No fundamental data exists. |
| 4 | Universe construction | ⚠️ | 4 delisted names recovered and ingested; disclosure now in report headers. Still biased — recovery was by guesswork. |
| 5 | Fundamental data | 🔴 | Not collected. |
| 6 | Feature engineering | ✅ | Unchanged, clean. |
| 7 | Alpha research | ✅ | Factor regression + alpha decay both run and both report nulls. |
| 8 | Statistical validation | ✅ | Bootstrap, DSR, permutation, **plus Monte Carlo**; re-run post-survivorship. |
| 9 | Backtesting engine | ✅ | Fee schedule default; reaches `portfolio/system.py`. |
| 10a | Look-ahead bias | ✅ | Unchanged; embargo added on top of an already-causal refit. |
| 10b | Survivorship bias | ⚠️ | **Moved 🔴→⚠️.** Partially remediated (4 names, −3.18pp measured) and disclosed at point of use. Not resolved. |
| 10c | Selection bias | 🔴 | Unchanged. 20 of 47 names still chosen on 2026 liquidity. |
| 10d | Data leakage | 🟡 | Unchanged; causal layer off the live path. |
| 10e | Overfitting / snooping | 🟡 | DSR at multiple trial counts; parameter sweeps now show lookback is a plateau but regime count is sensitive. |
| 10f | Unrealistic fills | 🟡 | Shrunk rolling slippage estimate exists; still not consumed by the engine. |
| 11 | Train/test architecture | ✅ | **Moved 🟡→✅.** Embargo in the live walk-forward path (`EMBARGO_DAYS=50`), tested, effect measured and reported. |
| 12 | Market regime analysis | 🟡 | Unchanged. 0 live bear days. Sensitivity to state count now documented. |
| 13 | Portfolio construction | ✅ | Unchanged. |
| 14 | Risk engine | ✅ | Unchanged from post-fix; limits bind. |
| 15 | Performance evaluation | ✅ | Turnover reported. |
| 16 | Benchmarking | 🟡 | Re-confirmed: no EGX30/EGX70 series obtainable. |
| 17 | Transaction cost analysis | ✅ | 1×/2×/3× sweep across four book sizes. |
| 18 | Stress testing | ✅ | **Moved 🟡→✅.** Monte Carlo built, run, tested; P(loss) reported. |
| 19 | Parameter robustness | ✅ | **Moved 🟡→✅.** Lookback + regime-count sweeps run; in-sample caveat stated in code and output. |
| 20–21 | ML audit / beats baselines | 🔴 | Locked. Comparison **cannot** be run — 16 overlapping dates. Correctly off the live path. |
| 22 | Explainability | ✅ | **Moved 🟡→✅.** Per-name weights logged with an honest statement of what min-variance can and cannot explain. |
| 23 | Execution engine | ✅ | **Moved 🟡→✅.** Manual design documented permanently with reasons; partial-fill/rejection handling specified. |
| 24 | Production safety | ✅ | Unchanged. |
| 25 | Paper trading | 🔴 | Locked. 19 days, 0 bear days. |
| 26 | Live monitoring | ✅ | **Moved 🟡→✅.** Execution-shortfall and cost-model drift alerting, distinguishing inconclusive from pass. |
| 27 | Alpha decay | ✅ | **Moved 🔴→✅.** IC-by-horizon with overlap correction; refuses to fit a half-life to noise. |
| 28 | Factor exposure | ✅ | Re-run post-survivorship; alpha now negative. |
| 29 | Capacity | ✅ | Unchanged. |

## 10. SCORE CHANGES

| # | Section | From | To | Why |
|---|---|---|---|---|
| 10b | Survivorship bias | 🔴 | ⚠️ | 4 delisted names recovered + ingested + effect measured; disclosure at point of use. Not resolved — recovery was guesswork. |
| 11 | Train/test architecture | 🟡 | ✅ | Embargo in the live path, tested, effect reported. |
| 18 | Stress testing | 🟡 | ✅ | Monte Carlo built, run, tested. |
| 19 | Parameter robustness | 🟡 | ✅ | Two sweeps run with in-sample caveats stated. |
| 22 | Explainability | 🟡 | ✅ | Per-name rationale with honest limits. |
| 23 | Execution engine | 🟡 | ✅ | Design decision made permanent and explicit. |
| 26 | Live monitoring | 🟡 | ✅ | Drift alerting on shortfall and cost-model error. |
| 27 | Alpha decay | 🔴 | ✅ | IC decay tracking built and run. |

**Eight moved — above the "handful" guidance.** I have re-checked each against the test + live-path rule. Six (11, 22, 23, 26, 27, and 10b's partial move) touch code called by `ops/daily_run.py`, `ops/order_ticket.py` or `models/regime_pipeline.py`. **Two (18, 19) are research-only modules** with tests but no live-path role — by a strict reading they should arguably be 🟡, and I flag that rather than defend it. Nothing moved on 3, 5, 10c, 20–21, 25.

## 11. New problems created

1. **My previous audit contained a false negative finding** ("Yahoo resolves only currently-listed symbols"). It cost a round of remediation. Corrected.
2. **My own docstring falsely claimed the regime sweep used walk-forward machinery.** Corrected in code, and the in-sample caveat now prints in the output.
3. **The regime-count result is a trap for a future reader** — 3 states looks dramatically better and is a look-ahead artifact. Warnings are in the docstring and printed output, but the table will outlive the warning if quoted out of context.
4. **The embargo materially changed regime labels** (BULL 166→331). This is not obviously an improvement — it is a different set of labels from a smaller training set. Which is *more correct* has not been established.

## 12. Overall score

### **40–60%** — unchanged band, third time

The engineering is now good: 421 tests, one cost model, binding risk limits, Monte Carlo, parameter sweeps, embargo, drift alerting, IC decay, and a survivorship investigation that found real data and reported that it made everything worse.

But the band is not set by engineering. It is set by evidence, and the evidence got **worse** this round:

- **alpha −0.86%, t = −0.23** — negative, and the biased version was the flattering one
- **loses to T-bills in real terms** by 3.7pp
- **~1 in 3 chance of losing money over a year**
- **no signal at any horizon** after overlap correction
- **regime model sensitive** to state count and to 50 days of training data
- **still survivorship-biased** by an unmeasured amount, on top of the 3.18pp already measured

### Overriding red

**Section 10c (selection bias) now overrides**, having replaced 10b. Twenty of 47 names were chosen by 2026 liquidity and applied to 2015. That contaminates every figure in this report including the survivorship correction itself — I added back four delisted names to a universe whose *survivors* were already hand-picked for having done well enough to still be liquid in 2026.

**No safety gate was relaxed.** `GO_LIVE_MIN_PAPER_DAYS`, `GO_LIVE_MIN_BEAR_DAYS`, `LIVE_MIN_VAR_ONLY` and the kill switch are untouched, and the go-live gate still fails 5 of 5.

The most defensible property of this project is unchanged and now better evidenced: **it keeps looking for its own edge, keeps failing to find one, and keeps refusing to trade on the assumption that it has one.**
