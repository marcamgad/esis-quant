# ESIS-QUANT — Re-Audit After Remediation

**Date:** 2026-08-05 (post-fix) · **Scope:** re-scored from scratch, not upgraded · **Tests:** 412 passing (was 401)

Scoring: ✅ implemented + tested + **live-path evidence** · 🟡 implemented but weak · 🔴 missing · ⚠️ implemented but dangerous/misleading

> **The headline finding is not in the score table.** With the cost model corrected, the strategy **no longer beats T-bills in real terms**: real CAGR **−3.94%** vs T-bills **−2.00%**, real Sharpe **−0.016** vs **+0.037**. And a factor regression shows the full system's alpha is **+0.08% with t = 0.02** — statistically indistinguishable from zero once market beta and a low-volatility tilt are accounted for. See Sections A and B below.

---

## A. The number that matters

**Question:** does the strategy still beat T-bills in real, risk-adjusted terms after the corrections?

**Answer: No.**

| | real (USD-deflated) CAGR | real Sharpe |
|---|---|---|
| Strategy (full system, 10k book) | **−3.94%** | **−0.016** |
| Egyptian T-bills (mean yield 16.9%) | −2.00% | +0.037 |
| **Excess** | **−1.94pp** | **worse** |

Both are negative in USD terms — Egyptian assets lost purchasing power over the sample. But the T-bill lost *less*, with *less* risk. `beats_tbill_real = False`.

**Deflated Sharpe** on the corrected series: nominal SR 0.968, skew −0.20, kurtosis 10.0.

| trials assumed | DSR | survives |
|---|---|---|
| 1 | 0.9991 | ✅ |
| 10 | 0.9413 | ❌ |
| 50 | 0.8081 | ❌ |

It survives only if you pretend this system tested exactly one hypothesis. It tested far more than ten.

## B. Is the "alpha" a factor tilt? (new — `analysis/factor_exposure.py`)

Regression on factors built from this repo's own price panel, Newey-West lag 5, n = 2,608:

```
core          ALPHA +5.93%/yr  t=1.84     MKT 0.703 (t=15.0)  VOL 0.095 (t=2.02)
full_system   ALPHA +0.08%/yr  t=0.02     MKT 0.558 (t=12.3)  VOL 0.075 (t=1.86)
```

R² = 0.683 (core). **The full system's alpha is zero to two decimal places.** The strategy is ~55–70% market beta plus a statistically significant low-volatility tilt — a well-documented, freely available risk premium, not skill. Neither MOM nor SIZE loads.

This is the single most important result of the remediation and it did not require fixing anything: nobody had asked.

## C. Cost sensitivity (new — `analysis/cost_sensitivity.py`)

```
      book EGP  mult  sys CAGR  sys Sharpe  core CAGR
         1,000    1x     7.87%     0.505      10.62%
         1,000    2x    -1.95%     0.098      -9.30%   <- DIES
         1,000    3x   -14.32%    -0.114     -31.36%
        10,000    1x    14.23%     0.958      25.84%
        10,000    3x    10.87%     0.746      19.95%   <- survives
       100,000    3x    11.67%     0.805      23.13%
     1,000,000    3x     8.60%     0.612      19.61%
```

At 1,000 EGP the corrected model drops system CAGR **13.69% → 7.87%** and the strategy **dies at 2× cost**. At 10,000+ it survives 3×. The distortion was real and concentrated exactly where the live book was trading.

## D. Headline numbers, before vs after

| book | cost model | CAGR | Sharpe | MaxDD | turnover/yr |
|---|---|---|---|---|---|
| core | OLD flat 0.55% | 26.08% | 1.479 | −44.7% | not reported |
| core | **NEW real schedule @10k** | **25.84%** | **1.463** | **−43.0%** | **3.80×** |
| full system | OLD flat 0.55% | 13.69% | 0.927 | −50.1% | not reported |
| full system | **NEW real schedule @10k** | **14.23%** | **0.958** | **−48.9%** | **3.80×** |

**Honest note:** at 10k the correction barely moved the headline, and full-system CAGR *rose* slightly. That is not the fix failing — it is because I also removed a **commission double-count** I introduced en route (`_impact_only` charged commission *and* `_schedule_cost` charged the broker leg). Correcting only the level while leaving the double-count would have produced a falsely punitive number. The correction bites hard at 1,000 EGP (13.69% → 7.87%), which is where it should.

---

## E. Section scores

| # | Section | Score | Evidence |
|---|---|---|---|
| 0 | System definition | ✅ | One cost model. `config.THNDR_COMMISSION_PCT`/`ROUND_TRIP_COST_PCT` removed; `config.__getattr__` raises on access; 9 call sites migrated to `fees.book_pct_per_side`. |
| 1 | Market data infrastructure | 🟡 | Unchanged. No tick/L1/L2; PIT reconstruction still partial for universe membership. |
| 2 | Data quality | 🟡 | Unchanged. Gate hard-fails (`main.py:362`), but still correct-in-place, no quarantine. |
| 3 | Point-in-time data | 🔴 | **Unchanged by rule.** No fundamental data exists in this system. P5 was scoped as "do not wire", so nothing moved. |
| 4 | Universe construction | ⚠️ | **Disclosure only.** Bias documented in `data/universe_discovery.py`; delisted history not sourced. Per your rule, disclosure ≠ remediation. |
| 5 | Fundamental data | 🔴 | Not collected. Unchanged. |
| 6 | Feature engineering | ✅ | Unchanged and still clean — live path uses only backward-looking covariance + regime features. |
| 7 | Alpha research | ✅ | Strengthened: `analysis/factor_exposure.py` asks whether the edge is a factor tilt and reports that it is. |
| 8 | Statistical validation | ✅ | Bootstrap + DSR + permutation, now **re-run on corrected costs** and reported at 1/10/50 trials. |
| 9 | Backtesting engine | ✅ | `use_fee_schedule=True` is the default in `BacktestEngine.__init__`; `portfolio/system.py` inherits it; commission double-count removed; tested (`tests/test_backtest.py`, `tests/test_fees.py`). |
| 10a | Look-ahead bias | ✅ | Unchanged. |
| 10b | Survivorship bias | 🔴 | **Unchanged by rule.** No delisted-name history sourced. Feasibility investigated: Yahoo `.CA` resolves only currently-listed symbols; no free source of delisted EGX history found. Reported, not fixed. |
| 10c | Selection bias | 🔴 | Unchanged. 20 of 47 names still chosen on 2026 liquidity. |
| 10d | Data leakage | 🟡 | Unchanged — contained, not eliminated; causal layer still off the live path. |
| 10e | Overfitting / snooping | 🟡 | DSR now reported at multiple trial counts, showing failure at ≥10. Parameters still tuned on one history. |
| 10f | Unrealistic fills | 🟡 | `ops/execution_quality.measured_slippage` now produces a shrunk rolling estimate (16 fills, 44% weight on data, flagged below the 20-fill floor). Not yet consumed by the engine. |
| 11 | Train/test architecture | 🟡 | Walk-forward unchanged. **Embargo NOT added** — see Deferred. |
| 12 | Market regime analysis | 🟡 | Unchanged. 0 live bear days. |
| 13 | Portfolio construction | ✅ | Unchanged and correct. |
| 14 | Risk engine | ✅ | **Fixed.** `risk/pretrade.py` rejects/resizes per order; wired into `ops/order_ticket.write_ticket` (rejections printed into the ticket header); 11 tests in `tests/test_pretrade.py`; fails closed on unknown liquidity. |
| 15 | Performance evaluation | ✅ | Turnover now reported alongside CAGR/Sharpe/MaxDD (`backtest/analytics.turnover_stats`, wired through `portfolio/system.summ`): 0.317/rebalance, 3.80×/yr. |
| 16 | Benchmarking | 🟡 | Re-confirmed: no EGX30/EGX70 series obtainable (`config.py:231` still accurate — `^CASE30`, `EGX30.CA` etc. all return empty). Equal-weight + T-bill only. |
| 17 | Transaction cost analysis | ✅ | `analysis/cost_sensitivity.py` sweeps 1×/2×/3× across four book sizes and prints a falsification verdict. Run; results in Section C. |
| 18 | Stress testing | 🟡 | **Monte Carlo NOT added** — see Deferred. Bootstrap + profit-range only. |
| 19 | Parameter robustness | 🟡 | **Lookback/regime-count sweep NOT added** — see Deferred. |
| 20–21 | ML audit / beats baselines | 🔴 | **Unchanged by rule.** No OOS comparison against momentum/buy-and-hold was run. Correctly still off the live path. |
| 22 | Explainability | 🟡 | Unchanged — rationale line still regime-only. |
| 23 | Execution engine | 🟡 | Manual by design; duplicate-order guard exists. Partial-fill/rejection handling **not** added and **not** documented as out-of-scope. |
| 24 | Production safety | ✅ | Kill switch live + per-order limits now binding (`MAX_POSITION_WEIGHT`, `MAX_DAILY_LOSS_BLOCK_NEW`, `ADV_LIMIT_ACTION`), deliberately separate from the kill switch. |
| 25 | Paper trading | 🔴 | **Untouched, by rule.** 19 days, 0 bear days, `sufficient: False`. No code makes time pass. |
| 26 | Live monitoring | 🟡 | **Drift alerting NOT added** — see Deferred. |
| 27 | Alpha decay | 🔴 | **Not added** — see Deferred. |
| 28 | Factor exposure | ✅ | `analysis/factor_exposure.py`: MKT/SIZE/MOM/VOL with Newey-West errors. Run, and it returns a result damaging to the strategy — which is the point. |
| 29 | Capacity | ✅ | Unchanged, still wired into the daily report. |

---

## F. SCORE CHANGES

| # | Section | From | To | Why |
|---|---|---|---|---|
| 0 | System definition | 🟡 | ✅ | Two cost models collapsed to one; removed constants now raise. |
| 9 | Backtesting engine | ⚠️ | ✅ | Fee schedule is the default and reaches `portfolio/system.py`; double-count fixed. |
| 14 | Risk engine | 🔴 | ✅ | Limits reject/resize orders and are wired into the ticket path, with tests. |
| 15 | Performance evaluation | 🟡 | ✅ | Turnover reported. |
| 17 | Transaction cost analysis | 🔴 | ✅ | 1×/2×/3× sweep built and run. |
| 28 | Factor exposure | 🟡 | ✅ | Regression built and run; result reported. |
| 8 | Statistical validation | ✅ | ✅ | No change in score; re-run on corrected costs. |
| 10f | Unrealistic fills | 🟡 | 🟡 | Estimate exists but isn't consumed by the engine — not enough for ✅. |

**Six sections moved to ✅** — at your cap. I have re-checked each against the "test + live-path evidence" rule; all six are called by `portfolio/system.py` or `ops/order_ticket.py` and covered by passing tests. Sections 10f, 11, 18, 19, 26, 27 deliberately did **not** move despite work being possible, because the work wasn't done.

**Nothing moved on 3, 4, 10b, 10c, 20–21, 25** — the structurally-locked set. Disclosure was added for the universe bias and is scored as disclosure.

---

## G. New problems created by the fixes

Per your instruction to flag trades of one distortion for another:

1. **`cost_per_turnover=0.0` no longer means zero cost.** The schedule now applies regardless, so a cost-free run requires `use_fee_schedule=False` explicitly. This broke a legitimate mechanics test, which I fixed by opting out rather than by weakening the default. Any old script assuming zero-cost semantics is now silently paying fees.
2. **I introduced and then removed a commission double-count.** Worth recording because it nearly produced a falsely *pessimistic* headline, which would have been just as dishonest as the flattering one it replaced.
3. **A regression I introduced:** defaulting `aum_egp` unconditionally broke the documented `aum_egp=None → flat cost` contract, silently charging impact costs to explicit cost-free runs. Fixed; the AUM now defaults only when the schedule is on.
4. **Test fixtures were unrealistic, and the new control exposed it.** 11 tests wrote `value_traded_egp = 0`, so the pre-trade screen correctly fail-closed on unknown liquidity. I fixed the fixtures (realistic volume, concentration limit relaxed *in ticket-mechanics tests only*, with the limit itself tested directly in `tests/test_pretrade.py`). I did not weaken the control.
5. **`MAX_POSITION_WEIGHT = 0.20` is looser than the optimizer's `max_weight = 0.15`.** Deliberate — the pre-trade limit is a backstop against sizing bugs, not a second optimizer — but it means the screen will rarely bind on optimizer output. It binds on integer-allocator output and manual edits, which is where the risk was.

---

## H. Deferred — not done, not disguised

These were in scope and I did not complete them. None is blocked; I ran out of session, not out of method.

- **P4.12 embargo** on the live walk-forward refit (`models/regime_pipeline.py`). Purged CV still exists only in the dead causal path.
- **P6.16 Monte Carlo** path perturbation.
- **P6.17 parameter sweep** over lookback and regime count.
- **P6.18 per-name rationale** logging.
- **P6.19 partial-fill/rejection** handling or an explicit out-of-scope statement.
- **P6.20 drift alerting** on execution shortfall and cost-model error.
- **P6.21 IC-decay / half-life** tracking.
- **P5.13–14** sentiment date-alignment fix and the OOS baseline comparison (correctly left off the live path either way).
- **P2.5** disclosure header is in the module docstring, **not** yet injected into every generated report header.

---

## I. Overall score

### **40–60%** — unchanged band, for a different reason

Before, the band was set by *not knowing* whether the numbers were real. Now we know, and the answer is worse than the uncertainty was.

The engineering is materially better: one cost model, risk limits that bind, turnover reported, cost sensitivity measured, factor exposure tested, 412 tests passing. Sections 0, 9, 14, 15, 17 and 28 are genuinely fixed.

But the corrected system **does not beat T-bills in real terms** (−3.94% vs −2.00%), its **alpha is 0.08% with t = 0.02** once market beta and a low-vol tilt are removed, its **DSR fails at ≥10 trials**, and it **dies at 2× cost on a 1,000 EGP book**. The universe remains survivorship-biased and the forward record still has zero bear days.

### Does a single 🔴 override?

**Yes — Section 10b/10c (survivorship + selection bias) now overrides**, having replaced Section 9 as the binding constraint. Every historical statistic above is computed on a universe selected by 2026 liquidity. That includes the factor regression and the cost sweep. The bias is disclosed, unmeasured, and unbounded.

The system continues to refuse to act on its own numbers — go-live gate 5/5 failing, `LIVE_MIN_VAR_ONLY=True`, kill switch armed, and now per-order limits that bind. **No safety gate was relaxed in this work.** That remains the most defensible thing about the project: the evidence is weak and the system behaves as though it knows that.
