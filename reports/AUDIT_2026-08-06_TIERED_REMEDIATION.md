# ESIS-QUANT — Tiered Remediation

**Date:** 2026-08-06 · **Scope:** TIER 1 ONLY (evidentiary integrity) · **Tests:** 426 passing

Tiers 2–5 not started, per the instruction to finish and report Tier 1 first.

---

## TIER 1 — Evidentiary integrity

### 1. Selection bias — point-in-time universe · **DONE**

**Changed:** new `data/pit_universe.py` (`build_pit_universe`, `store_pit_universe`,
`load_pit_tradable`, `turnover_report`). At each date it ranks *every* ticker with price
history by trailing 60-session median value traded and takes the top 50. Membership is
recomputed per date from data available at that date only.

Supporting data work: ingested **147 additional EGX tickers** (`ingest_full_egx.py`), taking
the price panel from 51 to **198** names.

**Proof the fix does what it claims** (`turnover_report`):

```
distinct names ever included : 147     (vs 47 hand-picked)
names present on EVERY date  : 0       (a fixed list would show ~47)
membership changes           : 1335 entries / 1335 exits
mean universe size           : 50 names
```

Zero names survive on every date, so this is genuinely dynamic membership, not the old list
in disguise.

**Metric delta — this is the headline of the entire session:**

| | BIASED (51 hand-picked) | POINT-IN-TIME (147) | delta |
|---|---|---|---|
| core CAGR | 24.27% | **13.75%** | **−10.52pp** |
| core Sharpe | 1.463 | **0.755** | **−0.708** |
| system CAGR | 13.27% | **3.14%** | −10.13pp |
| system Sharpe | 0.949 | **0.263** | −0.686 |
| system MaxDD | −51.2% | **−59.8%** | worse |
| **alpha /yr** | **+7.54% (t = 1.93)** | **−2.46% (t = −0.61)** | **positive → negative** |
| real (USD) CAGR | −4.22% | **−12.79%** | −8.57pp |
| beats T-bills real | False | **False** | unchanged |

**Selection bias was worth roughly 10pp of CAGR and 0.7 of Sharpe — three times the
survivorship effect (3.18pp) measured last session.** It was the single largest distortion in
the project, exactly as the previous audit's overriding-red called it.

**Scrutiny on the flattering number, per constraint #5.** The BIASED column here shows
alpha **+7.54%, t = 1.93** — much better than the −0.86% reported last session. That is *not*
new evidence of alpha. Two things changed underneath it: (a) the trading calendar is
recomputed over 198 tickers, so a different set of low-coverage dates is dropped; (b)
`build_factors` now builds the MKT proxy from 198 names instead of 51, and a broader, weaker
market proxy mechanically raises measured alpha against it. The yardstick moved, not the
strategy. When the *universe* is also corrected, alpha goes to −2.46%. Both columns above use
the identical 198-name factor set, so the delta between them is meaningful even though
neither is comparable to last session's number.

**Still unresolved:**
- The PIT universe is **measured, not adopted**. The live book still runs the 51-name
  universe. Switching the live path to PIT would be a strategy change (expected CAGR ~10pp
  lower) and is the operator's decision, not an audit side effect.
- PIT removes selection bias substantially but survivorship only partially — it can only rank
  tickers Yahoo still resolves (see item 2).
- Yahoo's EGX coverage may itself be non-random; thin names may be absent for reasons
  correlated with returns. Unmeasured.

---

### 2. Survivorship — systematic search · **PARTIAL (lower bound, not exhaustive)**

**What was done:** probed **242 candidate EGX symbols** against Yahoo (197 resolved), covering
the full current listing from a public EGX constituent list plus 23 hand-generated
delisting candidates, and classified every one by last print date.

**Found: 4 names that stopped trading before 2026-06** — the same four as last session:

| ticker | rows | first | last |
|---|---|---|---|
| ESRS | 2,775 | 2014-01-02 | 2025-03-13 |
| UASG | 2,673 | 2014-01-02 | 2024-10-23 |
| SUCE | 1,844 | 2014-01-02 | 2023-03-08 |
| OCIC | 126 | 2014-01-02 | 2015-10-14 |

**Was the search exhaustive? NO.** It was systematic over *Yahoo-resolvable symbols*, which is
not the same as systematic over *EGX delistings*. Specifically:

- **CBE and EGX portals are inaccessible to automated fetch** — both
  `cbe.org.eg/en/auctions/egp-t-bills` and the T-bill statistics page return
  *"The requested URL was rejected. Please consult with your administrator."*
- No FRA/EGX historical delisting bulletin was obtainable in machine-readable form.
- No academic EGX constituent dataset was located.
- A company that delisted **and** had its Yahoo symbol dropped or recycled is invisible to
  this method by construction.

**This remains a LOWER BOUND.** The count found this pass is 4; the true count is unknown and
larger. A correction of unknown size still sits on top of every number in this report.

---

### 3. T-bill auction yield series · **INCONCLUSIVE**

**Attempted, all failed:**

| source | result |
|---|---|
| CBE T-bill auctions page | HTTP rejected — *"The requested URL was rejected"* |
| CBE T-bill statistics page | HTTP rejected — same |
| World Bank API (`FR.INR.RINR`, `FR.INR.DPST`, `FR.INR.LEND`) | read timeout, 3/3 |
| Web search for a downloadable series | CEIC / Moody's / cbonds all identified as **subscription-gated**; MoF publishes individual auction *posts*, not a series |

**Consequence, stated plainly:** `data/rates.py` still reconstructs the CBE policy rate rather
than using fetched auction yields, and the "beats T-bills?" verdict still depends on that
reconstruction. **This check is INCONCLUSIVE, not passed.**

Mitigating fact worth noting *without* treating it as a pass: the verdict is `False` at both
ends of the plausible yield range in the corrected-universe run (strategy real CAGR −12.79%
is below any Egyptian T-bill yield on record), so the *direction* does not currently hinge on
the assumption — but the *margin* does, and the margin is what a future favourable result
would rest on.

---

### 4. Benchmark series · **PARTIAL — no EGX30/70, but a real external benchmark obtained**

**EGX30/EGX70 index history: NOT OBTAINABLE.** Probed `^CASE30`, `CASE30.CA`, `^EGX30`,
`EGX30.CA`, `EGX30`, `^EGX` — all return 0 or 1 rows. `config.py:231` remains accurate.

**One candidate was checked and REJECTED:** `CCSI` returns 1,216 rows and looked like a
plausible "Cairo & Alexandria Stock Exchange Index" ticker. It is **Consensus Cloud Solutions,
Inc.**, a US software company. Verified via `longName` before use. Had this been assumed
rather than checked, it would have silently corrupted every benchmark comparison.

**What WAS obtained: `EGPT` — VanEck Egypt Index ETF**, USD-denominated, 2,580 sessions
(2014-01-02 → 2024-04-04, fund closed). This is a genuine, externally-priced, investable
Egypt equity benchmark — arguably better than an index for this purpose, because it is net of
real-world costs and denominated in the currency the real-terms question is asked in.

**Benchmark comparison, same 1,830-session window, all USD:**

| series | CAGR | vol | Sharpe | MaxDD |
|---|---|---|---|---|
| EGPT (real Egypt equity ETF) | **−10.35%** | 30.3% | −0.341 | −71.8% |
| strategy USD — POINT-IN-TIME | **−10.28%** | 24.8% | −0.414 | **−58.5%** |
| strategy USD — BIASED | −3.99% | 22.7% | −0.176 | −52.2% |

**Reading, honestly:** on the corrected universe the strategy earned **the same USD return as
simply owning the Egypt ETF** (−10.28% vs −10.35%), with **18% less volatility** and **13pp
less drawdown**. That is a real risk-reduction benefit and *zero* return benefit — which is
precisely what a min-variance book with no alpha should look like.

*Sharpe caveat:* with negative returns, lower volatility makes Sharpe **more** negative, so the
strategy's worse Sharpe (−0.414 vs −0.341) is an artifact of the sign, not evidence it is
worse. The vol and drawdown columns are the meaningful ones here.

---

## Collateral finding: I broke the live path and fixed it

The 147-ticker research ingest extended the price panel to 2026-08-05, a session for which the
51 live names had no data. `generate_trade_list` then failed with *"only 0 eligible names"* —
**it refused to emit a book rather than emit a degraded one**, which is the gate working
correctly.

Fixed by refreshing live prices and rebuilding `tradable_universe` **restricted to the 51 live
names**. The 147 research tickers were deliberately *not* made tradable: widening the live
universe from 51 to 198 is a strategy change, not a data fix.

```
LIVE OK: as-of 2026-08-05  regime BULL  exposure 1.0  positions 27
live tradable tickers: 51  (unchanged)
```

---

## Constraint #8 — did the evidence for an edge get better, worse, or stay at zero?

**Worse. Materially and unambiguously worse.**

Last session the honest estimate of system alpha was −0.86% (t = −0.23) — indistinguishable
from zero. This session, correcting the selection bias that the previous audit had flagged as
its overriding red, the estimate is **−2.46% (t = −0.61)** on a universe that is no longer
chosen with knowledge of the answer, and core Sharpe fell from 1.463 to **0.755**.

Separately, the first *external* benchmark ever applied to this system shows it earning the
same USD return as a passive Egypt ETF over the same window. The system's measurable
contribution is **lower volatility and shallower drawdowns, at zero return benefit.**

That is a coherent and defensible thing for a min-variance book to do. It is not an edge, and
the evidence that it might be one is now weaker than at any prior audit.

---

## Go-live gate — unchanged

Tier 3 has not been touched, so no forward data has moved. Raw output:

```
============================================================
GO-LIVE GATE (forward paper record vs pre-committed criteria)
============================================================
  [FAIL] forward duration           20 sessions (need >= 189)
  [FAIL] survived a BEAR regime     0 BEAR sessions observed live (need >= 20)
  [FAIL] forward Sharpe             n/a (need >= 0.8)
  [FAIL] forward max drawdown       n/a (must be >= -0.25)
  [FAIL] positive REAL (USD) CAGR   n/a (need >= 0.0)
------------------------------------------------------------
  RESULT: NOT READY -- 5 criterion/criteria unmet. Keep paper trading.
  This gate reads the forward record only. It never reads the backtest.
============================================================
```

**No safety threshold was altered.** `GO_LIVE_MIN_PAPER_DAYS`, `GO_LIVE_MIN_BEAR_DAYS`,
`GO_LIVE_MIN_SHARPE`, `GO_LIVE_MAX_DRAWDOWN`, `GO_LIVE_MIN_REAL_CAGR`, `LIVE_MIN_VAR_ONLY` and
every kill-switch constant are untouched.

---

## Tier 1 status summary

| item | status | evidence |
|---|---|---|
| 1. Point-in-time universe | **DONE** | `data/pit_universe.py`; 147 names, 1335 entries/exits, 0 always-present; core CAGR −10.52pp, alpha +7.54% → −2.46% |
| 2. Survivorship systematic search | **PARTIAL** | 242 symbols probed, 4 delistings found; CBE/EGX/FRA archives HTTP-rejected; explicitly a lower bound |
| 3. T-bill auction yields | **INCONCLUSIVE** | 4 sources attempted, all blocked/timed out/paywalled; `data/rates.py` still a reconstruction |
| 4. Benchmark series | **PARTIAL** | no EGX30/70 obtainable (6 symbols probed); `CCSI` checked and rejected as a US company; EGPT ETF obtained and used |

**Not started:** Tier 2 (items 5–7), Tier 3 (8–10), Tier 4 (11–13), Tier 5 (14–15).


---

# TIER 2 - Statistical robustness of the regime model

*(appended 2026-08-08; 429 tests passing)*

### 5. HMM state-count sensitivity, WALK-FORWARD - **DONE**

**Changed:** `analysis/regime_validation.py:walkforward_bear_calls` + `walkforward_state_sweep`.

**Why a new function rather than the live one.** `models/regime_labeler.label_states`
deliberately RAISES for K != 4 - it refuses to guess a BULL/BEAR/HIGH_VOL/RECOVERY mapping for
a different state count. That is correct design, and it makes the live function unusable for a
sweep. `walkforward_bear_calls` mirrors `compute_regime_series(mode="walkforward")` exactly -
expanding annual refit, embargo before each boundary, frozen params, forward-only filtered
decode - and identifies BEAR by the SAME rule the labeler uses, `argmin(mean 20d return)`.

**Proof the mirror is faithful:** at K=4 it produces **401 BEAR days, identical to the live
pipeline**.

```
states  bear days     diff   t(raw)   t(adj)  verdict
     2       1433   -0.18%   -0.55   -0.12   worse, not significant
     3        640   -0.93%   -3.08   -0.67   worse, not significant
     4        401   -2.13%   -7.17   -1.57   worse, not significant  <-- live
     5        473   -1.69%   -4.89   -1.07   worse, not significant
     6        504   -0.34%   -1.15   -0.25   worse, not significant
```

**Findings.** (a) **4 states is defensible** - best of the grid on both effect size and
adjusted t. (b) This **REVERSES the in-sample sweep**, which implied 3 states was much better;
`robustness_suite.regime_count_sweep`'s own docstring called that a probable look-ahead
artifact and it was right. (c) **No state count produces a significant signal.** The live
setting is the best of five failing options.

**Still unresolved:** K=2 labels 1,433 of 1,889 days BEAR (76%) - a permanent bearish stance,
not regime detection. Any Sharpe comparison across state counts must be read with that in mind.

### 6. Embargo justification - **DONE, first evidence either way**

**Changed:** `analysis/regime_validation.py:embargo_sweep`. Scores each embargo on the only
thing a regime call is for: does a causally-made BEAR call precede bad returns?

```
embargo   BEAR   BULL  HI_VOL  RECOV     diff   t(adj)  verdict
      0    389    167     859    474    +1.32%   +0.86   NOT worse
     25    468    270     577    574    -1.47%   -0.99   worse, not significant
     50    401    331     800    357    -2.13%   -1.57   worse, not significant  <-- live
    100    498    271     507    419    -1.47%   -1.04   worse, not significant
```

**Without the embargo the signal points the WRONG WAY** - BEAR calls precede *higher* returns
(+1.32%). Every embargoed setting flips it to the correct direction and **50 days is the
strongest of the four.** The prior audit reported the BULL 166->331 shift and declined to call
it an improvement because nothing had been tested. It has now been tested: **the embargo is
justified, not merely different.**

**Still unresolved:** none of the four is statistically significant - this is directional
evidence from four grid points. 50 was already the live setting before this test, so this is a
VALIDATION that could have failed, not a selection.

### 7. Cross-market validation - **PARTIAL / IN PROGRESS**

The first run reported *"only 0 observations"* for Turkey, Brazil, South Africa and India. That
was a **FALSE NEGATIVE**: Yahoo rate-limits under sustained use and returns an empty frame
rather than raising, so a throttle was being reported as a data limitation. Retry logic added
(`cross_market`, 4 attempts with backoff); all five symbols verified to fetch (TUR/EWZ/EZA
4,173 rows each). Re-run in progress at time of writing.

**One result did complete, and it is unfavourable:**
`EGYPT (EGPT): 72 bear days, 2,487 other, diff +6.09%, t(adj) +1.68 - NOT worse.`
On the reduced feature set, the method's BEAR calls on the HOME market preceded *higher*
returns. Conclusion withheld until the other four markets land.

---

# TIER 3 - Forward/live evidence

### 8. Slippage measurement - **INCONCLUSIVE (8 measured fills; floor 20)**

**Changed:** `execution_quality` gained a `source` column; `summary()` now excludes
reconstructed fills.

**Why.** 16 fills were recorded but **8 had prices DERIVED from reported position values**
during the 2026-08-04 scale-in, not read from contract notes. Their "shortfall" partly
measures that reconstruction. Mixing them with real fills inflated the sample and combined two
different quantities.

```
SLIPPAGE - modelled 0.10% vs measured +2.745% over 8 fill(s)
  shrunk estimate 0.856% (29% weight on data)
  8 fills is below the 20-fill floor - indicative, not settled
```

**Explicitly INCONCLUSIVE.** Also an upper bound: shortfall is scored against the as-of close,
so trading a session later counts price drift as slippage.

### 9. Kill-switch / drift-alert live testing - **REPLAY-VERIFIED, not live-verified**

**Changed:** new `ops/control_replay.py`.

**Why.** Neither control has ever seen an adverse day - worst paper session -0.58%, worst book
day -0.02%, zero days below -1%. Every existing test feeds them synthetic numbers chosen to
trip them. So real historical EGX moves are replayed through the same functions the live path
calls.

```
MODEL KILL SWITCH (daily -7%, weekly -12%) over 2928 sessions
  trips on real history: 36  -> REPLAY-VERIFIED (fires)
  worst: 2020-03-19 weekly -14.94% | 2026-06-23 daily -9.92%
ACCOUNT DRAWDOWN GUARD (worst real drawdown -38.1%)
    warn -10%: 992 days (first 2014-12-14)
   block -15%: 684 days (first 2014-12-17)
 flatten -25%: 321 days (first 2015-08-23)  -> REPLAY-VERIFIED (escalates)
DRIFT: execution_drift INCONCLUSIVE (8 fills) | cost_model_drift ok
```

**REPLAY-VERIFIED means the logic reacts correctly to real historical magnitudes. It does NOT
mean the control has fired in production**, because production has never had a bad enough day.
The -9.92% day was 2026-06-23, two weeks before the forward record began.

### 10. Paper duration and bear-day exposure - **UNCHANGED, and the gate was greenwashing**

**A real defect found.** The gate was printing:

```
[PASS] forward Sharpe        7.673
[PASS] forward max drawdown  -0.0058
```

Both computed from 21 sessions that the same gate declares `sufficient: False`. A Sharpe of
7.67 is three weeks of a rising market annualised; "no drawdown yet" is not drawdown control.

**Changed:** `ops/go_live_gate._crit` gained `conclusive`. A criterion passes only if
**satisfied AND conclusive**, using the SAME `GO_LIVE_MIN_PAPER_DAYS` threshold - no new or
looser bar. Three tests in `tests/test_production_hardening.py` pin it, including one proving
the rule lifts once the record is long enough.

**This TIGHTENS the gate. No threshold was relaxed. INCONCLUSIVE blocks go-live exactly as a
fail does.**

```
[FAIL]   forward duration           21 sessions (need >= 189)
[FAIL]   survived a BEAR regime     0 BEAR sessions observed live (need >= 20)
[INCONC] forward Sharpe             7.67 (need >= 0.8) - from only 21 sessions, not meaningful
[INCONC] forward max drawdown       -0.58% (must be >= -25%) - only 21 sessions, no adverse day yet
[INCONC] positive REAL (USD) CAGR   n/a (need >= 0.0)
------------------------------------------------------------
  RESULT: NOT READY -- 2 failed, 3 inconclusive (insufficient record). Keep paper trading.
  INCONC = not enough forward data to judge. It is not a pass.
```

| | now | required | gap |
|---|---|---|---|
| forward sessions | 21 | 189 | 168 = **~8 months** |
| live BEAR days | 0 | 20 | cannot be scheduled |

Forward cumulative +8.39%.

---

## Constraint #8 - did the evidence for an edge change in Tier 2-3?

**It stayed at zero, and one prop was removed.**

Tier 2 did not find an edge: no state count produces a significant BEAR signal (best |t(adj)|
= 1.57, at the live setting), and on the reduced-feature cross-market test the home market's
BEAR calls preceded *higher* returns.

Tier 2 did produce one genuinely positive engineering result - **the 50-day embargo is now
evidenced rather than assumed**, having flipped the signal from the wrong direction to the
right one. That improves the *model*, not the *edge*: the corrected signal is still not
statistically distinguishable from noise.

Tier 3 removed a false positive: two gate criteria that read PASS now correctly read
INCONCLUSIVE. **The apparent evidence went down because it was never real.**

## Status

| item | status |
|---|---|
| 5. Walk-forward state sweep | **DONE** |
| 6. Embargo justification | **DONE** |
| 7. Cross-market validation | **PARTIAL** - retry running; EGPT control unfavourable |
| 8. Slippage measurement | **INCONCLUSIVE** (8 measured fills) |
| 9. Control live testing | **REPLAY-VERIFIED** (never fired in production) |
| 10. Paper duration / bear days | **UNCHANGED**; gate defect fixed |

**Not started:** Tier 4 (11-13), Tier 5 (14-15).


---

# TIER 2 item 7 - CROSS-MARKET VALIDATION (completed)

```
market            bear d  other d     diff   t(adj)  verdict
EGYPT (EGPT)      bear=12 - INCONCLUSIVE (below the 20-observation floor)
TURKEY (TUR)         154      654   +8.58%   +1.41   NOT worse
BRAZIL (EWZ)         133      675   +3.22%   +0.49   NOT worse
S.AFRICA (EZA)        94      714   -1.96%   -0.26   worse, not significant
INDIA (INDA)         240      462   -0.05%   -0.02   worse, not significant
```

**The method does not generalise.** Two of four markets have BEAR calls preceding *higher*
returns. Nothing is significant in either direction, so there is no evidence the HMM regime
methodology carries generalisable skill - and the test is equally underpowered to prove it
does not.

**COMPUTE CONCESSION, DECLARED.** A single from-scratch EM fit on ~4,000 observations exceeds
10 minutes, so the 15-fold annual roll used on EGX is infeasible across five markets. Each
market uses ONE expanding split, ONE EM restart, weekly-thinned to <=1,500 rows. That makes
each estimate noisier. It does NOT introduce look-ahead: the fit still sees only the training
prefix and the decode is still forward-only. A generalisation question survives a noisy
estimator; a fine-grained ranking of markets would not, and none is claimed.

The EGYPT control returned INCONCLUSIVE (12 BEAR days after thinning, below the floor) rather
than a number - the thinning that made the test tractable also made the home control unusable.

---

# TIER 4 - Structural / data gaps

*(2026-08-09; 451 tests passing)*

### 11. Point-in-time fundamental data - **CONCLUSIVE: not viable. Question closed.**

Fundamentals DO exist free via yfinance for EGX names (38 income-statement line items, 46
balance-sheet items, 5 annual + 7 quarterly periods for COMI). Tested empirically rather than
inferred from vendor marketing. They fail on BOTH requirements:

| requirement | measured reality |
|---|---|
| depth back to HISTORY_START_DATE (2015) | **4-5 annual periods, earliest 2021-12-31** - over half the backtest window has no data |
| point-in-time dating | **dated by FISCAL PERIOD END, not publication.** A 2025-12-31 annual is not public until ~March 2026 |
| filing/publication date exposed | **none.** Only `isEarningsDateEstimate` and `firstTradeDateMilliseconds` exist in the API |
| restatement vintages | **none.** Latest restated figures only - not what was knowable at the time |

Using this as-is would inject roughly three months of look-ahead per observation, on a series
that only covers the back half of the sample. A genuine PIT dataset would require scraping the
EGX disclosure portal WITH filing dates, and EGX/CBE portals are HTTP-blocked (established in
Tier 1, item 3).

**Documented conclusively: not worth re-checking without a paid vendor.** This closes the loop
the instruction asked to close.

### 12. Live/intraday data - **SKIPPED as instructed**

No genuinely new option surfaced. The prior conclusion ("no retail-viable real-time EGX feed
exists") stands unchallenged.

### 13. `data/microstructure.py` - **THE PREMISE WAS WRONG. Do not delete: it is load-bearing.**

The instruction described this as dead code, to be deleted once real fills confirmed it added
nothing. **It is not dead code.** `MicrostructureAnalyzer` computes `o5_avg_amihud`, one of the
six observation columns feeding the LIVE HMM regime model:

```
o1_market_return_20d   o2_realized_vol_20d   o3_breadth
o4_usd_egp_change_20d  o5_avg_amihud         o6_realized_vol_60d
```

It is additionally imported by `models/factor_model.py` and `models/factor_returns.py`.

The "Amihud explains only 4-9% of cost variance" finding that motivated the deletion plan was
about **cost modelling**. This is **regime detection** - a different use, and carrying the
conclusion across would have been a category error.

**ABLATION (replace o5 with a constant, keep dimensionality, re-run the causal walk-forward):**

```
WITH amihud      bear_days= 401  diff= -2.13%  t(adj)=-1.57  significant=False
WITHOUT amihud   bear_days= 401  diff= +0.42%  t(adj)=+0.25  significant=False
label agreement: 74.6% over 1889 shared sessions
```

**Removing the feature FLIPS THE SIGNAL DIRECTION** - from -2.13% (BEAR calls precede worse
returns, the intended behaviour) to +0.42% (BEAR calls precede *better* returns), and changes
a quarter of all labels. Deleting `microstructure.py` would not merely break an import; it
would invert the regime signal.

**What this does NOT establish:** neither version is statistically significant. Amihud turns an
insignificant-and-wrongly-directed signal into an insignificant-and-correctly-directed one.
That is a real contribution to the model and is not evidence of an edge.

Note the parallel with the embargo result (Tier 2, item 6): both the 50-day embargo and the
Amihud feature are *individually necessary* for the BEAR calls to point the right way at all,
and *neither* makes them significant. The signal appears to sit right at the boundary where
modelling choices determine its sign - which is itself a statement about how weak it is.

**No file was deleted and nothing was re-wired.**

### Edge-case hardening (requested alongside Tier 4)

New `tests/test_tier_modules.py` - **22 tests**, covering every module added during this
remediation at the points where each could return something plausible-looking and wrong:

- empty panels, single observations, constant/zero-variance series
- `pit_universe`: membership must be backward-looking only; a static universe must be FLAGGED
  rather than pass silently
- `score_calls`: refuses samples under 20; overlap correction pinned to `t_raw / sqrt(h)`
- `_regime_series`: uses the `date` COLUMN, not position (the misalignment bug caught by hand)
- `_bear_state`: lowest mean return, never lowest state index (HMM numbering is arbitrary)
- `control_replay`: thresholds read from config, never redefined; escalation ladder ordered;
  a calm series must report "never fires" rather than pass
- `factor_exposure`: a pure market replica must show alpha 0 and beta 1 - if that fails, every
  alpha figure in these audits is suspect
- `alpha_decay`: refuses to fit a half-life to insignificant ICs
- `monte_carlo`: percentiles ordered, resampled values never invented
- removed cost constants must RAISE, not return a number

Full suite: **451 passing** (was 429).

---

## Constraint #8 - did the evidence for an edge change in Tier 4?

**It stayed at zero, and the model looks more fragile than before.**

Tier 4 found no new evidence for or against an edge - it was a data-availability and dead-code
tier. But two results from this session compound:

1. Cross-market: the regime method shows **no generalisable skill**; on two of four foreign
   markets its BEAR calls precede *higher* returns.
2. Ablation: the EGX signal's *direction* depends on a single illiquidity feature, exactly as
   it also depends on the 50-day embargo.

A signal whose sign is determined by individual modelling choices, which fails to replicate on
four other markets, and which is not statistically significant under any configuration tested,
is best described as **not yet distinguishable from noise** - which is where the previous three
audits also left it.

## Tier 4 status

| item | status |
|---|---|
| 11. Fundamental data | **CONCLUSIVE - not viable** (no PIT dating, no pre-2021 depth) |
| 12. Live/intraday data | **SKIPPED** as instructed; nothing new surfaced |
| 13. microstructure.py | **PREMISE CORRECTED** - it is live and load-bearing; ablation flips the signal; nothing deleted |
| edge-case hardening | **DONE** - 22 new tests, suite 429 -> 451 |

**Not started:** Tier 5 (14-15).


---

# TIER 5 - Housekeeping

*(2026-08-09)*

Tier 5 asks for VERIFICATION, not construction: keep the unvalidated layers off the live path,
and re-verify only if the underlying data situation changed. It did change this session - the
price panel went from 51 to 198 tickers - so the re-verification was not a formality.

### 14. Sentiment (Layer 4) and TFT (Layer 6) - **VERIFIED OFF THE LIVE PATH; walls unchanged**

**Sentiment, measured now:**

```
signals rows        : 64,080
non-zero sentiment  : 0  (0.00%)
news articles       : 517, spanning 2026-06-11 -> 2026-07-10  (ONE MONTH)
daily_sentiment rows: 90
```

The wall is the **news corpus, not the ticker count**: 517 articles over one month against an
11-year backtest. Widening the universe 51 -> 198 does nothing for it. Combined with the
model's measured F1 of 0.33, this remains structural. **No attempt was made to "fix" it**, per
the instruction.

**TFT:** checkpoint exists at `config.TFT_MODEL_CHECKPOINT_PATH`, and so does the sentiment
checkpoint. **Neither is consumed by the live path.** That matters more than it sounds -
isolation here is NOT guaranteed by the models being absent, only by nothing importing them,
which is exactly the property that decays silently in a refactor.

**Transitive import check across the whole live path**
(`ops.daily_run`, `ops.trade_list`, `portfolio.system`, `ops.order_ticket`,
`ops.portfolio_tracker`):

```
BANNED layers reachable from the live path: NONE
```

### 15. Causal / Double-ML - **VERIFIED ISOLATED**

```
causal_signal populated rows: 63,194 of 64,080
pooled (leaky) function still present     : True
walk-forward (fixed) function present     : True
```

The signal is computed and stored, and **nothing on the live path reads it**. The pooled-theta
function that carried the target-leakage finding still exists alongside the walk-forward
version; that is acceptable *only* because the entire layer is off the live path. Its docstring
already records the honest detail that with `theta_sent == 0` the causal signal reduces to a
cross-sectional momentum z-score, making the leak fix currently a no-op.

**No further work done**, per the instruction.

### What was ADDED: a tripwire, because verification decays

`tests/test_live_path_isolation.py` - **5 tests**. Verification that is performed once and
written in a report is worth very little; the same verification as an assertion runs on every
future commit.

- **the tripwire**: no module matching `sentiment|tft|causal|double_ml|alpha_score|
  fama_macbeth|rank_persistence` may be reachable from any live entrypoint. If someone wires
  one in, the suite fails and explains why.
- `LIVE_MIN_VAR_ONLY` is armed, AND `generate_trade_list` provably raises when it is False -
  testing the flag's *effect*, not just its value.
- **sentiment coverage is asserted to stay below 5%.** This is the "re-verify only if the data
  situation changes" instruction turned into something that checks itself. When coverage
  becomes non-trivial the test FAILS - and its message says that is the signal to run the
  out-of-sample baseline comparison, *not* to wire anything in.
- checkpoints existing on disk must never be mistaken for the models being in use.

Re-enabling any of these layers is now a deliberate act that must delete a test.

---

# FINAL DELIVERABLE

## Tier-by-tier status

| # | item | status | evidence |
|---|---|---|---|
| 1 | Point-in-time universe | **DONE** | `data/pit_universe.py`; 147 names, 1335 entries/exits, 0 always-present; core CAGR **-10.52pp**, alpha **+7.54% -> -2.46%** |
| 2 | Survivorship search | **PARTIAL** | 242 symbols probed, 4 delistings; CBE/EGX/FRA archives HTTP-rejected; explicitly a LOWER BOUND |
| 3 | T-bill auction yields | **INCONCLUSIVE** | 4 sources attempted, all blocked/timed out/paywalled; `data/rates.py` still a reconstruction |
| 4 | Benchmark series | **PARTIAL** | no EGX30/70 (6 symbols probed); `CCSI` checked and REJECTED as a US company; EGPT ETF obtained and used |
| 5 | Walk-forward state sweep | **DONE** | mirror reproduces the live 401 BEAR days at K=4; 4 states best of grid; **none significant** |
| 6 | Embargo justification | **DONE** | embargo=0 points the WRONG way (+1.32%); 50d strongest; first evidence either way |
| 7 | Cross-market validation | **DONE** | 2 of 4 markets point the wrong way; **no generalisable skill**; compute concession declared |
| 8 | Slippage measurement | **INCONCLUSIVE** | 8 MEASURED fills (8 more were reconstructed and are now excluded); floor is 20 |
| 9 | Control live testing | **REPLAY-VERIFIED** | 36 kill-switch trips on real history; **never fired in production** |
| 10 | Paper duration / bear days | **UNCHANGED** | 21 of 189 sessions, 0 of 20 BEAR; gate greenwashing defect FIXED |
| 11 | Fundamental data | **CONCLUSIVE - not viable** | earliest annual 2021-12-31; no publication dates; no vintages |
| 12 | Live/intraday data | **SKIPPED** | as instructed; nothing new surfaced |
| 13 | microstructure.py | **PREMISE CORRECTED** | it is LIVE and load-bearing; ablation **flips the signal sign**; nothing deleted |
| 14 | Sentiment / TFT | **VERIFIED OFF LIVE PATH** | 0% coverage, 517 articles/1 month; tripwire test added |
| 15 | Causal / Double-ML | **VERIFIED ISOLATED** | computed and stored, never read by the live path |

## Constraint #8 - did the system's evidenced trading edge get better, worse, or stay at zero?

**Worse, and then it stayed at zero.**

The one number that moved is the one that matters: correcting the selection bias the previous
audit flagged as its overriding red took **system alpha from +7.54% (t = 1.93) to -2.46%
(t = -0.61)** and **core Sharpe from 1.463 to 0.755**. That is not a new discovery about the
market; it is the removal of a distortion that had been inflating every prior figure.

Everything else confirmed zero:
- no state count produces a significant regime signal (best |t(adj)| = 1.57)
- the method does not generalise - 2 of 4 foreign markets have BEAR calls preceding *higher*
  returns
- the EGX signal's *sign* depends on the 50-day embargo AND on one illiquidity feature; remove
  either and it points the wrong way
- against the first external benchmark ever applied (EGPT), the strategy earned the **same USD
  return as passively owning Egypt**, with less volatility and shallower drawdowns

**The strategy does not beat T-bills in real terms, and its alpha is not distinguishable from
zero** - with the survivorship caveat attached, as required: four delisted names were recovered
and an unknown number remain missing, so the corrected figures are still an UPPER BOUND.

## Go-live gate - unchanged, and now stricter

Tier 3 moved no forward data. The gate was found reporting PASS on statistics derived from a
record it had itself declared insufficient; those now read INCONCLUSIVE. **No threshold was
relaxed** - `GO_LIVE_MIN_PAPER_DAYS`, `GO_LIVE_MIN_BEAR_DAYS`, `GO_LIVE_MIN_SHARPE`,
`GO_LIVE_MAX_DRAWDOWN`, `GO_LIVE_MIN_REAL_CAGR`, `LIVE_MIN_VAR_ONLY` and every kill-switch
constant are untouched.

```
============================================================
GO-LIVE GATE (forward paper record vs pre-committed criteria)
============================================================
  [FAIL] forward duration           21 sessions (need >= 189)
  [FAIL] survived a BEAR regime     0 BEAR sessions observed live (need >= 20)
  [INCONC] forward Sharpe             7.67 (need >= 0.8) - from only 21 sessions, not meaningful
  [INCONC] forward max drawdown       -0.58% (must be >= -25%) - only 21 sessions, no adverse day yet
  [INCONC] positive REAL (USD) CAGR   n/a (need >= 0.0)
------------------------------------------------------------
  RESULT: NOT READY -- 2 failed, 3 inconclusive (insufficient record). Keep paper trading.
  INCONC = not enough forward data to judge. It is not a pass.
  This gate reads the forward record only. It never reads the backtest.
============================================================
```

## Damage I caused and repaired, stated plainly

1. **The live book silently widened from 27 to 73 positions.** Ingesting 147 research tickers
   made `build_tradable_universe` mark all 198 tradable. I corrected it by hand in Tier 1 and a
   later rebuild undid the correction, because a manual DELETE is not a policy. The live
   universe is now DECLARED in `data.universe.live_universe_tickers()` and enforced by default,
   with three tests pinning it.
2. **An empty panel crashed** with `ValueError: No objects to concatenate`. Now returns an empty
   universe with a warning.
3. **A false INCONCLUSIVE was nearly reported as a finding** - Yahoo rate-limiting returned
   empty frames for four markets, which read as "no data available". Retry logic added.
4. **The first cross-market design was computationally infeasible** (>10 min per EM fit). The
   simplification is declared in the report rather than buried.

## Test suite

**429 -> 459 tests**, all passing. New this session: 22 edge-case tests across every module
added during the remediation, 3 live-universe restriction tests, 5 live-path isolation
tripwires.
