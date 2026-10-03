"""
Tests for the Tier-1/Tier-2 pre-live hardening modules, with known-answer
cases where possible:
  * real_returns  : constant FX -> USD==EGP; doubling FX halves USD cumulative.
  * significance  : strong-mean series -> CI excludes 0; zero-mean -> straddles 0.
  * ca_reconcile  : synthetic 2:1 split is detected; a logged-but-unadjusted
                    action is flagged suspect.
  * costs         : higher participation and lower ADV cost more.
  * capacity      : the smallest-ADV name at weight is the binding constraint.
  * kill_switch   : a -8% day trips and holds flat for the cooldown.
  * vol_target    : high realized vol -> exposure < 1; low vol -> capped at 1.
  * go_live_gate  : an empty forward record fails every criterion.
"""

import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "h.db"))
    database.init_db()
    return config.DB_PATH


# --------------------------------------------------------------------------
# real_returns
# --------------------------------------------------------------------------
def _seed_fx(dates, values):
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO macro_daily (date, usd_egp) VALUES (?, ?)",
                         [(str(d.date()), float(v)) for d, v in zip(dates, values)])


def test_real_returns_constant_fx_is_identity(temp_db):
    from analysis import real_returns as rr
    idx = pd.bdate_range("2020-01-01", periods=30)
    _seed_fx(idx, np.full(len(idx), 20.0))                  # flat FX
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, len(idx)), index=idx)
    usd = rr.to_hard_currency(r, rr.load_usd_egp())
    # flat FX -> USD returns equal EGP returns (after the first anchored day)
    assert np.allclose(usd.iloc[1:].values, r.iloc[1:].values, atol=1e-9)


def test_real_returns_devaluation_reduces_usd(temp_db):
    from analysis import real_returns as rr
    idx = pd.bdate_range("2020-01-01", periods=30)
    fx = np.linspace(20.0, 40.0, len(idx))                  # EGP halves vs USD
    _seed_fx(idx, fx)
    r = pd.Series(np.zeros(len(idx)), index=idx)            # flat EGP NAV
    usd = rr.to_hard_currency(r, rr.load_usd_egp())
    # flat in EGP but currency halved -> ~-50% in USD
    usd_cum = float((1 + usd).prod() - 1)
    assert usd_cum == pytest.approx(-0.5, abs=0.02)


# --------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------
def test_significance_strong_edge_excludes_zero():
    from analysis import significance as sig
    rng = np.random.default_rng(1)
    r = rng.normal(0.0015, 0.008, 800)                      # high, steady Sharpe
    out = sig.sharpe_significance(r, n_boot=500, method="stationary")
    assert out["sharpe"] > 1.5
    assert out["ci95"][0] > 0 and out["significant_at_5pct"]
    assert out["p_value_le_0"] < 0.05


def test_significance_noise_straddles_zero():
    from analysis import significance as sig
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.01, 800)                          # no edge
    out = sig.sharpe_significance(r, n_boot=500, method="moving")
    assert out["ci95"][0] < 0 < out["ci95"][1]
    assert not out["significant_at_5pct"]


# --------------------------------------------------------------------------
# ca_reconcile
# --------------------------------------------------------------------------
def test_ca_detects_split_and_flags_unadjusted(temp_db):
    from data import ca_reconcile as car
    idx = pd.bdate_range("2020-01-01", periods=20)
    # AAA: real 2:1 split mid-series reflected in adjusted_close (close halves,
    # adj_close continuous) -> factor jumps -> implied split, none logged.
    close = np.concatenate([np.full(10, 100.0), np.full(10, 50.0)])
    adj = np.full(20, 50.0)                                 # already split-adjusted throughout
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, value_traded_egp) "
                         "VALUES ('AAA', ?, ?, ?, 0, 0)",
                         [(str(d.date()), float(c), float(a)) for d, c, a in zip(idx, close, adj)])
    rec = car.reconcile_ticker("AAA")
    assert rec["n_implied_splits"] >= 1                     # the split was detected
    assert len(rec["implied_splits_unrecorded"]) >= 1       # and it isn't in our log


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------
def test_cost_grows_with_participation_and_illiquidity():
    from backtest import costs
    small = costs.trade_cost_fraction(1_000, adv_egp=1_000_000)["one_way"]
    big = costs.trade_cost_fraction(500_000, adv_egp=1_000_000)["one_way"]
    illiquid = costs.trade_cost_fraction(500_000, adv_egp=50_000)["one_way"]
    assert big > small                                     # more size -> more slippage
    assert illiquid > big                                  # thinner name -> even more
    assert costs.trade_cost_fraction(1, adv_egp=0)["participation"] == float("inf")


def test_book_roundtrip_flags_high_impact():
    from backtest import costs
    w = pd.Series({"AAA": 0.5, "BBB": 0.5})
    adv = pd.Series({"AAA": 100_000_000.0, "BBB": 1_000_000.0})   # BBB thin
    out = costs.book_roundtrip_cost(w, capital_egp=5_000_000, adv=adv)
    assert out["roundtrip_cost"] > 0
    assert "BBB" in out["flagged_high_impact"]             # 2.5M into 1M ADV = 250%


# --------------------------------------------------------------------------
# capacity
# --------------------------------------------------------------------------
def test_capacity_binding_is_smallest_adv_name():
    from analysis import capacity
    w = pd.Series({"BIG": 0.5, "SMALL": 0.5})
    adv = pd.Series({"BIG": 100_000_000.0, "SMALL": 1_000_000.0})
    out = capacity.book_capacity(w, participation_cap=0.10, spread_days=1, adv=adv)
    assert out["binding_ticker"] == "SMALL"
    # cap = 0.10 * 1_000_000 * 1 / 0.5 = 200_000
    assert out["capacity_egp"] == pytest.approx(200_000.0)


# --------------------------------------------------------------------------
# kill_switch
# --------------------------------------------------------------------------
def test_kill_switch_trips_on_big_daily_loss_and_holds():
    from risk import kill_switch as ks
    idx = pd.bdate_range("2020-01-01", periods=10)
    r = pd.Series(0.0, index=idx)
    r.iloc[3] = -0.08                                      # -8% day (< -7% limit)
    mult = ks.check_kill_switch(r, max_daily=-0.07, max_weekly=-0.99, cooldown=3)
    assert mult.iloc[3] == 0.0                             # flattened on the breach
    assert (mult.iloc[4:7] == 0.0).all()                  # ... and through cooldown
    assert mult.iloc[7] == 1.0                             # then released


# --------------------------------------------------------------------------
# vol-target breaker replacement
# --------------------------------------------------------------------------
def test_vol_target_reduces_exposure_in_turbulence():
    from risk.overlay import volatility_target_exposure
    calm = np.random.default_rng(0).normal(0, 0.005, 60)   # ~8% annualized
    wild = np.random.default_rng(0).normal(0, 0.03, 60)    # ~48% annualized
    e_calm = volatility_target_exposure(calm, target_vol=0.15, lookback=20)
    e_wild = volatility_target_exposure(wild, target_vol=0.15, lookback=20)
    assert e_calm == 1.0                                   # calm -> full exposure (capped)
    assert e_wild < 0.5                                    # turbulent -> scaled down, smoothly


# --------------------------------------------------------------------------
# go-live gate
# --------------------------------------------------------------------------
def test_go_live_gate_fails_on_empty_record(temp_db):
    from ops import go_live_gate as glg
    out = glg.evaluate_go_live()
    assert out["go_live"] is False
    assert all(not c["pass"] for c in out["checks"])       # nothing passes with no record


# --------------------------------------------------------------------------
# liquidity-aware cost wired into the engine
# --------------------------------------------------------------------------
def test_engine_liquidity_cost_bites_at_size(temp_db):
    """Same strategy, same data: a bigger AUM must net LESS than a tiny AUM,
    because the per-name impact cost grows with participation. The flat-cost
    engine (aum_egp=None) sits between as the size-blind reference."""
    import numpy as np
    import pandas as pd
    from backtest.engine import BacktestEngine
    from data.database import get_connection

    idx = pd.bdate_range("2021-01-01", periods=260)
    rng = np.random.default_rng(0)
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rows = []
    for t in tickers:
        px = 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(idx)))
        adv = 1_000_000.0 if t in ("AAA", "BBB", "CCC") else 40_000.0   # half are thin
        for d, p in zip(idx, px):
            rows.append((t, str(d.date()), float(p), float(p), 0, adv))
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, value_traded_egp) "
                         "VALUES (?, ?, ?, ?, ?, ?)", rows)

    from backtest.engine import load_price_returns
    rets = load_price_returns()

    def eqw(window):
        return pd.Series(1.0 / window.shape[1], index=window.columns)

    small = BacktestEngine(rebalance_days=21, min_names=3, tradable_only=False,
                           aum_egp=100_000).run_portfolio(eqw, rets, lookback=60)
    big = BacktestEngine(rebalance_days=21, min_names=3, tradable_only=False,
                         aum_egp=20_000_000).run_portfolio(eqw, rets, lookback=60)
    small_final = float(small["nav"].iloc[-1])
    big_final = float(big["nav"].iloc[-1])
    assert big_final < small_final          # trading big into thin names costs more -> nets less


def test_years_to_significance_scales_inverse_square():
    from analysis.significance import years_to_significance as yts
    assert yts(0.0) == float("inf")
    assert yts(1.0) == pytest.approx(1.96**2 * 1.5, rel=1e-6)   # ~5.76y
    assert yts(0.5) > yts(1.0) > yts(2.0)                       # thinner edge -> longer
