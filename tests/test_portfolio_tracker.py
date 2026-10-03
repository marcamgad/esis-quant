"""
Tests for the interactive portfolio tracker + reconciler. Covers recording
trades (cash/holdings/ledger), the sell-more-than-held guard, weight
valuation, and the reconcile action logic (BUY/ADD/HOLD/TRIM/SELL, plus the
BEAR-regime de-risk path).
"""

import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from ops import portfolio_tracker as pt


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "trk.db"))
    database.init_db()
    return config.DB_PATH


def _price(ticker, price, date="2026-07-16"):
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, value_traded_egp) "
                     "VALUES (?, ?, ?, ?, 0, 0)", (ticker, date, price, price))


def _target(regime, weights: dict, exposure=1.0, run="r1"):
    now = pd.Timestamp.now().isoformat()
    rows = [(run, now, "2026-07-16", regime, exposure, t, w) for t, w in weights.items()] or \
           [(run, now, "2026-07-16", regime, exposure, "__CASH__", 1.0)]
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO trade_list_log "
                         "(run_id, generated_at, as_of_date, regime, exposure, ticker, weight) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def test_buy_updates_cash_holdings_and_ledger(temp_db):
    pt.set_cash(100_000)
    pt.record_trade("buy", "COMI", 100, 80.0)
    assert pt.get_cash() == pytest.approx(100_000 - 8_000)
    with get_connection() as conn:
        row = conn.execute("SELECT shares, avg_cost FROM user_positions WHERE ticker='COMI'").fetchone()
        ledger = conn.execute("SELECT COUNT(*) FROM user_trade_ledger").fetchone()[0]
    assert row["shares"] == 100 and row["avg_cost"] == pytest.approx(80.0)
    assert ledger == 1


def test_avg_cost_and_partial_sell(temp_db):
    pt.set_cash(100_000)
    pt.record_trade("buy", "COMI", 100, 80.0)
    pt.record_trade("buy", "COMI", 100, 100.0)     # avg cost -> 90
    with get_connection() as conn:
        row = conn.execute("SELECT shares, avg_cost FROM user_positions WHERE ticker='COMI'").fetchone()
    assert row["shares"] == 200 and row["avg_cost"] == pytest.approx(90.0)
    pt.record_trade("sell", "COMI", 50, 110.0)      # cash back, position shrinks
    assert pt.get_cash() == pytest.approx(100_000 - 8_000 - 10_000 + 5_500)


def test_cannot_sell_more_than_held(temp_db):
    pt.set_cash(50_000)
    pt.record_trade("buy", "COMI", 10, 80.0)
    with pytest.raises(ValueError):
        pt.record_trade("sell", "COMI", 20, 82.0)


def test_reconcile_actions(temp_db):
    _price("AAA", 100); _price("BBB", 50); _price("CCC", 25)
    pt.set_cash(100_000)
    # you hold AAA (way over) and CCC (not in target); you don't hold BBB
    pt.record_trade("buy", "AAA", 500, 100.0)   # 50k
    pt.record_trade("buy", "CCC", 400, 25.0)    # 10k
    _target("BULL", {"AAA": 0.20, "BBB": 0.30})  # target wants less AAA, some BBB, no CCC

    rec = pt.reconcile()
    acts = {a["ticker"]: a["action"] for a in rec["actions"]}
    assert acts["BBB"] == "BUY"     # target-only -> buy
    assert acts["CCC"] == "SELL"    # held, not in target -> exit
    assert acts["AAA"] == "TRIM"    # held far above target -> trim
    assert not rec["de_risked"]


def test_reconcile_bear_regime_says_go_to_cash(temp_db):
    _price("AAA", 100)
    pt.set_cash(50_000)
    pt.record_trade("buy", "AAA", 100, 100.0)
    _target("BEAR", {}, exposure=0.0)            # system de-risked -> cash target
    rec = pt.reconcile()
    assert rec["de_risked"] is True
    assert any(a["ticker"] == "AAA" and a["action"] == "SELL" for a in rec["actions"])
    assert "SELL all positions" in pt.format_advice(rec)


def test_reconcile_hold_when_aligned(temp_db):
    _price("AAA", 100); _price("BBB", 100)
    pt.set_cash(100_000)                          # start with cash, then deploy it
    pt.record_trade("buy", "AAA", 500, 100.0)    # 50k -> 50% (cash 50k)
    pt.record_trade("buy", "BBB", 500, 100.0)    # 50k -> 50% (cash 0)
    _target("BULL", {"AAA": 0.5, "BBB": 0.5})
    rec = pt.reconcile()
    assert all(a["action"] == "HOLD" for a in rec["actions"])


def test_accept_rebalances_to_target(temp_db):
    _price("AAA", 100); _price("BBB", 50)
    pt.set_cash(100_000)
    _target("BULL", {"AAA": 0.6, "BBB": 0.4})
    out = pt.accept_suggestion()
    assert not out["de_risked"]
    pos, cash = pt.holdings()
    w = pos.set_index("ticker")["weight"]
    assert abs(w["AAA"] - 0.6) < 0.02 and abs(w["BBB"] - 0.4) < 0.02   # matched target
    # after accepting, reconcile should show everything HOLD
    rec = pt.reconcile()
    assert all(a["action"] == "HOLD" for a in rec["actions"])
    # decision logged
    assert (pt.decision_history()["decision"] == "ACCEPTED").any()


def test_accept_in_bear_sells_to_cash(temp_db):
    _price("AAA", 100)
    pt.set_cash(50_000)
    pt.record_trade("buy", "AAA", 300, 100.0)   # 30k invested, 20k cash
    _target("BEAR", {}, exposure=0.0)           # system says de-risk
    out = pt.accept_suggestion()
    assert out["de_risked"] is True
    pos, cash = pt.holdings()
    assert pos.empty                             # sold everything
    assert cash > 49_000                         # back to (almost) all cash


def test_decline_and_own_log_decisions_without_trading(temp_db):
    _price("AAA", 100)
    pt.set_cash(10_000)
    _target("BULL", {"AAA": 1.0})
    pt.log_decision("DECLINED", "not this time")
    pt.log_decision("OWN", "buying my own picks")
    hist = pt.decision_history()
    assert set(hist["decision"]) >= {"DECLINED", "OWN"}
    # neither created positions
    pos, cash = pt.holdings()
    assert pos.empty and cash == 10_000


def test_pending_and_catchup(temp_db):
    _price("AAA", 100)
    _target("BULL", {"AAA": 1.0}, run="r1")           # a suggestion, undecided
    # simulate a second, older suggestion date with no decision
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO trade_list_log (run_id, generated_at, as_of_date, regime, exposure, ticker, weight) "
                     "VALUES ('r0','2026-07-10T00:00:00','2026-07-10','BULL',1.0,'AAA',1.0)")
    pend = pt.pending_decisions()
    assert len(pend) == 2                              # both dates unanswered
    n = pt.catchup()
    assert n == 2
    assert pt.pending_decisions().empty                # all cleared
    hist = pt.decision_history()
    assert (hist["decision"] == "MISSED").sum() == 2   # logged as MISSED, not a trade


def test_rebuild_cost_basis_replays_ledger(temp_db):
    """Correcting a fill price after the fact must be fixed by REPLAYING the
    ledger -- avg_cost is path-dependent, so it can't be patched arithmetically."""
    _price("AAA", 100)
    pt.set_cash(10_000)
    pt.record_trade("buy", "AAA", 10, 100.0)
    pt.record_trade("buy", "AAA", 10, 120.0)          # avg -> 110
    with get_connection() as conn:
        assert conn.execute("SELECT avg_cost FROM user_positions WHERE ticker='AAA'").fetchone()[0] == pytest.approx(110.0)
        # correct the second fill: it was really 130
        conn.execute("UPDATE user_trade_ledger SET price=130.0 WHERE ticker='AAA' AND price=120.0")
    df = pt.rebuild_cost_basis()
    with get_connection() as conn:
        new = conn.execute("SELECT avg_cost, shares FROM user_positions WHERE ticker='AAA'").fetchone()
    assert new["avg_cost"] == pytest.approx(115.0)     # (10*100 + 10*130)/20
    assert new["shares"] == 20                         # shares untouched
    assert (df["ticker"] == "AAA").any()


def test_rebuild_cost_basis_handles_sells(temp_db):
    """A sell reduces shares but must not corrupt the remaining cost basis."""
    _price("AAA", 100)
    pt.set_cash(10_000)
    pt.record_trade("buy", "AAA", 10, 100.0)
    pt.record_trade("sell", "AAA", 4, 150.0)
    pt.rebuild_cost_basis()
    with get_connection() as conn:
        row = conn.execute("SELECT avg_cost, shares FROM user_positions WHERE ticker='AAA'").fetchone()
    assert row["avg_cost"] == pytest.approx(100.0)     # unchanged by the sell
    assert row["shares"] == 6


# --- do not suggest trades that cannot clear their own fee -------------------

def test_uneconomic_rebalance_is_downgraded_to_hold(temp_db, monkeypatch):
    """A 42 EGP correction carrying a flat ~EGP 5 commission pays 12% to chase a
    few percent of drift. The gap test says rebalance; the arithmetic says the
    cure costs more than the disease."""
    _price("AAA", 100); _price("BBB", 100)
    pt.set_cash(1000)                          # funded, so the book totals 1,000
    pt.record_trade("buy", "AAA", 6, 100.0)
    pt.record_trade("buy", "BBB", 4, 100.0)
    _target("BULL", {"AAA": 0.40, "BBB": 0.60})        # BBB is 20pp underweight -> ADD
    rec = pt.reconcile()
    acts = {a["ticker"]: a for a in rec["actions"]}
    assert acts["BBB"]["action"] == "HOLD"
    assert acts["BBB"]["uneconomic"] is True
    assert acts["BBB"]["fee_share"] > pt.MAX_FEE_SHARE_OF_TRADE


def test_a_large_enough_trade_still_goes_through(temp_db):
    """The guard must not silence the system on a book where trading IS
    economic -- it is a cost test, not a blanket ban."""
    _price("AAA", 100); _price("BBB", 100)
    pt.set_cash(100_000)
    pt.record_trade("buy", "AAA", 600, 100.0)
    pt.record_trade("buy", "BBB", 400, 100.0)
    _target("BULL", {"AAA": 0.40, "BBB": 0.60})        # same 20pp gap, 100x the notional
    rec = pt.reconcile()
    acts = {a["ticker"]: a for a in rec["actions"]}
    assert acts["BBB"]["action"] == "ADD"
    assert acts["BBB"]["uneconomic"] is False


def test_holds_are_not_labelled_uneconomic(temp_db):
    """A name already at target has no trade to be uneconomic about. Flagging it
    would make the warning meaningless by attaching it to everything."""
    _price("AAA", 100)
    pt.set_cash(1000)
    pt.record_trade("buy", "AAA", 10, 100.0)
    _target("BULL", {"AAA": 1.0})
    rec = pt.reconcile()
    a = rec["actions"][0]
    assert a["action"] == "HOLD"
    assert a["uneconomic"] is False


def test_reconcile_uses_the_same_name_cap_as_the_ticket(temp_db):
    """If only the ticket caps, the reconciler compares an 8-name book against a
    24-name target: every held name reads TRIM and every excluded name reads BUY
    -- a permanent false rebalance signal that rebuilds the capped-away book."""
    tgt = {}
    for i in range(20):
        _price(f"T{i}", 100)
        tgt[f"T{i}"] = 1 / 20
    pt.set_cash(1000)
    for i in range(5):
        pt.record_trade("buy", f"T{i}", 2, 100.0)
    _target("BULL", tgt)
    rec = pt.reconcile()
    assert rec["capped"] is not None
    assert rec["capped"]["kept"] <= rec["capped"]["cap"] + 5
    # the 15 capped-away names must NOT appear as BUY suggestions
    buys = {a["ticker"] for a in rec["actions"] if a["action"] == "BUY"}
    assert buys.isdisjoint(set(rec["capped"]["dropped"]))


# --- the target must be recomputed when the data moves on --------------------

def test_stale_target_is_regenerated_not_reused(temp_db, monkeypatch):
    """reconcile() used to reuse the stored target whenever one existed, so a
    ticket built without running the daily job re-issued the PREVIOUS session's
    regime call with today's date on it."""
    from ops import trade_list as tlmod
    _price("AAA", 100); _price("BBB", 100)
    pt.set_cash(1000)
    _target("BULL", {"AAA": 1.0})                       # stored, as-of 2026-07-16
    monkeypatch.setattr(tlmod, "latest_session",
                        lambda: pd.Timestamp("2026-07-20"))   # newer data exists
    calls = {"n": 0}

    def fake_generate(**kw):
        calls["n"] += 1
        return {"weights": pd.Series({"BBB": 1.0}), "regime": "BEAR", "exposure": 0.0}

    monkeypatch.setattr(tlmod, "generate_trade_list", fake_generate)
    rec = pt.reconcile()
    assert calls["n"] == 1                              # it regenerated
    assert rec["regime"] == "BEAR"                      # and used the NEW call
    assert rec["signal_note"] is None


def test_fresh_target_is_not_needlessly_recomputed(temp_db, monkeypatch):
    """Regenerating is expensive. When the stored target already matches the
    newest session there is nothing to recompute."""
    from ops import trade_list as tlmod
    _price("AAA", 100); pt.set_cash(1000)
    _target("BULL", {"AAA": 1.0})
    monkeypatch.setattr(tlmod, "latest_session", lambda: pd.Timestamp("2026-07-16"))
    calls = {"n": 0}
    monkeypatch.setattr(tlmod, "generate_trade_list",
                        lambda **kw: calls.__setitem__("n", calls["n"] + 1))
    rec = pt.reconcile()
    assert calls["n"] == 0
    assert rec["regime"] == "BULL"


def test_failed_regeneration_falls_back_but_says_so(temp_db, monkeypatch):
    """Using a stale target is survivable. Using one SILENTLY is not."""
    from ops import trade_list as tlmod
    _price("AAA", 100); pt.set_cash(1000)
    _target("BULL", {"AAA": 1.0})
    monkeypatch.setattr(tlmod, "latest_session", lambda: pd.Timestamp("2026-07-20"))

    def boom(**kw):
        raise RuntimeError("macro data 23 days stale")

    monkeypatch.setattr(tlmod, "generate_trade_list", boom)
    rec = pt.reconcile()
    assert rec["regime"] == "BULL"                       # fell back
    assert rec["signal_note"] and "macro data" in rec["signal_note"]


def test_no_stored_target_and_failed_generation_raises(temp_db, monkeypatch):
    """With nothing to fall back to there is no honest output -- fail loudly
    rather than invent a book."""
    from ops import trade_list as tlmod
    _price("AAA", 100); pt.set_cash(1000)

    def boom(**kw):
        raise RuntimeError("no data")

    monkeypatch.setattr(tlmod, "generate_trade_list", boom)
    with pytest.raises(RuntimeError):
        pt.reconcile()
