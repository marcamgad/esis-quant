"""
Order-ticket round trip: write a ticket sized to the book, simulate the user
filling in real fills, apply it, and confirm the trades land in the DB (and
only filled rows are saved).
"""
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from ops import portfolio_tracker as pt
from ops import order_ticket as ot


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "t.db"))
    monkeypatch.setattr(ot, "TICKET_DIR", str(tmp_path / "orders"))
    # These tests exercise TICKET MECHANICS, not concentration limits, and many
    # use 1-2 synthetic names which are necessarily >20% of the book. The
    # concentration limit is tested directly in tests/test_pretrade.py; relaxing
    # it here keeps each test about one thing rather than weakening the control.
    monkeypatch.setattr(config, "MAX_POSITION_WEIGHT", 1.0)
    database.init_db()
    return config.DB_PATH


def _price(t, p, date="2026-07-16", value_traded=1e7):
    # value_traded must be REALISTIC, not 0. With 0 the pre-trade screen reads
    # "liquidity unknown" and fail-closes -- correct behaviour on a real order,
    # but it made these fixtures test the rejection path instead of the ticket.
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, value_traded_egp) "
                     "VALUES (?, ?, ?, ?, ?, ?)", (t, date, p, p, int(value_traded // max(p, 1)), value_traded))


def _target(weights, as_of="2026-07-16"):
    now = pd.Timestamp.now().isoformat()
    rows = [("r1", now, as_of, "BULL", 1.0, t, w) for t, w in weights.items()]
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO trade_list_log "
                         "(run_id, generated_at, as_of_date, regime, exposure, ticker, weight) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def test_ticket_write_never_oversuggests_cash(temp_db):
    _price("AAA", 100); _price("BBB", 50)
    pt.set_cash(1000)
    _target({"AAA": 0.6, "BBB": 0.4})
    out = ot.write_ticket()
    df = ot._read_ticket(out["path"])
    assert (df["suggested_egp"].sum()) <= 1000 + 1e-6      # never suggests more than cash
    assert out["suggested_total"] <= 1000


def test_ticket_apply_saves_only_filled_rows(temp_db):
    _price("AAA", 100); _price("BBB", 50)
    pt.set_cash(1000)
    _target({"AAA": 0.6, "BBB": 0.4})
    out = ot.write_ticket()
    df = ot._read_ticket(out["path"])
    # user fills AAA only (bought 4 for 404 total + 2.40 fee), leaves BBB blank
    df.loc[df["ticker"] == "AAA", "shares"] = 4
    df.loc[df["ticker"] == "AAA", "PAID_EGP"] = 404.0
    df.loc[df["ticker"] == "AAA", "FEE_EGP"] = 2.40
    # rewrite with the comment header preserved-ish (apply skips '#' lines)
    df.to_csv(out["path"], index=False)
    res = ot.apply_ticket(out["path"], as_of="2026-07-16")
    assert res["n"] == 1                                    # only the filled row
    pos, cash = pt.holdings()
    held = pos.set_index("ticker")
    assert "AAA" in held.index and "BBB" not in held.index
    assert res["applied"][0]["price"] == pytest.approx(101.0)   # derived 404/4
    assert cash == pytest.approx(1000 - 404.0 - 2.40)       # fee leaves the account too
    assert (pt.decision_history()["decision"] == "OWN").any()


def test_prune_keeps_newest_tickets(temp_db, tmp_path):
    import os
    os.makedirs(ot.TICKET_DIR, exist_ok=True)
    for d in ("2026-07-01", "2026-07-02", "2026-07-03"):
        open(os.path.join(ot.TICKET_DIR, f"orders_{d}.csv"), "w").close()
    deleted = ot.prune_old_tickets(keep=2)
    assert deleted == ["orders_2026-07-01.csv"]            # oldest removed
    left = sorted(os.listdir(ot.TICKET_DIR))
    assert left == ["orders_2026-07-02.csv", "orders_2026-07-03.csv"]


def test_write_survives_locked_ticket_file(temp_db, monkeypatch):
    """The daily popup writes a ticket every morning. If yesterday's file is
    still open in Excel, the write must fall back to a suffixed name rather
    than crashing the whole run."""
    import os, builtins
    _price("AAA", 100)
    pt.set_cash(1000)
    _target({"AAA": 1.0})
    first = ot.write_ticket()
    real_open = builtins.open

    def fake_open(f, mode="r", *a, **kw):
        # simulate Windows' lock: appending to the existing ticket is denied
        if str(f) == first["path"] and "a" in mode:
            raise PermissionError(13, "locked")
        return real_open(f, mode, *a, **kw)

    monkeypatch.setattr(builtins, "open", fake_open)
    second = ot.write_ticket()
    assert second["path"] != first["path"]      # fell back, did not crash
    assert os.path.exists(second["path"])


def test_ticket_named_by_generation_date_and_prunes_previous(temp_db):
    """The file is named for the day it was generated (not the signal's as-of),
    and writing a new one removes the superseded file."""
    import os, pandas as pd
    _price("AAA", 100); pt.set_cash(1000)
    _target({"AAA": 1.0}, as_of="2026-07-22")          # signal is from the 22nd
    out = ot.write_ticket()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    assert os.path.basename(out["path"]) == f"orders_{today}.csv"   # today, not 07-22
    assert out["as_of"] == "2026-07-22"                 # as-of still reported

    # a stale ticket from an earlier day must be swept when the new one lands
    stale = os.path.join(ot.TICKET_DIR, "orders_2026-01-01.csv")
    open(stale, "w").close()
    out2 = ot.write_ticket()
    assert not os.path.exists(stale)
    assert sorted(os.listdir(ot.TICKET_DIR)) == [f"orders_{today}.csv"]


def test_apply_finds_latest_ticket_regardless_of_name(temp_db):
    """apply() must locate the newest ticket by scanning, since the filename is
    the generation date and need not match the signal's as-of date."""
    import pandas as pd
    _price("AAA", 100); pt.set_cash(1000)
    _target({"AAA": 1.0}, as_of="2026-07-22")
    out = ot.write_ticket()
    df = ot._read_ticket(out["path"])
    df.loc[df["ticker"] == "AAA", "shares"] = 3
    df.loc[df["ticker"] == "AAA", "PAID_EGP"] = 297.0
    df.loc[df["ticker"] == "AAA", "FEE_EGP"] = 2.0
    df.to_csv(out["path"], index=False)
    res = ot.apply_ticket()                             # no path given
    assert res["n"] == 1
    pos, cash = pt.holdings()
    assert int(pos.set_index("ticker").loc["AAA", "shares"]) == 3


# --- two-column fills: PAID_EGP + FEE_EGP ----------------------------------

def test_fill_price_is_derived_from_total_paid(temp_db):
    """You type the total, not the per-share average. 350 for 22 shares must
    become 15.909.../share -- a number you should never have to compute."""
    _price("AAA", 16); pt.set_cash(1000)
    _target({"AAA": 1.0})
    out = ot.write_ticket()
    df = ot._read_ticket(out["path"])
    df.loc[df["ticker"] == "AAA", ["shares", "PAID_EGP", "FEE_EGP"]] = [22, 350.0, 2.35]
    df.to_csv(out["path"], index=False)
    res = ot.apply_ticket(out["path"], as_of="2026-07-16")
    assert res["applied"][0]["price"] == pytest.approx(350.0 / 22)


def test_paid_without_shares_is_refused_not_guessed(temp_db):
    """A total with no quantity cannot yield a price. Inventing one would write a
    false cost basis, so the row must be skipped and reported."""
    _price("AAA", 100); pt.set_cash(1000)
    _target({"AAA": 1.0})
    out = ot.write_ticket()
    df = ot._read_ticket(out["path"])
    df["shares"] = df["shares"].astype(object)
    df.loc[df["ticker"] == "AAA", "PAID_EGP"] = 300.0
    df.loc[df["ticker"] == "AAA", "shares"] = ""      # user forgot the quantity
    df.to_csv(out["path"], index=False)
    res = ot.apply_ticket(out["path"], as_of="2026-07-16")
    assert res["n"] == 0
    assert res["skipped"] and res["skipped"][0]["ticker"] == "AAA"
    _, cash = pt.holdings()
    assert cash == pytest.approx(1000)          # nothing was recorded


def test_hand_typed_cells_are_tolerated(temp_db):
    """This file is filled in by hand. '1,250 EGP' must parse; junk must read as
    'not traded' rather than as some number."""
    assert ot._num("1,250 EGP") == pytest.approx(1250.0)
    assert ot._num(" 2.35 ") == pytest.approx(2.35)
    for blank in ("", "  ", "nan", "None", "-", "n/a", None):
        assert ot._num(blank) == 0.0


def test_fee_enters_the_cost_basis(temp_db):
    """The basis is what the shares actually cost to acquire, fee included --
    otherwise every position looks cheaper than it was."""
    _price("AAA", 100); pt.set_cash(1000)
    _target({"AAA": 1.0})
    out = ot.write_ticket()
    df = ot._read_ticket(out["path"])
    df.loc[df["ticker"] == "AAA", ["shares", "PAID_EGP", "FEE_EGP"]] = [4, 400.0, 8.0]
    df.to_csv(out["path"], index=False)
    ot.apply_ticket(out["path"], as_of="2026-07-16")
    pos, _ = pt.holdings()
    assert pos.set_index("ticker").loc["AAA", "avg_cost"] == pytest.approx((400.0 + 8.0) / 4)


def test_fee_model_check_compares_against_the_real_schedule(temp_db):
    """The first real fee data the system has ever seen must be checked against
    what backtest/fees.py predicted, not silently absorbed."""
    _price("AAA", 100); pt.set_cash(1000)
    _target({"AAA": 1.0})
    out = ot.write_ticket()
    df = ot._read_ticket(out["path"])
    df.loc[df["ticker"] == "AAA", ["shares", "PAID_EGP", "FEE_EGP"]] = [4, 400.0, 50.0]
    df.to_csv(out["path"], index=False)
    res = ot.apply_ticket(out["path"], as_of="2026-07-16")
    fc = res["fee_check"]
    assert fc["actual_egp"] == pytest.approx(50.0)
    assert fc["predicted_egp"] < 10.0            # schedule says ~2.4 EGP
    assert fc["error_pct"] > 0.20                # a gap this size must be flagged


# --- name-count cap ---------------------------------------------------------

def test_ticket_caps_the_number_of_names(temp_db):
    """Each extra name is another flat EGP 2 per side, and measured risk stops
    falling past ~10 names, so the ticket must not print 24 lines on a small book."""
    tgt = {}
    for i in range(20):
        _price(f"T{i}", 10 + i)
        tgt[f"T{i}"] = 1 / 20
    pt.set_cash(10_000)
    _target(tgt)
    out = ot.write_ticket(max_names=6)
    assert out["rows"] <= 6
    assert out["capped"]["cap"] == 6
    assert len(out["capped"]["dropped"]) == 14


def test_cap_can_be_disabled(temp_db):
    tgt = {}
    for i in range(12):
        _price(f"T{i}", 10 + i)
        tgt[f"T{i}"] = 1 / 12
    pt.set_cash(10_000)
    _target(tgt)
    out = ot.write_ticket(max_names=0)
    assert out["capped"] is None
    assert out["rows"] > 6


def test_cap_never_evicts_a_name_you_already_hold(temp_db):
    """Dropping a held name would print a SELL purely because the cap moved --
    churn, not a signal, and it costs a fee to express."""
    tgt = {}
    for i in range(12):
        _price(f"T{i}", 10 + i)
        tgt[f"T{i}"] = (12 - i) / 78          # T0 heaviest, T11 lightest
    pt.set_cash(10_000)
    pt.record_trade("buy", "T11", 5, 21.0)    # hold the LIGHTEST name
    _target(tgt)
    out = ot.write_ticket(max_names=3)
    assert "T11" not in out["capped"]["dropped"]


# --- size-aware name cap ----------------------------------------------------

def test_name_cap_grows_with_the_book():
    """The flat fee is amortised over notional, so the affordable number of
    names must RISE with capital. A constant would be wrong at one end."""
    small = ot.suggested_name_cap(1_000)
    mid = ot.suggested_name_cap(10_000)
    big = ot.suggested_name_cap(100_000)
    assert small < mid <= big
    assert small >= ot.NAME_CAP_FLOOR
    assert big <= ot.NAME_CAP_CEILING


def test_name_cap_is_bounded_at_both_ends():
    """Never below the diversification floor, never above the point where
    measured risk stops falling."""
    assert ot.suggested_name_cap(1) == ot.NAME_CAP_FLOOR
    assert ot.suggested_name_cap(0) == ot.NAME_CAP_FLOOR
    assert ot.suggested_name_cap(10**9) == ot.NAME_CAP_CEILING


def test_name_cap_respects_the_cost_budget():
    """The chosen count's round-trip cost must actually fit the budget it was
    solved against (unless the floor forced it)."""
    from backtest.fees import one_side_cost
    for cap_egp in (5_000, 20_000, 100_000):
        n = ot.suggested_name_cap(cap_egp)
        rt = 2.0 * n * one_side_cost(cap_egp / n, spread_per_side=0.0)["total_egp"] / cap_egp
        assert rt <= ot.NAME_CAP_COST_BUDGET + 1e-9


def test_cheaper_fees_would_buy_more_names():
    """Sanity: the cap is driven by the fee schedule, not hardcoded. A looser
    budget must permit at least as many names."""
    tight = ot.suggested_name_cap(5_000, budget=0.01)
    loose = ot.suggested_name_cap(5_000, budget=0.10)
    assert loose >= tight


# --- double-book guard ------------------------------------------------------

def test_apply_refuses_after_the_day_was_already_accepted(temp_db):
    """ACCEPT and APPLY are alternatives, never both. Booking the same as-of
    twice is what put 13 phantom positions on top of 8 real ones on 2026-08-04."""
    _price("AAA", 100); pt.set_cash(1000)
    _target({"AAA": 1.0}, as_of="2026-07-22")
    out = ot.write_ticket()
    pt.log_decision("ACCEPTED", note="rebalanced to system target", as_of="2026-07-22")
    df = ot._read_ticket(out["path"])
    df.loc[df["ticker"] == "AAA", ["shares", "PAID_EGP", "FEE_EGP"]] = [3, 300.0, 5.0]
    df.to_csv(out["path"], index=False)
    res = ot.apply_ticket(out["path"], as_of="2026-07-22")
    assert res["n"] == 0
    assert res["blocked"]["decision"] == "ACCEPTED"
    _, cash = pt.holdings()
    assert cash == pytest.approx(1000)            # nothing was double-booked


def test_force_overrides_the_double_book_guard(temp_db):
    """The guard must be escapable for genuine additional fills, but only
    deliberately."""
    _price("AAA", 100); pt.set_cash(1000)
    _target({"AAA": 1.0}, as_of="2026-07-22")
    out = ot.write_ticket()
    pt.log_decision("ACCEPTED", note="x", as_of="2026-07-22")
    df = ot._read_ticket(out["path"])
    df.loc[df["ticker"] == "AAA", ["shares", "PAID_EGP", "FEE_EGP"]] = [3, 300.0, 5.0]
    df.to_csv(out["path"], index=False)
    res = ot.apply_ticket(out["path"], as_of="2026-07-22", force=True)
    assert res["n"] == 1


def test_a_different_day_is_not_blocked(temp_db):
    """The guard is per as-of date -- yesterday's ACCEPT must not block today."""
    _price("AAA", 100); pt.set_cash(1000)
    pt.log_decision("ACCEPTED", note="yesterday", as_of="2026-07-21")
    _target({"AAA": 1.0}, as_of="2026-07-22")
    out = ot.write_ticket()
    df = ot._read_ticket(out["path"])
    df.loc[df["ticker"] == "AAA", ["shares", "PAID_EGP", "FEE_EGP"]] = [3, 300.0, 5.0]
    df.to_csv(out["path"], index=False)
    res = ot.apply_ticket(out["path"], as_of="2026-07-22")
    assert res["n"] == 1


# --- deposit-aware scale-in planning ----------------------------------------

def test_deposit_sizes_the_plan_without_recording_the_money(temp_db):
    """Planning a scale-in must NOT credit cash that is not in the account --
    that is exactly how the book stops matching the broker."""
    _price("AAA", 100); _price("BBB", 100)
    pt.set_cash(50)
    pt.record_trade("buy", "AAA", 5, 100.0)      # 500 invested, 50 -> -450... fund first
    pt.set_cash(50)
    _target({"AAA": 0.5, "BBB": 0.5})
    out = ot.write_ticket(deposit=10_000)
    assert out["deposit"] == 10_000
    assert out["book"] > 10_000                  # sized against the bigger book
    assert pt.get_cash() == pytest.approx(50)    # but cash is UNTOUCHED


def test_deposit_plan_marks_itself_as_a_plan(temp_db):
    _price("AAA", 100); pt.set_cash(50)
    _target({"AAA": 1.0})
    out = ot.write_ticket(deposit=5_000)
    text = open(out["path"], encoding="utf-8-sig").read()
    assert "PLAN ONLY" in text
    assert "NOT yet in" in text


def test_scale_in_tops_up_rather_than_selling(temp_db):
    """Adding capital must never produce SELL lines for names already held and
    still in the target -- selling to rebuy the same thing is pure fee."""
    for t in ("AAA", "BBB", "CCC"):
        _price(t, 100)
    pt.set_cash(1000)
    pt.record_trade("buy", "AAA", 5, 100.0)
    pt.record_trade("buy", "BBB", 5, 100.0)
    _target({"AAA": 0.4, "BBB": 0.4, "CCC": 0.2})
    out = ot.write_ticket(deposit=50_000)
    df = ot._read_ticket(out["path"])
    assert not df.empty
    assert "SELL" not in set(df["action"])
    assert "TRIM" not in set(df["action"])
    assert set(df.loc[df["action"] == "ADD", "ticker"]) >= {"AAA", "BBB"}


def test_idle_is_leftover_cash_not_the_whole_book(temp_db):
    """On a top-up ticket, money already invested in existing positions is not
    idle. Subtracting from portfolio_value overstated it by the book's value."""
    _price("AAA", 100); _price("BBB", 100)
    pt.set_cash(1000)
    pt.record_trade("buy", "AAA", 9, 100.0)      # 900 invested, 100 cash
    _target({"AAA": 0.5, "BBB": 0.5})
    out = ot.write_ticket()
    assert out["idle"] <= 1000                   # never exceeds spendable cash
    assert out["idle"] < out["cash"] + 1e-6


# --- prune must not delete the ticket it just wrote -------------------------

def test_prune_orders_by_mtime_not_by_name(temp_db, monkeypatch):
    """The locked-file fallback writes orders_<date>_<HHMMSS>.csv, which sorts
    AFTER orders_<date>.csv. Name-ordering therefore deleted a freshly written
    ticket as if it were the older one -- observed live on 2026-08-04."""
    import os, time
    os.makedirs(ot.TICKET_DIR, exist_ok=True)
    old = os.path.join(ot.TICKET_DIR, "orders_2026-08-04_113205.csv")
    new = os.path.join(ot.TICKET_DIR, "orders_2026-08-04.csv")
    open(old, "w").close()
    time.sleep(0.02)
    open(new, "w").close()                       # written LAST, sorts FIRST by name
    deleted = ot.prune_old_tickets(keep=1)
    assert deleted == ["orders_2026-08-04_113205.csv"]
    assert os.path.exists(new)


def test_protect_never_deletes_the_named_file(temp_db):
    import os, time
    os.makedirs(ot.TICKET_DIR, exist_ok=True)
    a = os.path.join(ot.TICKET_DIR, "orders_2026-08-01.csv")
    b = os.path.join(ot.TICKET_DIR, "orders_2026-08-02.csv")
    open(a, "w").close(); time.sleep(0.02); open(b, "w").close()
    ot.prune_old_tickets(keep=1, protect=a)      # protect the OLDER one
    assert os.path.exists(a)


def test_write_leaves_its_own_ticket_on_disk(temp_db):
    """End-to-end guard on the bug: after write(), the returned path must exist."""
    import os
    _price("AAA", 100); pt.set_cash(1000)
    _target({"AAA": 1.0})
    stale = os.path.join(ot.TICKET_DIR, "orders_2026-08-04_999999.csv")
    os.makedirs(ot.TICKET_DIR, exist_ok=True)
    open(stale, "w").close()
    out = ot.write_ticket()
    assert os.path.exists(out["path"])


# --- allocator output must clear its own fee too -----------------------------

def test_allocator_dribbles_are_dropped_from_the_ticket(temp_db):
    """The integer allocator minimises tracking error and knows nothing about
    fees, so it spends residual cash on 1-share orders. A 9 EGP order carries a
    ~5 EGP commission -- 55% of itself. reconcile() screens its own actions;
    allocator output reaches the ticket by a different path and must be screened
    too, or the two paths disagree."""
    _price("AAA", 1000); _price("BBB", 9)
    pt.set_cash(5000)
    _target({"AAA": 0.98, "BBB": 0.02})
    out = ot.write_ticket()
    df = ot._read_ticket(out["path"])
    tiny = df[df["suggested_egp"] < 100]
    assert tiny.empty, f"uneconomic rows printed: {tiny[['ticker', 'suggested_egp']].to_dict('records')}"


def test_uneconomic_uses_the_same_bound_as_the_reconciler(temp_db):
    """One threshold, two call sites. If these ever diverge, the ticket and the
    advice screen disagree about what is worth trading."""
    assert ot._uneconomic(9.0) is True
    assert ot._uneconomic(5_000.0) is False
    assert ot._uneconomic(0.0) is True             # unpriceable -> too expensive
    boundary = pt.MAX_FEE_SHARE_OF_TRADE
    assert ot._fee_share(9.0) > boundary >= ot._fee_share(5_000.0)


def test_fee_share_treats_unknown_as_infinite_not_free(temp_db):
    """An order we cannot price must never read as costless."""
    assert ot._fee_share(0) == float("inf")
    assert ot._fee_share(-5) == float("inf")
