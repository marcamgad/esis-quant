"""4-week ranges and the high-risk rule."""

import numpy as np
import pandas as pd

from ops import forecast as fc
from ops import high_risk_picks as hr


def test_forecast_bands_are_ordered_and_centred_on_today():
    f = fc.forecast("COMI")
    q = f["quantiles"]
    assert q["p10"] < q["p25"] < q["p50"] < q["p75"] < q["p90"]
    assert abs(q["p50"] / f["price"] - 1) < 0.02                 # neutral centre
    assert 0 <= f["p_down10"] <= 1 and 0 <= f["p_up10"] <= 1 and f["month_vol"] > 0


def test_forecast_uses_only_data_up_to_as_of():
    a = fc.forecast("COMI", as_of="2024-06-30")
    b = fc.forecast("COMI", as_of="2024-06-30")
    assert a["price_date"] <= "2024-06-30" and a["quantiles"] == b["quantiles"]


def test_calibration_on_a_small_sample_is_roughly_right():
    c = fc.backtest_calibration(start="2023-01-01", step_months=3, max_names=15)
    assert c["n"] > 100
    assert 0.65 <= c["inside_80_band"] <= 0.92 and 0.35 <= c["inside_50_band"] <= 0.65


def _f(**kw):
    base = {"q": {"period": "Q2 2026", "value": 5.0}, "growth": {"change": 8.0, "label": "small base"},
            "de": 2.5, "financial": False, "negative_equity": False, "value": 400_000, "capital_change": None,
            "val": {"pe": 30.0}}
    base.update(kw)
    return base


def test_high_risk_gates_allow_what_the_careful_rule_refuses():
    assert all(ok for _, ok, _ in hr._gates(_f(), 0.15))          # small base, D/E 2.5, P/E 30, thin
    assert not all(ok for _, ok, _ in hr._gates(_f(), 0.08))      # not risky enough
    assert not all(ok for _, ok, _ in hr._gates(_f(de=3.5), 0.15))
    assert not all(ok for _, ok, _ in hr._gates(_f(q={"period": "Q2 2026", "value": -1.0}), 0.15))


def test_exits_fire_on_stop_target_and_time(monkeypatch):
    import ops.watchlist as w
    w.ensure_tables()
    import sqlite3, config
    con = sqlite3.connect(config.DB_PATH)
    day = str((pd.Timestamp.now() - pd.Timedelta(days=120)).date())
    con.execute("DELETE FROM system_signals WHERE rule = ?", (hr.RULE,))
    for t, p in (("AAA", 10.0), ("BBB", 10.0), ("CCC", 10.0)):
        con.execute("INSERT INTO system_signals (signal_date, ticker, action, shares, price, rule) VALUES (?,?,?,?,?,?)",
                    (day if t == "CCC" else str(pd.Timestamp.now().date()), t, "BUY", 10, p, hr.RULE))
    con.commit()
    held = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "shares": [10, 10, 10], "price": [7.5, 14.5, 10.5]})
    out = {x["ticker"]: x["why"][0].split(":")[0] for x in hr._exits(held, pd.Timestamp.now())}
    con.execute("DELETE FROM system_signals WHERE rule = ?", (hr.RULE,))
    con.commit()
    con.close()
    assert out == {"AAA": "STOP", "BBB": "TARGET", "CCC": "TIME"}
