"""
Tests for Phase D delivery pieces: the Telegram brief composes the key facts,
send_brief dry-runs without credentials (never fabricates a send), and the
report generator produces a file with the headline numbers. Uses a small
synthetic run_full_system result so the tests don't depend on a full backtest.
"""

import numpy as np
import pandas as pd

from alerts.telegram_bot import compose_daily_brief, send_brief


def _fake_result():
    idx = pd.bdate_range("2024-01-01", periods=10)
    return {
        "returns": {"full_system": pd.Series(0.001, index=idx)},
        "latest": {"date": idx[-1], "regime": "BEAR", "param_vintage": pd.Timestamp("2024-01-01"),
                   "exposure": 0.0, "breaker_active": False, "current_drawdown": -0.05},
        "metrics": {"full_system": {"sharpe": 1.70, "sortino": 1.50, "max_drawdown": -0.25,
                                    "calmar": 0.98, "cornish_fisher_var95": 0.0104, "ann_return": 0.245}},
        "attribution": {"timing": {"timing_contrib": 0.46}},
    }


def test_brief_contains_key_facts():
    txt = compose_daily_brief(_fake_result())
    assert "REGIME: BEAR" in txt
    assert "IN CASH" in txt            # exposure 0 -> in cash
    assert "Sharpe 1.70" in txt
    assert "DRAWDOWN CONTROL" in txt   # honesty note present


def test_send_brief_dry_runs_without_credentials(monkeypatch):
    # no token/chat configured -> must NOT send, must return composed text
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    out = send_brief(result=_fake_result())
    assert out["sent"] is False
    assert "ESIS-QUANT Morning Brief" in out["text"]


def test_report_generation_writes_headline_numbers(tmp_path):
    from reports.generate_report import generate_final_report
    res = _fake_result()
    # report needs core + equal_weight metrics too
    res["metrics"]["core"] = {"sharpe": 1.40, "sortino": 1.27, "max_drawdown": -0.428,
                              "calmar": 0.56, "cornish_fisher_var95": 0.0154, "ann_return": 0.239, "dd_duration": 487}
    res["metrics"]["equal_weight"] = {"sharpe": 1.24, "sortino": 1.05, "max_drawdown": -0.571,
                                      "calmar": 0.44, "cornish_fisher_var95": 0.0199, "ann_return": 0.252, "dd_duration": 481}
    res["metrics"]["full_system"]["dd_duration"] = 61
    res["attribution"]["brinson"] = {"total": {"allocation": 0.062, "selection": -0.184, "interaction": -0.058}}
    path = generate_final_report(res, path=str(tmp_path / "r.md"))
    text = open(path, encoding="utf-8").read()
    assert "Sharpe" in text and "1.70" in text
    assert "selection adds NO alpha" in text.replace("**", "")
    assert "Not investment advice" in text
