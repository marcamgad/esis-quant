"""
The unvalidated layers must stay OFF the live path.  (Tier 5, items 14-15)

Three model families exist in this repo and none has earned a place in a live
trade list:

  * SENTIMENT (Layer 4) - 0 of 64,080 signal rows carry a non-zero score, and
    the entire news corpus is 517 articles spanning ONE MONTH (2026-06-11 to
    2026-07-10) against an 11-year backtest. The model also measured F1 0.33.
  * TFT (Layer 6) - near-random IC; trained on Kaggle, checkpoint present on
    disk but never consumed by the live path.
  * CAUSAL / DOUBLE-ML (Layer 5) - carried a target-leakage finding (a pooled
    theta estimated on the full panel including the test period).

Both the TFT and sentiment CHECKPOINTS EXIST ON DISK. Isolation is therefore not
guaranteed by the models being absent -- it is guaranteed only by nothing
importing them. That is exactly the kind of property that decays silently during
a refactor, so it is pinned here.

WHAT THESE TESTS ARE FOR. Not to prove the layers are bad; the prior audits did
that. They are a tripwire: if anyone ever wires one of these into
ops/daily_run.py, ops/trade_list.py or portfolio/system.py, the suite fails and
says why. Re-enabling one is then a deliberate act that has to delete a test,
not an accident.
"""
import sys
from pathlib import Path

import pytest

import config

# Module-name fragments that must never be reachable from the live trade path.
BANNED = ("sentiment", "tft", "causal", "double_ml", "alpha_score",
          "fama_macbeth", "rank_persistence")

# The modules that constitute the live path: data in, trade list out.
LIVE_ENTRYPOINTS = ("ops.daily_run", "ops.trade_list", "portfolio.system",
                    "ops.order_ticket", "ops.portfolio_tracker")


def _modules_reachable_from(entrypoints=LIVE_ENTRYPOINTS):
    """Import the live entrypoints in a CLEAN SUBPROCESS and report which project
    modules got pulled in transitively.

    WHY A SUBPROCESS. The first version of this helper inspected `sys.modules`
    in-process after deleting banned entries. That was wrong twice over:

      * `name.startswith(("sentiment", "tft", ...))` tests the START of the
        dotted name, and the real names are `models.sentiment.model` -- so the
        cleanup deleted nothing.
      * more fundamentally, by the time this test runs under the full suite,
        dozens of OTHER tests have already imported those layers. Any in-process
        check therefore measures what pytest imported, not what the live path
        imports. It passed when run alone and failed in the suite -- the exact
        signature of a test that is measuring global state.

    A fresh interpreter has no such history, so what it loads is exactly what the
    live path pulls in.
    """
    import json
    import subprocess
    import textwrap

    code = textwrap.dedent(f"""
        import importlib, json, sys
        for ep in {list(entrypoints)!r}:
            importlib.import_module(ep)
        pre = ('models.', 'analysis.', 'data.', 'ops.', 'portfolio.',
               'risk.', 'backtest.', 'research.')
        print(json.dumps(sorted(m for m in sys.modules if m.startswith(pre))))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, f"live path failed to import:\n{out.stderr[-2000:]}"
    return set(json.loads(out.stdout.strip().splitlines()[-1]))


def test_no_unvalidated_layer_is_reachable_from_the_live_path():
    """The tripwire. If this fails, an unvalidated model is now in the live
    trade path and the go-live evidence base no longer describes what trades."""
    loaded = _modules_reachable_from()
    offenders = sorted(m for m in loaded if any(k in m for k in BANNED))
    assert not offenders, (
        f"unvalidated layers reachable from the live path: {offenders}. "
        "None of these has beaten a baseline out-of-sample. Wiring one in is a "
        "deliberate decision that must delete this test, not a refactor side effect."
    )


def test_live_min_var_only_flag_is_still_armed():
    """generate_trade_list raises if this is False. It is the last line of
    defence for the signal-SELECTION path, which measured -18% Brinson
    selection."""
    assert config.LIVE_MIN_VAR_ONLY is True


def test_trade_list_refuses_to_run_with_selection_enabled(monkeypatch):
    """Prove the flag actually blocks, rather than merely being True."""
    from ops import trade_list
    monkeypatch.setattr(config, "LIVE_MIN_VAR_ONLY", False)
    with pytest.raises(RuntimeError) as e:
        trade_list.generate_trade_list()
    assert "LIVE_MIN_VAR_ONLY" in str(e.value)


def test_sentiment_coverage_is_still_structurally_insufficient():
    """Item 14 says re-verify ONLY if the data situation changes. This makes
    'has it changed?' a test rather than something someone has to remember to
    check. It fails when coverage becomes non-trivial -- which is the signal to
    re-run the baseline comparison, not to wire anything in."""
    import pandas as pd
    from data.database import get_connection
    with get_connection() as conn:
        row = pd.read_sql(
            "SELECT COUNT(*) n, SUM(CASE WHEN sentiment_score IS NOT NULL "
            "AND sentiment_score != 0 THEN 1 ELSE 0 END) nz FROM signals", conn)
    n, nz = int(row["n"][0]), int(row["nz"][0] or 0)
    if n == 0:
        pytest.skip("no signals rows in this database")
    coverage = nz / n
    assert coverage < 0.05, (
        f"sentiment coverage is now {coverage:.1%} (was 0%). The data situation "
        "has changed -- re-run the out-of-sample baseline comparison against "
        "momentum/buy-and-hold BEFORE considering the live path. Coverage alone "
        "is not evidence the signal works."
    )


def test_checkpoints_on_disk_do_not_imply_wiring():
    """Both checkpoints exist. Their presence must not be mistaken for the
    models being in use -- this documents the distinction in an executable form."""
    import os
    tft = getattr(config, "TFT_MODEL_CHECKPOINT_PATH", None)
    sent = getattr(config, "SENTIMENT_MODEL_CHECKPOINT_PATH", None)
    # existence is fine; reachability is not
    loaded = _modules_reachable_from()
    assert not any("tft" in m for m in loaded)
    assert not any("sentiment" in m for m in loaded)
    # if a checkpoint is missing that is also fine -- the test is about wiring
    assert tft is None or isinstance(tft, str)
    assert sent is None or isinstance(sent, str)
