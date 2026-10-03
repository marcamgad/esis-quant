"""
Part 3 -- daily broker reconciliation: is the internal book still describing
reality?

WHY THIS IS NOT OPTIONAL. Everything in ops/portfolio_tracker.py is the system's
OWN arithmetic: it adds what it was told was bought and subtracts what it was
told was sold. Nothing in it is anchored to the account. Every source of drift
is silent -- a fill typed in wrong, a partial fill, a fee the model doesn't
charge, a dividend, a corporate action, an order that filled at a different
size. Each is individually small and none raises an error, which is exactly why
the internal record can stop matching the account without anyone noticing. A
trade record that has quietly stopped being true is worse than no record,
because decisions keep being made on it.

This module compares the internal book against the broker's own figures and
FAILS LOUD on divergence, reusing ops/health.alert -- the same mechanism that
already exists for the 2026-07-23 silent-outage class of failure. No new
alerting path is introduced.

INPUT, REALISTICALLY. Thndr has no public trading API, so there is no automated
pull. The input is therefore whatever the user can actually produce: a CSV
export, or the two numbers the app shows on its main screen (total stock value
and cash). Both are supported, because a reconciliation that demands data the
user cannot obtain will simply never be run:

  * `reconcile_from_csv(path)` -- columns ticker,shares[,price] plus an optional
    CASH row. Full position-level check.
  * `reconcile_totals(stocks_value, cash)` -- the two-number check. Weaker: it
    can detect that the book is wrong but not WHICH position is wrong.

TOLERANCES. Cash is compared in EGP with a small absolute tolerance (rounding
and sub-piastre fees are not breaks). Share counts must match EXACTLY -- a
fractional-share discrepancy is not a rounding artifact on an exchange that
trades whole shares, it is a real error.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection
from ops import portfolio_tracker as pt

logger = logging.getLogger("esis_quant.reconcile")

CASH_TOLERANCE_EGP = 1.00      # rounding / sub-piastre fees, not a break
VALUE_TOLERANCE_PCT = 0.01     # 1% on marked stock value (prices differ by timing)


def _internal_state() -> tuple:
    pos, cash = pt.holdings()
    shares = (pos.set_index("ticker")["shares"].astype(float).to_dict()
              if not pos.empty else {})
    value = float(pos["value"].sum()) if not pos.empty else 0.0
    return shares, float(cash), value


def reconcile_positions(broker_shares: dict, broker_cash: float,
                        recon_date=None, store: bool = True) -> dict:
    """Position-level diff between the internal book and the broker."""
    internal, cash_int, _ = _internal_state()
    names = sorted(set(internal) | set(broker_shares))
    breaks = []
    for t in names:
        i = float(internal.get(t, 0.0))
        b = float(broker_shares.get(t, 0.0))
        if abs(i - b) > 1e-9:
            breaks.append({"ticker": t, "internal": i, "broker": b, "diff": i - b})
    cash_diff = cash_int - float(broker_cash)
    ok = (not breaks) and abs(cash_diff) <= CASH_TOLERANCE_EGP

    detail = "; ".join(f"{b['ticker']}: internal {b['internal']:g} vs broker {b['broker']:g}"
                       for b in breaks) or "no position breaks"
    result = {"ok": ok, "breaks": breaks, "n_breaks": len(breaks),
              "cash_internal": cash_int, "cash_broker": float(broker_cash),
              "cash_diff": cash_diff, "detail": detail,
              "recon_date": str(recon_date or pd.Timestamp.now().strftime("%Y-%m-%d"))}
    if store:
        _store(result)
    if not ok:
        _alert(result)
    return result


def reconcile_totals(stocks_value: float, cash: float, recon_date=None,
                     store: bool = True) -> dict:
    """Two-number check when a full export isn't available.

    Deliberately labelled WEAKER in its own output: it can prove the book is
    wrong but cannot say which position caused it, so it must never be reported
    as an equivalent substitute for the position-level check.
    """
    _, cash_int, value_int = _internal_state()
    cash_diff = cash_int - float(cash)
    val_diff = value_int - float(stocks_value)
    val_pct = abs(val_diff) / float(stocks_value) if stocks_value else np.inf
    ok = abs(cash_diff) <= CASH_TOLERANCE_EGP and val_pct <= VALUE_TOLERANCE_PCT
    result = {"ok": ok, "breaks": [], "n_breaks": 0, "mode": "totals_only",
              "cash_internal": cash_int, "cash_broker": float(cash), "cash_diff": cash_diff,
              "value_internal": value_int, "value_broker": float(stocks_value),
              "value_diff": val_diff, "value_diff_pct": val_pct,
              "detail": (f"totals-only check: stock value internal {value_int:,.2f} vs broker "
                         f"{stocks_value:,.2f} ({val_pct*100:.2f}%); cash diff {cash_diff:+.2f} EGP"),
              "recon_date": str(recon_date or pd.Timestamp.now().strftime("%Y-%m-%d"))}
    if store:
        _store(result)
    if not ok:
        _alert(result)
    return result


def reconcile_from_csv(path: str, recon_date=None) -> dict:
    """Read a broker export: columns ticker,shares (+ optional CASH row)."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol, scol = cols.get("ticker"), cols.get("shares")
    if not tcol or not scol:
        raise ValueError("broker CSV needs at least 'ticker' and 'shares' columns")
    cash = 0.0
    shares = {}
    for r in df.itertuples():
        t = str(getattr(r, tcol)).strip().upper()
        v = float(getattr(r, scol) or 0.0)
        if t in ("CASH", "BALANCE"):
            cash = v
        elif t and t != "NAN":
            shares[t] = v
    return reconcile_positions(shares, cash, recon_date=recon_date)


def _store(result: dict):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO broker_reconciliation (recon_date, checked_at, ok, n_position_breaks, "
            "cash_internal, cash_broker, cash_diff, detail) VALUES (?,?,?,?,?,?,?,?)",
            (result["recon_date"], pd.Timestamp.now().isoformat(timespec="seconds"),
             1 if result["ok"] else 0, int(result["n_breaks"]),
             float(result["cash_internal"]), float(result["cash_broker"]),
             float(result["cash_diff"]), str(result["detail"])[:2000]))


def _alert(result: dict):
    """Reuse the existing health alert -- do not invent a second mechanism."""
    try:
        from ops.health import alert
        alert("ESIS-QUANT — BOOK DOES NOT MATCH BROKER",
              f"{result['n_breaks']} position break(s); cash differs by "
              f"{result['cash_diff']:+.2f} EGP.\n\n{result['detail']}\n\n"
              "The internal trade record has stopped describing the account. "
              "Fix before acting on any further suggestion.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconcile: alert failed: %s", exc)


def last_reconciliation() -> dict:
    with get_connection() as conn:
        row = conn.execute("SELECT recon_date, checked_at, ok, n_position_breaks, cash_diff, "
                           "detail FROM broker_reconciliation ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else {}


def is_stale(max_age_days: int = 3) -> bool:
    """Has the book gone unchecked for too long? Absence of a check is itself a
    problem -- the same reasoning as ops/health.is_stale."""
    last = last_reconciliation()
    if not last:
        return True
    age = (pd.Timestamp.now().normalize()
           - pd.Timestamp(last["checked_at"]).normalize()).days
    return age > max_age_days


def format_result(r: dict) -> str:
    if not r:
        return "  reconciliation: never run."
    if r["ok"]:
        return (f"  RECONCILED — book matches broker "
                f"(cash diff {r['cash_diff']:+.2f} EGP)")
    lines = ["  *** BOOK DOES NOT MATCH BROKER ***",
             f"    position breaks : {r['n_breaks']}",
             f"    cash            : internal {r['cash_internal']:,.2f} vs broker "
             f"{r['cash_broker']:,.2f}  (diff {r['cash_diff']:+.2f})"]
    for b in r.get("breaks", [])[:10]:
        lines.append(f"      {b['ticker']:6} internal {b['internal']:g} vs broker {b['broker']:g}")
    lines.append("    do not act on further suggestions until this is resolved.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.ERROR)
    if len(sys.argv) >= 4 and sys.argv[1] == "totals":
        print(format_result(reconcile_totals(float(sys.argv[2]), float(sys.argv[3]))))
    elif len(sys.argv) >= 3 and sys.argv[1] == "csv":
        print(format_result(reconcile_from_csv(sys.argv[2])))
    else:
        print(format_result(last_reconciliation()))
        print("\nusage:  py -3.11 -m ops.reconcile totals <stocks_value> <cash>")
        print("        py -3.11 -m ops.reconcile csv <broker_export.csv>")
