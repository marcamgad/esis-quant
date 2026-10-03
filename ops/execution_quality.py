"""
Phase 0.2 / Part 1 -- measured execution quality, closing the manual-execution
blind spot.

THE BLIND SPOT. The workflow is human-in-the-loop and always will be: Thndr has
no public trading API, so `ops/order_ticket.py` prints a target at the close of
session t, a person types it into the broker app some time later, and the fill
lands at whatever the market gave. Until now only shares and fill price were
kept. Nothing recorded WHEN the human executed, and nothing compared the fill to
the SIGNAL price the optimizer actually sized on.

WHY THAT MATTERS MORE THAN IT SOUNDS. Every cost number in this project is
modelled, not measured: backtest/costs.py assumes a slippage coefficient that is
a literature default, and backtest/fees.py's non-commission terms are stated
placeholders. Without fill data there is no way to check either against the
account's reality -- the system could be wrong about its single largest
controllable expense indefinitely and never find out. Implementation shortfall
is the number that closes that loop:

    shortfall = (fill_price - signal_price) / signal_price     [BUY]
    shortfall = (signal_price - fill_price) / signal_price     [SELL]

Signed so POSITIVE always means "worse than the price the decision was made at",
whichever side you were on. Aggregated over enough fills, its mean is the
account's true one-way slippage and its dispersion is the execution risk.

WHAT THIS DELIBERATELY DOES NOT DO. It does not build a broker integration --
none is available, and inventing one would be fiction. It does not replace the
ticket workflow. It makes the existing manual workflow MEASURABLE, which is the
only honest improvement available here.

DELAY IS RECORDED, NOT ASSUMED. `filled_at` is entered by the user. If it is
omitted the row is still stored with a NULL delay rather than a guessed one --
a missing measurement must never be silently replaced by a plausible number.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.execution_quality")

# A single fill worse than this vs the signal price is flagged for review --
# on a 1,000 EGP book that is real money and usually means the order was placed
# into a gap rather than at the price the book was sized on.
SHORTFALL_ALERT = 0.02


def implementation_shortfall(side: str, signal_price: float, fill_price: float) -> float:
    """Signed so positive is ALWAYS unfavourable, for buys and sells alike."""
    if not signal_price or signal_price <= 0:
        return float("nan")
    raw = (fill_price - signal_price) / signal_price
    return raw if str(side).lower() == "buy" else -raw


def record_fill(ticker: str, side: str, shares: float, signal_price: float,
                fill_price: float, ticket_date=None, as_of_date=None,
                ticket_generated_at=None, filled_at=None) -> dict:
    """Store one filled line item with its measured shortfall."""
    ticket_date = str(ticket_date or pd.Timestamp.now().strftime("%Y-%m-%d"))
    sf = implementation_shortfall(side, signal_price, fill_price)
    delay = None
    if ticket_generated_at and filled_at:
        try:
            delay = float((pd.Timestamp(filled_at) - pd.Timestamp(ticket_generated_at))
                          .total_seconds() / 60.0)
        except Exception:  # noqa: BLE001 -- a bad timestamp must not lose the fill
            delay = None
    notional = float(shares) * float(fill_price)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO execution_quality (ticket_date, as_of_date, ticket_generated_at, "
            "filled_at, side, ticker, shares, signal_price, fill_price, shortfall, "
            "notional, delay_minutes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (ticket_date, str(as_of_date) if as_of_date else None,
             str(ticket_generated_at) if ticket_generated_at else None,
             str(filled_at) if filled_at else None, str(side).lower(), ticker,
             float(shares), float(signal_price), float(fill_price),
             float(sf) if sf == sf else None, notional, delay))
    if sf == sf and sf > SHORTFALL_ALERT:
        logger.warning("record_fill: %s %s shortfall %.2f%% vs signal price %.2f",
                       side, ticker, sf * 100, signal_price)
    return {"ticker": ticker, "side": side, "shortfall": sf, "notional": notional,
            "delay_minutes": delay}


def load_fills() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT * FROM execution_quality ORDER BY id", conn)


def summary(min_fills: int = 1) -> dict:
    """Measured one-way slippage vs the modelled assumption."""
    import config
    df = load_fills()
    # A fill whose price was DERIVED from reported position values is not a
    # measurement of execution quality -- its "shortfall" partly measures the
    # reconstruction. Counting those alongside real contract notes inflated the
    # apparent sample and mixed two different quantities.
    # Only fills from the REAL account count. 'reconstructed' fills had prices
    # derived from reported position values (that measures the arithmetic, not
    # the execution), and 'simulator' fills came from the Thndr simulator before
    # the 2026-08-09 switch to real money -- simulator fills are not evidence
    # about real-money execution. Both are kept in the table and excluded here.
    if "source" in df.columns:
        df = df[df["source"].fillna("measured") == "measured"]
    if len(df) < min_fills or df.empty:
        return {"n_fills": int(len(df))}
    sf = df["shortfall"].dropna()
    # notional-weighted is the number that matters for cost: a bad fill on a
    # 15 EGP line matters far less than the same percentage on a 300 EGP line
    w = df.loc[sf.index, "notional"]
    wmean = float((sf * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
    return {"n_fills": int(len(df)), "mean_shortfall": float(sf.mean()),
            "weighted_mean_shortfall": wmean, "median_shortfall": float(sf.median()),
            "worst": float(sf.max()), "best": float(sf.min()),
            "std": float(sf.std(ddof=1)) if len(sf) > 1 else 0.0,
            "pct_unfavourable": float((sf > 0).mean()),
            "total_cost_egp": float((sf * w).sum()),
            "modelled_slippage": config.SLIPPAGE_PCT,
            "mean_delay_minutes": float(df["delay_minutes"].dropna().mean())
            if df["delay_minutes"].notna().any() else None}


def by_period(freq: str = "W") -> pd.DataFrame:
    """Weekly/monthly aggregation, so a drift in execution quality is visible."""
    df = load_fills()
    if df.empty:
        return pd.DataFrame()
    df["ticket_date"] = pd.to_datetime(df["ticket_date"])
    g = df.set_index("ticket_date").groupby(pd.Grouper(freq=freq))
    out = g.apply(lambda x: pd.Series({
        "n_fills": len(x),
        "wmean_shortfall": (x["shortfall"] * x["notional"]).sum() / x["notional"].sum()
        if x["notional"].sum() > 0 else np.nan,
        "notional": x["notional"].sum(),
        "cost_egp": (x["shortfall"] * x["notional"]).sum()}))
    return out[out["n_fills"] > 0]


def format_summary(s: dict) -> str:
    if not s or s.get("n_fills", 0) == 0:
        return ("  execution quality: no fills recorded yet.\n"
                "  (fills are captured by ops.order_ticket apply -- enter PAID_EGP + FEE_EGP)")
    lines = [f"EXECUTION QUALITY — {s['n_fills']} fill(s) measured",
             f"  weighted mean shortfall : {s['weighted_mean_shortfall']*100:+.3f}%"
             f"   (modelled slippage assumption {s['modelled_slippage']*100:.2f}%)",
             f"  median / worst / best   : {s['median_shortfall']*100:+.3f}% / "
             f"{s['worst']*100:+.3f}% / {s['best']*100:+.3f}%",
             f"  unfavourable fills      : {s['pct_unfavourable']*100:.0f}%",
             f"  total slippage cost     : {s['total_cost_egp']:+,.2f} EGP"]
    if s.get("mean_delay_minutes") is not None:
        lines.append(f"  mean delay to fill      : {s['mean_delay_minutes']:.0f} min")
    if s["n_fills"] < 20:
        lines.append(f"  NOTE: {s['n_fills']} fills is too few to conclude anything about "
                     "the cost model.")
    else:
        d = s["weighted_mean_shortfall"] - s["modelled_slippage"]
        verdict = ("model UNDERSTATES real slippage" if d > 0.001 else
                   "model overstates real slippage" if d < -0.001 else
                   "model is consistent with reality")
        lines.append(f"  -> measured minus modelled = {d*100:+.3f}pp — {verdict}")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    print(format_summary(summary()))


# --- feeding measured slippage back into the cost model  (audit P4.9) --------

MIN_FILLS_FOR_ESTIMATE = 20      # below this the estimate is noise, not evidence
SHRINK_PRIOR_FILLS = 20          # weight given to the modelled prior


def measured_slippage(min_fills: int = None) -> dict:
    """Rolling slippage estimate from real fills, shrunk toward the model.

    WHY SHRINKAGE RATHER THAN A HAND-EDITED CONSTANT. The audit found
    config.SLIPPAGE_PCT (0.10%) contradicted by a measured +2.745% weighted
    shortfall. The wrong response is to retype the constant as 2.745%: that
    number came from 8 fills, all on one day, all measured against a signal
    price two sessions stale -- so it overstates true slippage by an unknown
    amount. The other wrong response is to keep 0.10% because the sample is
    small.

    So: an empirical-Bayes blend. The estimate starts AT the model and moves
    toward the measurement as fills accumulate, with weight n / (n + prior). At
    8 fills the estimate is still mostly the prior; by 60 fills it is mostly the
    data. Nothing is hand-tuned to a target and the raw measurement is always
    reported alongside, so the shrinkage can never hide it.

    `usable` is False below MIN_FILLS_FOR_ESTIMATE -- callers should keep using
    the modelled prior until then, and the flag says so rather than leaving the
    caller to guess whether 8 fills is enough. (It is not.)
    """
    import config
    prior = float(config.SLIPPAGE_PCT)
    s = summary()
    n = int(s.get("n_fills", 0))
    if n == 0:
        return {"n_fills": 0, "prior": prior, "measured": float("nan"),
                "estimate": prior, "usable": False, "weight_on_data": 0.0,
                "note": "no fills recorded yet — using the modelled prior"}
    raw = float(s.get("weighted_mean_shortfall", float("nan")))
    if raw != raw:
        return {"n_fills": n, "prior": prior, "measured": float("nan"),
                "estimate": prior, "usable": False, "weight_on_data": 0.0,
                "note": "fills recorded but shortfall not computable"}
    w = n / (n + SHRINK_PRIOR_FILLS)
    est = (1 - w) * prior + w * max(raw, 0.0)
    usable = n >= (min_fills or MIN_FILLS_FOR_ESTIMATE)
    return {"n_fills": n, "prior": prior, "measured": raw, "estimate": est,
            "weight_on_data": w, "usable": usable,
            "note": ("" if usable else
                     f"{n} fills is below the {MIN_FILLS_FOR_ESTIMATE}-fill floor — "
                     "treat as indicative, not settled")}


def format_measured_slippage(d: dict) -> str:
    if not d:
        return ""
    lines = [f"SLIPPAGE — modelled {d['prior']*100:.2f}% vs measured "
             f"{d['measured']*100:+.3f}% over {d['n_fills']} fill(s)"]
    lines.append(f"  shrunk estimate {d['estimate']*100:.3f}% "
                 f"({d['weight_on_data']*100:.0f}% weight on data)")
    if d["note"]:
        lines.append(f"  {d['note']}")
    lines.append("  NOTE: shortfall is measured against the signal price (the as-of "
                 "close). When the ticket is traded a session or more later, price "
                 "DRIFT is counted as slippage — so this is an upper bound.")
    return "\n".join(lines)
