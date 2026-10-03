

# --- flat per-order fee must be charged, not just the percentage leg -------

def test_roundtrip_charges_the_flat_per_order_fee():
    """A percentage-only model reports near-zero cost on tiny orders. The flat
    EGP 2 is most of the real cost at that size and must appear."""
    import pandas as pd
    from backtest.costs import book_roundtrip_cost
    w = pd.Series({f"T{i}": 1 / 18 for i in range(18)})
    adv = pd.Series({t: 1e9 for t in w.index})        # infinite liquidity -> no impact
    d = book_roundtrip_cost(w, capital_egp=1_000.0, adv=adv)
    assert d["n_orders"] == 18
    assert d["flat_fee_share"] > 0.8                   # flat fee dominates at this size
    assert d["roundtrip_cost"] > 0.05                  # >5%, not the old ~0.5%


def test_splitting_the_same_capital_over_more_names_costs_more():
    """Same notional, more orders -> strictly more flat fee. This is the whole
    reason concentration matters at small size."""
    import pandas as pd
    from backtest.costs import book_roundtrip_cost
    adv = pd.Series({f"T{i}": 1e9 for i in range(20)})
    few = book_roundtrip_cost(pd.Series({f"T{i}": 0.2 for i in range(5)}), 1_000.0, adv=adv)
    many = book_roundtrip_cost(pd.Series({f"T{i}": 0.05 for i in range(20)}), 1_000.0, adv=adv)
    assert many["roundtrip_cost"] > few["roundtrip_cost"]
    assert many["total_egp"] - few["total_egp"] > 50   # 30 extra orders x ~2 EGP


def test_flat_fee_share_fades_as_the_book_grows():
    """At large size the percentage/impact leg takes over. The flat fee must
    become negligible rather than staying a fixed fraction."""
    import pandas as pd
    from backtest.costs import book_roundtrip_cost
    w = pd.Series({f"T{i}": 1 / 18 for i in range(18)})
    adv = pd.Series({t: 1e9 for t in w.index})
    small = book_roundtrip_cost(w, capital_egp=1_000.0, adv=adv)
    big = book_roundtrip_cost(w, capital_egp=10_000_000.0, adv=adv)
    assert big["flat_fee_share"] < 0.05 < small["flat_fee_share"]
    assert big["roundtrip_cost"] < small["roundtrip_cost"]
