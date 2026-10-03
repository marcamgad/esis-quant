"""
Layer 8: risk management + performance attribution, from scratch.

  metrics.py       VaR (historical + Cornish-Fisher), drawdown analytics,
                   normal inverse-CDF (Acklam) -- all no-scipy.
  overlay.py       regime de-risking overlay + drawdown circuit breaker (with
                   an explicit re-entry rule) + Kelly/drawdown position sizing.
  attribution.py   Brinson-Hood-Beebower allocation/selection/interaction plus
                   a regime-timing term, reconciled to realized active return.
"""
