"""
Layer 7: Portfolio construction, all optimizers implemented from scratch
(no scipy.optimize / cvxpy). Given the Phase-A finding that this system's
measured edge is risk management rather than stock selection, the
risk-based optimizers here (min-variance via Ledoit-Wolf-shrunk covariance,
equal-risk-contribution, CVaR) are the load-bearing components.

  covariance.py    Ledoit-Wolf shrinkage + capped-simplex projection
  mv_optimizer.py  mean-variance via projected gradient
  risk_parity.py   equal risk contribution via Newton on the log-barrier form
  cvar_optimizer.py Rockafellar-Uryasev CVaR LP via a from-scratch simplex
  ensemble.py      0.40 MV / 0.35 RP / 0.25 CVaR blend + constraints
  signals.py       composite signal (regime-weighted) feeding expected returns
"""
