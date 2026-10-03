"""
Production operations layer (audit Part 2). Everything here exists because
the audit found the system had NO live decision path at all: it could
backtest, but it could not produce, log, or forward-test an actual trade list.

  validate.py      startup config/DB sanity -- fail loudly, never degrade
  trade_list.py    produce + log today's target book (gated on data quality)
  paper_trading.py independent forward mark-to-market record (NOT the backtest)
"""
