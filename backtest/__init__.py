"""
Walk-forward backtesting for ESIS-QUANT. The single most important honesty
tool in the system: turns a per-(ticker, date) signal into a simulated
long portfolio, marks it to market on ADJUSTED prices, charges realistic
transaction costs on turnover, and reports P&L / Sharpe / drawdown against
baselines -- so "this signal looks good" becomes "this signal made or lost
X% after costs". engine.py runs the simulation; analytics.py computes the
performance statistics (all from scratch).
"""
