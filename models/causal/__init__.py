"""
Layer 5: Causal Signal Extraction.

Separates genuine alpha from spurious factor-driven correlation using
Double/Debiased Machine Learning (Robinson partialling-out with
cross-fitting) and Instrumental-Variables (2SLS) estimation -- every
estimator implemented from scratch in NumPy (no sklearn / statsmodels /
linearmodels). See lasso.py, double_ml.py, iv_estimator.py, signals.py.
"""
