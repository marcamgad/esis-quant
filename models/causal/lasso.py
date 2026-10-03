"""
Lasso regression via coordinate descent, implemented from scratch (no
sklearn). Used by Layer 5's Double ML to flexibly control for confounders
X in the two nuisance regressions Y ~ g(X) and D ~ m(X).

Objective (spec 5.1):
    (1 / 2N) * ||y - X beta||^2 + lambda * ||beta||_1

Coordinate descent with soft-thresholding. Standard practice -- and what
makes the L1 penalty scale-fair across regressors -- is to standardize
each column of X to zero mean / unit variance before descending, then map
the coefficients back to the original scale afterwards. With standardized
columns, X_j'X_j / N == 1, so the per-coordinate update collapses to the
soft-threshold of the partial-residual correlation rho_j exactly as the
spec writes it:

    r_j   = y - X_{-j} beta_{-j}          (partial residual, excl. column j)
    rho_j = X_j' r_j / N
    beta_j = sign(rho_j) * max(|rho_j| - lambda, 0)      (soft threshold)

The intercept is never penalized: we center y (and the standardized X has
zero-mean columns), fit the slopes penalized, then recover the intercept
as ybar - xbar . beta_original.
"""

import logging

import numpy as np

import config

logger = logging.getLogger("esis_quant.causal.lasso")


def soft_threshold(z: float, lam: float) -> float:
    """Soft-thresholding operator S(z, lambda) = sign(z) * max(|z| - lambda, 0),
    the proximal operator of the L1 norm and the closed-form coordinate
    update for Lasso. Scalar in, scalar out.
    """
    if z > lam:
        return z - lam
    if z < -lam:
        return z + lam
    return 0.0


class LassoCoordinateDescent:
    """Lasso by cyclic coordinate descent.

    fit(X, y, lam) with X shape (N, p), y shape (N,). Stores coefficients in
    the ORIGINAL feature scale (`coef_`, shape (p,)) plus `intercept_`.
    Columns of X with zero variance are handled by forcing their coefficient
    to 0 (they carry no information and would divide by zero on rescaling).
    """

    def __init__(self, max_iter: int = None, tol: float = None):
        self.max_iter = max_iter if max_iter is not None else config.CAUSAL_LASSO_MAX_ITER
        self.tol = tol if tol is not None else config.CAUSAL_LASSO_TOL
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray, lam: float) -> "LassoCoordinateDescent":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, p = X.shape

        mu = X.mean(axis=0)
        sigma = X.std(axis=0)  # population std (ddof=0) => sum(Xs_j^2) == n
        nonconstant = sigma > 1e-12
        Xs = np.zeros_like(X)
        Xs[:, nonconstant] = (X[:, nonconstant] - mu[nonconstant]) / sigma[nonconstant]

        ybar = y.mean()
        yc = y - ybar

        beta = np.zeros(p)
        residual = yc - Xs @ beta  # == yc initially
        for it in range(self.max_iter):
            max_change = 0.0
            for j in range(p):
                if not nonconstant[j]:
                    continue
                # add column j back in to form the partial residual r_j
                residual += Xs[:, j] * beta[j]
                rho = Xs[:, j] @ residual / n  # X_j'X_j/n == 1 for standardized col
                new_bj = soft_threshold(rho, lam)
                max_change = max(max_change, abs(new_bj - beta[j]))
                beta[j] = new_bj
                residual -= Xs[:, j] * beta[j]
            self.n_iter_ = it + 1
            if max_change < self.tol:
                break

        # map standardized-scale betas back to original feature scale
        coef = np.zeros(p)
        coef[nonconstant] = beta[nonconstant] / sigma[nonconstant]
        self.coef_ = coef
        self.intercept_ = float(ybar - mu @ coef)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_


def lambda_grid(n_lambdas: int = None) -> np.ndarray:
    """Log-spaced grid of lambda values (spec 5.1: 50 values, 1e-4 .. 1)."""
    n_lambdas = n_lambdas if n_lambdas is not None else config.CAUSAL_LASSO_N_LAMBDAS
    return np.logspace(
        np.log10(config.CAUSAL_LASSO_LAMBDA_MIN),
        np.log10(config.CAUSAL_LASSO_LAMBDA_MAX),
        n_lambdas,
    )


def time_series_folds(n: int, k: int, purge: int = 0):
    """Yield (train_idx, val_idx) for `k` CONTIGUOUS, time-ordered blocks of
    a length-`n` sample. Fold i validates on block i and trains on every
    OTHER block -- a blocked K-fold that keeps whole contiguous time
    segments together (no random shuffling, spec 5.1: "Use time-series CV
    (not random)").

    `purge` drops that many training observations immediately adjacent to
    each validation block on both sides (De Prado purging + embargo). This
    matters when the label is an h-day-forward return: the last rows of a
    validation block and the first rows of the neighbouring training block
    share overlapping return windows, so without a purge of at least h the
    two sets leak into each other (STAT-LEAK-03). purge=0 (default) leaves
    the plain blocked K-fold used for lambda cross-validation, where the
    Lasso target is contemporaneous and no overlap exists.
    """
    idx = np.arange(n)
    bounds = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        val = idx[lo:hi]
        left = idx[: max(0, lo - purge)]
        right = idx[min(n, hi + purge):]
        train = np.concatenate([left, right])
        if len(train) == 0:
            continue
        yield train, val


def lasso_cv(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray = None,
             k: int = None) -> dict:
    """Pick lambda by time-series cross-validated mean squared error.

    Returns {'best_lambda', 'cv_mse' (per-lambda mean val MSE), 'lambdas'}.
    Falls back to the smallest lambda if every fold is degenerate.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    lambdas = lambda_grid() if lambdas is None else np.asarray(lambdas, dtype=float)
    k = k if k is not None else config.CAUSAL_CV_N_FOLDS

    n = len(y)
    cv_mse = np.full(len(lambdas), np.nan)
    for li, lam in enumerate(lambdas):
        fold_mse = []
        for train, val in time_series_folds(n, k):
            model = LassoCoordinateDescent().fit(X[train], y[train], lam)
            pred = model.predict(X[val])
            fold_mse.append(float(np.mean((y[val] - pred) ** 2)))
        if fold_mse:
            cv_mse[li] = np.mean(fold_mse)

    if np.all(np.isnan(cv_mse)):
        best_lambda = float(lambdas[0])
        logger.warning("lasso_cv: all folds degenerate, defaulting to smallest lambda %g", best_lambda)
    else:
        best_lambda = float(lambdas[int(np.nanargmin(cv_mse))])
    return {"best_lambda": best_lambda, "cv_mse": cv_mse, "lambdas": lambdas}


def lasso_path(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray = None) -> dict:
    """Fit Lasso across the whole lambda grid on the full sample (for the
    regularization-path chart). Coefficients are returned in STANDARDIZED
    scale so magnitudes are comparable across confounders on one plot.

    Returns {'lambdas', 'coefs' (n_lambdas, p standardized), 'feature_std'}.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    lambdas = lambda_grid() if lambdas is None else np.asarray(lambdas, dtype=float)
    sigma = X.std(axis=0)
    coefs = np.zeros((len(lambdas), X.shape[1]))
    for li, lam in enumerate(lambdas):
        model = LassoCoordinateDescent().fit(X, y, lam)
        coefs[li] = model.coef_ * sigma  # original-scale coef * std = standardized-scale coef
    return {"lambdas": lambdas, "coefs": coefs, "feature_std": sigma}
