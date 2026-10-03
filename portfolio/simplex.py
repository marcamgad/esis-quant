"""
A from-scratch two-phase primal simplex LP solver (no scipy). Built to solve
the Rockafellar-Uryasev CVaR linear program in cvar_optimizer.py, but written
as a general solver so it can be unit-tested against hand-solvable LPs.

Solves:  minimize c' x  subject to  A_ub x <= b_ub,  A_eq x = b_eq,  x >= 0.

Method: convert to standard form (add non-negative slacks to the <=
constraints), then run two-phase simplex with artificial variables for the
equalities/converted rows. Phase 1 minimizes the sum of artificials to find a
basic feasible solution; Phase 2 minimizes the real objective. Pivoting uses
Bland's rule (smallest-index entering/leaving variable), which is provably
cycle-free -- slower than Dantzig's rule but it cannot stall on the degenerate
vertices the CVaR LP produces (many binding u_t >= 0 constraints).
"""

import numpy as np


def _simplex_phase(T, basis, n_cols, max_iter):
    """Run simplex pivots on tableau T (last column = RHS, last row = reduced
    costs) until optimal, using Bland's rule. Mutates T and basis in place.
    """
    m = T.shape[0] - 1
    for _ in range(max_iter):
        # entering: first column with negative reduced cost (Bland)
        reduced = T[-1, :n_cols]
        entering = -1
        for j in range(n_cols):
            if reduced[j] < -1e-9:
                entering = j
                break
        if entering == -1:
            return True  # optimal
        # ratio test: leaving = smallest index among min-ratio rows (Bland)
        col = T[:m, entering]
        rhs = T[:m, -1]
        best_ratio, leaving = np.inf, -1
        for i in range(m):
            if col[i] > 1e-9:
                ratio = rhs[i] / col[i]
                if ratio < best_ratio - 1e-12 or (abs(ratio - best_ratio) < 1e-12 and (leaving == -1 or basis[i] < basis[leaving])):
                    best_ratio, leaving = ratio, i
        if leaving == -1:
            return False  # unbounded
        # pivot
        T[leaving] /= T[leaving, entering]
        for i in range(T.shape[0]):
            if i != leaving and abs(T[i, entering]) > 1e-15:
                T[i] -= T[i, entering] * T[leaving]
        basis[leaving] = entering
    return True


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, max_iter: int = 20000) -> dict:
    """minimize c'x s.t. A_ub x <= b_ub, A_eq x = b_eq, x >= 0.

    Returns {'x', 'fun', 'status'} where status is 'optimal' | 'infeasible' |
    'unbounded'. x has the length of c (structural variables only).
    """
    c = np.asarray(c, dtype=float)
    n = len(c)
    rows_ub = np.asarray(A_ub, dtype=float) if A_ub is not None else np.zeros((0, n))
    b_ub = np.asarray(b_ub, dtype=float) if b_ub is not None else np.zeros(0)
    rows_eq = np.asarray(A_eq, dtype=float) if A_eq is not None else np.zeros((0, n))
    b_eq = np.asarray(b_eq, dtype=float) if b_eq is not None else np.zeros(0)

    n_ub, n_eq = len(b_ub), len(b_eq)
    # standard form: [structural | slacks(n_ub)] ; equalities have no slack
    n_slack = n_ub
    A = np.zeros((n_ub + n_eq, n + n_slack))
    b = np.zeros(n_ub + n_eq)
    if n_ub:
        A[:n_ub, :n] = rows_ub
        A[:n_ub, n:n + n_slack] = np.eye(n_ub)
        b[:n_ub] = b_ub
    if n_eq:
        A[n_ub:, :n] = rows_eq
        b[n_ub:] = b_eq
    # make RHS non-negative (flip rows) so artificials start feasible
    for i in range(len(b)):
        if b[i] < 0:
            A[i] *= -1
            b[i] *= -1

    m, total = A.shape
    # add artificial variables (one per row) for a trivial initial basis
    T = np.zeros((m + 1, total + m + 1))
    T[:m, :total] = A
    T[:m, total:total + m] = np.eye(m)
    T[:m, -1] = b
    basis = list(range(total, total + m))
    # Phase 1 objective: minimize sum of artificials
    T[-1, total:total + m] = 1.0
    for i in range(m):  # price out artificials from the objective row
        T[-1] -= T[i]
    if not _simplex_phase(T, basis, total + m, max_iter):
        return {"x": None, "fun": None, "status": "unbounded"}
    # phase-1 objective value = -T[-1,-1]; a positive residual (T[-1,-1] < 0)
    # means the artificials couldn't be driven out -> infeasible.
    if T[-1, -1] < -1e-7:
        return {"x": None, "fun": None, "status": "infeasible"}

    # Phase 2: restore real objective (structural costs only), drop artificials
    T[-1, :] = 0.0
    T[-1, :n] = c
    T[-1, -1] = 0.0
    for i in range(m):  # price out current basic variables
        if basis[i] < total and abs(T[-1, basis[i]]) > 1e-15:
            T[-1] -= T[-1, basis[i]] * T[i]
    # forbid artificials from re-entering by pricing them prohibitively (they
    # sit at zero; Bland won't pick a non-negative reduced cost anyway)
    if not _simplex_phase(T, basis, total, max_iter):
        return {"x": None, "fun": None, "status": "unbounded"}

    x = np.zeros(total)
    for i in range(m):
        if basis[i] < total:
            x[basis[i]] = T[i, -1]
    x_struct = x[:n]
    return {"x": x_struct, "fun": float(c @ x_struct), "status": "optimal"}
