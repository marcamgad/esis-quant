"""
Known-answer tests for the from-scratch two-phase simplex LP solver, against
LPs whose optima are derivable by hand.
"""

import numpy as np

from portfolio.simplex import linprog


def test_simple_max_as_min():
    # maximize x+y s.t. x+2y<=4, 3x+2y<=6, x,y>=0  ->  min -(x+y)
    # optimum at (1,1.5): objective 2.5 (vertex of the two constraints)
    res = linprog(c=[-1, -1], A_ub=[[1, 2], [3, 2]], b_ub=[4, 6])
    assert res["status"] == "optimal"
    assert np.allclose(res["x"], [1.0, 1.5], atol=1e-6)
    assert res["fun"] == pytest_approx(-2.5)


def test_equality_constraint():
    # min x+y s.t. x+y=1, x,y>=0 -> objective 1 for any split
    res = linprog(c=[1, 1], A_eq=[[1, 1]], b_eq=[1])
    assert res["status"] == "optimal"
    assert res["fun"] == pytest_approx(1.0)
    assert abs(res["x"].sum() - 1.0) < 1e-6


def test_bounded_min_picks_cheapest():
    # min 2x+3y s.t. x+y=1, x<=0.4 (as A_ub), x,y>=0
    # y is cheaper is FALSE (x cheaper): push x to its cap 0.4, y=0.6 -> 0.8+1.8=2.6
    res = linprog(c=[2, 3], A_ub=[[1, 0]], b_ub=[0.4], A_eq=[[1, 1]], b_eq=[1])
    assert res["status"] == "optimal"
    assert np.allclose(res["x"], [0.4, 0.6], atol=1e-6)
    assert res["fun"] == pytest_approx(2.6)


def test_infeasible_detected():
    # x>=0, x = -1 (via equality with positive var) is infeasible
    res = linprog(c=[1], A_eq=[[1]], b_eq=[-1])  # x = -1 but x>=0
    assert res["status"] == "infeasible"


def test_unbounded_detected():
    # min -x with no upper bound on x -> unbounded below
    res = linprog(c=[-1], A_ub=None, b_ub=None)
    assert res["status"] == "unbounded"


# tiny local approx helper (avoid importing pytest.approx at module top for clarity)
def pytest_approx(v, tol=1e-6):
    class _A:
        def __eq__(self, other):
            return abs(other - v) < tol
    return _A()
