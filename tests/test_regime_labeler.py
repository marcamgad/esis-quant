import numpy as np

from models.regime_labeler import label_states, label_columns_for_gamma, BULL, BEAR, HIGH_VOL, RECOVERY
from models.regime_features import OBSERVATION_COLUMNS

RET = OBSERVATION_COLUMNS.index("o1_market_return_20d")
VOL = OBSERVATION_COLUMNS.index("o2_realized_vol_20d")
D = len(OBSERVATION_COLUMNS)


def _mu_row(return_, vol_):
    row = np.zeros(D)
    row[RET] = return_
    row[VOL] = vol_
    return row


def _row_stochastic(rows):
    A = np.array(rows, dtype=float)
    return A / A.sum(axis=1, keepdims=True)


def test_label_states_clear_case():
    # state 0: strong positive return -> BULL
    # state 1: strong negative return -> BEAR
    # state 2: modest positive return, high vol -> HIGH_VOL
    # state 3: modest positive return, low vol, and BEAR transitions into it
    #          most often among the candidates -> RECOVERY
    mu = np.array([
        _mu_row(0.08, 0.15),
        _mu_row(-0.10, 0.35),
        _mu_row(0.02, 0.45),
        _mu_row(0.01, 0.20),
    ])
    A = _row_stochastic([
        [0.90, 0.03, 0.03, 0.04],
        [0.05, 0.85, 0.03, 0.07],  # from BEAR(1): highest off-diagonal mass goes to state 3
        [0.05, 0.05, 0.85, 0.05],
        [0.05, 0.05, 0.05, 0.85],
    ])

    labels = label_states(mu, A)
    assert labels[0] == BULL
    assert labels[1] == BEAR
    assert labels[2] == HIGH_VOL
    assert labels[3] == RECOVERY


def test_label_states_uses_transitions_not_just_volatility():
    # Same emission means as a case where a pure vol-ranking heuristic
    # would call state 2 HIGH_VOL (it has the higher vol, 0.50 > 0.10) --
    # but here BEAR transitions into state 2 far more than into state 3,
    # so the transition-aware heuristic must call state 2 RECOVERY instead,
    # even though it's the higher-vol state. This is the concrete
    # regression case for the mislabeling found on real EGX data (where a
    # vol-based tiebreak among leftover states gave an economically wrong
    # answer for RECOVERY).
    mu = np.array([
        _mu_row(0.08, 0.20),
        _mu_row(-0.10, 0.30),
        _mu_row(0.01, 0.50),
        _mu_row(0.02, 0.10),
    ])
    A = _row_stochastic([
        [0.90, 0.03, 0.03, 0.04],
        [0.05, 0.84, 0.08, 0.03],  # from BEAR(1): state 2 gets more mass than state 3
        [0.05, 0.05, 0.85, 0.05],
        [0.05, 0.05, 0.05, 0.85],
    ])

    labels = label_states(mu, A)
    assert labels[1] == BEAR
    assert labels[2] == RECOVERY  # higher-vol state, but that's where BEAR transitions to
    assert labels[3] == HIGH_VOL  # lower-vol state, but not where BEAR recovers into
    assert labels[0] == BULL


def test_label_states_rejects_non_four_state():
    mu = np.zeros((3, D))
    A = np.eye(3)
    try:
        label_states(mu, A)
        assert False, "expected ValueError for K != 4"
    except ValueError:
        pass


def test_label_columns_for_gamma_orders_correctly():
    mu = np.array([
        _mu_row(0.08, 0.15),
        _mu_row(-0.10, 0.35),
        _mu_row(0.02, 0.45),
        _mu_row(0.01, 0.20),
    ])
    A = _row_stochastic([
        [0.90, 0.03, 0.03, 0.04],
        [0.05, 0.85, 0.03, 0.07],
        [0.05, 0.05, 0.85, 0.05],
        [0.05, 0.05, 0.05, 0.85],
    ])
    labels = label_states(mu, A)
    cols = label_columns_for_gamma(labels)
    # state 0 -> BULL -> prob_bull should be first column since label_map[0] == BULL
    assert cols[0] == "prob_bull"
    assert cols[1] == "prob_bear"
    assert cols[2] == "prob_high_vol"
    assert cols[3] == "prob_recovery"
