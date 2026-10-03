"""
Maps a fitted GaussianHMM's arbitrary state indices (0..K-1, meaningless
on their own -- EM has no notion of "bull" or "bear") to the four
economically-meaningful labels from the spec. This is a heuristic, not a
mathematical guarantee -- it can mislabel states in unusual fits, which is
why regime_pipeline._warn_if_labels_look_inconsistent exists as a second
line of defense: it checks the RESULT of this heuristic against what
RECOVERY is supposed to look like (short, transitional) and logs loudly if
it doesn't, rather than trusting the label silently.

Heuristic (in order of precedence):
  1. BEAR     = state with the lowest mean 20d return (o1) -- the clearest,
               least ambiguous signal available.
  2. RECOVERY = among the OTHER three states (all with return >= BEAR's),
               whichever the fitted transition matrix says BEAR moves into
               most often. This directly encodes the spec's own definition
               of RECOVERY ("bottoming out, rising from bear") instead of
               an arbitrary emission-based tiebreak. An earlier version of
               this heuristic picked HIGH_VOL/RECOVERY purely by ranking
               volatility between whichever two states were left over --
               that produced a real mislabeling on real EGX data (a
               globally-low-vol state won "HIGH_VOL" just because it beat
               the other leftover state by a hair). Using the transition
               matrix is a legitimate, principled fix even though it
               happened to agree with the old result on that particular
               fit -- it will do the right thing on fits where the old
               tiebreak wouldn't.
  3. BULL     = among the two states still unlabeled, whichever has the
               higher mean return.
  4. HIGH_VOL = whatever's left.
"""

import numpy as np

from models.regime_features import OBSERVATION_COLUMNS

BULL, BEAR, HIGH_VOL, RECOVERY = "BULL", "BEAR", "HIGH_VOL", "RECOVERY"

_RETURN_COL = OBSERVATION_COLUMNS.index("o1_market_return_20d")


def label_states(mu: np.ndarray, A: np.ndarray) -> dict:
    """mu: (K, D) fitted emission means, columns ordered per
    models.regime_features.OBSERVATION_COLUMNS.
    A: (K, K) fitted transition matrix, A[i, j] = P(state j at t+1 | state i at t).
    Returns {state_index: label}. Only defined for K == 4 (the spec's four
    regimes); raises otherwise rather than silently guessing a mapping for
    a different K.
    """
    K = mu.shape[0]
    if K != 4:
        raise ValueError(f"label_states assumes 4 states (BULL/BEAR/HIGH_VOL/RECOVERY), got {K}")

    returns = mu[:, _RETURN_COL]

    bear_state = int(np.argmin(returns))
    recovery_candidates = [s for s in range(K) if s != bear_state and returns[s] >= returns[bear_state]]
    recovery_state = int(max(recovery_candidates, key=lambda s: A[bear_state, s]))

    remaining = [s for s in range(K) if s not in (bear_state, recovery_state)]
    bull_state = max(remaining, key=lambda s: returns[s])
    high_vol_state = [s for s in remaining if s != bull_state][0]

    return {
        bull_state: BULL,
        bear_state: BEAR,
        high_vol_state: HIGH_VOL,
        recovery_state: RECOVERY,
    }


def label_columns_for_gamma(label_map: dict) -> list:
    """Returns column names ['prob_bull', 'prob_bear', ...] ordered to
    match gamma's state-index columns (gamma[:, k] corresponds to
    label_map[k]), so callers can zip(label_columns_for_gamma(...), gamma.T).
    """
    name_by_regime = {BULL: "prob_bull", BEAR: "prob_bear", HIGH_VOL: "prob_high_vol", RECOVERY: "prob_recovery"}
    return [name_by_regime[label_map[k]] for k in range(len(label_map))]
