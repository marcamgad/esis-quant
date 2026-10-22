"""Exchange reference price as corporate-action evidence (librarian, 2026-10-04)."""

import pandas as pd

from ops import reference_librarian as rl


def test_reference_inside_window_and_same_direction_matches():
    refs = [(pd.Timestamp("2025-11-16"), 4.563)]                 # MCRO: EGX re-based a week later
    assert rl._match_ref("2025-11-09", 4.723, refs)               # vendor applied it early
    assert rl._match_ref("2025-08-23", 4.723, refs) is None       # outside the window
    assert rl._match_ref("2025-11-09", 0.8, refs) is None         # a rise is not this event


def test_ratio_far_from_the_reference_does_not_match():
    refs = [(pd.Timestamp("2026-08-30"), 1.639)]                  # LUTS rights issue
    assert rl._match_ref("2026-08-23", 1.743, refs)               # vendor's jump: close enough
    assert rl._match_ref("2026-08-23", 4.0, refs) is None         # a 4x drop is something else
