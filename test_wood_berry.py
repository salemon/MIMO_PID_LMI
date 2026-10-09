"""Regression test: the Wood-Berry design must reproduce the paper's result.

Run with  ``pytest -q``  (about one minute with N=60; the full N=300 paper
grid is exercised by ``wood_berry_example.py``).
"""

import numpy as np
import pytest

from auto_mimo_pid import closed_loop_sigmas
from wood_berry_example import PAPER_MIMO, design


@pytest.fixture(scope="module")
def result():
    return design(N=60, verbose=False)


def test_objective_matches_paper(result):
    # Paper: 2.25 on a 300-point grid.  Coarser grids give slightly smaller
    # (optimistic) values because fewer constraints are enforced.
    assert result["objval"] == pytest.approx(PAPER_MIMO["objval"], abs=0.15)


def test_gains_match_paper(result):
    for key in ("Kp", "Ki", "Kd"):
        np.testing.assert_allclose(result[key], PAPER_MIMO[key], atol=0.02)


def test_constraints_hold_on_design_grid(result):
    S, T, Q = closed_loop_sigmas(result["P"], result["controller"], result["w"])
    tol = 1e-3
    assert S.max() <= result["Smax"] + tol
    assert T.max() <= result["Tmax"] + tol
    assert Q.max() <= result["Qmax"] + tol


def test_objective_is_monotone(result):
    hist = result["history"]
    assert all(b <= a + 1e-6 for a, b in zip(hist, hist[1:]))
