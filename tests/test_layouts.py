"""Post 5's scorer (analysis/layouts.py): the label-free movement measures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import layouts as L  # noqa: E402


def test_moved_measures_change_against_the_reference():
    p0 = np.array([0.10, 0.20, 0.30, 0.40])
    p1 = np.array([0.10, 0.31, 0.27, 0.50])
    out = L.moved(p0, p1)
    assert out["mean_shift"] == pytest.approx((0 + 0.11 - 0.03 + 0.10) / 4)
    assert out["mean_abs_change"] == pytest.approx((0 + 0.11 + 0.03 + 0.10) / 4)
    assert out["moved_5pts_share"] == pytest.approx(0.5)  # 0.11 and 0.10; 0.03 is under 5 points
    assert out["spearman"] == pytest.approx(0.8)  # the second and third customers swap places


def test_moved_counts_exactly_five_points_as_moved():
    out = L.moved(np.array([0.20, 0.50]), np.array([0.25, 0.45]))
    assert out["moved_5pts_share"] == 1.0


def test_arrays_refuses_runs_with_different_labels():
    ref, other = {"a": (1, 0.3), "b": (0, 0.2)}, {"a": (0, 0.3), "b": (0, 0.2)}
    with pytest.raises(AssertionError):
        L.arrays(ref, other, ["a", "b"])


def test_every_layout_is_scored_against_the_same_holm_family():
    assert len(L.MODELS) * len(L.LAYOUTS) == 30
