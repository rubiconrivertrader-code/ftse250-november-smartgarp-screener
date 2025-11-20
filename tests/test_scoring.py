"""
Unit tests for scoring utilities.

Tests included:
- compute_percentiles handles NaNs as bottom correctly and small sample sizes
- compute_zscore_percentiles returns 50 for identical values and NaN preserved
- combine_scores renormalizes weights when requested vs treats missing as zero when not requested
"""

import pytest
import numpy as np
import pandas as pd

from scoring import compute_percentiles, compute_zscore_percentiles, combine_scores


def test_compute_percentiles_with_nans_bottom():
    vals = np.array([50.0, np.nan, 80.0, 30.0])
    pct = compute_percentiles(vals, na_option='bottom')
    # ordering: 30 -> 0.25, 50 -> 0.5, 80 -> 1.0, NaN -> 0.0 after fill
    # But rank(pct=True) yields fractional ranks; validate relative ordering and NaN->0
    assert np.isclose(pct[1], 0.0)  # NaN mapped to 0
    assert pct[3] < pct[0] < pct[2]  # 30 < 50 < 80 in percentiles


def test_compute_percentiles_all_nans():
    vals = np.array([np.nan, np.nan])
    pct = compute_percentiles(vals, na_option='bottom')
    assert np.allclose(pct, np.array([0.0, 0.0]))


def test_compute_zscore_percentiles_identical_values():
    vals = np.array([10.0, 10.0, 10.0, np.nan])
    zpct = compute_zscore_percentiles(vals)
    # non-NaN entries should be 50
    assert np.allclose(zpct[:3], np.array([50.0, 50.0, 50.0]))
    # NaN should remain NaN
    assert np.isnan(zpct[3])


def test_combine_scores_renormalize_behavior():
    # config with 60/40
    cfg = {'weights': {'technical_total': 0.6, 'fundamental_total': 0.4}, 'renormalize_missing': True}
    tech_val = 80.0
    fund_val = np.nan
    # With renormalize true, composite should equal tech_val (weight renormalized to 1)
    comp = combine_scores(tech_val, fund_val, cfg, renormalize_missing=True)
    assert pytest.approx(comp, rel=1e-6) == 80.0

    # With renormalize false, missing treated as zero => composite = tech_val*0.6
    comp2 = combine_scores(tech_val, fund_val, cfg, renormalize_missing=False)
    assert pytest.approx(comp2, rel=1e-6) == pytest.approx(80.0 * 0.6)


def test_combine_scores_both_present():
    cfg = {'weights': {'technical_total': 0.6, 'fundamental_total': 0.4}, 'renormalize_missing': True}
    tech_val = 70.0
    fund_val = 50.0
    comp = combine_scores(tech_val, fund_val, cfg, renormalize_missing=True)
    # Should equal 70*0.6 + 50*0.4 = 42 + 20 = 62
    assert pytest.approx(comp, rel=1e-6) == pytest.approx(62.0)