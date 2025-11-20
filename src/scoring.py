"""
scoring.py

Enhanced scoring utilities:
- compute_percentiles(series, na_option='bottom') -> numpy array of 0..100 percentiles (NaNs handled)
- compute_zscore_percentiles(series) -> numpy array of 0..100 percentiles based on z-scores + normal CDF
- combine_scores(value_dict_or_tuple, config, renormalize_missing=True) -> composite 0..100

combine_scores expects component values in 0..100 for whichever representation you use (absolute or percentile).
If renormalize_missing=True then weights for missing components are redistributed among present components.
"""

from typing import Tuple, Optional, Sequence, Union, Dict
import numpy as np
import pandas as pd
import math


def compute_percentiles(values: Sequence[float], na_option: str = 'bottom') -> np.ndarray:
    """
    Compute percentiles (0..100) for a sequence-like of numbers using pandas rank(pct=True).
    NaNs are handled per na_option: 'bottom' places NaNs as lowest percentiles (0),
    'top' places NaNs as highest (100), 'keep' will leave NaNs as np.nan.
    Returns numpy array aligned with input containing floats 0..100 (or np.nan).
    """
    s = pd.Series(values)
    if na_option not in ('bottom', 'top', 'keep'):
        raise ValueError("na_option must be 'bottom', 'top' or 'keep'")

    if na_option == 'keep':
        pct = s.rank(method='average', pct=True, na_option='keep') * 100.0
    else:
        pct = s.rank(method='average', pct=True, na_option=na_option) * 100.0

    # Convert any NaN rank to 0 if bottom, 100 if top, else keep NaN
    if na_option == 'bottom':
        pct = pct.fillna(0.0)
    elif na_option == 'top':
        pct = pct.fillna(100.0)

    return pct.to_numpy(dtype=float)


def compute_zscore_percentiles(values: Sequence[float]) -> np.ndarray:
    """
    Compute z-score percentiles: for each value compute z=(x-mean)/std (ignoring NaNs),
    then map z to percentile via normal CDF: pct = norm.cdf(z) * 100.
    If std == 0 (all values equal among non-NaN), return 50 for non-NaN values.
    NaNs remain NaN in output.
    """
    s = pd.Series(values, dtype='float64')
    mean = s.mean(skipna=True)
    std = s.std(skipna=True, ddof=0)  # population std to be stable for small samples

    out = np.full(len(s), np.nan, dtype=float)
    if std == 0 or np.isnan(std):
        # all values equal (or insufficient) -> neutral 50th percentile for present values
        mask = s.notna()
        out[mask.values] = 50.0
        return out

    # Compute z and map to CDF using erf
    for i, v in enumerate(s):
        if pd.isna(v):
            out[i] = np.nan
        else:
            z = (v - mean) / std
            # Normal CDF via erf
            cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            out[i] = float(max(0.0, min(100.0, cdf * 100.0)))
    return out


def combine_scores(tech_value: Optional[float],
                   fund_value: Optional[float],
                   config: Dict,
                   renormalize_missing: Optional[bool] = None) -> float:
    """
    Combine two component values (each 0..100) into a composite 0..100 using weights in config.
    - tech_value: technical component (0..100) or None/np.nan if missing
    - fund_value: fundamental component (0..100) or None/np.nan if missing
    Behavior:
    - If renormalize_missing is True, and one component is missing, the other's weight is renormalized to 1.0.
    - If renormalize_missing is False, missing component is treated as zero (conservative penalty).
    Returns composite float 0..100.
    """
    tech_weight = config.get('weights', {}).get('technical_total', 0.6)
    fund_weight = config.get('weights', {}).get('fundamental_total', 0.4)

    if renormalize_missing is None:
        renormalize_missing = bool(config.get('renormalize_missing', True))

    # Helper to identify missing
    def is_missing(x):
        return x is None or (isinstance(x, float) and np.isnan(x))

    tech_missing = is_missing(tech_value)
    fund_missing = is_missing(fund_value)

    # If both missing, return 0
    if tech_missing and fund_missing:
        return 0.0

    if renormalize_missing:
        # compute sum of weights of present components
        present = []
        present_weights = []
        if not tech_missing:
            present.append(('tech', float(tech_value)))
            present_weights.append(tech_weight)
        if not fund_missing:
            present.append(('fund', float(fund_value)))
            present_weights.append(fund_weight)
        total_present_weight = sum(present_weights)
        if total_present_weight <= 0:
            # degenerate, fallback: average present components
            vals = [v for _, v in present]
            return float(np.mean(vals) if vals else 0.0)
        # renormalize weights proportionally
        composite = 0.0
        for (name, val), w in zip(present, present_weights):
            normalized_w = w / total_present_weight
            composite += float(val) * normalized_w
        composite = float(max(0.0, min(100.0, composite)))
        return composite
    else:
        # conservative approach: missing -> treat as zero
        t = 0.0 if tech_missing else float(tech_value)
        f = 0.0 if fund_missing else float(fund_value)
        composite = t * tech_weight + f * fund_weight
        composite = float(max(0.0, min(100.0, composite)))
        return composite