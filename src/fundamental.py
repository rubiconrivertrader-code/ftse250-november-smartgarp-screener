"""
fundamental.py (refined scoring)

Fetch fundamentals via yfinance and apply smoother scoring curves for valuation and growth metrics.
The scoring functions are intentionally robust to missing fields and use continuous transforms
rather than harsh thresholds.
"""

from typing import Dict
import numpy as np
import yfinance as yf


def fetch_fundamentals(ticker: str) -> Dict:
    """Use yfinance to fetch basic fundamental metrics. Returns raw values (may be None)."""
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    # Field names used by yfinance
    peg = info.get('pegRatio', None)
    pe = info.get('trailingPE', None) or info.get('forwardPE', None)
    ps = info.get('priceToSalesTrailing12Months', None)
    # earnings growth: try multiple fields
    eg_q = info.get('earningsQuarterlyGrowth', None)
    eg_ann = info.get('earningsGrowth', None)  # sometimes present
    eg = None
    if eg_ann is not None:
        eg = eg_ann
    elif eg_q is not None:
        eg = eg_q

    # convert to percent if fractional
    earnings_growth = None
    if eg is not None:
        try:
            earnings_growth = float(eg) * 100.0
        except Exception:
            earnings_growth = None

    # Additional helpful fields for context
    market_cap = info.get('marketCap', None)
    forward_eps = info.get('forwardEps', None)

    return {
        'peg': float(peg) if peg is not None else None,
        'pe': float(pe) if pe is not None else None,
        'ps': float(ps) if ps is not None else None,
        'earnings_growth_percent': float(earnings_growth) if earnings_growth is not None else None,
        'market_cap': float(market_cap) if market_cap is not None else None,
        'forward_eps': float(forward_eps) if forward_eps is not None else None,
        'raw_info': info
    }


def score_fundamentals(fund: Dict, config: Dict) -> Dict:
    """
    Score fundamental metrics with smoother mappings:
      - PEG: ideal near 1.0 but values between 0.5-1.5 are considered good; penalize extremes.
      - P/E: continuous decay in score as PE increases.
      - P/S: similar continuous mapping with diminishing returns.
      - Earnings growth: reward positive multi-period growth (mapped smoothly).
    """

    weights = config.get('fundamental_weights', {})
    fund_total_weight = config.get('weights', {}).get('fundamental_total', 0.4)

    def is_nan(x):
        return x is None or (isinstance(x, float) and np.isnan(x))

    peg = fund.get('peg', None)
    pe = fund.get('pe', None)
    ps = fund.get('ps', None)
    eg = fund.get('earnings_growth_percent', None)

    subscores = {}

    # PEG scoring: smooth peaked score around 1.0; penalize >3
    if is_nan(peg):
        subscores['peg'] = np.nan
    else:
        # Use a smooth peaked function: score = exp(-((peg - 1)/sigma)^2) scaled, sigma controls width
        sigma = 0.75
        val = np.exp(-((peg - 1.0) ** 2) / (2 * sigma ** 2))
        # scale to 0-100 and penalize very low/negative PEGs less extremely
        subscores['peg'] = float(max(0.0, min(100.0, val * 110.0)))  # allows values slightly >100 to cap

    # P/E scoring: continuous decay; low PE preferred but suspiciously low PE may be due to issues
    if is_nan(pe):
        subscores['pe'] = np.nan
    else:
        # transform with a soft decay: score = 100 / (1 + (pe / 10)^1.2)
        score = 100.0 / (1.0 + (pe / 10.0) ** 1.15)
        subscores['pe'] = float(max(0.0, min(100.0, score)))

    # P/S scoring: similar continuous mapping
    if is_nan(ps):
        subscores['ps'] = np.nan
    else:
        # score = 100 if ps<=1, then decay
        score = 100.0 / (1.0 + (ps / 1.5) ** 1.1)
        subscores['ps'] = float(max(0.0, min(100.0, score)))

    # Earnings growth scoring: map -50 -> 0, 0 -> 40, +50 -> 100
    if is_nan(eg):
        subscores['earnings_growth'] = np.nan
    else:
        if eg <= -50:
            subscores['earnings_growth'] = 0.0
        elif eg >= 50:
            subscores['earnings_growth'] = 100.0
        else:
            subscores['earnings_growth'] = float(40.0 + (eg / 50.0) * 60.0)

    # Combine using weights (normalize for missing)
    subw = {
        'peg': weights.get('peg', 0.40),
        'pe': weights.get('pe', 0.25),
        'ps': weights.get('ps', 0.15),
        'earnings_growth': weights.get('earnings_growth', 0.20)
    }
    available = {k: v for k, v in subscores.items() if not is_nan(v)}
    if not available:
        fundamental_score_100 = np.nan
    else:
        total_weight = sum(subw[k] for k in available.keys())
        fundamental_score_100 = 0.0
        for k, v in available.items():
            w = subw[k] / total_weight if total_weight > 0 else 0
            fundamental_score_100 += v * w

    # Scale to configured fundamental_total (absolute scale)
    fundamental_score_scaled = None
    if is_nan(fundamental_score_100):
        fundamental_score_scaled = np.nan
    else:
        fundamental_score_scaled = (fundamental_score_100 / 100.0) * (fund_total_weight * 100.0)

    return {
        'raw': {'peg': peg, 'pe': pe, 'ps': ps, 'earnings_growth_percent': eg},
        'subscores_100': subscores,
        'fundamental_score_100': float(fundamental_score_100) if not is_nan(fundamental_score_100) else np.nan,
        'fundamental_score_scaled': float(fundamental_score_scaled) if fundamental_score_scaled is not None else np.nan
    }