"""
main.py (updated)

Enhancements:
- CLI flags to override ranking method and renormalize_missing behavior at runtime:
    --ranking-method {absolute,percentile,zscore,hybrid}
    --no-renormalize  (if passed, treats missing components as zero)
- Always compute and save:
    - TechnicalScore_100, FundamentalScore_100 (absolute mapped scores)
    - TechnicalPercentile, FundamentalPercentile (rank percentiles)
    - TechnicalZscorePercentile, FundamentalZscorePercentile (zscore->percentile)
    - CompositeScore_Absolute, CompositeScore_Percentile, CompositeScore_Zscore, CompositeScore_Hybrid
- Final CompositeScore column selected by method (config overridden by CLI).
- Respects renormalization option when combining components.
"""

import os
import sys
import yaml
import time
import traceback
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
import numpy as np

from technical import compute_indicators, score_technical
from fundamental import fetch_fundamentals, score_fundamentals
from scoring import combine_scores, compute_price_targets, compute_percentiles, compute_zscore_percentiles

# Setup logging
LOG_DIR = "data/results"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "screener.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_constituents(path="data/constituents.csv"):
    df = pd.read_csv(path, dtype=str)
    df = df.fillna("")
    return df


def parse_args():
    p = argparse.ArgumentParser(description="FTSE 250 SmartGARP Screener")
    p.add_argument("--ranking-method", choices=['absolute', 'percentile', 'zscore', 'hybrid'],
                   help="Override config ranking.method for this run")
    p.add_argument("--no-renormalize", action='store_true',
                   help="Treat missing components as zero instead of renormalizing weights")
    p.add_argument("--constituents", default="data/constituents.csv",
                   help="Path to constituents CSV")
    return p.parse_args()


def process_ticker(row, config):
    ticker = row['Ticker']
    name = row.get('Name', '')
    sector = row.get('Sector', '')
    result = {
        'Ticker': ticker,
        'Name': name,
        'Sector': sector,
        'Error': ''
    }
    try:
        # 1) Download price history
        period = config.get('pricing', {}).get('history_period', '1y')
        interval = config.get('pricing', {}).get('interval', '1d')
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        except Exception:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            raise ValueError("No historical price data")

        # ensure columns: Open, High, Low, Close, Volume
        if 'Close' not in df.columns:
            raise ValueError("Unexpected historical data format")

        last_close = float(df['Close'].dropna().iloc[-1])
        # Technical calculations
        indicators = compute_indicators(df, config)
        indicators['last_close'] = last_close
        tech = score_technical(indicators, config)

        # Fundamentals
        fund_raw = fetch_fundamentals(ticker)
        fund = score_fundamentals(fund_raw, config)

        # Composite absolute: use technical_score_100 and fundamental_score_100 combined using weights
        tech_100 = tech.get('technical_score_100', None)
        fund_100 = fund.get('fundamental_score_100', None)
        composite_abs = combine_scores(tech_100, fund_100, config, renormalize_missing=config.get('renormalize_missing', True))

        # Price targets
        t1, t2 = compute_price_targets(last_close, config)

        # Aggregate results
        result.update({
            'LastClose': last_close,
            'TechnicalScore_100': tech.get('technical_score_100'),
            'TechnicalScore_Scaled': tech.get('technical_score_scaled'),
            'FundamentalScore_100': fund.get('fundamental_score_100'),
            'FundamentalScore_Scaled': fund.get('fundamental_score_scaled'),
            'CompositeScore_Absolute': composite_abs,
            'Target1': t1,
            'Target2': t2,
            # store raw metrics for traceability
            'RSI': indicators.get('rsi'),
            'MACD': indicators.get('macd', {}).get('macd'),
            'MACD_Signal': indicators.get('macd', {}).get('signal'),
            'MACD_Crossover': indicators.get('macd', {}).get('crossover'),
            'ATR': indicators.get('atr'),
            'TrendPercent': indicators.get('trend_percent'),
            'EMA_Fast': indicators.get('ema_fast'),
            'EMA_Slow': indicators.get('ema_slow'),
            'EMA_Trend': indicators.get('ema_trend'),
            'EMA_PositionPct': indicators.get('ema_position_pct'),
            'OBV': indicators.get('obv'),
            'OBV_PctChange': indicators.get('obv_pct_change'),
            'ADX': indicators.get('adx'),
            'PEG': fund_raw.get('peg'),
            'PE': fund_raw.get('pe'),
            'PS': fund_raw.get('ps'),
            'EarningsGrowthPercent': fund_raw.get('earnings_growth_percent'),
        })

        logging.info(f"Processed {ticker}: composite_abs={composite_abs:.2f}")
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error processing {ticker}: {e}\n{tb}")
        result['Error'] = f"{e}"
    return result


def save_results(results_df: pd.DataFrame, config: dict):
    os.makedirs("data/results", exist_ok=True)
    full_csv = os.path.join("data/results", "full_rankings.csv")
    results_df.to_csv(full_csv, index=False)
    logging.info(f"Saved full rankings to {full_csv}")

    # Top 5 markdown — final CompositeScore (based on method)
    md_file = os.path.join("data/results", "latest_top_5.md")
    sort_col = "CompositeScore"
    top5 = results_df.sort_values(by=sort_col, ascending=False).head(5)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# FTSE 250 SmartGARP Screener — Top 5\n\n")
        f.write(f"Ranking method: {config.get('ranking', {}).get('method', 'absolute')}\n\n")
        f.write("| Rank | Ticker | Name | Sector | Composite Score | Composite (Abs) | Composite (Pct) | Composite (Zscore) | Tech %ile | Fund %ile | Last Close | Target1 | Target2 |\n")
        f.write("|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for idx, row in enumerate(top5.itertuples(), start=1):
            comp = getattr(row, 'CompositeScore', None)
            comp_abs = getattr(row, 'CompositeScore_Absolute', None)
            comp_pct = getattr(row, 'CompositeScore_Percentile', None)
            comp_z = getattr(row, 'CompositeScore_Zscore', None)
            tech_pct = getattr(row, 'TechnicalPercentile', None)
            fund_pct = getattr(row, 'FundamentalPercentile', None)
            f.write(f"| {idx} | {row.Ticker} | {row.Name} | {row.Sector} | "
                    f"{comp:.2f} | {comp_abs:.2f} | {comp_pct:.2f} | {comp_z:.2f} | "
                    f"{tech_pct:.2f} | {fund_pct:.2f} | "
                    f"{row.LastClose:.4f} | {row.Target1:.4f} | {row.Target2:.4f} |\n")
    logging.info(f"Saved top 5 to {md_file}")


def main():
    args = parse_args()
    cfg = load_config()
    # Override config using CLI flags if provided
    if args.ranking_method:
        cfg.setdefault('ranking', {})['method'] = args.ranking_method
    if args.no_renormalize:
        cfg['renormalize_missing'] = False

    constituents = load_constituents(path=args.constituents)

    results = []
    errors = []

    max_workers = cfg.get('max_workers', 4)
    logging.info(f"Starting enhanced analysis for {len(constituents)} tickers using {max_workers} workers with method={cfg.get('ranking', {}).get('method')} renorm={cfg.get('renormalize_missing')}")

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process_ticker, row, cfg): idx for idx, row in constituents.iterrows()}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if res.get('Error'):
                errors.append(res)

    # Build DataFrame
    df = pd.DataFrame(results)

    # Ensure numeric columns are casted properly when possible
    numeric_cols = ['LastClose', 'TechnicalScore_100', 'TechnicalScore_Scaled',
                    'FundamentalScore_100', 'FundamentalScore_Scaled',
                    'CompositeScore_Absolute', 'CompositeScore_Percentile', 'CompositeScore_Zscore', 'CompositeScore_Hybrid',
                    'Target1', 'Target2',
                    'RSI', 'MACD', 'MACD_Signal', 'ATR', 'TrendPercent',
                    'EMA_Fast', 'EMA_Slow', 'EMA_Trend', 'EMA_PositionPct',
                    'OBV', 'OBV_PctChange', 'ADX',
                    'PEG', 'PE', 'PS', 'EarningsGrowthPercent']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute percentiles across the full-run for technical and fundamental scores (rank-based)
    tech_col = 'TechnicalScore_100'
    fund_col = 'FundamentalScore_100'

    # Compute rank percentiles (treat NaNs as bottom for dashboards)
    df['TechnicalPercentile'] = compute_percentiles(df[tech_col].to_numpy() if tech_col in df.columns else np.array([]), na_option='bottom') if len(df) > 0 else np.array([])
    df['FundamentalPercentile'] = compute_percentiles(df[fund_col].to_numpy() if fund_col in df.columns else np.array([]), na_option='bottom') if len(df) > 0 else np.array([])

    # Compute zscore-based percentiles for both components
    df['TechnicalZscorePercentile'] = compute_zscore_percentiles(df[tech_col].to_numpy() if tech_col in df.columns else np.array([]))
    df['FundamentalZscorePercentile'] = compute_zscore_percentiles(df[fund_col].to_numpy() if fund_col in df.columns else np.array([]))

    # CompositeScore_Percentile and CompositeScore_Zscore
    tech_weight = cfg.get('weights', {}).get('technical_total', 0.6)
    fund_weight = cfg.get('weights', {}).get('fundamental_total', 0.4)

    df['CompositeScore_Percentile'] = (df['TechnicalPercentile'].fillna(0.0) * tech_weight + df['FundamentalPercentile'].fillna(0.0) * fund_weight).clip(0, 100)
    df['CompositeScore_Zscore'] = (df['TechnicalZscorePercentile'].fillna(0.0) * tech_weight + df['FundamentalZscorePercentile'].fillna(0.0) * fund_weight).clip(0, 100)

    # CompositeScore_Hybrid: tech percentile + fund absolute (example hybrid)
    df['CompositeScore_Hybrid'] = (df['TechnicalPercentile'].fillna(0.0) * tech_weight + df['FundamentalScore_100'].fillna(0.0) * fund_weight).clip(0, 100)

    # Ensure CompositeScore_Absolute exists
    if 'CompositeScore_Absolute' not in df.columns:
        df['CompositeScore_Absolute'] = df.apply(
            lambda r: combine_scores(r.get('TechnicalScore_100', np.nan), r.get('FundamentalScore_100', np.nan), cfg, renormalize_missing=cfg.get('renormalize_missing', True)),
            axis=1
        )

    # Final CompositeScore column selected by method
    method = cfg.get('ranking', {}).get('method', 'absolute')
    if method == 'percentile':
        df['CompositeScore'] = df['CompositeScore_Percentile']
    elif method == 'zscore':
        df['CompositeScore'] = df['CompositeScore_Zscore']
    elif method == 'hybrid':
        df['CompositeScore'] = df['CompositeScore_Hybrid']
    else:
        df['CompositeScore'] = df['CompositeScore_Absolute']

    save_results(df, cfg)

    # Save errors log
    if errors:
        err_file = os.path.join("data/results", "errors.log")
        with open(err_file, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(f"{e.get('Ticker')}: {e.get('Error')}\n")
        logging.info(f"Wrote errors to {err_file}")


if __name__ == "__main__":
    main()