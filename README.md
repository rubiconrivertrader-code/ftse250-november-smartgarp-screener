```markdown
# FTSE 250 SmartGARP Screener

This repository contains a Python-based stock screening system for the FTSE 250 that runs automatically via GitHub Actions.

Overview
- SmartGARP screener combining technical (60%) and fundamental (40%) scores.
- Indicators: RSI, MACD, ATR, EMA, OBV, ADX, trend; fundamentals: PEG, P/E, P/S, earnings growth.
- Ranking modes: absolute, percentile, zscore, hybrid.
- Automated fetching of FTSE 250 constituents and sector enrichment via yfinance.
- Weekly scheduled workflow and a manual dispatch workflow to refresh constituents immediately.

Quick start (local)
1. Create virtualenv and install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Fetch constituents (optional locally):
   python src/fetch_constituents.py

3. Run the screener:
   python src/main.py

CI & automation
- .github/workflows/weekly_screen.yml runs weekly and will refresh constituents before running the screener.
- .github/workflows/dispatch_fetch.yml allows you to manually trigger a fetch+commit of data/constituents.csv via the Actions UI.

Notes
- Sector enrichment is enabled by default in the fetch script (slower).
- Tests are included (tests/test_scoring.py); run with pytest.
```