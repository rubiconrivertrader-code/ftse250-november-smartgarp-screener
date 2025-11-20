"""
Fetch and populate data/constituents.csv with the authoritative FTSE 250 constituents
formatted for yfinance ('.L' suffix for LSE tickers).

This script attempts the following sources (in order) until it succeeds:
  1. Yahoo Finance FTSE 250 components table
  2. London Stock Exchange constituents table (if accessible)

Sector enrichment is enabled by default (slower) and uses yfinance.info to fill the Sector column.
The script writes data/constituents.csv with columns: Ticker,Name,Sector
"""

import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

YAHOO_URL = "https://uk.finance.yahoo.com/quote/%5EFTMC/components/"
LSE_URL = "https://www.londonstockexchange.com/indices/ftse-250/constituents/table"

OUT_PATH = os.path.join("data", "constituents.csv")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FTSE-250-screener/1.0; +https://github.com/)"
}


def _clean_symbol(sym: str) -> str:
    s = str(sym).strip()
    if s == "" or s.lower() in {"nan", "n/a", "-"}:
        return ""
    if "." in s:
        return s
    return s + ".L"


def fetch_from_yahoo() -> pd.DataFrame:
    resp = requests.get(YAHOO_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    for tb in tables:
        cols = [c.lower() for c in tb.columns.astype(str)]
        if any("symbol" in c or "ticker" in c for c in cols) and any("name" in c for c in cols):
            tb.columns = [str(c).strip() for c in tb.columns]
            symbol_col = next((c for c in tb.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
            name_col = next((c for c in tb.columns if "name" in c.lower()), None)
            result = tb[[symbol_col, name_col]].copy()
            result.columns = ["Symbol", "Name"]
            return result
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    rows = []
    if table:
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                if len(tds) >= 2:
                    rows.append((tds[0], tds[1]))
    if rows:
        return pd.DataFrame(rows, columns=["Symbol", "Name"])
    raise RuntimeError("Could not parse Yahoo components table")


def fetch_from_lse() -> pd.DataFrame:
    resp = requests.get(LSE_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    for tb in tables:
        cols = [c.lower() for c in tb.columns.astype(str)]
        if any("epic" in c or "ticker" in c or "symbol" in c for c in cols) and any("company" in c or "name" in c for c in cols):
            epic_col = next((c for c in tb.columns if "epic" in c.lower() or "symbol" in c.lower() or "ticker" in c.lower()), None)
            name_col = next((c for c in tb.columns if "company" in c.lower() or "name" in c.lower()), None)
            if epic_col and name_col:
                res = tb[[epic_col, name_col]].copy()
                res.columns = ["Symbol", "Name"]
                return res
    raise RuntimeError("Could not parse LSE constituents table")


def enrich_sectors(df: pd.DataFrame) -> pd.DataFrame:
    import yfinance as yf
    sectors = []
    for sym in df["Ticker"].tolist():
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            sector = info.get("sector", "")
            sectors.append(sector if sector is not None else "")
        except Exception:
            sectors.append("")
        time.sleep(0.1)
    df["Sector"] = sectors
    return df


def main():
    os.makedirs("data", exist_ok=True)
    last_exc = None
    for fetcher in (fetch_from_yahoo, fetch_from_lse):
        try:
            print(f"Attempting to fetch constituents from {fetcher.__name__}...")
            table = fetcher()
            table["Symbol"] = table["Symbol"].astype(str).str.strip()
            table["Name"] = table["Name"].astype(str).str.strip()
            table["Ticker"] = table["Symbol"].apply(_clean_symbol)
            table = table[table["Ticker"] != ""].drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
            out = pd.DataFrame({
                "Ticker": table["Ticker"],
                "Name": table["Name"],
                "Sector": [""] * len(table)
            })
            print("Enriching sectors via yfinance (this may take a while)...")
            out = enrich_sectors(out)
            out.to_csv(OUT_PATH, index=False)
            print(f"Wrote {len(out)} constituents to {OUT_PATH}")
            return
        except Exception as e:
            last_exc = e
            print(f"Fetcher {fetcher.__name__} failed: {e}")
            continue
    raise RuntimeError(f"All fetchers failed. Last error: {last_exc}")


if __name__ == "__main__":
    main()