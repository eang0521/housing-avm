"""
collect_data.py
Builds an expanded house-sales dataset for Contra Costa County by pulling
sold-home listings from Realtor.com (via HomeHarvest), then enriching each
sale with the prevailing 30-year mortgage rate (FRED) and local unemployment
rate (BLS) keyed to the month of sale.

Output: data/house_sales_extended.csv — same schema as house_sales.csv.

Usage:
    python scripts/collect_data.py [--days N]

    --days N  look back N days for sold listings (default: 1095 = 3 years)

Dependencies (all in pyproject.toml):
    pip install homeharvest requests pandas

Notes:
    - school_score is approximated with city-level medians from the original
      15-row dataset; swap in GreatSchools API values for property-level accuracy.
    - Unemployment: FRED series CAURN = California Not Seasonally Adjusted Unemployment Rate,
      used as a proxy for Contra Costa County. No API key required.
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from homeharvest import scrape_property

# ── Configuration ──────────────────────────────────────────────────────────────

CITIES = ["Concord", "Walnut Creek", "Martinez", "Pleasant Hill"]
STATE  = "CA"

# City-level median school scores estimated from the original 15-row dataset.
# Swap in GreatSchools API (https://developer.greatschools.org) for
# property-level accuracy.
CITY_SCHOOL_SCORES = {
    "Concord":       9,
    "Walnut Creek": 22,
    "Martinez":     19,
    "Pleasant Hill": 17,
}

PROPERTY_TYPE_MAP = {
    "SINGLE_FAMILY": "Single-family",
    "TOWNHOMES":     "Townhome",
    "TOWNHOUSE":     "Townhome",
    "CONDO":         "Condo",
    "CONDOS":        "Condo",
    "CONDO_TOWNHOME": "Condo",
}

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "house_sales_extended.csv"

# ── Realtor.com via HomeHarvest ────────────────────────────────────────────────

def fetch_sold_listings(city: str, date_from: str, date_to: str) -> pd.DataFrame:
    """
    Pull sold listings from Realtor.com for one city using HomeHarvest.
    Returns a DataFrame normalised to the project schema.
    """
    raw = scrape_property(
        location=f"{city}, {STATE}",
        listing_type="sold",
        date_from=date_from,
        date_to=date_to,
    )
    print(f"    {len(raw)} raw listings")

    df = raw.rename(columns={
        "full_street_line": "address",
        "sold_price":       "sold_price",
        "beds":             "bedrooms",
        "sqft":             "sq_ft",
        "style":            "type",
        "year_built":       "year_built",
        "last_sold_date":   "date_of_sale",
    }).copy()

    df["city"]         = city
    df["type"]         = df["type"].map(PROPERTY_TYPE_MAP)
    df["date_of_sale"] = pd.to_datetime(df["date_of_sale"], errors="coerce")
    df["build_age"]    = datetime.now().year - pd.to_numeric(df["year_built"], errors="coerce")
    df["school_score"] = CITY_SCHOOL_SCORES.get(city)

    keep = df["type"].notna() & df["date_of_sale"].notna() & df["sold_price"].notna()
    filtered = df[keep].copy()
    print(f"    {len(filtered)} after type/date/price filter")
    return filtered


# ── FRED (30-year fixed mortgage rate) ────────────────────────────────────────

def fetch_mortgage_rates() -> pd.DataFrame:
    """Download the MORTGAGE30US weekly series from FRED (no API key needed)."""
    from io import StringIO
    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US",
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().sort_values("date").reset_index(drop=True)


# ── BLS (Contra Costa County unemployment) ────────────────────────────────────

def fetch_unemployment() -> pd.DataFrame:
    """
    Fetch monthly California unemployment rate from FRED (series CAURN).
    CAURN = California Not Seasonally Adjusted Unemployment Rate (BLS LAUS, via FRED).
    Used as a proxy for Contra Costa County; county-level values track state trends closely.
    No API key required.
    """
    from io import StringIO
    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CAURN",
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", "unemployment"]
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df[["year", "month", "unemployment"]].dropna().reset_index(drop=True)


# ── Pipeline ───────────────────────────────────────────────────────────────────

def build_dataset(days: int = 1095, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    date_to   = datetime.now().strftime("%Y-%m-%d")
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # 1. Realtor.com: one request per city
    print(f"Fetching sold listings ({date_from} to {date_to})...")
    frames = []
    for city in CITIES:
        print(f"  {city}")
        try:
            frames.append(fetch_sold_listings(city, date_from, date_to))
        except Exception as exc:
            print(f"  WARNING: skipped {city} — {exc}")
        time.sleep(1)

    if not frames:
        raise RuntimeError("No listing data retrieved.")

    df = pd.concat(frames, ignore_index=True)
    print(f"\n  {len(df)} total listings across all cities")

    # 2. FRED: join interest rate on nearest date to sale
    print("\nFetching 30-year mortgage rates from FRED...")
    rates = fetch_mortgage_rates()
    df_sorted = df.sort_values("date_of_sale")
    df["interest_rate"] = pd.merge_asof(
        df_sorted[["date_of_sale"]],
        rates,
        left_on="date_of_sale",
        right_on="date",
        direction="nearest",
    )["rate"].values
    print(f"  Joined {len(rates)} weekly rate observations")

    # 3. BLS: join monthly unemployment by year + month of sale
    print("\nFetching California unemployment from FRED (proxy for Contra Costa)...")
    unemp = fetch_unemployment()
    df["year"]  = df["date_of_sale"].dt.year
    df["month"] = df["date_of_sale"].dt.month
    df = df.merge(unemp, on=["year", "month"], how="left")
    print(f"  Joined {len(unemp)} monthly unemployment observations")

    # 4. Final column selection and cleanup
    out_cols = [
        "address", "city", "date_of_sale", "sold_price",
        "bedrooms", "sq_ft", "type", "build_age",
        "school_score", "unemployment", "interest_rate",
    ]
    result = (
        df[out_cols]
        .dropna(subset=["sold_price", "sq_ft", "bedrooms", "build_age", "interest_rate"])
        .copy()
    )
    # Format date to match original CSV: M/D/YY (no leading zeros)
    result["date_of_sale"] = result["date_of_sale"].apply(
        lambda d: f"{d.month}/{d.day}/{d.strftime('%y')}"
    )
    result["sold_price"] = result["sold_price"].astype(int)
    result["bedrooms"]   = result["bedrooms"].astype(int)
    result["sq_ft"]      = result["sq_ft"].astype(int)
    result["build_age"]  = result["build_age"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved {len(result)} rows -> {output_path}")
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Contra Costa County home sales data.")
    parser.add_argument("--days", type=int, default=1095, help="Look-back window in days (default: 1095 = 3 years)")
    args = parser.parse_args()

    df = build_dataset(days=args.days)

    print(f"\nCity breakdown:\n{df['city'].value_counts().to_string()}")
    print(f"\nType breakdown:\n{df['type'].value_counts().to_string()}")
    print(f"\nSample:\n{df.head(3).to_string(index=False)}")
