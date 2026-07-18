"""
collect_data.py
Builds an enriched house-sales dataset for Contra Costa County by pulling
sold listings from Realtor.com (HomeHarvest), then adding:
  - 30-year mortgage rate (FRED MORTGAGE30US)
  - California unemployment rate (FRED CAURN)
  - Distance to nearest BART station (computed from lat/lon)
  - Census tract median household income (ACS 5-yr, Census Batch Geocoder)
  - Walk Score / Transit Score (optional — set WALKSCORE_API_KEY env var)

Output: data/house_sales_extended.csv

Usage:
    python scripts/collect_data.py [--days N]
"""

import argparse
import csv
import io
import json
import math
import os
import re
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from homeharvest import scrape_property

# ── Configuration ──────────────────────────────────────────────────────────────

CITIES = ["Concord", "Walnut Creek", "Martinez", "Pleasant Hill"]
STATE  = "CA"

CITY_SCHOOL_SCORES = {
    "Concord":       9,
    "Walnut Creek": 22,
    "Martinez":     19,
    "Pleasant Hill": 17,
}

PROPERTY_TYPE_MAP = {
    "SINGLE_FAMILY":  "Single-family",
    "TOWNHOMES":      "Townhome",
    "TOWNHOUSE":      "Townhome",
    "CONDO":          "Condo",
    "CONDOS":         "Condo",
    "CONDO_TOWNHOME": "Condo",
}

# All BART stations within ~20 miles of the four target cities
BART_STATIONS = {
    "Orinda":                   (37.8784, -122.1795),
    "Lafayette":                (37.8937, -122.1237),
    "Walnut Creek":             (37.9057, -122.0671),
    "Pleasant Hill/CC Centre":  (37.9282, -122.0567),
    "Concord":                  (37.9738, -122.0291),
    "North Concord/Martinez":   (37.9938, -122.0250),
    "Pittsburg/Bay Point":      (37.9961, -121.9449),
}

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "house_sales_extended.csv"
TRACT_CACHE = Path(__file__).parent.parent / "data" / "tract_cache.json"


# ── HomeHarvest ────────────────────────────────────────────────────────────────

def fetch_sold_listings(city: str, date_from: str, date_to: str) -> pd.DataFrame:
    raw = scrape_property(
        location=f"{city}, {STATE}",
        listing_type="sold",
        date_from=date_from,
        date_to=date_to,
    )
    print(f"    {len(raw)} raw listings")

    df = raw.rename(columns={
        "full_street_line": "address",
        "beds":             "bedrooms",
        "full_baths":       "bathrooms",
        "sqft":             "sq_ft",
        "style":            "type",
        "year_built":       "year_built",
        "last_sold_date":   "date_of_sale",
        "parking_garage":   "garage",
    }).copy()

    df["city"]         = city
    df["type"]         = df["type"].map(PROPERTY_TYPE_MAP)
    df["date_of_sale"] = pd.to_datetime(df["date_of_sale"], errors="coerce")
    df["build_age"]    = datetime.now().year - pd.to_numeric(df["year_built"], errors="coerce")
    df["school_score"] = CITY_SCHOOL_SCORES.get(city)

    df["lot_sqft"]  = pd.to_numeric(df.get("lot_sqft"),  errors="coerce")
    df["hoa_fee"]   = pd.to_numeric(df.get("hoa_fee"),   errors="coerce").fillna(0)
    df["stories"]   = pd.to_numeric(df.get("stories"),   errors="coerce")
    df["garage"]    = pd.to_numeric(df.get("garage"),    errors="coerce").fillna(0).astype(int)
    df["bathrooms"] = pd.to_numeric(df.get("bathrooms"), errors="coerce")
    df["latitude"]  = pd.to_numeric(df.get("latitude"),  errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["zip_code"]  = df.get("zip_code", "").fillna("").astype(str)

    keep = df["type"].notna() & df["date_of_sale"].notna() & df["sold_price"].notna()
    filtered = df[keep].copy()
    print(f"    {len(filtered)} after type/date/price filter")
    return filtered


# ── FRED ───────────────────────────────────────────────────────────────────────

def fetch_mortgage_rates() -> pd.DataFrame:
    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US", timeout=30
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna().sort_values("date").reset_index(drop=True)


def fetch_unemployment() -> pd.DataFrame:
    resp = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CAURN", timeout=30
    )
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = ["date", "unemployment"]
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df[["year", "month", "unemployment"]].dropna().reset_index(drop=True)


# ── BART distance ──────────────────────────────────────────────────────────────

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def add_bart_distance(df: pd.DataFrame) -> pd.DataFrame:
    coords = list(BART_STATIONS.values())

    def nearest(row):
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            return float("nan")
        return min(_haversine_miles(row["latitude"], row["longitude"], lat, lon) for lat, lon in coords)

    df["dist_bart_miles"] = df.apply(nearest, axis=1)
    valid = df["dist_bart_miles"].notna().sum()
    print(f"  {valid}/{len(df)} properties geocoded  |  median {df['dist_bart_miles'].median():.1f} mi to BART")
    return df


# ── Census ACS median household income ────────────────────────────────────────

def fetch_acs_income() -> pd.DataFrame:
    """
    ACS 5-year median household income for every census tract in Contra Costa County.
    Uses Census Reporter (censusreporter.org) — no API key required.
    geo_ids: 140 = census tracts, 05000US06013 = Contra Costa County CA.
    """
    url = (
        "https://api.censusreporter.org/1.0/data/show/latest"
        "?table_ids=B19013&geo_ids=140|05000US06013"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for geo_id, tables in data["data"].items():
        # geo_id format: '14000US06013XXXXXX' -> strip prefix to get GEOID
        tract_geoid = geo_id.replace("14000US", "")
        income = tables.get("B19013", {}).get("estimate", {}).get("B19013001")
        if income is not None and income > 0:
            rows.append({"tract_geoid": tract_geoid, "median_income": float(income)})

    return pd.DataFrame(rows)


def _geocode_batch(addresses: list[dict]) -> dict:
    """
    Census Batch Geocoder — up to 1,000 addresses per call.
    Each dict: {id, street, city, state, zip, _key}
    Returns {id -> tract_geoid}.
    """
    csv_body = "id,street,city,state,zip\n"
    for a in addresses:
        street = re.sub(r"\s+(unit|apt|#|suite)\s+[\w-]+", "", a["street"], flags=re.IGNORECASE).strip()
        csv_body += f'{a["id"]},"{street}","{a["city"]}",{a["state"]},{a["zip"]}\n'

    resp = requests.post(
        "https://geocoding.geo.census.gov/geocoder/geographies/addressbatch",
        data={
            "benchmark":  "Public_AR_Current",
            "vintage":    "Current_Current",
            "layers":     "Census Tracts",
            "returntype": "geographies",
        },
        files={"addressFile": ("addresses.csv", csv_body.encode(), "text/csv")},
        timeout=180,
    )
    resp.raise_for_status()

    result = {}
    reader = csv.reader(io.StringIO(resp.text))
    for parts in reader:
        if len(parts) < 11 or parts[2].strip().lower() != "match":
            continue
        uid       = parts[0].strip()
        state_fp  = parts[8].strip().zfill(2)
        county_fp = parts[9].strip().zfill(3)
        tract_fp  = parts[10].strip().zfill(6)
        if state_fp and county_fp and tract_fp:
            result[uid] = f"{state_fp}{county_fp}{tract_fp}"
    return result


def add_median_income(df: pd.DataFrame) -> pd.DataFrame:
    print("\nFetching ACS 5-year median household income (Contra Costa County tracts)...")
    income_df = fetch_acs_income()
    print(f"  {len(income_df)} census tracts loaded")

    # Cache keyed by "address|city|zip" so re-runs skip already-geocoded rows
    cache: dict = {}
    if TRACT_CACHE.exists():
        with open(TRACT_CACHE) as f:
            cache = json.load(f)
        print(f"  {len(cache)} cached tract lookups")

    def _cache_key(row):
        return f"{row.get('address', '')}|{row.get('city', '')}|{row.get('zip_code', '')}"

    df["ckey"] = df.apply(_cache_key, axis=1)
    new_mask = ~df["ckey"].isin(cache)
    new_rows = df[new_mask].copy().reset_index(drop=True)

    if len(new_rows) > 0:
        print(f"  Geocoding {len(new_rows)} new addresses in batches of 1,000...")
        for i in range(0, len(new_rows), 1000):
            batch = new_rows.iloc[i : i + 1000]
            addresses = [
                {
                    "id":    str(j),
                    "street": str(row.address),
                    "city":  str(row.city),
                    "state": STATE,
                    "zip":   str(row.zip_code),
                    "key":   row.ckey,
                }
                for j, row in enumerate(batch.itertuples())
            ]
            try:
                id_to_geoid = _geocode_batch(addresses)
                for addr in addresses:
                    geoid = id_to_geoid.get(addr["id"])
                    if geoid:
                        cache[addr["key"]] = geoid
                print(f"    Batch {i // 1000 + 1}: {len(id_to_geoid)}/{len(batch)} matched")
            except Exception as exc:
                print(f"    WARNING: batch {i // 1000 + 1} failed — {exc}")
            time.sleep(2)

        TRACT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACT_CACHE, "w") as f:
            json.dump(cache, f)

    df["tract_geoid"] = df["ckey"].map(cache).astype(str)
    df = df.drop(columns=["ckey"])
    income_df["tract_geoid"] = income_df["tract_geoid"].astype(str)
    df = df.merge(income_df, on="tract_geoid", how="left")
    matched = df["median_income"].notna().sum()
    print(f"  {matched}/{len(df)} properties matched to income data")
    return df


# ── Walk Score (optional) ──────────────────────────────────────────────────────

def add_walk_scores(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Walk Score + Transit Score per property.
    Free API key: https://www.walkscore.com/professional/api.php
    Set env var WALKSCORE_API_KEY to enable.
    """
    print("\nFetching Walk Scores...")
    walk_scores, transit_scores = [], []
    for _, row in df.iterrows():
        ws, ts = float("nan"), float("nan")
        if not (pd.isna(row.get("latitude")) or pd.isna(row.get("longitude"))):
            try:
                addr = requests.utils.quote(str(row.get("address", "")))
                url = (
                    f"https://api.walkscore.com/score/json?format=json"
                    f"&address={addr}&lat={row['latitude']}&lon={row['longitude']}"
                    f"&transit=1&wsapikey={api_key}"
                )
                data = requests.get(url, timeout=10).json()
                ws = data.get("walkscore")
                transit = data.get("transit")
                ts = transit.get("score") if isinstance(transit, dict) else None
            except Exception:
                pass
        walk_scores.append(ws)
        transit_scores.append(ts)
        time.sleep(0.12)

    df["walk_score"]    = walk_scores
    df["transit_score"] = transit_scores
    n = sum(x == x for x in walk_scores)
    print(f"  {n}/{len(df)} walk scores retrieved")
    return df


# ── Pipeline ───────────────────────────────────────────────────────────────────

def build_dataset(days: int = 1095, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    date_to   = datetime.now().strftime("%Y-%m-%d")
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # 1. Realtor.com via HomeHarvest
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

    # 2. FRED mortgage rates
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

    # 3. FRED unemployment
    print("\nFetching California unemployment from FRED...")
    unemp = fetch_unemployment()
    df["year"]  = df["date_of_sale"].dt.year
    df["month"] = df["date_of_sale"].dt.month
    df = df.merge(unemp, on=["year", "month"], how="left")
    print(f"  Joined {len(unemp)} monthly unemployment observations")

    # 4. BART distance
    print("\nComputing distance to nearest BART station...")
    df = add_bart_distance(df)

    # 5. Census median income
    df = add_median_income(df)

    # 6. Walk Score (optional)
    ws_key = os.environ.get("WALKSCORE_API_KEY")
    if ws_key:
        df = add_walk_scores(df, ws_key)
    else:
        print("\nSkipping Walk Score (set WALKSCORE_API_KEY env var to include it)")

    # 7. lot_sqft: condos/townhomes share a parcel — force 0 regardless of what Realtor.com reports.
    #    Single-family: impute missing values with SF median.
    sf_mask = df["type"] == "Single-family"
    sf_lot_median = df.loc[sf_mask & df["lot_sqft"].notna(), "lot_sqft"].median()
    df.loc[sf_mask & df["lot_sqft"].isna(), "lot_sqft"] = sf_lot_median
    df.loc[~sf_mask, "lot_sqft"] = 0

    # 8. Impute stories: 1 for missing (majority of homes are single-story)
    df["stories"] = df["stories"].fillna(1)

    # 9. Final column selection
    base_cols = [
        "address", "city", "date_of_sale", "sold_price",
        "type", "bedrooms", "bathrooms", "sq_ft", "lot_sqft",
        "build_age", "stories", "garage", "hoa_fee",
        "school_score", "median_income", "dist_bart_miles",
        "unemployment", "interest_rate",
    ]
    ws_cols = ["walk_score", "transit_score"] if ws_key else []
    out_cols = [c for c in base_cols + ws_cols if c in df.columns]

    result = (
        df[out_cols]
        .dropna(subset=["sold_price", "sq_ft", "bedrooms", "build_age", "interest_rate"])
        .copy()
    )
    result["date_of_sale"] = result["date_of_sale"].apply(
        lambda d: f"{d.month}/{d.day}/{d.strftime('%y')}" if pd.notna(d) else ""
    )
    for col in ["sold_price", "bedrooms", "sq_ft", "build_age"]:
        result[col] = result[col].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nSaved {len(result)} rows -> {output_path}")
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1095)
    args = parser.parse_args()

    df = build_dataset(days=args.days)
    print(f"\nCity breakdown:\n{df['city'].value_counts().to_string()}")
    print(f"\nType breakdown:\n{df['type'].value_counts().to_string()}")
    print(f"\nNew features sample:\n{df[['bathrooms','lot_sqft','garage','hoa_fee','stories','dist_bart_miles','median_income']].describe().to_string()}")
