"""
pipeline.py
End-to-end training pipeline for house price prediction.
Loads the enriched dataset, trains a Random Forest, and prints metrics.

Usage:
    python src/ElijahA/pipeline.py
"""

import math
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.ElijahA.preprocessing import TypeDummyCreator

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "house_sales_extended.csv"

FEATURES = [
    # Property
    "type", "city", "bedrooms", "bathrooms", "sq_ft", "lot_sqft",
    "build_age", "stories", "garage", "hoa_fee",
    # Neighborhood / spatial
    "school_score", "median_income", "dist_bart_miles",
    # Macro
    "unemployment", "interest_rate",
]
LABEL = "sold_price"

# Columns needing median imputation (NaN from scraper gaps or FRED publication lag)
_IMPUTE_MEDIAN = ["unemployment", "bathrooms", "dist_bart_miles", "median_income"]

# Hard caps on known HomeHarvest outliers (data entry errors in MLS)
_OUTLIER_CAPS = {
    "lot_sqft":  43_560,  # 1 acre — 99th pct for residential
    "garage":    8,
    "stories":   5,
    "bathrooms": 6,
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in _IMPUTE_MEDIAN:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col, cap in _OUTLIER_CAPS.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("encode", TypeDummyCreator(columns=["type", "city"])),
        ("model",  RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])


def main():
    df = load_data()
    print(f"Loaded {len(df):,} rows from {DATA_PATH.name}")

    X = df[FEATURES]
    y = df[LABEL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds   = pipeline.predict(X_test)
    rmse    = math.sqrt(mean_squared_error(y_test, preds))
    mae     = mean_absolute_error(y_test, preds)
    r2      = r2_score(y_test, preds)

    print(f"Test RMSE : ${rmse:>10,.0f}")
    print(f"Test MAE  : ${mae:>10,.0f}")
    print(f"Test R2   :  {r2:>9.3f}")

    cv = cross_val_score(
        pipeline, X, y,
        cv=KFold(5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
    )
    print(f"\n5-fold CV RMSE: ${-cv.mean():,.0f} (+/- ${(-cv).std():,.0f})")
    return pipeline


if __name__ == "__main__":
    main()
