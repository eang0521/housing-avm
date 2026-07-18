# House Price Prediction with Synthetic Data Augmentation

Predicts home sale prices in Contra Costa County, CA using a two-phase approach:
first overcoming a 15-row dataset with synthetic data augmentation (SDV), then
expanding to **9,049 real sales** via a purpose-built data collection pipeline.

Built for the **Data Detectives 2025** internship.

---

## Project Evolution

### Phase 1 — Synthetic Data (15 real rows)

The original dataset contained only 15 home sales — far too few to train a model.
[SDV (Synthetic Data Vault)](https://sdv.dev/) was used to fit a `GaussianCopulaSynthesizer`
on the real distribution and sample 1,000 synthetic rows, preserving feature correlations
(e.g. sq_ft vs. sold_price).

| Evaluation | RMSE |
|---|---|
| Synthetic test set (80/20 split) | $197,427 |
| Synthetic-trained model → 15 real rows | $151,077 |
| Leave-One-Out CV on real data | $139,915 |

### Phase 2 — Real Data (9,038 rows, 18 features)

`scripts/collect_data.py` built a pipeline to collect real sales data automatically,
making synthetic augmentation no longer necessary. Phase 2 adds bathrooms, lot size,
garage, HOA fee, stories, BART distance, census-tract median income, zip code, and
raw latitude/longitude for fine-grained spatial signal.

Four targeted improvements vs. the initial Phase 2 run:
1. **Log-transform target** — models train on `log1p(price)`, cutting RMSE ~15%
2. **Lat/lon as features** — raw coordinates capture within-city spatial variation
3. **Zip code** — 8 distinct zip codes give finer location resolution than 4 cities
4. **XGBoost** — gradient boosting with 500 trees and learning_rate=0.05

| Model | Test RMSE | Test MAE | Test R² | 5-Fold CV RMSE |
|---|---|---|---|---|
| Random Forest (log-target) | $176,305 | $108,666 | 0.867 | $195,542 ± $35,198 |
| XGBoost (log-target) | $177,063 | $108,074 | 0.866 | $193,054 ± $31,864 |

Both models now explain ~87% of price variance. RMSE dropped from $207K to $176K
(~15% improvement) primarily from the log transform compressing the right skew.

---

## Dataset

9,038 home sales across 4 Contra Costa County cities (2023–2026):

| Feature | Description | Source |
|---|---|---|
| `type` | Property type (Condo / Single-family / Townhome) | Realtor.com |
| `city` | Concord, Walnut Creek, Martinez, or Pleasant Hill | Realtor.com |
| `bedrooms` | Number of bedrooms | Realtor.com |
| `bathrooms` | Number of full bathrooms | Realtor.com |
| `sq_ft` | Interior square footage | Realtor.com |
| `lot_sqft` | Lot size in sq ft (0 for condos/townhomes) | Realtor.com |
| `build_age` | Years since the property was built | Realtor.com |
| `stories` | Number of floors | Realtor.com |
| `garage` | Number of garage spaces | Realtor.com |
| `hoa_fee` | Monthly HOA fee (0 if none) | Realtor.com |
| `school_score` | City-level school rating proxy¹ | Original 15-row dataset |
| `median_income` | Census tract median household income | ACS 5-yr (Census Reporter) |
| `dist_bart_miles` | Distance to nearest BART station | Computed from lat/lon |
| `unemployment` | California unemployment rate at time of sale | FRED (`CAURN`) |
| `interest_rate` | 30-year fixed mortgage rate at time of sale | FRED (`MORTGAGE30US`) |
| `zip_code` | 5-digit ZIP code (~8 distinct in the dataset) | Realtor.com |
| `latitude` | Property latitude | Realtor.com |
| `longitude` | Property longitude | Realtor.com |
| `sold_price` | **Target** — sale price in USD | Realtor.com |

¹ _Property-level values would require the GreatSchools API. The city-level proxy
(Concord: 9, Walnut Creek: 22, Martinez: 19, Pleasant Hill: 17) is a known limitation._

---

## How It Works

### Data Collection (`scripts/collect_data.py`)

```
Realtor.com (HomeHarvest) ──► address, price, beds, baths, sqft, lot, garage, HOA,
                               stories, type, year built, sale date, lat/lon
FRED MORTGAGE30US          ──► weekly 30-yr fixed rate  ──► joined on nearest date to sale
FRED CAURN                 ──► monthly CA unemployment  ──► joined by year + month of sale
Census Batch Geocoder      ──► address -> census tract GEOID (cached in data/tract_cache.json)
Census Reporter API        ──► tract median household income ──► joined by GEOID
BART station coordinates   ──► Haversine distance to nearest of 7 stations
                                        |
                                        v
                           data/house_sales_extended.csv
```

Re-run anytime to refresh with the latest sales:
```bash
python scripts/collect_data.py --days 1095   # default: 3 years
```

Walk Score data is optional — set the `WALKSCORE_API_KEY` environment variable to include
`walk_score` and `transit_score` columns (free key at walkscore.com/professional/api.php).

### Model Pipeline (`src/ElijahA/pipeline.py`)

```python
Pipeline([
    ('encode', TypeDummyCreator(columns=['type', 'city'])),   # one-hot, fit-aware
    ('model',  RandomForestRegressor(n_estimators=200)),
])
```

`TypeDummyCreator` (from `src/ElijahA/preprocessing.py`) stores seen categories at fit
time and uses `reindex` at transform time, so inference never breaks on missing categories.

---

## Notebooks

| Notebook | Description |
|---|---|
| [`model_training.ipynb`](notebooks/model_training.ipynb) | **Primary** — EDA, preprocessing, log-target RF and XGBoost training, CV, and feature importance on 9,038 real sales with 18 features |
| [`sdv_synthetic_model.ipynb`](notebooks/sdv_synthetic_model.ipynb) | Phase 1 archive — synthetic data generation and validation on 15 real rows |
| [`baseline_pipeline.ipynb`](notebooks/baseline_pipeline.ipynb) | Early pipeline exploration and 2-feature baseline model |

## Project Structure

```
ElijahA/
├── data/
│   ├── house_sales.csv              # Original 15 real sales
│   ├── house_sales_extended.csv     # 9,049 sales from collect_data.py
│   └── tract_cache.json             # Geocoded census tract IDs (cached)
├── notebooks/
│   ├── model_training.ipynb         # Primary model notebook (Phase 2)
│   ├── sdv_synthetic_model.ipynb    # Synthetic data approach (Phase 1)
│   └── baseline_pipeline.ipynb      # Pipeline exploration
├── scripts/
│   └── collect_data.py              # Data collection pipeline
├── src/ElijahA/
│   ├── preprocessing.py             # TypeDummyCreator + custom transformers
│   └── pipeline.py                  # End-to-end training script
└── tests/
    └── test_preprocessing.py        # 7 unit tests for TypeDummyCreator
```

---

## Setup

Requires Python 3.11–3.13 and [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run python -m ipykernel install --user --name elijaha --display-name "Python (elijaha)"
```

Open a notebook in VS Code and select the **Python (elijaha)** kernel, or run the
training pipeline directly:

```bash
python src/ElijahA/pipeline.py
```

To refresh the dataset:

```bash
python scripts/collect_data.py --days 1095
```

---

## Tech Stack

- **Python 3.13** · **scikit-learn** · **pandas** · **numpy** · **matplotlib**
- **SDV** — Gaussian Copula synthesizer (Phase 1)
- **HomeHarvest** — Realtor.com data collection
- **FRED API** — mortgage rates and unemployment
- **Census Batch Geocoder + Census Reporter** — tract-level median income
- **pytest** — 7 unit tests for custom transformers
