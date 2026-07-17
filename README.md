# House Price Prediction with Synthetic Data Augmentation

Predicts home sale prices in Contra Costa County, CA using a two-phase approach:
first overcoming a 15-row dataset with synthetic data augmentation (SDV), then
expanding to **3,173 real sales** via a purpose-built data collection pipeline.

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

### Phase 2 — Real Data (3,173 rows)

`scripts/collect_data.py` built a pipeline to collect real sales data automatically,
making synthetic augmentation no longer necessary.

| Model | Test RMSE | Test MAE | Test R² | 5-Fold CV RMSE |
|---|---|---|---|---|
| Random Forest (200 trees) | $171,164 | $114,886 | 0.862 | $195,769 ± $14,298 |
| Gradient Boosting (200 trees, depth 4) | $173,375 | $115,579 | 0.858 | $195,186 ± $12,487 |

Both models explain ~86% of variance in home sale prices on unseen data.

---

## Dataset

3,173 home sales across 4 Contra Costa County cities (2023–2026):

| Feature | Description | Source |
|---|---|---|
| `type` | Property type (Condo / Single-family / Townhome) | Realtor.com |
| `city` | Concord, Walnut Creek, Martinez, or Pleasant Hill | Realtor.com |
| `bedrooms` | Number of bedrooms | Realtor.com |
| `sq_ft` | Square footage | Realtor.com |
| `build_age` | Years since the property was built | Realtor.com |
| `school_score` | City-level school rating proxy¹ | Original 15-row dataset |
| `unemployment` | California unemployment rate at time of sale | FRED (`CAURN`) |
| `interest_rate` | 30-year fixed mortgage rate at time of sale | FRED (`MORTGAGE30US`) |
| `sold_price` | **Target** — sale price in USD | Realtor.com |

¹ _Property-level values would require the GreatSchools API. The city-level proxy
(Concord: 9, Walnut Creek: 22, Martinez: 19, Pleasant Hill: 17) is a known limitation._

---

## How It Works

### Data Collection (`scripts/collect_data.py`)

```
Realtor.com (HomeHarvest) ──► address, price, beds, sqft, type, year built, sale date
FRED MORTGAGE30US          ──► weekly 30-yr fixed rate  ──► joined on nearest date to sale
FRED CAURN                 ──► monthly CA unemployment  ──► joined by year + month of sale
                                        │
                                        ▼
                           data/house_sales_extended.csv
```

Re-run anytime to refresh with the latest sales:
```bash
python scripts/collect_data.py --days 1095   # default: 3 years
```

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
| [`model_training.ipynb`](notebooks/model_training.ipynb) | **Primary** — EDA, preprocessing, RF training, CV, feature importance, and Gradient Boosting comparison on 3,173 real sales |
| [`sdv_synthetic_model.ipynb`](notebooks/sdv_synthetic_model.ipynb) | Phase 1 archive — synthetic data generation and validation on 15 real rows |
| [`baseline_pipeline.ipynb`](notebooks/baseline_pipeline.ipynb) | Early pipeline exploration and 2-feature baseline model |

## Project Structure

```
ElijahA/
├── data/
│   ├── house_sales.csv              # Original 15 real sales
│   └── house_sales_extended.csv     # 3,173 sales from collect_data.py
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
- **pytest** — 7 unit tests for custom transformers
