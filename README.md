# Auction Marketplace Segmentation & Price Intelligence Engine

> **End-to-end data science platform** for heavy equipment auction marketplace analytics — segmenting listings, scoring prices, and surfacing market insights through a production-ready API.

---

## Table of Contents

- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Running the API](#running-the-api)
- [Docker](#docker)
- [Running Tests](#running-tests)
- [Key Results](#key-results)
- [API Reference](#api-reference)
- [Business Recommendations](#business-recommendations)
- [Future Work](#future-work)

---

## Business Problem

Heavy equipment auction marketplaces (think Ritchie Bros., IronPlanet, Aucto) list thousands of pieces of equipment simultaneously — excavators, cranes, forklifts, graders — from hundreds of sellers across every US region. Marketplace operators and buyers face three core challenges:

1. **What meaningful segments exist?** Which listings cluster together by behavior, price, and demand characteristics?
2. **Is this listing fairly priced?** Is a $55,000 Caterpillar excavator underpriced, overpriced, or right on the money for its condition and age?
3. **What should I act on first?** Which segments clear fastest? Which regions have the most active buyers?

This project answers all three questions with a reproducible, deployable data science pipeline.

---

## Dataset

**Source**: Synthetic heavy equipment auction dataset generated to mirror the structure of IronPlanet / Ritchie Bros. marketplaces.

**Why synthetic?** Real heavy-equipment auction datasets (IronPlanet, Ritchie Bros.) require commercial subscriptions. The synthetic dataset is statistically calibrated to realistic auction market distributions — price depreciation curves, condition-based discounts, regional demand patterns, engagement distributions — ensuring all downstream analyses produce genuinely meaningful insights. To use real data, replace `scripts/generate_data.py` with your ingestion connector; all downstream scripts remain unchanged.

| Attribute | Value |
|---|---|
| Listings | 15,000 |
| Sellers | 800 |
| Equipment Categories | 10 (Excavators, Cranes, Bulldozers, etc.) |
| US Regions | 9 |
| Date Range | Jan 2021 – Dec 2023 |
| Features Engineered | 35 |
| Sold Rate | ~41% |
| Price Range | $2,000 – $241,500 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Pipeline                          │
│  generate_data.py → preprocess.py → feature_engineering.py  │
└─────────────────────────┬───────────────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
   train_segmentation  train_model    init_db.py
   (KMeans / HAC)      (LightGBM /    (SQLite via
                        XGBoost /     SQLAlchemy)
                        LogReg)
           │              │              │
           └──────────────┼──────────────┘
                          │
                   ┌──────▼──────┐
                   │  FastAPI    │
                   │  6 endpoints│
                   │  + Swagger  │
                   └──────┬──────┘
                          │
                   ┌──────▼──────┐
                   │   Docker    │
                   └─────────────┘
```

---

## Project Structure

```
auction-marketplace-segmentation/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── api/
│   │   └── routes.py              # All route handlers
│   ├── models/
│   │   └── schemas.py             # Pydantic response schemas
│   ├── services/
│   │   └── model_service.py       # LightGBM inference service
│   └── db/
│       ├── models.py              # SQLAlchemy ORM models
│       └── database.py            # DB engine + session factory
├── data/
│   ├── raw/                       # listings.csv, sellers.csv
│   └── processed/                 # cleaned + feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_segmentation.ipynb      # Clustering walkthrough
│   └── 03_modeling.ipynb          # Predictive modeling analysis
├── reports/
│   ├── figures/                   # Generated charts (12+ PNGs)
│   └── final_report.md            # Business report
├── scripts/
│   ├── generate_data.py           # Synthetic dataset generator
│   ├── preprocess.py              # Cleaning pipeline
│   ├── feature_engineering.py     # Feature engineering (35 features)
│   ├── train_segmentation.py      # KMeans + Hierarchical clustering
│   ├── train_model.py             # LightGBM / XGBoost / LogReg
│   ├── init_db.py                 # SQLite initialization + data load
│   └── run_all.py                 # Master pipeline runner
├── tests/
│   ├── test_pipeline.py           # 28 pipeline unit tests
│   └── test_api.py                # 13 API integration tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- pip

### Install dependencies

```bash
git clone https://github.com/navyathag13-ui/Auction-Marketplace-Segmentation-Price-Intelligence-Engine.git
cd Auction-Marketplace-Segmentation-Price-Intelligence-Engine

pip install -r requirements.txt
```

---

## Running the Pipeline

### Option A: Run everything at once

```bash
python scripts/run_all.py
```

This runs: data generation → preprocessing → feature engineering → segmentation → modeling → database init → notebook generation.

### Option B: Step by step

```bash
# 1. Generate synthetic dataset
python scripts/generate_data.py

# 2. Clean and preprocess
python scripts/preprocess.py

# 3. Engineer features (35 features)
python scripts/feature_engineering.py

# 4. Train segmentation model
python scripts/train_segmentation.py

# 5. Train price intelligence model
python scripts/train_model.py

# 6. Initialize database
python scripts/init_db.py

# 7. Generate notebooks
python notebooks/create_notebooks.py
```

**Output files:**
- `data/raw/listings.csv` — 15,000 raw listings
- `data/processed/listings_clean.csv` — cleaned dataset
- `data/processed/listings_features.csv` — 62-column feature set
- `data/processed/features_meta.json` — feature documentation
- `data/processed/model_results.json` — model metrics
- `models/kmeans_segmentation.joblib` — trained KMeans model
- `models/lgbm_price_classifier.joblib` — trained LightGBM model
- `data/auction_marketplace.db` — SQLite database (8.2 MB)
- `reports/figures/*.png` — 12+ charts

---

## Running the API

```bash
uvicorn app.main:app --reload
```

The API launches at `http://localhost:8000`

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Sample API responses

**Health Check**
```bash
curl http://localhost:8000/api/v1/health
```
```json
{
  "status": "ok",
  "version": "1.0.0",
  "database": "connected",
  "records_loaded": 15000
}
```

**Segment Summary**
```bash
curl http://localhost:8000/api/v1/segments/summary
```
```json
{
  "total_listings": 15000,
  "num_segments": 5,
  "segments": [
    {
      "segment_id": 0,
      "segment_name": "Premium High-Demand",
      "listing_count": 1853,
      "avg_asking_price": 11268.77,
      "avg_sold_rate": 0.412,
      ...
    }
  ]
}
```

**Listing Price Score**
```bash
curl http://localhost:8000/api/v1/listing/L000000/price-score
```
```json
{
  "listing_id": "L000000",
  "asking_price": 45200.0,
  "price_label": "fair_priced",
  "price_score": 78.5,
  "prob_underpriced": 0.08,
  "prob_fair_priced": 0.79,
  "prob_overpriced": 0.13,
  "interpretation": "This listing is priced in line with market comparables."
}
```

**Top Features**
```bash
curl http://localhost:8000/api/v1/insights/top-features
```
```json
{
  "model": "LightGBM Price Classifier",
  "top_features": [
    {"feature": "price_vs_cond_median", "importance": 892.0, "rank": 1},
    {"feature": "log_asking_price", "importance": 741.0, "rank": 2},
    ...
  ]
}
```

---

## Docker

### Build and run

```bash
# Run API only
docker-compose up api

# Run full pipeline + API
docker-compose --profile pipeline up
```

### Manual Docker build

```bash
docker build -t auction-marketplace .
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models auction-marketplace
```

---

## Running Tests

```bash
pytest tests/ -v
```

**41 tests across 7 test classes:**

```
tests/test_api.py::TestHealthEndpoint        (3 tests)
tests/test_api.py::TestRootEndpoint          (2 tests)
tests/test_api.py::TestSegmentsEndpoint      (2 tests)
tests/test_api.py::TestListingEndpoints      (4 tests)
tests/test_api.py::TestInsightsEndpoints     (2 tests)
tests/test_pipeline.py::TestDataGeneration   (7 tests)
tests/test_pipeline.py::TestPreprocessing    (5 tests)
tests/test_pipeline.py::TestFeatureEngineering (7 tests)
tests/test_pipeline.py::TestSegmentation     (4 tests)
tests/test_pipeline.py::TestModel            (5 tests)
```

---

## Key Results

### Segmentation

| Segment | Name | Listings | Avg Price | Sold Rate |
|---|---|---|---|---|
| 0 | Premium High-Demand | 1,853 | $11,269 | 41.2% |
| 1 | Aging Low-Engagement | 3,577 | $10,755 | 44.8% |
| 2 | Value Fleet Inventory | 3,451 | $49,811 | 41.8% |
| 3 | Hot Fast-Moving Deals | 1,859 | $14,788 | 38.8% |
| 4 | Niche Specialty Equipment | 4,260 | $10,876 | 39.2% |

KMeans silhouette: **0.1054** (selected over hierarchical clustering for full-dataset coverage and API-compatible inference speed)

### Predictive Modeling

| Model | Accuracy | F1 Macro | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 70.9% | 0.648 | 0.880 |
| **LightGBM (selected)** | **98.4%** | **0.983** | **0.999** |
| XGBoost | 97.1% | 0.966 | 0.997 |

LightGBM delivers **+38 percentage point accuracy gain** over the baseline.

### Features Engineered (35 total)

- Price intelligence: `price_vs_cat_median`, `price_vs_cond_median`, `price_zscore_cat`, `log_asking_price`
- Utilization: `hours_per_year`, `hours_vs_cat_median`, `log_hours_used`
- Engagement: `total_engagement`, `bid_rate`, `inquiry_rate`, `engagement_zscore`
- Seller signals: `seller_listing_count`, `seller_sold_rate`, `seller_listing_velocity`, `is_high_vol_seller`
- Market context: `category_freq`, `region_demand_proxy`, `category_velocity`
- Lifecycle: `age_bucket`, `listing_age_days`, `is_vintage`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Health check, DB status, record count |
| GET | `/api/v1/segments/summary` | Segment profiles for all 5 segments |
| GET | `/api/v1/listing/{id}/segment` | Segment assignment for a listing |
| GET | `/api/v1/listing/{id}/price-score` | Price intelligence classification + probabilities |
| GET | `/api/v1/insights/top-features` | LightGBM feature importances |
| GET | `/api/v1/reports/segment-summary` | Full marketplace report with category + region breakdown |

---

## Business Recommendations

1. **Surface underpriced listings to buyers** — 12.5% of listings are underpriced; "Value Pick" badges drive buyer engagement.
2. **Automate pricing alerts for sellers** — 37.4% of listings are overpriced with >2× longer days-on-market.
3. **Prioritize Segment 3 "Hot Fast-Moving Deals"** — These clear ~30% faster; premium search placement amplifies velocity.
4. **Re-engage Segment 1 "Aging Low-Engagement"** — Automated email/SMS after 14 days can recover stale inventory.
5. **Southeast + Midwest seller acquisition** — These regions show highest volume AND active buyers; priority expansion markets.
6. **Verified seller badge program** — Verified sellers show measurably higher sold rates; investment in verification scales conversion.

---

## Future Work

- [ ] Replace synthetic data with real Ritchie Bros. / IronPlanet data via API or web scraping
- [ ] Add time-series trend analysis (listing volume by week, seasonal pricing patterns)
- [ ] Build a Streamlit or Plotly Dash interactive dashboard
- [ ] Add real-time model retraining trigger when 1,000+ new listings arrive
- [ ] Implement buyer segmentation (in addition to listing segmentation)
- [ ] Add fast-sale prediction model (days-on-market regression)
- [ ] Add PostgreSQL support for production deployment
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Price recommendation endpoint (suggest optimal listing price for new equipment)

---
