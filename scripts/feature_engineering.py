"""
Feature Engineering Pipeline
------------------------------
Reads cleaned listings and engineers business-relevant features for:
  - Segmentation (unsupervised)
  - Predictive modeling (supervised)

Features documented inline.

Output:
  data/processed/listings_features.csv   — full feature set
  data/processed/features_meta.json      — feature documentation
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

FEATURE_DOCS: dict[str, str] = {}


def doc(name: str, description: str):
    FEATURE_DOCS[name] = description


# ── Price intelligence features ──────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    # Category median asking price
    cat_median = df.groupby("category")["asking_price"].transform("median")
    df["price_vs_cat_median"] = df["asking_price"] / cat_median
    doc("price_vs_cat_median",
        "Ratio of listing asking price to its category median. >1 = premium priced, <1 = discount priced.")

    # Category-condition median
    cat_cond_median = df.groupby(["category", "condition"])["asking_price"].transform("median")
    df["price_vs_cond_median"] = df["asking_price"] / cat_cond_median
    doc("price_vs_cond_median",
        "Ratio of asking price to the median for same category+condition. Captures pricing relative to comparable peers.")

    # Price per year of life (depreciation proxy)
    df["price_per_year"] = df["asking_price"] / (df["age_years"].clip(lower=1))
    doc("price_per_year",
        "Asking price divided by equipment age. Higher = higher per-year retained value.")

    # Log price (stabilizes variance for modeling)
    df["log_asking_price"] = np.log1p(df["asking_price"])
    doc("log_asking_price",
        "Natural log of asking_price + 1. Used to normalize skewed price distribution.")

    # Z-score within category
    df["price_zscore_cat"] = df.groupby("category")["asking_price"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9)
    )
    doc("price_zscore_cat",
        "Z-score of asking price within its category. Captures how far a listing deviates from category average.")

    return df


# ── Listing age & lifecycle features ─────────────────────────────────────────

def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    # Age already in clean data; add buckets
    df["age_bucket"] = pd.cut(
        df["age_years"],
        bins=[-1, 3, 7, 12, 18, 100],
        labels=["0-3yr", "4-7yr", "8-12yr", "13-18yr", "18+yr"],
    ).astype(str)
    doc("age_bucket", "Equipment age bucketed into lifecycle stages.")

    # Vintage premium flag (older equipment sometimes commands premium)
    df["is_vintage"] = (df["age_years"] >= 20).astype(int)
    doc("is_vintage", "Binary: 1 if equipment is 20+ years old (potential vintage/classic value).")

    # Listing recency in days (from latest list date in dataset)
    max_date = df["list_date"].max()
    df["listing_age_days"] = (max_date - df["list_date"]).dt.days
    doc("listing_age_days",
        "Number of days since the listing was posted (relative to most recent listing in dataset).")

    return df


# ── Hours & utilization features ─────────────────────────────────────────────

def add_hours_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hours_per_year"] = df["hours_used"] / (df["age_years"].clip(lower=1))
    doc("hours_per_year",
        "Average hours used per year of equipment life. High = heavy use, Low = light use.")

    # Hours vs category median
    cat_hrs_median = df.groupby("category")["hours_used"].transform("median")
    df["hours_vs_cat_median"] = df["hours_used"] / (cat_hrs_median + 1)
    doc("hours_vs_cat_median",
        "Ratio of hours used to category median. >1 = higher utilization than peers.")

    # Log hours
    df["log_hours_used"] = np.log1p(df["hours_used"])
    doc("log_hours_used", "Log-transformed hours used. Normalizes right-skewed distribution.")

    return df


# ── Engagement & demand proxy features ───────────────────────────────────────

def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    # Total engagement
    df["total_engagement"] = df["views"] + df["bids"] * 3 + df["inquiries"] * 2
    doc("total_engagement",
        "Weighted sum of views (×1), bids (×3), inquiries (×2). Captures listing demand intensity.")

    # Bid-to-view ratio
    df["bid_rate"] = df["bids"] / (df["views"] + 1)
    doc("bid_rate", "Ratio of bids to views. High value = strong buyer interest per exposure.")

    # Inquiry-to-view ratio
    df["inquiry_rate"] = df["inquiries"] / (df["views"] + 1)
    doc("inquiry_rate", "Ratio of inquiries to views. Captures serious buyer intent.")

    # Engagement z-score
    df["engagement_zscore"] = (
        (df["total_engagement"] - df["total_engagement"].mean())
        / (df["total_engagement"].std() + 1e-9)
    )
    doc("engagement_zscore",
        "Z-score of total_engagement across all listings. Identifies high/low demand listings.")

    # Log views
    df["log_views"] = np.log1p(df["views"])
    doc("log_views", "Log-transformed view count.")

    return df


# ── Seller features ───────────────────────────────────────────────────────────

def add_seller_features(df: pd.DataFrame) -> pd.DataFrame:
    # Seller listing volume (computed from current data, not raw total_listings field)
    seller_vol = df.groupby("seller_id")["listing_id"].transform("count")
    df["seller_listing_count"] = seller_vol
    doc("seller_listing_count",
        "Number of listings this seller has in the dataset. Proxy for seller activity level.")

    # Seller avg price in dataset
    seller_avg_price = df.groupby("seller_id")["asking_price"].transform("mean")
    df["seller_avg_price"] = seller_avg_price
    doc("seller_avg_price",
        "Average asking price across all of this seller's listings. Captures seller market tier.")

    # Seller sell rate
    seller_sold_rate = df.groupby("seller_id")["sold"].transform("mean")
    df["seller_sold_rate"] = seller_sold_rate
    doc("seller_sold_rate",
        "Fraction of this seller's listings that sold. Proxy for seller effectiveness.")

    # High-volume seller flag (top quartile)
    vol_q75 = df["seller_listing_count"].quantile(0.75)
    df["is_high_vol_seller"] = (df["seller_listing_count"] >= vol_q75).astype(int)
    doc("is_high_vol_seller",
        "Binary: 1 if seller is in top 25% by listing volume (power seller).")

    # Seller verified flag (already in data, ensure numeric)
    df["seller_verified_flag"] = df["seller_verified"].astype(int)
    doc("seller_verified_flag", "Binary: 1 if seller is verified on the platform.")

    return df


# ── Category & region demand features ────────────────────────────────────────

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    # Category listing frequency (relative popularity)
    cat_freq = df["category"].value_counts(normalize=True)
    df["category_freq"] = df["category"].map(cat_freq)
    doc("category_freq",
        "Proportion of all listings that belong to this category. Captures category-level supply.")

    # Region demand proxy: avg engagement score per region
    region_demand = df.groupby("region")["total_engagement"].transform("mean")
    df["region_demand_proxy"] = region_demand
    doc("region_demand_proxy",
        "Mean total engagement for all listings in this region. Higher = more active market region.")

    # Region listing density
    region_freq = df["region"].value_counts(normalize=True)
    df["region_freq"] = df["region"].map(region_freq)
    doc("region_freq",
        "Proportion of all listings in this region. Captures regional supply concentration.")

    return df


# ── Condition & quality encoding ──────────────────────────────────────────────

def add_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    cond_map = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
    df["condition_score"] = df["condition"].map(cond_map).fillna(1)
    doc("condition_score",
        "Ordinal encoding of condition: Poor=0, Fair=1, Good=2, Excellent=3.")

    # Interaction: condition × price ratio
    df["condition_price_interaction"] = df["condition_score"] * df["price_vs_cat_median"]
    doc("condition_price_interaction",
        "Product of condition_score and price_vs_cat_median. Captures premium pricing given condition.")

    return df


# ── Velocity features ─────────────────────────────────────────────────────────

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    # Listings per seller per quarter (velocity)
    df["list_period"] = df["list_date"].dt.to_period("Q").astype(str)
    seller_qtr_vol = df.groupby(["seller_id", "list_period"])["listing_id"].transform("count")
    df["seller_listing_velocity"] = seller_qtr_vol
    doc("seller_listing_velocity",
        "Number of listings this seller posted in the same quarter. High = high-frequency seller.")

    # Category listing velocity in same period
    cat_qtr_vol = df.groupby(["category", "list_period"])["listing_id"].transform("count")
    df["category_velocity"] = cat_qtr_vol
    doc("category_velocity",
        "Number of listings in same category and quarter. Captures seasonal/cyclical supply surges.")

    return df


# ── Underpriced / Overpriced flag (target for classification) ─────────────────

def add_price_intelligence_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price intelligence label:
      - underpriced  (0): price_vs_cond_median < 0.80
      - fair_priced  (1): 0.80 <= price_vs_cond_median <= 1.20
      - overpriced   (2): price_vs_cond_median > 1.20
    """
    df["price_label"] = pd.cut(
        df["price_vs_cond_median"],
        bins=[-np.inf, 0.80, 1.20, np.inf],
        labels=["underpriced", "fair_priced", "overpriced"],
    ).astype(str)
    doc("price_label",
        "Price intelligence classification: underpriced/fair_priced/overpriced relative to category+condition peers.")

    label_enc = {"underpriced": 0, "fair_priced": 1, "overpriced": 2}
    df["price_label_enc"] = df["price_label"].map(label_enc)
    doc("price_label_enc", "Numeric encoding of price_label (0=under, 1=fair, 2=over).")

    return df


# ── Label encoding for categoricals ──────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["category", "make", "region", "condition", "seller_seller_type"]:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            doc(f"{col}_enc", f"Label-encoded version of '{col}' for model input.")
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("=== Feature Engineering Pipeline ===")

    clean_path = os.path.join(PROC_DIR, "listings_clean.csv")
    if not os.path.exists(clean_path):
        log.error("listings_clean.csv not found. Run preprocess.py first.")
        import sys; sys.exit(1)

    df = pd.read_csv(clean_path, parse_dates=["list_date"])
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} cols")

    log.info("Engineering features …")
    df = add_price_features(df)
    df = add_age_features(df)
    df = add_hours_features(df)
    df = add_engagement_features(df)
    df = add_seller_features(df)
    df = add_market_features(df)
    df = add_condition_features(df)
    df = add_velocity_features(df)
    df = add_price_intelligence_target(df)
    df = encode_categoricals(df)

    out_path = os.path.join(PROC_DIR, "listings_features.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Saved features → {out_path}  ({len(df):,} rows × {df.shape[1]} cols)")

    meta_path = os.path.join(PROC_DIR, "features_meta.json")
    with open(meta_path, "w") as f:
        json.dump(FEATURE_DOCS, f, indent=2)
    log.info(f"Saved feature docs → {meta_path}  ({len(FEATURE_DOCS)} features documented)")

    log.info(f"\nTotal features engineered: {len(FEATURE_DOCS)}")
    log.info("Feature engineering complete.")
    return df


if __name__ == "__main__":
    run_pipeline()
