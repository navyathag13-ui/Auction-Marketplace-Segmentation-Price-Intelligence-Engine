"""
Data Cleaning & Preprocessing Pipeline
---------------------------------------
Reads raw listings + sellers CSVs, performs:
  1. Type coercion
  2. Duplicate removal
  3. Missing value imputation
  4. Outlier capping
  5. Category/region normalization
  6. Merges into a single analytics-ready dataset

Output: data/processed/listings_clean.csv
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    listings_path = os.path.join(RAW_DIR, "listings.csv")
    sellers_path  = os.path.join(RAW_DIR, "sellers.csv")

    if not os.path.exists(listings_path):
        log.error("listings.csv not found. Run: python scripts/generate_data.py")
        sys.exit(1)

    listings = pd.read_csv(listings_path, parse_dates=["list_date"])
    sellers  = pd.read_csv(sellers_path)
    log.info(f"Loaded listings: {len(listings):,} rows | sellers: {len(sellers):,} rows")
    return listings, sellers


# ── Cleaning steps ───────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["listing_id"])
    removed = before - len(df)
    if removed:
        log.info(f"  Removed {removed} duplicate listing_ids")
    return df


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    df["year"]          = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["asking_price"]  = pd.to_numeric(df["asking_price"], errors="coerce")
    df["final_price"]   = pd.to_numeric(df["final_price"], errors="coerce")
    df["hours_used"]    = pd.to_numeric(df["hours_used"], errors="coerce")
    df["views"]         = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
    df["bids"]          = pd.to_numeric(df["bids"], errors="coerce").fillna(0).astype(int)
    df["inquiries"]     = pd.to_numeric(df["inquiries"], errors="coerce").fillna(0).astype(int)
    df["sold"]          = pd.to_numeric(df["sold"], errors="coerce").fillna(0).astype(int)
    df["days_on_market"]= pd.to_numeric(df["days_on_market"], errors="coerce")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Make: fill with category-level mode
    make_mode = df.groupby("category")["make"].transform(
        lambda s: s.fillna(s.mode()[0] if not s.mode().empty else "Unknown")
    )
    df["make"] = df["make"].fillna(make_mode)

    # Hours: fill with category-year median
    df["hours_used"] = df.groupby(["category", "year"])["hours_used"].transform(
        lambda s: s.fillna(s.median())
    )
    # Remaining: fill with category median
    df["hours_used"] = df.groupby("category")["hours_used"].transform(
        lambda s: s.fillna(s.median())
    )
    # Final fallback
    df["hours_used"] = df["hours_used"].fillna(df["hours_used"].median())

    # Asking price: should not be null; drop rows where it is
    before = len(df)
    df = df.dropna(subset=["asking_price"])
    if len(df) < before:
        log.warning(f"  Dropped {before - len(df)} rows with null asking_price")

    # Year: fill with category mode
    df["year"] = df.groupby("category")["year"].transform(
        lambda s: s.fillna(s.mode()[0] if not s.mode().empty else 2010)
    )

    log.info(f"  After imputation — nulls remaining:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Cap asking_price at 99.5th percentile per category
    def cap_group(s):
        upper = s.quantile(0.995)
        lower = s.quantile(0.005)
        return s.clip(lower=lower, upper=upper)

    df["asking_price"] = df.groupby("category")["asking_price"].transform(cap_group)

    # Cap hours_used at 30,000
    df["hours_used"] = df["hours_used"].clip(upper=30_000)

    # Cap days_on_market at 365
    df["days_on_market"] = df["days_on_market"].clip(upper=365)
    return df


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["category"]  = df["category"].str.strip().str.title()
    df["condition"] = df["condition"].str.strip().str.title()
    df["region"]    = df["region"].str.strip().str.title()
    df["make"]      = df["make"].str.strip().str.title()
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight derived columns useful in EDA."""
    df["age_years"]  = 2024 - df["year"].astype(float)
    df["price_band"] = pd.cut(
        df["asking_price"],
        bins=[0, 10_000, 30_000, 75_000, 150_000, 1_000_000],
        labels=["<$10K", "$10K–$30K", "$30K–$75K", "$75K–$150K", ">$150K"],
    )
    df["list_year"]  = df["list_date"].dt.year
    df["list_month"] = df["list_date"].dt.month
    df["list_qtr"]   = df["list_date"].dt.quarter
    return df


# ── Merge & save ─────────────────────────────────────────────────────────────

def merge_sellers(listings: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    # Prefix seller cols to avoid collision
    sellers = sellers.add_prefix("seller_").rename(columns={"seller_seller_id": "seller_id"})
    merged  = listings.merge(sellers, on="seller_id", how="left")
    missing = merged["seller_seller_type"].isnull().sum()
    if missing:
        log.warning(f"  {missing} listings have no matching seller record")
        merged["seller_seller_type"] = merged["seller_seller_type"].fillna("unknown")
        merged["seller_tenure_years"] = merged["seller_tenure_years"].fillna(merged["seller_tenure_years"].median())
        merged["seller_avg_rating"]   = merged["seller_avg_rating"].fillna(merged["seller_avg_rating"].median())
        merged["seller_verified"]     = merged["seller_verified"].fillna(False)
    return merged


def save(df: pd.DataFrame, filename: str):
    os.makedirs(PROC_DIR, exist_ok=True)
    path = os.path.join(PROC_DIR, filename)
    df.to_csv(path, index=False)
    log.info(f"Saved → {path}  ({len(df):,} rows × {df.shape[1]} cols)")


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("=== Preprocessing Pipeline ===")

    listings, sellers = load_raw()

    log.info("Step 1: Remove duplicates")
    listings = remove_duplicates(listings)

    log.info("Step 2: Fix types")
    listings = fix_types(listings)

    log.info("Step 3: Impute missing values")
    listings = impute_missing(listings)

    log.info("Step 4: Cap outliers")
    listings = cap_outliers(listings)

    log.info("Step 5: Normalize categoricals")
    listings = normalize_categoricals(listings)

    log.info("Step 6: Add derived columns")
    listings = add_derived_columns(listings)

    log.info("Step 7: Merge seller profiles")
    combined = merge_sellers(listings, sellers)

    log.info("Step 8: Save")
    save(combined, "listings_clean.csv")

    log.info("\n=== Summary ===")
    log.info(f"  Final dataset shape: {combined.shape}")
    log.info(f"  Sold rate: {combined['sold'].mean():.1%}")
    log.info(f"  Null count: {combined.isnull().sum().sum()}")
    log.info("Pipeline complete.")
    return combined


if __name__ == "__main__":
    run_pipeline()
