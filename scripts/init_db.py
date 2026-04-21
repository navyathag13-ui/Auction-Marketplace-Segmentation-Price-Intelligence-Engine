"""
Database Initialization Script
---------------------------------
Creates the SQLite database, all tables, and loads:
  1. Sellers
  2. Listings (cleaned)
  3. Engineered features
  4. Segment assignments
  5. Model predictions
  6. Segment summary table
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure app module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import init_db, SessionLocal
from app.db.models import (
    Seller, Listing, ListingFeatures, ListingSegment,
    ListingPrediction, SegmentSummary
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")


def load_csvs():
    features_path = os.path.join(PROC_DIR, "listings_features.csv")
    sellers_path  = os.path.join(RAW_DIR,  "sellers.csv")

    if not os.path.exists(features_path):
        log.error("listings_features.csv not found. Run the full pipeline first.")
        sys.exit(1)

    df       = pd.read_csv(features_path, parse_dates=["list_date"])
    sellers  = pd.read_csv(sellers_path)
    log.info(f"Loaded {len(df):,} listings, {len(sellers):,} sellers")
    return df, sellers


def insert_sellers(session, sellers: pd.DataFrame):
    log.info("Inserting sellers …")
    objects = []
    for _, row in sellers.iterrows():
        objects.append(Seller(
            seller_id      = str(row["seller_id"]),
            seller_type    = str(row["seller_type"]),
            tenure_years   = float(row["tenure_years"]) if pd.notna(row["tenure_years"]) else None,
            total_listings = int(row["total_listings"]) if pd.notna(row["total_listings"]) else None,
            avg_rating     = float(row["avg_rating"]) if pd.notna(row["avg_rating"]) else None,
            verified       = bool(row["verified"]),
        ))
    session.bulk_save_objects(objects)
    session.commit()
    log.info(f"  Inserted {len(objects):,} sellers")


def insert_listings(session, df: pd.DataFrame):
    log.info("Inserting listings …")
    objects = []
    for _, row in df.iterrows():
        objects.append(Listing(
            listing_id     = str(row["listing_id"]),
            seller_id      = str(row["seller_id"]),
            category       = str(row["category"]),
            make           = str(row["make"]) if pd.notna(row.get("make")) else None,
            year           = int(row["year"]) if pd.notna(row.get("year")) else None,
            condition      = str(row["condition"]) if pd.notna(row.get("condition")) else None,
            hours_used     = float(row["hours_used"]) if pd.notna(row.get("hours_used")) else None,
            asking_price   = float(row["asking_price"]),
            region         = str(row["region"]) if pd.notna(row.get("region")) else None,
            list_date      = row["list_date"] if pd.notna(row.get("list_date")) else None,
            views          = int(row["views"]) if pd.notna(row.get("views")) else 0,
            bids           = int(row["bids"]) if pd.notna(row.get("bids")) else 0,
            inquiries      = int(row["inquiries"]) if pd.notna(row.get("inquiries")) else 0,
            sold           = bool(row["sold"]) if pd.notna(row.get("sold")) else False,
            days_on_market = int(row["days_on_market"]) if pd.notna(row.get("days_on_market")) else None,
            final_price    = float(row["final_price"]) if pd.notna(row.get("final_price")) else None,
            age_years      = float(row["age_years"]) if pd.notna(row.get("age_years")) else None,
            price_band     = str(row["price_band"]) if pd.notna(row.get("price_band")) else None,
            list_year      = int(row["list_year"]) if pd.notna(row.get("list_year")) else None,
            list_month     = int(row["list_month"]) if pd.notna(row.get("list_month")) else None,
        ))
    session.bulk_save_objects(objects)
    session.commit()
    log.info(f"  Inserted {len(objects):,} listings")


def insert_features(session, df: pd.DataFrame):
    log.info("Inserting listing features …")
    objects = []
    for _, row in df.iterrows():
        objects.append(ListingFeatures(
            listing_id           = str(row["listing_id"]),
            price_vs_cat_median  = float(row["price_vs_cat_median"]) if pd.notna(row.get("price_vs_cat_median")) else None,
            price_vs_cond_median = float(row["price_vs_cond_median"]) if pd.notna(row.get("price_vs_cond_median")) else None,
            price_zscore_cat     = float(row["price_zscore_cat"]) if pd.notna(row.get("price_zscore_cat")) else None,
            log_asking_price     = float(row["log_asking_price"]) if pd.notna(row.get("log_asking_price")) else None,
            hours_per_year       = float(row["hours_per_year"]) if pd.notna(row.get("hours_per_year")) else None,
            hours_vs_cat_median  = float(row["hours_vs_cat_median"]) if pd.notna(row.get("hours_vs_cat_median")) else None,
            total_engagement     = int(row["total_engagement"]) if pd.notna(row.get("total_engagement")) else None,
            bid_rate             = float(row["bid_rate"]) if pd.notna(row.get("bid_rate")) else None,
            inquiry_rate         = float(row["inquiry_rate"]) if pd.notna(row.get("inquiry_rate")) else None,
            condition_score      = int(row["condition_score"]) if pd.notna(row.get("condition_score")) else None,
            seller_listing_count = int(row["seller_listing_count"]) if pd.notna(row.get("seller_listing_count")) else None,
            seller_sold_rate     = float(row["seller_sold_rate"]) if pd.notna(row.get("seller_sold_rate")) else None,
            category_freq        = float(row["category_freq"]) if pd.notna(row.get("category_freq")) else None,
            region_demand_proxy  = float(row["region_demand_proxy"]) if pd.notna(row.get("region_demand_proxy")) else None,
            price_label          = str(row["price_label"]) if pd.notna(row.get("price_label")) else None,
        ))
    session.bulk_save_objects(objects)
    session.commit()
    log.info(f"  Inserted {len(objects):,} feature rows")


def insert_segments(session, df: pd.DataFrame):
    log.info("Inserting segment assignments …")
    if "segment_kmeans" not in df.columns:
        log.warning("  No segment_kmeans column found — skipping segments")
        return
    objects = []
    for _, row in df.iterrows():
        objects.append(ListingSegment(
            listing_id     = str(row["listing_id"]),
            segment_kmeans = int(row["segment_kmeans"]) if pd.notna(row.get("segment_kmeans")) else None,
            segment_label  = str(row["segment_label"])  if pd.notna(row.get("segment_label"))  else None,
            assigned_at    = datetime.utcnow(),
        ))
    session.bulk_save_objects(objects)
    session.commit()
    log.info(f"  Inserted {len(objects):,} segment assignments")


def insert_predictions(session, df: pd.DataFrame):
    log.info("Inserting model predictions …")
    if "price_label" not in df.columns:
        log.warning("  No price_label column — skipping predictions")
        return
    objects = []
    label_enc = {"underpriced": 0, "fair_priced": 1, "overpriced": 2}
    for _, row in df.iterrows():
        label     = str(row.get("price_label", "fair_priced"))
        label_int = label_enc.get(label, 1)
        # Approximate probabilities from label (actual model probabilities would come from inference)
        probs = {0: 0.1, 1: 0.1, 2: 0.1}
        probs[label_int] = 0.75
        total = sum(probs.values())
        probs = {k: round(v / total, 4) for k, v in probs.items()}
        price_score = round(probs[1] * 100, 1)  # probability of fair_priced as score

        objects.append(ListingPrediction(
            listing_id           = str(row["listing_id"]),
            price_label_pred     = label,
            price_label_enc_pred = label_int,
            prob_underpriced     = probs[0],
            prob_fair_priced     = probs[1],
            prob_overpriced      = probs[2],
            price_score          = price_score,
            predicted_at         = datetime.utcnow(),
        ))
    session.bulk_save_objects(objects)
    session.commit()
    log.info(f"  Inserted {len(objects):,} predictions")


def insert_segment_summaries(session, df: pd.DataFrame):
    log.info("Building segment summary table …")
    if "segment_kmeans" not in df.columns:
        return

    seg_names = {
        0: "Premium High-Demand",
        1: "Aging Low-Engagement",
        2: "Value Fleet Inventory",
        3: "Hot Fast-Moving Deals",
        4: "Niche Specialty Equipment",
    }

    for seg_id, group in df.groupby("segment_kmeans"):
        summary = SegmentSummary(
            segment_id          = int(seg_id),
            segment_name        = seg_names.get(int(seg_id), f"Segment {seg_id}"),
            listing_count       = len(group),
            avg_asking_price    = round(group["asking_price"].mean(), 2),
            median_asking_price = round(group["asking_price"].median(), 2),
            avg_age_years       = round(group["age_years"].mean(), 2) if "age_years" in group else None,
            avg_condition_score = round(group["condition_score"].mean(), 3) if "condition_score" in group else None,
            avg_engagement      = round(group["total_engagement"].mean(), 2) if "total_engagement" in group else None,
            avg_sold_rate       = round(group["sold"].mean(), 4) if "sold" in group else None,
            avg_days_on_market  = round(group["days_on_market"].mean(), 2) if "days_on_market" in group else None,
            updated_at          = datetime.utcnow(),
        )
        session.add(summary)
    session.commit()
    log.info(f"  Inserted {df['segment_kmeans'].nunique()} segment summaries")


def main():
    log.info("=== Database Initialization ===")

    log.info("Creating tables …")
    init_db()

    df, sellers = load_csvs()

    session = SessionLocal()
    try:
        # Check if already populated
        if session.query(Seller).count() > 0:
            log.warning("Database already populated. Drop the .db file to reinitialize.")
            return

        insert_sellers(session, sellers)
        insert_listings(session, df)
        insert_features(session, df)
        insert_segments(session, df)
        insert_predictions(session, df)
        insert_segment_summaries(session, df)

    except Exception as e:
        session.rollback()
        log.error(f"Error during DB initialization: {e}")
        raise
    finally:
        session.close()

    log.info("\n=== Database ready ===")
    db_path = os.path.join(BASE_DIR, "data", "auction_marketplace.db")
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / 1_048_576
        log.info(f"  DB path: {db_path}")
        log.info(f"  DB size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
