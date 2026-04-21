"""
Synthetic Heavy Equipment Auction Marketplace Dataset Generator

Why synthetic data:
  Real heavy-equipment auction datasets (IronPlanet, Ritchie Bros, Aucto) are
  not freely downloadable without commercial accounts.  We generate a statistically
  realistic dataset that mirrors the structure, distributions, and business logic
  of those platforms so every downstream analysis is meaningful.

  The dataset (~15,000 listings) contains:
    - Equipment listings with category, make, model, year, condition, hours
    - Seller profiles with tenure, listing volume, historical rating
    - Auction outcome: sold flag, final price, days-on-market
    - Region and engagement proxies (views, bids, inquiries)
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_LISTINGS = 15_000
N_SELLERS  = 800
RAW_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# ── Domain fixtures ──────────────────────────────────────────────────────────

CATEGORIES = {
    "Excavators":       {"base_price": 85_000, "price_sd": 40_000, "hours_mean": 4_200, "weight": 0.18},
    "Wheel Loaders":    {"base_price": 75_000, "price_sd": 35_000, "hours_mean": 3_800, "weight": 0.14},
    "Motor Graders":    {"base_price": 110_000,"price_sd": 45_000, "hours_mean": 5_000, "weight": 0.08},
    "Bulldozers":       {"base_price": 95_000, "price_sd": 42_000, "hours_mean": 4_500, "weight": 0.10},
    "Cranes":           {"base_price": 180_000,"price_sd": 80_000, "hours_mean": 3_200, "weight": 0.06},
    "Forklifts":        {"base_price": 22_000, "price_sd": 10_000, "hours_mean": 6_000, "weight": 0.12},
    "Compactors":       {"base_price": 38_000, "price_sd": 15_000, "hours_mean": 3_000, "weight": 0.07},
    "Skid Steers":      {"base_price": 28_000, "price_sd": 12_000, "hours_mean": 2_800, "weight": 0.12},
    "Backhoes":         {"base_price": 45_000, "price_sd": 18_000, "hours_mean": 3_500, "weight": 0.08},
    "Aerial Work Platforms": {"base_price": 35_000, "price_sd": 14_000, "hours_mean": 2_500, "weight": 0.05},
}

MAKES = ["Caterpillar", "Komatsu", "John Deere", "Volvo CE", "Hitachi",
         "Liebherr", "JCB", "Doosan", "Terex", "Case CE"]

CONDITIONS = ["Excellent", "Good", "Fair", "Poor"]
CONDITION_MULTIPLIER = {"Excellent": 1.20, "Good": 1.00, "Fair": 0.78, "Poor": 0.52}
CONDITION_WEIGHTS    = [0.15, 0.45, 0.30, 0.10]

REGIONS = {
    "Southeast": 0.18, "Midwest": 0.17, "Southwest": 0.14, "Northeast": 0.13,
    "West Coast": 0.12, "Mountain West": 0.09, "Great Plains": 0.08,
    "Mid-Atlantic": 0.05, "Pacific Northwest": 0.04,
}

# ── Seller profiles ──────────────────────────────────────────────────────────

def generate_sellers(n: int) -> pd.DataFrame:
    seller_types = np.random.choice(
        ["dealer", "fleet_operator", "contractor", "individual"],
        size=n, p=[0.30, 0.25, 0.25, 0.20]
    )
    return pd.DataFrame({
        "seller_id":      [f"S{i:04d}" for i in range(n)],
        "seller_type":    seller_types,
        "tenure_years":   np.clip(np.random.exponential(5, n), 0.5, 25).round(1),
        "total_listings": np.random.negative_binomial(5, 0.25, n) + 1,
        "avg_rating":     np.clip(np.random.normal(4.1, 0.6, n), 1, 5).round(2),
        "verified":       np.random.choice([True, False], size=n, p=[0.70, 0.30]),
    })


# ── Listing generation ───────────────────────────────────────────────────────

def _listing_price(category: str, year: int, condition: str, make: str) -> float:
    info = CATEGORIES[category]
    age  = 2024 - year
    # depreciation: ~8% per year, floored at 15% of base
    depreciation = max(0.15, 1.0 - 0.08 * age)
    brand_premium = 1.10 if make in ["Caterpillar", "Komatsu", "Liebherr"] else 1.0
    noise = np.random.lognormal(0, 0.12)
    price = (info["base_price"] * depreciation
             * CONDITION_MULTIPLIER[condition]
             * brand_premium
             * noise)
    return max(2_000, round(price / 100) * 100)


def generate_listings(sellers: pd.DataFrame, n: int) -> pd.DataFrame:
    cat_names    = list(CATEGORIES.keys())
    cat_weights  = [CATEGORIES[c]["weight"] for c in cat_names]
    categories   = np.random.choice(cat_names, size=n, p=cat_weights)

    years        = np.random.randint(1998, 2024, size=n)
    conditions   = np.random.choice(CONDITIONS, size=n, p=CONDITION_WEIGHTS)
    makes        = np.random.choice(MAKES, size=n)
    regions      = np.random.choice(list(REGIONS.keys()), size=n,
                                     p=list(REGIONS.values()))
    seller_ids   = sellers["seller_id"].sample(n, replace=True).values

    # Generate hours worked (with some missing ~8%)
    hours = []
    for cat, yr in zip(categories, years):
        h = np.random.normal(CATEGORIES[cat]["hours_mean"], 1200)
        h = max(50, h + (2024 - yr) * 80)  # older = more hours
        hours.append(round(h) if random.random() > 0.08 else np.nan)

    # List date: last 3 years
    base_date = datetime(2021, 1, 1)
    list_dates = [base_date + timedelta(days=random.randint(0, 1095)) for _ in range(n)]

    # Asking price
    asking_prices = [
        _listing_price(cat, yr, cond, mk)
        for cat, yr, cond, mk in zip(categories, years, conditions, makes)
    ]

    # Engagement signals
    views     = np.random.negative_binomial(8, 0.35, n) + 1
    bids      = np.array([
        max(0, int(np.random.poisson(v * 0.12))) for v in views
    ])
    inquiries = np.array([
        max(0, int(np.random.poisson(v * 0.08))) for v in views
    ])

    # Sale outcome: higher engagement + good condition = more likely to sell
    engagement_score = (views / views.max()) * 0.4 + (bids / (bids.max() + 1)) * 0.4
    cond_score = np.array([
        {"Excellent": 0.9, "Good": 0.75, "Fair": 0.55, "Poor": 0.35}[c]
        for c in conditions
    ])
    sell_prob = np.clip(engagement_score * 0.5 + cond_score * 0.5, 0.1, 0.95)
    sold      = np.random.binomial(1, sell_prob)

    # Days on market
    days_on_market = np.where(
        sold == 1,
        np.clip(np.random.exponential(25, n), 1, 180).astype(int),
        np.clip(np.random.exponential(75, n), 1, 365).astype(int),
    )

    # Final sale price (with negotiation discount)
    discount = np.where(
        conditions == "Excellent", np.random.uniform(0.95, 1.02, n),
        np.where(conditions == "Good", np.random.uniform(0.88, 0.97, n),
        np.where(conditions == "Fair", np.random.uniform(0.78, 0.90, n),
                 np.random.uniform(0.60, 0.82, n))))
    final_prices = np.where(sold == 1,
                            (np.array(asking_prices) * discount).round(-2),
                            np.nan)

    # Inject ~3% nulls in non-critical columns to make cleaning realistic
    def inject_nulls(arr, rate=0.03):
        mask = np.random.random(len(arr)) < rate
        arr  = arr.astype(object)
        arr[mask] = np.nan
        return arr

    df = pd.DataFrame({
        "listing_id":     [f"L{i:06d}" for i in range(n)],
        "seller_id":      seller_ids,
        "category":       categories,
        "make":           inject_nulls(makes),
        "year":           years,
        "condition":      conditions,
        "hours_used":     hours,
        "asking_price":   asking_prices,
        "region":         regions,
        "list_date":      list_dates,
        "views":          views,
        "bids":           bids,
        "inquiries":      inject_nulls(inquiries),
        "sold":           sold,
        "days_on_market": days_on_market,
        "final_price":    final_prices,
    })

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Generating seller profiles …")
    sellers = generate_sellers(N_SELLERS)
    sellers_path = os.path.join(RAW_DIR, "sellers.csv")
    sellers.to_csv(sellers_path, index=False)
    print(f"  → {sellers_path}  ({len(sellers):,} rows)")

    print("Generating auction listings …")
    listings = generate_listings(sellers, N_LISTINGS)
    listings_path = os.path.join(RAW_DIR, "listings.csv")
    listings.to_csv(listings_path, index=False)
    print(f"  → {listings_path}  ({len(listings):,} rows)")

    print("\nDataset summary:")
    print(f"  Listings : {len(listings):,}")
    print(f"  Sellers  : {len(sellers):,}")
    print(f"  Sold rate: {listings['sold'].mean():.1%}")
    print(f"  Categories: {listings['category'].nunique()}")
    print(f"  Regions   : {listings['region'].nunique()}")
    print(f"  Avg asking price: ${listings['asking_price'].mean():,.0f}")
    print(f"  Price range: ${listings['asking_price'].min():,.0f} – ${listings['asking_price'].max():,.0f}")


if __name__ == "__main__":
    main()
