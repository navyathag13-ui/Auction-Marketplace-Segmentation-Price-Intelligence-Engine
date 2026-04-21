"""
Pipeline unit tests
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")


# ── Data generation tests ─────────────────────────────────────────────────────

class TestDataGeneration:
    def test_raw_files_exist(self):
        assert os.path.exists(os.path.join(RAW_DIR, "listings.csv")), \
            "listings.csv not found. Run: python scripts/generate_data.py"
        assert os.path.exists(os.path.join(RAW_DIR, "sellers.csv")), \
            "sellers.csv not found. Run: python scripts/generate_data.py"

    def test_listings_shape(self):
        df = pd.read_csv(os.path.join(RAW_DIR, "listings.csv"))
        assert len(df) >= 10_000, f"Expected >= 10,000 listings, got {len(df)}"
        assert len(df.columns) >= 14, f"Expected >= 14 columns, got {len(df.columns)}"

    def test_sellers_shape(self):
        df = pd.read_csv(os.path.join(RAW_DIR, "sellers.csv"))
        assert len(df) >= 500, f"Expected >= 500 sellers, got {len(df)}"

    def test_listings_required_columns(self):
        df = pd.read_csv(os.path.join(RAW_DIR, "listings.csv"))
        required = ["listing_id", "seller_id", "category", "asking_price", "sold"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_listing_ids_unique(self):
        df = pd.read_csv(os.path.join(RAW_DIR, "listings.csv"))
        assert df["listing_id"].nunique() == len(df), "Duplicate listing_ids found"

    def test_asking_price_positive(self):
        df = pd.read_csv(os.path.join(RAW_DIR, "listings.csv"))
        df["asking_price"] = pd.to_numeric(df["asking_price"], errors="coerce")
        assert (df["asking_price"].dropna() > 0).all(), "Found non-positive asking prices"

    def test_sold_binary(self):
        df = pd.read_csv(os.path.join(RAW_DIR, "listings.csv"))
        sold_vals = df["sold"].dropna().unique()
        assert set(sold_vals).issubset({0, 1, 0.0, 1.0}), f"Unexpected sold values: {sold_vals}"


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    @pytest.fixture(autouse=True)
    def require_clean_data(self):
        path = os.path.join(PROC_DIR, "listings_clean.csv")
        if not os.path.exists(path):
            pytest.skip("listings_clean.csv not found. Run preprocess.py first.")

    def test_clean_file_exists(self):
        assert os.path.exists(os.path.join(PROC_DIR, "listings_clean.csv"))

    def test_no_duplicate_listing_ids(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_clean.csv"))
        assert df["listing_id"].nunique() == len(df)

    def test_asking_price_no_nulls(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_clean.csv"))
        assert df["asking_price"].isnull().sum() == 0

    def test_age_years_present(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_clean.csv"))
        assert "age_years" in df.columns

    def test_sold_col_binary(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_clean.csv"))
        assert set(df["sold"].unique()).issubset({0, 1})


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    @pytest.fixture(autouse=True)
    def require_features(self):
        path = os.path.join(PROC_DIR, "listings_features.csv")
        if not os.path.exists(path):
            pytest.skip("listings_features.csv not found. Run feature_engineering.py first.")

    def test_features_file_exists(self):
        assert os.path.exists(os.path.join(PROC_DIR, "listings_features.csv"))

    def test_engineered_columns_present(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_features.csv"))
        required_features = [
            "price_vs_cat_median", "price_vs_cond_median",
            "log_asking_price", "total_engagement", "bid_rate",
            "condition_score", "seller_sold_rate", "price_label",
        ]
        for feat in required_features:
            assert feat in df.columns, f"Missing engineered feature: {feat}"

    def test_price_label_valid(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_features.csv"))
        valid = {"underpriced", "fair_priced", "overpriced"}
        actual = set(df["price_label"].unique())
        assert actual.issubset(valid), f"Invalid price labels: {actual - valid}"

    def test_log_price_non_negative(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_features.csv"))
        assert (df["log_asking_price"] >= 0).all()

    def test_condition_score_range(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_features.csv"))
        assert df["condition_score"].between(0, 3).all()

    def test_no_inf_values(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_features.csv"))
        num_cols = df.select_dtypes(include=[np.number]).columns
        assert not np.isinf(df[num_cols].values).any(), "Infinite values found in features"

    def test_features_meta_exists(self):
        assert os.path.exists(os.path.join(PROC_DIR, "features_meta.json"))


# ── Segmentation tests ────────────────────────────────────────────────────────

class TestSegmentation:
    @pytest.fixture(autouse=True)
    def require_segmented(self):
        df_path = os.path.join(PROC_DIR, "listings_features.csv")
        if not os.path.exists(df_path):
            pytest.skip("listings_features.csv not found.")
        df = pd.read_csv(df_path)
        if "segment_kmeans" not in df.columns:
            pytest.skip("segment_kmeans not found. Run train_segmentation.py first.")

    def test_segment_count(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_features.csv"))
        n_segs = df["segment_kmeans"].nunique()
        assert n_segs == 5, f"Expected 5 segments, got {n_segs}"

    def test_all_listings_assigned(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_features.csv"))
        null_count = df["segment_kmeans"].isnull().sum()
        assert null_count == 0, f"{null_count} listings without segment assignment"

    def test_segment_labels_present(self):
        df = pd.read_csv(os.path.join(PROC_DIR, "listings_features.csv"))
        assert "segment_label" in df.columns
        assert df["segment_label"].isnull().sum() == 0

    def test_cluster_profiles_exist(self):
        assert os.path.exists(os.path.join(PROC_DIR, "cluster_profiles.csv"))


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModel:
    @pytest.fixture(autouse=True)
    def require_model(self):
        models_dir = os.path.join(BASE_DIR, "models")
        if not os.path.exists(os.path.join(models_dir, "lgbm_price_classifier.joblib")):
            pytest.skip("LightGBM model not found. Run train_model.py first.")

    def test_model_file_exists(self):
        models_dir = os.path.join(BASE_DIR, "models")
        assert os.path.exists(os.path.join(models_dir, "lgbm_price_classifier.joblib"))

    def test_model_features_file_exists(self):
        models_dir = os.path.join(BASE_DIR, "models")
        assert os.path.exists(os.path.join(models_dir, "model_features.json"))

    def test_model_results_exist(self):
        assert os.path.exists(os.path.join(PROC_DIR, "model_results.json"))

    def test_model_accuracy_above_threshold(self):
        import json
        results_path = os.path.join(PROC_DIR, "model_results.json")
        with open(results_path) as f:
            results = json.load(f)
        lgbm_acc = results["lightgbm"]["accuracy"]
        assert lgbm_acc >= 0.55, f"LightGBM accuracy too low: {lgbm_acc}"

    def test_model_loads(self):
        import joblib
        models_dir = os.path.join(BASE_DIR, "models")
        model = joblib.load(os.path.join(models_dir, "lgbm_price_classifier.joblib"))
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
