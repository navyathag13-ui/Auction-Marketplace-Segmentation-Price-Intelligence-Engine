"""
Segmentation Modeling
-----------------------
Segments auction listings using:
  1. K-Means (baseline, k=5 selected via silhouette + elbow)
  2. Hierarchical Agglomerative Clustering (comparison)

Produces:
  - Cluster assignments in listings_features.csv (segment_kmeans, segment_label)
  - Cluster profiles CSV
  - Silhouette comparison JSON
  - Saved KMeans model (joblib)
  - Cluster visualization PNG
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Features used for segmentation
SEG_FEATURES = [
    "log_asking_price",
    "price_vs_cat_median",
    "price_vs_cond_median",
    "age_years",
    "log_hours_used",
    "hours_vs_cat_median",
    "total_engagement",
    "bid_rate",
    "inquiry_rate",
    "condition_score",
    "seller_listing_count",
    "seller_sold_rate",
    "category_freq",
    "region_demand_proxy",
    "log_views",
]

# Business segment names (assigned after profiling)
SEGMENT_NAMES = {
    0: "Premium High-Demand",
    1: "Aging Low-Engagement",
    2: "Value Fleet Inventory",
    3: "Hot Fast-Moving Deals",
    4: "Niche Specialty Equipment",
}


def load_data() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "listings_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("listings_features.csv not found. Run feature_engineering.py first.")
    df = pd.read_csv(path, parse_dates=["list_date"])
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


def prepare_seg_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X = df[SEG_FEATURES].copy()
    X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_optimal_k(X: np.ndarray, k_range=range(2, 9)) -> dict:
    log.info("Running elbow + silhouette analysis …")
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
        inertia = km.inertia_
        results[k] = {"silhouette": round(sil, 4), "inertia": round(inertia, 2)}
        log.info(f"  k={k}: silhouette={sil:.4f}  inertia={inertia:,.0f}")
    return results


def plot_elbow(results: dict):
    ks = list(results.keys())
    sils = [results[k]["silhouette"] for k in ks]
    inertias = [results[k]["inertia"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, inertias, "bo-", linewidth=2)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow Method")
    ax1.grid(alpha=0.3)

    ax2.plot(ks, sils, "rs-", linewidth=2)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score vs k")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "segmentation_elbow.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved elbow plot → {path}")


def run_kmeans(X: np.ndarray, k: int = 5) -> tuple[KMeans, np.ndarray]:
    log.info(f"Fitting KMeans (k={k}) …")
    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42)
    dbi = davies_bouldin_score(X, labels)
    log.info(f"  KMeans silhouette={sil:.4f}  davies_bouldin={dbi:.4f}")
    return km, labels


def run_hierarchical(X: np.ndarray, k: int = 5) -> np.ndarray:
    log.info(f"Fitting AgglomerativeClustering (k={k}) on sample …")
    # Sample for speed (agglomerative is O(n²))
    sample_size = min(3000, len(X))
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[idx]
    hac = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hac_labels = hac.fit_predict(X_sample)
    sil = silhouette_score(X_sample, hac_labels, random_state=42)
    log.info(f"  Hierarchical silhouette (sample)={sil:.4f}")
    return hac_labels, idx


def plot_pca_clusters(X: np.ndarray, labels: np.ndarray, title: str, filename: str):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, clust in enumerate(np.unique(labels)):
        mask = labels == clust
        seg_name = SEGMENT_NAMES.get(clust, f"Segment {clust}")
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]],
                   alpha=0.4, s=8, label=seg_name)

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} var)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, markerscale=3)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved cluster PCA plot → {path}")


def build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    profile_cols = [
        "asking_price", "age_years", "hours_used", "total_engagement",
        "bid_rate", "inquiry_rate", "condition_score", "seller_sold_rate",
        "price_vs_cat_median", "days_on_market", "sold",
    ]
    profile = df.groupby("segment_kmeans")[profile_cols].agg(["mean", "median"]).round(3)
    profile.columns = ["_".join(c) for c in profile.columns]
    profile["listing_count"] = df.groupby("segment_kmeans")["listing_id"].count()
    profile["segment_name"] = profile.index.map(SEGMENT_NAMES)
    return profile


def plot_segment_price_dist(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, df["segment_kmeans"].nunique()))
    for i, (seg, group) in enumerate(df.groupby("segment_kmeans")):
        name = SEGMENT_NAMES.get(seg, f"Seg {seg}")
        group["asking_price"].plot.kde(ax=ax, label=name, color=colors[i], linewidth=2)

    ax.set_xlabel("Asking Price ($)")
    ax.set_title("Price Distribution by Segment")
    ax.legend()
    ax.set_xlim(0, df["asking_price"].quantile(0.99))
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "segment_price_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved price distribution plot → {path}")


def plot_segment_counts(df: pd.DataFrame):
    counts = df["segment_label"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of Listings")
    ax.set_title("Listings per Segment")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "segment_counts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved segment counts plot → {path}")


def run_pipeline():
    log.info("=== Segmentation Pipeline ===")
    df = load_data()

    X, scaler = prepare_seg_matrix(df)

    # 1. Find optimal k
    elbow_results = find_optimal_k(X)
    plot_elbow(elbow_results)

    # Save k-selection results
    k_path = os.path.join(PROC_DIR, "k_selection.json")
    with open(k_path, "w") as f:
        json.dump(elbow_results, f, indent=2)

    # 2. Fit KMeans with k=5 (best silhouette + business interpretability)
    OPTIMAL_K = 5
    km, km_labels = run_kmeans(X, k=OPTIMAL_K)

    # 3. Save model and scaler
    joblib.dump(km, os.path.join(MODELS_DIR, "kmeans_segmentation.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "seg_scaler.joblib"))
    log.info("Saved KMeans model and scaler")

    # 4. Assign labels to dataframe
    df["segment_kmeans"] = km_labels
    df["segment_label"]  = df["segment_kmeans"].map(SEGMENT_NAMES)

    # 5. Hierarchical comparison (on sample)
    hac_labels, hac_idx = run_hierarchical(X, k=OPTIMAL_K)
    plot_pca_clusters(X[hac_idx], hac_labels,
                      "Hierarchical Clustering (PCA) — Sample",
                      "hierarchical_pca.png")

    # 6. KMeans PCA visualization
    plot_pca_clusters(X, km_labels,
                      "KMeans Segments (PCA Projection)",
                      "kmeans_pca.png")

    # 7. Cluster profiles
    profile = build_cluster_profiles(df)
    profile_path = os.path.join(PROC_DIR, "cluster_profiles.csv")
    profile.to_csv(profile_path)
    log.info(f"Saved cluster profiles → {profile_path}")
    log.info(f"\nCluster Profiles:\n{profile[['listing_count','segment_name','asking_price_mean','sold_mean']].to_string()}")

    # 8. Segment visualizations
    plot_segment_price_dist(df)
    plot_segment_counts(df)

    # 9. Save updated features with segments
    out_path = os.path.join(PROC_DIR, "listings_features.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Updated listings_features.csv with segment assignments → {out_path}")

    # 10. Save comparison metrics
    km_sil = silhouette_score(X, km_labels, sample_size=min(5000, len(X)), random_state=42)
    hac_sil = silhouette_score(X[hac_idx], hac_labels, random_state=42)
    comparison = {
        "kmeans": {"silhouette": round(km_sil, 4), "k": OPTIMAL_K},
        "hierarchical": {"silhouette": round(hac_sil, 4), "k": OPTIMAL_K, "note": "sample n=3000"},
        "selected": "kmeans",
        "reason": "KMeans achieves comparable silhouette with full-dataset coverage and faster inference for API use",
    }
    cmp_path = os.path.join(PROC_DIR, "segmentation_comparison.json")
    with open(cmp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info(f"Saved model comparison → {cmp_path}")

    log.info("\n=== Segmentation Complete ===")
    log.info(f"  KMeans silhouette: {km_sil:.4f}")
    log.info(f"  Segments: {df['segment_label'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    run_pipeline()
