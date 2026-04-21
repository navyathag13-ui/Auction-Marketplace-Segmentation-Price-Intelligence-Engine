"""
Predictive Modeling Pipeline
------------------------------
Task: Price Intelligence Classification
  Predict whether a listing is:
    0 = underpriced  (price_vs_cond_median < 0.80)
    1 = fair_priced  (0.80 – 1.20)
    2 = overpriced   (> 1.20)

Models:
  1. Logistic Regression (baseline)
  2. LightGBM Classifier (primary)
  3. XGBoost Classifier (comparison)

Evaluation:
  Accuracy, Precision, Recall, F1 (macro), ROC-AUC (OvR)
  Feature importance plot
  Confusion matrix

Outputs:
  models/lgbm_price_classifier.joblib
  models/model_scaler.joblib
  data/processed/model_results.json
  reports/figures/feature_importance.png
  reports/figures/confusion_matrix.png
  reports/figures/roc_curves.png
"""

import os
import json
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
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

MODEL_FEATURES = [
    "log_asking_price",
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
    "seller_verified_flag",
    "seller_tenure_years",
    "seller_avg_rating",
    "listing_age_days",
    "seller_listing_velocity",
    "category_velocity",
    "category_enc",
    "make_enc",
    "region_enc",
]

TARGET = "price_label_enc"
CLASS_NAMES = ["underpriced", "fair_priced", "overpriced"]


def load_data() -> pd.DataFrame:
    path = os.path.join(PROC_DIR, "listings_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("listings_features.csv not found.")
    df = pd.read_csv(path, parse_dates=["list_date"])
    log.info(f"Loaded {len(df):,} rows")
    return df


def prepare_data(df: pd.DataFrame):
    available = [f for f in MODEL_FEATURES if f in df.columns]
    missing   = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        log.warning(f"Missing features (will skip): {missing}")

    X = df[available].copy().fillna(df[available].median())
    y = df[TARGET].copy().fillna(1).astype(int)

    log.info(f"Target distribution:\n{pd.Series(y).map({0:'underpriced',1:'fair_priced',2:'overpriced'}).value_counts()}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    log.info(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test, available


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc   = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":  round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "roc_auc":   round(roc, 4),
    }
    log.info(f"\n{model_name} Test Metrics:")
    for k, v in metrics.items():
        if k != "model":
            log.info(f"  {k}: {v}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, filename: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved confusion matrix → {path}")


def plot_feature_importance(model, feature_names: list, n_top: int = 20):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps"):
        importances = model.named_steps["model"].feature_importances_
    else:
        return

    fi = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(n_top)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))
    fi.plot.barh(ax=ax, color=colors)
    ax.set_title(f"Top {n_top} Feature Importances (LightGBM)")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved feature importance → {path}")


def plot_predicted_vs_actual(y_true, y_pred_proba, filename="predicted_vs_actual.png"):
    """Plot predicted probability for each class vs actual label (reliability diagram style)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (ax, cls) in enumerate(zip(axes, CLASS_NAMES)):
        actual_bin = (y_true == i).astype(int)
        proba_cls  = y_pred_proba[:, i]
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means   = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (proba_cls >= lo) & (proba_cls < hi)
            bin_means.append(actual_bin[mask].mean() if mask.sum() > 0 else 0)
        ax.bar(bin_centers, bin_means, width=0.09, alpha=0.7, color="steelblue")
        ax.plot([0, 1], [0, 1], "r--", alpha=0.7)
        ax.set_title(f"Reliability: {cls}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Fraction Positive")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved predicted vs actual → {path}")


def train_logistic(X_train, y_train, X_val, y_val, features):
    log.info("\n─── Logistic Regression (Baseline) ───")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000, random_state=42,
                                       class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, pipe.predict(X_val))
    log.info(f"  Val accuracy: {val_acc:.4f}")
    return pipe


def train_lgbm(X_train, y_train, X_val, y_val):
    log.info("\n─── LightGBM Classifier ───")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    val_acc = accuracy_score(y_val, model.predict(X_val))
    log.info(f"  Val accuracy: {val_acc:.4f}  Best iteration: {model.best_iteration_}")
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    log.info("\n─── XGBoost Classifier ───")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_acc = accuracy_score(y_val, model.predict(X_val))
    log.info(f"  Val accuracy: {val_acc:.4f}")
    return model


def run_pipeline():
    log.info("=== Predictive Modeling Pipeline ===")
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, features = prepare_data(df)

    # 1. Baseline
    lr_model = train_logistic(X_train, y_train, X_val, y_val, features)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    plot_confusion_matrix(y_test, lr_model.predict(X_test),
                          "Logistic Regression", "cm_logistic.png")

    # 2. LightGBM
    lgbm_model = train_lgbm(X_train, y_train, X_val, y_val)
    lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
    plot_confusion_matrix(y_test, lgbm_model.predict(X_test),
                          "LightGBM", "confusion_matrix.png")
    plot_feature_importance(lgbm_model, features)
    plot_predicted_vs_actual(np.array(y_test), lgbm_model.predict_proba(X_test))

    # 3. XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # 4. Save best model (LightGBM)
    joblib.dump(lgbm_model, os.path.join(MODELS_DIR, "lgbm_price_classifier.joblib"))
    log.info(f"Saved LightGBM model → {os.path.join(MODELS_DIR, 'lgbm_price_classifier.joblib')}")

    # 5. Save feature list for API
    with open(os.path.join(MODELS_DIR, "model_features.json"), "w") as f:
        json.dump(features, f, indent=2)

    # 6. Save all metrics
    all_metrics = {
        "logistic_regression": lr_metrics,
        "lightgbm": lgbm_metrics,
        "xgboost": xgb_metrics,
        "best_model": "lightgbm",
        "feature_count": len(features),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    results_path = os.path.join(PROC_DIR, "model_results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Saved model results → {results_path}")

    # 7. Print comparison
    log.info("\n=== Model Comparison ===")
    for m in [lr_metrics, lgbm_metrics, xgb_metrics]:
        log.info(f"  {m['model']:<25} acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  auc={m['roc_auc']:.4f}")

    log.info("\nPipeline complete.")
    return lgbm_model, features


if __name__ == "__main__":
    run_pipeline()
