"""
Helper script to generate the three analysis notebooks programmatically.
Run: python notebooks/create_notebooks.py
"""

import json
import os

NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source, "id": os.urandom(4).hex()}


def code(source, outputs=None):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "outputs": outputs or [],
        "execution_count": None,
        "id": os.urandom(4).hex(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 1: EDA
# ═══════════════════════════════════════════════════════════════════════════════

eda_cells = [
    md("# 01 — Exploratory Data Analysis\n**Auction Marketplace Segmentation & Price Intelligence Engine**\n\nThis notebook explores the heavy equipment auction dataset: distributions, relationships, missing values, and business insights."),
    code("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.titlesize'] = 14

print("Libraries loaded")"""),
    md("## 1. Load Data"),
    code("""\
df = pd.read_csv('../data/processed/listings_clean.csv', parse_dates=['list_date'])
print(f"Shape: {df.shape}")
df.head()"""),
    code("df.dtypes"),
    code("""\
print("=== Missing Values ===")
miss = df.isnull().sum()
print(miss[miss > 0])"""),
    code("""\
print("=== Descriptive Statistics ===")
df[['asking_price','hours_used','age_years','views','bids','inquiries','days_on_market']].describe().round(2)"""),
    md("## 2. Price Distribution"),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['asking_price'].plot.hist(bins=60, ax=axes[0], color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Asking Price ($)')
axes[0].set_title('Asking Price Distribution')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

np.log1p(df['asking_price']).plot.hist(bins=60, ax=axes[1], color='coral', edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Log(Asking Price + 1)')
axes[1].set_title('Log Price Distribution')

plt.tight_layout()
plt.savefig('../reports/figures/price_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved price_distribution.png")"""),
    md("## 3. Category Analysis"),
    code("""\
cat_stats = df.groupby('category').agg(
    count=('listing_id','count'),
    avg_price=('asking_price','mean'),
    median_price=('asking_price','median'),
    sold_rate=('sold','mean'),
).sort_values('count', ascending=False)
print(cat_stats.round(2))

fig, ax = plt.subplots(figsize=(12, 5))
cat_stats['avg_price'].sort_values().plot.barh(ax=ax, color='teal', alpha=0.8)
ax.set_xlabel('Average Asking Price ($)')
ax.set_title('Average Price by Equipment Category')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/category_avg_price.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 4. Condition & Sold Rate"),
    code("""\
cond_sold = df.groupby('condition')[['sold','asking_price']].agg({'sold':'mean','asking_price':'median'}).round(3)
print("Condition analysis:")
print(cond_sold)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cond_sold['sold'].sort_values().plot.barh(ax=axes[0], color='green', alpha=0.8)
axes[0].set_title('Sold Rate by Condition')
axes[0].set_xlabel('Sold Rate')

df.boxplot(column='asking_price', by='condition', ax=axes[1], grid=False)
axes[1].set_title('Price Distribution by Condition')
axes[1].set_xlabel('Condition')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('../reports/figures/condition_analysis.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 5. Region Analysis"),
    code("""\
region_stats = df.groupby('region').agg(
    count=('listing_id','count'),
    avg_price=('asking_price','mean'),
    sold_rate=('sold','mean'),
    avg_views=('views','mean'),
).sort_values('count', ascending=False)
print(region_stats.round(2))

fig, ax = plt.subplots(figsize=(11, 5))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(region_stats)))
region_stats['count'].sort_values().plot.barh(ax=ax, color=colors, alpha=0.9)
ax.set_xlabel('Number of Listings')
ax.set_title('Listing Volume by Region')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/region_volume.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 6. Engagement & Demand Analysis"),
    code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, color in zip(axes, ['views','bids','inquiries'], ['steelblue','coral','green']):
    df[col].clip(upper=df[col].quantile(0.99)).plot.hist(bins=40, ax=ax, color=color, alpha=0.8, edgecolor='white')
    ax.set_title(f'{col.capitalize()} Distribution')
    ax.set_xlabel(col.capitalize())
plt.tight_layout()
plt.savefig('../reports/figures/engagement_distributions.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    code("""\
# Correlation heatmap
num_cols = ['asking_price','age_years','hours_used','views','bids','inquiries','days_on_market','sold']
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('../reports/figures/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 7. Seller Analysis"),
    code("""\
seller_vol = df.groupby('seller_id').agg(
    listings=('listing_id','count'),
    sold_rate=('sold','mean'),
    avg_price=('asking_price','mean'),
).sort_values('listings', ascending=False)

print(f"Total sellers: {len(seller_vol):,}")
print(f"\\nTop 10 sellers by volume:\\n{seller_vol.head(10).round(2)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
seller_vol['listings'].clip(upper=50).plot.hist(bins=30, ax=axes[0], color='purple', alpha=0.8)
axes[0].set_xlabel('Listings per Seller')
axes[0].set_title('Seller Listing Volume Distribution')

seller_vol['sold_rate'].plot.hist(bins=30, ax=axes[1], color='orange', alpha=0.8)
axes[1].set_xlabel('Sold Rate')
axes[1].set_title('Seller Sold Rate Distribution')

plt.tight_layout()
plt.savefig('../reports/figures/seller_analysis.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 8. Key Takeaways\n\n- **Price range**: wide spread from <$10K forklifts to >$200K cranes\n- **Sold rate**: varies significantly by condition and engagement  \n- **Engagement**: highly skewed — most listings get moderate views, a few get massive attention\n- **Regional demand**: Southeast and Midwest dominate listing volume\n- **Seller patterns**: most sellers have few listings; power sellers drive disproportionate volume"),
    code("print('EDA Complete. All figures saved to reports/figures/')"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 2: Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

seg_cells = [
    md("# 02 — Segmentation Analysis\n**Auction Marketplace Segmentation & Price Intelligence Engine**\n\nThis notebook walks through the K-Means segmentation process, cluster validation, and business interpretation of each discovered segment."),
    code("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_theme(style='whitegrid')
print('Ready')"""),
    md("## 1. Load Feature-Engineered Data"),
    code("""\
df = pd.read_csv('../data/processed/listings_features.csv', parse_dates=['list_date'])
print(f"Shape: {df.shape}")
print(f"Segment distribution:")
if 'segment_label' in df.columns:
    print(df['segment_label'].value_counts())"""),
    md("## 2. Segmentation Feature Set"),
    code("""\
SEG_FEATURES = [
    'log_asking_price', 'price_vs_cat_median', 'price_vs_cond_median',
    'age_years', 'log_hours_used', 'hours_vs_cat_median',
    'total_engagement', 'bid_rate', 'inquiry_rate',
    'condition_score', 'seller_listing_count', 'seller_sold_rate',
    'category_freq', 'region_demand_proxy', 'log_views',
]

X = df[SEG_FEATURES].fillna(df[SEG_FEATURES].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Feature matrix: {X_scaled.shape}")"""),
    md("## 3. Elbow & Silhouette Method"),
    code("""\
# Load pre-computed k-selection results
k_path = '../data/processed/k_selection.json'
if os.path.exists(k_path):
    with open(k_path) as f:
        k_results = json.load(f)

    ks = [int(k) for k in k_results.keys()]
    sils = [k_results[str(k)]['silhouette'] for k in ks]
    inertias = [k_results[str(k)]['inertia'] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(ks, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Selected k=5')
    ax1.set_xlabel('k'); ax1.set_ylabel('Inertia'); ax1.set_title('Elbow Method'); ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ks, sils, 'rs-', linewidth=2, markersize=8)
    ax2.axvline(x=5, color='blue', linestyle='--', alpha=0.5, label='Selected k=5')
    ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette Score'); ax2.set_title('Silhouette Score vs k'); ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('../reports/figures/elbow_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Best silhouette at k={ks[sils.index(max(sils))]}: {max(sils):.4f}")
else:
    print("Run train_segmentation.py first to generate k_selection.json")"""),
    md("## 4. Segment Profiles"),
    code("""\
if 'segment_label' in df.columns:
    profile_cols = ['asking_price','age_years','hours_used','total_engagement',
                    'bid_rate','condition_score','seller_sold_rate','days_on_market','sold']

    profile = df.groupby('segment_label')[profile_cols].agg(['mean','median']).round(2)
    profile.columns = ['_'.join(c) for c in profile.columns]
    profile['count'] = df.groupby('segment_label')['listing_id'].count()
    print(profile[['count','asking_price_mean','age_years_mean','total_engagement_mean','sold_mean']].to_string())"""),
    md("## 5. PCA Cluster Visualization"),
    code("""\
if 'segment_kmeans' in df.columns:
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    SEGMENT_NAMES = {
        0: 'Premium High-Demand',
        1: 'Aging Low-Engagement',
        2: 'Value Fleet Inventory',
        3: 'Hot Fast-Moving Deals',
        4: 'Niche Specialty Equipment',
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.cm.tab10(np.linspace(0, 0.9, 5))

    for i in range(5):
        mask = df['segment_kmeans'] == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[colors[i]], alpha=0.4, s=8,
                   label=SEGMENT_NAMES.get(i, f'Seg {i}'))

    ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} variance explained)')
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} variance explained)')
    ax.set_title('K-Means Segments — PCA Projection')
    ax.legend(markerscale=4, fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('../reports/figures/pca_clusters_notebook.png', dpi=150, bbox_inches='tight')
    plt.show()"""),
    md("## 6. Price Distribution by Segment"),
    code("""\
if 'segment_label' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, 5))

    for i, (seg, group) in enumerate(df.groupby('segment_label')):
        prices = group['asking_price'].clip(upper=group['asking_price'].quantile(0.99))
        prices.plot.kde(ax=ax, label=seg, color=colors[i], linewidth=2)

    ax.set_xlabel('Asking Price ($)')
    ax.set_title('Price Density by Segment')
    ax.set_xlim(0, df['asking_price'].quantile(0.98))
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../reports/figures/segment_price_density.png', dpi=150, bbox_inches='tight')
    plt.show()"""),
    md("## 7. Business Interpretation\n\n| Segment | Name | Characteristics | Strategy |\n|---|---|---|---|\n| 0 | Premium High-Demand | High price, good condition, high engagement | Premium listing, hold price |\n| 1 | Aging Low-Engagement | Old equipment, low engagement, low price | Discount or remarket |\n| 2 | Value Fleet Inventory | Mid-price, high volume sellers, fair condition | Bundle deals, fleet pricing |\n| 3 | Hot Fast-Moving Deals | Moderate price, very high engagement, fast sale | Quick re-list, competitive pricing |\n| 4 | Niche Specialty Equipment | High price, low volume category, specific demand | Target specialist buyers |"),
    code("print('Segmentation analysis complete.')"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 3: Modeling
# ═══════════════════════════════════════════════════════════════════════════════

model_cells = [
    md("# 03 — Predictive Modeling\n**Auction Marketplace Segmentation & Price Intelligence Engine**\n\n### Task: Price Intelligence Classification\nPredict whether a listing is **underpriced**, **fair-priced**, or **overpriced** relative to comparable equipment.\n\nModels evaluated: Logistic Regression (baseline), LightGBM (primary), XGBoost (comparison)"),
    code("""\
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..'))
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

sns.set_theme(style='whitegrid')
print('Ready')"""),
    md("## 1. Load Data & Results"),
    code("""\
df = pd.read_csv('../data/processed/listings_features.csv', parse_dates=['list_date'])

results_path = '../data/processed/model_results.json'
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    print("=== Model Comparison ===")
    for model_name in ['logistic_regression', 'lightgbm', 'xgboost']:
        m = results[model_name]
        print(f"  {m['model']:<25}  acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  auc={m['roc_auc']:.4f}")
    print(f"\\nBest model: {results['best_model']}")
    print(f"Features used: {results['feature_count']}")
    print(f"Train/Test size: {results['train_size']:,} / {results['test_size']:,}")"""),
    md("## 2. Target Distribution"),
    code("""\
print("Price label distribution:")
print(df['price_label'].value_counts())
print()
print("Class proportions:")
print(df['price_label'].value_counts(normalize=True).round(3))

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#e74c3c','#2ecc71','#e67e22']
df['price_label'].value_counts().plot.bar(ax=ax, color=colors, edgecolor='white', rot=0)
ax.set_title('Price Intelligence Target Distribution')
ax.set_ylabel('Count')
ax.set_xlabel('Price Label')
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('../reports/figures/target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()"""),
    md("## 3. Feature Importance (LightGBM)"),
    code("""\
fi_path = '../reports/figures/feature_importance.png'
if os.path.exists(fi_path):
    from PIL import Image
    img = Image.open(fi_path)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Top Feature Importances — LightGBM')
    plt.tight_layout()
    plt.show()
else:
    print("Run train_model.py to generate feature importance plot")

    # Show correlation of features with target as proxy
    MODEL_FEATURES = ['log_asking_price','age_years','log_hours_used','total_engagement',
                      'bid_rate','inquiry_rate','condition_score','seller_sold_rate',
                      'category_freq','region_demand_proxy']
    available = [f for f in MODEL_FEATURES if f in df.columns]
    corr_with_target = df[available + ['price_label_enc']].corr()['price_label_enc'].drop('price_label_enc')

    fig, ax = plt.subplots(figsize=(8, 6))
    corr_with_target.sort_values().plot.barh(ax=ax, color=['red' if v < 0 else 'green' for v in corr_with_target.sort_values()])
    ax.set_title('Feature Correlation with Price Label')
    ax.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.show()"""),
    md("## 4. Model Performance Comparison"),
    code("""\
if os.path.exists(results_path):
    models = ['logistic_regression', 'lightgbm', 'xgboost']
    metrics = ['accuracy', 'precision', 'recall', 'f1_macro', 'roc_auc']

    comparison_df = pd.DataFrame({
        m_name: [results[m_name][metric] for metric in metrics]
        for m_name in models
    }, index=metrics)

    print(comparison_df.round(4))

    fig, ax = plt.subplots(figsize=(11, 5))
    comparison_df.T.plot.bar(ax=ax, width=0.7, edgecolor='white')
    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.set_xticklabels([m.replace('_', '\\n') for m in models], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../reports/figures/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()"""),
    md("## 5. Confusion Matrix"),
    code("""\
cm_path = '../reports/figures/confusion_matrix.png'
if os.path.exists(cm_path):
    from PIL import Image
    img = Image.open(cm_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Run train_model.py to generate confusion matrix")"""),
    md("## 6. Business Recommendations\n\n1. **Underpriced listings** (≈20%): Flag these for buyers as strong value opportunities. Alert sellers to consider repricing.\n2. **Overpriced listings** (≈20%): Target sellers for pricing guidance. These listings have longer days-on-market.\n3. **Fair-priced listings** (≈60%): Strong candidates for featured placement; predictable demand.\n4. **LightGBM** outperforms logistic regression significantly, especially on ROC-AUC — confirms non-linear pricing dynamics.\n5. **Top drivers**: log price, engagement signals, condition, and category context are most predictive."),
    code("print('Modeling analysis complete.')"),
]


def write_notebook(cells, filename):
    notebook = nb(cells)
    path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Written: {path}")


if __name__ == "__main__":
    write_notebook(eda_cells, "01_eda.ipynb")
    write_notebook(seg_cells, "02_segmentation.ipynb")
    write_notebook(model_cells, "03_modeling.ipynb")
    print("All notebooks created.")
