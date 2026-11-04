# =============================================================================
# Smart Hunt: Data-Driven Product Hunting and Market Intelligence System
# Module 4.2 — Analytics & Visualization
# Author  : [Your Name]
# =============================================================================
#
# PURPOSE
# -------
# This script performs statistical analysis and creates visualizations on the
# featured dataset. It answers key business questions:
#   1. What price ranges are most common?
#   2. Which categories have the best ratings?
#   3. What is the relationship between discount and demand?
#   4. How are ratings distributed across all products?
#   5. Which categories have the most products (competition)?
#   6. What is the average profit margin per category?
#   7. Top 10 products by units sold (demand)
#
# HOW TO RUN
# ----------
#   python src/analytics.py
#
# OUTPUT
# ------
#   outputs/charts/  — folder containing all chart images (PNG)
#   outputs/analytics_report.txt — statistical summary report
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# 0.  FILE PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV    = os.path.join(PROJECT_ROOT, "data", "processed", "featured_products.csv")
CHARTS_DIR   = os.path.join(PROJECT_ROOT, "outputs", "charts")
REPORT_FILE  = os.path.join(PROJECT_ROOT, "outputs", "analytics_report.txt")

os.makedirs(CHARTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  LOAD DATA
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  Smart Hunt — Module 4.2: Analytics & Visualization")
print("=" * 70)

if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(
        f"\n[ERROR] Featured dataset not found at:\n  {INPUT_CSV}\n"
        "Please run simulate_features.py first."
    )

df = pd.read_csv(INPUT_CSV)
print(f"\n[1] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2.  GLOBAL CHART STYLE
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi"      : 120,
    "figure.figsize"  : (10, 6),
    "axes.titlesize"  : 14,
    "axes.labelsize"  : 12,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
})

# ---------------------------------------------------------------------------
# 3.  HELPER — save chart
# ---------------------------------------------------------------------------

def save_chart(filename, title=""):
    path = os.path.join(CHARTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"    ✔ Saved: outputs/charts/{filename}")
    return path

# ---------------------------------------------------------------------------
# 4.  CALCULATE PROFIT MARGIN COLUMN (needed for several charts)
# ---------------------------------------------------------------------------
# Profit margin = (discounted_price - supplier_cost - shipping_cost - ad_cost)
#                  divided by discounted_price × 100
# This gives profit as a percentage of selling price

df["profit_margin_pct"] = (
    (df["discounted_price"] - df["supplier_cost"] - df["shipping_cost"] - df["ad_cost"])
    / df["discounted_price"] * 100
).round(2)

print(f"\n[2] Profit margin column calculated.")
print(f"    Average profit margin : {df['profit_margin_pct'].mean():.2f}%")
print(f"    Min profit margin     : {df['profit_margin_pct'].min():.2f}%")
print(f"    Max profit margin     : {df['profit_margin_pct'].max():.2f}%")

# ---------------------------------------------------------------------------
# 5.  CHARTS
# ---------------------------------------------------------------------------

print("\n[3] Generating charts ...")

# ── Chart 1 : Price Distribution (Histogram) ────────────────────────────────
fig, ax = plt.subplots()
# Filter out very expensive outliers for clearer visualization (keep < 20000)
price_data = df[df["discounted_price"] < 20000]["discounted_price"]
ax.hist(price_data, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.set_title("Distribution of Discounted Prices (Products < Rs. 20,000)")
ax.set_xlabel("Discounted Price (Rs.)")
ax.set_ylabel("Number of Products")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
save_chart("01_price_distribution.png")

# ── Chart 2 : Rating Distribution (Histogram) ────────────────────────────────
fig, ax = plt.subplots()
rating_data = df["rating"].dropna()
ax.hist(rating_data, bins=20, color="#55A868", edgecolor="white", alpha=0.85)
ax.set_title("Distribution of Product Ratings")
ax.set_xlabel("Rating (out of 5)")
ax.set_ylabel("Number of Products")
ax.set_xlim(1, 5)
save_chart("02_rating_distribution.png")

# ── Chart 3 : Average Rating per Main Category (Bar Chart) ──────────────────
fig, ax = plt.subplots(figsize=(11, 6))
cat_rating = (
    df.groupby("main_category")["rating"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)
bars = ax.bar(cat_rating.index, cat_rating.values, color=sns.color_palette("muted", len(cat_rating)))
ax.set_title("Average Rating per Category")
ax.set_xlabel("Category")
ax.set_ylabel("Average Rating")
ax.set_ylim(0, 5.5)
ax.tick_params(axis="x", rotation=30)
# Add value labels on top of bars
for bar, val in zip(bars, cat_rating.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)
save_chart("03_avg_rating_per_category.png")

# ── Chart 4 : Product Count per Category (Competition) ───────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
cat_count = df["main_category"].value_counts()
bars = ax.bar(cat_count.index, cat_count.values,
              color=sns.color_palette("pastel", len(cat_count)))
ax.set_title("Number of Products per Category (Competition Level)")
ax.set_xlabel("Category")
ax.set_ylabel("Number of Products")
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, cat_count.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=9)
save_chart("04_products_per_category.png")

# ── Chart 5 : Discount % vs Rating Count (Scatter) ───────────────────────────
fig, ax = plt.subplots()
scatter_df = df[df["rating_count"].notna() & (df["rating_count"] < 200000)]
ax.scatter(
    scatter_df["discount_percentage"],
    scatter_df["rating_count"],
    alpha=0.4, color="#C44E52", s=20
)
ax.set_title("Discount Percentage vs Rating Count (Demand Proxy)")
ax.set_xlabel("Discount Percentage (%)")
ax.set_ylabel("Rating Count (Demand Proxy)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save_chart("05_discount_vs_demand.png")

# ── Chart 6 : Average Profit Margin per Category (Bar Chart) ─────────────────
fig, ax = plt.subplots(figsize=(11, 6))
cat_margin = (
    df.groupby("main_category")["profit_margin_pct"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)
bars = ax.bar(cat_margin.index, cat_margin.values,
              color=sns.color_palette("coolwarm", len(cat_margin)))
ax.set_title("Average Profit Margin (%) per Category")
ax.set_xlabel("Category")
ax.set_ylabel("Average Profit Margin (%)")
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, cat_margin.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
save_chart("06_profit_margin_per_category.png")

# ── Chart 7 : Top 10 Products by Units Sold ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
top10 = df.nlargest(10, "units_sold")[["product_name", "units_sold"]].copy()
# Shorten long product names for display
top10["short_name"] = top10["product_name"].str[:45] + "..."
bars = ax.barh(top10["short_name"], top10["units_sold"],
               color=sns.color_palette("Blues_d", 10))
ax.set_title("Top 10 Products by Units Sold (Demand)")
ax.set_xlabel("Units Sold (Simulated)")
ax.invert_yaxis()
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
save_chart("07_top10_units_sold.png")

# ── Chart 8 : Correlation Heatmap ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
corr_cols = [
    "discounted_price", "actual_price", "discount_percentage",
    "rating", "rating_count", "supplier_cost",
    "ad_cost", "cod_return_rate", "units_sold", "profit_margin_pct"
]
corr_matrix = df[corr_cols].corr().round(2)
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, ax=ax, annot_kws={"size": 8}
)
ax.set_title("Correlation Heatmap — Key Numeric Columns")
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)
save_chart("08_correlation_heatmap.png")

# ── Chart 9 : Boxplot — Rating by Category ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
# Only keep categories with enough products for a meaningful boxplot
valid_cats = df["main_category"].value_counts()
valid_cats = valid_cats[valid_cats >= 5].index
box_df = df[df["main_category"].isin(valid_cats)]
sns.boxplot(data=box_df, x="main_category", y="rating", ax=ax,
            palette="Set2")
ax.set_title("Rating Distribution by Category (Boxplot)")
ax.set_xlabel("Category")
ax.set_ylabel("Rating")
ax.tick_params(axis="x", rotation=30)
save_chart("09_rating_boxplot_by_category.png")

# ── Chart 10 : Profit Margin Distribution ────────────────────────────────────
fig, ax = plt.subplots()
margin_data = df["profit_margin_pct"]
ax.hist(margin_data, bins=40, color="#8172B2", edgecolor="white", alpha=0.85)
ax.axvline(margin_data.mean(), color="red", linestyle="--",
           label=f"Mean: {margin_data.mean():.1f}%")
ax.set_title("Distribution of Profit Margin (%)")
ax.set_xlabel("Profit Margin (%)")
ax.set_ylabel("Number of Products")
ax.legend()
save_chart("10_profit_margin_distribution.png")

# ---------------------------------------------------------------------------
# 6.  STATISTICAL SUMMARY
# ---------------------------------------------------------------------------

print("\n[4] Generating statistical summary ...")

report = []
report.append("=" * 70)
report.append("  SMART HUNT — Module 4.2: Analytics Report")
report.append("=" * 70)

report.append("\n\n  DATASET OVERVIEW")
report.append("  " + "-" * 40)
report.append(f"  Total products     : {len(df)}")
report.append(f"  Total columns      : {df.shape[1]}")
report.append(f"  Categories         : {df['main_category'].nunique()}")
report.append(f"  Price range        : Rs. {df['discounted_price'].min():.0f} — Rs. {df['discounted_price'].max():.0f}")
report.append(f"  Average price      : Rs. {df['discounted_price'].mean():.2f}")

report.append("\n\n  RATING STATISTICS")
report.append("  " + "-" * 40)
report.append(f"  Average rating     : {df['rating'].mean():.2f}")
report.append(f"  Median rating      : {df['rating'].median():.2f}")
report.append(f"  Min rating         : {df['rating'].min():.2f}")
report.append(f"  Max rating         : {df['rating'].max():.2f}")
report.append(f"  Products rated 4+  : {(df['rating'] >= 4).sum()} ({(df['rating'] >= 4).mean()*100:.1f}%)")

report.append("\n\n  DEMAND STATISTICS (rating_count as proxy)")
report.append("  " + "-" * 40)
report.append(f"  Average rating count : {df['rating_count'].mean():.0f}")
report.append(f"  Median rating count  : {df['rating_count'].median():.0f}")
report.append(f"  Max rating count     : {df['rating_count'].max():.0f}")

report.append("\n\n  DISCOUNT STATISTICS")
report.append("  " + "-" * 40)
report.append(f"  Average discount   : {df['discount_percentage'].mean():.1f}%")
report.append(f"  Median discount    : {df['discount_percentage'].median():.1f}%")
report.append(f"  Max discount       : {df['discount_percentage'].max():.0f}%")

report.append("\n\n  PROFIT MARGIN STATISTICS")
report.append("  " + "-" * 40)
report.append(f"  Average margin     : {df['profit_margin_pct'].mean():.2f}%")
report.append(f"  Median margin      : {df['profit_margin_pct'].median():.2f}%")
report.append(f"  Min margin         : {df['profit_margin_pct'].min():.2f}%")
report.append(f"  Max margin         : {df['profit_margin_pct'].max():.2f}%")

report.append("\n\n  AVERAGE RATING PER CATEGORY")
report.append("  " + "-" * 40)
for cat, val in cat_rating.items():
    report.append(f"  {cat:30s} : {val:.2f}")

report.append("\n\n  PRODUCT COUNT PER CATEGORY (Competition)")
report.append("  " + "-" * 40)
for cat, val in cat_count.items():
    report.append(f"  {cat:30s} : {val}")

report.append("\n\n  AVERAGE PROFIT MARGIN PER CATEGORY")
report.append("  " + "-" * 40)
for cat, val in cat_margin.items():
    report.append(f"  {cat:30s} : {val:.2f}%")

report.append("\n\n  CHARTS GENERATED")
report.append("  " + "-" * 40)
charts = [
    "01_price_distribution.png       — Histogram of discounted prices",
    "02_rating_distribution.png      — Histogram of product ratings",
    "03_avg_rating_per_category.png  — Bar chart: avg rating by category",
    "04_products_per_category.png    — Bar chart: competition per category",
    "05_discount_vs_demand.png       — Scatter: discount % vs rating count",
    "06_profit_margin_per_category.png — Bar chart: avg margin by category",
    "07_top10_units_sold.png         — Top 10 products by demand",
    "08_correlation_heatmap.png      — Heatmap of numeric correlations",
    "09_rating_boxplot_by_category.png — Boxplot: rating spread by category",
    "10_profit_margin_distribution.png — Histogram of profit margins",
]
for c in charts:
    report.append(f"  {c}")

report.append("\n\n" + "=" * 70)
report.append("  END OF ANALYTICS REPORT")
report.append("=" * 70 + "\n")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"    ✔ Saved: outputs/analytics_report.txt")

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  Analytics complete.")
print(f"  10 charts  →  outputs/charts/")
print(f"  Report     →  outputs/analytics_report.txt")
print("=" * 70 + "\n")
