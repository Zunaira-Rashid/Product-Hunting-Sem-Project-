# =============================================================================
# Smart Hunt: Data-Driven Product Hunting and Market Intelligence System
# Module 4.2 — Winning Product Score
# Author  : [Your Name]
# =============================================================================
#
# PURPOSE
# -------
# This script calculates a Winning Product Score (0 to 100) for every product
# by combining four indicators:
#
#   1. Profit Margin Score  — how profitable is the product?
#   2. Demand Score         — how much is it in demand?
#   3. Competition Score    — how crowded is its category?
#   4. Customer Rating Score — how well do customers rate it?
#
# FORMULA
# -------
#   winning_score = (margin_score  × 0.35)
#                 + (demand_score  × 0.30)
#                 + (comp_score    × 0.20)
#                 + (rating_score  × 0.15)
#
# WEIGHT JUSTIFICATION
# --------------------
#   Margin  35% — profit is the primary goal for any dropshipper
#   Demand  30% — no sales = no revenue, demand is critical
#   Competition 20% — high competition makes winning harder
#   Rating  15% — good ratings help conversion but are secondary
#
# HOW TO RUN
# ----------
#   python src/winning_score.py
#
# OUTPUT
# ------
#   data/processed/scored_products.csv  — dataset with scores added
#   outputs/score_report.txt            — scoring summary report
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# 0.  FILE PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "processed", "featured_products.csv")
OUTPUT_CSV  = os.path.join(PROJECT_ROOT, "data", "processed", "scored_products.csv")
CHARTS_DIR  = os.path.join(PROJECT_ROOT, "outputs", "charts")
REPORT_FILE = os.path.join(PROJECT_ROOT, "outputs", "score_report.txt")

os.makedirs(CHARTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  LOAD DATA
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  Smart Hunt — Module 4.2: Winning Product Score")
print("=" * 70)

if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(
        f"\n[ERROR] Featured dataset not found at:\n  {INPUT_CSV}\n"
        "Please run simulate_features.py first."
    )

df = pd.read_csv(INPUT_CSV)
print(f"\n[1] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2.  HELPER — Min-Max Normalization (scales any value to 0.0 – 1.0)
# ---------------------------------------------------------------------------

def min_max_normalize(series):
    """
    Scales a pandas Series to range [0, 1].
    Formula: (x - min) / (max - min)
    Returns 0 for all values if min == max (no variation).
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)

# ---------------------------------------------------------------------------
# 3.  CALCULATE THE FOUR INDICATOR SCORES
# ---------------------------------------------------------------------------

print("\n[2] Calculating indicator scores ...")

# ── 3.1  Profit Margin Score ─────────────────────────────────────────────────
# Profit margin = selling price - all costs, as % of selling price
# Higher margin → higher score (good for dropshipper)

df["profit_margin_pct"] = (
    (df["discounted_price"] - df["supplier_cost"] - df["shipping_cost"] - df["ad_cost"])
    / df["discounted_price"] * 100
).round(2)

# Normalize to 0–1
df["margin_score"] = min_max_normalize(df["profit_margin_pct"])

print(f"    ✔ margin_score    — avg: {df['margin_score'].mean():.3f}")

# ── 3.2  Demand Score ────────────────────────────────────────────────────────
# We use units_sold as the demand measure (simulated from rating_count)
# Higher units sold → higher demand → higher score

df["demand_score"] = min_max_normalize(df["units_sold"])

print(f"    ✔ demand_score    — avg: {df['demand_score'].mean():.3f}")

# ── 3.3  Competition Score ───────────────────────────────────────────────────
# Competition = how many products are in the same main_category
# MORE products in a category = MORE competition = LOWER score
# So we INVERT the normalization: (1 - normalized_count)

# Step 1: count products per category
category_counts = df["main_category"].value_counts()

# Step 2: map count back to each product row
df["category_product_count"] = df["main_category"].map(category_counts)

# Step 3: normalize and invert (less competition = higher score)
normalized_count = min_max_normalize(df["category_product_count"])
df["competition_score"] = (1 - normalized_count).round(4)

print(f"    ✔ competition_score — avg: {df['competition_score'].mean():.3f}")

# ── 3.4  Customer Rating Score ───────────────────────────────────────────────
# Higher rating → higher score
# Missing ratings get a score of 0 (we do not impute)

df["rating_score"] = min_max_normalize(df["rating"].fillna(0))

print(f"    ✔ rating_score    — avg: {df['rating_score'].mean():.3f}")

# ---------------------------------------------------------------------------
# 4.  COMBINE INTO WINNING PRODUCT SCORE
# ---------------------------------------------------------------------------

print("\n[3] Calculating Winning Product Score ...")

# Weights (must add up to 1.0)
W_MARGIN      = 0.35   # 35% — profit is the primary goal
W_DEMAND      = 0.30   # 30% — demand drives revenue
W_COMPETITION = 0.20   # 20% — low competition = easier to win
W_RATING      = 0.15   # 15% — customer satisfaction

df["winning_score"] = (
    (df["margin_score"]      * W_MARGIN)      +
    (df["demand_score"]      * W_DEMAND)      +
    (df["competition_score"] * W_COMPETITION) +
    (df["rating_score"]      * W_RATING)
)

# Scale to 0–100 for easier reading
df["winning_score"] = (df["winning_score"] * 100).round(2)

print(f"    ✔ Winning score range : {df['winning_score'].min():.2f} — {df['winning_score'].max():.2f}")
print(f"    ✔ Average score       : {df['winning_score'].mean():.2f}")

# ---------------------------------------------------------------------------
# 5.  RANK PRODUCTS (1 = best)
# ---------------------------------------------------------------------------

df["rank"] = df["winning_score"].rank(ascending=False, method="min").astype(int)
df.sort_values("rank", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"\n[4] Products ranked. Rank 1 = highest winning score.")

# ---------------------------------------------------------------------------
# 6.  SHOW TOP 10 WINNING PRODUCTS
# ---------------------------------------------------------------------------

display_cols = ["rank", "product_name", "main_category",
                "discounted_price", "profit_margin_pct",
                "winning_score"]

print("\n    TOP 10 WINNING PRODUCTS:")
print("    " + "-" * 60)
top10 = df[display_cols].head(10)
for _, row in top10.iterrows():
    name = str(row["product_name"])[:50]
    print(f"    #{int(row['rank']):4d} | Score: {row['winning_score']:5.2f} | "
          f"Margin: {row['profit_margin_pct']:5.1f}% | {name}...")

# ---------------------------------------------------------------------------
# 7.  CHARTS
# ---------------------------------------------------------------------------

print("\n[5] Generating score charts ...")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 6)})

def save_chart(filename):
    path = os.path.join(CHARTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"    ✔ Saved: outputs/charts/{filename}")

# ── Chart 11 : Winning Score Distribution ────────────────────────────────────
fig, ax = plt.subplots()
ax.hist(df["winning_score"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(df["winning_score"].mean(), color="red", linestyle="--",
           label=f"Mean: {df['winning_score'].mean():.1f}")
ax.set_title("Distribution of Winning Product Scores")
ax.set_xlabel("Winning Score (0–100)")
ax.set_ylabel("Number of Products")
ax.legend()
save_chart("11_winning_score_distribution.png")

# ── Chart 12 : Top 20 Products by Winning Score (Horizontal Bar) ─────────────
fig, ax = plt.subplots(figsize=(12, 8))
top20 = df.head(20).copy()
top20["short_name"] = top20["product_name"].str[:50] + "..."
bars = ax.barh(top20["short_name"], top20["winning_score"],
               color=sns.color_palette("Blues_d", 20))
ax.set_title("Top 20 Products by Winning Score")
ax.set_xlabel("Winning Score (0–100)")
ax.invert_yaxis()
for bar, val in zip(bars, top20["winning_score"]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=8)
save_chart("12_top20_winning_products.png")

# ── Chart 13 : Average Winning Score per Category ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
cat_score = (
    df.groupby("main_category")["winning_score"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)
bars = ax.bar(cat_score.index, cat_score.values,
              color=sns.color_palette("viridis", len(cat_score)))
ax.set_title("Average Winning Score per Category")
ax.set_xlabel("Category")
ax.set_ylabel("Average Winning Score")
ax.tick_params(axis="x", rotation=30)
for bar, val in zip(bars, cat_score.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)
save_chart("13_avg_score_per_category.png")

# ── Chart 14 : Score Components Side-by-Side (Top 15 products) ───────────────
fig, ax = plt.subplots(figsize=(13, 6))
top15 = df.head(15).copy()
top15["short_name"] = top15["product_name"].str[:30] + "..."
x = np.arange(len(top15))
width = 0.2

ax.bar(x - 1.5*width, top15["margin_score"],      width, label="Margin (35%)",      color="#4C72B0")
ax.bar(x - 0.5*width, top15["demand_score"],       width, label="Demand (30%)",       color="#55A868")
ax.bar(x + 0.5*width, top15["competition_score"],  width, label="Competition (20%)",  color="#C44E52")
ax.bar(x + 1.5*width, top15["rating_score"],       width, label="Rating (15%)",       color="#8172B2")

ax.set_title("Score Components for Top 15 Products")
ax.set_xlabel("Product")
ax.set_ylabel("Score (0–1)")
ax.set_xticks(x)
ax.set_xticklabels(top15["short_name"], rotation=45, ha="right", fontsize=8)
ax.legend()
save_chart("14_score_components_top15.png")

# ── Chart 15 : Profit Margin vs Winning Score (Scatter) ──────────────────────
fig, ax = plt.subplots()
scatter = ax.scatter(
    df["profit_margin_pct"], df["winning_score"],
    c=df["demand_score"], cmap="YlOrRd",
    alpha=0.5, s=20
)
plt.colorbar(scatter, ax=ax, label="Demand Score")
ax.set_title("Profit Margin vs Winning Score\n(colour = Demand Score)")
ax.set_xlabel("Profit Margin (%)")
ax.set_ylabel("Winning Score (0–100)")
save_chart("15_margin_vs_winning_score.png")

# ---------------------------------------------------------------------------
# 8.  SAVE SCORED DATASET
# ---------------------------------------------------------------------------

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n[6] Scored dataset saved to:")
print(f"    {OUTPUT_CSV}")
print(f"    Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 9.  WRITE SCORE REPORT
# ---------------------------------------------------------------------------

report = []
report.append("=" * 70)
report.append("  SMART HUNT — Module 4.2: Winning Product Score Report")
report.append("=" * 70)

report.append("\n\n  SCORING FORMULA")
report.append("  " + "-" * 40)
report.append("  winning_score = (margin_score  × 0.35)")
report.append("                + (demand_score  × 0.30)")
report.append("                + (comp_score    × 0.20)")
report.append("                + (rating_score  × 0.15)")
report.append("  Final score scaled to 0–100.")

report.append("\n\n  WEIGHT JUSTIFICATION")
report.append("  " + "-" * 40)
report.append("  Margin      35% — profit is the primary goal for any dropshipper")
report.append("  Demand      30% — no demand = no revenue")
report.append("  Competition 20% — less competition = easier market entry")
report.append("  Rating      15% — customer trust, affects conversion rate")

report.append("\n\n  SCORE STATISTICS")
report.append("  " + "-" * 40)
report.append(f"  Total products scored : {len(df)}")
report.append(f"  Average score         : {df['winning_score'].mean():.2f}")
report.append(f"  Median score          : {df['winning_score'].median():.2f}")
report.append(f"  Highest score         : {df['winning_score'].max():.2f}")
report.append(f"  Lowest score          : {df['winning_score'].min():.2f}")

report.append("\n\n  TOP 20 WINNING PRODUCTS")
report.append("  " + "-" * 40)
for _, row in df[display_cols].head(20).iterrows():
    name = str(row["product_name"])[:55]
    report.append(
        f"  Rank {int(row['rank']):3d} | Score: {row['winning_score']:5.2f} | "
        f"Margin: {row['profit_margin_pct']:5.1f}% | {name}"
    )

report.append("\n\n  AVERAGE WINNING SCORE PER CATEGORY")
report.append("  " + "-" * 40)
for cat, val in cat_score.items():
    report.append(f"  {cat:30s} : {val:.2f}")

report.append("\n\n  INDICATOR SCORE AVERAGES")
report.append("  " + "-" * 40)
for col in ["margin_score", "demand_score", "competition_score", "rating_score"]:
    report.append(f"  {col:25s} : {df[col].mean():.4f}")

report.append("\n\n  CHARTS GENERATED (Module 4.2 — Scores)")
report.append("  " + "-" * 40)
score_charts = [
    "11_winning_score_distribution.png  — Histogram of all winning scores",
    "12_top20_winning_products.png      — Top 20 products by score",
    "13_avg_score_per_category.png      — Avg score per category",
    "14_score_components_top15.png      — 4 indicator bars for top 15",
    "15_margin_vs_winning_score.png     — Scatter: margin vs score",
]
for c in score_charts:
    report.append(f"  {c}")

report.append("\n\n" + "=" * 70)
report.append("  END OF SCORE REPORT")
report.append("=" * 70 + "\n")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"    ✔ Saved: outputs/score_report.txt")

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  Winning Product Score complete.")
print(f"  Scored CSV  →  data/processed/scored_products.csv")
print(f"  5 charts    →  outputs/charts/11 to 15")
print(f"  Report      →  outputs/score_report.txt")
print("=" * 70 + "\n")
