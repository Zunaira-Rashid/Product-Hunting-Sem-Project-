# =============================================================================
# Smart Hunt: Data-Driven Product Hunting and Market Intelligence System
# Module 4.1 — Simulated Feature Engineering
# Author  : [Your Name]
# =============================================================================
#
# PURPOSE
# -------
# The original Amazon dataset does not contain supplier cost, shipping cost,
# advertising cost, COD return rate, or units sold. These fields are required
# for Module 4.2 (Winning Product Score) and Module 4.7 (Profit Calculator).
#
# As stated in the project proposal (page 3):
#   "Where supplier cost, advertising cost or return rate is not available
#    in public datasets, simulated values will be used and clearly stated."
#
# All five columns added here are SIMULATED based on industry norms.
# They are NOT from the original Amazon dataset.
#
# SIMULATION RULES (justified below)
# ------------------------------------
# supplier_cost     = 30% to 50% of actual_price
#                     (typical wholesale margin in e-commerce)
#
# shipping_cost     = Rs. 50 to Rs. 200 flat per order
#                     (average courier cost in Pakistan/India)
#
# ad_cost           = 10% to 15% of discounted_price
#                     (standard digital advertising budget)
#
# cod_return_rate   = 15% to 30% per product (as a decimal: 0.15 to 0.30)
#                     (Pakistan COD market average return rate)
#
# units_sold        = rating_count × random multiplier between 3 and 7
#                     (industry estimate: ~1 in 5 buyers leaves a review)
#
# HOW TO RUN
# ----------
#   python src/simulate_features.py
# =============================================================================

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0.  FILE PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV  = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_products.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "featured_products.csv")
REPORT_FILE = os.path.join(PROJECT_ROOT, "outputs", "simulation_report.txt")

# ---------------------------------------------------------------------------
# 1.  LOAD CLEANED DATASET
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  Smart Hunt — Simulated Feature Engineering")
print("=" * 70)

# Check file exists
if not os.path.isfile(INPUT_CSV):
    raise FileNotFoundError(
        f"\n[ERROR] Cleaned dataset not found at:\n  {INPUT_CSV}\n"
        "Please run data_cleaning.py first."
    )

df = pd.read_csv(INPUT_CSV)
print(f"\n[1] Loaded cleaned dataset: {df.shape[0]} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2.  SET RANDOM SEED (so results are reproducible every time you run)
# ---------------------------------------------------------------------------

# np.random.seed(42) means every time you run this script,
# you get the SAME simulated values — important for reproducibility
np.random.seed(42)
n = len(df)  # total number of products

print(f"    Random seed set to 42 — results will be same every run")

# ---------------------------------------------------------------------------
# 3.  SIMULATE MISSING COLUMNS
# ---------------------------------------------------------------------------

print("\n[2] Simulating missing columns ...\n")

# ── 3.1  Supplier Cost ───────────────────────────────────────────────────────
# Rule : 30% to 50% of actual_price
# Why  : In e-commerce wholesale, supplier usually sells at 30-50% of MRP
# Type : float, rounded to 2 decimal places

df["supplier_cost"] = (
    df["actual_price"] *
    np.random.uniform(0.30, 0.50, size=n)
).round(2)

# Cap supplier_cost at 80% of discounted_price so margin is never impossibly negative
# (a seller would not buy at more than 80% of their own selling price)
df["supplier_cost"] = df[["supplier_cost", "discounted_price"]].apply(
    lambda row: min(row["supplier_cost"], row["discounted_price"] * 0.80), axis=1
).round(2)

print(f"    ✔ supplier_cost     — 30% to 50% of actual_price")
print(f"      Sample values: {df['supplier_cost'].head(3).tolist()}")

# ── 3.2  Shipping Cost ───────────────────────────────────────────────────────
# Rule : Rs. 50 to Rs. 200 flat per order
# Why  : Average courier delivery cost in Pakistan/India for small parcels
# Type : float, rounded to 2 decimal places

df["shipping_cost"] = np.random.uniform(50, 200, size=n).round(2)

print(f"\n    ✔ shipping_cost     — Rs. 50 to Rs. 200 flat per order")
print(f"      Sample values: {df['shipping_cost'].head(3).tolist()}")

# ── 3.3  Advertising Cost ────────────────────────────────────────────────────
# Rule : 10% to 15% of discounted_price
# Why  : Standard digital marketing spend for e-commerce products
# Type : float, rounded to 2 decimal places

df["ad_cost"] = (
    df["discounted_price"] *
    np.random.uniform(0.10, 0.15, size=n)
).round(2)

print(f"\n    ✔ ad_cost           — 10% to 15% of discounted_price")
print(f"      Sample values: {df['ad_cost'].head(3).tolist()}")

# ── 3.4  COD Return Rate ─────────────────────────────────────────────────────
# Rule : 15% to 30% per product (stored as decimal, e.g. 0.18 = 18%)
# Why  : Cash-on-delivery return rate in Pakistan averages 15-30%
#        This is Pakistan-specific and no public dataset contains it
# Type : float, rounded to 4 decimal places

df["cod_return_rate"] = np.random.uniform(0.15, 0.30, size=n).round(4)

print(f"\n    ✔ cod_return_rate   — 15% to 30% (Pakistan COD average)")
print(f"      Sample values: {df['cod_return_rate'].head(3).tolist()}")

# ── 3.5  Units Sold ──────────────────────────────────────────────────────────
# Rule : rating_count × random multiplier between 3 and 7
# Why  : Research suggests roughly 1 in 5 buyers leaves a review,
#        so actual buyers = rating_count × ~5 (we use 3-7 range)
#        Where rating_count is NaN, we use 0
# Type : integer

rating_count_filled = df["rating_count"].fillna(0)

df["units_sold"] = (
    rating_count_filled *
    np.random.uniform(3, 7, size=n)
).astype(int)

print(f"\n    ✔ units_sold        — rating_count × 3 to 7 multiplier")
print(f"      Sample values: {df['units_sold'].head(3).tolist()}")

# ---------------------------------------------------------------------------
# 4.  ADD A CLEAR LABEL COLUMN
# ---------------------------------------------------------------------------

# This column will appear in the CSV so anyone reading it knows
# which columns are real and which are simulated
df["data_source"] = "real"  # default for all rows

# Mark which columns are simulated — add a note in the report
SIMULATED_COLUMNS = [
    "supplier_cost",
    "shipping_cost",
    "ad_cost",
    "cod_return_rate",
    "units_sold"
]

# ---------------------------------------------------------------------------
# 5.  QUICK VALIDATION — make sure nothing looks wrong
# ---------------------------------------------------------------------------

print("\n[3] Validation checks ...")

# supplier_cost should always be less than actual_price
bad_supplier = (df["supplier_cost"] >= df["actual_price"]).sum()
print(f"    supplier_cost >= actual_price : {bad_supplier} rows (should be 0)")

# shipping_cost should be between 50 and 200
bad_shipping = ((df["shipping_cost"] < 50) | (df["shipping_cost"] > 200)).sum()
print(f"    shipping_cost out of range    : {bad_shipping} rows (should be 0)")

# cod_return_rate should be between 0.15 and 0.30
bad_cod = ((df["cod_return_rate"] < 0.15) | (df["cod_return_rate"] > 0.30)).sum()
print(f"    cod_return_rate out of range  : {bad_cod} rows (should be 0)")

# units_sold should be >= 0
bad_units = (df["units_sold"] < 0).sum()
print(f"    units_sold negative           : {bad_units} rows (should be 0)")

# ---------------------------------------------------------------------------
# 6.  SAVE FEATURED DATASET
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n[4] Featured dataset saved to:")
print(f"    {OUTPUT_CSV}")
print(f"    Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 7.  WRITE SIMULATION REPORT
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

report = []
report.append("=" * 70)
report.append("  SMART HUNT — Simulation Report")
report.append("  Module 4.1 — Simulated Feature Engineering")
report.append("=" * 70)

report.append("\n  DISCLAIMER")
report.append("  " + "-" * 40)
report.append("  The following 5 columns are SIMULATED.")
report.append("  They are NOT from the original Amazon dataset.")
report.append("  They are based on industry norms and Pakistan market estimates.")
report.append("  As stated in the project proposal (page 3), simulated values")
report.append("  are used where real data is unavailable in public datasets.")

report.append("\n\n  INPUT FILE")
report.append("  " + "-" * 40)
report.append(f"  {INPUT_CSV}")
report.append(f"  Rows: {df.shape[0]}   Columns before simulation: {df.shape[1] - len(SIMULATED_COLUMNS) - 1}")

report.append("\n\n  OUTPUT FILE")
report.append("  " + "-" * 40)
report.append(f"  {OUTPUT_CSV}")
report.append(f"  Rows: {df.shape[0]}   Columns after simulation: {df.shape[1]}")

report.append("\n\n  SIMULATED COLUMNS — RULES AND JUSTIFICATION")
report.append("  " + "-" * 40)

sim_details = [
    (
        "supplier_cost",
        "30% to 50% of actual_price",
        "Wholesale margin in e-commerce. Suppliers typically sell at 30-50% of MRP.",
        df["supplier_cost"].describe()
    ),
    (
        "shipping_cost",
        "Rs. 50 to Rs. 200 (flat per order)",
        "Average courier delivery cost in Pakistan/India for small-to-medium parcels.",
        df["shipping_cost"].describe()
    ),
    (
        "ad_cost",
        "10% to 15% of discounted_price",
        "Standard digital advertising budget for e-commerce product promotion.",
        df["ad_cost"].describe()
    ),
    (
        "cod_return_rate",
        "0.15 to 0.30 (15% to 30%)",
        "Pakistan COD market return rate. No public dataset contains this — simulated.",
        df["cod_return_rate"].describe()
    ),
    (
        "units_sold",
        "rating_count × 3 to 7",
        "Approx. 1 in 5 buyers leaves a review. Multiplier 3-7 gives conservative estimate.",
        df["units_sold"].describe()
    ),
]

for col_name, rule, justification, stats in sim_details:
    report.append(f"\n  Column      : {col_name}")
    report.append(f"  Rule        : {rule}")
    report.append(f"  Justification: {justification}")
    report.append(f"  Statistics  :")
    for stat_name, val in stats.items():
        report.append(f"    {stat_name:8s} = {val:.4f}")

report.append("\n\n  VALIDATION RESULTS")
report.append("  " + "-" * 40)
report.append(f"  supplier_cost >= actual_price : {bad_supplier} rows")
report.append(f"  shipping_cost out of range    : {bad_shipping} rows")
report.append(f"  cod_return_rate out of range  : {bad_cod} rows")
report.append(f"  units_sold negative           : {bad_units} rows")
report.append("  All checks passed." if all(v == 0 for v in [bad_supplier, bad_shipping, bad_cod, bad_units])
              else "  WARNING: Some checks failed — review the data.")

report.append("\n\n  COLUMNS IN FINAL DATASET")
report.append("  " + "-" * 40)
for i, col in enumerate(df.columns, 1):
    tag = " ← SIMULATED" if col in SIMULATED_COLUMNS else ""
    report.append(f"  {i:2d}. {col}{tag}")

report.append("\n\n" + "=" * 70)
report.append("  END OF SIMULATION REPORT")
report.append("=" * 70 + "\n")

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print(f"\n[5] Simulation report saved to:")
print(f"    {REPORT_FILE}")

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  Simulation complete.")
print(f"  Featured CSV  →  data/processed/featured_products.csv")
print(f"  Report        →  outputs/simulation_report.txt")
print("=" * 70 + "\n")
