# =============================================================================
# Smart Hunt: Data-Driven Product Hunting and Market Intelligence System
# Module 4.1 — Data Collection & Cleaning
# Author  : [Your Name]
# Dataset : Amazon Sales Dataset (amazon.csv)
# =============================================================================
#
# HOW TO RUN
# ----------
#   Open a terminal, go to the project root folder, and run:
#       python src/data_cleaning.py
#
# WHAT THIS SCRIPT DOES
# ---------------------
#   Task 1 : Inspects the raw dataset (shape, dtypes, missing values, duplicates)
#   Task 2 : Cleans the data using justified, column-specific rules
#   Task 3 : Saves cleaned_products.csv and a cleaning_report.txt
# =============================================================================

import os
import re
import pandas as pd

# ---------------------------------------------------------------------------
# 0.  FILE PATHS  (relative to the project root)
# ---------------------------------------------------------------------------

# The script is inside  /src , so we go one level up to reach the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input  ─ raw CSV (sits inside a folder that is also named amazon.csv)
RAW_CSV      = os.path.join(PROJECT_ROOT, "data", "raw", "amazon.csv", "amazon.csv")

# Output ─ cleaned CSV
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CLEANED_CSV   = os.path.join(PROCESSED_DIR, "cleaned_products.csv")

# Output ─ cleaning report
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")
REPORT_FILE   = os.path.join(OUTPUTS_DIR, "cleaning_report.txt")

# ---------------------------------------------------------------------------
# HELPER  :  pretty section divider for the report
# ---------------------------------------------------------------------------

def section(title):
    """Return a formatted section header string."""
    return "\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70


# ---------------------------------------------------------------------------
# 1.  LOAD THE DATASET
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  Smart Hunt — Module 4.1: Data Collection & Cleaning")
print("=" * 70)

# Check the file exists before trying to load it
if not os.path.isfile(RAW_CSV):
    raise FileNotFoundError(
        f"\n[ERROR] Could not find the CSV file at:\n  {RAW_CSV}\n"
        "Please make sure the file is placed at:  data/raw/amazon.csv/amazon.csv"
    )

print(f"\n[1] Loading dataset from:\n    {RAW_CSV}")

# Read CSV — keep everything as strings at first so nothing is silently
# converted before we have a chance to inspect it
df_raw = pd.read_csv(RAW_CSV, dtype=str)

print(f"    ✔ Loaded successfully — {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# Keep a pristine copy for reference throughout the report
df = df_raw.copy()

# ── Save original shape for the report ──────────────────────────────────────
orig_rows, orig_cols = df.shape


# ---------------------------------------------------------------------------
# TASK 1  :  INSPECT THE DATASET
# ---------------------------------------------------------------------------

print(section("TASK 1 — DATASET INSPECTION"))

# ── 1.1  First five rows ────────────────────────────────────────────────────
print("\n[1.1] First 5 rows:")
print(df.head().to_string())

# ── 1.2  Shape ──────────────────────────────────────────────────────────────
print(f"\n[1.2] Shape:  {orig_rows} rows  ×  {orig_cols} columns")

# ── 1.3  Column names & data types ──────────────────────────────────────────
print("\n[1.3] Column names and data types:")
print(df.dtypes.to_string())

# ── 1.4  Missing values per column ──────────────────────────────────────────
missing_before = df.isnull().sum()
print("\n[1.4] Missing values per column (before cleaning):")
print(missing_before.to_string())

# ── 1.5  Completely duplicated rows ─────────────────────────────────────────
full_dupes = df.duplicated().sum()
print(f"\n[1.5] Completely duplicated rows: {full_dupes}")

# ── 1.6  Duplicate product_ids — inspect before any decision ────────────────
dup_pid_mask  = df.duplicated(subset="product_id", keep=False)
dup_pid_count = dup_pid_mask.sum()
print(f"\n[1.6] Rows sharing a duplicate product_id: {dup_pid_count}")

if dup_pid_count > 0:
    print("      Sample of duplicated product_id records:")
    sample_dup = (
        df[dup_pid_mask][["product_id", "product_name", "discounted_price", "rating"]]
        .sort_values("product_id")
        .head(10)
    )
    print(sample_dup.to_string(index=True))

# ── 1.7  Inconsistent formats in key numeric/category columns ───────────────
print("\n[1.7] Sample raw values — inconsistent formats:")
for col in ["discounted_price", "actual_price", "discount_percentage",
            "rating", "rating_count"]:
    unique_samples = df[col].dropna().unique()[:8]
    print(f"  {col:22s}: {list(unique_samples)}")

print("\n  category (first 5 values):")
for v in df["category"].dropna().unique()[:5]:
    print(f"    {v}")

# ── 1.8  Invalid / unexpected numeric values ─────────────────────────────────
print("\n[1.8] Checking for non-numeric values in price / rating columns …")

def count_non_numeric(series):
    """Count values that cannot be converted to float after basic symbol strip."""
    cleaned = (
        series.astype(str)
              .str.replace(r"[₹,% ]", "", regex=True)
              .str.strip()
    )
    bad = cleaned[pd.to_numeric(cleaned, errors="coerce").isna() & (cleaned != "nan")]
    return len(bad), bad.unique()[:5]

for col in ["discounted_price", "actual_price", "discount_percentage",
            "rating", "rating_count"]:
    n_bad, examples = count_non_numeric(df[col])
    print(f"  {col:22s}: {n_bad} non-numeric value(s)  — examples: {list(examples)}")


# ---------------------------------------------------------------------------
# TASK 2  :  DATA CLEANING
# ---------------------------------------------------------------------------

print(section("TASK 2 — DATA CLEANING"))

# We will track every operation in a list so it goes into the report
cleaning_log = []

# ── Snapshot of dtypes before cleaning ──────────────────────────────────────
dtypes_before = df.dtypes.copy()

# ── 2.1  Strip leading/trailing whitespace from all string columns ────────
df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)
cleaning_log.append("Stripped leading/trailing whitespace from all text columns.")
print("\n[2.1] Stripped whitespace from all text columns.")

# ── 2.2  Convert prices: remove ₹ and commas, cast to float ─────────────────
#   Rule: ₹ symbol and comma-thousands-separators are formatting artefacts.
#         Rows where price is completely absent (NaN / empty) will become NaN.
for price_col in ["discounted_price", "actual_price"]:
    df[price_col] = (
        df[price_col]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    msg = f"Converted '{price_col}': removed ₹/commas → float64."
    cleaning_log.append(msg)
    print(f"[2.2] {msg}")

# ── 2.3  Convert discount_percentage: remove % sign, cast to float ───────────
df["discount_percentage"] = (
    df["discount_percentage"]
    .str.replace("%", "", regex=False)
    .str.strip()
)
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")
msg = "Converted 'discount_percentage': removed % → float64."
cleaning_log.append(msg)
print(f"[2.3] {msg}")

# ── 2.4  Convert rating to float ─────────────────────────────────────────────
#   Rule: Some cells contain '|' or other stray characters; coerce to NaN.
df["rating"] = pd.to_numeric(df["rating"].str.strip(), errors="coerce")
msg = "Converted 'rating' → float64 (non-numeric values → NaN)."
cleaning_log.append(msg)
print(f"[2.4] {msg}")

# ── 2.5  Convert rating_count to integer ─────────────────────────────────────
df["rating_count"] = (
    df["rating_count"]
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
# Use Int64 (nullable integer) so NaN rows are preserved without becoming floats
df["rating_count"] = df["rating_count"].astype("Int64")
msg = "Converted 'rating_count': removed commas → Int64 (nullable integer)."
cleaning_log.append(msg)
print(f"[2.5] {msg}")

# ── 2.6  Validate numeric ranges ─────────────────────────────────────────────
#   Prices must be positive. Ratings must be between 1 and 5. Discount 0–100 %.

before_price_invalid = df[
    (df["discounted_price"].notna() & (df["discounted_price"] <= 0)) |
    (df["actual_price"].notna()     & (df["actual_price"]     <= 0))
].shape[0]

# Set invalid prices to NaN instead of deleting the row
df.loc[df["discounted_price"] <= 0, "discounted_price"] = pd.NA
df.loc[df["actual_price"]     <= 0, "actual_price"    ] = pd.NA

before_rating_invalid = df[
    df["rating"].notna() & ((df["rating"] < 1) | (df["rating"] > 5))
].shape[0]

# Set out-of-range ratings to NaN
df.loc[(df["rating"] < 1) | (df["rating"] > 5), "rating"] = pd.NA

msg = (f"Flagged {before_price_invalid} row(s) with zero/negative prices → set to NaN. "
       f"Flagged {before_rating_invalid} row(s) with rating outside [1,5] → set to NaN.")
cleaning_log.append(msg)
print(f"[2.6] {msg}")

# ── 2.7  Handle missing values — column-specific rules ───────────────────────
#
#   product_id / product_name  : essential — rows without these cannot be used
#   discounted_price           : essential — drop if missing
#   actual_price               : fill from discounted_price when missing (best proxy)
#   discount_percentage        : recalculate when both prices are available
#   rating                     : keep NaN — do NOT replace with 0 (that is invalid)
#   rating_count               : keep NaN — no rating count means unknown, not zero
#   about_product / review_*   : text columns — fill NaN with empty string
#   img_link / product_link    : fill NaN with empty string

# --- Drop rows missing essential identifiers ---
essential_cols = ["product_id", "product_name", "discounted_price"]
rows_before_essential_drop = len(df)
df.dropna(subset=essential_cols, inplace=True)
rows_dropped_essential = rows_before_essential_drop - len(df)
msg = (f"Dropped {rows_dropped_essential} row(s) missing essential fields "
       f"(product_id, product_name, or discounted_price).")
cleaning_log.append(msg)
print(f"[2.7a] {msg}")

# --- Fill actual_price with discounted_price when missing ---
filled_actual = df["actual_price"].isna().sum()
df["actual_price"] = df["actual_price"].fillna(df["discounted_price"])
msg = (f"Filled {filled_actual} missing 'actual_price' value(s) "
       "with 'discounted_price' as best proxy.")
cleaning_log.append(msg)
print(f"[2.7b] {msg}")

# --- Recalculate discount_percentage where missing but both prices exist ---
can_recalc = (
    df["discount_percentage"].isna() &
    df["actual_price"].notna()       &
    (df["actual_price"] > 0)
)
df.loc[can_recalc, "discount_percentage"] = (
    (df.loc[can_recalc, "actual_price"] - df.loc[can_recalc, "discounted_price"])
    / df.loc[can_recalc, "actual_price"] * 100
).round(2)
recalc_count = can_recalc.sum()
msg = (f"Recalculated {recalc_count} missing 'discount_percentage' value(s) "
       "from actual_price and discounted_price.")
cleaning_log.append(msg)
print(f"[2.7c] {msg}")

# --- Fill text / URL columns with empty string ---
text_fill_cols = [
    "about_product", "user_id", "user_name",
    "review_id", "review_title", "review_content",
    "img_link", "product_link"
]
for col in text_fill_cols:
    df[col] = df[col].fillna("")
msg = f"Filled missing text/URL columns with empty string: {text_fill_cols}."
cleaning_log.append(msg)
print(f"[2.7d] {msg}")

# ── 2.8  Remove completely duplicated rows ────────────────────────────────────
rows_before_full_dup = len(df)
df.drop_duplicates(inplace=True)
rows_dropped_full_dup = rows_before_full_dup - len(df)
msg = f"Removed {rows_dropped_full_dup} fully duplicated row(s)."
cleaning_log.append(msg)
print(f"[2.8] {msg}")

# ── 2.9  Handle duplicate product_ids ────────────────────────────────────────
#
#   Inspection shows the same product_id can appear multiple times because
#   each row aggregates reviews from multiple reviewers into pipe-separated
#   fields (user_id, review_id, etc.).  Different rows with the same product_id
#   may represent different review batches for the same product.
#
#   Decision: keep only the FIRST occurrence per product_id.
#   Rationale: for market intelligence we need one record per product.
#              The first occurrence typically has the most complete product info.

dup_pid_before_dedup = df.duplicated(subset="product_id", keep=False).sum()
rows_before_pid_dedup = len(df)
df.drop_duplicates(subset="product_id", keep="first", inplace=True)
rows_dropped_pid_dedup = rows_before_pid_dedup - len(df)
msg = (f"Found {dup_pid_before_dedup} row(s) sharing duplicate product_ids. "
       f"Kept first occurrence per product_id; dropped {rows_dropped_pid_dedup} duplicate(s).")
cleaning_log.append(msg)
print(f"[2.9] {msg}")

# ── 2.10  Standardise category column ────────────────────────────────────────
#
#   Categories use pipe-separated hierarchy:
#     Computers&Accessories|Accessories&Peripherals|Cables&Accessories|...
#
#   We extract the top-level category (first segment) as a new clean column
#   'main_category'. This makes grouping and analysis easy without destroying
#   the original detailed path.

df["main_category"] = df["category"].str.split("|").str[0].str.strip()
msg = ("Extracted 'main_category' as a new column "
       "(first segment of pipe-separated category path).")
cleaning_log.append(msg)
print(f"[2.10] {msg}")

# ── 2.11  Reset index ─────────────────────────────────────────────────────────
df.reset_index(drop=True, inplace=True)
cleaning_log.append("Reset DataFrame index after all row drops.")

# ── Snapshot of dtypes after cleaning ────────────────────────────────────────
dtypes_after = df.dtypes.copy()
missing_after = df.isnull().sum()

final_rows = len(df)
rows_removed_total = orig_rows - final_rows

print(f"\n    ✔ Cleaning complete.")
print(f"    Original rows : {orig_rows}")
print(f"    Final rows    : {final_rows}")
print(f"    Rows removed  : {rows_removed_total}")


# ---------------------------------------------------------------------------
# TASK 3  :  SAVE OUTPUTS
# ---------------------------------------------------------------------------

print(section("TASK 3 — SAVING OUTPUTS"))

# ── Create output directories if they don't already exist ────────────────────
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR,   exist_ok=True)

# ── 3.1  Save cleaned CSV ────────────────────────────────────────────────────
df.to_csv(CLEANED_CSV, index=False, encoding="utf-8-sig")
print(f"\n[3.1] Cleaned dataset saved to:\n      {CLEANED_CSV}")

# ── 3.2  Write the cleaning report ──────────────────────────────────────────
report_lines = []

report_lines.append("=" * 70)
report_lines.append("  SMART HUNT — Module 4.1: Data Cleaning Report")
report_lines.append("=" * 70)

# --- Original shape ---
report_lines.append(section("1. ORIGINAL DATASET SHAPE"))
report_lines.append(f"  Rows    : {orig_rows}")
report_lines.append(f"  Columns : {orig_cols}")

# --- Missing values before cleaning ---
report_lines.append(section("2. MISSING VALUES — BEFORE CLEANING"))
report_lines.append(missing_before.to_string())

# --- Missing values after cleaning ---
report_lines.append(section("3. MISSING VALUES — AFTER CLEANING"))
report_lines.append(missing_after.to_string())
report_lines.append("")
report_lines.append("  NOTE: 'rating' and 'rating_count' may still contain NaN.")
report_lines.append("        These are legitimately unknown — not replaced with 0.")

# --- Duplicate rows ---
report_lines.append(section("4. DUPLICATE ROWS IDENTIFIED"))
report_lines.append(f"  Completely duplicated rows (before cleaning) : {full_dupes}")
report_lines.append(f"  Rows sharing a duplicate product_id          : {dup_pid_before_dedup}")
report_lines.append("")
report_lines.append("  Decision: Kept first occurrence per product_id.")
report_lines.append("  Reason  : Each product_id represents a unique product.")
report_lines.append("            Duplicate rows arise from multi-reviewer aggregation")
report_lines.append("            in the source data, not from true product duplicates.")

# --- Data types before ---
report_lines.append(section("5. DATA TYPES — BEFORE CLEANING"))
report_lines.append(dtypes_before.to_string())

# --- Data types after ---
report_lines.append(section("6. DATA TYPES — AFTER CLEANING"))
report_lines.append(dtypes_after.to_string())

# --- Cleaning operations ---
report_lines.append(section("7. CLEANING OPERATIONS PERFORMED"))
for i, op in enumerate(cleaning_log, 1):
    report_lines.append(f"  {i:2d}. {op}")

# --- Rows retained / removed ---
report_lines.append(section("8. ROWS RETAINED AND REMOVED"))
report_lines.append(f"  Original rows                              : {orig_rows}")
report_lines.append(f"  Rows dropped (missing essential fields)    : {rows_dropped_essential}")
report_lines.append(f"  Rows dropped (fully duplicated)            : {rows_dropped_full_dup}")
report_lines.append(f"  Rows dropped (duplicate product_id)        : {rows_dropped_pid_dedup}")
report_lines.append(f"  -----------------------------------------------")
report_lines.append(f"  Total rows removed                         : {rows_removed_total}")
report_lines.append(f"  Final rows retained                        : {final_rows}")
report_lines.append(f"  Final columns                              : {len(df.columns)}")

# --- Remaining data quality issues ---
report_lines.append(section("9. REMAINING DATA QUALITY ISSUES"))

remaining_issues = []

null_rating       = int(df["rating"].isna().sum())
null_rating_count = int(df["rating_count"].isna().sum())
null_actual_price = int(df["actual_price"].isna().sum())

if null_rating > 0:
    remaining_issues.append(
        f"  • 'rating' still has {null_rating} NaN value(s). "
        "No imputation applied — unknown ratings should stay unknown."
    )
if null_rating_count > 0:
    remaining_issues.append(
        f"  • 'rating_count' still has {null_rating_count} NaN value(s). "
        "Represents products with no recorded review count."
    )
if null_actual_price > 0:
    remaining_issues.append(
        f"  • 'actual_price' still has {null_actual_price} NaN value(s) "
        "after filling from discounted_price."
    )

for col in ["user_id", "user_name", "review_id", "review_title", "review_content"]:
    n_empty = int((df[col] == "").sum())
    if n_empty > 0:
        remaining_issues.append(
            f"  • '{col}' has {n_empty} empty string(s) "
            "(original records had no review data for these products)."
        )

if not remaining_issues:
    report_lines.append("  None identified beyond acceptable NaN values noted above.")
else:
    report_lines.extend(remaining_issues)

report_lines.append("")
report_lines.append("  COLUMNS NOT CLEANED (out of scope for Module 4.1):")
report_lines.append("  • about_product  — free text; no normalisation applied.")
report_lines.append("  • review_content — free text; no normalisation applied.")
report_lines.append("  • img_link / product_link — URLs; validity not checked.")
report_lines.append("  • user_id / user_name / review_id — not needed for")
report_lines.append("    market intelligence; preserved as-is.")

report_lines.append("\n" + "=" * 70)
report_lines.append("  END OF REPORT")
report_lines.append("=" * 70 + "\n")

# Write report file
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"[3.2] Cleaning report saved to:\n      {REPORT_FILE}")

# ---------------------------------------------------------------------------
# DONE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  All tasks complete.")
print(f"  Cleaned CSV  →  data/processed/cleaned_products.csv")
print(f"  Report       →  outputs/cleaning_report.txt")
print("=" * 70 + "\n")
