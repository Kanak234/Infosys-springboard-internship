"""
HealthForecast AI — Milestone 1
Data cleaning pipeline for the UCI Diabetes 130-US Hospitals dataset.

Input : diabetic_data.csv, IDS_mapping.csv
Output: diabetic_data_clean.csv  (model-ready + DB-seed ready)
        cleaning_report.txt      (what happened at each step)
        data_dictionary.csv      (column reference for the backend team)

Cleaning follows the methodology of Strack et al. (2014), the paper the
dataset ships with.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import io

UPLOADS = Path("/mnt/user-data/uploads")
OUT = Path("/mnt/user-data/outputs")
OUT.mkdir(parents=True, exist_ok=True)

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
# keep_default_na=False so the literal string "None" in max_glu_serum /
# A1Cresult is preserved - it means "test was not performed", not missing data.
df = pd.read_csv(
    UPLOADS / "diabetic_data.csv",
    keep_default_na=False,
    na_values=["?", "Unknown/Invalid", ""],
)
log("=" * 70)
log("HEALTHFORECAST AI - MILESTONE 1 CLEANING REPORT")
log("=" * 70)
log(f"\n[1] LOADED")
log(f"    Rows: {len(df):,}   Columns: {df.shape[1]}")
log(f"    Unique patients: {df['patient_nbr'].nunique():,}")

start_rows = len(df)

# ----------------------------------------------------------------------
# 2. MISSING VALUE AUDIT
# ----------------------------------------------------------------------
miss = (df.isna().mean() * 100).round(2)
miss = miss[miss > 0].sort_values(ascending=False)
log(f"\n[2] MISSING VALUES (columns with any missing)")
for col, pct in miss.items():
    log(f"    {col:<25} {pct:>6.2f}%")

# ----------------------------------------------------------------------
# 3. DROP UNUSABLE COLUMNS
# ----------------------------------------------------------------------
drop_cols = []

# weight: 97% missing -> unusable
drop_cols.append("weight")
# payer_code: 52% missing and not clinically relevant to readmission
drop_cols.append("payer_code")

# constant columns (single value across the whole dataset -> zero information)
constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
drop_cols += constant_cols

df = df.drop(columns=drop_cols)
log(f"\n[3] DROPPED COLUMNS ({len(drop_cols)})")
log(f"    weight        - 96.86% missing")
log(f"    payer_code    - 52.00% missing, not outcome-relevant")
for c in constant_cols:
    log(f"    {c:<13} - constant value, no information")

# ----------------------------------------------------------------------
# 4. FILL REMAINING MISSING VALUES
# ----------------------------------------------------------------------
log(f"\n[4] FILLING MISSING VALUES")

# medical_specialty: 53% missing but predictive -> explicit 'Missing' category
df["medical_specialty"] = df["medical_specialty"].fillna("Missing")
log(f"    medical_specialty -> 'Missing' category (53% blank, still predictive)")

# race: 2% missing -> explicit category (do not guess ethnicity)
df["race"] = df["race"].fillna("Missing")
log(f"    race              -> 'Missing' category (2% blank)")

# diag_1/2/3: small % missing -> 'Missing' before grouping
for c in ["diag_1", "diag_2", "diag_3"]:
    df[c] = df[c].fillna("Missing")
log(f"    diag_1/2/3        -> 'Missing' before ICD9 grouping")

# gender: 3 invalid rows -> drop them
before = len(df)
df = df.dropna(subset=["gender"])
log(f"    gender            -> dropped {before - len(df)} rows with invalid gender")

# ----------------------------------------------------------------------
# 5. REMOVE DEATH / HOSPICE ENCOUNTERS
# ----------------------------------------------------------------------
# These patients cannot be readmitted, so they would bias the model.
DEAD_OR_HOSPICE = [11, 13, 14, 19, 20, 21]
before = len(df)
df = df[~df["discharge_disposition_id"].isin(DEAD_OR_HOSPICE)]
log(f"\n[5] REMOVED DEATH / HOSPICE DISCHARGES")
log(f"    discharge_disposition_id in {DEAD_OR_HOSPICE}")
log(f"    Removed {before - len(df):,} rows (they cannot be readmitted)")

# ----------------------------------------------------------------------
# 6. ONE ENCOUNTER PER PATIENT
# ----------------------------------------------------------------------
# Multiple visits by the same patient are not statistically independent.
before = len(df)
df = df.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")
log(f"\n[6] KEPT FIRST ENCOUNTER PER PATIENT")
log(f"    Removed {before - len(df):,} repeat encounters")
log(f"    Remaining: {len(df):,} independent patient records")

# ----------------------------------------------------------------------
# 7. DECODE ID COLUMNS USING IDS_mapping.csv
# ----------------------------------------------------------------------
raw = (UPLOADS / "IDS_mapping.csv").read_text()
blocks = [b.strip() for b in raw.split(",\n") if b.strip()]

mappings = {}
for block in blocks:
    tbl = pd.read_csv(io.StringIO(block))
    key = tbl.columns[0]
    mappings[key] = dict(zip(tbl[key], tbl["description"]))

log(f"\n[7] DECODED ID COLUMNS")
for id_col, name in [
    ("admission_type_id", "admission_type"),
    ("discharge_disposition_id", "discharge_disposition"),
    ("admission_source_id", "admission_source"),
]:
    df[name] = df[id_col].map(mappings[id_col]).fillna("Not Available")
    df[name] = df[name].replace({"NULL": "Not Available", "Not Mapped": "Not Available"})
    log(f"    {id_col} -> {name} ({df[name].nunique()} categories)")

# ----------------------------------------------------------------------
# 8. GROUP ICD9 DIAGNOSIS CODES
# ----------------------------------------------------------------------
# 848 raw codes are useless to a model. Group them the way the paper does.
def group_icd9(code):
    if code == "Missing":
        return "Missing"
    c = str(code)
    if c.startswith("V") or c.startswith("E"):
        return "Other"
    try:
        n = float(c)
    except ValueError:
        return "Other"
    if 390 <= n <= 459 or int(n) == 785:
        return "Circulatory"
    if 460 <= n <= 519 or int(n) == 786:
        return "Respiratory"
    if 520 <= n <= 579 or int(n) == 787:
        return "Digestive"
    if 250 <= n < 251:
        return "Diabetes"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if 580 <= n <= 629 or int(n) == 788:
        return "Genitourinary"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"


for c in ["diag_1", "diag_2", "diag_3"]:
    df[c + "_group"] = df[c].apply(group_icd9)

log(f"\n[8] GROUPED ICD9 DIAGNOSIS CODES")
log(f"    diag_1: {df['diag_1'].nunique()} raw codes -> {df['diag_1_group'].nunique()} groups")
log(f"    Primary diagnosis distribution:")
for g, n in df["diag_1_group"].value_counts().items():
    log(f"        {g:<18} {n:>7,}  ({n/len(df)*100:>5.1f}%)")

# ----------------------------------------------------------------------
# 9. AGE -> NUMERIC MIDPOINT
# ----------------------------------------------------------------------
age_mid = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}
df["age_numeric"] = df["age"].map(age_mid)
# also the 3-band split the paper found meaningful
df["age_group"] = pd.cut(
    df["age_numeric"], bins=[0, 30, 60, 100],
    labels=["<30", "30-60", "60+"], right=False,
)
log(f"\n[9] AGE CONVERTED")
log(f"    age bracket -> age_numeric (midpoint) + age_group (<30 / 30-60 / 60+)")
log(f"    Mean age: {df['age_numeric'].mean():.1f} years")

# ----------------------------------------------------------------------
# 10. TARGET VARIABLE
# ----------------------------------------------------------------------
df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
pos = int(df["readmitted_30d"].sum())
log(f"\n[10] TARGET VARIABLE CREATED")
log(f"    readmitted_30d = 1 if readmitted within 30 days, else 0")
log(f"    Positive class: {pos:,} ({pos/len(df)*100:.2f}%)")
log(f"    Negative class: {len(df)-pos:,} ({(len(df)-pos)/len(df)*100:.2f}%)")
log(f"    NOTE: imbalanced - use class_weight / SMOTE at model stage.")

# ----------------------------------------------------------------------
# 11. ENGINEERED FEATURES
# ----------------------------------------------------------------------
df["total_prior_visits"] = (
    df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
)

DOSAGE_VALUES = {"No", "Up", "Down", "Steady"}
med_cols = [
    c for c in df.columns
    if not pd.api.types.is_numeric_dtype(df[c])
    and set(map(str, df[c].dropna().unique())) <= DOSAGE_VALUES
    and df[c].nunique() > 1
]
df["num_med_changes"] = df[med_cols].isin(["Up", "Down"]).sum(axis=1)
df["num_meds_prescribed"] = df[med_cols].ne("No").sum(axis=1)

log(f"\n[11] ENGINEERED FEATURES")
log(f"    total_prior_visits   = outpatient + emergency + inpatient")
log(f"    num_med_changes      = count of drugs with dosage Up/Down ({len(med_cols)} drug cols)")
log(f"    num_meds_prescribed  = count of drugs actually prescribed")

# ----------------------------------------------------------------------
# 12. FINAL TIDY
# ----------------------------------------------------------------------
df = df.drop(columns=["admission_type_id", "discharge_disposition_id", "admission_source_id"])

lead = ["encounter_id", "patient_nbr", "race", "gender", "age", "age_numeric", "age_group"]
tail = ["readmitted", "readmitted_30d"]
mid = [c for c in df.columns if c not in lead + tail]
df = df[lead + mid + tail]

assert df.isna().sum().sum() == 0, "missing values remain"
assert df["patient_nbr"].is_unique, "duplicate patients remain"

out_csv = OUT / "diabetic_data_clean.csv"
df.to_csv(out_csv, index=False)

log(f"\n[12] FINAL DATASET")
log(f"    Rows:    {len(df):,}  (from {start_rows:,}, removed {start_rows-len(df):,})")
log(f"    Columns: {df.shape[1]}")
log(f"    Missing values: 0")
log(f"    Duplicate patients: 0")
log(f"    Saved: diabetic_data_clean.csv ({out_csv.stat().st_size/1e6:.1f} MB)")

# ----------------------------------------------------------------------
# 13. DATA DICTIONARY FOR THE BACKEND TEAM
# ----------------------------------------------------------------------
PG = {"int64": "INTEGER", "float64": "NUMERIC", "object": "VARCHAR", "category": "VARCHAR"}
rows = []
for c in df.columns:
    dt = str(df[c].dtype)
    rows.append({
        "column": c,
        "pandas_dtype": dt,
        "postgres_type": PG.get(dt, "VARCHAR"),
        "distinct_values": df[c].nunique(),
        "example": str(df[c].iloc[0]),
    })
pd.DataFrame(rows).to_csv(OUT / "data_dictionary.csv", index=False)
log(f"\n[13] data_dictionary.csv written - column types for the PostgreSQL schema")

(OUT / "cleaning_report.txt").write_text("\n".join(log_lines))
print("\nDone.")
