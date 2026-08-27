"""EDA on the cleaned dataset - produces eda_summary.png + printed stats."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/mnt/user-data/outputs")
df = pd.read_csv(OUT / "diabetic_data_clean.csv")

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("HealthForecast AI - Milestone 1 EDA (69,987 patients)",
             fontsize=15, fontweight="bold")

# 1 target balance
c = df["readmitted_30d"].value_counts().sort_index()
ax[0, 0].bar(["Not readmitted", "Readmitted <30d"], c.values,
             color=["#7fb3d5", "#e59866"])
ax[0, 0].set_title("Target balance")
for i, v in enumerate(c.values):
    ax[0, 0].text(i, v, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom")
ax[0, 0].set_ylim(0, max(c.values) * 1.2)

# 2 readmission by age group
g = df.groupby("age_group", observed=True)["readmitted_30d"].mean() * 100
ax[0, 1].bar(g.index.astype(str), g.values, color="#a9cce3")
ax[0, 1].set_title("Readmission rate by age group")
ax[0, 1].set_ylabel("% readmitted <30d")

# 3 readmission by primary diagnosis
g = (df.groupby("diag_1_group")["readmitted_30d"].mean() * 100).sort_values()
ax[0, 2].barh(g.index, g.values, color="#a2d9ce")
ax[0, 2].set_title("Readmission rate by primary diagnosis")
ax[0, 2].set_xlabel("% readmitted <30d")

# 4 time in hospital
ax[1, 0].hist(df["time_in_hospital"], bins=14, color="#d7bde2", edgecolor="white")
ax[1, 0].set_title("Length of stay (days)")

# 5 prior inpatient visits vs readmission
g = df.groupby(df["number_inpatient"].clip(upper=5))["readmitted_30d"].mean() * 100
ax[1, 1].plot(g.index, g.values, marker="o", color="#cd6155")
ax[1, 1].set_title("Prior inpatient visits vs readmission")
ax[1, 1].set_xlabel("prior inpatient visits (5 = 5 or more)")
ax[1, 1].set_ylabel("% readmitted <30d")

# 6 A1C test vs readmission
g = df.groupby("A1Cresult")["readmitted_30d"].mean() * 100
ax[1, 2].bar(g.index.astype(str), g.values, color="#f5b7b1")
ax[1, 2].set_title("HbA1c test result vs readmission")
ax[1, 2].set_ylabel("% readmitted <30d")

plt.tight_layout()
plt.savefig(OUT / "eda_summary.png", dpi=130, bbox_inches="tight")

print("Top signals found in EDA")
print("-" * 55)
for col in ["number_inpatient", "number_emergency", "time_in_hospital",
            "num_medications", "number_diagnoses", "age_numeric",
            "total_prior_visits", "num_med_changes"]:
    r = df[col].corr(df["readmitted_30d"])
    print(f"  {col:<22} corr with target: {r:+.3f}")

print("\nReadmission rate by discharge type (top 5 by volume)")
top = df["discharge_disposition"].value_counts().head(5).index
for d in top:
    s = df[df["discharge_disposition"] == d]
    print(f"  {d[:45]:<47} {s['readmitted_30d'].mean()*100:5.1f}%  (n={len(s):,})")
