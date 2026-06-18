"""
Junction-level hotspot analysis (Page 3 completion item).

Aggregates named junctions only (excludes "No Junction" rows, which represent
mid-block violations with no junction context — about 49.5% of all records).
Priority level uses the same percentile-based logic as the DBSCAN cluster ML
Risk Score for consistency, but is clearly labeled as junction-level so it
isn't confused with the cluster-level Method 1 / Method 2 scores.
"""
import pandas as pd
import numpy as np

df = pd.read_parquet("/home/claude/violations_featured.parquet")
named = df[df["junction_name"].str.lower().str.strip() != "no junction"].copy()
print(f"Rows with named junction: {len(named)} of {len(df)} total "
      f"({len(named)/len(df)*100:.1f}%)")

agg = named.groupby("junction_name").agg(
    violation_count=("id", "count"),
    police_station=("police_station", lambda x: x.value_counts().index[0]),
    top_violation=("primary_violation", lambda x: x.value_counts().index[0]),
    top_vehicle=("vehicle_type", lambda x: x.value_counts().index[0]),
    peak_hour=("hour", lambda x: x.value_counts().index[0]),
    avg_vehicle_severity=("vehicle_severity", "mean"),
    avg_violation_severity=("violation_severity", "mean"),
    weekend_ratio=("is_weekend", "mean"),
    latitude=("latitude", "mean"),
    longitude=("longitude", "mean"),
).reset_index()

# Time concentration (peak hour share)
hour_max = named.groupby("junction_name")["hour"].agg(lambda x: x.value_counts().iloc[0])
hour_total = named.groupby("junction_name")["hour"].count()
agg["time_concentration"] = (hour_max / hour_total).values

# Same scoring philosophy as the DBSCAN cluster ML Risk Score: density-dominant
# log scale + percentile-ranked modifiers, kept consistent across the app.
# First pass at 60/25/15 weighting still let a 573-violation junction (high
# percentile severity/time) rank above an 11,538-violation junction (KR Market) —
# same density-distortion failure mode as the cluster-level score earlier.
# Raised density weight to 80% to keep raw volume decisive, consistent with
# how the cluster-level ML Risk Score was corrected.
log_density = np.log1p(agg["violation_count"])
density_score = (log_density - log_density.min()) / (log_density.max() - log_density.min()) * 100
severity_score = (agg["avg_violation_severity"] + agg["avg_vehicle_severity"]).rank(pct=True) * 100
time_score = agg["time_concentration"].rank(pct=True) * 100

agg["junction_risk_score"] = (
    density_score * 0.80 +
    severity_score * 0.12 +
    time_score * 0.08
).round(1)

def priority_band(score):
    if score >= 81: return "Critical"
    elif score >= 61: return "High"
    elif score >= 31: return "Medium"
    else: return "Low"

agg["priority_level"] = agg["junction_risk_score"].apply(priority_band)
agg = agg.sort_values("junction_risk_score", ascending=False).reset_index(drop=True)
agg["rank"] = agg.index + 1

# Clean junction display name (strip the "BTP051 - " code prefix for readability)
agg["junction_display"] = agg["junction_name"].str.replace(r"^BTP\d+\s*-\s*", "", regex=True)

print(f"\nTotal named junctions: {len(agg)}")
print(agg["priority_level"].value_counts())
print("\nTop 10 junctions by risk score:")
print(agg[["rank", "junction_display", "police_station", "violation_count",
           "top_violation", "peak_hour", "junction_risk_score", "priority_level"]].head(10).to_string())

agg.to_parquet("/home/claude/junction_analysis.parquet", index=False)
print("\nSaved junction_analysis.parquet")
