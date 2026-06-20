# SmartPark AI —  Parking Enforcement Intelligence System

AI-driven parking violation analytics for targeted enforcement — built on 298,277
real parking violation records from Bengaluru (Nov 2023 – Apr 2024).

Built for a Flipkart hackathon, Problem Statement 1: Parking-Induced Congestion.

**[Live demo →](#)** *(https://smartpark-ai-v2vwwgxy69ekdafx8abdkt.streamlit.app/)*

---

## What this app does

Ten pages, two independent scoring methods, and a forecasting model — all built
and validated on a single real dataset (no synthetic or placeholder data).

| Page | What it shows |
|---|---|
| 📊 Overview | KPIs, violation/vehicle breakdowns, daily trend |
| 🗺️ Hotspot Map | **Method 2** — DBSCAN clusters + Isolation Forest ML Risk Score, bubble map + density heatmap toggle |
| 🏆 Heuristic Priority Zones | **Method 1** — grid-based weighted scoring (built first, kept for comparison) |
| 🚦 Junction Analysis | Top 168 named junctions ranked by risk, e.g. KR Market Junction |
| ⏰ Time Analysis | Hour/day/weekday violation patterns |
| 🚗 Vehicle Breakdown | Vehicle type analysis, repeat offenders |
| 🔍 Zone Deep Dive | Drill into any hotspot zone, plain-English enforcement recommendation |
| 📈 Forecasting | 7-day city-wide violation forecast (Ridge regression), allocated to top stations |
| 💡 Recommendations | Top 5 priority clusters with concrete deploy/time/reason cards |
| 🧠 How the ML Works | Full transparent methodology, including what was tried and rejected |

## Why two scoring methods (Method 1 and Method 2)

**Method 1 (Heuristic Priority Zones)** was built first: ~300m grid cells,
weighted formula (frequency, vehicle severity, violation severity, time
concentration, junction proximity), scored 0–10 by design (observed scores
range 0.86–5.86 across the 661 zones found).

**Method 2 (ML Risk Score)** was built second: DBSCAN clustering on a
stratified sample, Isolation Forest anomaly detection on cluster-level
features, score range 0–100. This is the more technically rigorous approach
and is the one referenced throughout the Recommendations, Forecasting, and
Junction Analysis pages.

Both are kept in the app and clearly labeled (color-coded badges on each
page) rather than merged into one number, since they measure things
differently and a fabricated unified scale would overstate how comparable
they actually are.

## Key validated finding

The top 5 zones identified by Method 2 — with no manual input, no hardcoded
landmark list — landed on real, independently recognizable Bengaluru
congestion points: Majestic Transit Hub (metro + 2 bus terminals), Commercial
Street (×2), and K R Market. Verified directly against Google Maps.

**Headline stat:** the top 50 hotspot zones (7.6% of all 661 zones) account
for roughly 40% of all recorded violations.

## What's *not* in this prototype

- **Event/Incident dataset** — evaluated and excluded. The `end_datetime`
  field was 94% null, making event duration and impact impossible to compute
  credibly. Rather than force a weak integration, this dataset was dropped
  entirely and the scope kept to what the violation dataset could support.
- **Real-time camera/sensor ingestion** — this is a batch analysis on a
  historical CSV, not a live system. See the in-app "Path to Production"
  framing for the realistic path from here to real-time.

## Repo contents

```
app.py                       Main Streamlit application (10 pages)
pipeline.py                  Original data cleaning + grid-based hotspot pipeline (Method 1)
dbscan_clustering.py         DBSCAN clustering on a stratified 80K-row sample (Method 2, stage 1)
ml_risk_model.py             Isolation Forest + ML Risk Score computation (Method 2, stage 2)
junction_analysis.py         Junction-level hotspot aggregation and risk scoring
forecasting_model.py         City-wide Ridge regression forecasting model + validation

violations_featured.parquet  Cleaned + feature-engineered violation data (298,277 rows)
hotspots.parquet             661 grid-based hotspot zones (Method 1 output)
dbscan_clusters_ml.parquet   305 DBSCAN clusters with ML Risk Scores (Method 2 output)
junction_analysis.parquet    168 named junctions ranked by risk
city_forecast.parquet        7-day city-wide violation forecast
station_forecast.parquet     City forecast allocated to top 10 stations
forecast_test_results.parquet  Held-out 21-day validation: actual vs. predicted
forecast_metrics.json        Model performance metrics (MAE, MAPE, baseline comparison)

requirements.txt             Python dependencies
```

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

All `.parquet` and `.json` data files must be in the same folder as `app.py` —
the app loads them with relative paths.

## How to deploy to Streamlit Cloud

1. Push this entire folder to a GitHub repo (the parquet files are small
   enough to commit directly — largest is ~11MB)
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   set the main file to `app.py`
3. Streamlit Cloud installs `requirements.txt` automatically on deploy

## Regenerating the data from scratch

If starting from the raw violation CSV instead of the included parquet files,
run the pipeline scripts in this order:

```bash
python pipeline.py              # cleans raw CSV → violations_featured.parquet, hotspots.parquet
python dbscan_clustering.py     # → dbscan_sample.parquet
python ml_risk_model.py         # → dbscan_clusters_ml.parquet
python junction_analysis.py     # → junction_analysis.parquet
python forecasting_model.py     # → city_forecast.parquet, station_forecast.parquet, etc.
```

Each script prints its own validation output (cluster counts, model MAE vs.
baseline, etc.) so you can confirm the regenerated numbers match the
methodology section below before trusting them.

## Methodology in detail

### Method 1 — Heuristic Priority Zones (grid-based)
~300m grid cells, zones with 50+ violations flagged as hotspots (661 zones found).

```
Priority Score = 0.40×frequency + 0.20×vehicle severity + 0.20×violation severity
               + 0.10×time concentration + 0.10×junction proximity
```

### Method 2 — ML Risk Score (DBSCAN + Isolation Forest)
- **DBSCAN:** run on an 80,000-row stratified sample (a full-dataset run on
  all 298K rows exceeded available memory in development). 100m radius,
  minimum 30 points per cluster → 305 real clusters, 21.9% noise.
- **Isolation Forest:** trained on cluster-level features (violation count,
  peak-hour ratio, violation-type mix, vehicle diversity) to flag abnormally
  severe clusters, since the dataset has no "normal, non-violating" baseline
  to classify against directly.
- **Final score:** density-dominant weighting (55% log-scaled violation
  count, 20% anomaly score, 15% severity, 10% time concentration), 0–100
  scale. Density was deliberately weighted heavily after an earlier
  equal-weighted version let a low-volume, high-anomaly cluster outrank the
  city's largest, already-validated hotspot — corrected in favor of what
  would actually matter for real enforcement deployment.

### Forecasting — what worked and what didn't
Three granularities were tested honestly; two were rejected before shipping
the third:
- **Grid-zone level:** rejected — median zone had recorded activity on only
  40 of 150 possible days, too sparse for a daily time series.
- **Police-station level:** rejected — coefficient of variation 0.85 (daily
  swings nearly as large as the average itself); a tested model lost to a
  naive 7-day-average baseline.
- **City-wide level:** used — coefficient of variation 0.19, genuinely
  forecastable. XGBoost was tried here first and *also* lost to the naive
  baseline (likely overfitting on only 124 training days). A simpler,
  regularized **Ridge regression** on the same features won, beating the
  naive baseline by +9.0% on a held-out 21-day test period (MAE 267/day,
  MAPE 16.3%).

## Known limitations (stated upfront, not discovered)

- No direct traffic speed or flow data exists in the source dataset.
  Congestion impact is modeled as a violation-density proxy, not a directly
  measured metric.
- The forecasting model's improvement over baseline is real but modest
  (+9%), not dramatic — presented honestly rather than oversold.
- DBSCAN and Isolation Forest run on an 80K-row sample, not the full 298K
  rows, due to memory constraints in the development environment.

## Dataset

`jan_to_may_police_violation_anonymized.csv` — Bengaluru parking violation
records, Nov 2023–Apr 2024, 298,450 raw rows / 298,277 after cleaning.
Anonymized vehicle identifiers throughout.
