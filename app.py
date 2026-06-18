import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import ast
import json
from datetime import datetime

st.set_page_config(
    page_title="SmartPark Enforcement Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border-radius: 12px; padding: 1rem 1.2rem;
        border-left: 4px solid #e94560; color: white;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-label { font-size: 0.8rem; color: #aaa; text-transform: uppercase; }
    .priority-high { color: #e74c3c; font-weight: 700; }
    .priority-med  { color: #f39c12; font-weight: 700; }
    .priority-low  { color: #2ecc71; font-weight: 700; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        border-bottom: 2px solid #e94560;
        padding-bottom: 4px; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet("violations_featured.parquet")
    hotspots = pd.read_parquet("hotspots.parquet")

    # Parse violation lists for display
    def parse_violations(x):
        try:
            return ", ".join(ast.literal_eval(x))
        except:
            return str(x)

    df["violations_display"] = df["violation_type"].apply(parse_violations)

    # Priority label
    def priority_label(score):
        if score >= 4: return "🔴 Critical"
        elif score >= 2.5: return "🟡 High"
        else: return "🟢 Moderate"

    hotspots["priority_label"] = hotspots["priority_score"].apply(priority_label)
    hotspots["top_time_period"] = hotspots["peak_hour"].apply(
        lambda h: f"{h:02d}:00–{(h+1):02d}:00"
    )

    # DBSCAN cluster table with ML Risk Score (Isolation Forest based)
    ml_clusters = pd.read_parquet("dbscan_clusters_ml.parquet")

    # Light sample of raw points for the density heatmap layer (pydeck HeatmapLayer
    # does its own client-side aggregation, so it doesn't need every one of 298K rows —
    # a representative sample keeps the map responsive without losing the density pattern)
    heatmap_sample = df.sample(n=min(40000, len(df)), random_state=42)[["latitude", "longitude"]]

    # Forecasting model artifacts
    city_forecast = pd.read_parquet("city_forecast.parquet")
    station_forecast = pd.read_parquet("station_forecast.parquet")
    test_results = pd.read_parquet("forecast_test_results.parquet")
    with open("forecast_metrics.json") as f:
        forecast_metrics = json.load(f)

    # Junction-level hotspot analysis (Page 3 completion item)
    junctions = pd.read_parquet("junction_analysis.parquet")

    return (df, hotspots, ml_clusters, heatmap_sample, city_forecast, station_forecast,
            test_results, forecast_metrics, junctions)

(df, hotspots, ml_clusters, df_for_heatmap, city_forecast, station_forecast,
 forecast_test_results, forecast_metrics, junctions) = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=60)
    st.markdown("## SmartPark AI")
    st.markdown("*Parking Enforcement Intelligence System*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🗺️ Hotspot Map", "🏆 Priority Zones", "🚦 Junction Analysis",
         "⏰ Time Analysis", "🚗 Vehicle Breakdown", "🔍 Zone Deep Dive", "📈 Forecasting",
         "💡 Recommendations", "🧠 How the ML Works"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Filters")

    police_stations = ["All"] + sorted(df["police_station"].dropna().unique().tolist())
    selected_station = st.selectbox("Police Station", police_stations)

    violation_types = ["All"] + sorted(df["primary_violation"].dropna().unique().tolist())
    selected_violation = st.selectbox("Violation Type", violation_types)

    time_slots = ["All"] + df["time_slot"].dropna().unique().tolist()
    selected_slot = st.selectbox("Time Slot", time_slots)

    st.markdown("---")
    st.markdown(f"**Dataset:** Bengaluru, Nov 2023 – Apr 2024")
    st.markdown(f"**Total Records:** {len(df):,}")
    st.markdown(f"**Hotspot Zones:** {len(hotspots):,}")

# ── Filter application ────────────────────────────────────────────────────────
filtered_df = df.copy()
filtered_hotspots = hotspots.copy()

if selected_station != "All":
    filtered_df = filtered_df[filtered_df["police_station"] == selected_station]
    filtered_hotspots = filtered_hotspots[filtered_hotspots["top_police_station"] == selected_station]

if selected_violation != "All":
    filtered_df = filtered_df[filtered_df["primary_violation"] == selected_violation]

if selected_slot != "All":
    filtered_df = filtered_df[filtered_df["time_slot"] == selected_slot]

if len(filtered_df) == 0:
    st.warning("⚠️ No violations match this filter combination. Try a broader selection (e.g. set Police Station or Time Slot back to 'All').")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<p class="main-header">🚦 SmartPark Enforcement Intelligence</p>', unsafe_allow_html=True)
    st.markdown("AI-driven parking violation analytics for targeted enforcement in Bengaluru")
    st.markdown("---")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(filtered_df):,}</div>
            <div class="metric-label">Total Violations</div></div>""", unsafe_allow_html=True)
    with col2:
        critical = len(filtered_hotspots[filtered_hotspots["priority_score"] >= 4])
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{critical}</div>
            <div class="metric-label">Critical Zones</div></div>""", unsafe_allow_html=True)
    with col3:
        unique_v = filtered_df["vehicle_number"].nunique()
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{unique_v:,}</div>
            <div class="metric-label">Unique Vehicles</div></div>""", unsafe_allow_html=True)
    with col4:
        top_station = filtered_df["police_station"].value_counts().index[0] if len(filtered_df) > 0 else "N/A"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="font-size:1rem">{top_station}</div>
            <div class="metric-label">Busiest Station</div></div>""", unsafe_allow_html=True)
    with col5:
        peak_h = int(filtered_df["hour"].mode()[0]) if len(filtered_df) > 0 else 0
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{peak_h:02d}:00</div>
            <div class="metric-label">Peak Violation Hour</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Violations by Type</p>', unsafe_allow_html=True)
        vtype_counts = filtered_df["primary_violation"].value_counts().head(10).reset_index()
        vtype_counts.columns = ["Violation Type", "Count"]
        fig = px.bar(vtype_counts, x="Count", y="Violation Type", orientation="h",
                     color="Count", color_continuous_scale="Reds",
                     template="plotly_dark")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                         yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Violations by Vehicle Type</p>', unsafe_allow_html=True)
        veh_counts = filtered_df["vehicle_type"].value_counts().head(8).reset_index()
        veh_counts.columns = ["Vehicle Type", "Count"]
        fig = px.pie(veh_counts, values="Count", names="Vehicle Type",
                     color_discrete_sequence=px.colors.sequential.RdBu,
                     template="plotly_dark", hole=0.4)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                         paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Daily Violation Trend</p>', unsafe_allow_html=True)
        daily = filtered_df.groupby(filtered_df["created_datetime"].dt.date).size().reset_index()
        daily.columns = ["Date", "Violations"]
        fig = px.line(daily, x="Date", y="Violations",
                      template="plotly_dark", color_discrete_sequence=["#e94560"])
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Top Police Stations by Violations</p>', unsafe_allow_html=True)
        ps_counts = filtered_df["police_station"].value_counts().head(10).reset_index()
        ps_counts.columns = ["Station", "Violations"]
        fig = px.bar(ps_counts, x="Station", y="Violations",
                     color="Violations", color_continuous_scale="Reds",
                     template="plotly_dark")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                         xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOTSPOT MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Hotspot Map":
    st.markdown('<p class="main-header">🗺️ AI Hotspot Map</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: inline-block; background: #064663; color: #5EC4D6; padding: 4px 14px;
                border-radius: 20px; font-weight: 700; font-size: 0.85rem; margin-bottom: 10px;">
        METHOD 2 — ML-BASED RISK RANKING &nbsp;·&nbsp; Score range 0–100
    </div>
    """, unsafe_allow_html=True)
    st.markdown("DBSCAN clusters violations into real geographic hotspots, then Isolation Forest flags "
                 "abnormally severe clusters. See the *Priority Zones* page for **Method 1** — our earlier "
                 "grid-based heuristic scoring (score range 0–10). These are two deliberately different, "
                 "independently built approaches — not the same score on two different scales.")

    map_view = st.radio(
        "Map view", ["Bubble Hotspot Map", "Density Heatmap"],
        horizontal=True,
        help="Bubble map shows individual DBSCAN clusters with click-to-inspect detail. Heatmap shows raw violation density."
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        top_n = st.slider("Show top N hotspot clusters", 10, len(ml_clusters), min(80, len(ml_clusters)))
    with col_b:
        priority_filter = st.multiselect(
            "Priority filter", ["Critical Priority", "High Priority", "Medium Priority", "Low Priority"],
            default=["Critical Priority", "High Priority", "Medium Priority", "Low Priority"]
        )

    map_clusters = ml_clusters[ml_clusters["priority_level"].isin(priority_filter)].nlargest(top_n, "ml_risk_score")

    PRIORITY_COLORS = {
        "Critical Priority": [231, 76, 60, 200],
        "High Priority": [243, 156, 18, 190],
        "Medium Priority": [241, 196, 15, 170],
        "Low Priority": [46, 204, 113, 150],
    }
    map_clusters = map_clusters.copy()
    map_clusters["color"] = map_clusters["priority_level"].map(PRIORITY_COLORS)
    map_clusters["radius"] = (map_clusters["violation_count"] ** 0.5) * 12

    if map_view == "Bubble Hotspot Map":
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_clusters,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            opacity=0.75,
            stroked=True,
            get_line_color=[255, 255, 255, 80],
            line_width_min_pixels=1,
        )
        tooltip = {
            "html": """
            <b>{cluster_id}</b><br/>
            Police Station: {top_police_station}<br/>
            Junction: {top_junction}<br/>
            Total Violations: {violation_count}<br/>
            Top Violation: {top_violation}<br/>
            Peak Hour: {peak_hour}:00<br/>
            ML Risk Score: {ml_risk_score}/100<br/>
            Priority: {priority_level}
            """,
            "style": {"backgroundColor": "#16213e", "color": "white", "fontSize": "13px"}
        }
    else:
        layer = pdk.Layer(
            "HeatmapLayer",
            data=df_for_heatmap,
            get_position=["longitude", "latitude"],
            opacity=0.7,
            aggregation="SUM",
            get_weight=1,
            radiusPixels=45,
        )
        tooltip = None

    view_state = pdk.ViewState(latitude=12.97, longitude=77.59, zoom=11.5, pitch=0)
    deck = pdk.Deck(
        layers=[layer], initial_view_state=view_state, tooltip=tooltip,
        # 'road' is one of pydeck's built-in named styles, served via CARTO with no API
        # token required, and shows street names/area labels more clearly than the prior
        # mapbox:// style URL (which silently renders near-blank without a Mapbox token —
        # the root cause of the "too plain/black" map feedback).
        map_style=pdk.map_styles.ROAD,
    )
    st.pydeck_chart(deck, use_container_width=True, height=600)

    if map_view == "Bubble Hotspot Map":
        st.markdown("**Circle size** = number of violations &nbsp;|&nbsp; **Color** = ML Risk priority &nbsp; "
                     "🔴 Critical &nbsp; 🟠 High &nbsp; 🟡 Medium &nbsp; 🟢 Low &nbsp;|&nbsp; "
                     "**Click any bubble** for full cluster detail &nbsp;|&nbsp; Bengaluru city center")
    else:
        st.markdown("**Glow intensity** = density of recorded violations in that area")

    st.markdown("---")
    st.markdown('<p class="section-header">Cluster Detail Lookup</p>', unsafe_allow_html=True)
    st.caption("pydeck tooltips work on hover/click on the map above. Use this table for a reliable, scrollable view of the same detail.")

    detail_cols = ["risk_rank", "cluster_id", "top_police_station", "top_junction",
                   "violation_count", "top_violation", "peak_hour", "ml_risk_score", "priority_level"]
    detail_display = map_clusters[detail_cols].sort_values("risk_rank").reset_index(drop=True)
    detail_display.columns = ["Rank", "Cluster ID", "Police Station", "Junction", "Violations",
                                "Top Violation", "Peak Hour", "ML Risk Score", "Priority"]
    st.dataframe(detail_display, use_container_width=True, height=300)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRIORITY ZONES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Priority Zones":
    st.markdown('<p class="main-header">🏆 Enforcement Priority Zones</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: inline-block; background: #064663; color: #5EC4D6; padding: 4px 14px;
                border-radius: 20px; font-weight: 700; font-size: 0.85rem; margin-bottom: 10px;">
        METHOD 1 — HEURISTIC SCORING &nbsp;·&nbsp; Score range 0–10
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Ranked hotspots using a weighted formula "
                 "(frequency, vehicle severity, violation severity, time concentration, junction proximity). "
                 "See the *Hotspot Map* page for **Method 2** — our DBSCAN + Isolation Forest ML-based ranking "
                 "(score range 0–100). These are two deliberately different, independently built approaches — "
                 "not the same score on two different scales.")

    col1, col2, col3 = st.columns(3)
    with col1:
        critical_zones = filtered_hotspots[filtered_hotspots["priority_score"] >= 4]
        st.metric("🔴 Critical Zones", len(critical_zones))
    with col2:
        high_zones = filtered_hotspots[filtered_hotspots["priority_score"].between(2.5, 4)]
        st.metric("🟡 High Priority", len(high_zones))
    with col3:
        mod_zones = filtered_hotspots[filtered_hotspots["priority_score"] < 2.5]
        st.metric("🟢 Moderate", len(mod_zones))

    st.markdown("---")

    top_n = st.slider("Show top N zones", 10, 100, 25)
    display = filtered_hotspots.nlargest(top_n, "priority_score")[[
        "priority_rank", "priority_label", "priority_score",
        "violation_count", "top_violation", "top_vehicle",
        "top_police_station", "top_time_period", "weekend_ratio",
        "latitude", "longitude"
    ]].copy()

    display["weekend_ratio"] = (display["weekend_ratio"] * 100).round(1).astype(str) + "%"
    display.columns = [
        "Rank", "Priority", "Score", "Violations",
        "Top Violation", "Top Vehicle", "Police Station",
        "Peak Time", "Weekend %", "Lat", "Lon"
    ]

    def color_priority(val):
        if "Critical" in str(val): return "color: #e74c3c; font-weight: bold"
        elif "High" in str(val): return "color: #f39c12; font-weight: bold"
        return "color: #2ecc71"

    try:
        styled = display.style.map(color_priority, subset=["Priority"])
    except AttributeError:
        styled = display.style.applymap(color_priority, subset=["Priority"])

    st.dataframe(styled, use_container_width=True, height=500)

    st.markdown("---")
    st.markdown("### Priority Score Formula")
    st.latex(r"""
    \text{Priority Score} = 0.40 \times \text{Freq Score} + 0.20 \times \text{Vehicle Severity}
    + 0.20 \times \text{Violation Severity} + 0.10 \times \text{Time Concentration}
    + 0.10 \times \text{Junction Flag}
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: JUNCTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚦 Junction Analysis":
    st.markdown('<p class="main-header">🚦 Top Junctions by Parking Violations</p>', unsafe_allow_html=True)
    st.markdown(
        "168 named junctions identified, covering **50.5%** of all violations. "
        "The remaining 49.5% occur away from a named junction (mid-block parking) and are excluded here "
        "since they have no junction context to analyze."
    )
    st.caption(
        "Risk scoring uses the same density-dominant methodology as the DBSCAN cluster ML Risk Score "
        "(Hotspot Map page), scoped to junction-level aggregation for consistency."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Named Junctions", f"{len(junctions)}")
    col2.metric("🔴 Critical", int((junctions["priority_level"] == "Critical").sum()))
    col3.metric("🟠 High", int((junctions["priority_level"] == "High").sum()))
    col4.metric("Top Junction", junctions.iloc[0]["junction_display"])

    st.markdown("---")
    top_n_junctions = st.slider("Show top N junctions", 10, len(junctions), min(25, len(junctions)))

    junction_display = junctions.nsmallest(top_n_junctions, "rank")[[
        "rank", "junction_display", "police_station", "violation_count",
        "top_violation", "top_vehicle", "peak_hour", "junction_risk_score", "priority_level"
    ]].copy()
    junction_display["peak_hour"] = junction_display["peak_hour"].apply(lambda h: f"{h:02d}:00–{(h+1):02d}:00")
    junction_display.columns = ["Rank", "Junction", "Police Station", "Violations",
                                  "Top Violation", "Top Vehicle", "Peak Hour", "Risk Score", "Priority"]

    def color_junction_priority(val):
        colors = {"Critical": "color: #e74c3c; font-weight: bold", "High": "color: #f39c12; font-weight: bold",
                  "Medium": "color: #f1c40f", "Low": "color: #2ecc71"}
        return colors.get(val, "")

    try:
        styled_junctions = junction_display.style.map(color_junction_priority, subset=["Priority"])
    except AttributeError:
        styled_junctions = junction_display.style.applymap(color_junction_priority, subset=["Priority"])
    st.dataframe(styled_junctions, use_container_width=True, height=500)

    st.markdown("---")
    st.markdown('<p class="section-header">Violations by Police Station (for comparison)</p>', unsafe_allow_html=True)
    ps_counts = df["police_station"].value_counts().head(10).reset_index()
    ps_counts.columns = ["Police Station", "Violations"]
    fig = px.bar(ps_counts, x="Police Station", y="Violations", color="Violations",
                 color_continuous_scale="Reds", template="plotly_dark")
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       xaxis_tickangle=-30, height=350)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⏰ Time Analysis":
    st.markdown('<p class="main-header">⏰ Temporal Violation Patterns</p>', unsafe_allow_html=True)
    st.markdown("When do parking violations peak? Optimize patrol deployment by time.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Violations by Hour of Day</p>', unsafe_allow_html=True)
        hourly = filtered_df.groupby("hour").size().reset_index(name="count")
        fig = px.bar(hourly, x="hour", y="count",
                     color="count", color_continuous_scale="Reds",
                     template="plotly_dark",
                     labels={"hour": "Hour of Day", "count": "Violations"})
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Violations by Day of Week</p>', unsafe_allow_html=True)
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        daily_dow = filtered_df.groupby("day_of_week").size().reset_index(name="count")
        daily_dow["day_of_week"] = pd.Categorical(daily_dow["day_of_week"], categories=day_order, ordered=True)
        daily_dow = daily_dow.sort_values("day_of_week")
        fig = px.bar(daily_dow, x="day_of_week", y="count",
                     color="count", color_continuous_scale="Blues",
                     template="plotly_dark",
                     labels={"day_of_week": "Day", "count": "Violations"})
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Violation Heatmap: Hour × Day of Week</p>', unsafe_allow_html=True)
    pivot = filtered_df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(day_order)
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                    template="plotly_dark",
                    labels=dict(x="Hour of Day", y="Day of Week", color="Violations"),
                    aspect="auto")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                     paper_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Violations by Time Slot</p>', unsafe_allow_html=True)
    slot_counts = filtered_df["time_slot"].value_counts().reset_index()
    slot_counts.columns = ["Time Slot", "Violations"]
    fig = px.bar(slot_counts, x="Time Slot", y="Violations",
                 color="Violations", color_continuous_scale="Reds",
                 template="plotly_dark")
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                     plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VEHICLE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚗 Vehicle Breakdown":
    st.markdown('<p class="main-header">🚗 Vehicle & Violation Analysis</p>', unsafe_allow_html=True)
    st.markdown("Which vehicles commit which violations, and where?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Vehicle Type Distribution</p>', unsafe_allow_html=True)
        veh = filtered_df["vehicle_type"].value_counts().reset_index()
        veh.columns = ["Vehicle", "Count"]
        fig = px.bar(veh, x="Vehicle", y="Count",
                     color="Count", color_continuous_scale="Blues",
                     template="plotly_dark")
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                         xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Violation Type × Vehicle Heatmap</p>', unsafe_allow_html=True)
        top_vehs = filtered_df["vehicle_type"].value_counts().head(6).index
        top_viols = filtered_df["primary_violation"].value_counts().head(6).index
        sub = filtered_df[
            filtered_df["vehicle_type"].isin(top_vehs) &
            filtered_df["primary_violation"].isin(top_viols)
        ]
        cross = sub.groupby(["vehicle_type","primary_violation"]).size().unstack(fill_value=0)
        fig = px.imshow(cross, color_continuous_scale="Reds", template="plotly_dark",
                        aspect="auto")
        fig.update_layout(margin=dict(l=0,r=0,t=10,b=0),
                         paper_bgcolor="rgba(0,0,0,0)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Repeat Offenders (Top 20 by Violation Count)</p>', unsafe_allow_html=True)

    # Vectorized aggregation. The original per-group lambda x: x.value_counts().index[0]
    # approach took ~148 seconds on the full 298K-row dataset (231K unique vehicle groups,
    # each running a full value_counts() inside the lambda) — confirmed via direct timing,
    # not assumed. This groupby+idxmax approach gets the same "most frequent value per
    # group" result in under 1 second.
    def fast_mode(frame, group_col, value_col):
        counts = frame.groupby([group_col, value_col]).size().reset_index(name="cnt")
        idx = counts.groupby(group_col)["cnt"].idxmax()
        return counts.loc[idx].set_index(group_col)[value_col]

    violations_ct = filtered_df.groupby("vehicle_number").size().rename("violations")
    vehicle_type_first = filtered_df.groupby("vehicle_number")["vehicle_type"].first()
    last_seen_max = filtered_df.groupby("vehicle_number")["created_datetime"].max()
    top_violation_mode = fast_mode(filtered_df, "vehicle_number", "primary_violation")
    top_station_mode = fast_mode(filtered_df, "vehicle_number", "police_station")

    repeat = pd.concat([
        violations_ct, vehicle_type_first,
        top_violation_mode.rename("primary_violation"),
        top_station_mode.rename("police_station"),
        last_seen_max,
    ], axis=1).reset_index().nlargest(20, "violations")

    repeat["created_datetime"] = repeat["created_datetime"].dt.strftime("%Y-%m-%d")
    repeat.columns = ["Vehicle No","Violations","Type","Top Violation","Station","Last Seen"]
    st.dataframe(repeat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ZONE DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Zone Deep Dive":
    st.markdown('<p class="main-header">🔍 Zone Deep Dive</p>', unsafe_allow_html=True)
    st.markdown("Drill into any specific hotspot zone for detailed enforcement intelligence")

    top50 = filtered_hotspots.nlargest(50, "priority_score").copy()
    top50["zone_label"] = top50.apply(
        lambda r: f"#{int(r['priority_rank'])} | {r['top_police_station']} | Score: {r['priority_score']}", axis=1
    )

    selected_zone = st.selectbox("Select a Zone", top50["zone_label"].tolist())
    zone_row = top50[top50["zone_label"] == selected_zone].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Priority Score", f"{zone_row['priority_score']:.2f}")
    col2.metric("Total Violations", f"{int(zone_row['violation_count']):,}")
    col3.metric("Unique Vehicles", f"{int(zone_row['unique_vehicles']):,}")
    col4.metric("Peak Hour", f"{int(zone_row['peak_hour']):02d}:00")

    st.markdown("---")

    # Get raw data for this zone
    zone_df = filtered_df[
        (filtered_df["grid_lat"] == zone_row["latitude"]) &
        (filtered_df["grid_lon"] == zone_row["longitude"])
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Police Station:** {zone_row['top_police_station']}")
        st.markdown(f"**Priority Level:** {zone_row['priority_label']}")
        st.markdown(f"**Top Violation:** {zone_row['top_violation']}")
        st.markdown(f"**Top Vehicle:** {zone_row['top_vehicle']}")
        st.markdown(f"**Peak Time:** {zone_row['top_time_period']}")
        st.markdown(f"**Weekend Share:** {zone_row['weekend_ratio']*100:.1f}%")
        st.markdown(f"**Junction Zone:** {'Yes' if zone_row['junction_flag'] else 'No'}")

    with col2:
        st.markdown('<p class="section-header">Violations by Hour</p>', unsafe_allow_html=True)
        hourly = zone_df.groupby("hour").size().reset_index(name="count")
        fig = px.bar(hourly, x="hour", y="count", color="count",
                     color_continuous_scale="Reds", template="plotly_dark")
        fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=5,b=0),
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                         height=250)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Enforcement Recommendation")

    peak_h = int(zone_row['peak_hour'])
    weekend_flag = "weekends and" if zone_row['weekend_ratio'] > 0.35 else ""
    junction_flag = "near a junction" if zone_row['junction_flag'] else "in a non-junction zone"

    st.info(f"""
    **Zone #{int(zone_row['priority_rank'])} — {zone_row['priority_label']}**

    This zone falls under **{zone_row['top_police_station']}** jurisdiction and is {junction_flag}.
    The dominant violation is **{zone_row['top_violation']}** committed primarily by **{zone_row['top_vehicle']}** vehicles.

    🕐 **Deploy enforcement between {peak_h:02d}:00–{(peak_h+2):02d}:00** — peak violation window.
    📅 **Increase patrols on {weekend_flag} weekdays** given the temporal pattern.
    🚗 **Focus challan drives on {zone_row['top_vehicle']} vehicles** in this zone.
    📍 **Coordinates:** {zone_row['latitude']:.4f}°N, {zone_row['longitude']:.4f}°E
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Forecasting":
    st.markdown('<p class="main-header">📈 Forecasting</p>', unsafe_allow_html=True)
    st.markdown("Predicting near-term violation volume from historical patterns — "
                 "city-wide, with an honest account of what didn't work along the way.")

    # ── Tomorrow's headline prediction cards ─────────────────────────────────
    tomorrow_date = station_forecast["forecast_date"].min()
    tomorrow_data = station_forecast[station_forecast["forecast_date"] == tomorrow_date].sort_values(
        "predicted_violations", ascending=False
    )
    top_predicted_station = tomorrow_data.iloc[0]["police_station"]
    top_predicted_violations = tomorrow_data.iloc[0]["predicted_violations"]

    # Derive risk level and enforcement window from REAL existing data for that station
    # (not invented) — the station's highest-risk DBSCAN cluster and its historical peak hour.
    station_top_cluster = ml_clusters[ml_clusters["top_police_station"] == top_predicted_station].nlargest(1, "ml_risk_score")
    if len(station_top_cluster) > 0:
        expected_risk = station_top_cluster.iloc[0]["priority_level"].replace(" Priority", "")
    else:
        expected_risk = "Unrated"
    station_peak_hour = df[df["police_station"] == top_predicted_station]["hour"].value_counts().index[0]
    enforcement_window = f"{station_peak_hour:02d}:00–{(station_peak_hour+2)%24:02d}:00"

    st.markdown('<p class="section-header">Tomorrow\'s Prediction at a Glance</p>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Predicted Highest-Risk Station", top_predicted_station)
    pc2.metric("Expected Violations", f"{top_predicted_violations:.0f}")
    pc3.metric("Expected Risk Level", expected_risk)
    pc4.metric("Recommended Enforcement Window", enforcement_window)
    st.caption(
        f"Station-level violation count is allocated from the city-wide forecast by historical share "
        f"(see methodology below). Risk level and enforcement window come from this station's highest-priority "
        f"DBSCAN cluster and its historical peak hour — both directly observed in the data, not forecasted."
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "Ridge Regression")
    col2.metric("MAE (test)", f"{forecast_metrics['mae']:.0f} /day")
    col3.metric("MAPE", f"{forecast_metrics['mape_pct']:.1f}%")
    col4.metric("vs. Naive Baseline", f"{forecast_metrics['improvement_over_baseline_pct']:+.1f}%")

    st.markdown("---")
    st.markdown('<p class="section-header">Why City-Wide, Not Per-Zone or Per-Station</p>', unsafe_allow_html=True)

    diag_col1, diag_col2, diag_col3 = st.columns(3)
    with diag_col1:
        st.markdown("**Grid-Zone Level**")
        st.markdown("❌ Abandoned")
        st.caption("Median hotspot zone had recorded activity on only 40 of 150 possible days — too sparse for a credible daily forecast.")
    with diag_col2:
        st.markdown("**Police-Station Level**")
        st.markdown("❌ Abandoned")
        st.caption(f"Coefficient of variation 0.85 — day-to-day swings nearly as large as the mean. A model here scored worse than a naive 7-day average.")
    with diag_col3:
        st.markdown("**City-Wide Level**")
        st.markdown("✅ Used")
        st.caption(f"Coefficient of variation {forecast_metrics['coefficient_of_variation_city']:.2f} — stable enough to show genuine signal.")

    st.markdown("---")
    st.markdown('<p class="section-header">Why Ridge Regression, Not XGBoost</p>', unsafe_allow_html=True)
    st.warning(
        f"**XGBoost was tried first and lost.** On the same city-wide features, XGBoost scored an MAE of "
        f"{forecast_metrics['xgboost_attempt_mae']:.0f}/day — {forecast_metrics['xgboost_attempt_improvement_pct']:.1f}% "
        f"*worse* than simply using the trailing 7-day average. With only 124 days of training data, XGBoost's "
        f"flexibility let it overfit noise rather than learn real structure. A simpler, regularized Ridge regression "
        f"on identical features generalizes better and actually beats the naive baseline."
    )

    st.markdown("---")
    st.markdown('<p class="section-header">Model Validation: Predicted vs. Actual (Held-Out Last 21 Days)</p>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_test_results["date"], y=forecast_test_results["violation_count"],
                              mode="lines+markers", name="Actual", line=dict(color="#5EC4D6", width=2)))
    fig.add_trace(go.Scatter(x=forecast_test_results["date"], y=forecast_test_results["predicted"],
                              mode="lines+markers", name="Model Prediction", line=dict(color="#e94560", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=forecast_test_results["date"], y=forecast_test_results["naive_predicted"],
                              mode="lines", name="Naive (7-day avg) Baseline", line=dict(color="#888", width=1, dash="dot")))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=10, b=0),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                       xaxis_title="Date", yaxis_title="Violations / day")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This is the actual held-out test period — the model never saw these 21 days during training. "
               "The gap between the red dashed line and the actual blue line is the real, honest error margin.")

    st.markdown("---")
    st.markdown('<p class="section-header">7-Day Forward Forecast (City-Wide)</p>', unsafe_allow_html=True)

    fc_display = city_forecast.copy()
    fc_display["forecast_date"] = pd.to_datetime(fc_display["forecast_date"]).dt.strftime("%a, %b %d")
    fig2 = px.bar(fc_display, x="forecast_date", y="predicted_total_violations",
                  color="predicted_total_violations", color_continuous_scale="Tealgrn",
                  template="plotly_dark", text="predicted_total_violations")
    fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="", yaxis_title="Predicted Violations")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Forecast Allocated to Top Stations</p>', unsafe_allow_html=True)
    st.caption("City-wide forecast distributed by each station's stable historical share of total violations — "
               "not an independently forecasted number per station, since station-level data was too noisy for that (see above).")

    station_pivot = station_forecast.copy()
    station_pivot["forecast_date"] = pd.to_datetime(station_pivot["forecast_date"]).dt.strftime("%a %m/%d")
    pivot_table = station_pivot.pivot(index="police_station", columns="forecast_date", values="predicted_violations")
    pivot_table = pivot_table.loc[pivot_table.mean(axis=1).sort_values(ascending=False).index]
    try:
        import matplotlib  # noqa: F401
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False

    if has_matplotlib:
        styled_pivot = pivot_table.style.background_gradient(cmap="Reds", axis=None).format("{:.0f}")
    else:
        styled_pivot = pivot_table.style.format("{:.0f}")  # matplotlib not available — plain table, still functional
    st.dataframe(styled_pivot, use_container_width=True, height=400)

    st.markdown("---")
    st.info(
        f"**Reading this page honestly:** the model beats a naive baseline by "
        f"{forecast_metrics['improvement_over_baseline_pct']:.1f}% on held-out data — a real but modest "
        f"improvement, not a dramatic one. We're presenting it as a genuine, validated forecast rather than "
        f"inflating the claim. The most defensible use of this page is the 7-day directional trend "
        f"(are violations rising or falling city-wide), not a precise day-level number for any single station."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Recommendations":
    st.markdown('<p class="main-header">💡 Top Enforcement Recommendations</p>', unsafe_allow_html=True)
    st.markdown(
        "The 5 highest-priority hotspot clusters, with a concrete enforcement action for each — "
        "derived from real cluster-level data (violation count, dominant violation type, peak hour, "
        "junction proximity), not invented."
    )

    top5 = ml_clusters.nsmallest(5, "risk_rank").copy()

    def vehicle_recommendation(top_vehicle, violation_count):
        if violation_count >= 8000:
            return "2 officers + 1 tow-away vehicle"
        elif violation_count >= 3000:
            return "1 officer + 1 tow-away vehicle"
        else:
            return "1 officer on patrol"

    for _, row in top5.iterrows():
        priority_colors = {
            "Critical Priority": "#e74c3c", "High Priority": "#f39c12",
            "Medium Priority": "#f1c40f", "Low Priority": "#2ecc71",
        }
        color = priority_colors.get(row["priority_level"], "#888")
        deploy_action = vehicle_recommendation(row["top_vehicle"], row["violation_count"])
        window_start = int(row["peak_hour"])
        window_end = (window_start + 2) % 24
        junction_text = row["top_junction"] if row["top_junction"] != "No Junction" else "a non-junction zone"
        reason_parts = [f"{int(row['violation_count']):,} total violations"]
        if row["time_concentration"] > 0.15:
            reason_parts.append("strong peak-hour concentration")
        if row["junction_flag"]:
            reason_parts.append("junction zone")
        if row["is_outlier"] == -1:
            reason_parts.append("flagged as anomalous by Isolation Forest")
        reason = ", ".join(reason_parts)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #16213e, #0f3460); border-left: 5px solid {color};
                    border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;">
            <div style="font-size: 1.1rem; font-weight: 700; color: white;">
                {int(row['risk_rank'])}. {row['top_police_station']} — {junction_text}
            </div>
            <div style="color: {color}; font-weight: 600; margin: 0.3rem 0;">
                Priority: {row['priority_level']} (Risk Score: {row['ml_risk_score']:.1f}/100)
            </div>
            <div style="color: #cbd5e1; margin-top: 0.5rem; line-height: 1.5;">
                🚓 <b>Deploy:</b> {deploy_action}<br/>
                🕐 <b>Time:</b> {window_start:02d}:00–{window_end:02d}:00<br/>
                🅿️ <b>Dominant violation:</b> {row['top_violation']} ({row['top_vehicle']})<br/>
                📋 <b>Reason:</b> {reason}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Deployment sizing (officers/tow-away vehicles) is a simple rule of thumb based on violation volume "
        "thresholds, not a separately trained model — shown transparently here rather than presented as ML output."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOW THE ML WORKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 How the ML Works":
    st.markdown('<p class="main-header">🧠 How the ML Works</p>', unsafe_allow_html=True)
    st.markdown("A transparent account of every model in this app, including what was tried and rejected.")

    st.markdown("---")
    st.markdown('<p class="section-header">1. DBSCAN — Hotspot Clustering</p>', unsafe_allow_html=True)
    st.markdown("""
    Groups nearby parking violations into real geographic hotspot clusters using latitude and longitude,
    without needing to predefine the number of clusters. Run on an 80,000-row stratified sample (a full-dataset
    run on all 298K rows exceeded available memory). Parameters: 100m radius, minimum 30 points per cluster.
    Result: **305 real clusters**, 21.9% of sampled points classified as noise (isolated, non-clustered violations).
    """)

    st.markdown('<p class="section-header">2. Isolation Forest — Anomaly Detection</p>', unsafe_allow_html=True)
    st.markdown("""
    Detects unusually severe hotspot clusters based on violation count, peak-hour concentration, vehicle
    diversity, and the mix of violation types (no-parking, wrong-parking, main-road, double-parking counts).
    Since the dataset contains only violation records (no "normal, non-violating" parking data), this model
    identifies which *violation* hotspots are abnormally severe relative to other hotspots — not violation
    versus non-violation.
    """)

    st.markdown('<p class="section-header">3. ML Risk Score — Combining Both</p>', unsafe_allow_html=True)
    st.markdown("""
    Combines log-scaled violation density (55%), the Isolation Forest anomaly score (20%), peak-hour
    concentration (10%), and violation/vehicle severity (15%) into a single 0–100 score. Density is weighted
    most heavily deliberately: an earlier equal-weighted version of this score let a low-volume cluster with a
    sharp anomaly signature outrank the city's largest, already-validated hotspot — corrected by making density
    the dominant term, which is also what would matter most for real enforcement deployment.
    """)

    st.markdown('<p class="section-header">4. Forecasting — What Worked and What Didn\'t</p>', unsafe_allow_html=True)
    st.markdown(f"""
    Three forecasting granularities were tested honestly, and two were rejected:
    - **Grid-zone level:** rejected — median hotspot zone had only 40 of 150 possible days with recorded
      activity, too sparse for a daily time series.
    - **Police-station level:** rejected — coefficient of variation 0.85 (day-to-day swings nearly as large
      as the average itself). A model tested here lost to a naive 7-day-average baseline.
    - **City-wide level:** used — coefficient of variation {forecast_metrics['coefficient_of_variation_city']:.2f},
      genuinely forecastable. XGBoost was tried first here too and *also* lost to the naive baseline
      ({forecast_metrics['xgboost_attempt_improvement_pct']:.1f}%), likely overfitting on only 124 training days.
      A simpler, regularized Ridge regression on the same features won, beating the baseline by
      {forecast_metrics['improvement_over_baseline_pct']:+.1f}%.
    """)

    st.markdown("---")
    st.markdown('<p class="section-header">Dataset Used</p>', unsafe_allow_html=True)
    st.info("""
    **Dataset used:** Parking Violation Dataset only (298,277 cleaned records, Bengaluru, Nov 2023–Apr 2024).

    **Dataset 2 (Event/Incident data) — not used in this prototype.** It was evaluated and found to have
    94% missing `end_datetime` values, making event duration and impact uncomputable. Rather than force a
    weak integration, the event dataset was excluded entirely, and the project scope was kept to what the
    available data could support credibly.
    """)