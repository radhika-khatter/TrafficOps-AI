import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Load Model
# =========================
model = joblib.load("road_closure_model_v3.pkl")
cause_lookup = joblib.load("cause_lookup.pkl")
station_lookup = joblib.load("station_lookup.pkl")
corridor_lookup = joblib.load("corridor_lookup.pkl")

forecast_df = pd.read_csv("forecast_output.csv")
hotspot_df = pd.read_csv("hotspot_intelligence.csv")
heatmap_df = pd.read_csv("heatmap_data.csv", index_col=0)
hotspot_locations = pd.read_csv("hotspot_locations.csv")
command_center_df = pd.read_csv("command_center.csv")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="TrafficOps AI — Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1600px;
}

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #0f1729 0%, #1a2744 40%, #0d2137 70%, #0a0e1a 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6, #38bdf8);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 400;
    margin: 0 0 0.75rem 0;
    letter-spacing: 0.02em;
}
.hero-badges {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #38bdf8;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge.green { background: rgba(34,197,94,0.1); border-color: rgba(34,197,94,0.25); color: #22c55e; }
.badge.purple { background: rgba(129,140,248,0.1); border-color: rgba(129,140,248,0.25); color: #818cf8; }
.badge.pink { background: rgba(244,114,182,0.1); border-color: rgba(244,114,182,0.25); color: #f472b6; }

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: linear-gradient(145deg, #0f1729, #131d35);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 14px 14px;
}
.kpi-card.blue { border: 1px solid rgba(56,189,248,0.25); }
.kpi-card.blue::after { background: linear-gradient(90deg, #38bdf8, #0ea5e9); }
.kpi-card.green { border: 1px solid rgba(34,197,94,0.25); }
.kpi-card.green::after { background: linear-gradient(90deg, #22c55e, #16a34a); }
.kpi-card.red { border: 1px solid rgba(239,68,68,0.25); }
.kpi-card.red::after { background: linear-gradient(90deg, #ef4444, #dc2626); }
.kpi-card.amber { border: 1px solid rgba(251,191,36,0.25); }
.kpi-card.amber::after { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
.kpi-card.purple { border: 1px solid rgba(129,140,248,0.25); }
.kpi-card.purple::after { background: linear-gradient(90deg, #818cf8, #6366f1); }
.kpi-card.pink { border: 1px solid rgba(244,114,182,0.25); }
.kpi-card.pink::after { background: linear-gradient(90deg, #f472b6, #ec4899); }

.kpi-icon { font-size: 1.6rem; margin-bottom: 0.4rem; display: block; }
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-card.blue .kpi-value { color: #38bdf8; }
.kpi-card.green .kpi-value { color: #22c55e; }
.kpi-card.red .kpi-value { color: #ef4444; }
.kpi-card.amber .kpi-value { color: #fbbf24; }
.kpi-card.purple .kpi-value { color: #818cf8; }
.kpi-card.pink .kpi-value { color: #f472b6; }
.kpi-label {
    font-size: 0.68rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Section Header ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(56,189,248,0.12);
}
.section-icon { font-size: 1.3rem; }
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.01em;
}
.section-count {
    background: rgba(56,189,248,0.12);
    color: #38bdf8;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Input Panel ── */
.input-panel {
    background: linear-gradient(145deg, #0f1729, #131d35);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.input-panel-title {
    font-size: 0.8rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

/* ── Result Cards ── */
.result-card {
    background: linear-gradient(145deg, #0f1729, #131d35);
    border-radius: 14px;
    padding: 1.25rem;
    margin: 0.5rem 0;
}
.result-card.critical { border-left: 4px solid #ef4444; border-top: 1px solid rgba(239,68,68,0.2); border-right: 1px solid rgba(239,68,68,0.1); border-bottom: 1px solid rgba(239,68,68,0.1); }
.result-card.high { border-left: 4px solid #f97316; border-top: 1px solid rgba(249,115,22,0.2); border-right: 1px solid rgba(249,115,22,0.1); border-bottom: 1px solid rgba(249,115,22,0.1); }
.result-card.medium { border-left: 4px solid #fbbf24; border-top: 1px solid rgba(251,191,36,0.2); border-right: 1px solid rgba(251,191,36,0.1); border-bottom: 1px solid rgba(251,191,36,0.1); }
.result-card.low { border-left: 4px solid #22c55e; border-top: 1px solid rgba(34,197,94,0.2); border-right: 1px solid rgba(34,197,94,0.1); border-bottom: 1px solid rgba(34,197,94,0.1); }
.result-card.info { border-left: 4px solid #38bdf8; border-top: 1px solid rgba(56,189,248,0.2); border-right: 1px solid rgba(56,189,248,0.1); border-bottom: 1px solid rgba(56,189,248,0.1); }

/* ── Risk Badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.risk-badge.critical { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.risk-badge.high { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); }
.risk-badge.medium { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.risk-badge.low { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }

/* ── Alert Banner ── */
.alert-banner {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.06));
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.alert-banner-icon { font-size: 1.5rem; }
.alert-banner-text { flex: 1; }
.alert-banner-title { font-size: 0.95rem; font-weight: 700; color: #ef4444; margin-bottom: 0.15rem; }
.alert-banner-sub { font-size: 0.8rem; color: #94a3b8; }

/* ── Stat Row ── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.stat-item {
    flex: 1;
    min-width: 120px;
    background: rgba(15,23,41,0.8);
    border: 1px solid rgba(56,189,248,0.1);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}
.stat-item-value {
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
}
.stat-item-label { font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stSlider > div {
    background: rgba(15,23,41,0.9) !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #38bdf8 !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,41,0.8) !important;
    border-radius: 12px !important;
    padding: 0.3rem !important;
    border: 1px solid rgba(56,189,248,0.1) !important;
    gap: 0.2rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #2d1b69) !important;
    color: #e2e8f0 !important;
}

/* ── DataFrame styling ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(56,189,248,0.1) !important;
}

/* ── Divider ── */
.cyber-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
    margin: 1.5rem 0;
}

/* ── Hotspot row card ── */
.hotspot-card {
    background: linear-gradient(145deg, #0f1729, #131d35);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    transition: transform 0.15s;
}
.hotspot-card:hover { transform: translateX(4px); }
.hotspot-card.critical { border-left: 4px solid #ef4444; border: 1px solid rgba(239,68,68,0.2); border-left: 4px solid #ef4444; }
.hotspot-card.high { border: 1px solid rgba(249,115,22,0.2); border-left: 4px solid #f97316; }
.hotspot-card.medium { border: 1px solid rgba(251,191,36,0.2); border-left: 4px solid #fbbf24; }

/* ── Briefing ── */
.briefing-header {
    background: linear-gradient(135deg, #0f1729, #1a2744);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.briefing-header::before {
    content: 'CLASSIFIED — OPERATIONAL USE ONLY';
    position: absolute;
    top: 0.6rem;
    right: 1rem;
    font-size: 0.55rem;
    color: rgba(239,68,68,0.5);
    font-weight: 700;
    letter-spacing: 0.15em;
}
.briefing-title { font-size: 1.4rem; font-weight: 800; color: #e2e8f0; margin-bottom: 0.2rem; }
.briefing-meta { font-size: 0.75rem; color: #64748b; font-family: 'JetBrains Mono', monospace; }

/* ── Map controls panel ── */
.map-legend {
    background: rgba(10,14,26,0.9);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.legend-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; font-size: 0.8rem; color: #94a3b8; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.legend-dot.red { background: #ef4444; }
.legend-dot.orange { background: #f97316; }
.legend-dot.blue { background: #38bdf8; }

/* ── Progress bar override ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    border-radius: 4px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Hero Header
# =========================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🚦 TrafficOps AI</div>
    <div class="hero-subtitle">Event-Driven Traffic Impact &amp; Resource Recommendation System — Bengaluru Metropolitan Area</div>
    <div class="hero-badges">
        <span class="badge">● LIVE OPERATIONS</span>
        <span class="badge green">AI-POWERED</span>
        <span class="badge purple">ML v3.0</span>
        <span class="badge pink">Smart City</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Executive KPI Bar
# =========================
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card blue">
        <span class="kpi-icon">📡</span>
        <div class="kpi-value">8,057</div>
        <div class="kpi-label">Events Analyzed</div>
    </div>
    <div class="kpi-card purple">
        <span class="kpi-icon">📈</span>
        <div class="kpi-value">528</div>
        <div class="kpi-label">Forecast Windows</div>
    </div>
    <div class="kpi-card red">
        <span class="kpi-icon">🚨</span>
        <div class="kpi-value">117</div>
        <div class="kpi-label">Actionable Alerts</div>
    </div>
    <div class="kpi-card amber">
        <span class="kpi-icon">📍</span>
        <div class="kpi-value">10+</div>
        <div class="kpi-label">Hotspot Clusters</div>
    </div>
    <div class="kpi-card green">
        <span class="kpi-icon">🚔</span>
        <div class="kpi-value">563</div>
        <div class="kpi-label">Officers Deployed</div>
    </div>
    <div class="kpi-card pink">
        <span class="kpi-icon">🚧</span>
        <div class="kpi-value">340</div>
        <div class="kpi-label">Barricades Planned</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚦 Event Analysis",
    "📈 Traffic Forecast",
    "📍 Hotspot Intelligence",
    "🗺️ Hotspot Map",
    "🚨 Command Center",
    "📋 Daily Briefing"
])

# ─────────────────────────────────────────────
# TAB 1 — EVENT ANALYSIS
# ─────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🎯</span>
        <span class="section-title">Event Impact Analyzer</span>
        <span class="section-count">ML MODEL v3.0</span>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_results = st.columns([1, 1.6], gap="large")

    with col_input:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown('<div class="input-panel-title">⚙ Event Parameters</div>', unsafe_allow_html=True)

        event_cause = st.selectbox("Event Cause", [
            "vehicle_breakdown","construction","accident","water_logging",
            "tree_fall","road_conditions","public_event","procession","congestion"
        ])
        corridor = st.selectbox("Corridor", sorted(corridor_lookup.keys()))
        police_station = st.selectbox("Police Station", sorted(station_lookup.keys()))
        hour = st.slider("Hour of Event", 0, 23, 12)

        st.markdown(f"""
        <div style="background:rgba(56,189,248,0.05);border:1px solid rgba(56,189,248,0.12);border-radius:8px;padding:0.6rem 0.8rem;margin:0.8rem 0;font-size:0.75rem;color:#94a3b8;">
            ⏰ Selected Time: <strong style="color:#38bdf8;">{hour:02d}:00 {'AM' if hour < 12 else 'PM'}</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            📅 Day: <strong style="color:#38bdf8;">Weekday</strong>
        </div>
        """, unsafe_allow_html=True)

        analyze_clicked = st.button("🔍 ANALYZE EVENT", key="analyze_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        if analyze_clicked:
            # ── Time period ──
            if 6 <= hour <= 9:
                time_period = "Morning Rush"
            elif 18 <= hour <= 21:
                time_period = "Evening Rush"
            elif 10 <= hour <= 17:
                time_period = "Normal"
            else:
                time_period = "Night"

            cause_rate = cause_lookup.get(event_cause, 0)
            station_rate = station_lookup.get(police_station, 0)
            corridor_rate = corridor_lookup.get(corridor, 0)

            input_df = pd.DataFrame({
                "event_type": ["unplanned"],
                "event_cause": [event_cause],
                "corridor": [corridor],
                "police_station": [police_station],
                "hour": [hour],
                "time_period": [time_period],
                "day_of_week": [3],
                "month": [6],
                "is_weekend": [0],
                "cause_closure_rate": [cause_rate],
                "station_closure_rate": [station_rate],
                "corridor_closure_rate": [corridor_rate]
            })

            try:
                closure_prob = model.predict_proba(input_df)[0][1]
                closure_percent = round(closure_prob * 100, 2)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.stop()

            # ── Impact ──
            if event_cause in ["construction","road_conditions","public_event"]:
                impact_level = "High"
            elif event_cause in ["water_logging","tree_fall","congestion"]:
                impact_level = "Medium"
            else:
                impact_level = "Low"

            # ── Resources ──
            if impact_level == "Low":
                officers, barricades, diversion = 2, 0, "No"
            elif impact_level == "Medium":
                officers, barricades, diversion = 4, 2, "No"
            else:
                officers, barricades, diversion = 8, 4, "Yes"

            if closure_prob > 0.70:
                officers += 4
                barricades += 2
                diversion = "Yes"

            # ── Risk level ──
            if closure_prob < 0.30:
                risk_label = "LOW RISK"
                risk_cls = "low"
                risk_color = "#22c55e"
            elif closure_prob < 0.70:
                risk_label = "MEDIUM RISK"
                risk_cls = "medium"
                risk_color = "#fbbf24"
            else:
                risk_label = "HIGH RISK"
                risk_cls = "critical"
                risk_color = "#ef4444"

            # ── Hotspot check ──
            hotspot_corridors = ["Bellary Road 1","Bellary Road 2","Mysore Road","ORR East 1","ORR North 1","Tumkur Road"]
            is_hotspot = corridor in hotspot_corridors

            # ── Risk Assessment Card ──
            st.markdown(f"""
            <div class="result-card {risk_cls}" style="margin-top:0;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.8rem;">
                    <div style="font-size:0.7rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">Traffic Risk Assessment</div>
                    <span class="risk-badge {risk_cls}">⬤ {risk_label}</span>
                </div>
                <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.5rem;">Road Closure Probability</div>
                <div style="font-size:3rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:{risk_color};line-height:1;">{closure_percent}%</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:0.3rem;">Time Period: <strong style="color:#94a3b8;">{time_period}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(int(closure_percent), 100))

            # ── Hotspot Status ──
            if is_hotspot:
                st.markdown(f"""
                <div class="result-card critical" style="margin-top:0.5rem;">
                    <div style="font-size:0.85rem;font-weight:700;color:#ef4444;">⚠ Known Traffic Hotspot Corridor</div>
                    <div style="font-size:0.75rem;color:#94a3b8;margin-top:0.3rem;">This corridor is flagged in the hotspot intelligence database. Expect elevated disruption.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card low" style="margin-top:0.5rem;">
                    <div style="font-size:0.85rem;font-weight:700;color:#22c55e;">✓ Corridor Not a Major Hotspot</div>
                    <div style="font-size:0.75rem;color:#94a3b8;margin-top:0.3rem;">This corridor is not flagged in the hotspot intelligence database.</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Resource Cards ──
            st.markdown("""
            <div class="section-header" style="margin-top:1rem;">
                <span class="section-icon">🚔</span>
                <span class="section-title">Resource Deployment Plan</span>
            </div>
            """, unsafe_allow_html=True)

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Impact Level", impact_level)
            with r2:
                st.metric("Officers", officers)
            with r3:
                st.metric("Barricades", barricades)
            with r4:
                st.metric("Diversion", diversion)

            # ── Deployment visual ──
            impact_colors = {"Low":"#22c55e","Medium":"#fbbf24","High":"#ef4444"}
            fig_resource = go.Figure()
            fig_resource.add_trace(go.Bar(
                x=["Officers Required","Barricades Required"],
                y=[officers, barricades],
                marker_color=["#38bdf8","#818cf8"],
                text=[officers, barricades],
                textposition="outside",
                textfont=dict(color="white", size=14, family="JetBrains Mono"),
            ))
            fig_resource.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,41,0.6)",
                font=dict(color="#94a3b8", family="Inter"),
                height=220,
                margin=dict(l=20,r=20,t=20,b=30),
                showlegend=False,
                yaxis=dict(gridcolor="rgba(56,189,248,0.08)", color="#64748b"),
                xaxis=dict(color="#64748b"),
            )
            st.plotly_chart(fig_resource, use_container_width=True, config={"displayModeBar": False})

            # ── Event Summary ──
            st.markdown(f"""
            <div class="result-card info">
                <div style="font-size:0.7rem;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">📋 Event Summary</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;font-size:0.8rem;">
                    <div><span style="color:#64748b;">Cause:</span> <strong style="color:#e2e8f0;">{event_cause.replace('_',' ').title()}</strong></div>
                    <div><span style="color:#64748b;">Hour:</span> <strong style="color:#e2e8f0;">{hour:02d}:00</strong></div>
                    <div><span style="color:#64748b;">Corridor:</span> <strong style="color:#e2e8f0;">{corridor}</strong></div>
                    <div><span style="color:#64748b;">Station:</span> <strong style="color:#e2e8f0;">{police_station}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:360px;text-align:center;opacity:0.5;">
                <div style="font-size:3rem;margin-bottom:1rem;">🔍</div>
                <div style="font-size:1rem;color:#64748b;font-weight:600;">Configure event parameters</div>
                <div style="font-size:0.8rem;color:#475569;margin-top:0.4rem;">and click Analyze Event to run the ML model</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 2 — TRAFFIC FORECAST
# ─────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📈</span>
        <span class="section-title">Traffic Forecast Dashboard</span>
        <span class="section-count">528 WINDOWS</span>
    </div>
    """, unsafe_allow_html=True)

    risk_counts = forecast_df["risk_level"].value_counts()
    red_cnt = risk_counts.get("RED", 0)
    amber_cnt = risk_counts.get("AMBER", 0)
    green_cnt = risk_counts.get("GREEN", 0)

    fa, fb, fc, fd = st.columns(4)
    with fa:
        st.markdown(f"""
        <div class="kpi-card red" style="padding:0.9rem;">
            <div class="kpi-icon">🔴</div>
            <div class="kpi-value">{red_cnt}</div>
            <div class="kpi-label">Critical Windows</div>
        </div>
        """, unsafe_allow_html=True)
    with fb:
        st.markdown(f"""
        <div class="kpi-card amber" style="padding:0.9rem;">
            <div class="kpi-icon">🟠</div>
            <div class="kpi-value">{amber_cnt}</div>
            <div class="kpi-label">Amber Windows</div>
        </div>
        """, unsafe_allow_html=True)
    with fc:
        st.markdown(f"""
        <div class="kpi-card green" style="padding:0.9rem;">
            <div class="kpi-icon">🟢</div>
            <div class="kpi-value">{green_cnt}</div>
            <div class="kpi-label">Green Windows</div>
        </div>
        """, unsafe_allow_html=True)
    with fd:
        st.markdown(f"""
        <div class="kpi-card blue" style="padding:0.9rem;">
            <div class="kpi-icon">📊</div>
            <div class="kpi-value">{len(forecast_df)}</div>
            <div class="kpi-label">Total Forecasts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="cyber-divider">', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1.4, 1])

    with chart_col1:
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">🏆</span>
            <span class="section-title">Top Corridor Risk Ranking</span>
        </div>
        """, unsafe_allow_html=True)

        top_corridors = (
            forecast_df.sort_values("forecast_score", ascending=False)
            .drop_duplicates("corridor")
            .head(10)
        )
        color_map = {"RED": "#ef4444", "AMBER": "#fbbf24", "GREEN": "#22c55e"}
        bar_colors = [color_map.get(r, "#38bdf8") for r in top_corridors["risk_level"]]

        fig_corridor = go.Figure(go.Bar(
            x=top_corridors["forecast_score"],
            y=top_corridors["corridor"],
            orientation='h',
            marker=dict(color=bar_colors, opacity=0.85),
            text=top_corridors["forecast_score"].round(1),
            textposition="outside",
            textfont=dict(color="white", size=11, family="JetBrains Mono"),
        ))
        fig_corridor.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,41,0.6)",
            font=dict(color="#94a3b8", family="Inter"),
            height=360,
            margin=dict(l=10,r=40,t=10,b=20),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(56,189,248,0.08)", color="#64748b", title="Forecast Score"),
            yaxis=dict(color="#94a3b8", autorange="reversed"),
        )
        st.plotly_chart(fig_corridor, use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">🥧</span>
            <span class="section-title">Risk Distribution</span>
        </div>
        """, unsafe_allow_html=True)

        fig_pie = go.Figure(go.Pie(
            labels=["RED — Critical", "AMBER — High", "GREEN — Normal"],
            values=[red_cnt, amber_cnt, green_cnt],
            hole=0.55,
            marker=dict(colors=["#ef4444","#fbbf24","#22c55e"],
                        line=dict(color="#0a0e1a", width=2)),
            textinfo="percent",
            textfont=dict(size=11, color="white"),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            height=360,
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            annotations=[dict(text=f"<b>{len(forecast_df)}</b><br>Windows", x=0.5, y=0.5,
                              font=dict(size=14, color="#e2e8f0", family="JetBrains Mono"),
                              showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

    # Top 10 table
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📋</span>
        <span class="section-title">Top 10 Highest Risk Windows</span>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(
        forecast_df.sort_values("forecast_score", ascending=False).head(10),
        use_container_width=True, hide_index=True
    )

    # Heatmap
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🔥</span>
        <span class="section-title">Traffic Risk Heatmap — Corridor × Hour</span>
    </div>
    """, unsafe_allow_html=True)

    fig_heat, ax_heat = plt.subplots(figsize=(18, 7))
    fig_heat.patch.set_facecolor("#0a0e1a")
    ax_heat.set_facecolor("#0f1729")
    sns.heatmap(heatmap_df, cmap="YlOrRd", ax=ax_heat,
                linewidths=0.5, linecolor="#0a0e1a",
                cbar_kws={"shrink": 0.6})
    ax_heat.tick_params(colors="#94a3b8", labelsize=8)
    ax_heat.set_xlabel("Hour of Day", color="#94a3b8", fontsize=9)
    ax_heat.set_ylabel("Corridor", color="#94a3b8", fontsize=9)
    plt.setp(ax_heat.get_xticklabels(), rotation=0, color="#94a3b8")
    plt.setp(ax_heat.get_yticklabels(), rotation=0, color="#94a3b8")
    ax_heat.figure.axes[-1].tick_params(colors="#94a3b8")
    plt.tight_layout()
    st.pyplot(fig_heat)

# ─────────────────────────────────────────────
# TAB 3 — HOTSPOT INTELLIGENCE
# ─────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📍</span>
        <span class="section-title">Hotspot Intelligence Dashboard</span>
        <span class="section-count">CLUSTER ANALYSIS</span>
    </div>
    """, unsafe_allow_html=True)

    hs_critical = hotspot_df["level"].eq("Critical").sum()
    hs_high = hotspot_df["level"].eq("High").sum()
    hs_medium = hotspot_df["level"].eq("Medium").sum()

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.markdown(f"""<div class="kpi-card red" style="padding:0.9rem;">
            <div class="kpi-icon">🔴</div><div class="kpi-value">{hs_critical}</div>
            <div class="kpi-label">Critical Clusters</div></div>""", unsafe_allow_html=True)
    with h2:
        st.markdown(f"""<div class="kpi-card amber" style="padding:0.9rem;">
            <div class="kpi-icon">🟠</div><div class="kpi-value">{hs_high}</div>
            <div class="kpi-label">High Clusters</div></div>""", unsafe_allow_html=True)
    with h3:
        st.markdown(f"""<div class="kpi-card purple" style="padding:0.9rem;">
            <div class="kpi-icon">🟡</div><div class="kpi-value">{hs_medium}</div>
            <div class="kpi-label">Medium Clusters</div></div>""", unsafe_allow_html=True)
    with h4:
        st.markdown(f"""<div class="kpi-card blue" style="padding:0.9rem;">
            <div class="kpi-icon">📍</div><div class="kpi-value">{len(hotspot_df)}</div>
            <div class="kpi-label">Total Clusters</div></div>""", unsafe_allow_html=True)

    st.markdown('<hr class="cyber-divider">', unsafe_allow_html=True)

    hs_col1, hs_col2 = st.columns([1.5, 1])

    with hs_col1:
        st.markdown("""<div class="section-header">
            <span class="section-icon">📊</span>
            <span class="section-title">Cluster Incident Ranking</span></div>""", unsafe_allow_html=True)

        hs_sorted = hotspot_df.sort_values("incidents", ascending=True)
        hs_colors = {"Critical":"#ef4444","High":"#f97316","Medium":"#fbbf24"}
        bar_c = [hs_colors.get(lv,"#38bdf8") for lv in hs_sorted["level"]]

        fig_hs = go.Figure(go.Bar(
            x=hs_sorted["incidents"],
            y=hs_sorted["cluster"].astype(str).apply(lambda x: f"Cluster {x}"),
            orientation='h',
            marker=dict(color=bar_c, opacity=0.85),
            text=hs_sorted["incidents"],
            textposition="outside",
            textfont=dict(color="white", size=10, family="JetBrains Mono"),
        ))
        fig_hs.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,41,0.6)",
            font=dict(color="#94a3b8", family="Inter"),
            height=350,
            margin=dict(l=10,r=40,t=10,b=20),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(56,189,248,0.08)", color="#64748b", title="Total Incidents"),
            yaxis=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_hs, use_container_width=True, config={"displayModeBar": False})

    with hs_col2:
        st.markdown("""<div class="section-header">
            <span class="section-icon">🥧</span>
            <span class="section-title">Severity Distribution</span></div>""", unsafe_allow_html=True)

        fig_hspie = go.Figure(go.Pie(
            labels=["Critical","High","Medium"],
            values=[hs_critical, hs_high, hs_medium],
            hole=0.5,
            marker=dict(colors=["#ef4444","#f97316","#fbbf24"],
                        line=dict(color="#0a0e1a", width=2)),
            textinfo="percent+label",
            textfont=dict(size=10, color="white"),
        ))
        fig_hspie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            height=350,
            margin=dict(l=10,r=10,t=10,b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_hspie, use_container_width=True, config={"displayModeBar": False})

    # Hotspot Recommendation Cards
    st.markdown("""<div class="section-header">
        <span class="section-icon">🚔</span>
        <span class="section-title">Deployment Recommendations by Cluster</span></div>""", unsafe_allow_html=True)

    level_cls = {"Critical": "critical", "High": "high", "Medium": "medium"}
    level_icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡"}

    for _, row in hotspot_df.iterrows():
        lvl = row.get("level","Medium")
        cls = level_cls.get(lvl,"medium")
        icon = level_icon.get(lvl,"🟡")
        st.markdown(f"""
        <div class="hotspot-card {cls}">
            <div>
                <div style="font-size:0.85rem;font-weight:700;color:#e2e8f0;">
                    {icon} Cluster {row['cluster']}
                    <span class="risk-badge {cls}" style="margin-left:0.5rem;font-size:0.65rem;">{lvl.upper()}</span>
                </div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:0.2rem;">{row['incidents']} incidents recorded</div>
            </div>
            <div style="display:flex;gap:1.5rem;text-align:center;">
                <div><div style="font-size:1.1rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:#38bdf8;">{row['officers']}</div>
                    <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;">Officers</div></div>
                <div><div style="font-size:1.1rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:#818cf8;">{row['barricades']}</div>
                    <div style="font-size:0.6rem;color:#64748b;text-transform:uppercase;">Barricades</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 4 — HOTSPOT MAP
# ─────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🗺️</span>
        <span class="section-title">Bengaluru Live Hotspot Map</span>
        <span class="section-count">GEOSPATIAL INTELLIGENCE</span>
    </div>
    """, unsafe_allow_html=True)

    map_col, legend_col = st.columns([3, 1])

    with legend_col:
        total_incidents = hotspot_locations["incidents"].sum() if "incidents" in hotspot_locations.columns else "—"
        crit_spots = (hotspot_locations["incidents"] >= 1000).sum() if "incidents" in hotspot_locations.columns else 0
        high_spots = ((hotspot_locations["incidents"] >= 300) & (hotspot_locations["incidents"] < 1000)).sum() if "incidents" in hotspot_locations.columns else 0
        med_spots = (hotspot_locations["incidents"] < 300).sum() if "incidents" in hotspot_locations.columns else 0

        st.markdown(f"""
        <div class="map-legend">
            <div style="font-size:0.7rem;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">📍 Map Legend</div>
            <div class="legend-row"><div class="legend-dot red"></div> Critical ≥1000 incidents</div>
            <div class="legend-row"><div class="legend-dot orange"></div> High ≥300 incidents</div>
            <div class="legend-row"><div class="legend-dot blue"></div> Medium &lt;300 incidents</div>
        </div>
        <div class="map-legend" style="margin-top:0.8rem;">
            <div style="font-size:0.7rem;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">📊 Cluster Summary</div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.5rem;">
                <span style="color:#ef4444;font-weight:700;">{crit_spots}</span> Critical Zones
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.5rem;">
                <span style="color:#f97316;font-weight:700;">{high_spots}</span> High Risk Zones
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.5rem;">
                <span style="color:#38bdf8;font-weight:700;">{med_spots}</span> Medium Risk Zones
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid rgba(56,189,248,0.1);">
                Total Incidents:<br><span style="font-size:1.2rem;font-weight:800;font-family:'JetBrains Mono',monospace;color:#38bdf8;">{total_incidents:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with map_col:
        m = folium.Map(
            location=[12.97, 77.59],
            zoom_start=11,
            tiles="CartoDB dark_matter"
        )

        for _, row in hotspot_locations.iterrows():
            if row["incidents"] >= 1000:
                color, level = "red", "Critical"
            elif row["incidents"] >= 300:
                color, level = "orange", "High"
            else:
                color, level = "#38bdf8", "Medium"

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=min(row["incidents"] / 50, 20),
                popup=folium.Popup(
                    f"<b>Cluster: {row['cluster']}</b><br>Incidents: {row['incidents']}<br>Level: {level}",
                    max_width=200
                ),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=2,
            ).add_to(m)

        st_folium(m, width=None, height=560, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 5 — COMMAND CENTER
# ─────────────────────────────────────────────
with tab5:
    # ── Compute data ──
    actionable_df = command_center_df[command_center_df["unified_risk_score"] >= 50]
    critical_count = actionable_df["unified_level"].eq("CRITICAL").sum()
    high_count = actionable_df["unified_level"].eq("HIGH").sum()
    medium_count = actionable_df["unified_level"].eq("MEDIUM").sum()
    actionable_slots = len(actionable_df)
    total_officers = critical_count * 11 + high_count * 7 + medium_count * 3
    total_barricades = critical_count * 6 + high_count * 4 + medium_count * 2

    # ── Alert Banner ──
    st.markdown(f"""
    <div class="alert-banner">
        <div class="alert-banner-icon">🚨</div>
        <div class="alert-banner-text">
            <div class="alert-banner-title">EXECUTIVE ALERT — ACTIVE OPERATIONS MODE</div>
            <div class="alert-banner-sub">
                {actionable_slots} actionable windows identified across Bengaluru corridors — 
                {critical_count} critical alerts require immediate deployment
            </div>
        </div>
        <span class="risk-badge critical">● LIVE</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Top KPIs ──
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    cards = [
        (k1, "blue", "🚨", actionable_slots, "Actionable Windows"),
        (k2, "red", "🔴", critical_count, "Critical Alerts"),
        (k3, "amber", "🟠", high_count, "High Risk Alerts"),
        (k4, "purple", "🟡", medium_count, "Medium Alerts"),
        (k5, "green", "🚔", total_officers, "Officers Required"),
        (k6, "pink", "🚧", total_barricades, "Barricades Required"),
    ]
    for col, cls, icon, val, label in cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}" style="padding:0.9rem;">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="cyber-divider">', unsafe_allow_html=True)

    cc_col1, cc_col2 = st.columns([1, 1.6])

    with cc_col1:
        st.markdown("""<div class="section-header">
            <span class="section-icon">🥧</span>
            <span class="section-title">Risk Distribution</span></div>""", unsafe_allow_html=True)

        fig_cc_pie = go.Figure(go.Pie(
            labels=["CRITICAL","HIGH","MEDIUM"],
            values=[critical_count, high_count, medium_count],
            hole=0.55,
            marker=dict(colors=["#ef4444","#f97316","#fbbf24"],
                        line=dict(color="#0a0e1a", width=3)),
            textinfo="label+percent",
            textfont=dict(size=11, color="white", family="Inter"),
            pull=[0.05, 0, 0],
        ))
        fig_cc_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            height=300,
            margin=dict(l=10,r=10,t=10,b=10),
            showlegend=False,
            annotations=[dict(text=f"<b>{actionable_slots}</b><br>Total", x=0.5, y=0.5,
                              font=dict(size=14,color="#e2e8f0",family="JetBrains Mono"),
                              showarrow=False)]
        )
        st.plotly_chart(fig_cc_pie, use_container_width=True, config={"displayModeBar": False})

        # Resource allocation
        st.markdown("""<div class="section-header">
            <span class="section-icon">📦</span>
            <span class="section-title">Resource Allocation</span></div>""", unsafe_allow_html=True)

        fig_res = go.Figure()
        categories = ["CRITICAL","HIGH","MEDIUM"]
        officer_vals = [critical_count*11, high_count*7, medium_count*3]
        barricade_vals = [critical_count*6, high_count*4, medium_count*2]

        fig_res.add_trace(go.Bar(name="Officers", x=categories, y=officer_vals,
            marker_color="#38bdf8", opacity=0.85,
            text=officer_vals, textposition="outside",
            textfont=dict(color="white", size=10, family="JetBrains Mono")))
        fig_res.add_trace(go.Bar(name="Barricades", x=categories, y=barricade_vals,
            marker_color="#818cf8", opacity=0.85,
            text=barricade_vals, textposition="outside",
            textfont=dict(color="white", size=10, family="JetBrains Mono")))

        fig_res.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,41,0.6)",
            font=dict(color="#94a3b8", family="Inter"),
            height=260,
            margin=dict(l=10,r=10,t=10,b=20),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(color="#64748b"),
            yaxis=dict(gridcolor="rgba(56,189,248,0.08)", color="#64748b"),
        )
        st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})

    with cc_col2:
        st.markdown("""<div class="section-header">
            <span class="section-icon">🏆</span>
            <span class="section-title">Top Deployment Priorities</span>
            <span class="section-count">TOP 15</span></div>""", unsafe_allow_html=True)

        top15 = (
            actionable_df[[
                "corridor","hour","unified_risk_score","unified_level",
                "officers_required","barricades_required","suggested_diversion","action_priority"
            ]]
            .sort_values("unified_risk_score", ascending=False)
            .head(15)
        )
        st.dataframe(top15, use_container_width=True, hide_index=True)

        # Highest Risk Corridor
        highest_risk = actionable_df.sort_values("unified_risk_score", ascending=False).iloc[0]
        st.markdown(f"""
        <div class="result-card critical" style="margin-top:1rem;">
            <div style="font-size:0.7rem;font-weight:700;color:#ef4444;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">📌 Highest Risk Corridor — Immediate Action Required</div>
            <div style="font-size:1.1rem;font-weight:800;color:#e2e8f0;margin-bottom:0.6rem;">{highest_risk['corridor']}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;font-size:0.8rem;">
                <div><span style="color:#64748b;">Peak Hour:</span> <strong style="color:#e2e8f0;font-family:'JetBrains Mono',monospace;">{int(highest_risk['hour']):02d}:00</strong></div>
                <div><span style="color:#64748b;">Risk Score:</span> <strong style="color:#ef4444;font-family:'JetBrains Mono',monospace;">{highest_risk['unified_risk_score']:.1f}</strong></div>
                <div><span style="color:#64748b;">Officers:</span> <strong style="color:#38bdf8;">{highest_risk['officers_required']}</strong></div>
                <div><span style="color:#64748b;">Barricades:</span> <strong style="color:#818cf8;">{highest_risk['barricades_required']}</strong></div>
                <div style="grid-column:span 2;"><span style="color:#64748b;">Priority:</span> <strong style="color:#fbbf24;">{highest_risk['action_priority']}</strong></div>
                <div style="grid-column:span 2;"><span style="color:#64748b;">Diversion:</span> <strong style="color:#f472b6;">{highest_risk['suggested_diversion']}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 6 — DAILY BRIEFING
# ─────────────────────────────────────────────
with tab6:
    briefing_df = (
        command_center_df[command_center_df["unified_risk_score"] >= 50]
        .sort_values("unified_risk_score", ascending=False)
    )

    # Recompute for tab6 (tab5 vars may or may not be in scope)
    _actionable = command_center_df[command_center_df["unified_risk_score"] >= 50]
    _critical = _actionable["unified_level"].eq("CRITICAL").sum()
    _high = _actionable["unified_level"].eq("HIGH").sum()
    _medium = _actionable["unified_level"].eq("MEDIUM").sum()
    _officers = _critical * 11 + _high * 7 + _medium * 3
    _barricades = _critical * 6 + _high * 4 + _medium * 2
    top_p = briefing_df.iloc[0]

    st.markdown(f"""
    <div class="briefing-header">
        <div style="font-size:0.65rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.15em;margin-bottom:0.4rem;">
            TRAFFICOPS AI — OPERATIONAL BRIEFING DOCUMENT
        </div>
        <div class="briefing-title">🗒️ Daily Traffic Operations Briefing</div>
        <div class="briefing-meta">
            Generated: Today &nbsp;|&nbsp; Coverage: Bengaluru Metropolitan Area &nbsp;|&nbsp; 
            Classification: OPERATIONAL USE ONLY &nbsp;|&nbsp; Version: v3.0-ML
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary Stats
    b1,b2,b3,b4,b5 = st.columns(5)
    briefing_cards = [
        (b1,"red","🔴",_critical,"Critical Windows"),
        (b2,"amber","🟠",_high,"High Risk Windows"),
        (b3,"purple","🟡",_medium,"Medium Windows"),
        (b4,"green","🚔",_officers,"Officers Required"),
        (b5,"pink","🚧",_barricades,"Barricades Required"),
    ]
    for col,cls,icon,val,label in briefing_cards:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}" style="padding:0.9rem;">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="cyber-divider">', unsafe_allow_html=True)

    br_col1, br_col2 = st.columns([1.5, 1])

    with br_col1:
        st.markdown("""<div class="section-header">
            <span class="section-icon">🚨</span>
            <span class="section-title">Top 10 Priority Deployment Corridors</span></div>""", unsafe_allow_html=True)

        top10_brief = briefing_df[[
            "corridor","hour","unified_level","officers_required","barricades_required","suggested_diversion"
        ]].head(10)
        st.dataframe(top10_brief, use_container_width=True, hide_index=True)

        # Bar chart
        st.markdown("""<div class="section-header" style="margin-top:1rem;">
            <span class="section-icon">📊</span>
            <span class="section-title">Priority Corridor Risk Scores</span></div>""", unsafe_allow_html=True)

        top10_scores = briefing_df.head(10)
        level_colors_map = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#fbbf24"}
        bar_cols_brief = [level_colors_map.get(l,"#38bdf8") for l in top10_scores["unified_level"]]

        fig_brief = go.Figure(go.Bar(
            x=top10_scores["corridor"],
            y=top10_scores["unified_risk_score"],
            marker=dict(color=bar_cols_brief, opacity=0.85),
            text=top10_scores["unified_risk_score"].round(1),
            textposition="outside",
            textfont=dict(color="white", size=10, family="JetBrains Mono"),
        ))
        fig_brief.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,41,0.6)",
            font=dict(color="#94a3b8", family="Inter"),
            height=260,
            margin=dict(l=10,r=10,t=10,b=60),
            showlegend=False,
            xaxis=dict(color="#64748b", tickangle=-35),
            yaxis=dict(gridcolor="rgba(56,189,248,0.08)", color="#64748b", title="Risk Score"),
        )
        st.plotly_chart(fig_brief, use_container_width=True, config={"displayModeBar": False})

    with br_col2:
        st.markdown("""<div class="section-header">
            <span class="section-icon">🎯</span>
            <span class="section-title">Operational Recommendation</span></div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-card critical">
            <div style="font-size:0.7rem;font-weight:700;color:#ef4444;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">⚠ Priority One — Immediate Deployment</div>
            <div style="font-size:1rem;font-weight:800;color:#e2e8f0;margin-bottom:0.8rem;">{top_p['corridor']}</div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.4rem;">
                <span style="color:#64748b;">Peak Hour:</span> <strong style="font-family:'JetBrains Mono',monospace;">{int(top_p['hour']):02d}:00</strong>
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.4rem;">
                <span style="color:#64748b;">Risk Level:</span> <span class="risk-badge critical">{top_p['unified_level']}</span>
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.4rem;">
                <span style="color:#64748b;">Officers:</span> <strong style="color:#38bdf8;">{top_p['officers_required']}</strong>
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.4rem;">
                <span style="color:#64748b;">Barricades:</span> <strong style="color:#818cf8;">{top_p['barricades_required']}</strong>
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;">
                <span style="color:#64748b;">Diversion:</span> <strong style="color:#f472b6;">{top_p['suggested_diversion']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk summary box
        st.markdown(f"""
        <div class="result-card info" style="margin-top:0.8rem;">
            <div style="font-size:0.7rem;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">📋 Executive Risk Summary</div>
            <div class="stat-row" style="flex-direction:column;gap:0.4rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:0.3rem 0;border-bottom:1px solid rgba(56,189,248,0.08);">
                    <span style="color:#64748b;">Total Actionable Windows</span>
                    <strong style="color:#e2e8f0;font-family:'JetBrains Mono',monospace;">{len(briefing_df)}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:0.3rem 0;border-bottom:1px solid rgba(56,189,248,0.08);">
                    <span style="color:#64748b;">Critical Deployments</span>
                    <strong style="color:#ef4444;font-family:'JetBrains Mono',monospace;">{_critical}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:0.3rem 0;border-bottom:1px solid rgba(56,189,248,0.08);">
                    <span style="color:#64748b;">High Risk Deployments</span>
                    <strong style="color:#f97316;font-family:'JetBrains Mono',monospace;">{_high}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:0.3rem 0;border-bottom:1px solid rgba(56,189,248,0.08);">
                    <span style="color:#64748b;">Medium Risk Deployments</span>
                    <strong style="color:#fbbf24;font-family:'JetBrains Mono',monospace;">{_medium}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:0.3rem 0;border-bottom:1px solid rgba(56,189,248,0.08);">
                    <span style="color:#64748b;">Total Officers Required</span>
                    <strong style="color:#38bdf8;font-family:'JetBrains Mono',monospace;">{_officers}</strong>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;padding:0.3rem 0;">
                    <span style="color:#64748b;">Total Barricades Required</span>
                    <strong style="color:#818cf8;font-family:'JetBrains Mono',monospace;">{_barricades}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Operational recommendations
        st.markdown("""
        <div class="result-card low" style="margin-top:0.8rem;">
            <div style="font-size:0.7rem;font-weight:700;color:#22c55e;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">✅ Operational Recommendations</div>
            <div style="font-size:0.78rem;color:#94a3b8;line-height:1.7;">
                → Deploy additional units to critical corridors by 06:00<br>
                → Activate diversion protocols for top 3 corridors<br>
                → Coordinate with BBMP for construction zone management<br>
                → Pre-position barricades before morning rush hours<br>
                → Enable real-time monitoring for hotspot clusters
            </div>
        </div>
        """, unsafe_allow_html=True)