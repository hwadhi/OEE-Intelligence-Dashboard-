import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import io

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="OEEiq — Manufacturing Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# GLOBAL CSS — Mobile + Desktop
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Mobile first base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit defaults ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Top brand bar ── */
.brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 20px 0;
    border-bottom: 1px solid #6C63FF30;
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 8px;
}
.brand-logo {
    font-size: 22px;
    font-weight: 800;
    color: #6C63FF;
    letter-spacing: -0.5px;
}
.brand-tagline {
    font-size: 12px;
    color: #888;
    margin-top: 2px;
}
.brand-badge {
    background: #6C63FF20;
    color: #6C63FF;
    border: 1px solid #6C63FF40;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
}

/* ── KPI Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1A1D2E 0%, #1E2240 100%);
    border: 1px solid #6C63FF25;
    border-radius: 16px;
    padding: 20px 16px;
    box-shadow: 0 4px 20px rgba(108,99,255,0.08);
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover {
    border-color: #6C63FF60;
}
[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #FAFAFA !important;
}
[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* ── Section headers ── */
.section-header {
    font-size: 14px;
    font-weight: 600;
    color: #6C63FF;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 24px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── OEE Score Ring ── */
.oee-ring-container {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1A1D2E, #1E2240);
    border: 1px solid #6C63FF25;
    border-radius: 16px;
}
.oee-score {
    font-size: 52px;
    font-weight: 800;
    color: #6C63FF;
    line-height: 1;
}
.oee-label {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.oee-status {
    display: inline-block;
    margin-top: 8px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

/* ── Alert cards ── */
.alert-card {
    background: #FF4B4B10;
    border: 1px solid #FF4B4B30;
    border-left: 3px solid #FF4B4B;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 13px;
}
.alert-card .machine {
    font-weight: 700;
    color: #FF4B4B;
}
.alert-card .detail {
    color: #aaa;
    font-size: 12px;
    margin-top: 2px;
}

/* ── Loss pill ── */
.loss-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
.loss-availability { background:#FF4B4B20; color:#FF4B4B; }
.loss-performance  { background:#FFB84B20; color:#FFB84B; }
.loss-quality      { background:#4B9FFF20; color:#4B9FFF; }
.loss-none         { background:#4BFF9120; color:#4BFF91; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D0F1A !important;
    border-right: 1px solid #6C63FF20;
}
.sidebar-brand {
    text-align: center;
    padding: 20px 0 10px;
    border-bottom: 1px solid #6C63FF20;
    margin-bottom: 16px;
}
.sidebar-brand h2 {
    color: #6C63FF;
    font-size: 24px;
    font-weight: 800;
    margin: 0;
}
.sidebar-brand p {
    color: #666;
    font-size: 11px;
    margin: 4px 0 0;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .brand-logo { font-size: 18px; }
    .oee-score  { font-size: 40px; }
    [data-testid="stMetricValue"] { font-size: 20px !important; }
    [data-testid="metric-container"] { padding: 14px 10px; }
    .block-container { padding: 1rem !important; }
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #6C63FF40 !important;
    border-radius: 12px !important;
    background: #6C63FF08 !important;
}

/* ── Plotly chart background ── */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #444;
    font-size: 11px;
    padding: 32px 0 16px;
    border-top: 1px solid #6C63FF15;
    margin-top: 40px;
}
.footer a { color: #6C63FF; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def calculate_oee(df):
    df = df.copy()
    df['Availability'] = (
        (df['Planned_Production_Time_mins'] - df['Downtime_mins']) /
        df['Planned_Production_Time_mins']
    ).clip(0, 1)

    df['Performance'] = (
        (df['Total_Parts_Produced'] * df['Ideal_Cycle_Time_sec']) /
        ((df['Planned_Production_Time_mins'] - df['Downtime_mins']) * 60)
    ).clip(0, 1)

    df['Quality'] = (
        df['Good_Parts'] / df['Total_Parts_Produced']
    ).clip(0, 1)

    df['OEE'] = (df['Availability'] * df['Performance'] * df['Quality'] * 100).round(2)
    df['Availability_pct'] = (df['Availability'] * 100).round(2)
    df['Performance_pct']  = (df['Performance']  * 100).round(2)
    df['Quality_pct']      = (df['Quality']       * 100).round(2)

    def classify(row):
        if row['Availability'] < 0.85: return 'Availability Loss'
        if row['Performance']  < 0.85: return 'Performance Loss'
        if row['Quality']      < 0.99: return 'Quality Loss'
        return 'No Significant Loss'

    df['Primary_Loss'] = df.apply(classify, axis=1)

    def category(v):
        if v >= 85: return '🟢 World Class'
        if v >= 65: return '🟡 Average'
        return '🔴 Below Average'

    df['OEE_Category'] = df['OEE'].apply(category)
    return df


def add_anomaly(df):
    features = ['OEE','Availability_pct','Performance_pct','Quality_pct','Downtime_mins']
    X = StandardScaler().fit_transform(df[features])
    df['Anomaly_Score'] = IsolationForest(contamination=0.1, random_state=42).fit_predict(X)
    df['Is_Anomaly'] = df['Anomaly_Score'].apply(lambda x: '⚠️ Anomaly' if x == -1 else '✅ Normal')
    return df


def add_rupee_loss(df, part_value, parts_per_min):
    df = df.copy()
    df['Quality_Loss_Rs']  = df['Rejected_Parts'] * part_value
    df['Downtime_Loss_Rs'] = df['Downtime_mins'] * parts_per_min * part_value
    df['Total_Loss_Rs']    = df['Quality_Loss_Rs'] + df['Downtime_Loss_Rs']
    return df


@st.cache_data
def load_sample():
    import random
    from datetime import timedelta
    random.seed(42); np.random.seed(42)
    rows = []
    base = datetime(2026, 7, 1)
    for day in range(90):
        d = base + timedelta(days=day)
        for shift in ['A','B','C']:
            for machine in ['M1-HPDC','M2-HPDC','M3-HPDC','M4-HPDC']:
                pt = 480
                dt = random.randint(60,150) if machine=='M3-HPDC' else \
                     random.randint(40,100)  if shift=='C'         else \
                     random.randint(10,60)
                ict = random.uniform(28,35)
                avail = pt - dt
                total = int((avail*60/ict)*random.uniform(0.65,0.95))
                rej   = int(total * random.uniform(0.02,0.12))
                rows.append({
                    'Date': d.strftime('%Y-%m-%d'),
                    'Shift': shift,
                    'Machine_ID': machine,
                    'Planned_Production_Time_mins': pt,
                    'Downtime_mins': dt,
                    'Ideal_Cycle_Time_sec': round(ict,2),
                    'Total_Parts_Produced': total,
                    'Rejected_Parts': rej,
                    'Good_Parts': total - rej
                })
    df = pd.DataFrame(rows)
    df = calculate_oee(df)
    df = add_anomaly(df)
    return df


CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#FAFAFA', size=12),
    margin=dict(l=10, r=10, t=36, b=10),
)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🏭 OEEiq</h2>
        <p>Manufacturing Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📁 Data Source**")
    uploaded = st.file_uploader(
        "Upload shift data (CSV / Excel)",
        type=['csv','xlsx'],
        help="Columns needed: Date, Shift, Machine_ID, Planned_Production_Time_mins, Downtime_mins, Ideal_Cycle_Time_sec, Total_Parts_Produced, Rejected_Parts, Good_Parts"
    )

    if not uploaded:
        st.info("No file uploaded — using sample HPDC data (90 days, 4 machines, 3 shifts)")

    st.markdown("---")
    st.markdown("**🔧 Filters**")

    machines_all = ['M1-HPDC','M2-HPDC','M3-HPDC','M4-HPDC']
    sel_machines = st.multiselect("Machine", machines_all, default=machines_all)
    sel_shifts   = st.multiselect("Shift",   ['A','B','C'], default=['A','B','C'])

    st.markdown("---")
    st.markdown("**₹ Loss Calculator**")
    part_value   = st.number_input("Part value (₹)", min_value=10, value=250, step=10)
    parts_per_min = st.number_input("Ideal parts/min", min_value=0.1, value=2.0, step=0.1)

    st.markdown("---")
    st.markdown("**🎯 OEE Targets**")
    target_oee   = st.slider("World class target %", 70, 95, 85)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#444;text-align:center'>
        OEEiq v1.0 BETA<br>
        Built by Heramb | VJTI'27<br>
        <a href='mailto:heramb@vjti.ac.in' style='color:#6C63FF'>
        Request a demo</a>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
if uploaded:
    try:
        raw = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') \
              else pd.read_excel(uploaded)
        df  = calculate_oee(raw)
        df  = add_anomaly(df)
        st.success("✅ Your data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}. Using sample data instead.")
        df = load_sample()
else:
    df = load_sample()

# Apply filters
fdf = df[
    df['Machine_ID'].isin(sel_machines) &
    df['Shift'].isin(sel_shifts)
].copy()

# Add rupee loss
fdf = add_rupee_loss(fdf, part_value, parts_per_min)


# ─────────────────────────────────────────
# BRAND BAR
# ─────────────────────────────────────────
st.markdown("""
<div class="brand-bar">
    <div>
        <div class="brand-logo">🏭 OEEiq</div>
        <div class="brand-tagline">Manufacturing Intelligence Platform</div>
    </div>
    <div class="brand-badge">BETA v1.0</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# COMPUTED METRICS
# ─────────────────────────────────────────
avg_oee   = fdf['OEE'].mean()
avg_avail = fdf['Availability_pct'].mean()
avg_perf  = fdf['Performance_pct'].mean()
avg_qual  = fdf['Quality_pct'].mean()
anomalies = (fdf['Anomaly_Score'] == -1).sum()
total_loss_rs   = fdf['Total_Loss_Rs'].sum()
downtime_loss_rs = fdf['Downtime_Loss_Rs'].sum()
quality_loss_rs  = fdf['Quality_Loss_Rs'].sum()

oee_gap   = avg_oee - target_oee
gap_color = "#4BFF91" if oee_gap >= 0 else "#FF4B4B"
gap_label = f"+{oee_gap:.1f}% vs target" if oee_gap >= 0 else f"{oee_gap:.1f}% vs target"

if avg_oee >= 85:   status_text, status_bg = "World Class", "#4BFF9120"
elif avg_oee >= 65: status_text, status_bg = "Average",     "#FFB84B20"
else:               status_text, status_bg = "Below Average","#FF4B4B20"


# ─────────────────────────────────────────
# ROW 1 — OEE SCORE + CORE KPIs
# ─────────────────────────────────────────
st.markdown('<div class="section-header">📊 Performance Overview</div>', unsafe_allow_html=True)

col_ring, col_kpis = st.columns([1, 3])

with col_ring:
    st.markdown(f"""
    <div class="oee-ring-container">
        <div class="oee-score">{avg_oee:.1f}%</div>
        <div class="oee-label">Overall OEE</div>
        <div class="oee-status" style="background:{status_bg};
             color:{'#4BFF91' if avg_oee>=85 else '#FFB84B' if avg_oee>=65 else '#FF4B4B'}">
            {status_text}
        </div>
        <div style="font-size:12px;color:{gap_color};margin-top:8px">
            {gap_label}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_kpis:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Availability",  f"{avg_avail:.1f}%", f"{'✅' if avg_avail>=85 else '⚠️'} {'OK' if avg_avail>=85 else 'Low'}")
    k2.metric("Performance",   f"{avg_perf:.1f}%",  f"{'✅' if avg_perf>=85  else '⚠️'} {'OK' if avg_perf>=85  else 'Low'}")
    k3.metric("Quality",       f"{avg_qual:.1f}%",  f"{'✅' if avg_qual>=99  else '⚠️'} {'OK' if avg_qual>=99  else 'Low'}")
    k4.metric("⚠️ Anomalies",  f"{anomalies}",      "shifts flagged")


# ─────────────────────────────────────────
# ROW 2 — RUPEE LOSS CARDS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">₹ Loss Intelligence</div>', unsafe_allow_html=True)

l1, l2, l3 = st.columns(3)
l1.metric("Total Loss",      f"₹{total_loss_rs:,.0f}",    "This period")
l2.metric("Downtime Loss",   f"₹{downtime_loss_rs:,.0f}", f"{downtime_loss_rs/max(total_loss_rs,1)*100:.0f}% of total")
l3.metric("Quality Loss",    f"₹{quality_loss_rs:,.0f}",  f"{quality_loss_rs/max(total_loss_rs,1)*100:.0f}% of total")


# ─────────────────────────────────────────
# ROW 3 — TREND + MACHINE CHART
# ─────────────────────────────────────────
st.markdown('<div class="section-header">📈 Trends & Comparisons</div>', unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])

with c1:
    daily = fdf.groupby('Date')['OEE'].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['Date'], y=daily['OEE'],
        fill='tozeroy',
        fillcolor='rgba(108,99,255,0.08)',
        line=dict(color='#6C63FF', width=2),
        name='Daily OEE'
    ))
    fig.add_hline(y=target_oee, line_dash="dash",
                  line_color="#4BFF91", line_width=1,
                  annotation_text=f"Target {target_oee}%",
                  annotation_font_color="#4BFF91")
    fig.add_hline(y=65, line_dash="dot",
                  line_color="#FF4B4B", line_width=1,
                  annotation_text="Min 65%",
                  annotation_font_color="#FF4B4B")
    fig.update_layout(**CHART_LAYOUT, title="Daily OEE Trend",
                      xaxis=dict(showgrid=False),
                      yaxis=dict(showgrid=True, gridcolor='#FFFFFF08', range=[0,100]))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    m_oee = fdf.groupby('Machine_ID')['OEE'].mean().reset_index().sort_values('OEE')
    colors = ['#FF4B4B' if v < 65 else '#FFB84B' if v < target_oee else '#4BFF91'
              for v in m_oee['OEE']]
    fig2 = go.Figure(go.Bar(
        x=m_oee['OEE'], y=m_oee['Machine_ID'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.1f}%" for v in m_oee['OEE']],
        textposition='outside'
    ))
    fig2.update_layout(**CHART_LAYOUT, title="OEE by Machine",
                       xaxis=dict(range=[0,110], showgrid=False),
                       yaxis=dict(showgrid=False))
    st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────
# ROW 4 — SHIFT + LOSS PIE
# ─────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    s_oee = fdf.groupby('Shift')['OEE'].mean().reset_index()
    s_colors = ['#FF4B4B' if v < 65 else '#FFB84B' if v < target_oee else '#4BFF91'
                for v in s_oee['OEE']]
    fig3 = go.Figure(go.Bar(
        x=s_oee['Shift'], y=s_oee['OEE'],
        marker_color=s_colors,
        text=[f"{v:.1f}%" for v in s_oee['OEE']],
        textposition='outside'
    ))
    fig3.add_hline(y=target_oee, line_dash="dash", line_color="#4BFF91", line_width=1)
    fig3.update_layout(**CHART_LAYOUT, title="OEE by Shift",
                       yaxis=dict(range=[0,110], showgrid=True, gridcolor='#FFFFFF08'),
                       xaxis=dict(showgrid=False))
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    loss_dist = fdf['Primary_Loss'].value_counts().reset_index()
    loss_dist.columns = ['Loss_Type','Count']
    loss_colors = {
        'Availability Loss': '#FF4B4B',
        'Performance Loss':  '#FFB84B',
        'Quality Loss':      '#4B9FFF',
        'No Significant Loss':'#4BFF91'
    }
    fig4 = go.Figure(go.Pie(
        labels=loss_dist['Loss_Type'],
        values=loss_dist['Count'],
        marker_colors=[loss_colors.get(l,'#888') for l in loss_dist['Loss_Type']],
        hole=0.5,
        textinfo='percent+label',
        textfont_size=11
    ))
    fig4.update_layout(**CHART_LAYOUT, title="Loss Distribution",
                       showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────
# ROW 5 — ANOMALY ALERTS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">⚠️ ML Anomaly Alerts</div>', unsafe_allow_html=True)

anomaly_df = fdf[fdf['Anomaly_Score'] == -1].sort_values('OEE').head(10)

if anomaly_df.empty:
    st.success("✅ No anomalous shifts detected in selected filters.")
else:
    a1, a2 = st.columns([2, 1])
    with a1:
        for _, row in anomaly_df.iterrows():
            st.markdown(f"""
            <div class="alert-card">
                <span class="machine">
                    {row['Machine_ID']} — Shift {row['Shift']}
                </span>
                &nbsp;|&nbsp; OEE: <b>{row['OEE']:.1f}%</b>
                &nbsp;|&nbsp; {row['Date']}
                <div class="detail">
                    Primary issue: {row['Primary_Loss']} &nbsp;|&nbsp;
                    Downtime: {row['Downtime_mins']} mins
                </div>
            </div>
            """, unsafe_allow_html=True)
    with a2:
        total_anomalies = len(anomaly_df)
        worst = anomaly_df.iloc[0]
        st.markdown(f"""
        <div style='background:#FF4B4B10;border:1px solid #FF4B4B30;
        border-radius:12px;padding:16px;text-align:center'>
            <div style='font-size:32px;font-weight:800;color:#FF4B4B'>
                {total_anomalies}
            </div>
            <div style='font-size:11px;color:#888;margin-top:4px'>
                Anomalous shifts
            </div>
            <hr style='border-color:#FF4B4B20;margin:12px 0'>
            <div style='font-size:12px;color:#aaa'>
                Worst: {worst['Machine_ID']}<br>
                OEE: {worst['OEE']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# ROW 6 — BENCHMARK TABLE
# ─────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Benchmark vs World Class</div>', unsafe_allow_html=True)

bench = pd.DataFrame({
    'Metric':       ['OEE', 'Availability', 'Performance', 'Quality'],
    'Your Line':    [f"{avg_oee:.1f}%", f"{avg_avail:.1f}%", f"{avg_perf:.1f}%", f"{avg_qual:.1f}%"],
    'World Class':  ['85%', '90%', '95%', '99.9%'],
    'Gap':          [
        f"{avg_oee-85:.1f}%", f"{avg_avail-90:.1f}%",
        f"{avg_perf-95:.1f}%", f"{avg_qual-99.9:.1f}%"
    ],
    'Status': [
        '✅' if avg_oee>=85   else '⚠️',
        '✅' if avg_avail>=90 else '⚠️',
        '✅' if avg_perf>=95  else '⚠️',
        '✅' if avg_qual>=99.9 else '⚠️'
    ]
})
st.dataframe(bench, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────
# ROW 7 — DOWNLOAD
# ─────────────────────────────────────────
st.markdown('<div class="section-header">📥 Export</div>', unsafe_allow_html=True)

d1, d2 = st.columns(2)
with d1:
    csv_data = fdf.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Full Report (CSV)",
        data=csv_data,
        file_name=f"OEEiq_Report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
        use_container_width=True
    )
with d2:
    anomaly_csv = anomaly_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⚠️ Download Anomaly Report (CSV)",
        data=anomaly_csv,
        file_name=f"OEEiq_Anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv',
        use_container_width=True
    )


# ─────────────────────────────────────────
# RAW DATA EXPANDER
# ─────────────────────────────────────────
with st.expander("📋 View Raw Shift Data"):
    st.dataframe(
        fdf[['Date','Shift','Machine_ID','OEE',
             'Availability_pct','Performance_pct','Quality_pct',
             'Primary_Loss','Downtime_mins','Is_Anomaly',
             'Total_Loss_Rs']].sort_values('OEE'),
        use_container_width=True,
        hide_index=True
    )


# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    🏭 <b>OEEiq</b> — Manufacturing Intelligence Platform &nbsp;|&nbsp;
    Built by <b>Heramb</b> | Production Engineering | VJTI Mumbai '27 &nbsp;|&nbsp;
    Dataset: HPDC Line Observations (CIE Automotive) &nbsp;|&nbsp;
    <a href="mailto:wadhiheramb@gmail.com">Request live demo</a>
</div>
""", unsafe_allow_html=True)
