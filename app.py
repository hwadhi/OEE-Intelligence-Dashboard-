import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="OEE Intelligence Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
}
.stMetric {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('oee_final.csv')
    return df

df = load_data()

# Sidebar
st.sidebar.image(
    "https://img.icons8.com/fluency/96/factory.png", 
    width=80
)
st.sidebar.title("🏭 OEE Dashboard")
st.sidebar.markdown("**HPDC Manufacturing Line**")
st.sidebar.markdown("---")

# Filters
selected_machines = st.sidebar.multiselect(
    "Select Machine",
    options=df['Machine_ID'].unique(),
    default=df['Machine_ID'].unique()
)

selected_shifts = st.sidebar.multiselect(
    "Select Shift",
    options=['A', 'B', 'C'],
    default=['A', 'B', 'C']
)

# Filter data
filtered_df = df[
    (df['Machine_ID'].isin(selected_machines)) &
    (df['Shift'].isin(selected_shifts))
]

# Header
st.title("🏭 OEE Intelligence Dashboard")
st.markdown("**Real-time Manufacturing Performance | HPDC Line**")
st.markdown("---")

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)

avg_oee = filtered_df['OEE'].mean()
avg_avail = filtered_df['Availability_pct'].mean()
avg_perf = filtered_df['Performance_pct'].mean()
avg_qual = filtered_df['Quality_pct'].mean()
anomalies = (filtered_df['Anomaly_Score'] == -1).sum()

col1.metric(
    "Overall OEE", 
    f"{avg_oee:.1f}%",
    f"{avg_oee - 85:.1f}% vs World Class"
)
col2.metric("Availability", f"{avg_avail:.1f}%")
col3.metric("Performance", f"{avg_perf:.1f}%")
col4.metric("Quality", f"{avg_qual:.1f}%")
col5.metric("⚠️ Anomalies", f"{anomalies}", "shifts flagged")

st.markdown("---")

# Row 2 — Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Daily OEE Trend")
    daily_oee = filtered_df.groupby('Date')['OEE'].mean().reset_index()
    fig = px.line(
        daily_oee, x='Date', y='OEE',
        title='Daily Average OEE (%)'
    )
    fig.add_hline(
        y=85, line_dash="dash", 
        line_color="green",
        annotation_text="World Class 85%"
    )
    fig.add_hline(
        y=65, line_dash="dash",
        line_color="red",
        annotation_text="Minimum 65%"
    )
    fig.update_traces(line_color='#667eea', line_width=2)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏭 OEE by Machine")
    machine_oee = filtered_df.groupby('Machine_ID')['OEE'].mean().reset_index()
    machine_oee = machine_oee.sort_values('OEE')
    colors = ['#FF4444' if x < 65 else '#FFD700' 
              if x < 85 else '#00CC44' 
              for x in machine_oee['OEE']]
    fig2 = px.bar(
        machine_oee, x='OEE', y='Machine_ID',
        orientation='h',
        title='Average OEE by Machine (%)',
        color='OEE',
        color_continuous_scale=['red', 'yellow', 'green']
    )
    fig2.add_vline(x=85, line_dash="dash", line_color="green")
    st.plotly_chart(fig2, use_container_width=True)

# Row 3
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔄 Shift Performance")
    shift_oee = filtered_df.groupby('Shift')['OEE'].mean().reset_index()
    fig3 = px.bar(
        shift_oee, x='Shift', y='OEE',
        title='Average OEE by Shift (%)',
        color='OEE',
        color_continuous_scale=['red', 'yellow', 'green']
    )
    fig3.add_hline(y=85, line_dash="dash", line_color="green")
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("📊 Loss Distribution")
    loss_dist = filtered_df['Primary_Loss'].value_counts().reset_index()
    loss_dist.columns = ['Loss_Type', 'Count']
    fig4 = px.pie(
        loss_dist, values='Count', names='Loss_Type',
        title='Primary Loss Distribution',
        color_discrete_map={
            'Availability Loss': '#FF4444',
            'Performance Loss': '#FF8C00',
            'Quality Loss': '#FFD700',
            'No Significant Loss': '#00CC44'
        }
    )
    st.plotly_chart(fig4, use_container_width=True)

# Row 4 — Anomaly Section
st.markdown("---")
st.subheader("⚠️ ML Anomaly Detection — Flagged Shifts")

anomaly_df = filtered_df[filtered_df['Anomaly_Score'] == -1][
    ['Date', 'Shift', 'Machine_ID', 'OEE', 
     'Primary_Loss', 'Downtime_mins']
].sort_values('OEE')

st.dataframe(
    anomaly_df.style.applymap(
        lambda x: 'background-color: #ffcccc' 
        if isinstance(x, float) and x < 65 else '',
        subset=['OEE']
    ),
    use_container_width=True
)

# Row 5 — Benchmark
st.markdown("---")
st.subheader("🎯 Benchmark Comparison — vs World Class")

benchmark_data = {
    'Metric': ['OEE', 'Availability', 'Performance', 'Quality'],
    'Your Line': [avg_oee, avg_avail, avg_perf, avg_qual],
    'World Class': [85, 90, 95, 99.9]
}
bench_df = pd.DataFrame(benchmark_data)
bench_df['Gap'] = bench_df['World Class'] - bench_df['Your Line']
bench_df['Gap'] = bench_df['Gap'].apply(lambda x: f"-{x:.1f}%")

st.dataframe(bench_df, use_container_width=True)

# Raw Data
st.markdown("---")
with st.expander("📋 View Raw Shift Data"):
    st.dataframe(
        filtered_df[[
            'Date', 'Shift', 'Machine_ID', 'OEE',
            'Availability_pct', 'Performance_pct', 
            'Quality_pct', 'Primary_Loss', 
            'Downtime_mins', 'Is_Anomaly'
        ]].sort_values('OEE'),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown(
    "**OEE Intelligence Dashboard** | "
    "Built by Heramb | VJTI'27 | "
    "Dataset: HPDC Line Observations"
)
