import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime
import io

# ─── PAGE CONFIG ───────────────────────────────
st.set_page_config(
    page_title="OEEiq | Production Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME TOGGLE ──────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

# ─── CSS ───────────────────────────────────────
bg        = "#0F1117" if dark else "#F8F9FA"
surface   = "#1A1D2E" if dark else "#FFFFFF"
surface2  = "#252840" if dark else "#F0F2F6"
text      = "#FAFAFA" if dark else "#1C1C1C"
subtext   = "#888888" if dark else "#666666"
accent    = "#6C63FF"
green     = "#4BFF91" if dark else "#00A651"
red       = "#FF4B4B"
amber     = "#FFB84B"
border    = "#6C63FF25" if dark else "#E0E0E0"
chart_bg  = "rgba(0,0,0,0)"

st.markdown(f"""
<style>
  html,body,[class*="css"]{{
    background-color:{bg};
    color:{text};
    font-family:'Inter',sans-serif;
  }}
  #MainMenu,footer,header{{visibility:hidden}}
  section[data-testid="stSidebar"]{{
    background:{surface} !important;
    border-right:1px solid {border};
  }}
  [data-testid="metric-container"]{{
    background:linear-gradient(135deg,{surface},{surface2});
    border:1px solid {border};
    border-radius:14px;
    padding:18px 14px;
    box-shadow:0 4px 20px rgba(108,99,255,0.08);
  }}
  [data-testid="stMetricValue"]{{
    font-size:26px!important;
    font-weight:700!important;
    color:{text}!important;
  }}
  [data-testid="stMetricLabel"]{{
    font-size:11px!important;
    color:{subtext}!important;
    text-transform:uppercase;
    letter-spacing:.5px;
  }}
  [data-testid="stFileUploader"]{{
    border:2px dashed {accent}40!important;
    border-radius:12px!important;
    background:{accent}08!important;
  }}
  .stDownloadButton>button{{
    background:{accent}!important;
    color:white!important;
    border:none!important;
    border-radius:8px!important;
    width:100%;
  }}
  .stButton>button{{
    background:{surface2}!important;
    color:{text}!important;
    border:1px solid {border}!important;
    border-radius:8px!important;
  }}
  @media(max-width:768px){{
    [data-testid="stMetricValue"]{{font-size:18px!important}}
    .block-container{{padding:0.5rem!important}}
  }}
</style>
""", unsafe_allow_html=True)

CHART = dict(
    paper_bgcolor=chart_bg,
    plot_bgcolor=chart_bg,
    font=dict(color=text, size=11),
    margin=dict(l=8,r=8,t=32,b=8)
)

# ─── HELPERS ───────────────────────────────────
@st.cache_data
def calc_oee(df):
    df = df.copy()
    df['Availability'] = ((df['Planned_Production_Time_mins']-df['Downtime_mins'])/df['Planned_Production_Time_mins']).clip(0,1)
    df['Performance']  = ((df['Total_Parts_Produced']*df['Ideal_Cycle_Time_sec'])/((df['Planned_Production_Time_mins']-df['Downtime_mins'])*60)).clip(0,1)
    df['Quality']      = (df['Good_Parts']/df['Total_Parts_Produced']).clip(0,1)
    df['OEE']          = (df['Availability']*df['Performance']*df['Quality']*100).round(2)
    df['Availability_pct'] = (df['Availability']*100).round(2)
    df['Performance_pct']  = (df['Performance']*100).round(2)
    df['Quality_pct']      = (df['Quality']*100).round(2)
    def cls(r):
        if r['Availability']<0.85: return 'Availability Loss'
        if r['Performance']<0.85:  return 'Performance Loss'
        if r['Quality']<0.99:      return 'Quality Loss'
        return 'No Significant Loss'
    df['Primary_Loss'] = df.apply(cls,axis=1)
    X = StandardScaler().fit_transform(df[['OEE','Availability_pct','Performance_pct','Quality_pct','Downtime_mins']])
    df['Anomaly_Score'] = IsolationForest(contamination=0.1,random_state=42).fit_predict(X)
    df['Is_Anomaly'] = df['Anomaly_Score'].apply(lambda x:'⚠️ Anomaly' if x==-1 else '✅ Normal')
    return df

@st.cache_data
def calc_scm(scm, oee):
    scm = scm.copy()
    daily_inv = scm.groupby('Date')['Inventory_Buffer_Days'].mean().reset_index()
    daily_oee = oee.groupby('Date')['OEE'].mean().reset_index()
    merged = daily_inv.merge(daily_oee, on='Date')
    corr, pval = stats.pearsonr(merged['Inventory_Buffer_Days'], merged['OEE'])
    supplier_perf = scm.groupby('Supplier').agg(
        OTIF_Rate=('OTIF','mean'),
        Avg_Delay=('Delivery_Delay_Days','mean'),
        Total_Deliveries=('OTIF','count')
    ).reset_index()
    supplier_perf['OTIF_Rate'] = (supplier_perf['OTIF_Rate']*100).round(1)
    supplier_perf['Risk'] = supplier_perf['OTIF_Rate'].apply(
        lambda x: '🔴 High Risk' if x<80 else '🟡 Monitor' if x<90 else '🟢 Good'
    )
    return corr, pval, supplier_perf, merged

@st.cache_data
def load_sample():
    import random as rnd
    rnd.seed(42); np.random.seed(42)
    rows=[]
    base=datetime(2026,7,1)
    for day in range(90):
        d=base+timedelta(days=day) if True else None
        from datetime import timedelta
        d=(base+timedelta(days=day)).strftime('%Y-%m-%d')
        for shift in ['A','B','C']:
            for machine in ['M1-HPDC','M2-HPDC','M3-HPDC','M4-HPDC']:
                pt=480
                dt=rnd.randint(60,150) if machine=='M3-HPDC' else rnd.randint(40,100) if shift=='C' else rnd.randint(10,60)
                ict=rnd.uniform(28,35)
                total=int(((pt-dt)*60/ict)*rnd.uniform(0.65,0.95))
                rej=int(total*rnd.uniform(0.02,0.12))
                rows.append({'Date':d,'Shift':shift,'Machine_ID':machine,
                    'Planned_Production_Time_mins':pt,'Downtime_mins':dt,
                    'Ideal_Cycle_Time_sec':round(ict,2),'Total_Parts_Produced':total,
                    'Rejected_Parts':rej,'Good_Parts':total-rej})
    return pd.DataFrame(rows)

# ─── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-size:26px;font-weight:800;color:{accent}'>🏭 OEEiq</div>
      <div style='font-size:11px;color:{subtext};margin-top:4px'>Production Intelligence Platform</div>
      <div style='display:inline-block;margin-top:8px;padding:3px 12px;
           border-radius:20px;background:{accent}20;color:{accent};
           font-size:10px;font-weight:600;border:1px solid {accent}40'>BETA v2.0</div>
    </div>
    <hr style='border-color:{border};margin:12px 0'>
    """, unsafe_allow_html=True)

    theme_label = "☀️ Light Mode" if dark else "🌙 Dark Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown(f"<div style='font-size:12px;font-weight:500;color:{subtext};margin:12px 0 6px'>📁 DATA SOURCE</div>", unsafe_allow_html=True)
    oee_file = st.file_uploader("OEE shift data (CSV)", type=['csv','xlsx'], key='oee')
    scm_file = st.file_uploader("SCM supplier data (CSV)", type=['csv','xlsx'], key='scm')

    if not oee_file:
        st.info("Using sample HPDC data — 90 days, 4 machines, 3 shifts")

    st.markdown(f"<hr style='border-color:{border}'><div style='font-size:12px;font-weight:500;color:{subtext};margin-bottom:8px'>🔧 FILTERS</div>", unsafe_allow_html=True)
    machines_all = ['M1-HPDC','M2-HPDC','M3-HPDC','M4-HPDC']
    sel_m = st.multiselect("Machine", machines_all, default=machines_all)
    sel_s = st.multiselect("Shift", ['A','B','C'], default=['A','B','C'])

    st.markdown(f"<hr style='border-color:{border}'><div style='font-size:12px;font-weight:500;color:{subtext};margin-bottom:8px'>₹ LOSS CALCULATOR</div>", unsafe_allow_html=True)
    part_val  = st.number_input("Part value (₹)", min_value=10, value=250, step=10)
    parts_min = st.number_input("Ideal parts/min", min_value=0.1, value=2.0, step=0.1)
    target    = st.slider("OEE target %", 70, 95, 85)

    st.markdown(f"""
    <hr style='border-color:{border}'>
    <div style='text-align:center;font-size:10px;color:{subtext};padding:8px 0'>
      Built by <b>Heramb</b> | VJTI'27<br>
      Production Engineering · Data Science<br>
      <a href='mailto:heramb@vjti.ac.in' style='color:{accent}'>Request demo</a>
    </div>
    """, unsafe_allow_html=True)

# ─── LOAD DATA ─────────────────────────────────
from datetime import timedelta
try:
    raw = pd.read_csv(oee_file) if oee_file else load_sample()
    df  = calc_oee(raw)
except:
    df  = calc_oee(load_sample())

try:
    scm_raw = pd.read_csv(scm_file) if scm_file else pd.read_csv('scm_data.csv')
    has_scm = True
except:
    has_scm = False

fdf = df[df['Machine_ID'].isin(sel_m) & df['Shift'].isin(sel_s)].copy()
fdf['Quality_Loss_Rs']  = fdf['Rejected_Parts'] * part_val
fdf['Downtime_Loss_Rs'] = fdf['Downtime_mins'] * parts_min * part_val
fdf['Total_Loss_Rs']    = fdf['Quality_Loss_Rs'] + fdf['Downtime_Loss_Rs']

avg_oee  = fdf['OEE'].mean()
avg_av   = fdf['Availability_pct'].mean()
avg_perf = fdf['Performance_pct'].mean()
avg_qual = fdf['Quality_pct'].mean()
anom_cnt = (fdf['Anomaly_Score']==-1).sum()
tot_loss = fdf['Total_Loss_Rs'].sum()
dt_loss  = fdf['Downtime_Loss_Rs'].sum()
ql_loss  = fdf['Quality_Loss_Rs'].sum()
gap      = avg_oee - target

# ─── HEADER ────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
     padding:0 0 16px;border-bottom:1px solid {border};margin-bottom:20px;flex-wrap:wrap;gap:8px'>
  <div>
    <div style='font-size:22px;font-weight:800;color:{accent}'>🏭 OEEiq</div>
    <div style='font-size:12px;color:{subtext}'>Production Intelligence Platform · HPDC Line</div>
  </div>
  <div style='background:{accent}15;border:1px solid {accent}30;border-radius:8px;
       padding:6px 14px;font-size:11px;color:{accent};font-weight:600'>
    {datetime.now().strftime('%d %b %Y')} · BETA v2.0
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ──────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 OEE Overview", "🏭 SCM Intelligence", "⚠️ Anomalies", "📥 Export"])

# ══════════════════════════════════════════════
# TAB 1 — OEE OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    # OEE Score + KPIs
    col_score, col_kpis = st.columns([1,3])
    with col_score:
        status_c = green if avg_oee>=85 else amber if avg_oee>=65 else red
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,{surface},{surface2});
             border:1px solid {border};border-radius:16px;
             padding:24px 16px;text-align:center'>
          <div style='font-size:54px;font-weight:800;color:{status_c};line-height:1'>
            {avg_oee:.1f}%
          </div>
          <div style='font-size:11px;color:{subtext};margin-top:6px;
               text-transform:uppercase;letter-spacing:1px'>Overall OEE</div>
          <div style='margin-top:10px;padding:4px 14px;border-radius:20px;
               display:inline-block;font-size:11px;font-weight:600;
               background:{status_c}20;color:{status_c}'>
            {"🟢 World Class" if avg_oee>=85 else "🟡 Average" if avg_oee>=65 else "🔴 Below Average"}
          </div>
          <div style='font-size:12px;margin-top:8px;
               color:{"#4BFF91" if gap>=0 else "#FF4B4B"}'>
            {f"+{gap:.1f}%" if gap>=0 else f"{gap:.1f}%"} vs target
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpis:
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Availability",  f"{avg_av:.1f}%",   f"{'✅ OK' if avg_av>=90  else '⚠️ Low'}")
        k2.metric("Performance",   f"{avg_perf:.1f}%", f"{'✅ OK' if avg_perf>=95 else '⚠️ Low'}")
        k3.metric("Quality",       f"{avg_qual:.1f}%", f"{'✅ OK' if avg_qual>=99 else '⚠️ Low'}")
        k4.metric("⚠️ Anomalies",  str(anom_cnt),      "ML flagged")

    st.markdown("<br>", unsafe_allow_html=True)

    # Rupee Loss Row
    st.markdown(f"<div style='font-size:11px;font-weight:600;color:{accent};text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>₹ Loss Intelligence</div>", unsafe_allow_html=True)
    l1,l2,l3 = st.columns(3)
    l1.metric("Total Loss",    f"₹{tot_loss:,.0f}", "This period")
    l2.metric("Downtime Loss", f"₹{dt_loss:,.0f}",  f"{dt_loss/max(tot_loss,1)*100:.0f}% of total")
    l3.metric("Quality Loss",  f"₹{ql_loss:,.0f}",  f"{ql_loss/max(tot_loss,1)*100:.0f}% of total")

    st.divider()

    # Charts Row 1
    c1,c2 = st.columns([2,1])
    with c1:
        daily = fdf.groupby('Date')['OEE'].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['Date'], y=daily['OEE'],
            fill='tozeroy', fillcolor=f'{accent}12',
            line=dict(color=accent, width=2), name='Daily OEE'
        ))
        fig.add_hline(y=target, line_dash="dash", line_color=green, line_width=1,
                      annotation_text=f"Target {target}%", annotation_font_color=green)
        fig.add_hline(y=65, line_dash="dot", line_color=red, line_width=1,
                      annotation_text="Min 65%", annotation_font_color=red)
        fig.update_layout(**CHART, title="Daily OEE Trend",
                          yaxis=dict(range=[0,100],showgrid=True,gridcolor=f'{text}08'),
                          xaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        m_oee = fdf.groupby('Machine_ID')['OEE'].mean().reset_index().sort_values('OEE')
        colors = [red if v<65 else amber if v<target else green for v in m_oee['OEE']]
        fig2 = go.Figure(go.Bar(
            x=m_oee['OEE'], y=m_oee['Machine_ID'],
            orientation='h', marker_color=colors,
            text=[f"{v:.1f}%" for v in m_oee['OEE']], textposition='outside'
        ))
        fig2.update_layout(**CHART, title="OEE by Machine",
                           xaxis=dict(range=[0,110],showgrid=False),
                           yaxis=dict(showgrid=False))
        st.plotly_chart(fig2, use_container_width=True)

    # Charts Row 2
    c3,c4 = st.columns(2)
    with c3:
        s_oee = fdf.groupby('Shift')['OEE'].mean().reset_index()
        s_col = [red if v<65 else amber if v<target else green for v in s_oee['OEE']]
        fig3 = go.Figure(go.Bar(
            x=s_oee['Shift'], y=s_oee['OEE'],
            marker_color=s_col,
            text=[f"{v:.1f}%" for v in s_oee['OEE']], textposition='outside'
        ))
        fig3.add_hline(y=target, line_dash="dash", line_color=green, line_width=1)
        fig3.update_layout(**CHART, title="OEE by Shift",
                           yaxis=dict(range=[0,110],showgrid=True,gridcolor=f'{text}08'),
                           xaxis=dict(showgrid=False))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        ld = fdf['Primary_Loss'].value_counts().reset_index()
        ld.columns=['Loss_Type','Count']
        lc = {'Availability Loss':red,'Performance Loss':amber,
              'Quality Loss':'#4B9FFF','No Significant Loss':green}
        fig4 = go.Figure(go.Pie(
            labels=ld['Loss_Type'], values=ld['Count'],
            marker_colors=[lc.get(l,'#888') for l in ld['Loss_Type']],
            hole=0.5, textinfo='percent+label', textfont_size=11
        ))
        fig4.update_layout(**CHART, title="Loss Distribution", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Benchmark
    st.divider()
    st.markdown(f"<div style='font-size:11px;font-weight:600;color:{accent};text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>🎯 Benchmark vs World Class</div>", unsafe_allow_html=True)
    bench = pd.DataFrame({
        'Metric':['OEE','Availability','Performance','Quality'],
        'Your Line':[f"{avg_oee:.1f}%",f"{avg_av:.1f}%",f"{avg_perf:.1f}%",f"{avg_qual:.1f}%"],
        'World Class':['85%','90%','95%','99.9%'],
        'Gap':[f"{avg_oee-85:.1f}%",f"{avg_av-90:.1f}%",f"{avg_perf-95:.1f}%",f"{avg_qual-99.9:.1f}%"],
        'Status':['✅' if avg_oee>=85 else '⚠️','✅' if avg_av>=90 else '⚠️',
                  '✅' if avg_perf>=95 else '⚠️','✅' if avg_qual>=99.9 else '⚠️']
    })
    st.dataframe(bench, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 2 — SCM INTELLIGENCE
# ══════════════════════════════════════════════
with tab2:
    if has_scm:
        corr, pval, sup_perf, merged = calc_scm(scm_raw, fdf)

        st.markdown(f"<div style='font-size:11px;font-weight:600;color:{accent};text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>📦 Supplier Performance — OTIF Scorecard</div>", unsafe_allow_html=True)

        sc1,sc2,sc3 = st.columns(3)
        sc1.metric("Pearson Correlation", f"{corr:.3f}", "Inventory → OEE")
        sc2.metric("P-Value", f"{pval:.4f}", "Statistically significant" if pval<0.05 else "Not significant")
        sc3.metric("Best Supplier", sup_perf.loc[sup_perf['OTIF_Rate'].idxmax(),'Supplier'], f"{sup_perf['OTIF_Rate'].max():.1f}% OTIF")

        st.divider()

        # OTIF Bar Chart
        sup_sorted = sup_perf.sort_values('OTIF_Rate')
        s_colors = [red if v<80 else amber if v<90 else green for v in sup_sorted['OTIF_Rate']]
        fig5 = go.Figure(go.Bar(
            x=sup_sorted['OTIF_Rate'], y=sup_sorted['Supplier'],
            orientation='h', marker_color=s_colors,
            text=[f"{v:.1f}%" for v in sup_sorted['OTIF_Rate']], textposition='outside'
        ))
        fig5.add_vline(x=95, line_dash="dash", line_color=green, line_width=1,
                       annotation_text="Target 95%", annotation_font_color=green)
        fig5.update_layout(**CHART, title="Supplier OTIF Score (%)",
                           xaxis=dict(range=[0,110],showgrid=False),
                           yaxis=dict(showgrid=False))
        st.plotly_chart(fig5, use_container_width=True)

        # Inventory vs OEE Scatter
        fig6 = go.Figure(go.Scatter(
            x=merged['Inventory_Buffer_Days'], y=merged['OEE'],
            mode='markers',
            marker=dict(color=accent, size=6, opacity=0.7),
            name='Daily data'
        ))
        m,b = np.polyfit(merged['Inventory_Buffer_Days'], merged['OEE'], 1)
        x_line = np.linspace(merged['Inventory_Buffer_Days'].min(), merged['Inventory_Buffer_Days'].max(), 100)
        fig6.add_trace(go.Scatter(
            x=x_line, y=m*x_line+b,
            mode='lines', line=dict(color=red, width=2, dash='dash'),
            name=f'Trend (r={corr:.2f})'
        ))
        fig6.update_layout(**CHART, title=f"Inventory Buffer vs OEE (Pearson r = {corr:.3f})",
                           xaxis=dict(title="Inventory Buffer Days", showgrid=True, gridcolor=f'{text}08'),
                           yaxis=dict(title="Average OEE %", showgrid=True, gridcolor=f'{text}08'))
        st.plotly_chart(fig6, use_container_width=True)

        # Supplier Table
        st.markdown(f"<div style='font-size:11px;font-weight:600;color:{accent};text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>Supplier Risk Ranking</div>", unsafe_allow_html=True)
        st.dataframe(sup_perf.sort_values('OTIF_Rate'), use_container_width=True, hide_index=True)

    else:
        st.info("📦 Upload scm_data.csv in the sidebar to see SCM Intelligence module.")
        st.markdown(f"""
        <div style='background:{surface};border:1px solid {border};border-radius:12px;padding:20px;margin-top:16px'>
          <div style='font-size:14px;font-weight:600;color:{text};margin-bottom:8px'>SCM Module includes:</div>
          <div style='font-size:12px;color:{subtext};line-height:1.8'>
            ✅ Supplier OTIF scorecard<br>
            ✅ Inventory vs OEE Pearson correlation<br>
            ✅ Supplier risk ranking<br>
            ✅ Delivery delay impact analysis<br>
            ✅ Reorder point recommendations
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — ANOMALIES
# ══════════════════════════════════════════════
with tab3:
    anom_df = fdf[fdf['Anomaly_Score']==-1].sort_values('OEE').head(15)

    a1,a2 = st.columns([1,3])
    with a1:
        st.markdown(f"""
        <div style='background:{red}12;border:1px solid {red}30;border-radius:14px;
             padding:24px;text-align:center'>
          <div style='font-size:48px;font-weight:800;color:{red}'>{anom_cnt}</div>
          <div style='font-size:11px;color:{subtext};margin-top:4px'>Anomalous shifts</div>
          <hr style='border-color:{red}20;margin:12px 0'>
          <div style='font-size:11px;color:{subtext}'>
            {anom_cnt/len(fdf)*100:.1f}% of total shifts<br>
            flagged by Isolation Forest
          </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        for _, row in anom_df.head(6).iterrows():
            loss_c = red if row['Primary_Loss']=='Availability Loss' else amber if row['Primary_Loss']=='Performance Loss' else '#4B9FFF'
            st.markdown(f"""
            <div style='background:{surface};border:1px solid {red}25;
                 border-left:3px solid {red};border-radius:0 10px 10px 0;
                 padding:10px 14px;margin-bottom:8px'>
              <span style='font-weight:600;color:{red}'>{row['Machine_ID']} — Shift {row['Shift']}</span>
              &nbsp;|&nbsp; OEE: <b>{row['OEE']:.1f}%</b>
              &nbsp;|&nbsp; {row['Date']}
              <div style='font-size:11px;color:{subtext};margin-top:3px'>
                <span style='color:{loss_c}'>{row['Primary_Loss']}</span>
                &nbsp;·&nbsp; Downtime: {row['Downtime_mins']} mins
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    with st.expander("📋 All anomalous shifts"):
        st.dataframe(anom_df[['Date','Shift','Machine_ID','OEE','Primary_Loss','Downtime_mins','Is_Anomaly']],
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 4 — EXPORT
# ══════════════════════════════════════════════
with tab4:
    st.markdown(f"<div style='font-size:13px;font-weight:500;color:{text};margin-bottom:16px'>Download reports for shift handover and procurement review</div>", unsafe_allow_html=True)
    e1,e2,e3 = st.columns(3)
    with e1:
        csv1 = fdf[['Date','Shift','Machine_ID','OEE','Availability_pct','Performance_pct',
                    'Quality_pct','Primary_Loss','Downtime_mins','Is_Anomaly','Total_Loss_Rs']].to_csv(index=False).encode()
        st.download_button("📥 Full OEE Report (CSV)", csv1,
            f"OEEiq_Report_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
    with e2:
        anom_csv = anom_df[['Date','Shift','Machine_ID','OEE','Primary_Loss','Downtime_mins']].to_csv(index=False).encode()
        st.download_button("⚠️ Anomaly Report (CSV)", anom_csv,
            f"OEEiq_Anomalies_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)
    with e3:
        loss_csv = fdf[['Date','Shift','Machine_ID','Total_Loss_Rs','Quality_Loss_Rs','Downtime_Loss_Rs']].to_csv(index=False).encode()
        st.download_button("₹ Loss Report (CSV)", loss_csv,
            f"OEEiq_Losses_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)

    st.divider()
    st.markdown(f"""
    <div style='background:{surface};border:1px solid {border};border-radius:12px;padding:20px;margin-top:8px'>
      <div style='font-size:13px;font-weight:600;color:{text};margin-bottom:12px'>🏭 OEEiq — Platform Summary</div>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:12px;color:{subtext}'>
        <div>📊 Shifts analysed: <b style='color:{text}'>{len(fdf):,}</b></div>
        <div>🏭 Machines monitored: <b style='color:{text}'>{fdf['Machine_ID'].nunique()}</b></div>
        <div>⚠️ Anomalies flagged: <b style='color:{red}'>{anom_cnt}</b></div>
        <div>₹ Total loss: <b style='color:{red}'>₹{tot_loss:,.0f}</b></div>
        <div>📅 Date range: <b style='color:{text}'>{fdf['Date'].min()} → {fdf['Date'].max()}</b></div>
        <div>🎯 Avg OEE: <b style='color:{accent}'>{avg_oee:.1f}%</b></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style='text-align:center;color:{subtext};font-size:11px;
     padding:32px 0 16px;border-top:1px solid {border};margin-top:40px'>
  🏭 <b>OEEiq</b> — Production Intelligence Platform &nbsp;|&nbsp;
  Built by <b>Heramb</b> | Production Engineering | VJTI Mumbai '27 &nbsp;|&nbsp;
  Dataset: HPDC Line (CIE Automotive) &nbsp;|&nbsp;
  <a href='mailto:heramb@vjti.ac.in' style='color:{accent}'>Request live demo</a>
</div>
""", unsafe_allow_html=True)
