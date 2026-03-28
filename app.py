import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BC Population Health Equity Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Light / Clinical theme ────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #F7F9FC; color: #1A2235; }
  [data-testid="stSidebar"]          { background: #FFFFFF; border-right: 1px solid #E2E8F0; }
  [data-testid="stSidebar"] * { color: #2D3748 !important; }
  .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  [data-testid="metric-container"] {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 8px; padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  [data-testid="metric-container"] label {
    color: #64748B !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.07em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1A2235 !important; font-size: 1.55rem !important; font-weight: 700 !important;
  }

  .alert-box {
    background: #FFF5F5; border-left: 4px solid #E53E3E;
    border-radius: 6px; padding: 0.9rem 1.2rem; margin-bottom: 1rem;
  }
  .alert-box h4 { color: #C53030; margin: 0 0 0.35rem 0; font-size: 0.78rem; letter-spacing: 0.06em; }
  .alert-box p  { color: #2D3748; margin: 0; font-size: 0.9rem; line-height: 1.55; }

  .briefing-box {
    background: #EBF8FF; border: 1px solid #BEE3F8;
    border-radius: 8px; padding: 1.2rem 1.5rem; margin-top: 0.5rem;
  }
  .briefing-box p { color: #1A365D; font-size: 0.97rem; line-height: 1.75; margin: 0; }
  .briefing-source { font-size: 0.68rem; color: #718096; margin-top: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; }

  .section-label {
    font-size: 0.68rem; color: #718096; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 0.4rem; font-weight: 600;
  }

  .vuln-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
  .vuln-table th {
    background: #F1F5F9; color: #475569; font-size: 0.68rem;
    letter-spacing: 0.09em; text-transform: uppercase;
    padding: 9px 12px; text-align: left; border-bottom: 2px solid #E2E8F0;
  }
  .vuln-table td { padding: 9px 12px; border-bottom: 1px solid #F1F5F9; color: #2D3748; }
  .vuln-table tr:hover td { background: #F8FAFC; }
  .badge-red   { background:#FEE2E2; color:#DC2626; padding:2px 9px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-amber { background:#FEF3C7; color:#B45309; padding:2px 9px; border-radius:20px; font-size:0.75rem; font-weight:600; }
  .badge-green { background:#DCFCE7; color:#166534; padding:2px 9px; border-radius:20px; font-size:0.75rem; font-weight:600; }

  .stButton>button {
    background: #2B6CB0; color: #FFFFFF; border: none;
    border-radius: 6px; font-weight: 600; padding: 0.55rem 1rem; font-size: 0.85rem;
  }
  .stButton>button:hover { background: #2C5282; }

  [data-testid="stFileUploader"] {
    background: #FFFFFF; border: 1px dashed #CBD5E0; border-radius: 8px; padding: 0.5rem;
  }

  h1, h2, h3 { color: #1A2235 !important; }
  hr { border-color: #E2E8F0; }
  p, li { color: #4A5568; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
REQUIRED_COLS = [
    "chsa_name", "health_authority", "population",
    "pct_without_family_doctor", "pct_below_poverty_line",
    "opioid_overdose_rate", "median_household_income",
    "life_expectancy", "er_visits_per_1000",
]

def find_csv(name):
    for p in [name, f"data/{name}", f"/mnt/user-data/outputs/{name}"]:
        if os.path.exists(p):
            return p
    return None

def validate_df(df):
    return [c for c in REQUIRED_COLS if c not in df.columns]

def compute_vulnerability(df_region, df_full):
    d = df_region.copy()
    def norm(col):
        mn, mx = df_full[col].min(), df_full[col].max()
        return (d[col] - mn) / (mx - mn + 1e-9)
    d["score_gp"]      = norm("pct_without_family_doctor") * 40
    d["score_poverty"] = norm("pct_below_poverty_line")    * 30
    d["score_opioid"]  = norm("opioid_overdose_rate")      * 30
    d["vulnerability_score"] = (d["score_gp"] + d["score_poverty"] + d["score_opioid"]).round(1)
    return d

def get_briefing(region_name, top, stats):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        prompt = f"""You are a senior public health analyst writing for a BC Health Authority director.

Region: {region_name}
Most vulnerable community: {top['chsa_name']}
  - Vulnerability score: {top['vulnerability_score']}/100
  - % without family doctor: {top['pct_without_family_doctor']}%
  - % below poverty line: {top['pct_below_poverty_line']}%
  - Opioid overdose rate: {top['opioid_overdose_rate']} per 100k
  - Life expectancy: {top['life_expectancy']} years
  - ER visits per 1,000: {top['er_visits_per_1000']}
Region averages — Avg % without GP: {stats['avg_no_gp']:.1f}%, Avg life expectancy: {stats['avg_le']:.1f} years, Avg opioid rate: {stats['avg_opioid']:.1f}

Write exactly 3 sentences as an Executive Briefing:
1. Name the top-priority community and its most critical indicator.
2. Explain the equity gap relative to the regional average.
3. Give a specific, actionable recommendation.
Be direct, clinical, and data-driven. No bullet points. No headers."""
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 300,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()["content"][0]["text"].strip(), "Generated by Claude AI · claude-sonnet-4"
        except Exception:
            pass

    # Rule-based fallback
    gap_gp = top["pct_without_family_doctor"] - stats["avg_no_gp"]
    gap_le = stats["avg_le"] - top["life_expectancy"]
    if top["pct_without_family_doctor"] > 30:
        action = "deployment of a mobile primary care clinic combined with GP recruitment incentives"
    elif top["opioid_overdose_rate"] > 40:
        action = "targeted opioid harm-reduction outreach and supervised consumption services"
    else:
        action = "community health worker deployment and social determinants screening program"
    briefing = (
        f"{top['chsa_name']} is the highest-priority community in {region_name}, "
        f"with {top['pct_without_family_doctor']}% of residents lacking a family doctor "
        f"and an opioid overdose rate of {top['opioid_overdose_rate']} per 100,000. "
        f"This community sits {gap_gp:.1f} percentage points above the regional GP access average "
        f"and has a life expectancy {gap_le:.1f} years below the regional mean, indicating compounding inequity. "
        f"The recommended immediate intervention is {action} in {top['chsa_name']}."
    )
    return briefing, "Generated by rule-based scoring engine (set ANTHROPIC_API_KEY to enable AI)"


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏥 BC Health Equity Dashboard")
    st.markdown("---")
    st.markdown('<div class="section-label">📂 Data Source</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your own CSV",
        type=["csv"],
        help=(
            "Required columns: chsa_name, health_authority, population, "
            "pct_without_family_doctor, pct_below_poverty_line, opioid_overdose_rate, "
            "median_household_income, life_expectancy, er_visits_per_1000"
        ),
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            missing = validate_df(df_raw)
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                df_raw = None
            else:
                st.success(f"✅ Loaded {len(df_raw)} rows")
        except Exception as e:
            st.error(f"Could not read file: {e}")
            df_raw = None
    else:
        p = find_csv("bc_health_indicators.csv")
        df_raw = pd.read_csv(p) if p else None

    if df_raw is None:
        st.error("No valid data. Upload a CSV or ensure bc_health_indicators.csv is present.")
        st.stop()

    st.markdown("---")
    st.markdown('<div class="section-label">🏛 Health Authority</div>', unsafe_allow_html=True)
    authorities = sorted(df_raw["health_authority"].unique())
    selected_ha = st.selectbox("", authorities, label_visibility="collapsed")

    st.markdown("---")
    st.caption("Vulnerability score: 40% GP access · 30% poverty · 30% opioid harm.")
    st.caption(f"Data: {len(df_raw)} CHSAs · BuildersVault × UVic Hacks · HealthHack 2026")


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPUTE
# ═══════════════════════════════════════════════════════════════════════════════
region_df    = df_raw[df_raw["health_authority"] == selected_ha].copy()
region_df    = compute_vulnerability(region_df, df_raw)
region_sorted = region_df.sort_values("vulnerability_score", ascending=False)
top = region_sorted.iloc[0].to_dict()
region_stats = {
    "avg_no_gp":  region_df["pct_without_family_doctor"].mean(),
    "avg_le":     region_df["life_expectancy"].mean(),
    "avg_opioid": region_df["opioid_overdose_rate"].mean(),
}

# Auto-generate briefing on load / HA change
cache_key = f"briefing_{selected_ha}_{uploaded_file.name if uploaded_file else 'default'}"
if cache_key not in st.session_state:
    with st.spinner("Generating executive briefing…"):
        b_text, b_source = get_briefing(selected_ha, top, region_stats)
    st.session_state[cache_key] = (b_text, b_source)
b_text, b_source = st.session_state[cache_key]


# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🏥  Community Health Equity",
    "⏱  BC Wait Times",
    "💊  Opioid Crisis",
])


# ╔═══════════════════════════╗
# ║  TAB 1 — Health Equity   ║
# ╚═══════════════════════════╝
with tab1:
    st.markdown(f"## {selected_ha} · Population Health Equity")
    st.markdown(f"*{len(region_df)} CHSAs · Population: {region_df['population'].sum():,.0f}*")
    st.markdown("---")

    st.markdown(
        f"""<div class="alert-box">
            <h4>⚠ HIGHEST PRIORITY COMMUNITY FLAGGED</h4>
            <p><strong>{top['chsa_name']}</strong> — Score: <strong>{top['vulnerability_score']}/100</strong>
            &nbsp;·&nbsp; {top['pct_without_family_doctor']}% without GP
            &nbsp;·&nbsp; Opioid: {top['opioid_overdose_rate']}/100k
            &nbsp;·&nbsp; Life exp: {top['life_expectancy']} yrs</p>
        </div>""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Highest ER Pressure",
              f"{region_df.loc[region_df['er_visits_per_1000'].idxmax(), 'er_visits_per_1000']:.0f}/1k",
              region_df.loc[region_df["er_visits_per_1000"].idxmax(), "chsa_name"])
    m2.metric("Lowest Life Expectancy",
              f"{region_df['life_expectancy'].min()} yrs",
              region_df.loc[region_df["life_expectancy"].idxmin(), "chsa_name"])
    m3.metric("Worst GP Access Gap",
              f"{region_df['pct_without_family_doctor'].max()}%",
              region_df.loc[region_df["pct_without_family_doctor"].idxmax(), "chsa_name"])
    m4.metric("Highest Opioid Rate",
              f"{region_df['opioid_overdose_rate'].max()}/100k",
              region_df.loc[region_df["opioid_overdose_rate"].idxmax(), "chsa_name"])

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    # Visual 1 — Bar
    with col_l:
        st.markdown('<div class="section-label">Visual 1 — Primary Care Access Gap</div>', unsafe_allow_html=True)
        bar_df = region_df.sort_values("pct_without_family_doctor", ascending=True)
        colors = ["#E53E3E" if v > 30 else "#D69E2E" if v > 20 else "#38A169"
                  for v in bar_df["pct_without_family_doctor"]]
        fig_bar = go.Figure(go.Bar(
            x=bar_df["pct_without_family_doctor"], y=bar_df["chsa_name"],
            orientation="h", marker_color=colors,
            text=[f"{v}%" for v in bar_df["pct_without_family_doctor"]],
            textposition="outside", textfont=dict(color="#4A5568", size=10),
        ))
        fig_bar.add_vline(x=region_stats["avg_no_gp"], line_dash="dash", line_color="#A0AEC0",
                          annotation_text=f"Avg {region_stats['avg_no_gp']:.1f}%",
                          annotation_font_color="#718096")
        fig_bar.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#F7F9FC", font_color="#4A5568",
            height=max(280, len(bar_df) * 28),
            xaxis=dict(title="% Without Family Doctor", gridcolor="#EDF2F7", color="#718096"),
            yaxis=dict(gridcolor="#EDF2F7", color="#2D3748"),
            margin=dict(l=0, r=50, t=10, b=30), showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Visual 2 — Scatter
    with col_r:
        st.markdown('<div class="section-label">Visual 2 — Wealth–Health Equity Gap</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            region_df, x="median_household_income", y="life_expectancy",
            size="population", color="vulnerability_score",
            color_continuous_scale=["#38A169", "#D69E2E", "#E53E3E"],
            hover_name="chsa_name",
            hover_data={"median_household_income": ":,.0f", "life_expectancy": True,
                        "vulnerability_score": True, "population": ":,.0f"},
            labels={"median_household_income": "Median Household Income ($)",
                    "life_expectancy": "Life Expectancy (years)",
                    "vulnerability_score": "Vulnerability Score"},
            size_max=40,
        )
        fig_scatter.add_annotation(
            x=top["median_household_income"], y=top["life_expectancy"],
            text=f"▲ {top['chsa_name']}", showarrow=True, arrowhead=2,
            arrowcolor="#E53E3E", font=dict(color="#E53E3E", size=11),
            bgcolor="#FFF5F5", bordercolor="#E53E3E", borderwidth=1,
        )
        fig_scatter.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#F7F9FC", font_color="#4A5568",
            height=max(280, len(bar_df) * 28),
            xaxis=dict(gridcolor="#EDF2F7", color="#718096", tickprefix="$", tickformat=","),
            yaxis=dict(gridcolor="#EDF2F7", color="#718096"),
            coloraxis_colorbar=dict(title="Vuln.", tickfont=dict(color="#718096"), title_font_color="#718096"),
            margin=dict(l=0, r=20, t=10, b=30),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Visual 3 — Vulnerability table
    st.markdown("---")
    st.markdown('<div class="section-label">Visual 3 — Community Vulnerability Ranking</div>', unsafe_allow_html=True)
    st.caption("Score: 40% GP access · 30% poverty · 30% opioid  |  🔴 ≥ 60  🟡 ≥ 35  🟢 < 35")

    def badge(score):
        if score >= 60: return f'<span class="badge-red">{score}</span>'
        if score >= 35: return f'<span class="badge-amber">{score}</span>'
        return f'<span class="badge-green">{score}</span>'

    rows_html = ""
    for _, r in region_sorted.iterrows():
        rows_html += f"""<tr>
            <td><strong>{r['chsa_name']}</strong></td>
            <td>{badge(r['vulnerability_score'])}</td>
            <td>{r['pct_without_family_doctor']}%</td>
            <td>{r['pct_below_poverty_line']}%</td>
            <td>{r['opioid_overdose_rate']}</td>
            <td>{r['life_expectancy']}</td>
            <td>{r['er_visits_per_1000']:.0f}</td>
        </tr>"""
    st.markdown(f"""
    <table class="vuln-table">
      <thead><tr>
        <th>Community (CHSA)</th><th>Vuln. Score</th><th>% No GP</th>
        <th>% Poverty</th><th>Opioid/100k</th><th>Life Exp.</th><th>ER/1k</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    # AI Briefing — auto on load
    st.markdown("---")
    st.markdown('<div class="section-label">🤖 AI Synthesis Engine — Executive Briefing</div>', unsafe_allow_html=True)
    st.caption("*Which community needs the next outreach clinic, and why?*")
    st.markdown(
        f"""<div class="briefing-box">
            <p>{b_text}</p>
            <div class="briefing-source">⚙ {b_source}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 Refresh Briefing"):
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()


# ╔═══════════════════════════╗
# ║  TAB 2 — Wait Times      ║
# ╚═══════════════════════════╝
with tab2:
    st.markdown("## BC Wait Times — Surgical & Diagnostic Trends")
    st.markdown("*Provincial-level CIHI data · 2014–2025 · BC vs national peers*")
    st.markdown("---")

    p = find_csv("wait_times_mock.csv")
    if not p:
        st.warning("wait_times_mock.csv not found. Place it in the same folder as app.py.")
    else:
        wait_df = pd.read_csv(p)
        procedures = sorted(wait_df["procedure"].unique())
        sel_proc = st.selectbox("Select Procedure", procedures)

        proc_df = wait_df[wait_df["procedure"] == sel_proc]
        bc_proc = proc_df[proc_df["province"] == "BC"].sort_values("year")
        nat_avg = proc_df.groupby("year")["median_wait_days"].mean().reset_index()

        latest_bc   = bc_proc.iloc[-1]
        earliest_bc = bc_proc.iloc[0]
        change = latest_bc["median_wait_days"] - earliest_bc["median_wait_days"]

        wm1, wm2, wm3, wm4 = st.columns(4)
        wm1.metric("BC Median Wait (Latest)", f"{latest_bc['median_wait_days']} days", f"{int(latest_bc['year'])}")
        wm2.metric("Change Since 2014", f"{change:+.0f} days", "improvement" if change < 0 else "increase")
        wm3.metric("% Within Benchmark", f"{latest_bc['pct_within_benchmark']}%", f"Benchmark: {int(latest_bc['benchmark_days'])}d")
        wm4.metric("Annual Volume (BC)", f"{int(latest_bc['volume']):,}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Trend chart: BC vs all provinces
        fig_wait = go.Figure()
        for prov in proc_df["province"].unique():
            if prov == "BC":
                continue
            pdata = proc_df[proc_df["province"] == prov].sort_values("year")
            fig_wait.add_trace(go.Scatter(x=pdata["year"], y=pdata["median_wait_days"],
                mode="lines", name=prov, line=dict(color="#CBD5E0", width=1),
                hovertemplate=f"{prov}: %{{y}}d<extra></extra>"))
        fig_wait.add_trace(go.Scatter(x=nat_avg["year"], y=nat_avg["median_wait_days"],
            mode="lines", name="National Avg", line=dict(color="#D69E2E", width=2, dash="dash")))
        fig_wait.add_trace(go.Scatter(x=bc_proc["year"], y=bc_proc["median_wait_days"],
            mode="lines+markers", name="BC", line=dict(color="#2B6CB0", width=3),
            marker=dict(size=7, color="#2B6CB0")))
        fig_wait.add_hline(y=int(bc_proc.iloc[0]["benchmark_days"]),
                           line_dash="dot", line_color="#E53E3E",
                           annotation_text=f"Benchmark ({int(bc_proc.iloc[0]['benchmark_days'])}d)",
                           annotation_font_color="#E53E3E")
        fig_wait.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#F7F9FC", font_color="#4A5568", height=380,
            title=dict(text=f"Median Wait Days — {sel_proc}", font_color="#1A2235", font_size=14),
            xaxis=dict(title="Year", gridcolor="#EDF2F7", color="#718096"),
            yaxis=dict(title="Median Wait Days", gridcolor="#EDF2F7", color="#718096"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#E2E8F0"),
            margin=dict(l=0, r=0, t=40, b=30),
        )
        st.plotly_chart(fig_wait, use_container_width=True)

        # % within benchmark — all procedures, latest year
        st.markdown('<div class="section-label">BC Performance Across All Procedures — Latest Year</div>', unsafe_allow_html=True)
        latest_yr = wait_df["year"].max()
        bc_latest = wait_df[(wait_df["province"] == "BC") & (wait_df["year"] == latest_yr)].sort_values("pct_within_benchmark")
        colors_b = ["#E53E3E" if v < 75 else "#D69E2E" if v < 85 else "#38A169"
                    for v in bc_latest["pct_within_benchmark"]]
        fig_bench = go.Figure(go.Bar(
            x=bc_latest["pct_within_benchmark"], y=bc_latest["procedure"],
            orientation="h", marker_color=colors_b,
            text=[f"{v}%" for v in bc_latest["pct_within_benchmark"]],
            textposition="outside", textfont=dict(color="#4A5568", size=11),
        ))
        fig_bench.add_vline(x=85, line_dash="dash", line_color="#A0AEC0",
                            annotation_text="85% target", annotation_font_color="#718096")
        fig_bench.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#F7F9FC", font_color="#4A5568", height=300,
            xaxis=dict(title="% Patients Within Benchmark", gridcolor="#EDF2F7", range=[0, 115]),
            yaxis=dict(gridcolor="#EDF2F7"),
            margin=dict(l=0, r=60, t=10, b=30), showlegend=False,
        )
        st.plotly_chart(fig_bench, use_container_width=True)
        st.caption("Source: wait_times_mock.csv · CIHI Wait Times · Canadian Wait Time Alliance benchmarks")


# ╔═══════════════════════════╗
# ║  TAB 3 — Opioid Crisis   ║
# ╚═══════════════════════════╝
with tab3:
    st.markdown("## BC Opioid Crisis — Provincial Surveillance")
    st.markdown("*Quarterly opioid toxicity deaths, hospitalizations & ED visits · BC · 2016–2025*")
    st.markdown("---")

    p2 = find_csv("opioid_harms_mock.csv")
    if not p2:
        st.warning("opioid_harms_mock.csv not found. Place it in the same folder as app.py.")
    else:
        opioid_df = pd.read_csv(p2)
        bc_opioid = opioid_df[opioid_df["province"] == "BC"].copy()
        bc_opioid["period"] = bc_opioid["year"].astype(str) + " " + bc_opioid["quarter"]
        bc_opioid = bc_opioid.sort_values(["year", "quarter"])

        annual = bc_opioid.groupby("year").agg(
            deaths=("apparent_opioid_toxicity_deaths", "sum"),
            hospitalizations=("opioid_hospitalizations", "sum"),
            ed_visits=("opioid_ed_visits", "sum"),
        ).reset_index()

        latest_a = annual.iloc[-1]
        prev_a   = annual.iloc[-2]

        om1, om2, om3, om4 = st.columns(4)
        om1.metric(f"Deaths ({int(latest_a['year'])})", f"{int(latest_a['deaths']):,}",
                   f"{int(latest_a['deaths'] - prev_a['deaths']):+,} vs prior year", delta_color="inverse")
        om2.metric("Hospitalizations", f"{int(latest_a['hospitalizations']):,}",
                   f"{int(latest_a['hospitalizations'] - prev_a['hospitalizations']):+,} vs prior year", delta_color="inverse")
        om3.metric("ED Visits", f"{int(latest_a['ed_visits']):,}",
                   f"{int(latest_a['ed_visits'] - prev_a['ed_visits']):+,} vs prior year", delta_color="inverse")
        om4.metric("Death Rate/100k (Latest Qtr)", f"{bc_opioid.iloc[-1]['rate_per_100k_deaths']}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Quarterly stacked bar + deaths line
        fig_opioid = go.Figure()
        fig_opioid.add_trace(go.Bar(x=bc_opioid["period"], y=bc_opioid["opioid_ed_visits"],
            name="ED Visits", marker_color="#BEE3F8", opacity=0.7))
        fig_opioid.add_trace(go.Bar(x=bc_opioid["period"], y=bc_opioid["opioid_hospitalizations"],
            name="Hospitalizations", marker_color="#D69E2E"))
        fig_opioid.add_trace(go.Scatter(x=bc_opioid["period"], y=bc_opioid["apparent_opioid_toxicity_deaths"],
            mode="lines+markers", name="Toxicity Deaths",
            line=dict(color="#E53E3E", width=2.5), marker=dict(size=6), yaxis="y2"))
        fig_opioid.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#F7F9FC", font_color="#4A5568",
            height=400, barmode="stack",
            title=dict(text="BC Opioid Harms — Quarterly (2016–2025)", font_color="#1A2235", font_size=14),
            xaxis=dict(title="Quarter", gridcolor="#EDF2F7", color="#718096",
                       tickangle=-45, tickvals=bc_opioid["period"].tolist()[::4]),
            yaxis=dict(title="ED Visits + Hospitalizations", gridcolor="#EDF2F7", color="#718096"),
            yaxis2=dict(title="Toxicity Deaths", overlaying="y", side="right",
                        color="#E53E3E", showgrid=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08),
            margin=dict(l=0, r=60, t=50, b=80),
        )
        st.plotly_chart(fig_opioid, use_container_width=True)

        # Provincial comparison — latest year
        st.markdown('<div class="section-label">BC vs All Provinces — Death Rate per 100k (Latest Year)</div>', unsafe_allow_html=True)
        latest_yr = opioid_df["year"].max()
        prov_comp = opioid_df[opioid_df["year"] == latest_yr].groupby("province").agg(
            avg_rate=("rate_per_100k_deaths", "mean")).reset_index().sort_values("avg_rate", ascending=True)
        colors_prov = ["#E53E3E" if p == "BC" else "#90CDF4" for p in prov_comp["province"]]
        fig_prov = go.Figure(go.Bar(
            x=prov_comp["avg_rate"], y=prov_comp["province"],
            orientation="h", marker_color=colors_prov,
            text=[f"{v:.1f}" for v in prov_comp["avg_rate"]],
            textposition="outside", textfont=dict(color="#4A5568", size=11),
        ))
        fig_prov.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#F7F9FC", font_color="#4A5568", height=300,
            xaxis=dict(title="Avg Death Rate per 100k", gridcolor="#EDF2F7"),
            yaxis=dict(gridcolor="#EDF2F7"),
            margin=dict(l=0, r=50, t=10, b=30), showlegend=False,
        )
        st.plotly_chart(fig_prov, use_container_width=True)

        # Crisis callout
        peak_row = bc_opioid.loc[bc_opioid["apparent_opioid_toxicity_deaths"].idxmax()]
        st.markdown(
            f"""<div class="alert-box">
                <h4>📍 CRISIS CONTEXT FOR HEALTH AUTHORITY BRIEFINGS</h4>
                <p>BC's quarterly peak was <strong>{int(peak_row['apparent_opioid_toxicity_deaths'])} deaths</strong>
                in <strong>{peak_row['period']}</strong>. Cross-reference with the Community Health Equity tab
                to identify CHSAs where poverty and opioid rates are simultaneously elevated — these communities
                are your highest dual-burden intervention targets.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption("Source: opioid_harms_mock.csv · BC Centre for Disease Control surveillance")

# ── Footer
st.markdown("---")
st.caption("BC Population Health Equity Dashboard · HealthHack 2026 · Track 2 · BuildersVault × UVic Hacks")
