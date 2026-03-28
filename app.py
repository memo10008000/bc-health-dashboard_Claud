import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BC Population Health Equity Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  [data-testid="stAppViewContainer"] { background: #0F1117; color: #E8EAF0; }
  [data-testid="stSidebar"] { background: #161B27; border-right: 1px solid #2A3040; }
  [data-testid="stSidebar"] * { color: #C8D0E0 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #1A2035;
    border: 1px solid #2A3040;
    border-radius: 6px;
    padding: 16px;
  }
  [data-testid="metric-container"] label { color: #7B8FAB !important; font-size: 0.75rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E8EAF0 !important; font-size: 1.6rem !important; font-weight: 700 !important;
  }

  /* Alert box */
  .alert-box {
    background: #1E1020;
    border-left: 4px solid #C0392B;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
  }
  .alert-box h4 { color: #E74C3C; margin: 0 0 0.4rem 0; font-size: 0.85rem; letter-spacing: 0.06em; }
  .alert-box p  { color: #C8D0E0; margin: 0; font-size: 0.9rem; line-height: 1.5; }

  /* Briefing box */
  .briefing-box {
    background: #0D1B2A;
    border: 1px solid #1A4A6B;
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-top: 0.5rem;
  }
  .briefing-box p { color: #B8D4E8; font-size: 0.95rem; line-height: 1.7; margin: 0; }
  .briefing-source {
    font-size: 0.7rem; color: #4A6A84; margin-top: 0.75rem;
    letter-spacing: 0.05em; text-transform: uppercase;
  }

  /* Section headers */
  .section-label {
    font-size: 0.7rem; color: #4A6A84; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 0.5rem;
  }

  /* Vulnerability table */
  .vuln-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .vuln-table th {
    background: #1A2035; color: #7B8FAB; font-size: 0.7rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 8px 12px; text-align: left; border-bottom: 1px solid #2A3040;
  }
  .vuln-table td { padding: 8px 12px; border-bottom: 1px solid #1A2035; color: #C8D0E0; }
  .vuln-table tr:hover td { background: #1A2035; }
  .badge-red   { background:#4A1010; color:#E74C3C; padding:2px 8px; border-radius:3px; font-size:0.75rem; }
  .badge-amber { background:#3A2A00; color:#F0A500; padding:2px 8px; border-radius:3px; font-size:0.75rem; }
  .badge-green { background:#0A2A1A; color:#27AE60; padding:2px 8px; border-radius:3px; font-size:0.75rem; }

  h1, h2, h3 { color: #E8EAF0 !important; }
  hr { border-color: #2A3040; }
  .stButton>button {
    background: #1A4A6B; color: #E8EAF0; border: 1px solid #2A6A9B;
    border-radius: 4px; font-weight: 600; width: 100%;
  }
  .stButton>button:hover { background: #2A5A8B; }
</style>
""", unsafe_allow_html=True)

# ── Load data ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Support running from any working directory
    for path in ["bc_health_indicators.csv", "data/bc_health_indicators.csv"]:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("bc_health_indicators.csv not found. Place it in the same folder as app.py.")

df = load_data()

# ── Vulnerability Score ─────────────────────────────────────────────────────────
def compute_vulnerability(df_region: pd.DataFrame) -> pd.DataFrame:
    """
    Composite vulnerability score (0-100).
    Weights: 40% no family doctor | 30% below poverty | 30% opioid overdose rate
    Each indicator normalised to 0-1 within the full province for comparability.
    """
    d = df_region.copy()
    # Normalise against full province range
    def norm(col):
        mn, mx = df[col].min(), df[col].max()
        return (d[col] - mn) / (mx - mn + 1e-9)

    d["score_gp"]     = norm("pct_without_family_doctor") * 40
    d["score_poverty"] = norm("pct_below_poverty_line")   * 30
    d["score_opioid"]  = norm("opioid_overdose_rate")     * 30
    d["vulnerability_score"] = (d["score_gp"] + d["score_poverty"] + d["score_opioid"]).round(1)
    return d

# ── Claude API Briefing ─────────────────────────────────────────────────────────
def get_claude_briefing(region_name: str, top_community: dict, stats: dict) -> tuple[str, str]:
    """
    Returns (briefing_text, source_label).
    Falls back to rule-based if API key missing or call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if api_key:
        prompt = f"""You are a senior public health analyst writing for a BC Health Authority director.

Region: {region_name}
Most vulnerable community: {top_community['chsa_name']}
  - Vulnerability score: {top_community['vulnerability_score']}/100
  - % without family doctor: {top_community['pct_without_family_doctor']}%
  - % below poverty line: {top_community['pct_below_poverty_line']}%
  - Opioid overdose rate: {top_community['opioid_overdose_rate']} per 100k
  - Life expectancy: {top_community['life_expectancy']} years
  - ER visits per 1,000: {top_community['er_visits_per_1000']}

Region averages:
  - Avg % without GP: {stats['avg_no_gp']:.1f}%
  - Avg life expectancy: {stats['avg_le']:.1f} years
  - Avg opioid rate: {stats['avg_opioid']:.1f}

Write exactly 3 sentences as an Executive Briefing:
1. Name the top-priority community and its most critical indicator.
2. Explain the equity gap relative to the regional average.
3. Give a specific, actionable recommendation (e.g., mobile clinic, outreach, GP incentive).

Be direct, clinical, and data-driven. No bullet points. No headers."""

        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=15,
            )
            if resp.status_code == 200:
                text = resp.json()["content"][0]["text"].strip()
                return text, "Generated by Claude AI · claude-sonnet-4"
        except Exception:
            pass  # fall through to rule-based

    # ── Rule-based fallback ────────────────────────────────────────────────────
    c = top_community
    gap_gp     = c["pct_without_family_doctor"] - stats["avg_no_gp"]
    gap_le     = stats["avg_le"] - c["life_expectancy"]
    action     = (
        "a mobile primary care clinic and GP recruitment incentives"
        if c["pct_without_family_doctor"] > 30
        else "targeted opioid harm-reduction outreach and supervised consumption services"
        if c["opioid_overdose_rate"] > 40
        else "community health worker deployment and social determinants screening"
    )

    briefing = (
        f"{c['chsa_name']} is the highest-priority community in {region_name}, "
        f"with {c['pct_without_family_doctor']}% of residents lacking a family doctor "
        f"and an opioid overdose rate of {c['opioid_overdose_rate']} per 100,000. "
        f"This community sits {gap_gp:.1f} percentage points above the regional average for GP access gaps "
        f"and has a life expectancy {gap_le:.1f} years below the regional mean, "
        f"indicating compounding inequity. "
        f"The recommended immediate intervention is {action} in {c['chsa_name']}."
    )
    return briefing, "Generated by rule-based scoring engine (set ANTHROPIC_API_KEY to enable AI)"

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏥 BC Health Equity Dashboard")
    st.markdown('<div class="section-label">Health Authority</div>', unsafe_allow_html=True)

    authorities = sorted(df["health_authority"].unique())
    selected_ha = st.selectbox("", authorities, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">AI Synthesis Engine</div>', unsafe_allow_html=True)
    generate_btn = st.button("⚡ Generate Executive Briefing")

    st.markdown("---")
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.caption(
        "Tool for BC Health Authority population health analysts. "
        "Identifies underserved communities using composite vulnerability scoring across "
        "GP access, poverty, and opioid harm indicators."
    )
    st.caption("Data: BC Open Health Data · 78 CHSAs · 2024")

# ── Filter data ─────────────────────────────────────────────────────────────────
region_df = df[df["health_authority"] == selected_ha].copy()
region_df = compute_vulnerability(region_df)
region_df_sorted = region_df.sort_values("vulnerability_score", ascending=False)
top = region_df_sorted.iloc[0].to_dict()

region_stats = {
    "avg_no_gp":   region_df["pct_without_family_doctor"].mean(),
    "avg_le":      region_df["life_expectancy"].mean(),
    "avg_opioid":  region_df["opioid_overdose_rate"].mean(),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## {selected_ha} · Population Health Equity Overview")
st.markdown(
    f"*{len(region_df)} Community Health Service Areas · "
    f"Population: {region_df['population'].sum():,.0f}*"
)
st.markdown("---")

# ── ALERT: top priority community ──────────────────────────────────────────────
st.markdown(
    f"""<div class="alert-box">
        <h4>⚠ HIGHEST PRIORITY COMMUNITY FLAGGED</h4>
        <p><strong>{top['chsa_name']}</strong> — Vulnerability Score: <strong>{top['vulnerability_score']}/100</strong> &nbsp;·&nbsp;
        {top['pct_without_family_doctor']}% without GP &nbsp;·&nbsp;
        Opioid rate: {top['opioid_overdose_rate']}/100k &nbsp;·&nbsp;
        Life expectancy: {top['life_expectancy']} yrs</p>
    </div>""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROW 1 — KEY METRICS
# ═══════════════════════════════════════════════════════════════════════════════
m1, m2, m3, m4 = st.columns(4)

highest_er    = region_df.loc[region_df["er_visits_per_1000"].idxmax()]
lowest_le     = region_df.loc[region_df["life_expectancy"].idxmin()]
highest_no_gp = region_df.loc[region_df["pct_without_family_doctor"].idxmax()]
highest_opioid = region_df.loc[region_df["opioid_overdose_rate"].idxmax()]

m1.metric(
    "Highest ER Pressure",
    f"{highest_er['er_visits_per_1000']:.0f}/1k",
    highest_er["chsa_name"],
)
m2.metric(
    "Lowest Life Expectancy",
    f"{lowest_le['life_expectancy']} yrs",
    lowest_le["chsa_name"],
)
m3.metric(
    "Worst GP Access Gap",
    f"{highest_no_gp['pct_without_family_doctor']}%",
    highest_no_gp["chsa_name"],
)
m4.metric(
    "Highest Opioid Rate",
    f"{highest_opioid['opioid_overdose_rate']}/100k",
    highest_opioid["chsa_name"],
)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROW 2 — VISUAL 1 (Bar) + VISUAL 2 (Scatter)
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns(2, gap="large")

# ── Visual 1: GP Access Gap Bar Chart ──────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">Visual 1 — Primary Care Access Gap</div>', unsafe_allow_html=True)

    bar_df = region_df.sort_values("pct_without_family_doctor", ascending=True)
    colors = [
        "#E74C3C" if v > 30
        else "#F0A500" if v > 20
        else "#27AE60"
        for v in bar_df["pct_without_family_doctor"]
    ]

    fig_bar = go.Figure(go.Bar(
        x=bar_df["pct_without_family_doctor"],
        y=bar_df["chsa_name"],
        orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in bar_df["pct_without_family_doctor"]],
        textposition="outside",
        textfont=dict(color="#C8D0E0", size=11),
    ))
    fig_bar.add_vline(
        x=region_df["pct_without_family_doctor"].mean(),
        line_dash="dash", line_color="#4A6A84",
        annotation_text=f"Avg {region_stats['avg_no_gp']:.1f}%",
        annotation_font_color="#4A6A84",
    )
    fig_bar.update_layout(
        plot_bgcolor="#0F1117", paper_bgcolor="#0F1117",
        font_color="#C8D0E0", height=max(280, len(bar_df) * 28),
        xaxis=dict(title="% Without Family Doctor", gridcolor="#1A2035", color="#7B8FAB"),
        yaxis=dict(gridcolor="#1A2035", color="#C8D0E0"),
        margin=dict(l=0, r=40, t=10, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Visual 2: Wealth-Health Scatter Plot ───────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">Visual 2 — Wealth–Health Equity Gap</div>', unsafe_allow_html=True)

    fig_scatter = px.scatter(
        region_df,
        x="median_household_income",
        y="life_expectancy",
        size="population",
        color="vulnerability_score",
        color_continuous_scale=["#27AE60", "#F0A500", "#E74C3C"],
        hover_name="chsa_name",
        hover_data={
            "median_household_income": ":,.0f",
            "life_expectancy": True,
            "vulnerability_score": True,
            "population": ":,.0f",
        },
        labels={
            "median_household_income": "Median Household Income ($)",
            "life_expectancy": "Life Expectancy (years)",
            "vulnerability_score": "Vulnerability",
        },
        size_max=35,
    )
    fig_scatter.update_layout(
        plot_bgcolor="#0F1117", paper_bgcolor="#0F1117",
        font_color="#C8D0E0", height=max(280, len(bar_df) * 28),
        xaxis=dict(gridcolor="#1A2035", color="#7B8FAB", tickprefix="$", tickformat=","),
        yaxis=dict(gridcolor="#1A2035", color="#7B8FAB"),
        coloraxis_colorbar=dict(
            title="Vuln.", tickfont=dict(color="#7B8FAB"), title_font_color="#7B8FAB"
        ),
        margin=dict(l=0, r=20, t=10, b=30),
    )
    # Annotate top priority community
    fig_scatter.add_annotation(
        x=top["median_household_income"],
        y=top["life_expectancy"],
        text=f"▲ {top['chsa_name']}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#E74C3C",
        font=dict(color="#E74C3C", size=11),
        bgcolor="#1E1020",
        bordercolor="#E74C3C",
        borderwidth=1,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  ROW 3 — VULNERABILITY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">Visual 3 — Community Vulnerability Ranking</div>', unsafe_allow_html=True)
st.caption("Composite score: 40% GP access · 30% poverty rate · 30% opioid harm. Ranked highest to lowest.")

def score_badge(score):
    if score >= 60: return f'<span class="badge-red">{score}</span>'
    if score >= 35: return f'<span class="badge-amber">{score}</span>'
    return f'<span class="badge-green">{score}</span>'

display_cols = [
    "chsa_name", "vulnerability_score",
    "pct_without_family_doctor", "pct_below_poverty_line",
    "opioid_overdose_rate", "life_expectancy", "er_visits_per_1000"
]
table_df = region_df_sorted[display_cols].head(len(region_df))

rows_html = ""
for _, r in table_df.iterrows():
    rows_html += f"""<tr>
        <td>{r['chsa_name']}</td>
        <td>{score_badge(r['vulnerability_score'])}</td>
        <td>{r['pct_without_family_doctor']}%</td>
        <td>{r['pct_below_poverty_line']}%</td>
        <td>{r['opioid_overdose_rate']}</td>
        <td>{r['life_expectancy']}</td>
        <td>{r['er_visits_per_1000']:.0f}</td>
    </tr>"""

st.markdown(f"""
<table class="vuln-table">
  <thead>
    <tr>
      <th>Community (CHSA)</th>
      <th>Vuln. Score</th>
      <th>% No GP</th>
      <th>% Poverty</th>
      <th>Opioid Rate</th>
      <th>Life Exp.</th>
      <th>ER/1k</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  AI EXECUTIVE BRIEFING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-label">AI Synthesis Engine — Executive Briefing</div>', unsafe_allow_html=True)
st.markdown("*Answers: Which community needs the next outreach clinic, and why?*")

if generate_btn or "briefing" in st.session_state:
    if generate_btn:
        with st.spinner("Synthesising regional health data..."):
            briefing, source = get_claude_briefing(selected_ha, top, region_stats)
        st.session_state["briefing"] = briefing
        st.session_state["briefing_source"] = source

    st.markdown(
        f"""<div class="briefing-box">
            <p>{st.session_state['briefing']}</p>
            <div class="briefing-source">⚙ {st.session_state['briefing_source']}</div>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.info("👈 Select a Health Authority then click **Generate Executive Briefing** in the sidebar.")

# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "BC Population Health Equity Dashboard · Built for HealthHack 2026 · Track 2: Population Health & Health Equity · "
    "Data: BC Open Health · BuildersVault × UVic Hacks"
)
