# BC Population Health Equity Dashboard
### HealthHack 2026 · Track 2: Population Health & Health Equity
**BuildersVault × UVic Hacks · March 27–28, 2026**

---

## The Problem

Health outcomes vary dramatically across BC communities, yet the data that could reveal these gaps is scattered, unfiltered, and hard to act on. Public health analysts spend hours aggregating indicators — and still struggle to justify *which* community should receive the next clinic or outreach program.

Most health research is built on WEIRD populations (Western, Educated, Industrialized, Rich, Democratic), leaving rural, Indigenous, and low-income communities systematically underrepresented in resource allocation decisions.

## Our Solution

A **Streamlit dashboard** that helps BC Health Authority population health analysts:

1. **Identify** the most underserved communities across any Health Authority
2. **Understand** the compounding equity gaps driving poor outcomes
3. **Communicate** where support should be prioritized — with a defensible, data-backed AI summary

### The Synthesis Engine

A composite **Vulnerability Score (0–100)** ranks every CHSA using:
- **40%** — % of residents without a family doctor (access gap)
- **30%** — % below the poverty line (social determinant)
- **30%** — Opioid overdose rate per 100k (crisis indicator)

An **AI Executive Briefing** (Claude API + rule-based fallback) answers the question every health authority director needs answered: *"Which community needs the next outreach clinic, and why?"*

---

## Team Members

> *(Add your names here)*

## Challenge Track

**Track 2: Population Health & Health Equity**

## Tech Stack

| Layer | Tool |
|---|---|
| App framework | Streamlit |
| Data processing | Pandas |
| Visualizations | Plotly |
| AI Briefing | Anthropic Claude API (claude-sonnet-4) + rule-based fallback |
| Data | BC Open Health — 78 CHSAs across 5 Health Authorities |

---

## How to Run

### Option 1 — Streamlit Cloud (Recommended for judges)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your fork
3. Set secret: `ANTHROPIC_API_KEY = "your-key"` (optional — rule-based fallback works without it)
4. Deploy

### Option 2 — Local

```bash
pip install streamlit pandas plotly requests
# Place bc_health_indicators.csv in the same folder as app.py
streamlit run app.py
```

### Environment Variable (optional — enables Claude AI briefing)

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Without the key, the app automatically uses the rule-based scoring engine. **Both modes are fully functional.**

---

## Data

`bc_health_indicators.csv` — 78 Community Health Service Areas (CHSAs) across:
- Island Health · Fraser · Interior · Northern · Vancouver Coastal

Key indicators: `pct_without_family_doctor`, `median_household_income`, `life_expectancy`, `opioid_overdose_rate`, `er_visits_per_1000`, `pct_below_poverty_line`, and more.

---

## Demo Video

> *(Add Loom / YouTube link here if submitting locally)*

---

## Slides

> *(Add Google Slides or PDF link here)*
