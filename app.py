"""
Duplicate Content Detector — Main Entry Point
A Streamlit app for detecting duplicate and cannibalizing content using:
  1. Screaming Frog embeddings (semantic similarity)
  2. Google Search Console data (keyword cannibalization)
  3. AI-powered recommendations via OpenRouter
"""

import streamlit as st

st.set_page_config(
    page_title="Duplicate Content Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Duplicate Content Detector — SEO tool for finding and fixing content duplication issues.",
    },
)

# ─── Initialize session state defaults ───────────────────────────────────────

defaults = {
    "sf_loaded": False,
    "gsc_loaded": False,
    "url_df": None,
    "embeddings_matrix": None,
    "sim_matrix": None,
    "sim_matrix_url_count": 0,
    "gsc_data": None,
    "gsc_properties": None,
    "selected_property": None,
    "gsc_token_data": None,
    "cannibalization_findings": None,
    "url_cannibal_summary": {},
    "combined_df": None,
    "clusters": None,
    "cluster_df": None,
    "url_risk_df": None,
    "filter_urls": [],
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── Home page content ────────────────────────────────────────────────────────

st.title("🔎 Duplicate Content Detector")
st.markdown(
    "**Identify and prioritize duplicate and cannibalized content across your website "
    "using embeddings, Search Console data, and AI-powered recommendations.**"
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Layer 1: Semantic Similarity")
    st.markdown(
        "Upload your **Screaming Frog embeddings CSV** to detect pages that are "
        "semantically similar — even if the exact words differ. Uses cosine similarity "
        "on pre-computed embedding vectors."
    )
    st.markdown("**Signal:** Content sounds the same even with different wording")
    if st.session_state.sf_loaded:
        n = len(st.session_state.url_df) if st.session_state.url_df is not None else 0
        st.success(f"✅ {n} URLs loaded")
    else:
        st.info("⬆️ Upload SF embeddings to activate")

with col2:
    st.markdown("### Layer 2: Keyword Cannibalization")
    st.markdown(
        "Connect to **Google Search Console** to find queries where multiple pages "
        "are competing for clicks and impressions — a direct SEO impact signal."
    )
    st.markdown("**Signal:** Multiple URLs splitting traffic for the same queries")
    if st.session_state.gsc_loaded:
        n = len(st.session_state.gsc_data) if st.session_state.gsc_data is not None else 0
        st.success(f"✅ {n:,} GSC rows loaded")
    else:
        st.info("🔐 Connect to GSC to activate")

with col3:
    st.markdown("### Layer 3: AI Recommendations")
    st.markdown(
        "For flagged URL pairs, get specific remediation advice from **GPT-5.1, "
        "Claude Sonnet 4.6, or Gemini Flash** — including which page to keep, "
        "redirect, canonicalize, or differentiate."
    )
    st.markdown("**Output:** Actionable SEO instructions per duplicate pair")
    try:
        has_key = bool(st.secrets.get("OPENROUTER_API_KEY", ""))
    except Exception:
        has_key = False
    if has_key:
        st.success("✅ OpenRouter configured")
    else:
        st.info("🔑 Add OpenRouter API key to enable")

st.divider()

# ─── How it works ────────────────────────────────────────────────────────────

with st.expander("📖 How to use this app", expanded=True):
    st.markdown("""
    ### Step-by-step

    **1. Start on the Data Input page** (sidebar → 📥 Data Input)
    - Upload your Screaming Frog embeddings CSV
    - Connect to Google Search Console via OAuth
    - Optionally paste/upload a URL list to filter the analysis

    **2. Run Semantic Similarity Analysis** (sidebar → 🔍 Semantic Similarity)
    - See all URL pairs ranked by cosine similarity
    - Explore clusters of similar pages
    - Download results as CSV

    **3. Check GSC Cannibalization** (sidebar → 📊 GSC Cannibalization)
    - Find queries where multiple URLs are splitting traffic
    - Ranked by SEO impact score (clicks + impressions × competing URLs)

    **4. Review Combined Risk Dashboard** (sidebar → 🎯 Combined Risk)
    - See which URL pairs are flagged by BOTH signals = Critical priority
    - Scatter plot: semantic similarity vs. GSC impact

    **5. Get AI Recommendations** (sidebar → 🤖 AI Recommendations)
    - Select flagged pairs and choose your preferred AI model
    - Receive specific advice: consolidate, redirect, canonicalize, or differentiate

    ---

    ### Similarity Score Guide

    | Score | Meaning | Action |
    |-------|---------|--------|
    | **> 0.85** | Likely duplicate content | Consolidate or redirect |
    | **0.60 – 0.85** | Overlapping intent/topic | Review and differentiate |
    | **< 0.60** | Distinct content | Usually fine |

    ---

    ### What Screaming Frog embeddings export?
    Screaming Frog SEO Spider (v20+) can generate AI embeddings for every crawled page.
    Go to **Bulk Export → Content → Embeddings** after your crawl completes.
    The CSV will contain the page URL and 384 numeric embedding dimensions.
    """)

# ─── Quick status panel ───────────────────────────────────────────────────────

st.subheader("Current Session Status")

status_cols = st.columns(5)
status_cols[0].metric(
    "SF URLs",
    len(st.session_state.url_df) if st.session_state.url_df is not None else 0,
    label_visibility="visible",
)
status_cols[1].metric(
    "GSC Rows",
    f"{len(st.session_state.gsc_data):,}" if st.session_state.gsc_data is not None else 0,
)
status_cols[2].metric(
    "Cannibal Queries",
    len(st.session_state.cannibalization_findings or []),
)
status_cols[3].metric(
    "Clusters Found",
    len(st.session_state.clusters or []),
)
status_cols[4].metric(
    "Combined Pairs",
    len(st.session_state.combined_df) if st.session_state.combined_df is not None else 0,
)

st.divider()
st.caption(
    "Built with Streamlit · Powered by Screaming Frog embeddings, Google Search Console, "
    "and OpenRouter AI · No data is stored or shared externally."
)
