"""
Page 1 — Data Input
Handles three data sources:
  A) Screaming Frog embeddings CSV upload
  B) Google Search Console OAuth connection + data fetch
  C) Optional URL list (GSC sitemap / CSV upload / paste)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from urllib.parse import urlparse

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings_handler import parse_sf_embeddings, get_summary_stats
from utils.gsc_handler import (
    init_from_stored_refresh_token,
    is_authenticated,
    fetch_properties,
    fetch_gsc_data,
    fetch_sitemap_urls,
    logout,
)

st.set_page_config(
    page_title="Data Input — Duplicate Content Detector",
    page_icon="📥",
    layout="wide",
)

st.title("📥 Data Input")
st.caption("Load your data sources before running analysis on the other pages.")

# ─── Auto-login from stored refresh token (runs silently on every page load) ─

if not is_authenticated():
    init_from_stored_refresh_token()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — Screaming Frog Embeddings
# ═══════════════════════════════════════════════════════════════════════════

st.header("A — Screaming Frog Embeddings", divider="blue")

with st.expander("ℹ️ How to export embeddings from Screaming Frog", expanded=False):
    st.markdown("""
    1. Run a crawl in Screaming Frog SEO Spider
    2. Go to **Bulk Export → Content → Embeddings** (requires Screaming Frog v20+)
    3. The export will be a CSV with the page URL and embedding vector columns
    4. Upload that CSV file below

    **Expected format:**
    - Column 1: `Address` (the page URL)
    - Remaining columns: numeric values (e.g., `dim_0`, `dim_1`, ... `dim_383`)

    The app auto-detects the URL column and all numeric embedding dimensions.
    """)

uploaded_embeddings = st.file_uploader(
    "Upload Screaming Frog embeddings CSV",
    type=["csv"],
    key="sf_embeddings_upload",
    help="CSV with URL column + numeric embedding dimensions",
)

if uploaded_embeddings:
    with st.spinner("Parsing embeddings..."):
        url_df, embeddings_matrix, error = parse_sf_embeddings(uploaded_embeddings)

    if error:
        st.error(f"**Parse error:** {error}")
    else:
        n_urls = len(url_df)
        n_dims = embeddings_matrix.shape[1]

        # Warn on large datasets
        if n_urls > 300:
            st.warning(
                f"**{n_urls} URLs detected.** Pairwise similarity computation scales as N², "
                "which may be slow on Streamlit free tier. Consider filtering to a URL subset "
                "using the URL List section below."
            )
        elif n_urls > 150:
            st.info(f"**{n_urls} URLs** — analysis will take a few seconds.")

        col1, col2, col3 = st.columns(3)
        col1.metric("URLs Loaded", n_urls)
        col2.metric("Embedding Dimensions", n_dims)
        col3.metric("Pairs to Analyze", f"{n_urls * (n_urls - 1) // 2:,}")

        st.dataframe(url_df.head(5), use_container_width=True, hide_index=True)

        # Store in session state
        st.session_state.url_df = url_df
        st.session_state.embeddings_matrix = embeddings_matrix
        st.session_state.sf_loaded = True
        st.success(f"Embeddings loaded: {n_urls} URLs × {n_dims} dimensions")

elif st.session_state.get("sf_loaded"):
    st.info(
        f"Embeddings already loaded: "
        f"{len(st.session_state.url_df)} URLs × "
        f"{st.session_state.embeddings_matrix.shape[1]} dimensions"
    )
    if st.button("Clear embeddings", key="clear_sf"):
        for key in ["url_df", "embeddings_matrix", "sf_loaded", "sim_matrix"]:
            st.session_state.pop(key, None)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — Google Search Console
# ═══════════════════════════════════════════════════════════════════════════

st.header("B — Google Search Console", divider="green")

if not is_authenticated():
    # Check which secrets are present to give a precise error message
    try:
        gsc_secrets = st.secrets.get("gsc", {})
        has_client_id = bool(gsc_secrets.get("client_id", "").strip().replace("YOUR_GOOGLE_OAUTH_CLIENT_ID.apps.googleusercontent.com", ""))
        has_client_secret = bool(gsc_secrets.get("client_secret", "").strip().replace("YOUR_GOOGLE_OAUTH_CLIENT_SECRET", ""))
        has_refresh_token = bool(gsc_secrets.get("refresh_token", "").strip().replace("YOUR_REFRESH_TOKEN", ""))
    except Exception:
        has_client_id = has_client_secret = has_refresh_token = False

    if not has_client_id or not has_client_secret:
        st.warning(
            "**GSC credentials not configured.** "
            "Add your `client_id` and `client_secret` to Streamlit secrets under `[gsc]`. "
            "See the README for Google Cloud setup instructions."
        )
    elif not has_refresh_token:
        st.error(
            "**Refresh token missing.** Your `client_id` and `client_secret` are set, "
            "but `refresh_token` is empty. \n\n"
            "**To get your refresh token:**\n"
            "1. Run locally: `python scripts/get_refresh_token.py`\n"
            "2. A browser opens → sign in with your Google account\n"
            "3. Copy the printed `refresh_token` value\n"
            "4. Add it to your Streamlit secrets as `gsc.refresh_token`\n\n"
            "No redirect URI configuration is needed — the script handles everything locally."
        )
    else:
        st.error(
            "Authentication failed. Your `refresh_token` may be invalid or revoked. "
            "Run `scripts/get_refresh_token.py` again to get a fresh token."
        )

else:
    # Already authenticated
    st.success("✅ Connected to Google Search Console")

    col_logout, _ = st.columns([1, 3])
    with col_logout:
        if st.button("Disconnect", key="gsc_logout"):
            logout()
            st.rerun()

    # Property selector
    if "gsc_properties" not in st.session_state:
        with st.spinner("Fetching properties..."):
            st.session_state.gsc_properties = fetch_properties()

    properties = st.session_state.get("gsc_properties", [])

    if not properties:
        st.warning("No verified GSC properties found for this account.")
    else:
        selected_property = st.selectbox(
            "Select GSC Property",
            options=properties,
            index=0 if "selected_property" not in st.session_state else (
                properties.index(st.session_state.selected_property)
                if st.session_state.selected_property in properties else 0
            ),
            key="property_selector",
        )
        st.session_state.selected_property = selected_property

        # Date range
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "Start Date",
                value=date.today() - timedelta(days=90),
                max_value=date.today() - timedelta(days=2),
                key="gsc_start_date",
            )
        with col_end:
            end_date = st.date_input(
                "End Date",
                value=date.today() - timedelta(days=2),
                max_value=date.today() - timedelta(days=2),
                key="gsc_end_date",
            )

        if st.button("🔄 Fetch GSC Data", type="primary", key="fetch_gsc"):
            with st.spinner("Fetching Search Analytics data..."):
                rows = fetch_gsc_data(
                    selected_property,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )

            if rows:
                df = pd.DataFrame(rows)
                st.session_state.gsc_data = df
                st.session_state.gsc_property = selected_property
                st.session_state.gsc_start = str(start_date)
                st.session_state.gsc_end = str(end_date)
                st.session_state.gsc_loaded = True
                st.success(f"Fetched {len(df):,} rows from GSC")
            else:
                st.error("No data returned. Check your property selection and date range.")

        if st.session_state.get("gsc_loaded"):
            gsc_df = st.session_state.gsc_data
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{len(gsc_df):,}")
            col2.metric("Unique Queries", f"{gsc_df['query'].nunique():,}")
            col3.metric("Unique URLs", f"{gsc_df['page'].nunique():,}")
            col4.metric("Total Clicks", f"{gsc_df['clicks'].sum():,}")

            with st.expander("Preview GSC data"):
                st.dataframe(
                    gsc_df.head(20),
                    use_container_width=True,
                    hide_index=True,
                )

# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — URL List (Optional Filter)
# ═══════════════════════════════════════════════════════════════════════════

st.header("C — URL List Filter (Optional)", divider="gray")
st.caption(
    "Optionally restrict the analysis to a specific set of URLs. "
    "If left empty, all URLs from the embeddings file / GSC data will be used."
)

url_input_tab1, url_input_tab2, url_input_tab3 = st.tabs([
    "📋 Paste URLs",
    "📁 Upload CSV/TXT",
    "🗺 Pull from GSC Sitemap",
])

filter_urls: list[str] = []

with url_input_tab1:
    pasted = st.text_area(
        "Paste URLs (one per line)",
        height=150,
        placeholder="https://example.com/page-1\nhttps://example.com/page-2",
        key="pasted_urls",
    )
    if pasted:
        filter_urls = [u.strip() for u in pasted.strip().splitlines() if u.strip().startswith("http")]
        if filter_urls:
            st.info(f"{len(filter_urls)} URLs parsed")

with url_input_tab2:
    uploaded_urls = st.file_uploader(
        "Upload CSV or TXT with URLs",
        type=["csv", "txt"],
        key="url_file_upload",
    )
    if uploaded_urls:
        content = uploaded_urls.read().decode("utf-8", errors="ignore")
        # Try CSV first, then plain text
        try:
            df_urls = pd.read_csv(pd.io.common.StringIO(content))
            url_col = None
            for col in df_urls.columns:
                if df_urls[col].astype(str).str.startswith("http").any():
                    url_col = col
                    break
            if url_col:
                filter_urls = df_urls[url_col].dropna().astype(str).tolist()
        except Exception:
            filter_urls = [u.strip() for u in content.splitlines() if u.strip().startswith("http")]

        if filter_urls:
            st.info(f"{len(filter_urls)} URLs loaded from file")
        else:
            st.warning("No URLs found in uploaded file.")

with url_input_tab3:
    if is_authenticated() and st.session_state.get("selected_property"):
        if st.button("Fetch sitemap URLs from GSC", key="fetch_sitemaps"):
            with st.spinner("Fetching sitemaps..."):
                sitemap_paths = fetch_sitemap_urls(st.session_state.selected_property)
            if sitemap_paths:
                st.info(f"Found {len(sitemap_paths)} sitemap(s):")
                st.write(sitemap_paths)
            else:
                st.warning("No sitemaps found or GSC sitemap data is unavailable.")
    else:
        st.info("Connect to GSC in Section B first to use this option.")

# Save URL filter to session
if filter_urls:
    st.session_state.filter_urls = list(set(filter_urls))
    st.success(f"URL filter active: {len(st.session_state.filter_urls)} URLs")

# ─── Status Summary ──────────────────────────────────────────────────────────

st.divider()
st.subheader("Status Summary")

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.get("sf_loaded"):
        st.success("✅ SF Embeddings loaded")
    else:
        st.error("❌ SF Embeddings not loaded")

with col2:
    if st.session_state.get("gsc_loaded"):
        st.success("✅ GSC Data loaded")
    else:
        st.warning("⚠️ GSC Data not loaded (optional for Layer 1)")

with col3:
    if st.session_state.get("filter_urls"):
        st.info(f"🔍 URL filter: {len(st.session_state.filter_urls)} URLs")
    else:
        st.info("No URL filter applied (all URLs used)")

st.markdown("---")
st.markdown(
    "**Next step →** Navigate to **Semantic Similarity** or **GSC Cannibalization** "
    "in the sidebar to run the analysis."
)
