"""
Page 2 — Semantic Similarity Analysis
Computes and displays pairwise cosine similarity from Screaming Frog embeddings.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings_handler import (  # noqa: E402
    THRESHOLD_HIGH,
    THRESHOLD_MEDIUM,
    compute_similarity_matrix,
    get_pairs_above_threshold,
    get_summary_stats,
)
from utils.url_exclusions import is_homepage, should_exclude  # noqa: E402

st.set_page_config(
    page_title="Semantic Similarity — Duplicate Content Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Semantic Similarity Analysis")
st.caption(
    "Detects pages with highly similar content using pre-computed Screaming Frog embeddings.")

# ─── Guard: require SF embeddings ────────────────────────────────────────────

if not st.session_state.get(
        "sf_loaded") or st.session_state.get("url_df") is None:
    st.warning(
        "No embeddings loaded. Please go to **Data Input** and upload your Screaming Frog embeddings CSV.")
    st.stop()

url_df_orig: pd.DataFrame = st.session_state.url_df
embeddings_orig: np.ndarray = st.session_state.embeddings_matrix

# ─── Controls ────────────────────────────────────────────────────────────────

st.sidebar.header("Analysis Settings")

threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.40,
    max_value=1.00,
    value=THRESHOLD_HIGH,
    step=0.01,
    help="Pairs above this score are flagged. >0.85 = likely duplicate, 0.60–0.85 = overlapping.",
)

max_pairs_display = st.sidebar.number_input(
    "Max pairs to display",
    min_value=10,
    max_value=5000,
    value=500,
    step=50,
)

# ─── Compute similarity matrix from ORIGINAL data (cached) ───────────────────
# Must use original (pre-exclusion) count as cache key so Combined Risk page
# can safely slice the same matrix with its own exclusion mask.

if (
    "sim_matrix" not in st.session_state
    or st.session_state.get("sim_matrix_url_count") != len(url_df_orig)
):
    with st.spinner(f"Computing pairwise similarity for {len(url_df_orig)} URLs..."):
        sim_matrix_full = compute_similarity_matrix(embeddings_orig)
        st.session_state.sim_matrix = sim_matrix_full
        st.session_state.sim_matrix_url_count = len(url_df_orig)
else:
    sim_matrix_full = st.session_state.sim_matrix

# ─── Build combined exclusion mask over the original url_df ──────────────────

keep = np.ones(len(url_df_orig), dtype=bool)

exclude_patterns = st.session_state.get("exclude_patterns", [])
if exclude_patterns:
    keep &= ~url_df_orig["url"].apply(
        lambda u: should_exclude(u, exclude_patterns)).values

_prop = st.session_state.get("selected_property") or ""
if st.session_state.get("exclude_homepage", True) and _prop:
    keep &= ~url_df_orig["url"].apply(
        lambda u: is_homepage(u, _prop)).values

# Apply mask once to both url_df and sim_matrix
url_df = url_df_orig[keep].reset_index(drop=True)
sim_matrix = sim_matrix_full[np.ix_(keep, keep)]

_n_excluded = len(url_df_orig) - len(url_df)
if _n_excluded:
    st.info(
        f"Exclusions active: {_n_excluded} URLs filtered out "
        f"({len(url_df)} remaining)"
    )

if len(url_df) < 2:
    st.error("Need at least 2 URLs to compute similarity. Adjust your URL filter.")
    st.stop()

# ─── Summary metrics ─────────────────────────────────────────────────────────

stats = get_summary_stats(url_df, sim_matrix, threshold, THRESHOLD_MEDIUM)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("URLs Analyzed", stats["total_urls"])
col2.metric("Total Pairs", f"{stats['total_pairs']:,}")
col3.metric(
    f"Pairs ≥ {threshold:.0%}",
    stats["high_risk_pairs"],
    delta="High Risk" if stats["high_risk_pairs"] > 0 else None,
    delta_color="inverse",
)
col4.metric(
    f"Pairs 60%–{threshold:.0%}",
    stats["medium_risk_pairs"],
)
col5.metric("Avg Similarity", f"{stats['avg_similarity']:.1%}")

st.divider()

# ─── Similar Pairs ───────────────────────────────────────────────────────────

st.subheader(f"URL Pairs with Similarity ≥ {threshold:.0%}")

pairs_df = get_pairs_above_threshold(
    url_df, sim_matrix, threshold=threshold, max_pairs=int(max_pairs_display)
)

if pairs_df.empty:
    st.info(
        f"No pairs found above {threshold:.0%} threshold. Try lowering the threshold.")
else:
    # Color-code risk levels
    def color_risk(val):
        colors = {
            "High": "background-color:#ffcccc",
            "Medium": "background-color:#fff3cc",
            "Low": "background-color:#e8f5e9"}
        return colors.get(val, "")

    styled = pairs_df.style.applymap(color_risk, subset=["risk_level"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        f"Showing {len(pairs_df):,} pairs (max {int(max_pairs_display):,})")

    csv = pairs_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download pairs as CSV",
        data=csv,
        file_name="similar_pairs.csv",
        mime="text/csv",
    )

# ─── Navigation hint ─────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**Next steps →** Check **GSC Cannibalization** for keyword overlap analysis, "
    "or go to **AI Recommendations** to get specific remediation advice."
)
