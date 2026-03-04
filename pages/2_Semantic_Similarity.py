"""
Page 2 — Semantic Similarity Analysis
Computes and displays pairwise cosine similarity from Screaming Frog embeddings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings_handler import (
    compute_similarity_matrix,
    get_pairs_above_threshold,
    build_heatmap_data,
    get_summary_stats,
    THRESHOLD_HIGH,
    THRESHOLD_MEDIUM,
)
from utils.clustering import build_clusters, clusters_to_dataframe, compute_url_risk_summary

st.set_page_config(
    page_title="Semantic Similarity — Duplicate Content Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Semantic Similarity Analysis")
st.caption("Detects pages with highly similar content using pre-computed Screaming Frog embeddings.")

# ─── Guard: require SF embeddings ────────────────────────────────────────────

if not st.session_state.get("sf_loaded") or st.session_state.get("url_df") is None:
    st.warning("No embeddings loaded. Please go to **Data Input** and upload your Screaming Frog embeddings CSV.")
    st.stop()

url_df: pd.DataFrame = st.session_state.url_df
embeddings: np.ndarray = st.session_state.embeddings_matrix

# Apply URL filter if set
filter_urls = st.session_state.get("filter_urls", [])
if filter_urls:
    mask = url_df["url"].isin(filter_urls)
    url_df = url_df[mask].reset_index(drop=True)
    embeddings = embeddings[mask.values]
    st.info(f"URL filter applied: showing {len(url_df)} of {len(st.session_state.url_df)} URLs")

if len(url_df) < 2:
    st.error("Need at least 2 URLs to compute similarity. Adjust your URL filter.")
    st.stop()

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

cluster_threshold = st.sidebar.slider(
    "Cluster Threshold",
    min_value=0.40,
    max_value=1.00,
    value=THRESHOLD_HIGH,
    step=0.01,
    help="Minimum similarity to link URLs into the same cluster.",
)

# ─── Compute similarity matrix (cached in session state) ─────────────────────

if (
    "sim_matrix" not in st.session_state
    or st.session_state.get("sim_matrix_url_count") != len(url_df)
):
    with st.spinner(f"Computing pairwise similarity for {len(url_df)} URLs..."):
        sim_matrix = compute_similarity_matrix(embeddings)
        st.session_state.sim_matrix = sim_matrix
        st.session_state.sim_matrix_url_count = len(url_df)
else:
    sim_matrix = st.session_state.sim_matrix

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

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Similar Pairs",
    "🔵 Clusters",
    "🗺 Heatmap",
    "📊 URL Risk Summary",
])

# ── Tab 1: Similar Pairs ─────────────────────────────────────────────────────

with tab1:
    st.subheader(f"URL Pairs with Similarity ≥ {threshold:.0%}")

    pairs_df = get_pairs_above_threshold(
        url_df, sim_matrix, threshold=threshold, max_pairs=int(max_pairs_display)
    )

    if pairs_df.empty:
        st.info(f"No pairs found above {threshold:.0%} threshold. Try lowering the threshold.")
    else:
        # Color-code risk levels
        def color_risk(val):
            colors = {"High": "background-color:#ffcccc", "Medium": "background-color:#fff3cc", "Low": "background-color:#e8f5e9"}
            return colors.get(val, "")

        styled = pairs_df.style.applymap(color_risk, subset=["risk_level"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(pairs_df):,} pairs (max {int(max_pairs_display):,})")

        csv = pairs_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download pairs as CSV",
            data=csv,
            file_name="similar_pairs.csv",
            mime="text/csv",
        )

# ── Tab 2: Clusters ───────────────────────────────────────────────────────────

with tab2:
    st.subheader(f"Duplicate Clusters (threshold = {cluster_threshold:.0%})")

    with st.spinner("Building clusters..."):
        clusters = build_clusters(url_df, sim_matrix, threshold=cluster_threshold)

    if not clusters:
        st.info(f"No clusters found at {cluster_threshold:.0%} threshold. Try lowering the cluster threshold.")
    else:
        cluster_df = clusters_to_dataframe(clusters, url_df, sim_matrix)
        st.metric("Clusters Found", len(clusters))

        for i, cluster in enumerate(clusters, start=1):
            avg_sim = cluster_df[cluster_df["cluster_id"] == i]["avg_similarity"].iloc[0]
            with st.expander(
                f"Cluster {i} — {len(cluster)} URLs (avg similarity: {avg_sim:.1%})",
                expanded=(i <= 5),
            ):
                for url in cluster:
                    st.markdown(f"- `{url}`")

        csv = cluster_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download clusters as CSV",
            data=csv,
            file_name="url_clusters.csv",
            mime="text/csv",
        )

        # Save clusters to session for use in AI page
        st.session_state.clusters = clusters
        st.session_state.cluster_df = cluster_df

# ── Tab 3: Heatmap ────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Similarity Heatmap (Top 50 URLs)")

    max_heatmap = st.slider("Max URLs in heatmap", 10, 50, 30, key="heatmap_size")

    labels, sub_matrix = build_heatmap_data(url_df, sim_matrix.copy(), max_urls=max_heatmap)

    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(
            z=sub_matrix,
            x=labels,
            y=labels,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            text=np.round(sub_matrix, 2),
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(title="Cosine Similarity"),
        ))
        fig.update_layout(
            height=600,
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Install `plotly` to enable the heatmap visualization. Showing raw matrix instead.")
        st.dataframe(
            pd.DataFrame(sub_matrix, index=labels, columns=labels),
            use_container_width=True,
        )

# ── Tab 4: URL Risk Summary ───────────────────────────────────────────────────

with tab4:
    st.subheader("Per-URL Risk Summary")
    st.caption("Each URL ranked by its highest similarity score to any other URL.")

    with st.spinner("Computing per-URL risk..."):
        risk_df = compute_url_risk_summary(url_df, sim_matrix)

    if risk_df.empty:
        st.info("No data to display.")
    else:
        def color_risk_row(row):
            colors = {"High": "#ffcccc", "Medium": "#fff3cc", "Low": "#e8f5e9", "Minimal": ""}
            bg = colors.get(row["risk_level"], "")
            return [f"background-color: {bg}"] * len(row)

        styled = risk_df.style.apply(color_risk_row, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Save for combined risk page
        st.session_state.url_risk_df = risk_df

        csv = risk_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download URL risk summary as CSV",
            data=csv,
            file_name="url_risk_summary.csv",
            mime="text/csv",
        )

# ─── Navigation hint ─────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "**Next steps →** Check **GSC Cannibalization** for keyword overlap analysis, "
    "or go to **AI Recommendations** to get specific remediation advice."
)
