"""
Page 4 — Combined Risk Dashboard
Cross-references semantic similarity (Layer 1) with GSC cannibalization (Layer 2)
to produce a prioritized, combined risk score per URL pair.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings_handler import get_pairs_above_threshold, THRESHOLD_MEDIUM

st.set_page_config(
    page_title="Combined Risk — Duplicate Content Detector",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Combined Risk Dashboard")
st.caption(
    "URL pairs flagged by BOTH semantic similarity (Screaming Frog) "
    "AND keyword cannibalization (GSC) represent the highest priority for action."
)

# ─── Check what data is available ────────────────────────────────────────────

has_embeddings = st.session_state.get("sf_loaded") and st.session_state.get("sim_matrix") is not None
has_gsc = st.session_state.get("gsc_loaded") and st.session_state.get("cannibalization_findings") is not None

if not has_embeddings and not has_gsc:
    st.warning(
        "No analysis data found. Run **Semantic Similarity** and/or **GSC Cannibalization** first."
    )
    st.stop()

if not has_embeddings:
    st.info("Semantic similarity data not available. Showing GSC cannibalization only.")
if not has_gsc:
    st.info("GSC cannibalization data not available. Showing semantic similarity only.")

# ─── Build combined dataset ──────────────────────────────────────────────────

# --- Semantic pairs ---
semantic_pairs_df = pd.DataFrame()
if has_embeddings:
    url_df = st.session_state.url_df
    sim_matrix = st.session_state.sim_matrix
    filter_urls = st.session_state.get("filter_urls", [])
    if filter_urls:
        mask = url_df["url"].isin(filter_urls)
        url_df = url_df[mask].reset_index(drop=True)
        sim_matrix = sim_matrix[np.ix_(mask.values, mask.values)]

    semantic_pairs_df = get_pairs_above_threshold(
        url_df, sim_matrix, threshold=THRESHOLD_MEDIUM, max_pairs=10000
    )

# --- GSC cannibalization data ---
cannibal_findings = []
url_cannibal_summary = {}
if has_gsc:
    cannibal_findings = st.session_state.cannibalization_findings
    url_cannibal_summary = st.session_state.get("url_cannibal_summary", {})

    # Build URL-pair level cannibalization lookup
    cannibal_pair_lookup: dict[tuple, dict] = {}
    for finding in cannibal_findings:
        shared_queries = [finding["query"]]
        urls = finding["urls"]
        for i in range(len(urls)):
            for j in range(i + 1, len(urls)):
                pair_key = tuple(sorted([urls[i], urls[j]]))
                if pair_key not in cannibal_pair_lookup:
                    cannibal_pair_lookup[pair_key] = {
                        "shared_queries": [],
                        "total_clicks": 0,
                        "total_impressions": 0,
                        "max_impact": 0.0,
                    }
                cannibal_pair_lookup[pair_key]["shared_queries"].append(finding["query"])
                cannibal_pair_lookup[pair_key]["total_clicks"] += finding["total_clicks"]
                cannibal_pair_lookup[pair_key]["total_impressions"] += finding["total_impressions"]
                cannibal_pair_lookup[pair_key]["max_impact"] = max(
                    cannibal_pair_lookup[pair_key]["max_impact"],
                    finding["impact_score"],
                )


# ─── Build combined DataFrame ─────────────────────────────────────────────────

def build_combined_df() -> pd.DataFrame:
    rows = []

    if has_embeddings and not semantic_pairs_df.empty:
        for _, pair in semantic_pairs_df.iterrows():
            pair_key = tuple(sorted([pair["url_a"], pair["url_b"]]))
            gsc_info = cannibal_pair_lookup.get(pair_key, {}) if has_gsc else {}

            semantic_score = float(pair["similarity"])
            gsc_impact = gsc_info.get("max_impact", 0.0)

            # Normalize GSC impact to 0–1 range
            max_impact = max((f["impact_score"] for f in cannibal_findings), default=1.0) if cannibal_findings else 1.0
            gsc_norm = min(gsc_impact / max_impact, 1.0) if max_impact > 0 else 0.0

            combined_score = round(semantic_score * 0.5 + gsc_norm * 0.5, 4)

            # Determine alert level
            in_gsc = bool(gsc_info)
            if semantic_score >= 0.85 and in_gsc:
                alert = "Critical"
            elif semantic_score >= 0.85:
                alert = "High — Semantic"
            elif in_gsc and gsc_norm >= 0.3:
                alert = "High — GSC"
            elif semantic_score >= 0.60 and in_gsc:
                alert = "Medium"
            else:
                alert = "Monitor"

            rows.append({
                "url_a": pair["url_a"],
                "url_b": pair["url_b"],
                "semantic_similarity": semantic_score,
                "gsc_impact_score": round(gsc_impact, 2),
                "shared_queries": len(gsc_info.get("shared_queries", [])),
                "combined_score": combined_score,
                "alert_level": alert,
                "gsc_clicks": gsc_info.get("total_clicks", 0),
                "gsc_impressions": gsc_info.get("total_impressions", 0),
            })

    elif has_gsc and not has_embeddings:
        # GSC only mode
        for finding in cannibal_findings:
            urls = finding["urls"]
            for i in range(len(urls)):
                for j in range(i + 1, len(urls)):
                    rows.append({
                        "url_a": urls[i],
                        "url_b": urls[j],
                        "semantic_similarity": None,
                        "gsc_impact_score": finding["impact_score"],
                        "shared_queries": 1,
                        "combined_score": finding["impact_score"],
                        "alert_level": "High — GSC",
                        "gsc_clicks": finding["total_clicks"],
                        "gsc_impressions": finding["total_impressions"],
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return df


with st.spinner("Building combined risk dataset..."):
    combined_df = build_combined_df()

# ─── Summary ─────────────────────────────────────────────────────────────────

if combined_df.empty:
    st.success("No combined risk findings. All pages appear sufficiently distinct.")
    st.stop()

critical = combined_df[combined_df["alert_level"] == "Critical"]
high = combined_df[combined_df["alert_level"].str.startswith("High")]
medium = combined_df[combined_df["alert_level"] == "Medium"]
monitor = combined_df[combined_df["alert_level"] == "Monitor"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Flagged Pairs", len(combined_df))
col2.metric("🔴 Critical", len(critical))
col3.metric("🟠 High", len(high))
col4.metric("🟡 Medium", len(medium))
col5.metric("🔵 Monitor", len(monitor))

# Save for AI page
st.session_state.combined_df = combined_df

st.divider()

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 Critical & High",
    "📋 All Findings",
    "📈 Risk Scatter Plot",
    "🏆 URL Leaderboard",
])

ALERT_COLORS = {
    "Critical": "background-color:#ff4d4d;color:white",
    "High — Semantic": "background-color:#ff9966",
    "High — GSC": "background-color:#ffb366",
    "Medium": "background-color:#ffff99",
    "Monitor": "",
}


def color_alert(val):
    return ALERT_COLORS.get(val, "")


# ── Tab 1: Critical & High ────────────────────────────────────────────────────

with tab1:
    priority_df = combined_df[combined_df["alert_level"].isin(
        ["Critical", "High — Semantic", "High — GSC"]
    )].copy()

    if priority_df.empty:
        st.success("No critical or high-risk pairs found.")
    else:
        display_cols = [
            "url_a", "url_b", "semantic_similarity", "gsc_impact_score",
            "shared_queries", "gsc_clicks", "combined_score", "alert_level",
        ]
        available_cols = [c for c in display_cols if c in priority_df.columns]
        styled = priority_df[available_cols].style.applymap(color_alert, subset=["alert_level"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if not priority_df.empty:
            # Expandable critical pairs
            if len(critical) > 0:
                st.subheader("Critical Pairs Detail")
                for _, row in critical.head(10).iterrows():
                    with st.expander(f"{row['url_a'][:60]}... ↔ {row['url_b'][:60]}..."):
                        c1, c2 = st.columns(2)
                        c1.metric("Semantic Similarity", f"{row['semantic_similarity']:.1%}")
                        c2.metric("Shared GSC Queries", row["shared_queries"])
                        c1.metric("Combined Score", f"{row['combined_score']:.2f}")
                        c2.metric("GSC Clicks at Risk", row["gsc_clicks"])
                        st.markdown(f"**URL A:** `{row['url_a']}`")
                        st.markdown(f"**URL B:** `{row['url_b']}`")

        csv = priority_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download critical/high findings",
            data=csv,
            file_name="critical_high_pairs.csv",
            mime="text/csv",
        )

# ── Tab 2: All Findings ───────────────────────────────────────────────────────

with tab2:
    st.subheader(f"All Flagged Pairs ({len(combined_df):,})")

    alert_filter = st.multiselect(
        "Filter by alert level",
        options=combined_df["alert_level"].unique().tolist(),
        default=combined_df["alert_level"].unique().tolist(),
        key="alert_filter",
    )

    filtered = combined_df[combined_df["alert_level"].isin(alert_filter)] if alert_filter else combined_df
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(filtered)} of {len(combined_df)} pairs")

    csv = filtered.to_csv(index=False)
    st.download_button(
        "⬇️ Download filtered results",
        data=csv,
        file_name="combined_risk_findings.csv",
        mime="text/csv",
    )

# ── Tab 3: Scatter Plot ───────────────────────────────────────────────────────

with tab3:
    if has_embeddings and has_gsc:
        try:
            import plotly.express as px

            plot_df = combined_df.dropna(subset=["semantic_similarity"]).copy()
            plot_df["url_label"] = (
                plot_df["url_a"].str.replace("https://", "").str[:40]
                + " ↔ "
                + plot_df["url_b"].str.replace("https://", "").str[:40]
            )

            fig = px.scatter(
                plot_df,
                x="semantic_similarity",
                y="gsc_impact_score",
                color="alert_level",
                size="combined_score",
                hover_name="url_label",
                hover_data=["shared_queries", "gsc_clicks"],
                color_discrete_map={
                    "Critical": "#ff0000",
                    "High — Semantic": "#ff7700",
                    "High — GSC": "#ff9900",
                    "Medium": "#ffcc00",
                    "Monitor": "#99bbff",
                },
                labels={
                    "semantic_similarity": "Semantic Similarity (SF Embeddings)",
                    "gsc_impact_score": "GSC Cannibalization Impact Score",
                    "alert_level": "Alert Level",
                },
                title="Combined Risk: Semantic Similarity vs. GSC Impact",
            )

            # Quadrant lines
            fig.add_vline(x=0.85, line_dash="dash", line_color="gray", annotation_text="High similarity")
            fig.add_hline(y=plot_df["gsc_impact_score"].quantile(0.80), line_dash="dash", line_color="gray", annotation_text="High GSC impact")

            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Top-right quadrant (high similarity + high GSC impact) = Critical risk. "
                "Bubble size = combined score."
            )
        except ImportError:
            st.warning("Install `plotly` to see the scatter plot.")
    else:
        st.info("This chart requires both SF embeddings and GSC data.")

# ── Tab 4: URL Leaderboard ────────────────────────────────────────────────────

with tab4:
    st.subheader("URLs with Most Duplicate Risk")
    st.caption("Aggregated per-URL stats combining semantic and cannibalization signals.")

    url_agg: dict[str, dict] = {}

    for _, row in combined_df.iterrows():
        for url in [row["url_a"], row["url_b"]]:
            if url not in url_agg:
                url_agg[url] = {
                    "url": url,
                    "flagged_pairs": 0,
                    "max_similarity": 0.0,
                    "total_combined_score": 0.0,
                    "critical_count": 0,
                }
            url_agg[url]["flagged_pairs"] += 1
            url_agg[url]["total_combined_score"] += row["combined_score"]
            if row.get("semantic_similarity"):
                url_agg[url]["max_similarity"] = max(
                    url_agg[url]["max_similarity"], row["semantic_similarity"]
                )
            if row["alert_level"] == "Critical":
                url_agg[url]["critical_count"] += 1

    leaderboard_df = pd.DataFrame(list(url_agg.values()))
    leaderboard_df = leaderboard_df.sort_values("total_combined_score", ascending=False).reset_index(drop=True)

    # Add GSC cannibalization stats
    if url_cannibal_summary:
        leaderboard_df["gsc_queries_competing"] = leaderboard_df["url"].map(
            lambda u: url_cannibal_summary.get(u, {}).get("queries_competing", 0)
        )

    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    csv = leaderboard_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download URL leaderboard",
        data=csv,
        file_name="url_risk_leaderboard.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    "**Next →** Go to **AI Recommendations** to get specific remediation advice "
    "for your highest-risk URL pairs."
)
