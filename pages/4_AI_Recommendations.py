"""
Page 5 — AI Recommendations
Uses OpenRouter to generate specific remediation advice for flagged URL pairs/groups.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings_handler import (  # noqa: E402
    THRESHOLD_HIGH,
    get_pairs_above_threshold,
)
from utils.openrouter_handler import stream_recommendation  # noqa: E402

st.set_page_config(
    page_title="AI Recommendations — Duplicate Content Detector",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Recommendations")
st.caption(
    "Get specific, data-driven remediation advice for your duplicate content issues "
    "powered by your choice of AI model."
)

# ─── Check OpenRouter API key ───────────────────────────────────────────

try:
    has_api_key = bool(st.secrets.get("OPENROUTER_API_KEY", ""))
except Exception:
    has_api_key = False

if not has_api_key:
    st.error(
        "**OpenRouter API key not configured.** "
        "Add `OPENROUTER_API_KEY` to your `.streamlit/secrets.toml` to use AI recommendations."
    )
    st.stop()

# ─── Check for analysis data ─────────────────────────────────────────────────

has_combined = "combined_df" in st.session_state and not st.session_state.combined_df.empty
has_semantic = "sim_matrix" in st.session_state and st.session_state.get(
    "sf_loaded")
has_gsc = "cannibalization_findings" in st.session_state

if not has_combined and not has_semantic and not has_gsc:
    st.warning(
        "No analysis data found. Run the **Semantic Similarity**, **GSC Cannibalization**, "
        "or **Combined Risk** pages first."
    )
    st.stop()

# ─── Model selection (reads from session state set on setup page) ────────────

from utils.openrouter_handler import AVAILABLE_MODELS  # noqa: E402

model_options = {m["label"]: m["id"] for m in AVAILABLE_MODELS}
model_labels = list(model_options.keys())

# Default to whatever was chosen on the setup page
saved_label = st.session_state.get("selected_model_label", model_labels[0])
default_idx = (
    model_labels.index(saved_label) if saved_label in model_labels else 0
)

with st.container(border=True):
    mcol, pcol = st.columns([3, 1])
    with mcol:
        selected_label = st.selectbox(
            "🤖 AI Model",
            options=model_labels,
            index=default_idx,
            key="ai_page_model",
        )
    with pcol:
        max_pairs_to_analyze = st.number_input(
            "Max pairs",
            min_value=1,
            max_value=10,
            value=3,
            help="URL pairs per AI call",
        )

model_id = model_options[selected_label]
# Sync back to session state so setup page stays in sync
st.session_state.selected_model_label = selected_label
st.session_state.selected_model_id = model_id

# ─── URL pair selection ─────────────────────────────────────────────────

st.subheader("Select URL Pairs to Analyze")

pair_source = st.radio(
    "Select pairs from:",
    options=(
        ["Combined Risk Dashboard"]
        + (["Semantic Similarity Pairs"] if has_semantic else [])
        + (["GSC Cannibalization"] if has_gsc else [])
        + ["Manual Entry"]
    ),
    horizontal=True,
    key="pair_source",
)

selected_pairs: list[dict] = []

if pair_source == "Combined Risk Dashboard" and has_combined:
    combined_df: pd.DataFrame = st.session_state.combined_df

    alert_filter = st.multiselect(
        "Filter by alert level",
        options=combined_df["alert_level"].unique().tolist(),
        default=["Critical", "High — Semantic", "High — GSC"],
        key="ai_alert_filter",
    )
    filtered_df = combined_df[combined_df["alert_level"].isin(
        alert_filter)].head(50)

    if filtered_df.empty:
        st.info("No pairs match your filter.")
    else:
        # Let user pick pairs
        pair_labels = [
            f"{row['url_a'][:50]}... ↔ {row['url_b'][:50]}... "
            f"[sim={row.get('semantic_similarity', 0) or 0:.0%}, "
            f"alert={row['alert_level']}]"
            for _, row in filtered_df.iterrows()
        ]

        selected_indices = st.multiselect(
            f"Choose up to {int(max_pairs_to_analyze)} pairs to analyze",
            options=range(len(pair_labels)),
            format_func=lambda i: pair_labels[i],
            max_selections=int(max_pairs_to_analyze),
            key="selected_pair_indices",
        )

        for idx in selected_indices:
            row = filtered_df.iloc[idx]
            # Find shared queries from GSC if available
            shared_queries = []
            if has_gsc:
                cannibal_findings = st.session_state.cannibalization_findings
                for f in cannibal_findings:
                    pair_key = tuple(sorted([row["url_a"], row["url_b"]]))
                    if tuple(sorted(
                            [f["urls"][0], f["urls"][-1] if len(f["urls"]) > 1 else f["urls"][0]])) == pair_key:
                        shared_queries.append(f["query"])

            # Get per-URL click/impression data from GSC
            gsc_data = st.session_state.get("gsc_data")
            clicks_a = clicks_b = impressions_a = impressions_b = None
            if gsc_data is not None:
                url_a_data = gsc_data[gsc_data["page"] == row["url_a"]]
                url_b_data = gsc_data[gsc_data["page"] == row["url_b"]]
                if not url_a_data.empty:
                    clicks_a = int(url_a_data["clicks"].sum())
                    impressions_a = int(url_a_data["impressions"].sum())
                if not url_b_data.empty:
                    clicks_b = int(url_b_data["clicks"].sum())
                    impressions_b = int(url_b_data["impressions"].sum())

            selected_pairs.append({
                "url_a": row["url_a"],
                "url_b": row["url_b"],
                "similarity": row.get("semantic_similarity"),
                "gsc_shared_queries": shared_queries[:10],
                "clicks_a": clicks_a,
                "clicks_b": clicks_b,
                "impressions_a": impressions_a,
                "impressions_b": impressions_b,
            })

elif pair_source == "Semantic Similarity Pairs" and has_semantic:
    from utils.embeddings_handler import get_pairs_above_threshold, THRESHOLD_HIGH

    url_df = st.session_state.url_df
    sim_matrix = st.session_state.sim_matrix
    pairs_df = get_pairs_above_threshold(
        url_df, sim_matrix, threshold=THRESHOLD_HIGH, max_pairs=50)

    if pairs_df.empty:
        st.info(
            "No high-similarity pairs found. Lower the threshold on the Semantic Similarity page.")
    else:
        pair_labels = [
            f"{row['url_a'][:50]}... ↔ {row['url_b'][:50]}... [sim={row['similarity']:.0%}]"
            for _, row in pairs_df.iterrows()
        ]
        selected_indices = st.multiselect(
            f"Choose up to {int(max_pairs_to_analyze)} pairs",
            options=range(len(pair_labels)),
            format_func=lambda i: pair_labels[i],
            max_selections=int(max_pairs_to_analyze),
            key="sim_pair_indices",
        )
        for idx in selected_indices:
            row = pairs_df.iloc[idx]
            selected_pairs.append({
                "url_a": row["url_a"],
                "url_b": row["url_b"],
                "similarity": row["similarity"],
            })

elif pair_source == "GSC Cannibalization" and has_gsc:
    cannibal_findings = st.session_state.cannibalization_findings[:50]
    finding_labels = [
        f"Query: '{
            f['query']}' — {
            f['num_competing_urls']} URLs — Impact: {
            f['impact_score']:.1f}"
        for f in cannibal_findings
    ]
    selected_finding_indices = st.multiselect(
        f"Choose up to {int(max_pairs_to_analyze)} queries to analyze",
        options=range(len(finding_labels)),
        format_func=lambda i: finding_labels[i],
        max_selections=int(max_pairs_to_analyze),
        key="cannibal_indices",
    )
    for idx in selected_finding_indices:
        f = cannibal_findings[idx]
        # Convert to pairs (dominant vs first competing)
        if len(f["urls"]) >= 2:
            selected_pairs.append({
                "url_a": f["urls"][0],
                "url_b": f["urls"][1],
                "gsc_shared_queries": [f["query"]],
                "gsc_impact_score": f["impact_score"],
                "clicks_a": f["dominant_clicks"],
                "clicks_b": None,
                "impressions_a": f["dominant_impressions"],
                "impressions_b": None,
            })

elif pair_source == "Manual Entry":
    st.markdown("Enter up to 2 URLs to compare manually:")
    col1, col2 = st.columns(2)
    with col1:
        url_a = st.text_input(
            "URL A",
            placeholder="https://example.com/page-1",
            key="manual_url_a")
    with col2:
        url_b = st.text_input(
            "URL B",
            placeholder="https://example.com/page-2",
            key="manual_url_b")

    if url_a and url_b:
        selected_pairs = [{"url_a": url_a.strip(), "url_b": url_b.strip()}]

# ─── Additional context ─────────────────────────────────────────────────

if selected_pairs:
    additional_context = st.text_area(
        "Additional context for AI (optional)",
        placeholder="e.g., These pages target the same audience segment, "
                    "or they were created at different times for different campaigns...",
        height=80,
        key="ai_additional_context",
    )

# ─── Run analysis ───────────────────────────────────────────────────────

if selected_pairs and st.button(
        "🤖 Generate AI Recommendations", type="primary", key="run_ai"):
    st.divider()
    st.subheader("AI Analysis")
    st.caption(
        f"Model: **{selected_label}**  |  Pairs: **{len(selected_pairs)}**")

    output_placeholder = st.empty()
    full_output = ""

    with st.spinner("Generating recommendations..."):
        for chunk in stream_recommendation(
            model_id=model_id,
            url_pairs=selected_pairs,
            additional_context=additional_context if selected_pairs else "",
        ):
            full_output += chunk
            output_placeholder.markdown(full_output + "▌")

    output_placeholder.markdown(full_output)

    # Copy button
    st.text_area(
        "Raw output (for copying)",
        value=full_output,
        height=200,
        key="ai_raw_output",
    )
    st.download_button(
        "⬇️ Download recommendation as Markdown",
        data=full_output,
        file_name="ai_recommendations.md",
        mime="text/markdown",
    )

elif not selected_pairs:
    st.info("Select URL pairs above to enable AI analysis.")

# ─── Instructions ───────────────────────────────────────────────────────

with st.expander("ℹ️ How to interpret AI recommendations"):
    st.markdown("""
    The AI will recommend one of these actions per URL pair:

    | Action | When to Use |
    |--------|-------------|
    | **Consolidate** | Two pages cover identical content — merge into one, stronger page |
    | **Canonicalize** | Keep both pages but tell Google which is the "master" |
    | **Redirect (301)** | Permanently redirect the weaker page to the stronger one |
    | **Differentiate** | Both pages serve a purpose — update them to be clearly distinct |
    | **No action** | Pages are similar but serve different intents — leave as-is |

    **Priority levels:**
    - 🔴 **Critical** — Act immediately (cannibalizing high-traffic queries + high semantic overlap)
    - 🟠 **High** — Address in next sprint
    - 🟡 **Medium** — Address in next content audit cycle
    - 🔵 **Low** — Monitor over time
    """)
