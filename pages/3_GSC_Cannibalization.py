"""
Page 3 — GSC Keyword Cannibalization
Identifies queries where multiple URLs are competing for clicks and impressions.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gsc_handler import detect_cannibalization, get_url_cannibalization_summary

st.set_page_config(
    page_title="GSC Cannibalization — Duplicate Content Detector",
    page_icon="📊",
    layout="wide",
)

st.title("📊 GSC Keyword Cannibalization")
st.caption(
    "Finds queries where multiple URLs are competing for clicks and impressions — "
    "a strong signal that content is cannibalizing itself."
)

# ─── Guard: require GSC data ─────────────────────────────────────────────────

if not st.session_state.get("gsc_loaded") or st.session_state.get("gsc_data") is None:
    st.warning("No GSC data loaded. Please go to **Data Input** and connect to Google Search Console.")
    st.stop()

gsc_df: pd.DataFrame = st.session_state.gsc_data

# Apply URL filter if set
filter_urls = st.session_state.get("filter_urls", [])
if filter_urls:
    filtered = gsc_df[gsc_df["page"].isin(filter_urls)]
    if len(filtered) > 0:
        gsc_df = filtered
        st.info(f"URL filter applied: {gsc_df['page'].nunique()} URLs, {len(gsc_df):,} rows")

# ─── Controls ────────────────────────────────────────────────────────────────

st.sidebar.header("Cannibalization Settings")

min_impressions = st.sidebar.number_input(
    "Min Impressions per URL",
    min_value=0,
    max_value=10000,
    value=10,
    step=5,
    help="Only consider URL/query pairs with at least this many impressions.",
)

min_clicks = st.sidebar.number_input(
    "Min Clicks per URL",
    min_value=0,
    max_value=1000,
    value=0,
    step=1,
    help="Only consider URL/query pairs with at least this many clicks.",
)

show_top_pct = st.sidebar.slider(
    "High Impact cutoff (top %)",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    help="The top X% of findings by impact score are labelled 'High Impact'.",
)

# ─── Run analysis ────────────────────────────────────────────────────────────

with st.spinner("Detecting cannibalization..."):
    gsc_rows = gsc_df.to_dict("records")
    findings = detect_cannibalization(
        gsc_rows,
        min_impressions=int(min_impressions),
        min_clicks=int(min_clicks),
    )

# Store for combined risk page
st.session_state.cannibalization_findings = findings
st.session_state.url_cannibal_summary = get_url_cannibalization_summary(findings)

# ─── Summary metrics ─────────────────────────────────────────────────────────

total_affected_urls = set()
for f in findings:
    for u in f["urls"]:
        total_affected_urls.add(u)

total_click_impact = sum(f["total_clicks"] for f in findings)
top_n = max(1, int(len(findings) * show_top_pct / 100))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cannibalized Queries", len(findings))
col2.metric("Affected URLs", len(total_affected_urls))
col3.metric("Total Clicks at Risk", f"{total_click_impact:,}")
col4.metric(f"High Impact (top {show_top_pct}%)", top_n)

st.divider()

if not findings:
    st.success(
        "No cannibalization detected with current settings. "
        "Try lowering the minimum impressions threshold."
    )
    st.stop()

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "⚡ High Impact",
    "📋 All Queries",
    "🔗 URL Summary",
])

# ── Build display DataFrame ───────────────────────────────────────────────────

def findings_to_df(findings_list: list[dict]) -> pd.DataFrame:
    rows = []
    for f in findings_list:
        competing_str = "\n".join(f["competing_urls"][:5])
        if len(f["competing_urls"]) > 5:
            competing_str += f"\n(+{len(f['competing_urls']) - 5} more)"
        rows.append({
            "Query": f["query"],
            "# URLs": f["num_competing_urls"],
            "Dominant URL": f["dominant_url"],
            "Dominant Clicks": f["dominant_clicks"],
            "Competing URLs": ", ".join(f["competing_urls"][:3]) + (
                f" (+{len(f['competing_urls']) - 3} more)" if len(f["competing_urls"]) > 3 else ""
            ),
            "Total Clicks": f["total_clicks"],
            "Total Impressions": f["total_impressions"],
            "Impact Score": f["impact_score"],
        })
    return pd.DataFrame(rows)


# ── Tab 1: High Impact ────────────────────────────────────────────────────────

with tab1:
    st.subheader(f"Top {show_top_pct}% — Highest Impact Cannibalization")
    high_impact = findings[:top_n]

    if not high_impact:
        st.info("No high-impact findings.")
    else:
        hi_df = findings_to_df(high_impact)
        st.dataframe(hi_df, use_container_width=True, hide_index=True)

        # Detailed expandable view
        st.subheader("Detailed View")
        for i, finding in enumerate(high_impact[:20], start=1):
            with st.expander(
                f"**{finding['query']}** — {finding['num_competing_urls']} URLs competing "
                f"({finding['total_clicks']} clicks, Impact: {finding['impact_score']:.1f})",
                expanded=(i <= 3),
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Dominant URL** (most clicks):")
                    st.code(finding["dominant_url"])
                    st.markdown(
                        f"Clicks: **{finding['dominant_clicks']}** | "
                        f"Impressions: **{finding['dominant_impressions']}**"
                    )
                with col_b:
                    st.markdown(f"**Competing URLs** ({len(finding['competing_urls'])}):")
                    for url in finding["competing_urls"]:
                        url_data = [
                            r for r in gsc_rows
                            if r["query"] == finding["query"] and r["page"] == url
                        ]
                        clicks = url_data[0]["clicks"] if url_data else 0
                        impressions = url_data[0]["impressions"] if url_data else 0
                        st.markdown(f"- `{url}` — {clicks} clicks / {impressions} impr.")

        csv = hi_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download high impact findings as CSV",
            data=csv,
            file_name="cannibalization_high_impact.csv",
            mime="text/csv",
        )

# ── Tab 2: All Queries ────────────────────────────────────────────────────────

with tab2:
    st.subheader(f"All Cannibalized Queries ({len(findings):,})")

    # Search filter
    search_term = st.text_input("Filter queries", placeholder="Type to search...", key="cannibal_search")

    filtered_findings = [
        f for f in findings
        if not search_term or search_term.lower() in f["query"].lower()
    ]

    all_df = findings_to_df(filtered_findings)
    st.dataframe(all_df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(filtered_findings)} of {len(findings)} findings")

    csv = all_df.to_csv(index=False)
    st.download_button(
        "⬇️ Download all findings as CSV",
        data=csv,
        file_name="cannibalization_all.csv",
        mime="text/csv",
    )

# ── Tab 3: URL Summary ────────────────────────────────────────────────────────

with tab3:
    st.subheader("URL-Level Cannibalization Summary")
    st.caption("How many queries is each URL involved in as either the dominant or a competing page?")

    url_summary = st.session_state.url_cannibal_summary

    if not url_summary:
        st.info("No URL summary available.")
    else:
        summary_rows = []
        for url, data in url_summary.items():
            summary_rows.append({
                "URL": url,
                "Queries Competing": data["queries_competing"],
                "As Dominant URL": data["as_dominant"],
                "As Competing URL": data["as_competing"],
                "Total Impact Score": round(data["total_impact"], 1),
            })

        summary_df = pd.DataFrame(summary_rows).sort_values("Total Impact Score", ascending=False)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        csv = summary_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download URL summary as CSV",
            data=csv,
            file_name="url_cannibalization_summary.csv",
            mime="text/csv",
        )

# ─── Chart: Top 20 queries by impact ─────────────────────────────────────────

st.divider()
st.subheader("Top 20 Queries by Impact Score")

try:
    import plotly.express as px

    chart_data = pd.DataFrame([
        {"query": f["query"][:50], "impact": f["impact_score"], "urls": f["num_competing_urls"]}
        for f in findings[:20]
    ])

    fig = px.bar(
        chart_data,
        x="impact",
        y="query",
        orientation="h",
        color="urls",
        color_continuous_scale="Reds",
        labels={"impact": "Impact Score", "query": "Query", "urls": "# Competing URLs"},
        title="Top 20 Cannibalized Queries",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig, use_container_width=True)
except ImportError:
    st.info("Install `plotly` to see the impact chart.")

st.divider()
st.markdown(
    "**Next →** View the **Combined Risk Dashboard** to see which URLs are flagged by both "
    "semantic similarity AND keyword cannibalization."
)
