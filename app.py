"""
Duplicate Content Detector — Setup & Configuration
This IS the first page users see. All data sources are configured here.
"""

import os
import sys
import calendar
from datetime import date, timedelta

import pandas as pd
import streamlit as st

# Streamlit adds the project root to sys.path automatically.
# The explicit insert below keeps local dev working regardless.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.embeddings_handler import parse_sf_embeddings  # noqa: E402
from utils.gsc_handler import (  # noqa: E402
    fetch_gsc_data,
    fetch_properties,
    fetch_sitemap_urls,
    init_from_stored_refresh_token,
    is_authenticated,
    logout,
)
from utils.openrouter_handler import AVAILABLE_MODELS  # noqa: E402
from utils.url_exclusions import (  # noqa: E402
    DEFAULT_EXCLUDE_PATTERNS,
    apply_exclusions,
    patterns_from_text,
    patterns_to_text,
)


st.set_page_config(
    page_title="Duplicate Content Detector",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state defaults ─────────────────────────────────────────────

defaults = {
    "sf_loaded": False, "gsc_loaded": False,
    "url_df": None, "embeddings_matrix": None, "sim_matrix": None,
    "sim_matrix_url_count": 0, "gsc_data": None, "gsc_properties": None,
    "selected_property": None, "gsc_token_data": None,
    "cannibalization_findings": None, "url_cannibal_summary": {},
    "combined_df": None, "clusters": None, "cluster_df": None,
    "url_risk_df": None,
    "exclude_patterns": list(DEFAULT_EXCLUDE_PATTERNS),
    "selected_model_id": AVAILABLE_MODELS[0]["id"],
    "selected_model_label": AVAILABLE_MODELS[0]["label"],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Auto-authenticate from stored refresh token ────────────────────────

if not is_authenticated():
    init_from_stored_refresh_token()

# ─── Header ─────────────────────────────────────────────────────────────

st.title("🔎 Duplicate Content Detector")
st.caption(
    "Configure your data sources below, then use the sidebar to run each analysis."
)

# ─── Status bar ─────────────────────────────────────────────────────────

sf_ok = st.session_state.sf_loaded and st.session_state.url_df is not None
gsc_ok = is_authenticated() and st.session_state.gsc_loaded
try:
    # Check top-level key first, then fallback to [gsc] section (common paste mistake)
    _or_key = (
        st.secrets.get("OPENROUTER_API_KEY", "")
        or st.secrets.get("gsc", {}).get("OPENROUTER_API_KEY", "")
    )
    ai_ok = bool(_or_key.strip()) and "YOUR_OPENROUTER" not in _or_key
except Exception:
    ai_ok = False

c1, c2, c3 = st.columns(3)
c1.metric(
    "Screaming Frog",
    f"{len(st.session_state.url_df):,} URLs" if sf_ok else "Not loaded",
    delta="Ready" if sf_ok else None,
    delta_color="normal",
)
c2.metric(
    "Google Search Console",
    f"{len(st.session_state.gsc_data):,} rows" if gsc_ok else "Not connected",
    delta="Ready" if gsc_ok else None,
    delta_color="normal",
)
c3.metric(
    "AI Model",
    st.session_state.selected_model_label.split(
        " (")[0] if ai_ok else "API key missing",
    delta="Ready" if ai_ok else None,
    delta_color="normal",
)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Google Search Console
# ═══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    hcol, tcol = st.columns([5, 1])
    with hcol:
        st.subheader("1️⃣  Google Search Console")
        st.caption(
            "Detects keyword cannibalization — queries where multiple pages split your clicks")
    with tcol:
        gsc_on = st.toggle("Enable", value=True, key="gsc_toggle")

    if gsc_on:
        if is_authenticated():
            st.success("✅  Connected to Google Search Console")

            dcol, _ = st.columns([1, 4])
            with dcol:
                if st.button("Disconnect", key="gsc_logout"):
                    logout()
                    st.rerun()

            # Property selector
            if not st.session_state.gsc_properties:
                with st.spinner("Fetching properties…"):
                    st.session_state.gsc_properties = fetch_properties()

            props = st.session_state.gsc_properties or []
            if not props:
                st.warning("No GSC properties found for this account.")
            else:
                prop_idx = 0
                if st.session_state.selected_property in props:
                    prop_idx = props.index(st.session_state.selected_property)

                # Property selector
                sel = st.selectbox(
                    "GSC Property", props, index=prop_idx, key="prop_sel")
                st.session_state.selected_property = sel

                # ── Date range ───────────────────────────────────────────
                _PRESETS = [
                    "Last 30 days", "Last 60 days", "Last 90 days",
                    "Last 180 days", "Last 360 days",
                    "Calendar month", "Custom range",
                ]
                _PRESET_DAYS = {
                    "Last 30 days": 30, "Last 60 days": 60,
                    "Last 90 days": 90, "Last 180 days": 180,
                    "Last 360 days": 360,
                }
                _MONTH_NAMES = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November",
                    "December",
                ]

                date_mode = st.selectbox(
                    "Date range",
                    options=_PRESETS,
                    index=2,  # default: Last 90 days
                    key="date_mode",
                )

                _today = date.today()
                _max_d = _today - timedelta(days=2)  # GSC lags ~2 days

                if date_mode in _PRESET_DAYS:
                    start_d = _today - timedelta(days=_PRESET_DAYS[date_mode])
                    end_d = _max_d
                    st.caption(
                        f"{start_d.strftime('%b %d, %Y')}"
                        f" → {end_d.strftime('%b %d, %Y')}"
                    )

                elif date_mode == "Calendar month":
                    _prev = (_today.replace(day=1) - timedelta(days=1))
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        month_name = st.selectbox(
                            "Month", _MONTH_NAMES,
                            index=_prev.month - 1,
                            key="cal_month",
                        )
                    with mc2:
                        _years = list(range(_today.year - 3, _today.year + 1))
                        cal_year = st.selectbox(
                            "Year", _years,
                            index=_years.index(_prev.year),
                            key="cal_year",
                        )
                    _mnum = _MONTH_NAMES.index(month_name) + 1
                    _, _last = calendar.monthrange(cal_year, _mnum)
                    start_d = date(cal_year, _mnum, 1)
                    end_d = min(date(cal_year, _mnum, _last), _max_d)
                    st.caption(
                        f"{start_d.strftime('%b %d, %Y')}"
                        f" → {end_d.strftime('%b %d, %Y')}"
                    )

                else:  # Custom range
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        start_d = st.date_input(
                            "Start date",
                            value=_today - timedelta(days=90),
                            max_value=_max_d,
                            key="gsc_start",
                        )
                    with cc2:
                        end_d = st.date_input(
                            "End date",
                            value=_max_d,
                            max_value=_max_d,
                            key="gsc_end",
                        )

                if st.button(
                    "⬇️  Fetch GSC Data",
                    type="primary",
                        key="fetch_gsc"):
                    with st.spinner("Fetching Search Analytics data…"):
                        rows = fetch_gsc_data(
                            sel,
                            start_d.strftime("%Y-%m-%d"),
                            end_d.strftime("%Y-%m-%d"),
                        )
                    if rows:
                        df = pd.DataFrame(rows)
                        st.session_state.gsc_data = df
                        st.session_state.selected_property = sel
                        st.session_state.gsc_loaded = True
                        st.success(
                            f"Loaded {
                                len(df):,} rows — {
                                df['query'].nunique():,} queries across {
                                df['page'].nunique():,} URLs")
                        st.rerun()
                    else:
                        st.error(
                            "No data returned. Check the property and date range.")

                if st.session_state.gsc_loaded and st.session_state.gsc_data is not None:
                    gdf = st.session_state.gsc_data
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Rows", f"{len(gdf):,}")
                    m2.metric("Queries", f"{gdf['query'].nunique():,}")
                    m3.metric("URLs", f"{gdf['page'].nunique():,}")
                    m4.metric("Total Clicks", f"{gdf['clicks'].sum():,}")

        else:
            # ── Not connected — show precise error + instructions ────────────
            try:
                gsc_sec = st.secrets.get("gsc", {})
                raw_id = gsc_sec.get("client_id", "")
                raw_secret = gsc_sec.get("client_secret", "")
                raw_token = gsc_sec.get("refresh_token", "")
                has_id = bool(raw_id) and "YOUR_GOOGLE" not in raw_id
                has_secret = bool(
                    raw_secret) and "YOUR_GOOGLE" not in raw_secret
                has_token = bool(raw_token) and "YOUR_REFRESH" not in raw_token
            except Exception:
                has_id = has_secret = has_token = False

            if has_token and has_id and has_secret:
                st.error(
                    "❌  Authentication failed — your refresh token may be expired or revoked. Re-run `scripts/get_refresh_token.py` to get a new one.")
            elif has_id and has_secret and not has_token:
                st.warning(
                    "⚠️  Credentials found, but `refresh_token` is missing from Streamlit secrets.")
            else:
                st.info("Not connected. Follow the steps below to connect.")

            with st.expander("🔧  How to connect — step by step", expanded=not has_id):
                st.markdown("""
**Step 1 — Create a Google Cloud project and enable the API**

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project dropdown at the top → **New Project** → give it any name → **Create**
3. In the left menu: **APIs & Services → Library**
4. Search for **"Google Search Console API"** → click it → **Enable**

---

**Step 2 — Create OAuth credentials (Desktop App)**

1. Go to **APIs & Services → Credentials → + Create Credentials → OAuth 2.0 Client ID**
2. If prompted to configure the consent screen:
   - User type: **External** → Create
   - Fill in App name (anything) + your email → Save and Continue through all steps
   - Back on the Credentials page, retry creating the OAuth Client ID
3. Application type: **Desktop App** → Name it anything → **Create**
4. A dialog shows your **Client ID** and **Client Secret** — copy both

> 💡 Desktop App credentials do **not** need any redirect URI configuration.

---

**Step 3 — Get your refresh token**

Open a terminal in this project's folder and run:

```bash
python scripts/get_refresh_token.py
```

The script will ask for your Client ID and Client Secret interactively,
then open a browser for you to sign in with your Google account.
After approving access, it prints your `refresh_token`.

---

**Step 4 — Add to Streamlit secrets**

In Streamlit Cloud → your app → **Settings → Secrets**, paste:

```toml
OPENROUTER_API_KEY = "sk-or-XXXX"

[gsc]
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"
refresh_token = "1//0gXXXXXXXXXXXXXXXXXXXXXXXXX"
```

> ⚠️ `OPENROUTER_API_KEY` **must be above** `[gsc]` — keys below a section header belong to that section in TOML.

Save → the app will reconnect automatically.
                """)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Screaming Frog Embeddings
# ═══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    hcol, tcol = st.columns([5, 1])
    with hcol:
        st.subheader("2️⃣  Screaming Frog Embeddings")
        st.caption(
            "Detects pages with semantically similar content using pre-computed embedding vectors")
    with tcol:
        sf_on = st.toggle("Enable", value=True, key="sf_toggle")

    if sf_on:
        if st.session_state.sf_loaded:
            n_urls = len(st.session_state.url_df)
            n_dims = st.session_state.embeddings_matrix.shape[1]
            st.success(
                f"✅  {
                    n_urls:,} URLs loaded · {n_dims} embedding dimensions · {
                    n_urls * (
                        n_urls - 1) // 2:,} pairs to analyze")
            if st.button("🗑  Clear & re-upload", key="clear_sf"):
                for k in [
                    "url_df",
                    "embeddings_matrix",
                    "sf_loaded",
                    "sim_matrix",
                        "sim_matrix_url_count"]:
                    st.session_state[k] = None if k != "sf_loaded" else False
                    if k == "sim_matrix_url_count":
                        st.session_state[k] = 0
                st.rerun()
        else:
            uploaded = st.file_uploader(
                "Upload Screaming Frog embeddings (CSV or Excel)",
                type=["csv", "xlsx", "xls"],
                key="sf_upload",
                help="Bulk Export → Content → Embeddings in Screaming Frog v20+",
            )

            with st.expander("How to export embeddings from Screaming Frog"):
                st.markdown("""
1. Run a crawl in **Screaming Frog SEO Spider v20+**
2. After the crawl: **Bulk Export → Content → Embeddings**
3. Save the CSV — it contains one row per page with `Address` + 384 numeric columns
4. Upload that file here

The app auto-detects the URL column and all embedding dimensions.
                """)

            if uploaded:
                with st.spinner("Parsing embeddings…"):
                    url_df, emb_matrix, err = parse_sf_embeddings(uploaded)
                if err:
                    st.error(f"**Parse error:** {err}")
                else:
                    n = len(url_df)
                    if n > 300:
                        st.warning(
                            f"**{
                                n:,} URLs detected.** Pairwise similarity is O(N²) — "
                            "may be slow on Streamlit free tier. Consider enabling the URL filter below."
                        )
                    st.session_state.url_df = url_df
                    st.session_state.embeddings_matrix = emb_matrix
                    st.session_state.sf_loaded = True
                    st.success(
                        f"Loaded {
                            n:,} URLs · {
                            emb_matrix.shape[1]} dimensions")
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — URL Exclusions
# ═══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    hcol, tcol = st.columns([5, 1])
    with hcol:
        st.subheader("3️⃣  URL Exclusions")
        st.caption(
            "Filter OUT pages that don't benefit from duplicate content analysis — "
            "contact, legal, blog posts, PPC landing pages, and more"
        )
    with tcol:
        excl_on = st.toggle(
            "Enable",
            value=True,
            key="excl_toggle",
        )

    if excl_on:
        current_patterns = st.session_state.exclude_patterns

        # ── Pattern editor ────────────────────────────────────────────────
        edited_text = st.text_area(
            "Exclusion patterns (one per line)",
            value=patterns_to_text(current_patterns),
            height=220,
            key="excl_patterns_text",
            help=(
                "Plain text → substring match (case-insensitive). "
                "Add $ at the end for end-of-URL match (keeps subpages). "
                "Lines starting with # are comments."
            ),
        )

        bcol1, bcol2, _ = st.columns([1, 1, 3])
        with bcol1:
            if st.button("✅ Apply", key="apply_excl", type="primary"):
                st.session_state.exclude_patterns = patterns_from_text(
                    edited_text)
                st.session_state.sim_matrix = None  # invalidate cache
                st.success("Exclusions updated.")
                st.rerun()
        with bcol2:
            if st.button("↺ Reset defaults", key="reset_excl"):
                st.session_state.exclude_patterns = list(
                    DEFAULT_EXCLUDE_PATTERNS)
                st.session_state.sim_matrix = None
                st.success("Reset to defaults.")
                st.rerun()

        # ── Live preview (requires embeddings) ───────────────────────────
        if st.session_state.sf_loaded and st.session_state.url_df is not None:
            all_urls = st.session_state.url_df["url"].tolist()
            kept, excluded = apply_exclusions(all_urls, current_patterns)
            p1, p2, p3 = st.columns(3)
            p1.metric("Total URLs", len(all_urls))
            p2.metric("Excluded", len(excluded),
                      delta=f"−{len(excluded)}", delta_color="inverse")
            p3.metric("Remaining for analysis", len(kept),
                      delta=f"+{len(kept)}", delta_color="normal")

            if excluded:
                with st.expander(
                    f"Preview excluded URLs ({len(excluded)} total)"
                ):
                    st.dataframe(
                        pd.DataFrame({"excluded_url": excluded}),
                        use_container_width=True,
                        hide_index=True,
                    )

        # ── Help ─────────────────────────────────────────────────────────
        with st.expander("ℹ️ Pattern syntax guide"):
            st.markdown("""
| Pattern | Behaviour | Example match |
|---------|-----------|---------------|
| `ppc` | Any URL containing "ppc" | `/ppc-landing`, `/google-ppc/` |
| `/contact` | Any URL containing "/contact" | `/contact-us`, `/contact` |
| `/blog/` | Any URL with "/blog/" in path | `/blog/`, `/blog/my-post/` |
| `/service-area$` | URL **ends with** /service-area (ignores trailing slash) | `/service-area/` ✓ — `/service-area/dallas/` ✗ |
| `/locations$` | URL **ends with** /locations | `/locations/` ✓ — `/locations/houston/` ✗ |
| `# comment` | Ignored | — |

**Tip:** Click **Apply** after editing to immediately preview how many URLs are affected.
The exclusion runs automatically on every analysis page.
            """)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — AI Model
# ═══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    hcol, tcol = st.columns([5, 1])
    with hcol:
        st.subheader("4️⃣  AI Recommendations Model")
        st.caption(
            "Used on the AI Recommendations page to explain and prioritize duplicate issues")
    with tcol:
        ai_on = st.toggle("Enable", value=True, key="ai_toggle")

    if ai_on:
        model_labels = [m["label"] for m in AVAILABLE_MODELS]
        current_label = st.session_state.selected_model_label
        default_idx = model_labels.index(
            current_label) if current_label in model_labels else 0

        chosen = st.selectbox(
            "Select model",
            options=model_labels,
            index=default_idx,
            key="model_select",
        )
        chosen_id = next(m["id"]
                         for m in AVAILABLE_MODELS if m["label"] == chosen)
        st.session_state.selected_model_label = chosen
        st.session_state.selected_model_id = chosen_id

        if ai_ok:
            st.success("✅  OpenRouter API key configured")
        else:
            st.error(
                "❌  OpenRouter API key missing — add `OPENROUTER_API_KEY` to Streamlit secrets. "
                "Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys)")

# ═══════════════════════════════════════════════════════════════════════════════
# Navigation shortcuts
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("Run Analysis")

n1, n2, n3, n4 = st.columns(4)

with n1:
    ready = st.session_state.sf_loaded
    st.page_link(
        "pages/1_Semantic_Similarity.py",
        label="🔍 Semantic Similarity",
        disabled=not ready,
    )
    if not ready:
        st.caption("Upload SF embeddings first")

with n2:
    ready = st.session_state.gsc_loaded
    st.page_link(
        "pages/2_GSC_Cannibalization.py",
        label="📊 GSC Cannibalization",
        disabled=not ready,
    )
    if not ready:
        st.caption("Fetch GSC data first")

with n3:
    ready = st.session_state.sf_loaded or st.session_state.gsc_loaded
    st.page_link(
        "pages/3_Combined_Risk.py",
        label="🎯 Combined Risk",
        disabled=not ready,
    )
    if not ready:
        st.caption("Load at least one data source first")

with n4:
    ready = ai_ok
    st.page_link(
        "pages/4_AI_Recommendations.py",
        label="🤖 AI Recommendations",
        disabled=not ready,
    )
    if not ready:
        st.caption("Configure API key first")
