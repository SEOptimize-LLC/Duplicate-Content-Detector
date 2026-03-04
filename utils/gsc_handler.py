"""
GSC Handler — Google Search Console OAuth + Search Analytics API
Supports two auth patterns:
  1. Stored refresh token in Streamlit secrets (recommended for Streamlit Cloud)
  2. Interactive OAuth flow (for local development / first-time setup)
"""

import json
import math
import time
import requests
import streamlit as st
from datetime import datetime, date, timedelta
from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

# Google OAuth endpoints
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GSC_API_BASE = "https://www.googleapis.com/webmasters/v3"
SCOPES = "https://www.googleapis.com/auth/webmasters.readonly"


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

def _get_credentials_from_secrets() -> tuple[str, str]:
    """Read client_id and client_secret from st.secrets."""
    try:
        client_id = st.secrets["gsc"]["client_id"]
        client_secret = st.secrets["gsc"]["client_secret"]
        return client_id, client_secret
    except (KeyError, AttributeError):
        return "", ""


def _refresh_access_token(client_id: str, client_secret: str,
                          refresh_token: str) -> Optional[dict]:
    """Exchange a refresh token for a new access token."""
    resp = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        data["expires_at"] = time.time() + data.get("expires_in", 3600) - 60
        return data
    return None


def _exchange_code_for_token(code: str, redirect_uri: str,
                             client_id: str, client_secret: str) -> Optional[dict]:
    """Exchange an authorization code for access + refresh tokens."""
    resp = requests.post(TOKEN_URL, data={
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        data["expires_at"] = time.time() + data.get("expires_in", 3600) - 60
        return data
    return None


def get_valid_access_token() -> Optional[str]:
    """
    Return a valid access token from session state.
    Auto-refreshes if expired.
    """
    token_data = st.session_state.get("gsc_token_data")
    if not token_data:
        return None

    # Check expiry
    if time.time() >= token_data.get("expires_at", 0):
        client_id, client_secret = _get_credentials_from_secrets()
        refresh_token = token_data.get("refresh_token")
        if refresh_token and client_id and client_secret:
            new_data = _refresh_access_token(
                client_id, client_secret, refresh_token)
            if new_data:
                # Preserve refresh_token (not always returned on refresh)
                if "refresh_token" not in new_data:
                    new_data["refresh_token"] = refresh_token
                st.session_state.gsc_token_data = new_data
                return new_data["access_token"]
        return None

    return token_data.get("access_token")


def is_authenticated() -> bool:
    return get_valid_access_token() is not None


# ---------------------------------------------------------------------------
# Auth flow helpers
# ---------------------------------------------------------------------------

def build_auth_url(redirect_uri: str) -> str:
    """Build the Google OAuth authorization URL."""
    client_id, _ = _get_credentials_from_secrets()
    if not client_id:
        return ""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent",  # Forces refresh_token to be issued
    }
    from urllib.parse import urlencode
    return f"{AUTH_URL}?{urlencode(params)}"


def handle_oauth_callback(code: str, redirect_uri: str) -> bool:
    """
    Exchange the authorization code for tokens.
    Returns True on success.
    """
    client_id, client_secret = _get_credentials_from_secrets()
    if not client_id:
        return False
    token_data = _exchange_code_for_token(
        code, redirect_uri, client_id, client_secret)
    if token_data:
        st.session_state.gsc_token_data = token_data
        return True
    return False


def init_from_stored_refresh_token() -> bool:
    """
    Initialize session from a refresh_token stored in secrets.
    Useful for single-user Streamlit Cloud deployments.
    Returns True if successful.
    """
    try:
        stored_refresh = st.secrets["gsc"].get("refresh_token", "")
    except (KeyError, AttributeError):
        stored_refresh = ""

    if not stored_refresh:
        return False

    client_id, client_secret = _get_credentials_from_secrets()
    if not client_id:
        return False

    token_data = _refresh_access_token(
        client_id, client_secret, stored_refresh)
    if token_data:
        if "refresh_token" not in token_data:
            token_data["refresh_token"] = stored_refresh
        st.session_state.gsc_token_data = token_data
        return True
    return False


def logout():
    """Clear GSC authentication from session state."""
    st.session_state.pop("gsc_token_data", None)
    st.session_state.pop("gsc_properties", None)
    st.session_state.pop("gsc_data", None)
    st.session_state.pop("selected_property", None)


# ---------------------------------------------------------------------------
# GSC API calls
# ---------------------------------------------------------------------------

def fetch_properties() -> list[str]:
    """Return list of all verified GSC properties for the authenticated user."""
    token = get_valid_access_token()
    if not token:
        return []

    resp = requests.get(
        f"{GSC_API_BASE}/sites",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    if resp.status_code == 200:
        sites = resp.json().get("siteEntry", [])
        return [s["siteUrl"] for s in sites]
    return []


def fetch_gsc_data(
    site_url: str,
    start_date: str,
    end_date: str,
    row_limit: int = 25000,
) -> list[dict]:
    """
    Pull query + URL performance data from GSC Search Analytics.
    Paginates automatically up to row_limit total rows.

    Returns a list of dicts with keys:
        query, page, clicks, impressions, ctr, position
    """
    token = get_valid_access_token()
    if not token:
        return []

    url = f"{GSC_API_BASE}/sites/{
        requests.utils.quote(
            site_url,
            safe='')}/searchAnalytics/query"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"}

    all_rows = []
    start_row = 0
    batch_size = min(25000, row_limit)

    while len(all_rows) < row_limit:
        payload = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": ["query", "page"],
            "rowLimit": batch_size,
            "startRow": start_row,
            "dataState": "final",
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            break

        rows = resp.json().get("rows", [])
        if not rows:
            break

        for row in rows:
            keys = row.get("keys", [])
            all_rows.append({
                "query": keys[0] if len(keys) > 0 else "",
                "page": keys[1] if len(keys) > 1 else "",
                "clicks": row.get("clicks", 0),
                "impressions": row.get("impressions", 0),
                "ctr": round(row.get("ctr", 0) * 100, 2),
                "position": round(row.get("position", 0), 1),
            })

        if len(rows) < batch_size:
            break

        start_row += len(rows)

    return all_rows[:row_limit]


def fetch_sitemap_urls(site_url: str) -> list[str]:
    """Fetch all URLs from the GSC sitemap index for a property."""
    token = get_valid_access_token()
    if not token:
        return []

    url = f"{GSC_API_BASE}/sites/{
        requests.utils.quote(
            site_url, safe='')}/sitemaps"
    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {token}"},
        timeout=15)
    if resp.status_code != 200:
        return []

    sitemaps = resp.json().get("sitemap", [])
    urls = []
    for sitemap in sitemaps:
        for content in sitemap.get("contents", []):
            # The contents field shows indexed URL counts but not individual URLs
            # We use sitemap path as reference
            pass
    # Return sitemap paths instead (actual URL discovery requires crawling)
    return [s.get("path", "") for s in sitemaps if s.get("path")]


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
    "utm_id", "utm_name", "utm_reader", "utm_place", "utm_pubreferrer",
    "gclid", "gclsrc", "dclid", "fbclid", "msclkid", "twclid", "ttclid",
    "_ga", "_gac", "_gl", "mc_eid", "mc_cid", "igshid", "s_kwcid",
    "ref", "referrer",
}


def normalize_url(url: str) -> str:
    """Strip tracking/UTM parameters from a URL, preserving other query params."""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=True)
        clean = {k: v for k, v in params.items()
                 if k.lower() not in _TRACKING_PARAMS}
        return urlunparse(parsed._replace(query=urlencode(clean, doseq=True)))
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Cannibalization analysis
# ---------------------------------------------------------------------------

def detect_cannibalization(
    gsc_rows: list[dict],
    min_impressions: int = 10,
    min_clicks: int = 0,
) -> list[dict]:
    """
    Identify queries where multiple URLs are competing.

    Returns sorted list of cannibalization findings:
        query, urls (list), total_clicks, total_impressions,
        dominant_url, competing_urls, impact_score
    """
    from collections import defaultdict

    # Normalize URLs to strip tracking params, then re-aggregate by
    # (query, normalized_page) so UTM variants count as the same URL.
    aggregated: dict[tuple, dict] = {}
    for row in gsc_rows:
        if (row["impressions"] >= min_impressions
                and row["clicks"] >= min_clicks):
            norm_page = normalize_url(row["page"])
            key = (row["query"], norm_page)
            if key not in aggregated:
                aggregated[key] = {
                    "query": row["query"],
                    "page": norm_page,
                    "clicks": 0,
                    "impressions": 0,
                }
            aggregated[key]["clicks"] += row["clicks"]
            aggregated[key]["impressions"] += row["impressions"]

    # Group normalized rows by query
    query_map: dict[str, list[dict]] = defaultdict(list)
    for row in aggregated.values():
        query_map[row["query"]].append(row)

    findings = []
    for query, pages in query_map.items():
        if len(pages) < 2:
            continue

        # Sort by clicks desc to find dominant URL
        pages_sorted = sorted(
            pages, key=lambda x: x["clicks"], reverse=True)
        dominant = pages_sorted[0]
        competing = pages_sorted[1:]

        total_clicks = sum(p["clicks"] for p in pages)
        total_impressions = sum(p["impressions"] for p in pages)

        # Impact score: more clicks + more impressions + more competing
        # URLs = higher risk
        impact_score = round(
            (total_clicks + math.log1p(total_impressions)) * len(pages),
            2,
        )

        findings.append({
            "query": query,
            "urls": [p["page"] for p in pages_sorted],
            "dominant_url": dominant["page"],
            "dominant_clicks": dominant["clicks"],
            "dominant_impressions": dominant["impressions"],
            "competing_urls": [p["page"] for p in competing],
            "total_clicks": total_clicks,
            "total_impressions": total_impressions,
            "num_competing_urls": len(pages),
            "impact_score": impact_score,
        })

    # Sort by impact score descending
    findings.sort(key=lambda x: x["impact_score"], reverse=True)
    return findings


def get_url_cannibalization_summary(
        cannibalization_findings: list[dict]) -> dict[str, dict]:
    """
    Build a per-URL summary of how many queries it's involved in as a cannibalizer.
    Returns dict keyed by URL.
    """
    from collections import defaultdict
    summary: dict[str, dict] = defaultdict(lambda: {
        "queries_competing": 0,
        "total_impact": 0.0,
        "as_dominant": 0,
        "as_competing": 0,
    })

    for finding in cannibalization_findings:
        dom_url = finding["dominant_url"]
        summary[dom_url]["queries_competing"] += 1
        summary[dom_url]["as_dominant"] += 1
        summary[dom_url]["total_impact"] += finding["impact_score"]

        for url in finding["competing_urls"]:
            summary[url]["queries_competing"] += 1
            summary[url]["as_competing"] += 1
            summary[url]["total_impact"] += finding["impact_score"]

    return dict(summary)
