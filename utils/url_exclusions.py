"""
URL Exclusions — pattern-based URL filtering for duplicate content analysis.

Pattern syntax:
  - Plain string   → case-insensitive substring match on the full URL
                     e.g. "ppc" matches any URL containing "ppc"
  - Ends with $    → matches only if the URL path ENDS with the pattern
                     (trailing slashes are ignored before comparing)
                     e.g. "/service-area$" matches /service-area/ but NOT
                     /service-area/dallas/ — useful for excluding a parent
                     page while keeping city/location subpages.
  - regex:<pattern>→ full regex match (re.search, case-insensitive) on the
                     full URL. e.g. "regex:/page/\d+" matches any pagination
                     URL like /blog/page/2/ or /testimonials/page/3/.
  - Lines starting with # → treated as comments and ignored
"""

import re
from urllib.parse import urlparse

DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    # PPC landing pages
    "ppc",

    # Contact / about
    "/contact",
    "/about",

    # Blog index only ($ = end-of-URL match — keeps /blog/my-post/ etc.)
    "/blog$",

    # Pagination — any subfolder: /blog/page/2/, /testimonials/page/3/, etc.
    "regex:/page/\\d+",

    # Membership / subscription
    "membership",

    # Auth pages
    "/login",
    "/sign-in",
    "/register",
    "/my-account",

    # Social proof pages
    "/testimonial",
    "/review",

    # Promotions / one-time pages
    "/giveaway",
    "/specials",
    "/coupon",

    # Conversion confirmations
    "/thank-you",
    "/thank_you",
    "/thankyou",

    # Booking / scheduling
    "/appointment",
    "/booking",
    "/schedule",

    # Location hub pages only (NOT city subpages)
    # $ = end-of-URL match → keeps /service-area/dallas/ etc.
    "/service-area$",
    "/locations$",

    # Content authorship
    "/author/",

    # Financing / payments
    "/financing",

    # Social / community
    "/stay-connected",

    # Legal / compliance
    "/terms",
    "/privacy",
    "/disclaimer",
    "/return-policy",
    "/refund-policy",
    "/legal",

    # Careers
    "/career",
    "/careers",
    "/jobs",

    # Support
    "/customer-support",
    "/support",

    # FAQ
    "/faq",
]


def should_exclude(url: str, patterns: list[str]) -> bool:
    """
    Return True if the URL matches any exclusion pattern.

    Pattern rules:
      - Strip leading/trailing whitespace and skip blank lines / comments (#)
      - If pattern starts with 'regex:': run re.search(pattern, url, IGNORECASE)
      - If pattern ends with '$': end-of-URL match (ignores trailing slashes)
      - Otherwise: case-insensitive substring match on the full URL
    """
    url_lower = url.lower().rstrip("/")
    for raw in patterns:
        p = raw.strip()
        if not p or p.startswith("#"):
            continue
        if p.startswith("regex:"):
            try:
                if re.search(p[6:], url, re.IGNORECASE):
                    return True
            except re.error:
                pass  # Malformed regex — skip silently
        elif p.endswith("$"):
            needle = p[:-1].lower().rstrip("/")
            if url_lower.endswith(needle):
                return True
        else:
            if p.lower() in url_lower:
                return True
    return False


def apply_exclusions(
    urls: list[str],
    patterns: list[str],
) -> tuple[list[str], list[str]]:
    """
    Split a list of URLs into (kept, excluded) based on patterns.
    Returns (kept_urls, excluded_urls).
    """
    kept, excluded = [], []
    for url in urls:
        if should_exclude(url, patterns):
            excluded.append(url)
        else:
            kept.append(url)
    return kept, excluded


def _strip_www(netloc: str) -> str:
    """Remove 'www.' prefix from a netloc string."""
    return netloc[4:] if netloc.startswith("www.") else netloc


def is_homepage(url: str, property_url: str) -> bool:
    """
    Return True if url is the homepage (root) of the given GSC property.

    Handles both URL-prefix properties (https://example.com/) and
    domain properties (sc-domain:example.com), including http/www variants.
    Trailing slashes and www prefixes are normalised before comparing.
    """
    if not property_url or not url:
        return False

    url_norm = url.rstrip("/").lower()

    if property_url.startswith("sc-domain:"):
        domain = property_url.replace("sc-domain:", "").rstrip("/").lower()
        return url_norm in {
            f"https://{domain}",
            f"http://{domain}",
            f"https://www.{domain}",
            f"http://www.{domain}",
        }

    # URL-prefix property — try exact match first
    prop_norm = property_url.rstrip("/").lower()
    if url_norm == prop_norm:
        return True

    # Fallback: handle www ↔ non-www mismatch between property and page URL.
    # Only consider root-path URLs (path == "" or "/") as candidates.
    try:
        parsed = urlparse(url_norm)
        prop_parsed = urlparse(prop_norm)
        if parsed.path not in ("", "/"):
            return False  # Has a non-root path → definitely not homepage
        if parsed.scheme != prop_parsed.scheme:
            return False  # Different protocol
        return _strip_www(parsed.netloc) == _strip_www(prop_parsed.netloc)
    except Exception:
        return False


def patterns_from_text(text: str) -> list[str]:
    """Parse a newline-separated text block into a list of pattern strings."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def patterns_to_text(patterns: list[str]) -> str:
    """Format a list of pattern strings as a newline-separated text block."""
    return "\n".join(patterns)
