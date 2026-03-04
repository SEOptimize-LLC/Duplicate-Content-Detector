"""
URL Exclusions — pattern-based URL filtering for duplicate content analysis.

Pattern syntax:
  - Plain string  → case-insensitive substring match on the full URL
                    e.g. "ppc" matches any URL containing "ppc"
  - Ends with $   → matches only if the URL path ENDS with the pattern
                    (trailing slashes are ignored before comparing)
                    e.g. "/service-area$" matches /service-area/ but NOT
                    /service-area/dallas/ — useful for excluding a parent
                    page while keeping city/location subpages.
  - Lines starting with # → treated as comments and ignored
"""

DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    # PPC landing pages
    "ppc",

    # Contact / about
    "/contact",
    "/about",

    # Blog index and posts
    "/blog/",

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
      - If pattern ends with '$': end-of-URL match (ignores trailing slashes)
      - Otherwise: case-insensitive substring match on the full URL
    """
    url_lower = url.lower().rstrip("/")
    for raw in patterns:
        p = raw.strip()
        if not p or p.startswith("#"):
            continue
        if p.endswith("$"):
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


def patterns_from_text(text: str) -> list[str]:
    """Parse a newline-separated text block into a list of pattern strings."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def patterns_to_text(patterns: list[str]) -> str:
    """Format a list of pattern strings as a newline-separated text block."""
    return "\n".join(patterns)
