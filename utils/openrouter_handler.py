"""
OpenRouter Handler — AI recommendations via OpenRouter API (OpenAI-compatible).

Models available:
  - openai/gpt-5.1
  - anthropic/claude-sonnet-4-6
  - google/gemini-3.1-flash-lite-preview
"""

import streamlit as st
from openai import OpenAI
from typing import Optional, Generator

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

AVAILABLE_MODELS = [
    {
        "label": "GPT-5.1 (OpenAI)",
        "id": "openai/gpt-5.1",
    },
    {
        "label": "Claude Sonnet 4.6 (Anthropic)",
        "id": "anthropic/claude-sonnet-4-6",
    },
    {
        "label": "Gemini 3.1 Flash Lite Preview (Google)",
        "id": "google/gemini-3.1-flash-lite-preview",
    },
]


def get_client() -> Optional[OpenAI]:
    """Return an OpenAI client pointed at OpenRouter."""
    try:
        api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        return None

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )


def build_duplicate_prompt(
    url_pairs: list[dict],
    additional_context: str = "",
) -> str:
    """
    Build the AI prompt for analyzing a group of similar URLs.

    url_pairs: list of dicts with keys:
        url_a, url_b, similarity, gsc_shared_queries (optional list),
        clicks_a, clicks_b, impressions_a, impressions_b
    """
    prompt_lines = [
        "You are an expert SEO consultant specializing in content strategy and technical SEO.",
        "",
        "I need your help analyzing a group of web pages that have been flagged as "
        "potential duplicates or cannibalizing content. Please analyze the data below and "
        "provide specific, actionable recommendations.",
        "",
        "## Flagged URL Groups",
        "",
    ]

    for i, pair in enumerate(url_pairs, start=1):
        prompt_lines.append(f"### Group {i}")
        prompt_lines.append(f"- **URL A:** {pair.get('url_a', 'N/A')}")
        prompt_lines.append(f"- **URL B:** {pair.get('url_b', 'N/A')}")

        sim = pair.get('similarity')
        if sim is not None:
            prompt_lines.append(
                f"- **Semantic Similarity Score:** {sim:.2%} (cosine similarity)")

        shared_queries = pair.get('gsc_shared_queries', [])
        if shared_queries:
            top_queries = shared_queries[:5]
            prompt_lines.append(f"- **Shared GSC Queries ({len(shared_queries)} total):** "
                                + ", ".join(f'"{q}"' for q in top_queries))

        clicks_a = pair.get('clicks_a')
        clicks_b = pair.get('clicks_b')
        if clicks_a is not None and clicks_b is not None:
            prompt_lines.append(
                f"- **GSC Clicks — URL A:** {clicks_a}, **URL B:** {clicks_b}")

        impr_a = pair.get('impressions_a')
        impr_b = pair.get('impressions_b')
        if impr_a is not None and impr_b is not None:
            prompt_lines.append(
                f"- **GSC Impressions — URL A:** {impr_a}, **URL B:** {impr_b}")

        prompt_lines.append("")

    if additional_context:
        prompt_lines += [
            "## Additional Context",
            additional_context,
            "",
        ]

    prompt_lines += [
        "## Your Analysis Should Cover:",
        "",
        "1. **Root Cause** — Why are these pages likely duplicating or cannibalizing each other?",
        "2. **Recommended Action** — Choose one:",
        "   - Consolidate (merge content into one URL)",
        "   - Canonicalize (set canonical tag pointing to the preferred URL)",
        "   - Redirect (301 redirect weaker page to stronger page)",
        "   - Differentiate (make the pages clearly distinct in topic/intent)",
        "   - No action needed (explain why these are actually fine)",
        "3. **Which URL to Keep** (if consolidating/redirecting) — and why",
        "4. **Content Differentiation Suggestions** (if keeping both) — specific angles to make each page unique",
        "5. **Priority Level** — Critical / High / Medium / Low",
        "",
        "Be specific and data-driven. Reference the similarity scores and GSC metrics in your reasoning.",
        "Format your response in clear markdown with headers.",
    ]

    return "\n".join(prompt_lines)


def stream_recommendation(
    model_id: str,
    url_pairs: list[dict],
    additional_context: str = "",
) -> Generator[str, None, None]:
    """
    Stream AI recommendation text for the given URL pairs.
    Yields string chunks as they arrive.
    """
    client = get_client()
    if client is None:
        yield "**Error:** OpenRouter API key not configured. Add `OPENROUTER_API_KEY` to your Streamlit secrets."
        return

    prompt = build_duplicate_prompt(url_pairs, additional_context)

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=2000,
            temperature=0.3,  # Lower temp for more consistent SEO advice
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
    except Exception as e:
        yield f"\n\n**Error calling OpenRouter API:** {e}"


def get_recommendation(
    model_id: str,
    url_pairs: list[dict],
    additional_context: str = "",
) -> str:
    """
    Non-streaming version — returns the full recommendation as a string.
    """
    return "".join(stream_recommendation(
        model_id, url_pairs, additional_context))
