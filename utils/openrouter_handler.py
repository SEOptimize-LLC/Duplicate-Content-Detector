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
        # Check top-level first, then [gsc] section (common paste mistake)
        api_key = (
            st.secrets.get("OPENROUTER_API_KEY", "")
            or st.secrets.get("gsc", {}).get("OPENROUTER_API_KEY", "")
        )
    except Exception:
        api_key = ""

    if not api_key or "YOUR_OPENROUTER" in api_key:
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
    Build a prompt that produces ONE unified, client-facing audit report
    covering all URL pairs — no batches, no repeated group numbering.

    url_pairs: list of dicts with keys:
        url_a, url_b, similarity, gsc_shared_queries (optional list),
        clicks_a, clicks_b, impressions_a, impressions_b
    """
    n = len(url_pairs)
    lines = [
        "You are a senior SEO consultant writing a professional, client-facing"
        " duplicate content audit report.",
        "",
        f"The analysis tool has flagged {n} URL pair(s) for duplicate content"
        " or keyword cannibalization. Use the data below to write ONE cohesive"
        " report — not separate independent analyses. Number every pair"
        " sequentially (Pair 1, Pair 2, …) and keep that numbering consistent"
        " throughout the entire document.",
        "",
        "---",
        "",
        "## Input Data",
        "",
    ]

    for i, pair in enumerate(url_pairs, start=1):
        lines.append(f"### Pair {i}")
        lines.append(f"- **URL A:** {pair.get('url_a', 'N/A')}")
        lines.append(f"- **URL B:** {pair.get('url_b', 'N/A')}")

        sim = pair.get('similarity')
        if sim is not None:
            lines.append(f"- **Semantic Similarity:** {sim:.0%}")

        shared_queries = pair.get('gsc_shared_queries', [])
        if shared_queries:
            q_str = ", ".join(f'"{q}"' for q in shared_queries[:5])
            lines.append(
                f"- **Shared GSC Queries ({len(shared_queries)} total):**"
                f" {q_str}"
            )

        clicks_a = pair.get('clicks_a')
        clicks_b = pair.get('clicks_b')
        if clicks_a is not None and clicks_b is not None:
            lines.append(
                f"- **GSC Clicks — URL A:** {clicks_a} |"
                f" **URL B:** {clicks_b}"
            )

        impr_a = pair.get('impressions_a')
        impr_b = pair.get('impressions_b')
        if impr_a is not None and impr_b is not None:
            lines.append(
                f"- **GSC Impressions — URL A:** {impr_a} |"
                f" **URL B:** {impr_b}"
            )

        lines.append("")

    if additional_context:
        lines += ["## Additional Context", additional_context, ""]

    lines += [
        "---",
        "",
        "## Required Report Structure",
        "",
        "Write the report using exactly this structure:",
        "",
        "### Executive Summary",
        "2–3 sentences covering: how many pairs were analyzed, severity"
        " breakdown (how many Critical / High / Medium / Low), and the"
        " dominant issue pattern observed across the site.",
        "",
        "### Pair-by-Pair Analysis",
        "For each pair use this exact sub-structure (keep Pair numbers"
        " matching the input data above):",
        "",
        "**Pair N — [Short descriptive title, e.g., 'Service hub vs."
        " individual service page']**",
        "- **URL A:** (paste the exact URL A from the input data)",
        "- **URL B:** (paste the exact URL B from the input data)",
        "- **Priority:** 🔴 Critical / 🟠 High / 🟡 Medium / 🔵 Low",
        "- **Recommended Action:** Consolidate / Canonicalize /"
        " Redirect (301) / Differentiate / No Action Needed",
        "- **Root Cause:** Why are these URLs competing in search?",
        "- **Which URL to Keep:** (only if consolidating or redirecting)"
        " State which URL to keep and why, referencing clicks/impressions"
        " where available.",
        "- **Implementation Steps:** Numbered list of concrete next steps.",
        "",
        "### Action Plan Summary",
        "A markdown table with these columns:",
        "Pair # | URL A | URL B | Recommended Action | Priority",
        "Sort rows by priority: Critical first, then High, Medium, Low.",
        "",
        "---",
        "",
        "Rules:",
        "- Do NOT use the words 'Group' or 'Batch' anywhere in the report.",
        "- Be concise and data-driven; reference similarity scores and GSC"
        " metrics directly.",
        "- Write as if delivering this report to a client — clear, professional,"
        " and actionable.",
    ]

    return "\n".join(lines)


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
        yield (
            "**Error:** OpenRouter API key not configured."
            " Add `OPENROUTER_API_KEY` to your Streamlit secrets."
        )
        return

    prompt = build_duplicate_prompt(url_pairs, additional_context)

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=4096,
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
