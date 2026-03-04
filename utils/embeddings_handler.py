"""
Embeddings Handler — Screaming Frog CSV parser + cosine similarity computation.

Supports two CSV formats:

  Format A — Wide (standard SF embeddings export):
    Address, dim_0, dim_1, ..., dim_383
    One numeric column per embedding dimension.

  Format B — Packed (custom JS extraction, e.g. ChatGPT embeddings):
    Address, Status Code, "(ChatGPT) Extract embeddings from page content 1"
    Embedding values are a comma-separated string inside a single cell.

All pairwise cosine similarity is computed with sklearn for efficiency.
"""

import io
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

# Recognized URL column names (case-insensitive match)
URL_COLUMN_CANDIDATES = [
    "address",
    "url",
    "page",
    "page_url",
    "landing page",
    "source"]

# Risk thresholds
THRESHOLD_HIGH = 0.85    # likely duplicate
THRESHOLD_MEDIUM = 0.60  # overlapping intent
THRESHOLD_LOW = 0.40     # borderline


def detect_url_column(df: pd.DataFrame) -> Optional[str]:
    """Return the column name that contains URLs, or None if not found."""
    for col in df.columns:
        if col.strip().lower() in URL_COLUMN_CANDIDATES:
            return col
    # Fallback: first column that contains http
    for col in df.columns:
        sample = df[col].dropna().head(10).astype(str)
        if sample.str.startswith("http").any():
            return col
    return None


# Metadata column names to skip when looking for embedding dimensions
_METADATA_COLS = {
    "status code", "status", "indexability", "indexability status",
    "title 1", "meta description 1", "h1-1", "h1-2", "h2-1", "h2-2",
    "word count", "crawl depth", "inlinks", "unique inlinks",
    "outlinks", "unique outlinks", "content type", "response time",
}


def _parse_packed_embeddings(
        series: pd.Series) -> Optional[np.ndarray]:
    """
    Try to parse a Series of comma-separated float strings (Format B).
    Returns a 2D float32 numpy array, or None if the column doesn't match.
    """
    try:
        sample = series.dropna().iloc[0] if not series.dropna().empty else ""
        if not isinstance(sample, str):
            return None
        parts = [p.strip() for p in sample.split(",")]
        if len(parts) < 10:
            return None
        # Validate first 20 values are floats
        [float(p) for p in parts[:20]]
        # Parse all rows
        matrix = np.array(
            [[float(v) for v in str(row).split(",")]
             for row in series],
            dtype=np.float32,
        )
        return matrix
    except (ValueError, AttributeError, IndexError):
        return None


def parse_sf_embeddings(
        file_obj) -> tuple[Optional[pd.DataFrame], Optional[np.ndarray], str]:
    """
    Parse a Screaming Frog embeddings CSV (Format A wide or Format B packed).

    Returns:
        (url_df, embeddings_matrix, error_message)
        - url_df: DataFrame with column 'url'
        - embeddings_matrix: numpy array of shape (n_urls, n_dims)
        - error_message: empty string on success, description on failure
    """
    try:
        # Detect file type by name if available, else try CSV then Excel
        fname = getattr(file_obj, "name", "")
        if fname.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_obj)
        else:
            try:
                df = pd.read_csv(file_obj, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                file_obj.seek(0)
                df = pd.read_csv(
                    file_obj, encoding="latin-1", low_memory=False)
    except Exception as e:
        return None, None, f"Failed to read file: {e}"

    if df.empty:
        return None, None, "The uploaded file is empty."

    # Find URL column
    url_col = detect_url_column(df)
    if url_col is None:
        return None, None, (
            "Could not detect the URL column. Expected a column named one of: "
            + ", ".join(URL_COLUMN_CANDIDATES)
        )

    # ── Format A: wide (one numeric col per embedding dimension) ────────
    non_url_cols = [c for c in df.columns if c != url_col]
    numeric_cols = df[non_url_cols].select_dtypes(
        include=[np.number]).columns.tolist()
    # Exclude obvious metadata columns
    embedding_cols = [
        c for c in numeric_cols
        if c.strip().lower() not in _METADATA_COLS
    ]

    if len(embedding_cols) >= 10:
        df_clean = df[[url_col] + embedding_cols].dropna()
        df_clean = df_clean.rename(columns={url_col: "url"})
        df_clean["url"] = df_clean["url"].str.strip()
        df_clean = df_clean[df_clean["url"].str.startswith("http")]
        if len(df_clean) >= 2:
            embeddings = df_clean[embedding_cols].values.astype(np.float32)
            embeddings = _normalize(embeddings)
            return df_clean[["url"]].reset_index(drop=True), embeddings, ""

    # ── Format B: packed (comma-separated floats in a single text col) ──
    object_cols = [
        c for c in non_url_cols
        if c.strip().lower() not in _METADATA_COLS
        and df[c].dtype == object
    ]
    for col in object_cols:
        # Only try rows where the URL is valid
        url_mask = df[url_col].astype(str).str.startswith("http")
        sub = df[url_mask][[url_col, col]].dropna()
        matrix = _parse_packed_embeddings(sub[col])
        if matrix is not None and matrix.shape[0] >= 2:
            url_df = sub[[url_col]].rename(
                columns={url_col: "url"}).reset_index(drop=True)
            url_df["url"] = url_df["url"].str.strip()
            embeddings = _normalize(matrix)
            return url_df, embeddings, ""

    # ── Neither format matched ───────────────────────────────────────────
    n_num = len(embedding_cols)
    return None, None, (
        f"Could not parse embeddings. Found {n_num} numeric column(s) and "
        "no packed-embedding text columns. "
        "Supported formats: (A) one numeric column per dimension, or "
        "(B) a single text column with comma-separated float values. "
        "Check that your export includes the embedding data."
    )


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows so cosine similarity = dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (matrix / norms).astype(np.float32)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise cosine similarity matrix.
    Returns (n, n) float32 array.
    """
    sim = cosine_similarity(embeddings)
    return sim.astype(np.float32)


def get_pairs_above_threshold(
    url_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    threshold: float = THRESHOLD_MEDIUM,
    max_pairs: int = 5000,
) -> pd.DataFrame:
    """
    Extract all URL pairs with similarity >= threshold.
    Returns DataFrame sorted by similarity desc.
    """
    n = len(url_df)
    urls = url_df["url"].tolist()

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score >= threshold:
                rows.append({
                    "url_a": urls[i],
                    "url_b": urls[j],
                    "similarity": round(score, 4),
                    "risk_level": _risk_label(score),
                })
                if len(rows) >= max_pairs:
                    break
        if len(rows) >= max_pairs:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            "similarity",
            ascending=False).reset_index(
            drop=True)
    return df


def _risk_label(score: float) -> str:
    if score >= THRESHOLD_HIGH:
        return "High"
    elif score >= THRESHOLD_MEDIUM:
        return "Medium"
    elif score >= THRESHOLD_LOW:
        return "Low"
    return "Minimal"


def get_top_similar_for_url(
    url: str,
    url_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    top_n: int = 10,
    min_threshold: float = THRESHOLD_LOW,
) -> pd.DataFrame:
    """Return the top_n most similar URLs to a given URL."""
    urls = url_df["url"].tolist()
    if url not in urls:
        return pd.DataFrame()

    idx = urls.index(url)
    scores = sim_matrix[idx].copy()
    scores[idx] = 0  # exclude self

    top_indices = np.argsort(scores)[::-1][:top_n]
    rows = []
    for i in top_indices:
        score = float(scores[i])
        if score < min_threshold:
            break
        rows.append({
            "url": urls[i],
            "similarity": round(score, 4),
            "risk_level": _risk_label(score),
        })
    return pd.DataFrame(rows)


def build_heatmap_data(
    url_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    max_urls: int = 50,
) -> tuple[list[str], np.ndarray]:
    """
    Return (labels, submatrix) for heatmap visualization.
    Selects top max_urls URLs with highest average similarity to others.
    """
    n = min(len(url_df), max_urls)
    urls = url_df["url"].tolist()

    if len(urls) <= max_urls:
        labels = [_shorten_url(u) for u in urls[:n]]
        return labels, sim_matrix[:n, :n]

    # Pick URLs with highest mean off-diagonal similarity
    np.fill_diagonal(sim_matrix, 0)
    avg_sim = sim_matrix.mean(axis=1)
    np.fill_diagonal(sim_matrix, 1)

    top_idx = np.argsort(avg_sim)[::-1][:max_urls]
    top_idx_sorted = sorted(top_idx.tolist())

    labels = [_shorten_url(urls[i]) for i in top_idx_sorted]
    sub = sim_matrix[np.ix_(top_idx_sorted, top_idx_sorted)]
    return labels, sub


def _shorten_url(url: str, max_len: int = 50) -> str:
    """Shorten a URL for display in charts."""
    url = url.replace("https://", "").replace("http://", "").rstrip("/")
    if len(url) > max_len:
        return "..." + url[-(max_len - 3):]
    return url


def get_summary_stats(
    url_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    high_threshold: float = THRESHOLD_HIGH,
    medium_threshold: float = THRESHOLD_MEDIUM,
) -> dict:
    """Compute summary statistics for the analysis."""
    n = len(url_df)
    total_pairs = n * (n - 1) // 2

    # Vectorized upper triangle counting
    upper = np.triu(sim_matrix, k=1)
    high_pairs = int(np.sum(upper >= high_threshold))
    medium_pairs = int(
        np.sum(
            (upper >= medium_threshold) & (
                upper < high_threshold)))

    return {
        "total_urls": n,
        "total_pairs": total_pairs,
        "high_risk_pairs": high_pairs,
        "medium_risk_pairs": medium_pairs,
        "embedding_dimensions": sim_matrix.shape[1] if sim_matrix.ndim > 1 else 0,
        "avg_similarity": round(float(np.mean(upper[upper > 0])), 4) if np.any(upper > 0) else 0.0,
    }
