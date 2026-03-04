"""
Embeddings Handler — Screaming Frog CSV parser + cosine similarity computation.

Expected SF export format:
  - One column containing the URL (auto-detected by name: 'Address', 'URL', 'url', etc.)
  - Remaining numeric columns = embedding vector dimensions
  - Example: Address, dim_0, dim_1, ..., dim_383  (384-dim all-MiniLM-L6-v2)

All pairwise cosine similarity is computed with sklearn for efficiency.
"""

import io
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

# Recognized URL column names (case-insensitive match)
URL_COLUMN_CANDIDATES = ["address", "url", "page", "page_url", "landing page", "source"]

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


def parse_sf_embeddings(file_obj) -> tuple[Optional[pd.DataFrame], Optional[np.ndarray], str]:
    """
    Parse a Screaming Frog embeddings CSV file.

    Returns:
        (url_df, embeddings_matrix, error_message)
        - url_df: DataFrame with columns ['url'] + any metadata columns
        - embeddings_matrix: numpy array of shape (n_urls, n_dims)
        - error_message: empty string on success, description on failure
    """
    try:
        # Read CSV — handle UTF-8 and Latin-1 encodings
        try:
            df = pd.read_csv(file_obj, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, encoding="latin-1", low_memory=False)
    except Exception as e:
        return None, None, f"Failed to read CSV: {e}"

    if df.empty:
        return None, None, "The uploaded file is empty."

    # Find URL column
    url_col = detect_url_column(df)
    if url_col is None:
        return None, None, (
            "Could not detect the URL column. Expected a column named one of: "
            + ", ".join(URL_COLUMN_CANDIDATES)
        )

    # Identify numeric embedding columns (all numeric columns except URL)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    embedding_cols = [c for c in numeric_cols]  # all numeric = embedding dims

    if len(embedding_cols) < 10:
        return None, None, (
            f"Only {len(embedding_cols)} numeric columns found. "
            "Expected at least 10 embedding dimensions. "
            "Ensure you exported the embeddings (not just metadata) from Screaming Frog."
        )

    # Drop rows with missing URL or NaN in embeddings
    df_clean = df[[url_col] + embedding_cols].dropna()
    df_clean = df_clean.rename(columns={url_col: "url"})
    df_clean["url"] = df_clean["url"].str.strip()
    df_clean = df_clean[df_clean["url"].str.startswith("http")]  # sanity filter

    if len(df_clean) < 2:
        return None, None, "Need at least 2 valid URLs with embeddings to run analysis."

    # Extract matrix
    embeddings = df_clean[embedding_cols].values.astype(np.float32)

    # L2-normalize rows so cosine sim = dot product (speeds up computation)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero
    embeddings = embeddings / norms

    url_df = df_clean[["url"]].reset_index(drop=True)
    return url_df, embeddings, ""


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
        df = df.sort_values("similarity", ascending=False).reset_index(drop=True)
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
    medium_pairs = int(np.sum((upper >= medium_threshold) & (upper < high_threshold)))

    return {
        "total_urls": n,
        "total_pairs": total_pairs,
        "high_risk_pairs": high_pairs,
        "medium_risk_pairs": medium_pairs,
        "embedding_dimensions": sim_matrix.shape[1] if sim_matrix.ndim > 1 else 0,
        "avg_similarity": round(float(np.mean(upper[upper > 0])), 4) if np.any(upper > 0) else 0.0,
    }
