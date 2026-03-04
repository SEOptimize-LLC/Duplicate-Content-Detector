"""
Clustering — Group URLs into duplicate clusters using connected-component analysis.
Uses a graph-based approach: URLs are nodes, edges = similarity above threshold.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import Optional


def build_clusters(
    url_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    threshold: float = 0.85,
) -> list[list[str]]:
    """
    Group URLs into clusters where each pair in the cluster has
    similarity >= threshold (transitive: A-B and B-C means A,B,C are clustered).

    Returns list of clusters (each cluster is a list of URLs), sorted by cluster size desc.
    Only returns clusters with 2+ members.
    """
    urls = url_df["url"].tolist()
    n = len(urls)

    # Build adjacency list
    adj: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                adj[i].append(j)
                adj[j].append(i)

    # BFS to find connected components
    visited = set()
    clusters = []

    for start in range(n):
        if start in visited:
            continue
        if not adj[start]:
            continue  # isolated node — not a cluster

        component = []
        queue = deque([start])
        visited.add(start)

        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(component) >= 2:
            clusters.append([urls[i] for i in component])

    # Sort by cluster size (largest first)
    clusters.sort(key=len, reverse=True)
    return clusters


def clusters_to_dataframe(
    clusters: list[list[str]],
    url_df: pd.DataFrame,
    sim_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Convert cluster list to a flat DataFrame for display.

    Columns: cluster_id, url, cluster_size, avg_similarity_in_cluster
    """
    urls = url_df["url"].tolist()
    url_to_idx = {u: i for i, u in enumerate(urls)}

    rows = []
    for cluster_id, cluster in enumerate(clusters, start=1):
        indices = [url_to_idx[u] for u in cluster if u in url_to_idx]

        # Average pairwise similarity within cluster
        if len(indices) >= 2:
            sims = [
                sim_matrix[i, j]
                for ii, i in enumerate(indices)
                for j in indices[ii + 1:]
            ]
            avg_sim = round(float(np.mean(sims)), 4)
        else:
            avg_sim = 1.0

        for url in cluster:
            rows.append({
                "cluster_id": cluster_id,
                "url": url,
                "cluster_size": len(cluster),
                "avg_similarity": avg_sim,
            })

    if not rows:
        return pd.DataFrame(columns=["cluster_id", "url", "cluster_size", "avg_similarity"])

    return pd.DataFrame(rows)


def get_cluster_for_url(url: str, clusters: list[list[str]]) -> Optional[list[str]]:
    """Return the cluster containing a given URL, or None."""
    for cluster in clusters:
        if url in cluster:
            return cluster
    return None


def compute_url_risk_summary(
    url_df: pd.DataFrame,
    sim_matrix: np.ndarray,
    threshold_high: float = 0.85,
    threshold_medium: float = 0.60,
) -> pd.DataFrame:
    """
    For each URL, compute:
    - max_similarity: highest similarity to any other URL
    - avg_similarity: average similarity to all other URLs
    - high_risk_count: number of URLs with similarity >= threshold_high
    - medium_risk_count: number of URLs with threshold_medium <= sim < threshold_high
    - risk_level: overall risk classification

    Returns DataFrame sorted by max_similarity desc.
    """
    urls = url_df["url"].tolist()
    n = len(urls)
    rows = []

    for i in range(n):
        row_sims = sim_matrix[i].copy()
        row_sims[i] = 0  # exclude self

        max_sim = float(np.max(row_sims))
        avg_sim = float(np.mean(row_sims))
        high_count = int(np.sum(row_sims >= threshold_high))
        medium_count = int(np.sum((row_sims >= threshold_medium) & (row_sims < threshold_high)))

        if max_sim >= threshold_high:
            risk = "High"
        elif max_sim >= threshold_medium:
            risk = "Medium"
        elif max_sim >= 0.40:
            risk = "Low"
        else:
            risk = "Minimal"

        rows.append({
            "url": urls[i],
            "max_similarity": round(max_sim, 4),
            "avg_similarity": round(avg_sim, 4),
            "high_risk_pairs": high_count,
            "medium_risk_pairs": medium_count,
            "risk_level": risk,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("max_similarity", ascending=False).reset_index(drop=True)
    return df
