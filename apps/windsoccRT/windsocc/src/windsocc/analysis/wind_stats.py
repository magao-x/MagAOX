"""
Wind statistics: cluster vetted wind tracks in (vu, vv) velocity space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN


def flatten_vetted_tracks_to_features(wind_summaries: list[dict[str, Any]]) -> np.ndarray:
    """
    Collect all vetted tracks from per-cube summaries into (N, 2) feature rows [vu, vv].

    Direction is degrees in the camwfs frame; components use radians:
    vu = v * cos(theta), vv = v * sin(theta).
    """
    rows: list[list[float]] = []
    for summary in wind_summaries:
        for track in summary.get("tracks", []):
            try:
                v = float(track["velocity_m_per_s"])
                direction_deg = float(track["direction"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(v) or not np.isfinite(direction_deg):
                continue
            theta_rad = np.deg2rad(direction_deg)
            vu = float(v * np.cos(theta_rad))
            vv = float(v * np.sin(theta_rad))
            rows.append([vu, vv])
    if not rows:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def cluster_wind_tracks_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 3,
    min_samples: int = 3,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, HDBSCAN | None]:
    """
    Run HDBSCAN on (vu, vv) features.

    Returns (labels, probabilities, model). If X is empty, returns empty arrays and None.
    """
    if X.size == 0 or X.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), None
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        **kwargs,
    )
    model.fit(X)
    labels = np.asarray(model.labels_, dtype=np.int64)
    probs = np.asarray(model.probabilities_, dtype=np.float64)
    return labels, probs, model


def per_cluster_vu_vv_stats(
    X: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[dict[str, Any]], int]:
    """
    Mean and std of vu, vv for each cluster label >= 0.

    Returns (rows, noise_count) where noise_count is the number of points with label -1.
    """
    if X.size == 0 or labels.size == 0:
        return [], 0
    noise_count = int(np.sum(labels == -1))
    rows: list[dict[str, Any]] = []
    for lab in sorted(set(labels.tolist())):
        if lab < 0:
            continue
        mask = labels == lab
        pts = X[mask]
        if pts.shape[0] == 0:
            continue
        vu = pts[:, 0]
        vv = pts[:, 1]
        rows.append(
            {
                "cluster_id": int(lab),
                "n_points": int(pts.shape[0]),
                "mean_vu": float(np.mean(vu)),
                "std_vu": float(np.std(vu, ddof=0)),
                "mean_vv": float(np.mean(vv)),
                "std_vv": float(np.std(vv, ddof=0)),
            }
        )
    return rows, noise_count
