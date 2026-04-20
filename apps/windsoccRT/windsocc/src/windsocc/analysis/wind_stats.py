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
    features: list[list[float]] = []
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
            features.append([vu, vv])
    if not features:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(features, dtype=np.float64)


def cluster_wind_tracks_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int = 3,
    cluster_selection_epsilon: float = 0.0,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, HDBSCAN | None]:
    """
    Run HDBSCAN on (vu, vv) features.

    Returns (labels, probabilities, model). If X is empty, returns empty arrays and None.
    """
    if X.size == 0 or X.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), None
    # HDBSCAN's Cython layer expects Python scalars; numpy scalar types from YAML/config
    # can raise: TypeError: only 0-dimensional arrays can be converted to Python scalars.
    mcs = int(np.asarray(min_cluster_size).item())
    eps = float(np.asarray(cluster_selection_epsilon).item())
    X_fit = np.ascontiguousarray(np.asarray(X, dtype=np.float64), dtype=np.float64)
    hdb_kwargs = dict(kwargs)
    if "min_samples" in hdb_kwargs and hdb_kwargs["min_samples"] is not None:
        hdb_kwargs["min_samples"] = int(np.asarray(hdb_kwargs["min_samples"]).item())
    hdb_kwargs.setdefault("copy", False)
    # sklearn's HDBSCAN: any cluster_selection_epsilon > 0 runs epsilon_search in Cython
    # (_tree.pyx traverse_upwards). On Python 3.13 + current sklearn, that path can raise
    # TypeError: only 0-dimensional arrays can be converted to Python scalars. Epsilon=0
    # skips that branch and uses plain EOM cluster selection.
    model = HDBSCAN(
        cluster_selection_epsilon=eps,
        min_cluster_size=mcs,
        **hdb_kwargs,
    )
    model.fit(X_fit)
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
    rows: list[str] = []
    for lab in sorted(set(labels.tolist())):
        if lab < 0:
            continue
        mask = labels == lab
        pts = X[mask]
        if pts.shape[0] == 0:
            continue
        vv = pts[:, 0]
        vu = pts[:, 1]
        speeds = np.sqrt(vu**2 + vv**2)
        # directions = np.arctan2(vv, vu)
        directions = np.arctan2(vu, vv)
        std_speeds = np.std(speeds)
        std_directions = np.std(directions)
        mean_vu = float(np.mean(vu))
        std_vu = float(np.std(vu))
        mean_vv = float(np.mean(vv))
        std_vv = float(np.std(vv))
        mean_speed = np.sqrt(mean_vu**2 + mean_vv**2)
        mean_dir_rad = np.mean(directions)
        mean_dir_deg = np.degrees(mean_dir_rad)
        mean_dir_deg = (mean_dir_deg + 360.0) % 360.0
        rows.append(
            [
                f"Layer {int(lab)}",
                f"n_points: {int(pts.shape[0])}",
                f"Mean U-component speed: {mean_vu:.2f}",
                f"Std U-component speed: {std_vu:.2f}",
                f"Mean V-component speed: {mean_vv:.2f}",
                f"Std V-component speed: {std_vv:.2f}",
                rf"Layer speed: {float(mean_speed):.2f} $\pm$ {float(std_speeds):.2f} m/s",
                rf"Layer direction: {float(mean_dir_deg):.2f} $\pm$ {float(std_directions):.2f} deg",
            ]
        )
    return rows, noise_count
