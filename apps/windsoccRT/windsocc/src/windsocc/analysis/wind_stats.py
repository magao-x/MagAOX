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


def cluster_centroids_table(
    X: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Per-cluster mean/std in (vu, vv) feature space for labels >= 0.

    Rows are ordered by ``cluster_id``. ``X`` columns are ``[vu, vv]`` (same as
    ``flatten_vetted_tracks_to_features``).
    """
    if X.size == 0 or labels.size == 0:
        return []
    out: list[dict[str, Any]] = []
    for lab in sorted(x for x in np.unique(labels) if x >= 0):
        mask = labels == lab
        pts = X[mask]
        if pts.shape[0] == 0:
            continue
        vu = pts[:, 0]
        vv = pts[:, 1]
        mean_vu = float(np.mean(vu))
        std_vu = float(np.std(vu))
        mean_vv = float(np.mean(vv))
        std_vv = float(np.std(vv))
        speeds = np.hypot(vu, vv)
        mean_speed = float(np.hypot(mean_vu, mean_vv))
        std_speeds = float(np.std(speeds))
        mean_dir_deg = float((np.degrees(np.arctan2(mean_vv, mean_vu)) + 360.0) % 360.0)
        dir_deg_pts = (np.degrees(np.arctan2(vv, vu)) + 360.0) % 360.0
        std_directions = float(np.std(dir_deg_pts))
        out.append(
            {
                "cluster_id": int(lab),
                "n_points": int(pts.shape[0]),
                "mean_vu": mean_vu,
                "std_vu": std_vu,
                "mean_vv": mean_vv,
                "std_vv": std_vv,
                "mean_speed": mean_speed,
                "std_speeds": std_speeds,
                "mean_direction_deg": mean_dir_deg,
                "std_direction_deg": std_directions,
            }
        )
    return out


def cluster_membership_sigma_mask(
    vu: np.ndarray,
    vv: np.ndarray,
    centroid: dict[str, Any],
    sigma: float,
    *,
    std_floor: float = 1e-6,
) -> np.ndarray:
    """
    Boolean mask: rows within ``sigma`` (independent) of centroid in ``(vu, vv)``.

    Uses ``max(std_*, std_floor)`` so a degenerate zero std does not collapse the gate.
    """
    vu = np.asarray(vu, dtype=np.float64).ravel()
    vv = np.asarray(vv, dtype=np.float64).ravel()
    if vu.shape != vv.shape:
        raise ValueError("vu and vv must have the same shape.")
    su = max(float(centroid["std_vu"]), std_floor)
    sv = max(float(centroid["std_vv"]), std_floor)
    sig = float(sigma)
    return (np.abs(vu - float(centroid["mean_vu"])) <= sig * su) & (
        np.abs(vv - float(centroid["mean_vv"])) <= sig * sv
    )


def wind_cluster_stats_report_rows_from_centroids(
    centroids: list[dict[str, Any]],
) -> list[list[str]]:
    """Human-readable blocks for ``write_wind_cluster_stats_report``."""
    rows: list[list[str]] = []
    for c in centroids:
        lab = int(c["cluster_id"])
        rows.append(
            [
                f"Layer {lab}",
                f"n_points: {int(c['n_points'])}",
                f"Mean U-component speed: {float(c['mean_vu']):.2f}",
                f"Std U-component speed: {float(c['std_vu']):.2f}",
                f"Mean V-component speed: {float(c['mean_vv']):.2f}",
                f"Std V-component speed: {float(c['std_vv']):.2f}",
                rf"Layer speed: {float(c['mean_speed']):.2f} $\pm$ {float(c['std_speeds']):.2f} m/s",
                rf"Layer direction: {float(c['mean_direction_deg']):.2f} $\pm$ {float(c['std_direction_deg']):.2f} deg",
            ]
        )
    return rows


def per_cluster_vu_vv_stats(
    X: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[list[str]], int]:
    """
    Mean and std of vu, vv for each cluster label >= 0.

    Returns (report_row_blocks, noise_count) where noise_count is the number of
    points with label -1. Each block is a list of lines for one cluster.
    """
    if X.size == 0 or labels.size == 0:
        return [], 0
    noise_count = int(np.sum(labels == -1))
    centroids = cluster_centroids_table(X, labels)
    return wind_cluster_stats_report_rows_from_centroids(centroids), noise_count
