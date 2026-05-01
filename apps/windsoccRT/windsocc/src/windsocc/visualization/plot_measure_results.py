"""
TODO the extrapolated and filtered sep results movies
are showing inconsistencies.
"""

import sep
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import os
import warnings
from typing import Any

from astropy.io import fits
from matplotlib.lines import Line2D

from windsocc.visualization.ffmpeg_matplotlib import configure_matplotlib_ffmpeg_path
from windsocc.analysis.cross_correlation import compute_aperture_overlap

TRACKED_SOURCE_DTYPE = np.dtype(
    [
        ("track_id", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("a", np.float64),
        ("b", np.float64),
        ("theta", np.float64),
    ]
)
def build_detection_grid(
    x_coords: list[float],
    y_coords: list[float],
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=np.uint8)
    if not x_coords:
        return grid
    xs = np.clip(np.rint(x_coords).astype(int), 0, shape[1] - 1)
    ys = np.clip(np.rint(y_coords).astype(int), 0, shape[0] - 1)
    unique_coords = set(zip(ys.tolist(), xs.tolist()))
    for y, x in unique_coords:
        grid[y, x] = 1
    return grid

def _ellipse_mask(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    a: float,
    b: float,
    theta: float,
    scale: float = 3.0,
) -> np.ndarray:
    """Return a boolean mask for a scaled ellipse aperture."""
    if not np.isfinite([x0, y0, a, b, theta]).all():
        return np.zeros(shape, dtype=bool)
    a_pix = float(abs(a) * scale)
    b_pix = float(abs(b) * scale)
    if a_pix <= 0.0 or b_pix <= 0.0:
        return np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    cos_t = np.cos(float(theta))
    sin_t = np.sin(float(theta))
    x_shift = xx - float(x0)
    y_shift = yy - float(y0)
    x_rot = x_shift * cos_t + y_shift * sin_t
    y_rot = -x_shift * sin_t + y_shift * cos_t
    return (x_rot / a_pix) ** 2 + (y_rot / b_pix) ** 2 <= 1.0


def _circle_mask(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    radius: float,
) -> np.ndarray:
    """Return a boolean mask for a circular aperture."""
    if not np.isfinite([x0, y0, radius]).all():
        return np.zeros(shape, dtype=bool)
    r_pix = float(abs(radius))
    if r_pix <= 0.0:
        return np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - float(x0)) ** 2 + (yy - float(y0)) ** 2 <= r_pix ** 2


def _load_xcorr_aperture_response_curve(
    measure_root: str,
    diam_pupils: int | None = None,
    fft_pad_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load distance (pixels) vs aperture CC response from ``camwfs/response_curve.txt``.

    If the file is not present and enough geometry is provided, generate it via
    ``compute_aperture_overlap`` and save it in the main camwfs directory.
    """
    camwfs_root = os.path.dirname(os.path.abspath(measure_root))
    path = os.path.join(camwfs_root, "response_curve.txt")
    if not os.path.isfile(path):
        if diam_pupils is None:
            warnings.warn(
                f"No aperture response curve at {path} and DIAM_PUPILS is unavailable; "
                "CC extraction plots use raw sums.",
                stacklevel=2,
            )
            return None
        if fft_pad_shape is None:
            warnings.warn(
                f"No aperture response curve at {path} and fft_pad_shape is unavailable; "
                "CC extraction plots use raw sums.",
                stacklevel=2,
            )
            return None
        try:
            aperture_center = ((float(diam_pupils) - 1.0) / 2.0, (float(diam_pupils) - 1.0) / 2.0)
            response_curve = compute_aperture_overlap(
                int(diam_pupils),
                aperture_center,
                (int(fft_pad_shape[0]), int(fft_pad_shape[1])),
            )
            np.savetxt(path, response_curve, fmt="%d %.6f")
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Could not create aperture response curve at {path}: {exc}; "
                "CC extraction plots use raw sums.",
                stacklevel=2,
            )
            return None
    try:
        data = np.loadtxt(path)
    except OSError as exc:
        warnings.warn(
            f"Could not read {path}: {exc}; CC extraction plots use raw sums.",
            stacklevel=2,
        )
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 1 or data.shape[1] < 2:
        warnings.warn(
            f"Unexpected shape in {path}: {data.shape}; CC extraction plots use raw sums.",
            stacklevel=2,
        )
        return None
    d_raw = np.asarray(data[:, 0], dtype=np.float64)
    v_raw = np.asarray(data[:, 1], dtype=np.float64)
    order = np.argsort(d_raw)
    d_sorted = d_raw[order]
    v_sorted = v_raw[order]
    uniq_d, inv = np.unique(d_sorted, return_inverse=True)
    sum_v = np.bincount(inv, weights=v_sorted)
    cnt = np.bincount(inv)
    v_agg = sum_v / np.maximum(cnt.astype(np.float64), 1.0)
    return uniq_d.astype(np.float64), v_agg.astype(np.float64)


def _subtract_background_sep_per_frame(cube: np.ndarray) -> np.ndarray:
    """Return a copy of ``cube`` with SEP ``Background`` subtracted per 2D frame."""
    n = len(cube)
    out = np.empty((n,) + cube.shape[1:], dtype=np.float32)
    for i in range(n):
        sep_frame = np.ascontiguousarray(cube[i], dtype=np.float32)
        estimate_bkg = sep.Background(sep_frame, bw=32)
        out[i] = sep_frame - estimate_bkg
    return out


def plot_wind_track_clusters(
    vu: np.ndarray,
    vv: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    output_plot_fname: str,
    title: str | None = None,
) -> None:
    """Scatter vu vs vv with HDBSCAN clusters; noise drawn first in faint gray.

    Cluster points use per-point alpha from ``probabilities`` (clipped to stay visible).
    """
    vu = np.asarray(vu, dtype=np.float64).ravel()
    vv = np.asarray(vv, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    n = vu.shape[0]
    if n == 0:
        warnings.warn("plot_wind_track_clusters: no points; skipping figure.", stacklevel=2)
        return
    if vv.shape[0] != n or labels.shape[0] != n:
        raise ValueError("vu, vv, labels must have the same length.")
    if probs.shape[0] != n:
        raise ValueError("probabilities must match vu length.")

    fig, ax = plt.subplots(figsize=(7, 7))
    noise = labels == -1
    if np.any(noise):
        ax.scatter(
            vv[noise],
            vu[noise],
            c="lightgray",
            s=24,
            alpha=0.35,
            zorder=1,
            edgecolors="none",
        )

    cmap = plt.get_cmap("tab10")
    cluster_ids = sorted(x for x in np.unique(labels) if x >= 0)
    for idx, lab in enumerate(cluster_ids):
        mask = labels == lab
        if not np.any(mask):
            continue
        rgb = cmap(idx % 10)[:3]
        n_pts = int(np.sum(mask))
        rgba = np.zeros((n_pts, 4), dtype=np.float64)
        rgba[:, :3] = rgb
        rgba[:, 3] = np.clip(probs[mask], 0.12, 1.0)
        ax.scatter(
            vv[mask],
            vu[mask],
            c=rgba,
            s=36,
            zorder=2,
            edgecolors="none",
        )
        centroid_u = float(np.mean(vv[mask]))
        centroid_v = float(np.mean(vu[mask]))
        ax.scatter(
            [centroid_u],
            [centroid_v],
            s=720,
            marker="o",
            facecolors=[(*rgb, 0.36)],
            edgecolors=[(*rgb, 1.0)],
            linewidths=1.6,
            zorder=4,
        )
        ax.text(
            centroid_u,
            centroid_v,
            f"{int(lab)}",
            color="white",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            zorder=5,
        )

    ax.set_ylabel(r"$V$-component Speed (m/s)", fontsize=24)
    ax.set_xlabel(r"$U$-component Speed (m/s)", fontsize=24)
    ax.grid(True, linestyle="--", alpha=0.3)
    compass_origin = (0.88, 0.16)
    compass_delta = 0.065
    compass_style = {
        "arrowstyle": "-|>",
        "color": "0.2",
        "linewidth": 1.2,
        "shrinkA": 0.0,
        "shrinkB": 0.0,
    }
    for label, offset, alignment in (
        ("N", (0.0, compass_delta), ("center", "bottom")),
        ("E", (compass_delta, 0.0), ("left", "center")),
        ("W", (-compass_delta, 0.0), ("right", "center")),
        ("S", (0.0, -compass_delta), ("center", "top")),
    ):
        ax.annotate(
            "",
            xy=(compass_origin[0] + offset[0], compass_origin[1] + offset[1]),
            xytext=compass_origin,
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=compass_style,
            zorder=6,
        )
        ax.text(
            compass_origin[0] + 1.25 * offset[0],
            compass_origin[1] + 1.25 * offset[1],
            label,
            transform=ax.transAxes,
            color="0.2",
            fontsize=16,
            fontweight="bold",
            ha=alignment[0],
            va=alignment[1],
            zorder=7,
        )
    # if title:
    #     ax.set_title(title)
    # else:
    #     ax.set_title("Wind tracks in velocity space (HDBSCAN)")
    legend_elements: list[Line2D] = []
    # add the noise to the legend
    if np.any(noise):
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Noise",
                markerfacecolor="lightgray",
                markersize=12,
                alpha=0.5,
            )
        )
    if legend_elements:
        ax.legend(handles=legend_elements, loc="best", fontsize=16)

    # ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_plot_dir = os.path.dirname(os.path.abspath(output_plot_fname))
    if out_plot_dir:
        os.makedirs(out_plot_dir, exist_ok=True)
    fig.savefig(f"{output_plot_fname}.png", dpi=150)
    fig.savefig(f"{output_plot_fname}.pdf")
    plt.close(fig)


def write_wind_cluster_stats_report(
    path: str,
    rows: list[list[str]],
    noise_count: int,
) -> None:
    """Write cluster mean/std for vu and vv plus noise point count."""
    lines = [
        "Wind track clusters (HDBSCAN) — vu, vv velocity components (m/s)",
        f"Noise points (label -1): {noise_count}",
        "",
    ]
    for row in rows:
        for line in row:
            lines.append(line)
        lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_wind_direction_vs_time_for_cluster(
    df: pl.DataFrame,
    output_png: str,
    *,
    cluster_id: int,
    sigma: float,
    mean_speed_mps: float,
    mean_direction_deg: float,
) -> None:
    """Scatter of wind direction (deg) vs cube timestamp for rows in ``df``.

    Rows are expected to already be filtered (e.g. by a ``sigma`` gate in ``(vu, vv)``
    around the cluster centroid using per-cluster std from HDBSCAN feature members).
    """
    if df.is_empty():
        warnings.warn(
            f"plot_wind_direction_vs_time_for_cluster: no points for cluster {cluster_id}; skipping.",
            stacklevel=2,
        )
        return
    times = df["time"].to_list()
    directions = df["direction"].cast(pl.Float64).to_list()
    velocities = np.asarray(df["velocity_m_per_s"].cast(pl.Float64).to_list(), dtype=np.float64)
    # Normalize speeds by the cluster mean speed for per-point color encoding.
    if mean_speed_mps > 0:
        velocities_norm = velocities / float(mean_speed_mps)
    else:
        velocities_norm = np.ones_like(velocities, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vmin = float(np.nanmin(velocities_norm))
    vmax = float(np.nanmax(velocities_norm))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    speed_norm = plt.Normalize(vmin=vmin, vmax=vmax)
    scat = ax.scatter(
        times,
        directions,
        s=28,
        alpha=0.9,
        c=velocities_norm,
        cmap="Blues",
        norm=speed_norm,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel("Time Obs. (UTC)")
    ax.set_ylabel("Wind direction (deg)")
    ax.set_title(
        rf"Cluster {cluster_id}: direction vs time ($\sigma$={sigma:g}; "
        rf"$\langle v\rangle$={mean_speed_mps:.1f} m/s, $\langle\theta\rangle$={mean_direction_deg:.1f}°)"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    cbar = fig.colorbar(scat, ax=ax, pad=0.02)
    cbar.set_label(r"$v / \langle v \rangle$")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_png_dir = os.path.dirname(os.path.abspath(output_png))
    if out_png_dir:
        os.makedirs(out_png_dir, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def plot_flux_decay(
    og_cc_cube: np.ndarray | None,
    og_cc_cube_fname: str | None,
    extrapolated_sources_by_frame: list[np.ndarray] | None = None,
    center_x: float | None = None,
    center_y: float | None = None,
    output_dir: str | None = None,
    spatial_noise_map: np.ndarray | None = None,
    track_velocity_mps: dict[int, float] | None = None,
    track_direction_deg: dict[int, float] | None = None,
    diam_pupils: int | None = None,
) -> list[str]:
    """Plot OG extracted flux-vs-distance curves with spatial-noise uncertainty bands."""
    saved_paths: list[str] = []

    if (
        og_cc_cube is None
        or og_cc_cube_fname is None
        or extrapolated_sources_by_frame is None
        or center_x is None
        or center_y is None
        or output_dir is None
    ):
        return saved_paths

    og_cube = np.asarray(og_cc_cube)
    if og_cube.ndim != 3:
        warnings.warn(
            f"OG response cube {og_cc_cube_fname!r} is not 3D (shape {og_cube.shape}); "
            "skipping extracted_cc_responses plots.",
            stacklevel=2,
        )
        return saved_paths

    n_og = len(og_cube)
    n_list = len(extrapolated_sources_by_frame)
    n_frames = min(n_og, n_list)
    if n_frames < n_og or n_frames < n_list:
        warnings.warn(
            f"Truncating OG aperture analysis to {n_frames} frames "
            f"(OG={n_og}, extrapolated list={n_list}).",
            stacklevel=2,
        )

    og_cube = og_cube[:n_frames]
    sigma_map: np.ndarray | None = None
    if spatial_noise_map is not None:
        sigma_map = np.asarray(spatial_noise_map, dtype=np.float64)
        if sigma_map.ndim != 2 or sigma_map.shape != og_cube.shape[1:]:
            warnings.warn(
                f"Spatial noise map shape {getattr(sigma_map, 'shape', None)} does not match "
                f"OG cube frame shape {og_cube.shape[1:]}; ignoring uncertainties.",
                stacklevel=2,
            )
            sigma_map = None

    measure_root = os.path.dirname(output_dir)
    og_stem = os.path.splitext(og_cc_cube_fname)[0]
    out_dir = os.path.join(measure_root, "extracted_cc_responses", og_stem)
    os.makedirs(out_dir, exist_ok=True)
    # Drop stale per-track PNGs so reruns with new track IDs do not leave old files behind.
    for name in os.listdir(out_dir):
        if name.lower().endswith(".png"):
            try:
                os.remove(os.path.join(out_dir, name))
            except OSError:
                pass

    ref_curve = _load_xcorr_aperture_response_curve(
        measure_root,
        diam_pupils=diam_pupils,
        fft_pad_shape=(int(og_cube.shape[1]), int(og_cube.shape[2])),
    )

    # Omit noisy outer radii from plots (matches typical pupil scale ~60 px).
    max_plot_distance_px = diam_pupils * 0.9

    track_distances: dict[int, list[float]] = {}
    track_sums: dict[int, list[float]] = {}
    track_sigmas: dict[int, list[float]] = {}

    for frame_idx in range(n_frames):
        frame = og_cube[frame_idx]
        sources = extrapolated_sources_by_frame[frame_idx]
        if sources.size == 0:
            continue
        for source in sources:
            tid = int(source["track_id"])
            if tid < 0:
                continue
            x = float(source["x"])
            y = float(source["y"])
            dist = float(np.hypot(x - center_x, y - center_y))
            # Elliptical aperture path (temporarily disabled while debugging circular apertures):
            # aperture_mask = _ellipse_mask(
            #     shape=frame.shape,
            #     x0=x,
            #     y0=y,
            #     a=float(source["a"]) / 2.0,
            #     b=float(source["b"]) / 2.0,
            #     theta=float(source["theta"]),
            #     scale=1.0,
            # )
            semimajor_radius = max(float(source["a"]), float(source["b"])) / 2.0
            aperture_mask = _circle_mask(
                shape=frame.shape,
                x0=x,
                y0=y,
                radius=semimajor_radius,
            )
            if not np.any(aperture_mask):
                continue
            aperture_sum = float(np.nansum(frame[aperture_mask]))
            if sigma_map is not None:
                sigma_sum = float(np.sqrt(np.nansum(np.square(sigma_map[aperture_mask]))))
            else:
                sigma_sum = float("nan")
            track_distances.setdefault(tid, []).append(dist)
            track_sums.setdefault(tid, []).append(aperture_sum)
            track_sigmas.setdefault(tid, []).append(sigma_sum)

    y_axis_label = "Extracted flux sum (OG response)"
    for track_id in sorted(track_distances):
        dists = np.asarray(track_distances[track_id], dtype=np.float64)
        sums = np.asarray(track_sums.get(track_id, []), dtype=np.float64)
        sigma_sums = np.asarray(track_sigmas.get(track_id, []), dtype=np.float64)
        if dists.size == 0 or dists.shape != sums.shape:
            continue
        order = np.argsort(dists)
        dists = dists[order]
        sums = sums[order]
        if sigma_sums.size == sums.size:
            sigma_sums = sigma_sums[order]
        else:
            sigma_sums = np.full_like(sums, np.nan)

        keep = dists <= max_plot_distance_px
        if not np.any(keep):
            continue
        dists = dists[keep]
        sums = sums[keep]
        sigma_sums = sigma_sums[keep]

        if ref_curve is not None:
            xp, fp = ref_curve
            ref_at_dist = np.interp(dists, xp, fp)
            ref_pos = ref_at_dist[ref_at_dist > 0.0]
            fp_scale = float(np.nanmax(np.abs(ref_at_dist))) if ref_at_dist.size else 1.0
            eps = max(
                np.finfo(np.float64).eps * max(fp_scale, 1.0),
                float(np.nanmin(ref_pos)) if ref_pos.size else np.finfo(np.float64).eps,
            )
            denom = np.maximum(ref_at_dist, eps)
            y_plot = sums / denom
            y_sigma = sigma_sums / denom
            y_label = "Extracted flux sum / aperture CC response"
        else:
            y_plot = sums
            y_sigma = sigma_sums
            y_label = "Extracted flux sum (OG response)"
        y_axis_label = y_label
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dists, y_plot, linewidth=1.2, color="tab:blue")
        if np.isfinite(y_sigma).any():
            ax.fill_between(dists, y_plot - y_sigma, y_plot + y_sigma, alpha=0.2, color="tab:blue")
        ax.set_xlabel("Distance from origin (pixels)")
        ax.set_ylabel(y_label)
        vel = None if track_velocity_mps is None else track_velocity_mps.get(int(track_id))
        direction = None if track_direction_deg is None else track_direction_deg.get(int(track_id))
        if vel is None or not np.isfinite(vel):
            vel_txt = "v=NA m/s"
        else:
            vel_txt = f"v={float(vel):.2f} m/s"
        if direction is None or not np.isfinite(direction):
            dir_txt = "dir=NA deg"
        else:
            dir_txt = f"dir={float(direction):.1f} deg"
        ax.set_title(f"Track {track_id} ({vel_txt}, {dir_txt})")
        ax.set_xlim(-1.0, max_plot_distance_px)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(
            out_dir, f"track_{track_id}_cc_response_vs_distance.png"
        )
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        saved_paths.append(plot_path)

    # Also save one master plot with all tracks overlaid.
    if track_distances:
        fig, ax = plt.subplots(figsize=(10, 6))
        for track_id in sorted(track_distances):
            dists = np.asarray(track_distances[track_id], dtype=np.float64)
            sums = np.asarray(track_sums.get(track_id, []), dtype=np.float64)
            sigma_sums = np.asarray(track_sigmas.get(track_id, []), dtype=np.float64)
            if dists.size == 0 or dists.shape != sums.shape:
                continue
            order = np.argsort(dists)
            dists = dists[order]
            sums = sums[order]
            if sigma_sums.size == sums.size:
                sigma_sums = sigma_sums[order]
            else:
                sigma_sums = np.full_like(sums, np.nan)

            keep = dists <= max_plot_distance_px
            if not np.any(keep):
                continue
            dists = dists[keep]
            sums = sums[keep]
            sigma_sums = sigma_sums[keep]

            if ref_curve is not None:
                xp, fp = ref_curve
                ref_at_dist = np.interp(dists, xp, fp)
                ref_pos = ref_at_dist[ref_at_dist > 0.0]
                fp_scale = float(np.nanmax(np.abs(ref_at_dist))) if ref_at_dist.size else 1.0
                eps = max(
                    np.finfo(np.float64).eps * max(fp_scale, 1.0),
                    float(np.nanmin(ref_pos)) if ref_pos.size else np.finfo(np.float64).eps,
                )
                denom = np.maximum(ref_at_dist, eps)
                y_plot = sums / denom
                y_sigma = sigma_sums / denom
                y_axis_label = "Extracted flux sum / aperture CC response"
            else:
                y_plot = sums
                y_sigma = sigma_sums
                y_axis_label = "Extracted flux sum (OG response)"

            vel = None if track_velocity_mps is None else track_velocity_mps.get(int(track_id))
            direction = None if track_direction_deg is None else track_direction_deg.get(int(track_id))
            if vel is None or not np.isfinite(vel):
                vel_txt = "v=NA m/s"
            else:
                vel_txt = f"v={float(vel):.2f} m/s"
            if direction is None or not np.isfinite(direction):
                dir_txt = "dir=NA deg"
            else:
                dir_txt = f"dir={float(direction):.1f} deg"
            label = f"track {int(track_id)} ({vel_txt}, {dir_txt})"
            ax.plot(dists, y_plot, linewidth=2.0, alpha=0.65, label=label)
            if np.isfinite(y_sigma).any():
                ax.fill_between(dists, y_plot - y_sigma, y_plot + y_sigma, alpha=0.12)

        ax.set_xlabel("Distance from origin (pixels)")
        ax.set_ylabel(y_axis_label)
        ax.set_title(f"{og_stem}: All-track flux decay")
        ax.set_xlim(-1.0, max_plot_distance_px)
        ax.grid(True, linestyle="--", alpha=0.3)
        if len(track_distances) > 0:
            ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        master_path = os.path.join(out_dir, f"{og_stem}_all_tracks_flux_decay.png")
        fig.savefig(master_path, dpi=150)
        plt.close(fig)
        saved_paths.append(master_path)

    return saved_paths


def _empty_sources_array() -> np.ndarray:
    return np.zeros(0, dtype=TRACKED_SOURCE_DTYPE)


def tracked_df_to_frame_sources(
    cube_sources: object,
    n_frames: int,
) -> list[np.ndarray]:
    if isinstance(cube_sources, list):
        return cube_sources
    if not isinstance(cube_sources, pl.DataFrame) or cube_sources.is_empty():
        return [_empty_sources_array() for _ in range(n_frames)]

    tracked = cube_sources.clone()
    required_cols = {"frames", "x_coords", "y_coords", "a", "b", "theta"}
    if not required_cols.issubset(set(tracked.columns)):
        return [_empty_sources_array() for _ in range(n_frames)]

    cols: list[pl.Expr] = [
        pl.col("frames").cast(pl.Float64, strict=False),
        pl.col("x_coords").cast(pl.Float64, strict=False).alias("x"),
        pl.col("y_coords").cast(pl.Float64, strict=False).alias("y"),
        pl.col("a").cast(pl.Float64, strict=False),
        pl.col("b").cast(pl.Float64, strict=False),
        pl.col("theta").cast(pl.Float64, strict=False),
    ]
    if "track_id" in tracked.columns:
        cols.append(pl.col("track_id").cast(pl.Float64, strict=False))
    tracked = tracked.with_columns(cols).drop_nulls(
        subset=["frames", "x", "y", "a", "b", "theta"]
    )
    tracked = tracked.filter(
        (pl.col("frames") >= 0) & (pl.col("frames") < float(n_frames))
    )

    if "track_id" in tracked.columns:
        tracked = tracked.sort("frames").unique(
            subset=["frames", "track_id"], keep="last"
        )

    grouped: dict[int, pl.DataFrame] = {}
    for frame_val in tracked["frames"].unique().sort().to_list():
        if frame_val is None or (isinstance(frame_val, float) and np.isnan(frame_val)):
            continue
        fi = int(frame_val)
        grouped[fi] = tracked.filter(pl.col("frames") == frame_val)

    frame_sources = []
    for frame_idx in range(n_frames):
        if frame_idx not in grouped:
            frame_sources.append(_empty_sources_array())
            continue
        frame_df = grouped[frame_idx]
        arr = np.zeros(len(frame_df), dtype=TRACKED_SOURCE_DTYPE)
        if "track_id" in frame_df.columns:
            arr["track_id"] = (
                frame_df["track_id"].fill_null(-1).cast(pl.Int64).to_numpy()
            )
        else:
            arr["track_id"] = -1
        arr["x"] = frame_df["x"].to_numpy()
        arr["y"] = frame_df["y"].to_numpy()
        arr["a"] = frame_df["a"].to_numpy() * 3
        arr["b"] = frame_df["b"].to_numpy() * 3
        arr["theta"] = frame_df["theta"].to_numpy()
        frame_sources.append(arr)
    return frame_sources


def tracked_df_to_origin_propagated_sources(
    cube_sources: object,
    n_frames: int,
    center_x: float,
    center_y: float,
) -> list[np.ndarray]:
    if not isinstance(cube_sources, pl.DataFrame) or cube_sources.is_empty():
        return [_empty_sources_array() for _ in range(n_frames)]

    tracked = cube_sources.clone()
    required_cols = {
        "track_id",
        "frames",
        "a",
        "b",
        "theta",
        "direction",
        "velocity_px_per_frame",
    }
    if not required_cols.issubset(set(tracked.columns)):
        return [_empty_sources_array() for _ in range(n_frames)]

    tracked = tracked.with_columns(
        pl.col("track_id").cast(pl.Float64, strict=False),
        pl.col("frames").cast(pl.Float64, strict=False),
        pl.col("a").cast(pl.Float64, strict=False),
        pl.col("b").cast(pl.Float64, strict=False),
        pl.col("theta").cast(pl.Float64, strict=False),
        pl.col("direction").cast(pl.Float64, strict=False),
        pl.col("velocity_px_per_frame").cast(pl.Float64, strict=False),
    ).drop_nulls(
        subset=[
            "track_id",
            "frames",
            "a",
            "b",
            "theta",
            "direction",
            "velocity_px_per_frame",
        ]
    )
    if tracked.is_empty():
        return [_empty_sources_array() for _ in range(n_frames)]

    per_frame: list[list[tuple[int, float, float, float, float, float]]] = [
        [] for _ in range(n_frames)
    ]
    for track_id in tracked["track_id"].unique().sort().to_list():
        if track_id is None or (
            isinstance(track_id, float) and np.isnan(track_id)
        ):
            continue
        tid = int(track_id)
        group = tracked.filter(pl.col("track_id") == track_id)
        mean_a = float(np.mean(group["a"].to_numpy())) * 5
        mean_b = float(np.mean(group["b"].to_numpy())) * 5
        mean_theta = float(np.mean(group["theta"].to_numpy()))
        last_one = group.sort("frames").tail(1)
        direction_rad = np.deg2rad(float(last_one["direction"][0]))
        speed_px = float(last_one["velocity_px_per_frame"][0])
        dx_per_frame = speed_px * np.cos(direction_rad + np.pi / 2.0)
        dy_per_frame = speed_px * np.sin(direction_rad + np.pi / 2.0)
        for frame_idx in range(n_frames):
            x_here = center_x + dx_per_frame * frame_idx
            y_here = center_y + dy_per_frame * frame_idx
            per_frame[frame_idx].append(
                (tid, x_here, y_here, mean_a, mean_b, mean_theta)
            )

    frame_sources = []
    for frame_idx in range(n_frames):
        entries = per_frame[frame_idx]
        arr = np.zeros(len(entries), dtype=TRACKED_SOURCE_DTYPE)
        for i, (tid, x_here, y_here, a_val, b_val, theta_val) in enumerate(
            entries
        ):
            arr["track_id"][i] = tid
            arr["x"][i] = x_here
            arr["y"][i] = y_here
            arr["a"][i] = a_val
            arr["b"][i] = b_val
            arr["theta"][i] = theta_val
        frame_sources.append(arr)
    return frame_sources


def make_source_detection_movie(
    mf_response_cube_path: str,
    mf_response_cube_fname: str,
    sources_all: list,
    output_dir: str,
    fps: int = 10,
    cmap: str = "Blues_r",
    png_only: bool = False,
    og_cc_cube_path: str | None = None,
    og_cc_cube_fname: str | None = None,
    spatial_noise_map: np.ndarray | None = None,
    diam_pupils: int | None = None,
) -> bool:
    """Make a movie of the source detections.

    When ``og_cc_cube_path`` and ``og_cc_cube_fname`` are set, the OG
    cube is SEP background-subtracted once per frame; that cube is used for the
    ``{og_stem}_extrapolated.mp4`` (or PNG frames) and per-track extracted
    CC response plots under
    ``<measure_results>/extracted_cc_responses/<og_cube_stem>/``.
    """
    detections_dir = os.path.join(os.path.dirname(output_dir), "sep_detections")
    os.makedirs(detections_dir, exist_ok=True)
    cube_sources = sources_all[-1]
    track_velocity_mps: dict[int, float] = {}
    track_direction_deg: dict[int, float] = {}
    if isinstance(cube_sources, pl.DataFrame) and not cube_sources.is_empty():
        if {"track_id", "velocity_m_per_s"}.issubset(set(cube_sources.columns)):
            vel_df = (
                cube_sources.with_columns(
                    pl.col("track_id").cast(pl.Int64, strict=False).alias("track_id_i64"),
                    pl.col("velocity_m_per_s").cast(pl.Float64, strict=False).alias("velocity_mps_f64"),
                )
                .drop_nulls(subset=["track_id_i64", "velocity_mps_f64"])
                .group_by("track_id_i64")
                .agg(pl.col("velocity_mps_f64").mean().alias("velocity_mps"))
            )
            for row in vel_df.iter_rows(named=True):
                track_velocity_mps[int(row["track_id_i64"])] = float(row["velocity_mps"])
        if {"track_id", "direction"}.issubset(set(cube_sources.columns)):
            dir_df = cube_sources.with_columns(
                pl.col("track_id").cast(pl.Int64, strict=False).alias("track_id_i64"),
                pl.col("direction").cast(pl.Float64, strict=False).alias("direction_deg_f64"),
            ).drop_nulls(subset=["track_id_i64", "direction_deg_f64"])
            for tid in dir_df["track_id_i64"].unique().to_list():
                group = dir_df.filter(pl.col("track_id_i64") == int(tid))
                if group.is_empty():
                    continue
                vals = group["direction_deg_f64"].to_numpy().astype(np.float64)
                mean_sin = np.mean(np.sin(np.radians(vals)))
                mean_cos = np.mean(np.cos(np.radians(vals)))
                mean_dir = float(np.degrees(np.arctan2(mean_sin, mean_cos))) % 360.0
                track_direction_deg[int(tid)] = mean_dir
    mf_response_cube = fits.getdata(mf_response_cube_path) #3D array of shape (n_frames, y_size, x_size)
    center_x = mf_response_cube.shape[2] / 2.0 - 0.5
    center_y = mf_response_cube.shape[1] / 2.0 - 0.5

    extrapolated_sources_by_frame = tracked_df_to_origin_propagated_sources(
        cube_sources=cube_sources,
        n_frames=len(mf_response_cube),
        center_x=center_x,
        center_y=center_y,
    )
    
    sep_sources_by_frame = tracked_df_to_frame_sources(
        cube_sources=cube_sources,
        n_frames=len(mf_response_cube),
    )
    response_stack = np.stack(mf_response_cube)
    vmin, vmax = np.percentile(response_stack, [0.1, 99.9])
    cube_stem = os.path.splitext(mf_response_cube_fname)[0]

    og_cube_bgsub: np.ndarray | None = None
    extrap_for_og: list[np.ndarray] | None = None
    if og_cc_cube_path is not None and og_cc_cube_fname is not None:
        og_cube_raw = np.asarray(fits.getdata(og_cc_cube_path))
        if og_cube_raw.ndim == 3 and og_cube_raw.shape[1:] == mf_response_cube.shape[1:]:
            n_mf_og = len(mf_response_cube)
            n_og_raw = len(og_cube_raw)
            n_ext_og = len(extrapolated_sources_by_frame)
            n_sync_og = min(n_mf_og, n_og_raw, n_ext_og)
            og_cube_sync = og_cube_raw[:n_sync_og]
            og_cube_bgsub = _subtract_background_sep_per_frame(og_cube_sync)
            extrap_for_og = extrapolated_sources_by_frame[:n_sync_og]
        else:
            warnings.warn(
                f"Skipping OG background subtraction and OG visuals: shape "
                f"{getattr(og_cube_raw, 'shape', None)} vs MF spatial "
                f"{mf_response_cube.shape[1:]}.",
                stacklevel=2,
            )

    def _render_movie_funcanimation(
        sources_by_frame: list[np.ndarray],
        movie_name: str,
        cube_data: np.ndarray,
        cube_fname: str,
        vmin_c: float,
        vmax_c: float,
        use_circular_apertures: bool = False,
    ) -> None:
        fig, ax = plt.subplots()
        image = ax.imshow(
            cube_data[0], origin="lower", cmap=cmap, vmin=vmin_c, vmax=vmax_c
        )
        title = ax.set_title("Source Extractor Detections")
        ax.set_xlabel("X pixels")
        ax.set_ylabel("Y pixels")
        fig.tight_layout()

        current_overlay: list[Artist] = []

        def _clear_overlay() -> None:
            for artist in list(current_overlay):
                artist.remove()
            current_overlay.clear()

        def _add_overlay(sources: np.ndarray) -> None:
            for source in sources:
                if use_circular_apertures:
                    circle_diam = float(max(source["a"], source["b"]))
                    aperture_artist = Ellipse(
                        (source["x"], source["y"]),
                        width=circle_diam,
                        height=circle_diam,
                        angle=0.0,
                        fill=False,
                        edgecolor="red",
                        linewidth=1.5,
                    )
                    r = 0.5 * circle_diam
                else:
                    aperture_artist = Ellipse(
                        (source["x"], source["y"]),
                        width=source["a"],
                        height=source["b"],
                        angle=np.degrees(source["theta"]),
                        fill=False,
                        edgecolor="red",
                        linewidth=1.5,
                    )
                    r = float(max(source["a"], source["b"], 2.0))
                ax.add_patch(aperture_artist)
                current_overlay.append(aperture_artist)
                tid = int(source["track_id"])
                if tid >= 0:
                    r = max(r, 2.0)
                    tx = float(source["x"]) + 0.35 * r
                    ty = float(source["y"]) + 0.35 * r
                    label = ax.text(
                        tx,
                        ty,
                        str(tid),
                        color="red",
                        fontsize=8,
                        fontweight="bold",
                        ha="left",
                        va="bottom",
                        clip_on=True,
                    )
                    current_overlay.append(label)

        def _update(frame_idx: int):
            image.set_data(cube_data[frame_idx])
            _clear_overlay()
            _add_overlay(sources_by_frame[frame_idx])
            split_fname = cube_fname.split("_")
            timestamp = split_fname[1] if len(split_fname) > 1 else cube_fname
            title.set_text(f"{timestamp}; Slice {frame_idx} of {len(cube_data)}")
            return [image, title, *current_overlay]

        anim = FuncAnimation(
            fig,
            _update,
            frames=len(cube_data),
            interval=1000.0 / max(fps, 1e-3),
            blit=False,
        )
        configure_matplotlib_ffmpeg_path()
        writer = FFMpegWriter(fps=fps)
        final_path = os.path.join(output_dir, movie_name)
        anim.save(final_path, writer=writer)
        plt.close(fig)

    def _render_movie_png_frames(
        sources_by_frame: list[np.ndarray],
        movie_name: str,
        cube_data: np.ndarray,
        cube_fname: str,
        vmin_c: float,
        vmax_c: float,
        ellipse_scale: float = 3.0,
        use_circular_apertures: bool = False,
    ) -> None:
        """Render individual PNG frames instead of an MP4 movie."""
        measure_results_dir = os.path.dirname(output_dir)
        pngs_base_dir = os.path.join(measure_results_dir, "pngs")
        os.makedirs(pngs_base_dir, exist_ok=True)

        movie_stem, _ = os.path.splitext(movie_name)
        movie_png_dir = os.path.join(pngs_base_dir, movie_stem)
        os.makedirs(movie_png_dir, exist_ok=True)

        n_frames = len(cube_data)
        for frame_idx in range(n_frames):
            fig, ax = plt.subplots()
            image = ax.imshow(
                cube_data[frame_idx],
                origin="lower",
                cmap=cmap,
                vmin=vmin_c,
                vmax=vmax_c,
            )
            title = ax.set_title("Source Extractor Detections")
            ax.set_xlabel("X pixels")
            ax.set_ylabel("Y pixels")
            fig.tight_layout()

            # Draw overlays for this frame (mirrors _add_overlay logic).
            sources = sources_by_frame[frame_idx]
            current_overlay: list[Artist] = []
            for source in sources:
                if use_circular_apertures:
                    circle_diam = ellipse_scale * float(max(source["a"], source["b"]))
                    aperture_artist = Ellipse(
                        (source["x"], source["y"]),
                        width=circle_diam,
                        height=circle_diam,
                        angle=0.0,
                        fill=False,
                        edgecolor="red",
                        linewidth=1.5,
                    )
                    r = 0.5 * circle_diam
                else:
                    aperture_artist = Ellipse(
                        (source["x"], source["y"]),
                        width=ellipse_scale * source["a"],
                        height=ellipse_scale * source["b"],
                        angle=np.degrees(source["theta"]),
                        fill=False,
                        edgecolor="red",
                        linewidth=1.5,
                    )
                    r = float(
                        max(
                            ellipse_scale * source["a"],
                            ellipse_scale * source["b"],
                            2.0,
                        )
                    )
                ax.add_patch(aperture_artist)
                current_overlay.append(aperture_artist)
                tid = int(source["track_id"])
                if tid >= 0:
                    r = max(r, 2.0)
                    tx = float(source["x"]) + 0.35 * r
                    ty = float(source["y"]) + 0.35 * r
                    label = ax.text(
                        tx,
                        ty,
                        str(tid),
                        color="red",
                        fontsize=8,
                        fontweight="bold",
                        ha="left",
                        va="bottom",
                        clip_on=True,
                    )
                    current_overlay.append(label)

            split_fname = cube_fname.split("_")
            timestamp = split_fname[1] if len(split_fname) > 1 else cube_fname
            title.set_text(f"{timestamp}; Slice {frame_idx} of {len(cube_data)}")

            frame_name = f"frame_{frame_idx:04d}.png"
            frame_path = os.path.join(movie_png_dir, frame_name)
            fig.savefig(frame_path, dpi=150)
            plt.close(fig)

    if png_only:
        _render_movie_png_frames(
            extrapolated_sources_by_frame,
            f"{cube_stem}_extrapolated.mp4",
            mf_response_cube,
            mf_response_cube_fname,
            vmin,
            vmax,
            ellipse_scale=3.0,
        )
        _render_movie_png_frames(
            sep_sources_by_frame,
            f"{cube_stem}_sep_results.mp4",
            mf_response_cube,
            mf_response_cube_fname,
            vmin,
            vmax,
            ellipse_scale=3.0,
        )
    else:
        _render_movie_funcanimation(
            extrapolated_sources_by_frame,
            f"{cube_stem}_extrapolated.mp4",
            mf_response_cube,
            mf_response_cube_fname,
            vmin,
            vmax,
        )
        _render_movie_funcanimation(
            sep_sources_by_frame,
            f"{cube_stem}_sep_results.mp4",
            mf_response_cube,
            mf_response_cube_fname,
            vmin,
            vmax,
        )

    if og_cube_bgsub is not None and extrap_for_og is not None:
        og_stack = np.stack(og_cube_bgsub)
        og_vmin, og_vmax = np.percentile(og_stack, [0.1, 99.9])
        og_stem = os.path.splitext(og_cc_cube_fname)[0]
        og_movie_name = f"{og_stem}_extrapolated.mp4"
        if png_only:
            _render_movie_png_frames(
                extrap_for_og,
                og_movie_name,
                og_cube_bgsub,
                og_cc_cube_fname,
                og_vmin,
                og_vmax,
                ellipse_scale=1.0,
                use_circular_apertures=True,
            )
        else:
            _render_movie_funcanimation(
                extrap_for_og,
                og_movie_name,
                og_cube_bgsub,
                og_cc_cube_fname,
                og_vmin,
                og_vmax,
                use_circular_apertures=True,
            )

    plot_flux_decay(
        og_cc_cube=og_cube_bgsub,
        og_cc_cube_fname=og_cc_cube_fname,
        extrapolated_sources_by_frame=extrapolated_sources_by_frame,
        center_x=center_x,
        center_y=center_y,
        output_dir=output_dir,
        spatial_noise_map=spatial_noise_map,
        track_velocity_mps=track_velocity_mps,
        track_direction_deg=track_direction_deg,
        diam_pupils=diam_pupils,
    )
    return True