"""
TODO the extrapolated and filtered sep results movies
are showing inconsistencies.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import os
from astropy.io import fits

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


def plot_flux_decay(
    mf_response_cube: np.ndarray,
    sep_sources_by_frame: list[np.ndarray],
    cube_stem: str,
    decay_dir: str,
) -> str | None:
    """Plot normalized aperture-summed flux curves per track across frames."""
    track_fluxes: dict[int, list[float]] = {}
    track_frames: dict[int, list[int]] = {}
    n_frames = min(len(mf_response_cube), len(sep_sources_by_frame))
    for frame_idx in range(n_frames):
        frame = mf_response_cube[frame_idx]
        sources = sep_sources_by_frame[frame_idx]
        if sources.size == 0:
            continue
        for source in sources:
            track_id = int(source["track_id"])
            aperture_mask = _ellipse_mask(
                shape=frame.shape,
                x0=float(source["x"]),
                y0=float(source["y"]),
                a=float(source["a"]),
                b=float(source["b"]),
                theta=float(source["theta"]),
                scale=3.0,
            )
            if not np.any(aperture_mask):
                continue
            extracted_flux = float(np.nansum(frame[aperture_mask]))
            track_fluxes.setdefault(track_id, []).append(extracted_flux)
            track_frames.setdefault(track_id, []).append(frame_idx)

    if not track_fluxes:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for track_id in sorted(track_fluxes):
        flux_values = np.asarray(track_fluxes[track_id], dtype=np.float64)
        frame_values = np.asarray(track_frames[track_id], dtype=np.int64)
        if flux_values.size == 0:
            continue
        max_flux = float(np.nanmax(flux_values))
        if np.isfinite(max_flux) and max_flux > 0.0:
            normalized = flux_values / max_flux
        else:
            normalized = np.zeros_like(flux_values)
        ax.plot(frame_values, normalized, linewidth=1.2, alpha=0.75, label=f"track {track_id}")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized Extracted Flux")
    ax.set_title(f"{cube_stem}: Normalized Flux Decay by Track")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    # Avoid an unreadable legend when many tracks are present.
    if 0 < len(track_fluxes) <= 15:
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    decay_plot_path = os.path.join(decay_dir, f"{cube_stem}_flux_decay.png")
    fig.savefig(decay_plot_path, dpi=150)
    plt.close(fig)
    return decay_plot_path


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
        arr["a"] = frame_df["a"].to_numpy()
        arr["b"] = frame_df["b"].to_numpy()
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
        mean_a = float(np.mean(group["a"].to_numpy()))
        mean_b = float(np.mean(group["b"].to_numpy()))
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
) -> bool:
    """Make a movie of the source detections."""
    detections_dir = os.path.join(os.path.dirname(output_dir), "sep_detections")
    decay_dir = os.path.join(os.path.dirname(output_dir), "decay_plots")
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(decay_dir, exist_ok=True)
    cube_sources = sources_all[-1]
    mf_response_cube = fits.getdata(mf_response_cube_path) #3D array of shape (n_frames, y_size, x_size)
    ny, nx = mf_response_cube.shape[1:]
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

    def _render_movie_funcanimation(
        sources_by_frame: list[np.ndarray], movie_name: str
    ) -> None:
        fig, ax = plt.subplots()
        image = ax.imshow(
            mf_response_cube[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
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
                ellipse = Ellipse(
                    (source["x"], source["y"]),
                    width=3.0 * source["a"],
                    height=3.0 * source["b"],
                    angle=np.degrees(source["theta"]),
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
                ax.add_patch(ellipse)
                current_overlay.append(ellipse)
                tid = int(source["track_id"])
                if tid >= 0:
                    r = float(
                        max(3.0 * source["a"], 3.0 * source["b"], 2.0)
                    )
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
            image.set_data(mf_response_cube[frame_idx])
            _clear_overlay()
            _add_overlay(sources_by_frame[frame_idx])
            split_fname = mf_response_cube_fname.split("_")
            timestamp = split_fname[1]
            title.set_text(f"{timestamp}; Slice {frame_idx} of {len(mf_response_cube)}")
            return [image, title, *current_overlay]

        anim = FuncAnimation(
            fig,
            _update,
            frames=len(mf_response_cube),
            interval=1000.0 / max(fps, 1e-3),
            blit=False,
        )
        writer = FFMpegWriter(fps=fps)
        final_path = os.path.join(output_dir, movie_name)
        anim.save(final_path, writer=writer)
        plt.close(fig)

    def _render_movie_png_frames(
        sources_by_frame: list[np.ndarray], movie_name: str
    ) -> None:
        """Render individual PNG frames instead of an MP4 movie."""
        measure_results_dir = os.path.dirname(output_dir)
        pngs_base_dir = os.path.join(measure_results_dir, "pngs")
        os.makedirs(pngs_base_dir, exist_ok=True)

        movie_stem, _ = os.path.splitext(movie_name)
        movie_png_dir = os.path.join(pngs_base_dir, movie_stem)
        os.makedirs(movie_png_dir, exist_ok=True)

        n_frames = len(mf_response_cube)
        for frame_idx in range(n_frames):
            fig, ax = plt.subplots()
            image = ax.imshow(
                mf_response_cube[frame_idx],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            title = ax.set_title("Source Extractor Detections")
            ax.set_xlabel("X pixels")
            ax.set_ylabel("Y pixels")
            fig.tight_layout()

            # Draw overlays for this frame (mirrors _add_overlay logic).
            sources = sources_by_frame[frame_idx]
            current_overlay: list[Artist] = []
            for source in sources:
                ellipse = Ellipse(
                    (source["x"], source["y"]),
                    width=3.0 * source["a"],
                    height=3.0 * source["b"],
                    angle=np.degrees(source["theta"]),
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
                ax.add_patch(ellipse)
                current_overlay.append(ellipse)
                tid = int(source["track_id"])
                if tid >= 0:
                    r = float(max(3.0 * source["a"], 3.0 * source["b"], 2.0))
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

            split_fname = mf_response_cube_fname.split("_")
            timestamp = split_fname[1] if len(split_fname) > 1 else mf_response_cube_fname
            title.set_text(f"{timestamp}; Slice {frame_idx} of {len(mf_response_cube)}")

            frame_name = f"frame_{frame_idx:04d}.png"
            frame_path = os.path.join(movie_png_dir, frame_name)
            fig.savefig(frame_path, dpi=150)
            plt.close(fig)

    if png_only:
        _render_movie_png_frames(
            extrapolated_sources_by_frame,
            f"{cube_stem}_extrapolated.mp4",
        )
        _render_movie_png_frames(
            sep_sources_by_frame,
            f"{cube_stem}_sep_results.mp4",
        )
    else:
        _render_movie_funcanimation(
            extrapolated_sources_by_frame,
            f"{cube_stem}_extrapolated.mp4",
        )
        _render_movie_funcanimation(
            sep_sources_by_frame,
            f"{cube_stem}_sep_results.mp4",
        )
    plot_flux_decay(
        mf_response_cube=mf_response_cube,
        sep_sources_by_frame=sep_sources_by_frame,
        cube_stem=cube_stem,
        decay_dir=decay_dir,
    )

        # x_coords = []
        # y_coords = []
        # for frame_idx, sources in enumerate(sep_sources_by_frame):
        #     for source in sources:
        #         x_coords.append(source["x"])
        #         y_coords.append(source["y"])

        # plot_stem = f"{cube_stem}_mf_response_sep_results"
        # if x_coords:
        #     fig_xy, ax_xy = plt.subplots()
        #     ax_xy.scatter(
        #         x_coords, y_coords,
        #         s=12, alpha=0.15,
        #         color="k",
        #         marker="o",
        #     )
        #     ax_xy.grid(True, linestyle="--", alpha=0.33)
        #     # ax_xy.set_facecolor("linen")
        #     ax_xy.set_xlabel("X-coords")
        #     ax_xy.set_ylabel("Y-coords")
        #     ax_xy.set_title("ALL SExtractor src coords")
        #     ax_xy.set_aspect("equal", adjustable="box")
        #     fig_xy.tight_layout()
        #     fig_xy.savefig(os.path.join(detections_dir, f"{plot_stem}_xy.png"))
        #     plt.close(fig_xy)

        # radius = 0.5 * min(nx, ny)
        # grid = build_detection_grid(x_coords, y_coords, (ny, nx))
        # angles, counts = line_sweep_histogram(
        #     grid,
        #     center_x=center_x,
        #     center_y=center_y,
        #     radius=radius,
        #     max_dist=3.0,
        #     bin_step_deg=1.0,
        # )
        # counts = counts - np.mean(counts)
        # counts = np.clip(counts, 0.0, None)
        # theta = np.deg2rad(angles)
        # theta = np.append(theta, theta[0])
        # r_vals = np.append(counts, counts[0])
        # fig_hist, ax_hist = plt.subplots(subplot_kw={"projection": "polar"})
        # ax_hist.plot(theta, r_vals, color="tab:blue", linewidth=1.5)
        # ax_hist.fill(theta, r_vals, color="tab:blue", alpha=0.25)
        # ax_hist.set_title(f"{cube_stem}: Line-Sweep Radar Plot")
        # fig_hist.tight_layout()
        # fig_hist.savefig(os.path.join(detections_dir, f"{plot_stem}_line_hist.png"))
        # plt.close(fig_hist)
    return True