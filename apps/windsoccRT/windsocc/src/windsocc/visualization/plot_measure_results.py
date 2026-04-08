"""
TODO the extrapolated and filtered sep results movies
are showing inconsistencies.
"""

import os

import numpy as np
import polars as pl
from astropy.io import fits
from matplotlib import pyplot as plt
from skimage.draw import ellipse_perimeter
from skimage.transform import resize
import imageio

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


_DIGIT_FONT_5X3: dict[str, np.ndarray] = {
    "0": np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "1": np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "2": np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "3": np.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    "4": np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    "5": np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    "6": np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "7": np.array(
        [
            [1, 1, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "8": np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "9": np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    "-": np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
}


def _draw_label_small_font(
    rgb: np.ndarray,
    text: str,
    row: int,
    col: int,
    color: tuple[int, int, int] = (255, 0, 0),
) -> None:
    """Draw a small raster-font label onto an RGB image in-place."""
    h, w, _ = rgb.shape
    max_chars = 4
    text = text[:max_chars]
    glyph_h = 5
    glyph_w = 3
    spacing = 1
    for i, ch in enumerate(text):
        glyph = _DIGIT_FONT_5X3.get(ch)
        if glyph is None:
            continue
        r0 = row
        c0 = col + i * (glyph_w + spacing)
        r1 = r0 + glyph_h
        c1 = c0 + glyph_w
        if r0 >= h or c0 >= w or r1 <= 0 or c1 <= 0:
            continue
        gr0 = max(r0, 0)
        gc0 = max(c0, 0)
        gr1 = min(r1, h)
        gc1 = min(c1, w)
        sub_glyph = glyph[gr0 - r0 : gr1 - r0, gc0 - c0 : gc1 - c0]
        mask = sub_glyph
        if not np.any(mask):
            continue
        rgb[gr0:gr1, gc0:gc1][mask] = color


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
):
    """Make a movie of the source detections."""
    detections_dir = os.path.join(os.path.dirname(output_dir), "sep_detections")
    decay_dir = os.path.join(os.path.dirname(output_dir), "decay_plots")
    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(decay_dir, exist_ok=True)
    cube_sources = sources_all[-1]
    mf_response_cube = fits.getdata(
        mf_response_cube_path
    )  # 3D array of shape (n_frames, y_size, x_size)
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
    vmin, vmax = np.percentile(mf_response_cube, [0.1, 99.9])
    cube_stem = os.path.splitext(mf_response_cube_fname)[0]

    def _write_movie_imageio(
        sources_by_frame: list[np.ndarray],
        movie_name: str,
        scale_factor: float = 3.0,
    ) -> None:
        final_path = os.path.join(output_dir, movie_name)
        n_frames = len(mf_response_cube)
        h, w = mf_response_cube.shape[1:]
        with imageio.get_writer(
            final_path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
        ) as writer:
            for frame_idx in range(n_frames):
                frame = mf_response_cube[frame_idx].astype(np.float32)
                if not np.isfinite(frame).any():
                    frame[:] = 0.0
                if vmax > vmin:
                    norm = (frame - vmin) / (vmax - vmin)
                else:
                    norm = np.zeros_like(frame, dtype=np.float32)
                norm = np.clip(norm, 0.0, 1.0)
                # Map normalized intensities to a simple dark-to-light blue gradient,
                # roughly resembling matplotlib's ``Blues_r``.
                # low -> dark blue, high -> very light blue.
                dark_blue = np.array([8, 48, 107], dtype=np.float32)
                light_blue = np.array([239, 243, 255], dtype=np.float32)
                rgb = (dark_blue + (light_blue - dark_blue) * norm[..., None]).astype(
                    np.uint8
                )

                sources = sources_by_frame[frame_idx]
                if sources.size > 0:
                    for source in sources:
                        y0 = float(source["y"])
                        x0 = float(source["x"])
                        a_val = float(source["a"])
                        b_val = float(source["b"])
                        theta = float(source["theta"])
                        if not np.isfinite([x0, y0, a_val, b_val, theta]).all():
                            continue
                        r_radius = max(int(abs(1.5 * b_val)), 1)
                        c_radius = max(int(abs(1.5 * a_val)), 1)
                        rr, cc = ellipse_perimeter(
                            int(round(y0)),
                            int(round(x0)),
                            r_radius,
                            c_radius,
                            orientation=theta,
                            shape=(h, w),
                        )
                        rgb[rr, cc] = (255, 0, 0)

                        tid = int(source["track_id"])
                        if tid >= 0:
                            r_scale = float(
                                max(3.0 * a_val, 3.0 * b_val, 2.0)
                            )
                            tx = int(round(x0 + 0.35 * r_scale))
                            ty = int(round(y0 + 0.35 * r_scale))
                            _draw_label_small_font(
                                rgb,
                                str(tid),
                                row=ty,
                                col=tx,
                                color=(255, 0, 0),
                            )

                # Optional upsampling to improve visual resolution for track-id labels.
                if scale_factor != 1.0:
                    out_h = max(int(round(h * scale_factor)), 1)
                    out_w = max(int(round(w * scale_factor)), 1)
                    rgb_out = resize(
                        rgb,
                        (out_h, out_w, 3),
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                    ).astype(np.uint8)
                else:
                    rgb_out = rgb

                writer.append_data(rgb_out)

    try:
        _write_movie_imageio(
            extrapolated_sources_by_frame,
            f"{cube_stem}_extrapolated.mp4",
        )
        _write_movie_imageio(
            sep_sources_by_frame,
            f"{cube_stem}_sep_results.mp4",
        )
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(
            f"WARNING: failed to write detection movies for {cube_stem}: {exc!r}"
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