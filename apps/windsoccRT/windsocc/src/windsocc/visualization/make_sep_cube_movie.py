import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import os
from astropy.io import fits

TRACKED_SOURCE_DTYPE = np.dtype(
    [
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

def _empty_sources_array() -> np.ndarray:
    return np.zeros(0, dtype=TRACKED_SOURCE_DTYPE)


def _as_pandas_tracked(cube_sources: object) -> object:
    if isinstance(cube_sources, pl.DataFrame):
        return pd.DataFrame(cube_sources.to_dicts(), columns=cube_sources.columns)
    return cube_sources

def tracked_df_to_frame_sources(
    cube_sources: object,
    n_frames: int,
) -> list[np.ndarray]:
    if isinstance(cube_sources, list):
        return cube_sources
    cube_sources = _as_pandas_tracked(cube_sources)
    if not isinstance(cube_sources, pd.DataFrame) or cube_sources.empty:
        return [_empty_sources_array() for _ in range(n_frames)]

    tracked = cube_sources.copy()
    required_cols = {"frames", "x_coords", "y_coords", "a", "b", "theta"}
    if not required_cols.issubset(set(tracked.columns)):
        return [_empty_sources_array() for _ in range(n_frames)]

    tracked = tracked.assign(
        frames=pd.to_numeric(tracked["frames"], errors="coerce"),
        x=pd.to_numeric(tracked["x_coords"], errors="coerce"),
        y=pd.to_numeric(tracked["y_coords"], errors="coerce"),
        a=pd.to_numeric(tracked["a"], errors="coerce"),
        b=pd.to_numeric(tracked["b"], errors="coerce"),
        theta=pd.to_numeric(tracked["theta"], errors="coerce"),
        track_id=pd.to_numeric(tracked.get("track_id"), errors="coerce"),
    ).dropna(subset=["frames", "x", "y", "a", "b", "theta"])
    tracked = tracked[(tracked["frames"] >= 0) & (tracked["frames"] < n_frames)]

    if "track_id" in tracked.columns:
        tracked = tracked.sort_values("frames").drop_duplicates(
            subset=["frames", "track_id"], keep="last"
        )

    grouped = {int(frame): frame_df for frame, frame_df in tracked.groupby("frames")}
    frame_sources = []
    for frame_idx in range(n_frames):
        if frame_idx not in grouped:
            frame_sources.append(_empty_sources_array())
            continue
        frame_df = grouped[frame_idx]
        arr = np.zeros(len(frame_df), dtype=TRACKED_SOURCE_DTYPE)
        arr["x"] = frame_df["x"].to_numpy(dtype=np.float64)
        arr["y"] = frame_df["y"].to_numpy(dtype=np.float64)
        arr["a"] = frame_df["a"].to_numpy(dtype=np.float64)
        arr["b"] = frame_df["b"].to_numpy(dtype=np.float64)
        arr["theta"] = frame_df["theta"].to_numpy(dtype=np.float64)
        frame_sources.append(arr)
    return frame_sources


def tracked_df_to_origin_propagated_sources(
    cube_sources: object,
    n_frames: int,
    center_x: float,
    center_y: float,
) -> list[np.ndarray]:
    cube_sources = _as_pandas_tracked(cube_sources)
    if not isinstance(cube_sources, pd.DataFrame) or cube_sources.empty:
        return [_empty_sources_array() for _ in range(n_frames)]

    tracked = cube_sources.copy()
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

    tracked = tracked.assign(
        track_id=pd.to_numeric(tracked["track_id"], errors="coerce"),
        frames=pd.to_numeric(tracked["frames"], errors="coerce"),
        a=pd.to_numeric(tracked["a"], errors="coerce"),
        b=pd.to_numeric(tracked["b"], errors="coerce"),
        theta=pd.to_numeric(tracked["theta"], errors="coerce"),
        direction=pd.to_numeric(tracked["direction"], errors="coerce"),
        velocity_px_per_frame=pd.to_numeric(
            tracked["velocity_px_per_frame"], errors="coerce"
        ),
    ).dropna(
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
    if tracked.empty:
        return [_empty_sources_array() for _ in range(n_frames)]

    per_frame: list[list[tuple[float, float, float, float, float]]] = [
        [] for _ in range(n_frames)
    ]
    for _, group in tracked.groupby("track_id"):
        mean_a = float(np.mean(group["a"].to_numpy(dtype=np.float64)))
        mean_b = float(np.mean(group["b"].to_numpy(dtype=np.float64)))
        mean_theta = float(np.mean(group["theta"].to_numpy(dtype=np.float64)))
        last_row = group.sort_values("frames").iloc[-1]
        direction_rad = np.deg2rad(float(last_row["direction"]))
        speed_px = float(last_row["velocity_px_per_frame"])
        dx_per_frame = speed_px * np.cos(direction_rad + np.pi / 2.0)
        dy_per_frame = speed_px * np.sin(direction_rad + np.pi / 2.0)
        for frame_idx in range(n_frames):
            x_here = center_x + dx_per_frame * frame_idx
            y_here = center_y + dy_per_frame * frame_idx
            per_frame[frame_idx].append(
                (x_here, y_here, mean_a, mean_b, mean_theta)
            )

    frame_sources = []
    for frame_idx in range(n_frames):
        entries = per_frame[frame_idx]
        arr = np.zeros(len(entries), dtype=TRACKED_SOURCE_DTYPE)
        for i, (x_here, y_here, a_val, b_val, theta_val) in enumerate(entries):
            arr["x"][i] = x_here
            arr["y"][i] = y_here
            arr["a"][i] = a_val
            arr["b"][i] = b_val
            arr["theta"][i] = theta_val
        frame_sources.append(arr)
    return frame_sources

def make_source_detection_movie(
    mf_response_cube_paths: list,
    mf_response_cube_fnames: list,
    sources_all: list,
    output_dir: str,
    fps: int = 5,
    cmap: str = "Blues_r",
):
    """Make a movie of the source detections."""
    detections_dir = os.path.join(os.path.dirname(output_dir), "sep_detections")
    os.makedirs(detections_dir, exist_ok=True)
    for cube_idx, (cube_path, cube_fname) in enumerate(zip(mf_response_cube_paths, mf_response_cube_fnames)):
        cube_sources = sources_all[cube_idx]
        mf_response_cube = fits.getdata(cube_path) #3D array of shape (n_frames, y_size, x_size)
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
        cube_stem = os.path.splitext(cube_fname)[0]

        def _render_movie(sources_by_frame: list[np.ndarray], movie_name: str) -> None:
            fig, ax = plt.subplots()
            image = ax.imshow(
                mf_response_cube[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
            )
            title = ax.set_title("Source Extractor Detections")
            ax.set_xlabel("X pixels")
            ax.set_ylabel("Y pixels")
            fig.tight_layout()

            current_patches: list[Ellipse] = []

            def _clear_patches() -> None:
                for patch in list[Ellipse](current_patches):
                    patch.remove()
                current_patches.clear()

            def _add_patches(sources: np.ndarray) -> None:
                for source in sources:
                    ellipse = Ellipse(
                        (source["x"], source["y"]),
                        width=3.0 * source["a"],
                        height=3.0 * source["b"],
                        angle=np.degrees(source["theta"]),
                        fill=False,
                        edgecolor="lime",
                        linewidth=1.0,
                    )
                    ax.add_patch(ellipse)
                    current_patches.append(ellipse)

            def _update(frame_idx: int):
                image.set_data(mf_response_cube[frame_idx])
                _clear_patches()
                _add_patches(sources_by_frame[frame_idx])
                title.set_text(f"{cube_fname}; Slice {frame_idx} of {len(mf_response_cube)}")
                return [image, title, *current_patches]

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

        _render_movie(
            extrapolated_sources_by_frame,
            f"{cube_stem}_extrapolated.mp4",
        )
        _render_movie(
            sep_sources_by_frame,
            f"{cube_stem}_sep_results.mp4",
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