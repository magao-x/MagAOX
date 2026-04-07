import numpy as np
import polars as pl

from windsocc.core.masks import make_annular_mask
from windsocc.core.windtracker import WindTracker


def _angle_diff_deg(a: float, b: float) -> float:
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def _numeric_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64, strict=False)


def _stitch_dropped_tracks(
    cube_sources: pl.DataFrame,
    max_gap_frames: int,
    direction_tol_deg: float = 1.0,
    velocity_tol_mps: float = 1.0,
) -> pl.DataFrame:
    if cube_sources.is_empty():
        return cube_sources
    required_cols = {"track_id", "frames", "direction", "velocity_m_per_s", "matches"}
    if not required_cols.issubset(set(cube_sources.columns)):
        return cube_sources

    numeric = cube_sources.with_columns(
        track_id_num=_numeric_expr("track_id"),
        frame_num=_numeric_expr("frames"),
        direction_num=_numeric_expr("direction"),
        velocity_num=_numeric_expr("velocity_m_per_s"),
        matches_num=_numeric_expr("matches"),
    ).drop_nulls(
        subset=["track_id_num", "frame_num", "direction_num", "velocity_num", "matches_num"]
    )
    if numeric.is_empty():
        return cube_sources

    summaries: dict[int, dict[str, float]] = {}
    track_ids = sorted(
        int(track_id)
        for track_id in numeric.get_column("track_id_num").drop_nulls().unique().to_list()
    )
    for track_id in track_ids:
        group = (
            numeric.filter(pl.col("track_id_num") == float(track_id))
            .sort("frame_num")
            .unique(subset=["frame_num"], keep="last")
        )
        if group.is_empty():
            continue
        matches = group.get_column("matches_num").to_numpy().astype(np.int64)
        mean_angle = np.mean(group.get_column("direction_num").to_numpy())
        summaries[track_id] = {
            "start_frame": float(group.get_column("frame_num").min()),
            "end_frame": float(group.get_column("frame_num").max()),
            "direction": float(mean_angle),
            "velocity": float(np.median(group.get_column("velocity_num").to_numpy())),
            "matches": int(np.sum(matches)),
        }

    if len(track_ids) < 2:
        return cube_sources

    parent = {tid: tid for tid in track_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        parent[max(ra, rb)] = min(ra, rb)

    for i, tid_i in enumerate(track_ids):
        si = summaries[tid_i]
        for tid_j in track_ids[i + 1 :]:
            sj = summaries[tid_j]
            if si["end_frame"] <= sj["start_frame"]:
                gap = sj["start_frame"] - si["end_frame"]
            elif sj["end_frame"] <= si["start_frame"]:
                gap = si["start_frame"] - sj["end_frame"]
            else:
                # Overlapping in time; treat as separate concurrent tracks.
                continue

            if gap > max_gap_frames:
                continue
            if _angle_diff_deg(si["direction"], sj["direction"]) > direction_tol_deg:
                continue
            if abs(si["velocity"] - sj["velocity"]) > velocity_tol_mps:
                continue
            union(tid_i, tid_j)

    remap = {tid: find(tid) for tid in track_ids}
    if all(src == dst for src, dst in remap.items()):
        return cube_sources

    out = cube_sources.clone()
    for old_id, new_id in remap.items():
        if old_id == new_id:
            continue
        out = out.with_columns(
            pl.when(_numeric_expr("track_id") == float(old_id))
            .then(pl.lit(new_id))
            .otherwise(pl.col("track_id"))
            .alias("track_id")
        )
    return out


MODEL_REJECTED_SCHEMA: dict[str, pl.DataType] = {
    "track_id": pl.Int64,
    "reject_reason": pl.String,
    "frames": pl.Int64,
    "direction": pl.Float64,
    "velocity_m_per_s": pl.Float64,
    "matches": pl.Int64,
    "inferred_origin": pl.Float64,
}


def _empty_model_rejected_df() -> pl.DataFrame:
    return pl.DataFrame(schema=MODEL_REJECTED_SCHEMA)


def _model_reject_row(
    track_id: int,
    reject_reason: str,
    group: pl.DataFrame,
) -> dict[str, object]:
    if group.is_empty():
        return {
            "track_id": int(track_id),
            "reject_reason": reject_reason,
            "frames": 0,
            "direction": float("nan"),
            "velocity_m_per_s": float("nan"),
            "matches": 0,
            "inferred_origin": float("nan"),
        }
    frame_num = group.get_column("frame_num")
    last_frame = int(frame_num.max()) if len(frame_num) > 0 else 0
    direction_vals = group.get_column("direction_num").to_numpy()
    velocity_vals = group.get_column("velocity_num").to_numpy()
    matches_vals = group.get_column("matches_num").to_numpy()
    inferred_origin_vals = group.get_column("inferred_origin_num").to_numpy()
    return {
        "track_id": int(track_id),
        "reject_reason": reject_reason,
        "frames": last_frame,
        "direction": float(np.mean(direction_vals)),
        "velocity_m_per_s": float(np.mean(velocity_vals)),
        "matches": int(np.max(matches_vals)),
        "inferred_origin": float(np.mean(inferred_origin_vals)),
    }    



def _keep_track_ids_by_model(
    cube_sources: pl.DataFrame,
    image_center: tuple[int, int],
    inner_bound: int,
    meters_per_pixel: float,
    time_per_frame: float,
    min_matches: int = 10,
    min_detections: int = 10,
    origin_tol_px: float = 10.0,
    rmse_tol_px: float = 10.0,
    outward_tol_px: float = 0.0,
    direction_scatter_tol_deg: float = 1.0,
    velocity_scatter_tol_mps: float = 1.0,
) -> tuple[list[int], pl.DataFrame]:
    if cube_sources.is_empty():
        return [], _empty_model_rejected_df()
    required_cols = {
        "track_id",
        "frames",
        "x_coords",
        "y_coords",
        "dist_traveled_px",
        "matches",
        "direction",
        "velocity_m_per_s",
        "inferred_origin",
    }
    if not required_cols.issubset(set(cube_sources.columns)):
        return [], _empty_model_rejected_df()

    numeric = cube_sources.with_columns(
        track_id_num=_numeric_expr("track_id"),
        frame_num=_numeric_expr("frames"),
        x_num=_numeric_expr("x_coords"),
        y_num=_numeric_expr("y_coords"),
        dist_num=_numeric_expr("dist_traveled_px"),
        matches_num=_numeric_expr("matches"),
        direction_num=_numeric_expr("direction"),
        velocity_num=_numeric_expr("velocity_m_per_s"),
        inferred_origin_num=_numeric_expr("inferred_origin"),
    ).drop_nulls(
        subset=[
            "track_id_num",
            "frame_num",
            "x_num",
            "y_num",
            "dist_num",
            "matches_num",
            "direction_num",
            "velocity_num",
            "inferred_origin_num",
        ]
    )
    if numeric.is_empty():
        return [], _empty_model_rejected_df()

    keep_track_ids: list[int] = []
    reject_rows: list[dict[str, object]] = []
    track_ids = sorted(
        int(track_id)
        for track_id in numeric.get_column("track_id_num").drop_nulls().unique().to_list()
    )
    for track_id in track_ids:
        group = (
            numeric.filter(pl.col("track_id_num") == float(track_id))
            .sort("frame_num")
            .unique(subset=["frame_num"], keep="last")
        )
        # if track_id == 40:
        #     print(group.select("frame_num", "matches_num"))
        #     exit()
        if group.is_empty():
            continue
        # TODO see if these two checks can be consolidated
        if float(group.get_column("matches_num").max()) < float(min_matches):
            reject_rows.append(_model_reject_row(track_id, "min_matches", group))
            continue
        if group.height < max(2, min_detections):
            reject_rows.append(_model_reject_row(track_id, "min_detections", group))
            continue

        radial_dist = group.get_column("dist_num").to_numpy()
        # if np.any(np.diff(radial_dist) < -outward_tol_px):
        #     reject_rows.append(_model_reject_row(track_id, "outward_motion", group))
        #     continue
        
        # # filter by unphysical velocity
        # velocity_vals = group.get_column("velocity_num").to_numpy()
        # avg_velocity = np.mean(velocity_vals)
        # last_frame = group.get_column("frame_num").to_numpy().max()
        # first_frame = group.get_column("frame_num").to_numpy().min()
        # # max velocity is radius of tripwire region / time per frame
        # radius_of_interest = inner_bound * meters_per_pixel
        # max_dist_traveled_px = group.get_column("dist_traveled_px").to_numpy().max()
        # max_velocity = (inner_bound * meters_per_pixel) / (time_per_frame * first_frame)
        # fudge_factor = 1.05
        # if avg_velocity > max_velocity * fudge_factor:
        #     reject_rows.append(_model_reject_row(track_id, "unphysical_velocity", group))
        #     continue

        # the direction col contains the PAs of the tracks wrt the center of image
        # if this peak is garbage, this direction is *not* meaningful
        # we can filter garbage by comparing this value from the direction
        # measured manually using the recorded coordinates
        direction_vals = group.get_column("direction_num").to_numpy() % 360.0
        angles = np.deg2rad(direction_vals)
        mean_angle = (np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))) + 360.0) % 360.0
        xs = group.get_column("x_num").to_numpy()
        ys = group.get_column("y_num").to_numpy()
        xs_c = xs - image_center[0]
        ys_c = ys - image_center[1]
        delta_x = xs_c[-1] - xs_c[0]
        delta_y = ys_c[-1] - ys_c[0]
        measured_direction = np.arctan2(delta_y, delta_x) - np.pi / 2 #radians
        measured_direction_deg = np.rad2deg(measured_direction) % 360.0
        direction_deltas = np.asarray(
            [_angle_diff_deg(float(angle), float(mean_angle)) for angle in direction_vals],
            dtype=np.float64,
        )
        # if track_id == 2014:
        #     print(f"track_id: {track_id}")
        #     print(f"xs: {xs}")
        #     print(f"ys: {ys}")
        #     print(f"xs_c: {xs_c}")
        #     print(f"ys_c: {ys_c}")
        #     print(f"delta_x: {delta_x}")
        #     print(f"delta_y: {delta_y}")
        #     print(f"direction_vals: {direction_vals}")
        #     print(f"mean_angle: {mean_angle}")
        #     print(f"measured_direction_deg: {measured_direction_deg}")
        #     exit()
        if np.max(direction_deltas) > direction_scatter_tol_deg:
            reject_rows.append(_model_reject_row(track_id, "direction_scatter", group))
            continue
        
        if np.abs(measured_direction_deg - mean_angle) > direction_scatter_tol_deg:
            reject_rows.append(_model_reject_row(track_id, "non_physical_trajectory", group))
            continue

        velocity_vals = group.get_column("velocity_num").to_numpy()
        if len(velocity_vals) >= 2 and np.std(velocity_vals) > velocity_scatter_tol_mps:
            reject_rows.append(_model_reject_row(track_id, "velocity_scatter", group))
            continue
        
        x_coords = group.get_column("x_num").to_numpy()
        y_coords = group.get_column("y_num").to_numpy()
        try:
            slope, intercept = np.polyfit(x_coords - image_center[0], y_coords - image_center[1], 1)
            slope = max(slope, 1e-3)
            y_dist_origin = np.abs(intercept)
            x_intercept = -intercept / slope
            x_dist_origin = np.abs(x_intercept)
            xy_dist_origin = np.asarray([x_dist_origin, y_dist_origin])
            dist_closest = np.abs(intercept) / np.sqrt(1 + slope**2)
            if dist_closest > origin_tol_px:
                reject_rows.append(_model_reject_row(track_id, "origin_distance", group))
                continue
        except ValueError:
            reject_rows.append(_model_reject_row(track_id, "slope_divergence", group))
            continue
        # if track_id == 36:
        #     print(f"track_id: {track_id}")
        #     print(f"y_coords: {y_coords}")
        #     print(f"x_coords: {x_coords}")
        #     print(f"slope: {slope}")
        #     print(f"intercept: {intercept}")
        #     print(f"y_dist_origin: {y_dist_origin}")
        #     print(f"x_intercept: {x_intercept}")
        #     print(f"x_dist_origin: {x_dist_origin}")
        #     print(f"xy_dist_origin: {xy_dist_origin}")
        #     print(f"dist_closest: {dist_closest}")
        #     exit()
        x_centered = group.get_column("x_num").to_numpy() - image_center[0]
        y_centered = group.get_column("y_num").to_numpy() - image_center[1]
        points_centered = np.column_stack((x_centered, y_centered))
        centroid = np.mean(points_centered, axis=0)
        _, _, vh = np.linalg.svd(points_centered - centroid, full_matrices=False)
        direction = vh[0]
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            reject_rows.append(_model_reject_row(track_id, "degenerate_line", group))
            continue
        direction = direction / direction_norm
        origin_distances = group.get_column("inferred_origin_num").to_numpy()
        mean_origin_distance = np.mean(origin_distances)
        along_line = np.outer((points_centered - centroid) @ direction, direction)
        residuals = points_centered - centroid - along_line
        rmse = np.sqrt(np.mean(np.sum(residuals**2, axis=1)))
        if mean_origin_distance > origin_tol_px:
            reject_rows.append(_model_reject_row(track_id, "origin_distance", group))
            continue
        if rmse > rmse_tol_px:
            reject_rows.append(_model_reject_row(track_id, "linearity_rmse", group))
            continue

        keep_track_ids.append(int(track_id))

    rejected_df = (
        pl.DataFrame(reject_rows, schema=MODEL_REJECTED_SCHEMA)
        if reject_rows
        else _empty_model_rejected_df()
    )
    return sorted(set(keep_track_ids)), rejected_df


def _normalize_matches_by_track(sources: pl.DataFrame) -> pl.DataFrame:
    if sources.is_empty() or "track_id" not in sources.columns or "matches" not in sources.columns:
        return sources
    matches_lookup = (
        sources.with_columns(
            track_id_num=_numeric_expr("track_id"),
            matches_num=_numeric_expr("matches"),
        )
        .drop_nulls(subset=["track_id_num", "matches_num"])
        .group_by("track_id_num")
        .agg(pl.col("matches_num").max().alias("canonical_matches"))
    )
    if matches_lookup.is_empty():
        return sources
    return (
        sources.with_columns(track_id_num=_numeric_expr("track_id"))
        .join(matches_lookup, on="track_id_num", how="left")
        .with_columns(
            pl.col("canonical_matches")
            .cast(pl.Int64, strict=False)
            .fill_null(_numeric_expr("matches").cast(pl.Int64, strict=False).fill_null(0))
            .alias("matches")
        )
        .drop(["track_id_num", "canonical_matches"])
    )


def _enforce_monotonic_matches(sources: pl.DataFrame) -> pl.DataFrame:
    if sources.is_empty() or "track_id" not in sources.columns or "matches" not in sources.columns:
        return sources
    working = (
        sources.with_columns(
            track_id_num=_numeric_expr("track_id"),
            frame_num=_numeric_expr("frames"),
            matches_num=_numeric_expr("matches").cast(pl.Int64, strict=False).fill_null(0),
        )
        .drop_nulls(subset=["track_id_num", "frame_num"])
        .sort(["track_id_num", "frame_num"])
        .with_columns(
            pl.col("matches_num").cum_max().over("track_id_num").alias("matches"),
        )
        .drop(["track_id_num", "frame_num", "matches_num"])
    )
    return working

def process_single_cc_cube(
    cc_cube: np.ndarray,
    error_map: np.ndarray | None,
    image_center: tuple[int, int],
    meters_per_pixel: float,
    sep_thresh: float,
    sep_minarea: int,
    inner_bound: int,
    outer_bound: int,
    time_per_frame: float,
    min_matches_for_dynamic_mask: int = 3,
    dynamic_window_radius: int = 10,
    tripwire_smoothing_sigma: float = 1.0,
    min_track_matches: int = 10,
    min_track_detections: int = 10,
    track_direction_scatter_tol_deg: float = 20.0,
    track_velocity_scatter_tol_mps: float = 5.0,
    ) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    - Initialize the wind tracker object
    - Apply the tripwire mask to the CC cube
    - For each frame:
        - Unmask predicted locations using previous frame sources
        - Use sep to extract all sources in the unmasked regions
        - Update the bank of candidate sources
        - 
    - Return movie-facing vetted detections, summary-facing final tracks,
      the masked response cube, tracker-side rejected detections, and
      per-track rows for tracks rejected by whole-cube model filtering.
    """
    # define the image center
    image_center = (cc_cube[0].shape[1] // 2, cc_cube[0].shape[0] // 2)
    # init the wind tracker
    wind_tracker = WindTracker(
        image_center=image_center,
        max_distance=10.0,
        time_per_frame=time_per_frame,
        meters_per_pixel=meters_per_pixel,
    )
    base_tripwire_mask = make_annular_mask(cc_cube[0].shape, inner_bound, outer_bound).astype(np.float32)
    yy, xx = np.ogrid[:cc_cube[0].shape[0], :cc_cube[0].shape[1]]
    mask_frames: list[np.ndarray] = []

    for frame_index, frame in enumerate(cc_cube):
        # Predict where wind peaks will be in this frame
        # updated_mask = wind_tracker.predict()
        # Any sources detected in the first 15 frames are unphysical...
        # ...and should be ignored
        if frame_index <= 15:
            mask_frames.append(base_tripwire_mask.copy())
            continue

        # elif frame_index == 25:
        #     exit()
        
        else:
            tripwire_mask = base_tripwire_mask.copy()
            dynamic_centers = wind_tracker.get_dynamic_window_centers(
                min_matches=min_matches_for_dynamic_mask,
                frame_index=frame_index,
            )
            for x_center, y_center in dynamic_centers:
                distance2 = (xx - x_center) ** 2 + (yy - y_center) ** 2
                circle_mask = distance2 <= dynamic_window_radius ** 2
                tripwire_mask[circle_mask] = 0.0

            # if tripwire_smoothing_sigma > 0:
            #     smoothed_mask = gaussian_filter(
            #         tripwire_mask.astype(np.float32), sigma=tripwire_smoothing_sigma
            #     )
            #     tripwire_mask = (smoothed_mask >= 0.5).astype(np.float32)

            mask_frames.append(tripwire_mask.copy())

            wind_tracker.track(
            frame,
            frame_index,
            image_center,
            error_map,
            tripwire_mask,
            sep_thresh,
            sep_minarea,
            meters_per_pixel,
        )

    cube_sources = wind_tracker.vetted_sources.clone()
    cube_sources = _stitch_dropped_tracks(
        cube_sources,
        max_gap_frames=wind_tracker.max_missed_frames,
        direction_tol_deg=1.0,
        velocity_tol_mps=1.0,
    )
    cube_sources = _normalize_matches_by_track(cube_sources)
    cube_sources = _enforce_monotonic_matches(cube_sources)
    keep_track_ids, model_rejected_by_track = _keep_track_ids_by_model(
        cube_sources,
        image_center=image_center,
        meters_per_pixel=meters_per_pixel,
        inner_bound=inner_bound,
        time_per_frame=time_per_frame,
        min_matches=min_track_matches,
        min_detections=min_track_detections,
        origin_tol_px=10.0,
        rmse_tol_px=10.0,
        outward_tol_px=0.0,
        direction_scatter_tol_deg=track_direction_scatter_tol_deg,
        velocity_scatter_tol_mps=track_velocity_scatter_tol_mps,
    )
    if "track_id" in cube_sources.columns:
        if keep_track_ids:
            cube_sources = cube_sources.filter(
                pl.col("track_id").cast(pl.Int64, strict=False).is_in(keep_track_ids)
            )
        else:
            cube_sources = cube_sources.head(0)
    flux_summary_lookup = pl.DataFrame(
        schema={
            "track_id_num": pl.Int64,
            "flux": pl.Float64,
            "flux_err": pl.Float64,
            "source_area": pl.Float64,
        }
    )
    if (
        not cube_sources.is_empty()
        and {"track_id", "frames", "flux", "flux_err", "source_area"}.issubset(set(cube_sources.columns))
    ):
        first_five_flux = (
            cube_sources.with_columns(
                track_id_num=_numeric_expr("track_id").cast(pl.Int64, strict=False),
                frame_num=_numeric_expr("frames"),
                flux_num=_numeric_expr("flux"),
                flux_err_num=_numeric_expr("flux_err"),
                source_area_num=_numeric_expr("source_area"),
            )
            .drop_nulls(subset=["track_id_num", "frame_num"])
            .sort(["track_id_num", "frame_num"])
            .group_by("track_id_num", maintain_order=True)
            .head(5)
        )
        if not first_five_flux.is_empty():
            flux_summary_lookup = first_five_flux.group_by("track_id_num").agg(
                [
                    pl.col("flux_num").mean().alias("flux"),
                    pl.col("flux_err_num").mean().alias("flux_err"),
                    pl.col("source_area_num").mean().alias("source_area"),
                ]
            )
        # drop the first 3 measurements of flux bc they are noisy
        flux_clipped = first_five_flux.tail(3)
        if not flux_clipped.is_empty():
            flux_summary_lookup = flux_clipped.group_by("track_id_num").agg(
                [
                    pl.col("flux_num").mean().alias("flux"),
                    pl.col("flux_err_num").mean().alias("flux_err"),
                    pl.col("source_area_num").mean().alias("source_area"),
                ]
            )
    summary_tracks = cube_sources.clone()
    if not summary_tracks.is_empty() and "track_id" in summary_tracks.columns:
        summary_tracks = (
            summary_tracks.with_columns(
                track_id_num=_numeric_expr("track_id"),
                frame_num=_numeric_expr("frames"),
            )
            .drop_nulls(subset=["track_id_num", "frame_num"])
        )
        if not summary_tracks.is_empty():
            summary_tracks = (
                summary_tracks.sort("frame_num")
                .unique(subset=["track_id_num"], keep="last")
                .select(cube_sources.columns)
            )
            summary_tracks = (
                summary_tracks.with_columns(
                    track_id_num=_numeric_expr("track_id").cast(pl.Int64, strict=False)
                )
                .join(flux_summary_lookup, on="track_id_num", how="left", suffix="_first3")
                .with_columns(
                    pl.coalesce([pl.col("flux_first3"), pl.col("flux")]).alias("flux"),
                    pl.coalesce([pl.col("flux_err_first3"), pl.col("flux_err")]).alias("flux_err"),
                    pl.coalesce([pl.col("source_area_first3"), pl.col("source_area")]).alias(
                        "source_area"
                    ),
                )
                .drop(["track_id_num", "flux_first3", "flux_err_first3", "source_area_first3"])
            )
    # cc_cube_clamped = np.clip(cc_cube, 0, None)
    mask_cube = np.stack(mask_frames, axis=0) if mask_frames else np.empty((0, *cc_cube[0].shape))
    mask_cube = (1 - mask_cube)
    rejected_sources = _normalize_matches_by_track(wind_tracker.rejected_sources.clone())
    rejected_sources = _enforce_monotonic_matches(rejected_sources)
    return (
        cube_sources,
        summary_tracks,
        cc_cube * mask_cube,
        rejected_sources,
        wind_tracker.track_history.clone(),
        model_rejected_by_track,
    )

