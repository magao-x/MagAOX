from typing import Any, Mapping

import numpy as np
import polars as pl
import sep
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

SOURCE_SCHEMA: dict[str, pl.DataType] = {
    "x_coords": pl.Float64,
    "y_coords": pl.Float64,
    "x_err": pl.Float64,
    "y_err": pl.Float64,
    "frames": pl.Int64,
    "track_id": pl.Int64,
    "dist_traveled_px": pl.Float64,
    "dist_traveled_ms": pl.Float64,
    "direction": pl.Float64,
    "velocity_m_per_s": pl.Float64,
    "velocity_px_per_frame": pl.Float64,
    "inferred_origin": pl.Float64,
    "strikes": pl.Int64,
    "matches": pl.Int64,
    "flux": pl.Float64,
    "flux_err": pl.Float64,
    "a": pl.Float64,
    "b": pl.Float64,
    "theta": pl.Float64,
    "source_area": pl.Float64,
}
SOURCE_COLUMNS = list(SOURCE_SCHEMA.keys())
REJECTED_SOURCE_SCHEMA: dict[str, pl.DataType] = {
    **SOURCE_SCHEMA,
    "reject_reason": pl.String,
}
REJECTED_SOURCE_COLUMNS = list(REJECTED_SOURCE_SCHEMA.keys())


def _angle_diff_deg(a: float, b: float) -> float:
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def _require_finite(name: str, value: Any) -> None:
    """Raise ValueError if value is not a finite float (for gating diagnostics)."""
    if value is None:
        raise ValueError(f"non-finite value: {name} (None)")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-finite value: {name} (not convertible to float)") from exc
    if not np.isfinite(x):
        raise ValueError(f"non-finite value: {name} ({x!r})")


def _require_all_finite(name_value_pairs: list[tuple[str, Any]]) -> None:
    """Raise ValueError on the first name whose value is not finite."""
    for name, value in name_value_pairs:
        _require_finite(name, value)


def make_empty_source_dataframe() -> pl.DataFrame:
    return pl.DataFrame(schema=SOURCE_SCHEMA)


def make_empty_rejected_source_dataframe() -> pl.DataFrame:
    return pl.DataFrame(schema=REJECTED_SOURCE_SCHEMA)


class WindTracker:
    """
    Track the wind direction and speed using a tripwire region,
    which is a thin annulus that is sufficiently far from the center
    of the image to allow overlapping CC peaks to spread out.

    Uses sep to detect candidate CC peaks in the tripwire region.

    Then we add or update rows in the candidate source DataFrame.


    Parameters:
    -----------
    max_distance : int
        The maximum distance in pixels to track the CC peaks.
    max_missed_frames : int
        The maximum number of frames to miss before declaring the wind lost.
    """

    def __init__(
        self,
        max_distance: int = 56,
        max_missed_frames: int = 5,
        image_center: tuple[int, int] = (None, None),
        time_per_frame: float = 0.004,
        meters_per_pixel: float = (6.5 / 60),
    ):
        if image_center[0] is None or image_center[1] is None:
            raise ValueError("image_center must be defined (not None)")
        self.time_per_frame = time_per_frame
        self.meters_per_pixel = meters_per_pixel
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.image_center = image_center
        self.confirmation_matches = 2
        self.max_direction_delta_deg = 2000.0
        self.max_theta_delta_deg = 35.0
        self.max_speed_delta_px = 2.5
        self.prune_immunity_matches = 30
        self.candidate_sources = make_empty_source_dataframe()
        self.predicted_sources = make_empty_source_dataframe()
        self.vetted_sources = make_empty_source_dataframe()
        self.track_history = make_empty_source_dataframe()
        # Row-level rejection events (gating/assignment/etc).
        self.rejected_detections = make_empty_rejected_source_dataframe()
        # Terminally rejected tracks (definitive loss/prune).
        self.rejected_sources = make_empty_rejected_source_dataframe()
        self._next_track_id = 0
        self.confirmed_track_ids: set[int] = set()
        self.track_match_counts: dict[int, int] = {}

    def _numeric_expr(self, column: str) -> pl.Expr:
        """Cast the column to a float.
        """
        return pl.col(column).cast(pl.Float64, strict=False)

    def _int_expr(self, column: str) -> pl.Expr:
        """Cast the column to an integer.
        """
        return pl.col(column).cast(pl.Int64, strict=False)

    def _df_to_numeric(self, df: pl.DataFrame, column: str) -> pl.Series:
        return df.get_column(column).cast(pl.Float64, strict=False)

    def _allocate_track_ids(self, n_sources: int) -> np.ndarray:
        new_ids = np.arange(self._next_track_id, self._next_track_id + n_sources)
        self._next_track_id += n_sources
        for track_id in new_ids:
            self.track_match_counts[int(track_id)] = 0
        return new_ids

    def get_track_matches(self, track_id: int) -> int:
        """Pull the match count for a given track_id.
        Returns 0 if the track_id is not found.

        track_match_counts is a *dict* of track_id to match count.
        """
        return int(self.track_match_counts.get(int(track_id), 0))

    def set_track_matches(self, track_id: int, matches: int) -> int:
        """Set the match count for a given track_id.

        Normalize the matches to be at least 0....
        I don't think this is necessary?.. but its cheap
        """
        normalized = max(0, int(matches))
        self.track_match_counts[int(track_id)] = normalized
        return normalized

    def increment_track_matches(self, track_id: int) -> int:
        current = self.get_track_matches(track_id)
        return self.set_track_matches(track_id, current + 1)

    def merge_track_matches(self, old_id: int, new_id: int) -> int:
        """
        Merge track IDs by preferring the stronger track.
        The old_id should always have 0 matches by design.
        """
        matches = max(self.get_track_matches(old_id), self.get_track_matches(new_id))
        self.set_track_matches(new_id, matches)
        # remove the old_id from the track_match_counts if it exists
        self.track_match_counts.pop(int(old_id), None)
        return matches

    def _frame_from_dicts(self, rows: list[dict[str, Any]]) -> pl.DataFrame:
        if not rows:
            return make_empty_source_dataframe()
        normalized = [
            {column: row.get(column) for column in SOURCE_COLUMNS}
            for row in rows
        ]
        return pl.DataFrame(normalized, schema=SOURCE_SCHEMA)

    def _rejected_frame_from_dicts(self, rows: list[dict[str, Any]]) -> pl.DataFrame:
        if not rows:
            return make_empty_rejected_source_dataframe()
        normalized = []
        for row in rows:
            rejected_row = {column: row.get(column) for column in SOURCE_COLUMNS}
            rejected_row["reject_reason"] = row.get("reject_reason")
            normalized.append(rejected_row)
        return pl.DataFrame(normalized, schema=REJECTED_SOURCE_SCHEMA)

    def _replace_row(
        self,
        df: pl.DataFrame,
        row_index: int,
        row_values: dict[str, Any],
    ) -> pl.DataFrame:
        new_row = self._frame_from_dicts([row_values])
        pieces: list[pl.DataFrame] = []
        if row_index > 0:
            pieces.append(df.slice(0, row_index))
        pieces.append(new_row)
        if row_index + 1 < df.height:
            pieces.append(df.slice(row_index + 1))
        return pl.concat(pieces, how="vertical") if pieces else make_empty_source_dataframe()

    def _find_row_index_by_track_id(self, df: pl.DataFrame, track_id: int) -> int | None:
        if df.is_empty():
            return None
        # add a "row_index" as the first col to the dataframe
        df_with_index = df.with_row_index("row_index")
        matches = (
            df_with_index
            .filter(self._int_expr("track_id") == track_id)
            .select("row_index")
            .to_series()
            .to_list()
        )
        if not matches:
            return None
        return int(matches[0])

    def _as_float(self, value: Any, default: float = np.nan) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _as_int(self, value: Any, default: int = 0) -> int:
        numeric = self._as_float(value, default=float(default))
        if np.isnan(numeric):
            return default
        return int(numeric)

    def _make_df_new_sources(
        self,
        sources_in_frame: np.ndarray,
        image_center: tuple[int, int],
        frame_index: int,
        time_per_frame: float = 0.004,
        meters_per_pixel: float = 0.108,
    ) -> pl.DataFrame:
        """Build a frame-local detections table with provisional track IDs."""
        if sources_in_frame.size == 0:
            return make_empty_source_dataframe()

        source_xs = sources_in_frame["x"]
        source_ys = sources_in_frame["y"]
        xs_c = source_xs - image_center[0]
        ys_c = source_ys - image_center[1]
        source_dists_from_origin = np.sqrt(xs_c**2 + ys_c**2)
        source_errx2 = sources_in_frame["errx2"]
        source_erry2 = sources_in_frame["erry2"]
        source_flux = sources_in_frame["flux"]
        source_field_names = set(sources_in_frame.dtype.names or ())
        if "flux_err" in source_field_names:
            source_flux_err = sources_in_frame["flux_err"]
        elif "fluxerr" in source_field_names:
            source_flux_err = sources_in_frame["fluxerr"]
        else:
            source_flux_err = np.full(len(source_xs), np.nan, dtype=np.float64)
        source_a = sources_in_frame["a"]
        source_b = sources_in_frame["b"]
        source_theta = sources_in_frame["theta"]
        if "npix" in source_field_names:
            source_area = np.asarray(sources_in_frame["npix"], dtype=np.float64)
        elif "tnpix" in source_field_names:
            source_area = np.asarray(sources_in_frame["tnpix"], dtype=np.float64)
        else:
            source_area = np.full(len(source_xs), np.nan, dtype=np.float64)

        time_elapsed = max(frame_index * time_per_frame, time_per_frame)
        distance_traveled_px = source_dists_from_origin
        distance_traveled_ms = source_dists_from_origin * meters_per_pixel
        source_vs_px_per_frame = distance_traveled_px / max(frame_index + 1, 1)
        source_vs_m_per_s = distance_traveled_ms / time_elapsed

        source_pas = np.arctan2(-xs_c, ys_c)
        source_pas_deg = (np.degrees(source_pas) + 360.0) % 360.0

        direction_rad = np.deg2rad(source_pas_deg)
        inferred_traceback_x = source_xs - source_vs_px_per_frame * np.cos(direction_rad + np.pi / 2.0) * frame_index
        inferred_traceback_y = source_ys - source_vs_px_per_frame * np.sin(direction_rad + np.pi / 2.0) * frame_index
        inferred_origin_dist = np.sqrt(
            (inferred_traceback_x - image_center[0]) ** 2 + (inferred_traceback_y - image_center[1]) ** 2
        )
        new_source_ids = self._allocate_track_ids(len(source_xs))

        return pl.DataFrame(
            {
                "x_coords": source_xs,
                "y_coords": source_ys,
                "x_err": source_errx2,
                "y_err": source_erry2,
                "frames": np.full(len(source_xs), frame_index, dtype=np.int64),
                "track_id": new_source_ids.astype(np.int64),
                "dist_traveled_px": distance_traveled_px,
                "dist_traveled_ms": distance_traveled_ms,
                "direction": source_pas_deg,
                "velocity_m_per_s": source_vs_m_per_s,
                "velocity_px_per_frame": source_vs_px_per_frame,
                "inferred_origin": inferred_origin_dist,
                "strikes": np.zeros(len(source_xs), dtype=np.int64),
                "matches": np.zeros(len(source_xs), dtype=np.int64),
                "flux": source_flux,
                "flux_err": source_flux_err,
                "a": source_a,
                "b": source_b,
                "theta": source_theta,
                "source_area": source_area,
            },
            schema=SOURCE_SCHEMA,
        )

    def _append_history_sources(self, detections: pl.DataFrame) -> None:
        if detections.is_empty():
            return
        self.track_history = pl.concat(
            [self.track_history, detections.select(SOURCE_COLUMNS)],
            how="vertical",
        )

    def _append_vetted_sources(self, detections: pl.DataFrame) -> None:
        if detections.is_empty():
            return
        combined = pl.concat(
            [self.vetted_sources, detections.select(SOURCE_COLUMNS)],
            how="vertical",
        )
        if {"track_id", "frames"}.issubset(set(combined.columns)):
            combined = (
                combined.with_columns(
                    track_id_num=self._int_expr("track_id"),
                    frame_num=self._int_expr("frames"),
                )
                .drop_nulls(subset=["track_id_num", "frame_num"])
                .sort(["frame_num", "track_id_num"])
                .unique(subset=["track_id_num", "frame_num"], keep="last")
                .select(SOURCE_COLUMNS)
            )
        self.vetted_sources = combined

    def _append_rejections(
        self,
        target_attr: str,
        detections: pl.DataFrame | None = None,
        reject_reason: str | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        rejected_frame = make_empty_rejected_source_dataframe()
        if rows:
            rejected_frame = self._rejected_frame_from_dicts(rows)
        elif detections is not None and not detections.is_empty():
            rejected_frame = detections.select(SOURCE_COLUMNS)
            rejected_frame = rejected_frame.with_columns(
                pl.lit(reject_reason).cast(pl.String).alias("reject_reason")
            )
        if rejected_frame.is_empty():
            return
        rejected_frame = rejected_frame.with_columns(
            pl.col("track_id")
            .cast(pl.Int64, strict=False)
            .map_elements(
                lambda tid: (
                    self.get_track_matches(int(tid)) if tid is not None else None
                ),
                return_dtype=pl.Int64,
            )
            .alias("matches")
        )

        target_df = getattr(self, target_attr)
        combined = pl.concat([target_df, rejected_frame], how="vertical")
        combined = (
            combined.with_columns(
                track_id_num=self._int_expr("track_id"),
                frame_num=self._int_expr("frames"),
                reject_reason_norm=pl.col("reject_reason").cast(pl.String, strict=False),
            )
            .sort(["frame_num", "track_id_num"])
            .unique(
                subset=["track_id_num", "frame_num", "reject_reason_norm"],
                keep="last",
            )
            .select(REJECTED_SOURCE_COLUMNS)
        )
        setattr(self, target_attr, combined)

    def _append_rejected_sources(
        self,
        detections: pl.DataFrame | None = None,
        reject_reason: str | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self._append_rejections(
            target_attr="rejected_detections",
            detections=detections,
            reject_reason=reject_reason,
            rows=rows,
        )

    def _append_terminal_rejected_sources(
        self,
        detections: pl.DataFrame | None = None,
        reject_reason: str | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self._append_rejections(
            target_attr="rejected_sources",
            detections=detections,
            reject_reason=reject_reason,
            rows=rows,
        )

    def _promote_confirmed_track_rows(self, track_id: int, extra_rows: pl.DataFrame) -> None:
        """Call this func once per unique matched track ID in the current frame.


        Args:
            track_id (int): _description_
            extra_rows (pl.DataFrame): 
        """
        if extra_rows.is_empty():
            print(f"extra_rows in promote_confirmed_track_rows is empty,\
                this should not happen")
            return
        track_rows = extra_rows.filter(self._int_expr("track_id") == track_id)
        if track_rows.is_empty():
            print(f"track_rows in promote_confirmed_track_rows is empty,\
                this should not happen")
            return

        active_rows = self.candidate_sources.filter(self._int_expr("track_id") == track_id)
        if active_rows.is_empty():
            print(f"active_rows in promote_confirmed_track_rows is empty,\
                this should not happen")
            return
        current_matches = self.get_track_matches(track_id)
        if current_matches is None:
            print(f"current_matches in promote_confirmed_track_rows is None,\
                this should not happen")
            return
        if track_id in self.confirmed_track_ids:
            self._append_vetted_sources(track_rows)
            return
        if current_matches < self.confirmation_matches:
            return

        history_with_current = pl.concat(
            [self.track_history, extra_rows.select(SOURCE_COLUMNS)],
            how="vertical",
        )
        promote_rows = history_with_current.filter(self._int_expr("track_id") == track_id)
        if promote_rows.is_empty():
            print(f"promote_rows in promote_confirmed_track_rows is empty,\
                this should not happen")
            return
        self._append_vetted_sources(promote_rows)
        self.confirmed_track_ids.add(track_id)

    def _match_passes_gating(
        self,
        active_row: Mapping[str, Any],
        this_frame_row: Mapping[str, Any],
    ) -> tuple[bool, str | None, float]:
        prior_direction = self._as_float(active_row.get("direction"))
        prior_x = self._as_float(active_row.get("x_coords"))
        prior_y = self._as_float(active_row.get("y_coords"))
        prior_speed_px = self._as_float(active_row.get("velocity_px_per_frame"))
        prior_speed_m_per_s = self._as_float(active_row.get("velocity_m_per_s"))
        prior_distance_px = self._as_float(active_row.get("dist_traveled_px"))
        prior_a = self._as_float(active_row.get("a"))
        prior_b = self._as_float(active_row.get("b"))
        prior_theta = self._as_float(active_row.get("theta"))
        prior_matches = self._as_float(active_row.get("matches"), default=0.0)
        prior_frame = self._as_float(active_row.get("frames"))

        current_direction = self._as_float(this_frame_row.get("direction"))
        current_x = self._as_float(this_frame_row.get("x_coords"))
        current_y = self._as_float(this_frame_row.get("y_coords"))
        current_distance_px = self._as_float(this_frame_row.get("dist_traveled_px"))
        current_distance_m = self._as_float(this_frame_row.get("dist_traveled_ms"))
        current_speed_px = self._as_float(this_frame_row.get("velocity_px_per_frame"))
        current_speed_m_per_s = self._as_float(this_frame_row.get("velocity_m_per_s"))
        current_a = self._as_float(this_frame_row.get("a"))
        current_b = self._as_float(this_frame_row.get("b"))
        current_theta = self._as_float(this_frame_row.get("theta"))
        current_frame = self._as_int(this_frame_row.get("frames"))

        # recompute the speed based on current and prior distance
        update_dist_traveled = current_distance_px - prior_distance_px #pixels
        update_time_elapsed = (current_frame - prior_frame) #frames
        update_speed_px = update_dist_traveled / update_time_elapsed #pixels per frame
        update_speed_m_per_s = np.mean([prior_speed_m_per_s, current_speed_m_per_s])
        inferred_distance_traveled = update_speed_m_per_s * current_frame * self.time_per_frame
        inferred_speed = current_distance_px / current_frame #
        inferred_direction = np.deg2rad(np.mean([prior_direction, current_direction])) + np.pi / 2.0
        prior_xc = prior_x - self.image_center[0]
        prior_yc = prior_y - self.image_center[1]
        current_xc = current_x - self.image_center[0]
        current_yc = current_y - self.image_center[1]
        prior_coords = np.asarray([prior_x, prior_y])
        current_coords = np.asarray([current_x, current_y])
        delta_coords = current_coords - prior_coords
        measured_direction = np.arctan2(delta_coords[1], delta_coords[0]) - np.pi / 2 #radians
        measured_direction_deg = np.rad2deg(measured_direction) % 360.0
        assumed_direction = np.arctan2(-current_xc, current_yc) + 2*np.pi #radians
        assumed_direction_deg = np.rad2deg(assumed_direction) % 360.0
        # traceback_x = current_x - inferred_speed * np.cos(inferred_direction) * current_frame
        # traceback_y = current_y - inferred_speed * np.sin(inferred_direction) * current_frame
        traceback_x = current_x - update_speed_px * np.cos(inferred_direction) * current_frame
        traceback_y = current_y - update_speed_px * np.sin(inferred_direction) * current_frame

        inferred_origin_dist = np.sqrt(
            (traceback_x - self.image_center[0]) ** 2 + (traceback_y - self.image_center[1]) ** 2
        )
        # inferred_origin_dist = inferred_distance_traveled - current_distance_m
        # compare inferred direction with measured direction
        diff_direction = measured_direction_deg - assumed_direction_deg # degrees
        # if active_row.get('track_id') == 21:
        #     print(f"track_id: {this_frame_row.get('track_id')}")
        #     print(f"prior_coords: {prior_coords}")
        #     print(f"current_coords: {current_coords}")
        #     print(f"delta_coords: {delta_coords}")
        #     print(f"measured_direction: {measured_direction_deg}")
        #     print(f"assumed_direction: {assumed_direction_deg}")
        #     print(f"diff_direction: {diff_direction}")
        #     print(f"prior_distance_px: {prior_distance_px}")
        #     print(f"current_distance_px: {current_distance_px}")
        #     # exit()
        diff_distance = inferred_distance_traveled - current_distance_m
        valid_dist = diff_distance <= self.max_distance
        # valid_dist = inferred_origin_dist <= self.max_distance
        diff_speed = current_speed_px - inferred_speed
        valid_speed = np.abs(diff_speed) < 1.0

        if prior_matches >= 1:
            _require_all_finite(
                [
                    ("prior_direction", prior_direction),
                    ("prior_x", prior_x),
                    ("prior_y", prior_y),
                    ("prior_speed_px", prior_speed_px),
                    ("prior_speed_m_per_s", prior_speed_m_per_s),
                    ("prior_distance_px", prior_distance_px),
                    ("prior_a", prior_a),
                    ("prior_b", prior_b),
                    ("prior_frame", prior_frame),
                    ("current_direction", current_direction),
                    ("current_x", current_x),
                    ("current_y", current_y),
                    ("current_distance_px", current_distance_px),
                    ("current_distance_m", current_distance_m),
                    ("current_speed_px", current_speed_px),
                    ("current_speed_m_per_s", current_speed_m_per_s),
                    ("current_a", current_a),
                    ("current_b", current_b),
                    ("current_frame", float(current_frame)),
                    ("update_speed_px", update_speed_px),
                    ("update_speed_m_per_s", update_speed_m_per_s),
                    ("inferred_speed", inferred_speed),
                    ("inferred_direction", inferred_direction),
                    ("measured_direction_deg", measured_direction_deg),
                    ("assumed_direction_deg", assumed_direction_deg),
                    ("diff_direction", diff_direction),
                    ("diff_speed", diff_speed),
                    ("diff_distance", diff_distance),
                    ("inferred_origin_dist", inferred_origin_dist),
                ]
            )
            if current_frame == 0:
                raise ValueError(
                    "invalid gating: current_frame (must be non-zero when prior_matches >= 1)"
                )
            if update_time_elapsed == 0:
                raise ValueError(
                    "invalid gating: update_time_elapsed (zero when prior_matches >= 1)"
                )

        velocity_min_px_per_frame = current_distance_px / (current_frame * self.time_per_frame)
        velocity_min_m_per_s = velocity_min_px_per_frame * self.meters_per_pixel
        fudge_factor = 1.01
        if current_speed_px > velocity_min_px_per_frame * fudge_factor:
            reject_reason = "unphysical_velocity"
            return False, reject_reason, float(inferred_origin_dist)
        if update_speed_m_per_s < 1.0:
            reject_reason = "static_source"
            return False, reject_reason, float(inferred_origin_dist)
        if prior_matches >= 1:
            if current_distance_px - prior_distance_px < 0.0:
                reject_reason = "inward_motion"
                return False, reject_reason, float(inferred_origin_dist)
            if _angle_diff_deg(prior_direction, current_direction) > self.max_direction_delta_deg:
                reject_reason = "prior_frame_direction_mismatch"
                return False, reject_reason, float(inferred_origin_dist)
            if abs(current_speed_px - prior_speed_px) > self.max_speed_delta_px:
                reject_reason = "prior_frame_speed_mismatch"
                return False, reject_reason, float(inferred_origin_dist)
            if current_a > (2.0 * max(prior_a, 1e-6)) or current_a < (0.5 * prior_a):
                reject_reason = "a_delta"
                return False, reject_reason, float(inferred_origin_dist)
            if current_b > (2.0 * max(prior_b, 1e-6)) or current_b < (0.5 * prior_b):
                reject_reason = "b_delta"
                return False, reject_reason, float(inferred_origin_dist)
            if not valid_speed:
                reject_reason = "inferred_speed_mismatch"
                return False, reject_reason, float(inferred_origin_dist)
            if not valid_dist:
                reject_reason = "origin_traceback_invalid"
                return False, reject_reason, float(inferred_origin_dist)

        elongated = False
        if prior_a > 0 and prior_b > 0:
            axis_ratio = max(prior_a / prior_b, prior_b / prior_a)
            elongated = axis_ratio >= 1.3
        if elongated and prior_matches >= 1:
            _require_all_finite(
                [
                    ("prior_theta", prior_theta),
                    ("current_theta", current_theta),
                ]
            )
            theta_delta = abs(np.degrees(current_theta - prior_theta)) % 180.0
            theta_delta = min(theta_delta, 180.0 - theta_delta)
            if theta_delta > self.max_theta_delta_deg:
                reject_reason = "theta_delta"
                return False, reject_reason, float(inferred_origin_dist)

        return True, None, float(inferred_origin_dist)

    def predict(
        self,
        frame_index: int,
        image_center: tuple[int, int],
        meters_per_pixel: float,
    ) -> pl.DataFrame:
        """
        Take each xy-coordinate and predict where the next one will be.

        Two points define a line; even if we only have one detection for
        a candidate source we still have 2 points (the origin...!)

        Actually, we also have a velocity even for new points by virtue
        of knowing the radius of the tripwire region and frame index.

        ...aaaand, we can also filter out unphysical velocities here.
        e.g., if a source is moving with a speed such that it would not
        have originated from the origin, it gets pruned.
        """
        if self.candidate_sources.is_empty():
            return make_empty_source_dataframe()

        matched_sources = self.candidate_sources.clone()
        direction_rad = np.deg2rad(
            self._df_to_numeric(matched_sources, "direction").to_numpy()
        )
        speed_px = (
            self._df_to_numeric(matched_sources, "velocity_px_per_frame")
            .fill_null(0.0)
            .to_numpy()
        )
        source_xs = self._df_to_numeric(matched_sources, "x_coords").to_numpy()
        source_ys = self._df_to_numeric(matched_sources, "y_coords").to_numpy()

        source_xs_pred = source_xs + speed_px * np.cos(direction_rad + np.pi / 2.0)
        source_ys_pred = source_ys + speed_px * np.sin(direction_rad + np.pi / 2.0)
        xs_c_pred = source_xs_pred - image_center[0]
        ys_c_pred = source_ys_pred - image_center[1]
        dist_px_pred = np.sqrt(xs_c_pred**2 + ys_c_pred**2)
        dist_ms_pred = dist_px_pred * meters_per_pixel

        return matched_sources.with_columns(
            [
                pl.Series("x_coords", source_xs_pred),
                pl.Series("y_coords", source_ys_pred),
                pl.Series(
                    "frames",
                    np.full(matched_sources.height, frame_index + 1, dtype=np.int64),
                ),
                pl.Series("dist_traveled_px", dist_px_pred),
                pl.Series("dist_traveled_ms", dist_ms_pred),
            ]
        ).select(SOURCE_COLUMNS)

    def extract(
        self,
        cc_frame: np.ndarray,
        error_map: np.ndarray | None,
        tripwire_mask: np.ndarray,
        sep_thresh: float,
        sep_minarea: int,
        image_center: tuple[int, int],
        cc_frame_ind: int,
        time_per_frame: float,
        meters_per_pixel: float,
    ) -> pl.DataFrame:
        """
        Just extract *all* sources from the CC frame using sep and tripwire mask.
        They will be filtered out later.
        """
        sep_frame = np.ascontiguousarray(cc_frame, dtype=np.float32)
        # sep_frame_clamped = np.clip(sep_frame, 0, None)
        # data_sub = sep_frame_clamped
        bkg = sep.Background(sep_frame)
        data_sub = sep_frame - bkg.rms()
        sep_err = None
        if error_map is not None:
            sep_err = np.ascontiguousarray(error_map, dtype=np.float32)
        sources_in_frame = sep.extract(
            data_sub,
            mask=tripwire_mask,
            thresh=sep_thresh,
            # err=sep_err,
            err=bkg.globalrms,
            minarea=sep_minarea,
            filter_kernel=None,
            deblend_cont=0.0005, # default is 0.005 (0.05%), lower is more sensitive
            deblend_nthresh=32,    # default is 32, higher better for saddles
            clean=False,
        )

        if sources_in_frame.size > 0 and sources_in_frame.shape[0] > 0:
            order = np.argsort(sources_in_frame["flux"])[::-1]
            sources_in_frame = sources_in_frame[order]

        return self._make_df_new_sources(
            sources_in_frame,
            image_center,
            cc_frame_ind,
            time_per_frame,
            meters_per_pixel,
        )

    def match(
        self,
        sources_this_frame: pl.DataFrame,
        frame_index: int,
    ) -> pl.DataFrame:
        """Take all the current frame detections and try to 
        match them to the predicted sources from the previous frame.

        If a detection is matched to a predicted source, update the
        active track's state.
        """
        if sources_this_frame.is_empty():
            # if there are no detections in this frame...
            if not self.candidate_sources.is_empty():
                # but there are active tracks from the previous frame...
                prior_ids = set(
                    self.candidate_sources.get_column("track_id")
                    .cast(pl.Int64, strict=False)
                    .drop_nulls()
                    .to_list()
                )
                # then everything gets a strike
                candidate_with_strikes = self.candidate_sources.with_columns(
                        (self._int_expr("strikes").fill_null(0) + 1).alias("strikes")
                    )
                rejected_rows = candidate_with_strikes.filter(
                    (self._int_expr("strikes").fill_null(0) > self.max_missed_frames)
                    & (self._int_expr("matches").fill_null(0) < self.prune_immunity_matches)
                )
                self._append_terminal_rejected_sources(
                    rejected_rows,
                    reject_reason="max_missed_frames",
                )
                # prune the striked out tracks
                self.candidate_sources = (
                    candidate_with_strikes.filter(
                        (self._int_expr("strikes").fill_null(0) <= self.max_missed_frames)
                        | (self._int_expr("matches").fill_null(0) >= self.prune_immunity_matches)
                    )
                    .select(SOURCE_COLUMNS)
                # select SOURCE_COLUMNS is likely redundant, but it's cheap
                )
                kept_ids = set(
                    self.candidate_sources.get_column("track_id")
                    .cast(pl.Int64, strict=False)
                    .drop_nulls()
                    .to_list()
                )
                for removed_track_id in (prior_ids - kept_ids): # set operations are cool
                    # pop method for dict removes the key-value pair for the given key
                    # syntax dict.pop(keyname, defaultvalue) where defaultvalue is optional
                    self.track_match_counts.pop(int(removed_track_id), None)
            return self.candidate_sources
        # add a row index as the first col to the sources_this_frame dataframe
        # source_index is the named col
        history_rows = sources_this_frame.with_row_index("source_index")
        # filter the predicted_sources dataframe to only include the
        # sources from the current frame
        predicted_subset = self.predicted_sources.filter(
            self._int_expr("frames") == frame_index
        )

        if predicted_subset.is_empty() or self.candidate_sources.is_empty():
            # process is still spooling up...
            self.candidate_sources = sources_this_frame.select(SOURCE_COLUMNS)
            self._append_history_sources(history_rows)
            return self.candidate_sources

        this_frame_data = ( #still a dataframe
            sources_this_frame.with_row_index("source_index")
            .select(
                [
                    pl.col("source_index"),
                    self._numeric_expr("x_coords").alias("x"),
                    self._numeric_expr("y_coords").alias("y"),
                    self._numeric_expr("dist_traveled_px").alias("dist_traveled_px"),
                    self._numeric_expr("velocity_m_per_s").alias("velocity_m_per_s"),
                    self._numeric_expr("track_id").alias("track_id"),
                ]
            )
            .drop_nulls(subset=["x", "y", "track_id", "dist_traveled_px"])
        )
        predicted_data = predicted_subset.select( #still a dataframe
            [
                self._numeric_expr("x_coords").alias("x"),
                self._numeric_expr("y_coords").alias("y"),
                self._numeric_expr("track_id").alias("track_id"),
            ]
        ).drop_nulls(subset=["x", "y", "track_id"])

        if this_frame_data.is_empty() or predicted_data.is_empty():
            self._append_history_sources(history_rows)
            if this_frame_data.is_empty():
                return self.candidate_sources
            # no predictions yet, so just add the new detections as new tracks
            # and return the candidate sources
            self.candidate_sources = pl.concat(
                [self.candidate_sources, sources_this_frame.select(SOURCE_COLUMNS)],
                how="vertical",
            )
            return self.candidate_sources

        new_detections_coords = this_frame_data.select(["x", "y"]).to_numpy()
        predicted_coords = predicted_data.select(["x", "y"]).to_numpy()
        # print(f"predicted_coords.shape: {predicted_coords.shape}")
        raw_distance = cdist(predicted_coords, new_detections_coords)
        distance = raw_distance.copy()

        unphysically_fast_candidates = (
            this_frame_data.get_column("velocity_m_per_s").fill_null(0.0).to_numpy() > 50.0
        )
        if np.any(unphysically_fast_candidates):
            distance[:, unphysically_fast_candidates] += 100.0

        # # Check if the distance matrix is feasible
        # print(f"Number of current candidates: {this_frame_data.height}")
        # print(f"Distance matrix: {distance}")
        # print(f"Number of unphysically fast candidates: {np.sum(unphysically_fast_candidates)}")
        # exit()

        row_ind, col_ind = linear_sum_assignment(distance)
        assignment_distance = raw_distance[row_ind, col_ind]
        accepted = assignment_distance < 1.0

        matched_pred_track_ids: set[int] = set()
        inward_pred_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        track_id_remap: dict[int, int] = {}
        vetted_rows: list[dict[str, Any]] = []
        inferred_origin_updates: dict[int, float] = {}

        for row_i, col_i, is_match in zip(row_ind, col_ind, accepted):
            predicted_row = predicted_data.row(int(row_i), named=True)
            new_detection_row = this_frame_data.row(int(col_i), named=True)
            predicted_track_id = int(predicted_row["track_id"])
            new_detection_track_id = int(new_detection_row["track_id"])
            source_idx = int(new_detection_row["source_index"])
            if not is_match:
                rejected_row = dict(sources_this_frame.row(source_idx, named=True))
                rejected_row["track_id"] = predicted_track_id
                rejected_row["reject_reason"] = "assignment_distance_exceeded"
                self._append_rejected_sources(rows=[rejected_row])
                continue
            # get the row index of candidate_sources using the track_id
            active_idx = self._find_row_index_by_track_id(self.candidate_sources, predicted_track_id)
            if active_idx is None:
                rejected_row = dict(sources_this_frame.row(source_idx, named=True))
                rejected_row["track_id"] = predicted_track_id
                rejected_row["reject_reason"] = "no_active_track"
                self._append_rejected_sources(rows=[rejected_row])
                continue

            # return the row as a dictionary (named=True triggers dict not tuple)
            active_row = self.candidate_sources.row(active_idx, named=True)
            prior_radius = self._as_float(active_row.get("dist_traveled_px"))
            current_radius = float(new_detection_row["dist_traveled_px"])
            if not np.isnan(prior_radius) and current_radius < prior_radius:
                inward_pred_track_ids.add(predicted_track_id)
                rejected_row = dict(sources_this_frame.row(source_idx, named=True))
                rejected_row["track_id"] = predicted_track_id
                rejected_row["reject_reason"] = "inward_motion"
                self._append_rejected_sources(rows=[rejected_row])
                continue
            #TODO add the inward motion check to the gating
            # prior_matches = self.get_track_matches(predicted_track_id)
            # return the row as a dictionary (named=True triggers dict not tuple)
            updated_row = dict(sources_this_frame.row(source_idx, named=True))

            valid_match, reject_reason, inferred_origin = self._match_passes_gating(active_row, updated_row)
            if not valid_match:
                rejected_row = dict(updated_row)
                rejected_row["track_id"] = predicted_track_id
                rejected_row["reject_reason"] = reject_reason
                self._append_rejected_sources(rows=[rejected_row])
                continue

            updated_row["track_id"] = predicted_track_id
            updated_row["matches"] = self.increment_track_matches(predicted_track_id)
            updated_row["strikes"] = 0
            updated_row["inferred_origin"] = inferred_origin
            self.candidate_sources = self._replace_row(
                self.candidate_sources,
                active_idx,
                updated_row,
            )
            matched_pred_track_ids.add(predicted_track_id)
            matched_detection_indices.add(source_idx)
            # track_id_remap is a dict of candidate_track_id to predicted_track_id
            track_id_remap[new_detection_track_id] = predicted_track_id
            vetted_rows.append(updated_row)
            inferred_origin_updates[source_idx] = float(inferred_origin)

        if inferred_origin_updates and "source_index" in history_rows.columns:
            update_frame = pl.DataFrame(
                {
                    "source_index": list(inferred_origin_updates.keys()),
                    "inferred_origin": list(inferred_origin_updates.values()),
                }
            )
            history_rows = (
                history_rows.join(update_frame, on="source_index", how="left", suffix="_new")
                .with_columns(
                    pl.coalesce(
                        [pl.col("inferred_origin_new"), pl.col("inferred_origin")]
                    ).alias("inferred_origin")
                )
                .drop("inferred_origin_new")
            )
        # need to match the track_ids
        # recall track_id_remap[candidate_track_id] = predicted_track_id; from above
        if track_id_remap:
            for old_track_id, new_track_id in track_id_remap.items():
                merged_matches = self.merge_track_matches(old_track_id, new_track_id)
                history_rows = history_rows.with_columns(
                    pl.when(self._int_expr("track_id") == old_track_id)
                    .then(pl.lit(new_track_id))
                    .otherwise(pl.col("track_id"))
                    .alias("track_id")
                ).with_columns( #add new col "track_id"
                    pl.when(self._int_expr("track_id") == new_track_id)
                    .then(pl.lit(merged_matches))
                    .otherwise(pl.col("matches"))
                    .alias("matches") #rename added col "track_id" to "matches"
                )
        if vetted_rows: # True if vetted_rows is not empty
            vetted_frame = self._frame_from_dicts(vetted_rows)
            # here we are tracking the active vetted tracks in this frame
            unique_track_ids = (
                vetted_frame.get_column("track_id")
                .cast(pl.Int64, strict=False)
                .drop_nulls()
                .unique()
                .to_list()
            )

            for track_id in unique_track_ids:
                self._promote_confirmed_track_rows(int(track_id), vetted_frame)
        # aggregate the track_ids with predictions into a set
        # so that we can use set operations
        predicted_track_ids = {
            int(tid)
            for tid in (
                predicted_data.get_column("track_id")
                .cast(pl.Int64, strict=False)
                .drop_nulls()
                .to_list()
            )
        }
        # not every prediction will get a match
        strike_track_ids = predicted_track_ids - matched_pred_track_ids
        # and some will be inward predictions
        strike_track_ids = strike_track_ids.union(inward_pred_track_ids)
        # these all get a strike
        if strike_track_ids: # if not empty
            # increment a strike for these tracks
            for track_id in strike_track_ids:
                self.candidate_sources = self.candidate_sources.with_columns(
                    pl.when(self._int_expr("track_id") == track_id)
                    .then(self._int_expr("strikes").fill_null(0) + 1)
                    .otherwise(pl.col("strikes"))
                    .alias("strikes")
                )
        # de-increment strikes for the matched tracks
        if matched_pred_track_ids:
            for track_id in matched_pred_track_ids:
                self.candidate_sources = self.candidate_sources.with_columns(
                    pl.when(self._int_expr("track_id") == track_id)
                    .then((self._int_expr("strikes").fill_null(0) - 1).clip(lower_bound=0))
                    .otherwise(pl.col("strikes"))
                    .alias("strikes")
                )
        # of course, some sources have just appeared in this frame
        unmatched_sources = (
            sources_this_frame.with_row_index("source_index")
            .filter(~pl.col("source_index").is_in(list(matched_detection_indices)))
            .drop("source_index")
            .select(SOURCE_COLUMNS)
        )
        # treat them as new candidates
        if not unmatched_sources.is_empty():
            self.candidate_sources = pl.concat(
                [self.candidate_sources, unmatched_sources],
                how="vertical",
            )

        # append all detections to the history df
        self._append_history_sources(history_rows)
        # a bit redundant but need to convert these values to ints
        # for the unique operation
        active_state = ( #add cols track_id_num and frame_num
            self.candidate_sources.with_columns(
                track_id_num=self._int_expr("track_id"),
                frame_num=self._int_expr("frames"),
            )
            .drop_nulls(subset=["track_id_num", "frame_num"])
            .sort("frame_num")
        )
        if not active_state.is_empty():
            # collapse to one row per track_id
            # retaining the row from the most recent frame
            self.candidate_sources = (
                active_state.unique(subset=["track_id_num"], keep="last")
                .select(SOURCE_COLUMNS) #drop helper cols track_id_num and frame_num
                .with_columns(
                    pl.col("track_id")
                    .cast(pl.Int64, strict=False)
                    .map_elements( #use track_id to pull match count
                        lambda tid: (
                            self.get_track_matches(int(tid))
                            if tid is not None
                            else None
                        ),
                        return_dtype=pl.Int64,
                    )
                    .alias("matches") #rename added col "track_id" to "matches"
                )
            )
        # TODO somehow a track gets a detection added to it many frames later
        # how can this happen? Might be during stitching of tracks, or from
        # a track with immunity that is getting matched much later than expected
        # stitching of tracks not likely bc some tracks are only getting a single detection
        # much later, which is not a valid track typically. I think candidate_sources
        # is not getting cleared out correctly, and thus also predicted_sources.
        # Indeed, immune tracks are just left alone and the Hungarian algo is then
        # just always looking for a match even in much later frames.
        rejected_rows = self.candidate_sources.filter(
            (self._int_expr("strikes").fill_null(0) > self.max_missed_frames)
            & (self._int_expr("matches").fill_null(0) < self.prune_immunity_matches)
        )
        self._append_terminal_rejected_sources(
            rejected_rows,
            reject_reason="max_missed_frames",
        )
        prior_ids = set(
            self.candidate_sources.get_column("track_id")
            .cast(pl.Int64, strict=False)
            .drop_nulls()
            .to_list()
        )
        self.candidate_sources = (
            self.candidate_sources.filter(
                (self._int_expr("strikes").fill_null(0) <= self.max_missed_frames)
                # | (self._int_expr("matches").fill_null(0) >= self.prune_immunity_matches)
            )
            .select(SOURCE_COLUMNS)
        )
        kept_ids = set(
            self.candidate_sources.get_column("track_id")
            .cast(pl.Int64, strict=False)
            .drop_nulls()
            .to_list()
        )
        for removed_track_id in (prior_ids - kept_ids):
            self.track_match_counts.pop(int(removed_track_id), None)
        return self.candidate_sources

    def get_dynamic_window_centers(
        self,
        min_matches: int,
        frame_index: int,
    ) -> list[tuple[float, float]]:
        if self.candidate_sources.is_empty():
            return []

        candidate_data = (
            self.candidate_sources.select(
                [
                    self._numeric_expr("track_id").alias("track_id"),
                    self._numeric_expr("x_coords").alias("x"),
                    self._numeric_expr("y_coords").alias("y"),
                    self._numeric_expr("frames").alias("frames"),
                    self._numeric_expr("matches").alias("matches"),
                    self._numeric_expr("strikes").alias("strikes"),
                ]
            )
            .drop_nulls(subset=["track_id", "x", "y", "frames", "matches", "strikes"])
            .filter(
                (pl.col("matches") >= max(min_matches, self.confirmation_matches))
                & (pl.col("strikes") <= self.max_missed_frames)
                & ((frame_index - pl.col("frames")) <= self.max_missed_frames)
            )
        )
        if candidate_data.is_empty():
            return []

        candidate_data = candidate_data.filter(
            pl.col("track_id").cast(pl.Int64, strict=False).is_in(list(self.confirmed_track_ids))
        )
        if candidate_data.is_empty():
            return []
        return [
            (float(x), float(y))
            for x, y in candidate_data.select(["x", "y"]).iter_rows()
        ]

    def track(
        self,
        cc_frame: np.ndarray,
        cc_frame_ind: int,
        image_center: tuple[int, int],
        error_map: np.ndarray | None,
        tripwire_mask: np.ndarray,
        sep_thresh: float,
        sep_minarea: int,
        meters_per_pixel: float,
    ) -> pl.DataFrame:
        if cc_frame_ind <= 5:
            print("cold start")
            return self.candidate_sources

        sources_in_frame = self.extract(
            cc_frame,
            error_map,
            tripwire_mask,
            sep_thresh,
            sep_minarea,
            image_center,
            cc_frame_ind,
            self.time_per_frame,
            meters_per_pixel,
        )
        self.candidate_sources = self.match(
            sources_this_frame=sources_in_frame,
            frame_index=cc_frame_ind,
        )
        self.predicted_sources = self.predict(
            frame_index=cc_frame_ind,
            image_center=image_center,
            meters_per_pixel=meters_per_pixel,
        )
        return self.candidate_sources
