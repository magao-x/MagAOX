"""Aggregate wind JSON from ``ws_measure`` and run wind-layer statistics (HDBSCAN)."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
from datetime import datetime

import numpy as np
import polars as pl

from windsocc.analysis.wind_stats import (
    cluster_centroids_table,
    cluster_wind_tracks_hdbscan,
    layer_reference_points_from_date_obs,
    per_cluster_vu_vv_stats,
)
from windsocc.io.config_handling import parse_config_file
from windsocc.io.fits_handling import convert_time_to_datetime, extract_time_from_fname
from windsocc.visualization.plot_measure_results import (
    plot_wind_direction_vs_time_for_clusters,
    plot_wind_track_clusters,
    write_wind_cluster_stats_report,
)

WIND_DATA_SCHEMA: dict[str, pl.DataType] = {
    "track_id": pl.Int64,
    "direction": pl.Float64,
    "velocity_m_per_s": pl.Float64,
    "matches": pl.Int64,
    "flux": pl.Float64,
    "source_area": pl.Float64,
    "inferred_origin": pl.Float64,
    "raw_direction": pl.Float64,
    "time": pl.Datetime,
    "time_raw": pl.String,
}

WIND_DATA_COLUMNS = list(WIND_DATA_SCHEMA.keys())


def _parse_two_float_range(
    raw_value: object,
    config_key: str,
) -> tuple[float, float] | None:
    """Parse a two-value config entry into a float pair."""
    if raw_value is None:
        return None
    if isinstance(raw_value, np.ndarray):
        values = raw_value.ravel().tolist()
    elif isinstance(raw_value, (list, tuple)):
        values = list(raw_value)
    else:
        logging.warning("%s must be a list/tuple with two float values.", config_key)
        return None
    if len(values) != 2:
        logging.warning("%s must contain exactly two values; got %d.", config_key, len(values))
        return None
    try:
        low = float(values[0])
        high = float(values[1])
    except (TypeError, ValueError):
        logging.warning("%s values must be numeric: %r", config_key, values)
        return None
    if not np.isfinite(low) or not np.isfinite(high):
        logging.warning("%s contains non-finite values: %r", config_key, values)
        return None
    return low, high


def _load_lco_surface_wind_points(
    csv_path: str,
    time_min: datetime,
    time_max: datetime,
) -> list[dict[str, object]]:
    """Load LCO wind rows in the observation window and map to ``(vu, vv)``."""
    if not os.path.isfile(csv_path):
        logging.warning("LCO wind CSV not found: %s", csv_path)
        return []
    try:
        lco_df = pl.read_csv(csv_path)
    except Exception as exc:
        logging.warning("Could not read LCO wind CSV %s: %s", csv_path, exc)
        return []
    required_cols = {"ts", "wind_dir_avg", "wind_speed_avg"}
    if not required_cols.issubset(set(lco_df.columns)):
        logging.warning(
            "LCO wind CSV %s missing required columns: %s",
            csv_path,
            sorted(required_cols),
        )
        return []

    lco_df = lco_df.with_columns(
        pl.col("ts").str.to_datetime(strict=False).alias("ts_dt"),
        pl.col("wind_dir_avg").cast(pl.Float64, strict=False).alias("wind_dir_avg_f64"),
        pl.col("wind_speed_avg").cast(pl.Float64, strict=False).alias("wind_speed_avg_f64"),
    ).filter(
        pl.col("ts_dt").is_not_null()
        & pl.col("wind_dir_avg_f64").is_finite()
        & pl.col("wind_speed_avg_f64").is_finite()
        & (pl.col("ts_dt") >= pl.lit(time_min))
        & (pl.col("ts_dt") <= pl.lit(time_max))
    )
    if lco_df.is_empty():
        return []

    # Clip obvious speed outliers before component conversion.
    # A broad +/- 5 sigma gate preserves normal variability while removing spikes.
    speed_sigma = 5.0
    speed_vals = lco_df["wind_speed_avg_f64"].to_numpy()
    speed_mean = float(np.mean(speed_vals))
    speed_std = float(np.std(speed_vals))
    if np.isfinite(speed_std) and speed_std > 0.0:
        speed_min = speed_mean - speed_sigma * speed_std
        speed_max = speed_mean + speed_sigma * speed_std
        n_before = lco_df.height
        lco_df = lco_df.filter(
            (pl.col("wind_speed_avg_f64") >= speed_min)
            & (pl.col("wind_speed_avg_f64") <= speed_max)
        )
        n_after = lco_df.height
        if n_after < n_before:
            logging.info(
                "LCO CSV sigma clip removed %d/%d rows outside %.2f +/- %.1f sigma.",
                n_before - n_after,
                n_before,
                speed_mean,
                speed_sigma,
            )
    if lco_df.is_empty():
        return []

    lco_df = lco_df.with_columns(
        # LCO CSV direction is already in the analysis convention.
        wind_dir_corr_deg=pl.col("wind_dir_avg_f64"),
    )

    out: list[dict[str, object]] = []
    for row in lco_df.select(
        ["ts_dt", "wind_speed_avg_f64", "wind_dir_corr_deg"]
    ).iter_rows(named=True):
        speed = float(row["wind_speed_avg_f64"])
        direction_deg = float(row["wind_dir_corr_deg"])
        theta_rad = math.radians(direction_deg)
        out.append(
            {
                "time": row["ts_dt"],
                "speed_m_per_s": speed,
                "direction_deg": direction_deg,
                "vu": float(speed * math.cos(theta_rad)),
                "vv": float(speed * math.sin(theta_rad)),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the results and calculate statistics of ws_measure output."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default=".",
        help="Path to the camwfs directory containing ws_config.yaml and measure_results/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output directory (default: data_dir/wind_stats)",
    )
    return parser.parse_args()


def _allocate_dirs(data_dir: str, output_dir: str) -> dict[str, str]:
    """Resolve measure and stats output paths."""
    dirs: dict[str, str] = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "measure_results_dir": os.path.join(data_dir, "measure_results"),
        "wind_data_dir": os.path.join(data_dir, "measure_results", "wind_data"),
        "wind_stats_dir": output_dir,
    }
    os.makedirs(output_dir, exist_ok=True)
    return dirs


def _load_wind_attributes_dataframe(wind_data_dir: str) -> pl.DataFrame:
    """Concatenate all ``*_wind_attributes.json`` rows into one frame."""
    pattern = os.path.join(wind_data_dir, "*_wind_attributes.json")
    wind_json_files = sorted(glob.glob(pattern))
    wind_df = pl.DataFrame(schema=WIND_DATA_SCHEMA)
    for wind_json_file in wind_json_files:
        time_raw = extract_time_from_fname(os.path.basename(wind_json_file))
        timestamp = convert_time_to_datetime(time_raw)
        with open(wind_json_file, encoding="utf-8") as f:
            data_json = json.load(f)
        tracks_data = data_json.get("tracks") or []
        for track_data in tracks_data:
            if not isinstance(track_data, dict):
                continue
            row: dict[str, object | None] = {
                c: track_data.get(c) for c in WIND_DATA_COLUMNS if c not in ("time", "time_raw")
            }
            row["time_raw"] = time_raw
            row["time"] = timestamp
            track_data_df = pl.DataFrame([row])
            for col in WIND_DATA_COLUMNS:
                track_data_df = track_data_df.with_columns(
                    pl.col(col).cast(WIND_DATA_SCHEMA[col], strict=False).alias(col)
                )
            # Match schema column order (dict insertion used time_raw before time).
            track_data_df = track_data_df.select(WIND_DATA_COLUMNS)
            wind_df = pl.concat([wind_df, track_data_df], how="vertical")
    return wind_df


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(data_dir, "wind_stats")
    output_dir = os.path.abspath(output_dir)

    dirs_dict = _allocate_dirs(data_dir, output_dir)
    path_yaml = os.path.join(data_dir, "ws_config.yaml")
    config_params = parse_config_file(path_yaml)
    date_obs = config_params.get("DATE_OBS", None)
    if date_obs is None:
        logging.warning("DATE_OBS not found in config file; using default value.")
        date_obs = "Unknown"

    min_cluster_size = int(np.asarray(config_params.get("MIN_CLUSTER_SIZE", 5)).item())
    cluster_selection_epsilon = float(
        np.asarray(config_params.get("CLUSTER_SELECTION_EPSILON", 0.0)).item()
    )

    layer_sigma = float(np.asarray(config_params.get("WIND_CLUSTER_SIGMA", 3.0)).item())
    u_component_range = _parse_two_float_range(
        config_params.get("U_COMPONENT_RANGE"),
        "U_COMPONENT_RANGE",
    )
    v_component_range = _parse_two_float_range(
        config_params.get("V_COMPONENT_RANGE"),
        "V_COMPONENT_RANGE",
    )

    wind_df = _load_wind_attributes_dataframe(dirs_dict["wind_data_dir"])
    if wind_df.is_empty():
        logging.warning(
            "No rows loaded from %s; exiting stats stage without cluster outputs.",
            dirs_dict["wind_data_dir"],
        )
        return

    rad_per_deg = math.pi / 180.0
    wind_df = wind_df.with_columns(
        theta_r=pl.col("direction").cast(pl.Float64) * rad_per_deg,
    ).with_columns(
        vu=pl.col("velocity_m_per_s").cast(pl.Float64) * pl.col("theta_r").cos(),
        vv=pl.col("velocity_m_per_s").cast(pl.Float64) * pl.col("theta_r").sin(),
    ).drop("theta_r")

    wind_feat = wind_df.filter(
        pl.col("vu").is_finite()
        & pl.col("vv").is_finite()
        & pl.col("direction").is_finite()
        & pl.col("velocity_m_per_s").is_finite()
        & pl.col("time").is_not_null()
    )
    if wind_feat.is_empty():
        logging.warning("No finite wind rows after filtering; exiting stats stage.")
        return

    X = wind_feat.select(["vu", "vv"]).to_numpy()
    if X.shape[0] < min_cluster_size:
        logging.warning(
            "Only %d wind rows (< MIN_CLUSTER_SIZE=%d); skipping HDBSCAN.",
            X.shape[0],
            min_cluster_size,
        )
        return

    labels, probabilities, _hdb = cluster_wind_tracks_hdbscan(
        X,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    stats_rows, noise_count = per_cluster_vu_vv_stats(X, labels)
    centroids = cluster_centroids_table(X, labels)

    out_dir = dirs_dict["wind_stats_dir"]
    cluster_plot = os.path.join(out_dir, "wind_track_clusters")
    direction_plot = os.path.join(out_dir, "wind_direction_vs_time_clusters")
    cluster_txt = os.path.join(out_dir, "wind_track_stats.txt")
    layer_reference_points = layer_reference_points_from_date_obs(date_obs)
    lco_surface_points: list[dict[str, object]] = []
    lco_wind_csv = config_params.get("LCO_WIND_CSV")
    if lco_wind_csv:
        lco_wind_csv_path = str(lco_wind_csv)
        time_min = wind_feat["time"].min()
        time_max = wind_feat["time"].max()
        if time_min is not None and time_max is not None:
            lco_surface_points = _load_lco_surface_wind_points(
                csv_path=lco_wind_csv_path,
                time_min=time_min,
                time_max=time_max,
            )


    write_wind_cluster_stats_report(path=cluster_txt, rows=stats_rows, noise_count=noise_count)
    plot_wind_track_clusters(
        vu=X[:, 0],
        vv=X[:, 1],
        labels=labels,
        probabilities=probabilities,
        output_plot_fname=cluster_plot,
        layer_reference_points=layer_reference_points,
        lco_surface_points=lco_surface_points,
        u_component_range=u_component_range,
        v_component_range=v_component_range,
    )

    wind_feat_labeled = wind_feat.with_columns(
        pl.Series(name="cluster_id", values=labels)
    )
    cluster_direction_frames: list[tuple[int, pl.DataFrame, float]] = []
    for c in centroids:
        cid = int(c["cluster_id"])
        sub = wind_feat_labeled.filter(pl.col("cluster_id") == cid).drop("cluster_id")
        if sub.is_empty():
            logging.info(
                "Cluster %d: no points assigned by HDBSCAN; skipping direction plot.",
                cid,
            )
            continue
        cluster_direction_frames.append((cid, sub, float(c["mean_scalar_speed"])))
        logging.info(
            "Cluster %d: retained %d HDBSCAN-assigned points (no sigma clipping).",
            cid,
            sub.height,
        )
    plot_wind_direction_vs_time_for_clusters(
        cluster_direction_frames,
        output_plot_fname=direction_plot,
        sigma=layer_sigma,
        date_obs=date_obs,
        lco_surface_points=lco_surface_points,
    )
    logging.info("Wrote consolidated direction plot: %s", direction_plot)

    logging.info("Wind stats written under %s", out_dir)


if __name__ == "__main__":
    main()
