"""Realtime orchestration for the WindsoCC pipeline.


TODO check the pupil coordinates using references made from
today's data
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
from astropy.io import fits

from windsocc.distill import run_distill_stage, run_distill_stage_in_memory
from windsocc.measure import run_measure_stage
from windsocc.reduce import (
    get_pupil_geometry,
    parse_config_file,
    process_batch_in_memory,
    process_dataset,
    save_reduced_quadrant_cubes,
)
from windsocc.xcorr import run_xcorr_stage, run_xcorr_stage_in_memory

DEFAULT_FRAMES_PER_CUBE = 512
DEFAULT_FRAME_SHAPE = (120, 120)

# Embedded callers treat the entrypoints below as a cross-language ABI. Keep the
# callable names, positional argument order, keyword names, float32 buffer
# expectation, and return-dict keys stable unless the C++ caller changes too.


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_timestamp(value) -> datetime:
    """Normalize supported timestamp inputs to timezone-aware UTC datetimes.

    Embedded callers currently pass either:
    - a timezone-aware or naive `datetime`
    - a Unix timestamp as `int` or `float`
    - a string in ISO-8601 form
    - a compact UTC string in `YYYYMMDDTHHMMSSffffff` format

    The compact form matches `windsoccRT.formatTimestamp()` and is treated as UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = datetime.strptime(normalized, "%Y%m%dT%H%M%S%f")
            parsed = parsed.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise TypeError(f"Unsupported timestamp type: {type(value)!r}")


def format_batch_timestamp(timestamp: datetime) -> str:
    """Format timestamps consistently for run directories and filenames."""
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


def parity_flip_needed_from_hour_angle(ha_deg: float) -> bool:
    """Infer parity-flip bookkeeping from hour angle (degrees) alone.

    Uses the usual equatorial convention: negative HA is east of the meridian
    (before transit), positive HA is west (after transit). Here
    "parity_flip_needed" is True when ``ha_deg > 0`` (after meridian). At
    ``ha_deg == 0`` returns False.
    """
    return ha_deg > 0.0


def _indi_number_member_to_float(node) -> float:
    """Best-effort float from a purepyindi2 number element or mapping."""
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, dict):
        for key in ("current", "value"):
            if key in node and node[key] is not None:
                return float(node[key])
    for attr in ("value",):
        if hasattr(node, attr):
            v = getattr(node, attr)
            if v is not None:
                return float(v)
    if hasattr(node, "__getitem__"):
        for key in ("current", "value"):
            try:
                return float(node[key])
            except (KeyError, TypeError, ValueError):
                continue
    return float(node)


def _construct_indi_client(host: str, port: int):
    """Instantiate ``IndiClient`` with host/port when the installed API supports it."""
    import purepyindi2 as indi

    cls = indi.client.IndiClient
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
    except (TypeError, ValueError):
        return cls()

    kwargs = {}
    if "host" in params:
        kwargs["host"] = host
    if "port" in params:
        kwargs["port"] = port
    if "hostname" in params and "host" not in kwargs:
        kwargs["hostname"] = host
    if "address" in params:
        kwargs["address"] = (host, port)
    try:
        return cls(**kwargs) if kwargs else cls()
    except TypeError:
        return cls()


WINDSOC_INDI_DEVICE_NAME = "windsocc"
_windsoc_indi_publisher = None


class _WindsocIndiPublisher:
    """Publish windsocc wind layers to INDI using a fixed property schema."""

    def __init__(self, max_layers: int):
        self.max_layers = int(max_layers)

        try:
            from magaox.indi.device import XDevice
            from purepyindi2 import properties, constants
            from purepyindi2.messages import DefNumber
        except ImportError as exc:
            raise RuntimeError(
                "INDI publishing requires optional dependency `purepyindi2` "
                "and MagAOX Python INDI wrapper modules."
            ) from exc

        # Minimal config object; MagAOX XDevice only requires this attribute.
        class _DummyConfig:
            sleep_interval_sec = 1.0

        self._device = XDevice(WINDSOC_INDI_DEVICE_NAME, _DummyConfig())

        # `windsocc.nlayers.current`
        nlayers_prop = properties.NumberVector(
            name="nlayers",
            perm=constants.PropertyPerm.READ_ONLY,
        )
        nlayers_prop.add_element(
            DefNumber(
                name="current",
                label="Number of wind layers",
                format="%i",
                min=0,
                max=self.max_layers,
                step=1,
                _value=0,
            )
        )
        self._device.add_property(nlayers_prop)

        # `windsocc.layer_XX.{speed,dir,str}`
        for i in range(self.max_layers):
            layer_prop = properties.NumberVector(
                name=f"layer_{i:02d}",
                perm=constants.PropertyPerm.READ_ONLY,
            )
            layer_prop.add_element(
                DefNumber(
                    name="speed",
                    label=f"Layer {i:02d} speed (m/s)",
                    format="%0.4f",
                    min=-1e6,
                    max=1e6,
                    step=0.0001,
                    _value=0.0,
                )
            )
            layer_prop.add_element(
                DefNumber(
                    name="dir",
                    label=f"Layer {i:02d} dir (deg)",
                    format="%0.3f",
                    min=0.0,
                    max=360.0,
                    step=0.001,
                    _value=0.0,
                )
            )
            layer_prop.add_element(
                DefNumber(
                    name="str",
                    label=f"Layer {i:02d} rel strength",
                    format="%0.4f",
                    min=0.0,
                    max=1.0,
                    step=0.0001,
                    _value=0.0,
                )
            )
            self._device.add_property(layer_prop)

        # Ensure subscribers see a defined state immediately.
        self.update_layers([])

    def update_layers(self, layers: list[dict[str, float]]):
        """Update all INDI properties for a new wind layer set."""
        nlayers_current = min(len(layers), self.max_layers)

        nlayers_prop = self._device.properties["nlayers"]
        nlayers_prop["current"] = float(nlayers_current)
        self._device.update_property(nlayers_prop)

        for i in range(self.max_layers):
            layer_name = f"layer_{i:02d}"
            layer_prop = self._device.properties[layer_name]

            if i < nlayers_current:
                layer = layers[i]
                layer_prop["speed"] = float(layer.get("speed", 0.0))
                layer_prop["dir"] = float(layer.get("dir", 0.0))
                layer_prop["str"] = float(layer.get("str", 0.0))
            else:
                # Keep arrow length at 0 by forcing normalized strength to 0.
                layer_prop["speed"] = 0.0
                layer_prop["dir"] = 0.0
                layer_prop["str"] = 0.0

            self._device.update_property(layer_prop)


def _maybe_get_windsoc_publisher(max_layers: int) -> _WindsocIndiPublisher | None:
    global _windsoc_indi_publisher

    if (
        _windsoc_indi_publisher is not None
        and _windsoc_indi_publisher.max_layers == int(max_layers)
    ):
        return _windsoc_indi_publisher

    try:
        _windsoc_indi_publisher = _WindsocIndiPublisher(max_layers=max_layers)
        return _windsoc_indi_publisher
    except Exception as exc:
        logging.warning("Could not initialize windsocc INDI publisher: %s", exc)
        _windsoc_indi_publisher = None
        return None


def fetch_hour_angle_deg(config_params: dict) -> float | None:
    """Read ``telpos.ha`` (degrees) from INDI when ``FETCH_HOUR_ANGLE`` is true.

    Requires optional dependency ``purepyindi2`` (``pip install windsocc[indi]``).
    Returns ``None`` if disabled, import fails, or the read errors.
    """
    if not config_params.get("FETCH_HOUR_ANGLE", False):
        return None
    try:
        import purepyindi2  # noqa: F401
    except ImportError:
        logging.warning(
            "FETCH_HOUR_ANGLE is enabled but purepyindi2 is not installed "
            "(install: pip install 'windsocc[indi]')."
        )
        return None

    device = str(config_params.get("TCS_INDI_DEVICE", "tcsi"))
    host = str(config_params.get("INDI_HOST", "127.0.0.1"))
    port = int(config_params.get("INDI_PORT", 7624))

    try:
        client = _construct_indi_client(host, port)
        client.connect()
        client.wait_to_connect()
        client.get_properties_and_wait(device)
        telpos = client[device]["telpos"]
        ha_deg = _indi_number_member_to_float(telpos["ha"])
        logging.info("INDI telpos.ha = %.6f deg (device=%s)", ha_deg, device)
        return ha_deg
    except Exception as exc:
        logging.warning("Could not read hour angle from INDI: %s", exc)
        return None


@dataclass
class FrameBatch:
    """A collected batch of camwfs frames."""

    frames: np.ndarray
    first_timestamp: datetime


@dataclass
class BatchRunSummary:
    """Structured summary of one realtime batch execution."""

    run_dir: str
    raw_cube_paths: list[str]
    timings_s: dict[str, float]
    json_paths: list[str]
    movie_paths: list[str]
    hour_angle_deg: float | None = None
    parity_flip_needed: bool | None = None


class FrameSource(ABC):
    """Abstract frame source for realtime and offline development."""

    @abstractmethod
    def collect_frames(self, target_frames: int) -> FrameBatch:
        """Collect a fixed number of frames."""


class OfflineFitsFrameSource(FrameSource):
    """Read frames from FITS files for local testing and validation."""

    def __init__(self, source_path: str):
        self.source_path = source_path

    def _iter_fits_paths(self) -> list[str]:
        if os.path.isdir(self.source_path):
            return sorted(
                os.path.join(self.source_path, name)
                for name in os.listdir(self.source_path)
                if name.endswith(".fits")
            )
        if os.path.isfile(self.source_path) and self.source_path.endswith(".fits"):
            return [self.source_path]
        raise FileNotFoundError(
            f"Offline FITS source {self.source_path} is not a FITS file or directory."
        )

    @staticmethod
    def _timestamp_from_header(header) -> datetime | None:
        for key in ("DATE-OBS", "DATE", "DATE_OBS"):
            value = header.get(key)
            if not value:
                continue
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
        return None

    def collect_frames(self, target_frames: int) -> FrameBatch:
        frames = []
        first_timestamp = None
        for fits_path in self._iter_fits_paths():
            with fits.open(fits_path) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
                header_timestamp = self._timestamp_from_header(hdul[0].header)
            if first_timestamp is None:
                first_timestamp = header_timestamp or _utcnow()

            if data.ndim == 2:
                frames.append(data)
            elif data.ndim == 3:
                frames.extend(np.asarray(frame, dtype=np.float32) for frame in data)
            else:
                raise ValueError(f"Unsupported FITS dimensionality for {fits_path}: {data.shape}")

            if len(frames) >= target_frames:
                break

        if len(frames) < target_frames:
            raise ValueError(
                f"Needed {target_frames} frames from offline source, found only {len(frames)}."
            )

        batch = np.stack(frames[:target_frames], axis=0)
        return FrameBatch(frames=batch, first_timestamp=first_timestamp or _utcnow())


class MagAOXCircularBufferFrameSource(FrameSource):
    """
    Adapter for a MagAO-X circular buffer reader.

    The actual shared-memory/buffer access is intentionally isolated here so
    the rest of the realtime pipeline can be tested offline. A reader callable
    can be supplied from an external module when Python bindings are available.
    """

    def __init__(
        self,
        stream_name: str,
        reader_callable: str | None = None,
        timeout_s: float = 5.0,
    ):
        self.stream_name = stream_name
        self.reader_callable = reader_callable
        self.timeout_s = timeout_s

    def _load_reader(self):
        if self.reader_callable is None:
            raise RuntimeError(
                "No Python buffer reader was provided. Pass --reader-callable "
                "module:function for live MagAO-X acquisition, or use --source-type offline-fits."
            )
        module_name, function_name = self.reader_callable.split(":", 1)
        module = importlib.import_module(module_name)
        reader = getattr(module, function_name)
        return reader

    def collect_frames(self, target_frames: int) -> FrameBatch:
        reader = self._load_reader()
        result = reader(
            stream_name=self.stream_name,
            frame_count=target_frames,
            timeout_s=self.timeout_s,
        )
        if isinstance(result, FrameBatch):
            return result
        if isinstance(result, tuple) and len(result) == 2:
            frames, first_timestamp = result
            return FrameBatch(
                frames=np.asarray(frames, dtype=np.float32),
                first_timestamp=first_timestamp,
            )
        raise TypeError(
            "Reader callable must return FrameBatch or (frames, first_timestamp)."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one realtime-style WindsoCC batch in-process."
    )
    parser.add_argument(
        "--source-type",
        choices=("offline-fits", "magaox-buffer"),
        default="offline-fits",
        help="Frame source backend.",
    )
    parser.add_argument(
        "--offline-source",
        type=str,
        default=None,
        help="Path to FITS file or directory for offline validation.",
    )
    parser.add_argument(
        "--stream-name",
        type=str,
        default="camwfs",
        help="MagAO-X stream name for live acquisition.",
    )
    parser.add_argument(
        "--reader-callable",
        type=str,
        default=None,
        help="Import path for a live reader callback as module:function.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to ws_config.yaml used for the batch.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help="Directory where camwfs_<timestamp> batch directories should be written.",
    )
    parser.add_argument(
        "--integration-seconds",
        type=float,
        default=10.0,
        help="Length of one processing batch in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2000.0,
        help="Frame rate used to convert integration time into frame count.",
    )
    parser.add_argument(
        "--frames-per-cube",
        type=int,
        default=DEFAULT_FRAMES_PER_CUBE,
        help="Number of raw frames to pack into each intermediate FITS cube.",
    )
    parser.add_argument(
        "--no-movie",
        action="store_true",
        help="Skip movie generation even if MAKE_MOVIE is enabled in config.",
    )
    parser.add_argument(
        "--save-distill-pngs",
        action="store_true",
        help="Keep distill PNG products; default skips them for lower latency.",
    )
    parser.add_argument(
        "--cleanup-intermediate",
        action="store_true",
        help="Delete xcorr and distill products after measure completes.",
    )
    return parser.parse_args()


def build_frame_source(args) -> FrameSource:
    if args.source_type == "offline-fits":
        if args.offline_source is None:
            raise ValueError("--offline-source is required for --source-type offline-fits")
        return OfflineFitsFrameSource(args.offline_source)
    return MagAOXCircularBufferFrameSource(
        stream_name=args.stream_name,
        reader_callable=args.reader_callable,
    )


def required_batch_frames(fps: float, integration_seconds: float) -> int:
    return max(1, int(round(fps * integration_seconds)))


def validate_frame_batch(frames: np.ndarray, frame_shape: tuple[int, int] = DEFAULT_FRAME_SHAPE):
    if frames.ndim != 3:
        raise ValueError(f"Expected a 3D batch of frames, got shape {frames.shape}")
    if tuple(frames.shape[1:]) != tuple(frame_shape):
        raise ValueError(
            f"Expected frame shape {frame_shape}, got {tuple(frames.shape[1:])}"
        )


def materialize_batch_directory(
    batch: FrameBatch,
    output_root: str,
    config_path: str,
    file_prefix: str,
    frames_per_cube: int,
) -> tuple[str, list[str]]:
    """Create a batch directory and write fixed-length raw cubes."""
    batch_timestamp = format_batch_timestamp(batch.first_timestamp)
    run_dir = os.path.join(output_root, f"camwfs_{batch_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(run_dir, "ws_config.yaml"))

    n_cubes = batch.frames.shape[0] // frames_per_cube
    if n_cubes == 0:
        raise ValueError(
            f"Need at least {frames_per_cube} frames to materialize one raw cube."
        )

    usable_frames = n_cubes * frames_per_cube
    frame_array = batch.frames[:usable_frames]
    raw_cube_paths = []
    for cube_idx in range(n_cubes):
        start = cube_idx * frames_per_cube
        stop = start + frames_per_cube
        cube = frame_array[start:stop]
        cube_name = f"{file_prefix}{batch_timestamp}_{cube_idx:05d}.fits"
        cube_path = os.path.join(run_dir, cube_name)
        fits.writeto(cube_path, cube.astype(np.float32), overwrite=True)
        raw_cube_paths.append(cube_path)

    if usable_frames < batch.frames.shape[0]:
        logging.info(
            "Dropped %d trailing frames to preserve %d-frame cube boundaries.",
            batch.frames.shape[0] - usable_frames,
            frames_per_cube,
        )

    return run_dir, raw_cube_paths


def prepare_batch_run_directory(
    batch: FrameBatch,
    output_root: str,
    config_path: str,
) -> str:
    """Create the realtime run directory without writing raw cubes."""
    batch_timestamp = format_batch_timestamp(batch.first_timestamp)
    run_dir = os.path.join(output_root, f"camwfs_{batch_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(config_path, os.path.join(run_dir, "ws_config.yaml"))
    return run_dir


def resolve_reduce_settings(config_params: dict) -> dict:
    return {
        "nstack": config_params.get("NSTACK"),
        "start_frame": int(config_params.get("START_FRAME", 0)),
        "step_frame": int(config_params.get("STEP_FRAME", 1)),
        "group_size": int(config_params.get("GROUP_SIZE", 8)),
        "create_reference": bool(config_params.get("CREATE_REFERENCE", True)),
        "remake_reference": bool(
            config_params.get("REMAKE_REFERENCE", config_params.get("REMAKE_REF", False))
        ),
        "skip_dark": bool(config_params.get("SKIP_DARK", config_params.get("SUBTRACT_DARK", False))),
        "subtract_reference": bool(config_params.get("SUBTRACT_REFERENCE", True)),
        "inspect_reduction": bool(config_params.get("INSPECT_REDUCTION", False)),
    }


def resolve_safe_segment_cubes(
    config_params: dict,
    raw_cube_count: int,
    frames_per_cube: int,
) -> int:
    """Choose a segment_cube_count that is valid for the current batch."""
    group_size = int(config_params.get("GROUP_SIZE", 8))
    if frames_per_cube % group_size != 0:
        raise ValueError(
            f"frames_per_cube={frames_per_cube} must be divisible by GROUP_SIZE={group_size}"
        )
    reduced_frames_per_cube = frames_per_cube // group_size
    max_delay = config_params.get("MAX_DELAY")
    if max_delay is None:
        raise ValueError("MAX_DELAY must be set in ws_config.yaml for realtime xcorr.")

    desired = int(config_params.get("SEGMENT_CUBES", config_params.get("SEGMENT_LENGTH", 22)))
    total_reduced_frames = raw_cube_count * reduced_frames_per_cube
    max_valid_segment_cubes = (total_reduced_frames - int(max_delay)) // reduced_frames_per_cube
    if max_valid_segment_cubes < 1:
        raise ValueError(
            "Not enough frames in this batch for xcorr with the current GROUP_SIZE/MAX_DELAY settings."
        )
    return max(1, min(desired, max_valid_segment_cubes))


def maybe_cleanup_intermediate(run_dir: str, config_params: dict):
    """Remove distill products after the final outputs are written."""
    distill_dir = os.path.join(run_dir, config_params.get("DISTILL_DIR", "distill_results"))
    if os.path.exists(distill_dir):
        shutil.rmtree(distill_dir)


def write_batch_summary(summary: BatchRunSummary):
    summary_path = os.path.join(summary.run_dir, "realtime_batch_summary.json")
    payload = {
        "run_dir": summary.run_dir,
        "raw_cube_paths": summary.raw_cube_paths,
        "timings_s": summary.timings_s,
        "json_paths": summary.json_paths,
        "movie_paths": summary.movie_paths,
    }
    if summary.hour_angle_deg is not None:
        payload["hour_angle_deg"] = summary.hour_angle_deg
    if summary.parity_flip_needed is not None:
        payload["parity_flip_needed"] = summary.parity_flip_needed
    with open(summary_path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


def process_collected_batch(
    batch: FrameBatch,
    config_path: str,
    output_root: str,
    frames_per_cube: int = DEFAULT_FRAMES_PER_CUBE,
    no_movie: bool = False,
    save_distill_pngs: bool = False,
    cleanup_intermediate: bool = False,
    initial_timings_s: dict[str, float] | None = None,
) -> BatchRunSummary:
    """Run the realtime pipeline for an already collected frame batch."""
    config_params = parse_config_file(config_path)
    file_prefix = config_params.get("FILE_PREFIX", "camwfs_")
    validate_frame_batch(batch.frames)

    hour_angle_deg: float | None = None
    parity_flip_needed: bool | None = None
    if config_params.get("FETCH_HOUR_ANGLE", False):
        hour_angle_deg = fetch_hour_angle_deg(config_params)
        if hour_angle_deg is not None:
            parity_flip_needed = parity_flip_needed_from_hour_angle(hour_angle_deg)
            logging.info(
                "Hour angle %.6f deg -> parity_flip_needed=%s",
                hour_angle_deg,
                parity_flip_needed,
            )

    timings_s = dict(initial_timings_s or {})

    t0 = perf_counter()
    run_dir = prepare_batch_run_directory(
        batch=batch,
        output_root=output_root,
        config_path=config_path,
    )
    timings_s["run_dir_setup"] = perf_counter() - t0

    t0 = perf_counter()
    reduce_settings = resolve_reduce_settings(config_params)
    pupil_centers, pupil_mask_radius = get_pupil_geometry(config_params)
    reduce_result = process_batch_in_memory(
        batch.frames,
        run_dir,
        run_dir,
        reduce_settings["nstack"],
        pupil_centers,
        pupil_mask_radius,
        frames_per_cube,
        reduce_settings["start_frame"],
        reduce_settings["step_frame"],
        reduce_settings["group_size"],
        reduce_settings["create_reference"],
        reduce_settings["remake_reference"],
        reduce_settings["skip_dark"],
        subtract_reference=reduce_settings["subtract_reference"],
        inspect_reduction=reduce_settings["inspect_reduction"],
    )
    if reduce_result["dropped_frames"] > 0:
        logging.info(
            "Dropped %d trailing frames to preserve %d-frame cube boundaries.",
            reduce_result["dropped_frames"],
            frames_per_cube,
        )
    reduce_result["group_suffix"] = f"{file_prefix}{format_batch_timestamp(batch.first_timestamp)}_00000"
    if reduce_settings["inspect_reduction"]:
        reduce_dir = config_params.get("REDUCE_DIR", "reduce_results")
        save_reduced_quadrant_cubes(
            run_dir,
            reduce_result["reduced_quadrants"],
            reduce_result["group_suffix"],
            reduce_dir_name=reduce_dir,
            max_frames=reduce_result["reduced_frames_per_cube"],
            name_suffix="_inspect",
        )
    timings_s["reduce"] = perf_counter() - t0

    t0 = perf_counter()
    segment_cubes = resolve_safe_segment_cubes(
        config_params=config_params,
        raw_cube_count=reduce_result["raw_cube_count"],
        frames_per_cube=frames_per_cube,
    )
    xcorr_result = run_xcorr_stage_in_memory(
        run_dir,
        reduce_result,
        config_params=config_params,
        overrides={"segment_cubes": segment_cubes},
    )
    if not all(success for _, success in xcorr_result["results"]):
        raise RuntimeError("Xcorr stage failed for one or more quadrants.")
    timings_s["xcorr"] = perf_counter() - t0

    t0 = perf_counter()
    distill_result = run_distill_stage_in_memory(
        run_dir,
        xcorr_result,
        config_params=config_params,
        save_pngs=save_distill_pngs,
    )
    if not distill_result["processed_groups"]:
        raise RuntimeError("Distill stage produced no processed groups.")
    timings_s["distill"] = perf_counter() - t0

    t0 = perf_counter()
    measure_result = run_measure_stage(
        basedir=run_dir,
        config_params=config_params,
        make_movie=(not no_movie) and config_params.get("MAKE_MOVIE", False),
        limit_cubes=None,
        parity_flip_needed=parity_flip_needed,
    )
    if not measure_result["json_paths"]:
        raise RuntimeError("Measure stage produced no wind-summary JSON outputs.")
    timings_s["measure"] = perf_counter() - t0

    if config_params.get("PUBLISH_WINDSOC_INDI", False):
        max_layers = int(config_params.get("WINDSOC_MAX_LAYERS", 10))

        wind_summaries = measure_result.get("wind_summaries", [])
        tracks = wind_summaries[0].get("tracks", []) if wind_summaries else []

        # Select top tracks by `flux` and compute relative normalized strength.
        def _flux_sort_key(t: dict) -> float:
            f = float(t.get("flux", float("-inf")))
            return f if np.isfinite(f) else float("-inf")

        tracks_sorted = sorted(
            tracks,
            key=_flux_sort_key,
            reverse=True,
        )
        selected = tracks_sorted[:max_layers]

        flux_vals: list[float] = []
        for t in selected:
            f = float(t.get("flux", 0.0))
            if np.isfinite(f):
                flux_vals.append(f)
        max_flux = max(flux_vals) if flux_vals else 0.0

        layer_values: list[dict[str, float]] = []
        for t in selected:
            flux = float(t.get("flux", 0.0))
            if (not np.isfinite(flux)) or (max_flux == 0.0) or (not np.isfinite(max_flux)):
                rel_strength = 0.0
            else:
                rel_strength = flux / max_flux

            speed = float(t.get("velocity_m_per_s", 0.0))
            dir_deg = float(t.get("direction", 0.0))
            if not np.isfinite(speed):
                speed = 0.0
            if not np.isfinite(dir_deg):
                dir_deg = 0.0
            layer_values.append(
                {
                    "speed": float(speed),
                    "dir": float(dir_deg),
                    "str": float(rel_strength),
                }
            )

        publisher = _maybe_get_windsoc_publisher(max_layers=max_layers)
        if publisher is not None:
            publisher.update_layers(layer_values)

    timings_s["total"] = sum(timings_s.values())

    if cleanup_intermediate:
        maybe_cleanup_intermediate(run_dir, config_params)

    summary = BatchRunSummary(
        run_dir=run_dir,
        raw_cube_paths=[],
        timings_s=timings_s,
        json_paths=measure_result["json_paths"],
        movie_paths=measure_result["movie_paths"],
        hour_angle_deg=hour_angle_deg,
        parity_flip_needed=parity_flip_needed,
    )
    write_batch_summary(summary)

    logging.info("Realtime batch complete: %s", run_dir)
    logging.info("Generated %d JSON output(s)", len(summary.json_paths))
    logging.info("Generated %d movie output(s)", len(summary.movie_paths))
    logging.info("Applied xcorr settings: %s", xcorr_result["settings"])
    return summary


def run_embedded_batch(
    batch_frames,
    first_timestamp,
    config_path: str,
    output_root: str,
    frames_per_cube: int = DEFAULT_FRAMES_PER_CUBE,
    no_movie: bool = False,
    save_distill_pngs: bool = False,
    cleanup_intermediate: bool = False,
) -> dict:
    """Embedded entrypoint for callers that already own a batch array.

    Stable call contract:
    - Callable name: `run_embedded_batch`
    - Positional args:
      `batch_frames`, `first_timestamp`, `config_path`, `output_root`
    - Keyword args:
      `frames_per_cube`, `no_movie`, `save_distill_pngs`,
      `cleanup_intermediate`
    - `batch_frames` is coerced to a float32 NumPy array with shape
      `(frame_count, frame_height, frame_width)`.
    - Returns a dict with stable keys:
      `run_dir`, `raw_cube_paths`, `timings_s`, `json_paths`, `movie_paths`
    """
    batch_array = np.asarray(batch_frames, dtype=np.float32)
    batch = FrameBatch(
        frames=batch_array,
        first_timestamp=_coerce_timestamp(first_timestamp),
    )
    summary = process_collected_batch(
        batch=batch,
        config_path=config_path,
        output_root=output_root,
        frames_per_cube=frames_per_cube,
        no_movie=no_movie,
        save_distill_pngs=save_distill_pngs,
        cleanup_intermediate=cleanup_intermediate,
    )
    return {
        "run_dir": summary.run_dir,
        "raw_cube_paths": summary.raw_cube_paths,
        "timings_s": summary.timings_s,
        "json_paths": summary.json_paths,
        "movie_paths": summary.movie_paths,
    }


def run_embedded_batch_buffer(
    batch_buffer,
    frame_count: int,
    first_timestamp,
    config_path: str,
    output_root: str,
    frame_height: int = DEFAULT_FRAME_SHAPE[0],
    frame_width: int = DEFAULT_FRAME_SHAPE[1],
    frames_per_cube: int = DEFAULT_FRAMES_PER_CUBE,
    no_movie: bool = False,
    save_distill_pngs: bool = False,
    cleanup_intermediate: bool = False,
) -> dict:
    """Embedded entrypoint that rebuilds a batch array from a raw buffer.

    Stable call contract used by `windsoccRT`:
    - Callable name: `run_embedded_batch_buffer`
    - Positional args:
      `batch_buffer`, `frame_count`, `first_timestamp`, `config_path`,
      `output_root`
    - Keyword args:
      `frame_height`, `frame_width`, `frames_per_cube`, `no_movie`,
      `save_distill_pngs`, `cleanup_intermediate`
    - `batch_buffer` must expose a contiguous buffer of `float32` values laid
      out as `frame_count * frame_height * frame_width`.
    - `first_timestamp` must be accepted by `_coerce_timestamp()`. The C++
      wrapper passes the compact UTC string from `formatTimestamp()`.
    """
    expected_values = int(frame_count) * int(frame_height) * int(frame_width)
    batch_array = np.frombuffer(batch_buffer, dtype=np.float32, count=expected_values)
    batch_array = batch_array.reshape((int(frame_count), int(frame_height), int(frame_width)))
    return run_embedded_batch(
        batch_frames=batch_array,
        first_timestamp=first_timestamp,
        config_path=config_path,
        output_root=output_root,
        frames_per_cube=frames_per_cube,
        no_movie=no_movie,
        save_distill_pngs=save_distill_pngs,
        cleanup_intermediate=cleanup_intermediate,
    )


def run_single_batch(args) -> BatchRunSummary:
    """Collect one batch, materialize it, and run the full pipeline."""
    target_frames = required_batch_frames(args.fps, args.integration_seconds)

    source = build_frame_source(args)

    t0 = perf_counter()
    batch = source.collect_frames(target_frames)
    validate_frame_batch(batch.frames)
    collection_elapsed = perf_counter() - t0

    return process_collected_batch(
        batch=batch,
        config_path=args.config,
        output_root=args.output_root,
        frames_per_cube=args.frames_per_cube,
        no_movie=args.no_movie,
        save_distill_pngs=args.save_distill_pngs,
        cleanup_intermediate=args.cleanup_intermediate,
        initial_timings_s={"frame_collection": collection_elapsed},
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    run_single_batch(args)


if __name__ == "__main__":
    main()