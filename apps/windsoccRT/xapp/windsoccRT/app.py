"""MagAO-X Python INDI device that runs WindsoCC realtime batches in ``loop()``."""

from __future__ import annotations

import argparse
import logging
import time

import xconf
from magaox.indi.device import BaseConfig, XDevice
from purepyindi2 import constants, properties
from purepyindi2.messages import DefNumber, DefText

from windsocc.realtime import run_single_batch

from .readiness import (
    BACKOFF_SECONDS,
    backoff_seconds,
    evaluate_readiness,
    readiness_gate_devices,
    shmim_exists,
    shmim_path,
)

# Pipeline INDI state values exposed on ``pipeline.state``.
PIPELINE_STATE_STANDBY = "standby"
PIPELINE_STATE_ACTIVE = "active"
PIPELINE_STATE_MISSING_STREAM = "missing_stream"


@xconf.config
class WindsoccRTConfig(BaseConfig):
    """Configuration for RTC WindsoCC batches (loaded from ``<device>.conf``).

    When started as ``windsoccRT -n windsocc``, settings are read from
    ``/opt/MagAOX/config/windsocc.conf`` (TOML / ``xconf``).
    """

    ws_yaml_path: str = xconf.field(
        default="/opt/MagAOX/config/ws_config.yaml",
        help="YAML pipeline config passed to windsocc (typically ws_config.yaml).",
    )
    output_root: str = xconf.field(
        default="/tmp/windsocc-python-stream",
        help="Directory for camwfs_<timestamp> batch folders.",
    )
    stream_name: str = xconf.field(
        default="aol1_imWFS2",
        help="MagAO-X shmim stream basename (without ``_cbuff`` suffix).",
    )
    frame_count: int = xconf.field(
        default=40000,
        help="Frames per pipeline batch when source_type is shmim (must be >= 1).",
    )
    frame_height: int = xconf.field(default=120, help="Expected shmim frame height.")
    frame_width: int = xconf.field(default=120, help="Expected shmim frame width.")

    wait_new_frame: bool = xconf.field(
        default=True,
        help="If true, block on the shmim semaphore each sample.",
    )
    timeout_seconds: float = xconf.field(
        default=5.0,
        help="Per-frame timeout when wait_new_frame is true.",
    )
    check_before_wait: bool = xconf.field(
        default=False,
        help="Stat shmim inode before semaphore wait.",
    )
    cnt0_diagnostics: bool = xconf.field(
        default=True,
        help="Log cnt0 skip/duplicate diagnostics for shmim.",
    )

    frames_per_cube: int = xconf.field(default=512, help="Frames per FITS raw cube.")
    no_movie: bool = xconf.field(
        default=True,
        help="Force-disable movies in continuous RTC operation.",
    )
    save_distill_pngs: bool = xconf.field(default=False, help="Keep distill PNGs.")
    cleanup_intermediate: bool = xconf.field(
        default=True,
        help="Remove heavier intermediates after measure completes.",
    )

    publish_windsoc_indi: bool = xconf.field(
        default=True,
        help="Publish measured wind layers to the windsocc INDI device.",
    )
    windsoc_max_layers: int = xconf.field(
        default=10,
        help="Maximum number of wind layers exposed as INDI layer_XX properties.",
    )

    pipeline_log_level: str = xconf.field(
        default="INFO",
        help="Stdlib logging level for windsocc.realtime (DEBUG/INFO/WARNING/ERROR).",
    )

    enable_readiness_gating: bool = xconf.field(
        default=True,
        help="If true, run INDI readiness checks before each batch.",
    )
    shm_missing_log_interval_sec: float = xconf.field(
        default=60.0,
        help="Minimum seconds between ERROR logs when the shmim file is missing.",
    )

    lab_mode_device: str = xconf.field(default="tcsi", help="INDI device for lab mode.")
    lab_mode_property: str = xconf.field(default="labMode", help="Lab mode property name.")
    lab_mode_element: str = xconf.field(default="toggle", help="Lab mode switch element.")

    stagepickoff_device: str = xconf.field(
        default="stagepickoff",
        help="Stage pickoff INDI device (moves mirror in/out of telescope beam).",
    )
    stagepickoff_property: str = xconf.field(
        default="presetName",
        help="Pickoff preset selector property name.",
    )
    stagepickoff_element: str = xconf.field(
        default="tel",
        help="Pickoff element indicating telescope beam is selected (readiness passes only when ON).",
    )

    camwfs_device: str = xconf.field(default="camwfs", help="WFS camera INDI device.")
    shutter_property: str = xconf.field(default="shutter", help="Shutter property name.")
    shutter_element: str = xconf.field(default="toggle", help="Shutter switch element.")
    shutter_closed_is_toggle_on: bool = xconf.field(
        default=True,
        help="If true, shutter.toggle ON means closed (ocam2K convention).",
    )

    holoop_device: str = xconf.field(
        default="holoop",
        help="High-order loop INDI device.",
    )
    loop_state_property: str = xconf.field(
        default="loop_state",
        help="Loop state property on holoop.",
    )
    loop_state_element: str = xconf.field(
        default="toggle",
        help="Loop state switch; ON means closed loop.",
    )


class windsoccRT(XDevice):
    """FIFO-backed INDI device; each ``loop()`` iteration runs one full batch."""

    config: WindsoccRTConfig

    def __init__(self, name, config, *args, verbose=False, all_verbose=False, **kwargs):
        super().__init__(name, config, *args, verbose=verbose, all_verbose=all_verbose, **kwargs)
        level = getattr(logging, self.config.pipeline_log_level.upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
        self._pipeline_args = self._build_pipeline_namespace(self.config)
        self._backoff_failure_index = 0
        self._suppress_readiness_logs = False
        self._last_shm_missing_log_monotonic = 0.0
        self._indi_props_ready = False
        self._indi_props_ready_err: str | None = None

    def setup(self) -> None:
        """Define INDI properties owned by this windsocc device."""
        max_layers = int(self.config.windsoc_max_layers)

        pipeline_state_prop = properties.TextVector(
            name="pipeline",
            perm=constants.PropertyPerm.READ_ONLY,
        )
        pipeline_state_prop.add_element(
            DefText(
                name="state",
                label="Pipeline state",
                _value=PIPELINE_STATE_STANDBY,
            )
        )
        self.add_property(pipeline_state_prop)

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
                max=max_layers,
                step=1,
                _value=0,
            )
        )
        self.add_property(nlayers_prop)

        for i in range(max_layers):
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
            self.add_property(layer_prop)

        if self.config.enable_readiness_gating:
            self._ensure_readiness_props()

    def _set_pipeline_state(self, state: str) -> None:
        """Update the read-only ``pipeline.state`` INDI property."""
        if "pipeline" not in self.properties:
            return
        prop = self.properties["pipeline"]
        try:
            current = prop["state"]
        except (KeyError, TypeError):
            current = None
        if current != state:
            prop["state"] = state
            self.update_property(prop)

    def _ensure_readiness_props(self) -> bool:
        """Ensure readiness INDI device properties are available on the client."""
        devices = readiness_gate_devices(self.config)

        try:
            self.client.get_properties_and_wait(devices)
        except TimeoutError as exc:
            self._indi_props_ready = False
            self._indi_props_ready_err = (
                f"Timed out waiting for readiness INDI devices {devices}: {exc}"
            )
            return False
        except Exception as exc:  # pragma: no cover - defensive
            self._indi_props_ready = False
            self._indi_props_ready_err = (
                f"Failed to load readiness INDI properties {devices}: {exc}"
            )
            return False

        self._indi_props_ready = True
        self._indi_props_ready_err = None
        return True

    @staticmethod
    def _build_pipeline_namespace(cfg: WindsoccRTConfig) -> argparse.Namespace:
        """Build an ``argparse.Namespace`` compatible with ``run_single_batch``."""
        return argparse.Namespace(
            source_type="shmim",
            offline_source=None,
            frame_count=int(cfg.frame_count),
            fps=2000.0,
            integration_seconds=10.0,
            stream_name=cfg.stream_name,
            reader_callable=None,
            config=cfg.ws_yaml_path,
            output_root=cfg.output_root,
            frame_height=int(cfg.frame_height),
            frame_width=int(cfg.frame_width),
            wait_new_frame=bool(cfg.wait_new_frame),
            timeout_seconds=float(cfg.timeout_seconds),
            check_before_wait=bool(cfg.check_before_wait),
            cnt0_diagnostics=bool(cfg.cnt0_diagnostics),
            frames_per_cube=int(cfg.frames_per_cube),
            no_movie=bool(cfg.no_movie),
            save_distill_pngs=bool(cfg.save_distill_pngs),
            cleanup_intermediate=bool(cfg.cleanup_intermediate),
            config_overrides={
                "MAKE_MOVIE": False,
                "PUBLISH_WINDSOC_INDI": False,
                "WINDSOC_MAX_LAYERS": int(cfg.windsoc_max_layers),
            },
            profile=None,
            profile_scope="collect",
            iter_timing_topk=0,
        )

    def _update_wind_layer_properties(self, layers: list[dict[str, float]]) -> None:
        """Update the read-only wind-layer INDI properties on this device."""
        if not self.config.publish_windsoc_indi:
            return

        max_layers = int(self.config.windsoc_max_layers)
        nlayers_current = min(len(layers), max_layers)

        nlayers_prop = self.properties["nlayers"]
        nlayers_prop["current"] = float(nlayers_current)
        self.update_property(nlayers_prop)

        for i in range(max_layers):
            layer_name = f"layer_{i:02d}"
            layer_prop = self.properties[layer_name]

            if i < nlayers_current:
                layer = layers[i]
                layer_prop["speed"] = float(layer.get("speed", 0.0))
                layer_prop["dir"] = float(layer.get("dir", 0.0))
                layer_prop["str"] = float(layer.get("str", 0.0))
            else:
                layer_prop["speed"] = 0.0
                layer_prop["dir"] = 0.0
                layer_prop["str"] = 0.0

            self.update_property(layer_prop)

    def _check_shm_stream(self) -> bool:
        """Return True if the configured shmim file exists."""
        stream = self.config.stream_name
        if shmim_exists(stream):
            return True

        path = shmim_path(stream)
        now = time.monotonic()
        interval = float(self.config.shm_missing_log_interval_sec)
        if now - self._last_shm_missing_log_monotonic >= interval:
            self.log.error("Image stream missing: %s", path)
            self._last_shm_missing_log_monotonic = now

        self._set_pipeline_state(PIPELINE_STATE_MISSING_STREAM)
        return False

    def _check_readiness(self) -> bool:
        """Return True if readiness gating passes (or is disabled)."""
        if not self.config.enable_readiness_gating:
            return True

        if not self._indi_props_ready:
            self._ensure_readiness_props()

        if not self._indi_props_ready:
            ready = False
            reasons = [
                self._indi_props_ready_err
                or "Readiness INDI properties unavailable"
            ]
        else:
            ready, reasons = evaluate_readiness(self.client, self.config)
        if ready:
            self._suppress_readiness_logs = False
            self._backoff_failure_index = 0
            return True

        if not self._suppress_readiness_logs:
            self.log.warning("Readiness check failed: %s", "; ".join(reasons))
            self._suppress_readiness_logs = True

        sleep_s = backoff_seconds(self._backoff_failure_index)
        if self._backoff_failure_index < len(BACKOFF_SECONDS) - 1:
            self._backoff_failure_index += 1

        self._set_pipeline_state(PIPELINE_STATE_STANDBY)
        time.sleep(sleep_s)
        return False

    def loop(self) -> None:
        """Collect one shmim batch and run reduce / xcorr / distill / measure."""
        if not self._check_shm_stream():
            return

        if not self._check_readiness():
            return

        self._set_pipeline_state(PIPELINE_STATE_ACTIVE)

        try:
            summary = run_single_batch(self._pipeline_args)
            self._update_wind_layer_properties(summary.wind_layers or [])
            self.log.info(
                "Realtime batch complete run_dir=%s json_paths=%d",
                summary.run_dir,
                len(summary.json_paths),
            )
        except Exception:
            self.log.exception("Realtime batch failed")


main = windsoccRT.console_app
