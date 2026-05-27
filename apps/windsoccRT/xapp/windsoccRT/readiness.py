"""Standby readiness checks for the windsoccRT INDI device."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from purepyindi2 import constants

# MagAO-X milk shmim path convention (see magaox.shmim.Image).
MILK_SHM_PREFIX = Path("/milk/shm")

BACKOFF_SECONDS: tuple[float, ...] = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


class ReadinessConfig(Protocol):
    """Minimal config surface used by :func:`evaluate_readiness`."""

    lab_mode_device: str
    lab_mode_property: str
    lab_mode_element: str
    stagepickoff_device: str
    stagepickoff_property: str
    stagepickoff_element: str
    camwfs_device: str
    shutter_property: str
    shutter_element: str
    shutter_closed_is_toggle_on: bool
    holoop_device: str
    loop_state_property: str
    loop_state_element: str


def shmim_path(stream_name: str) -> Path:
    """Return the expected ImageStreamIO shm path for ``stream_name``."""
    return MILK_SHM_PREFIX / f"{stream_name}.im.shm"


def shmim_exists(stream_name: str) -> bool:
    """Return whether the shmim file exists on disk."""
    return shmim_path(stream_name).is_file()


def backoff_seconds(failure_index: int) -> float:
    """Return sleep duration for the given consecutive readiness failure index."""
    if failure_index < 0:
        failure_index = 0
    if failure_index >= len(BACKOFF_SECONDS):
        return BACKOFF_SECONDS[-1]
    return BACKOFF_SECONDS[failure_index]


def _indi_key(device: str, prop: str, element: str) -> str:
    return f"{device}.{prop}.{element}"


def _read_switch(client: Any, key: str) -> bool | None:
    """Return switch ON/OFF, or ``None`` if the property cannot be read."""
    try:
        value = client[key]
    except (KeyError, TypeError, AttributeError):
        return None
    if value == constants.SwitchState.ON:
        return True
    if value == constants.SwitchState.OFF:
        return False
    return None


def evaluate_readiness(
    client: Any,
    cfg: ReadinessConfig,
) -> tuple[bool, list[str]]:
    """Evaluate whether the instrument state allows running the WindsoCC pipeline.

    Returns ``(ready, reasons)``. ``ready`` is False if any blocker is active or
    if a required INDI switch cannot be read (fail closed).
    """
    reasons: list[str] = []

    lab_key = _indi_key(cfg.lab_mode_device, cfg.lab_mode_property, cfg.lab_mode_element)
    lab_on = _read_switch(client, lab_key)
    if lab_on is None:
        reasons.append(f"{lab_key} unavailable")
    elif lab_on:
        reasons.append("lab mode on")

    # Pickoff mirror must be set to the telescope beam to be considered "on-sky".
    stagepickoff_key = _indi_key(
        cfg.stagepickoff_device,
        cfg.stagepickoff_property,
        cfg.stagepickoff_element,
    )
    stagepickoff_tel_on = _read_switch(client, stagepickoff_key)
    if stagepickoff_tel_on is None:
        reasons.append(f"{stagepickoff_key} unavailable")
    elif not stagepickoff_tel_on:
        reasons.append("stagepickoff mirror out (tel not in beam)")

    shutter_key = _indi_key(
        cfg.camwfs_device,
        cfg.shutter_property,
        cfg.shutter_element,
    )
    shutter_toggle = _read_switch(client, shutter_key)
    if shutter_toggle is None:
        reasons.append(f"{shutter_key} unavailable")
    else:
        shutter_closed = (
            shutter_toggle if cfg.shutter_closed_is_toggle_on else not shutter_toggle
        )
        if shutter_closed:
            reasons.append("camwfs shutter closed")

    loop_key = _indi_key(
        cfg.holoop_device,
        cfg.loop_state_property,
        cfg.loop_state_element,
    )
    loop_toggle = _read_switch(client, loop_key)
    if loop_toggle is None:
        reasons.append(f"{loop_key} unavailable")
    elif not loop_toggle:
        reasons.append("holoop loop open")

    return (len(reasons) == 0, reasons)


def readiness_gate_devices(cfg: ReadinessConfig) -> list[str]:
    """INDI device names to subscribe to before evaluating readiness."""
    return [
        cfg.lab_mode_device,
        cfg.stagepickoff_device,
        cfg.camwfs_device,
        cfg.holoop_device,
    ]
