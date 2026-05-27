"""Unit tests for windsoccRT readiness gating."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from purepyindi2 import constants

from .readiness import (
    BACKOFF_SECONDS,
    backoff_seconds,
    evaluate_readiness,
    shmim_path,
)


@dataclass
class _Cfg:
    lab_mode_device: str = "tcsi"
    lab_mode_property: str = "labMode"
    lab_mode_element: str = "toggle"
    stagepickoff_device: str = "stagepickoff"
    stagepickoff_property: str = "presetName"
    stagepickoff_element: str = "tel"
    camwfs_device: str = "camwfs"
    shutter_property: str = "shutter"
    shutter_element: str = "toggle"
    shutter_closed_is_toggle_on: bool = True
    holoop_device: str = "holoop"
    loop_state_property: str = "loop_state"
    loop_state_element: str = "toggle"


class _MockClient(dict):
    """Dict keyed by full INDI switch paths."""

    def __getitem__(self, key: str):
        if key not in self:
            raise KeyError(key)
        return super().__getitem__(key)


def _ready_client() -> _MockClient:
    return _MockClient(
        {
            "tcsi.labMode.toggle": constants.SwitchState.OFF,
            "stagepickoff.presetName.tel": constants.SwitchState.ON,
            "camwfs.shutter.toggle": constants.SwitchState.OFF,
            "holoop.loop_state.toggle": constants.SwitchState.ON,
        }
    )


def test_shmim_path():
    assert shmim_path("aol1_imWFS2") == "/milk/shm/aol1_imWFS2.im.shm"


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, 1.0),
        (1, 5.0),
        (2, 10.0),
        (3, 30.0),
        (4, 60.0),
        (5, 120.0),
        (6, 300.0),
        (7, 300.0),
        (100, 300.0),
    ],
)
def test_backoff_seconds(index: int, expected: float) -> None:
    assert backoff_seconds(index) == expected
    assert BACKOFF_SECONDS[-1] == 300.0


def test_evaluate_readiness_all_clear() -> None:
    ready, reasons = evaluate_readiness(_ready_client(), _Cfg())
    assert ready is True
    assert reasons == []


def test_evaluate_readiness_lab_mode() -> None:
    client = _ready_client()
    client["tcsi.labMode.toggle"] = constants.SwitchState.ON
    ready, reasons = evaluate_readiness(client, _Cfg())
    assert ready is False
    assert "lab mode on" in reasons


def test_evaluate_readiness_stagepickoff_out() -> None:
    client = _ready_client()
    client["stagepickoff.presetName.tel"] = constants.SwitchState.OFF
    ready, reasons = evaluate_readiness(client, _Cfg())
    assert ready is False
    assert "stagepickoff mirror out" in reasons


def test_evaluate_readiness_shutter_closed() -> None:
    client = _ready_client()
    client["camwfs.shutter.toggle"] = constants.SwitchState.ON
    ready, reasons = evaluate_readiness(client, _Cfg())
    assert ready is False
    assert "camwfs shutter closed" in reasons


def test_evaluate_readiness_shutter_open_when_toggle_on_means_open() -> None:
    client = _ready_client()
    client["camwfs.shutter.toggle"] = constants.SwitchState.ON
    cfg = _Cfg(shutter_closed_is_toggle_on=False)
    ready, reasons = evaluate_readiness(client, cfg)
    assert ready is True
    assert reasons == []


def test_evaluate_readiness_loop_open() -> None:
    client = _ready_client()
    client["holoop.loop_state.toggle"] = constants.SwitchState.OFF
    ready, reasons = evaluate_readiness(client, _Cfg())
    assert ready is False
    assert "holoop loop open" in reasons


def test_evaluate_readiness_missing_property() -> None:
    client = _MockClient({"tcsi.labMode.toggle": constants.SwitchState.OFF})
    ready, reasons = evaluate_readiness(client, _Cfg())
    assert ready is False
    assert any("unavailable" in r for r in reasons)


def test_evaluate_readiness_multiple_blockers() -> None:
    client = _ready_client()
    client["tcsi.labMode.toggle"] = constants.SwitchState.ON
    client["holoop.loop_state.toggle"] = constants.SwitchState.OFF
    ready, reasons = evaluate_readiness(client, _Cfg())
    assert ready is False
    assert len(reasons) >= 2
