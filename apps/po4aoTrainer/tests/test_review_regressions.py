## \file test_review_regressions.py
# \brief Cross-boundary regressions from the independent NN/PO4AO review.
# \ingroup po4aoTrainer_unit_test
"""Use the actual client cache, runtime adapter, supervisor and CPU policy-reference bookkeeping."""

from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from purepyindi2 import constants as C, messages as M, properties as P
from purepyindi2.client import IndiClient
from purepyindi2.transports import IndiConnection

from test_supervisor import config, finish
from xapp.po4aoTrainer.switching import Runtime
from xapp.po4aoTrainer.supervisor import Supervisor
from xapp.po4aoTrainer.configuration import load_training, revision
from xapp.po4aoTrainer.artifacts import engine_id, verify_candidate
from xapp.po4aoTrainer.process import atomic_json
from xapp.po4aoTrainer.training import TrainingSession
from xapp.po4aoTrainer.po4ao_runtime import ReplaySnapshot, save_training_snapshot


class Wire(IndiConnection):
    """Socket-free server boundary; the unmodified client still owns its actual retained cache."""

    def __init__(self, rig):
        """Record requests separately from simulated remote completion."""
        super().__init__()
        self.rig = rig
        self.status = C.ConnectionStatus.CONNECTED

    def send(self, message):
        """Respond to discovery or queue an explicit controller request."""
        if isinstance(message, M.GetProperties):
            if self.rig.discovery:
                for (device, name), prop in self.rig.props.items():
                    if (message.device is None or message.device == device) and (
                        message.name is None or message.name == name
                    ):
                        self.rig.emit(prop)
        else:
            self.rig.requests.append(
                (message.device, message.name, {key: message[key] for key in message})
            )
            self.rig.command(message)


class Rig:
    """Delayed frame/load model around real INDI transport classes; no instrument resources."""

    def __init__(self, timeout=0.15):
        """Define all remote capabilities before attaching the production observer."""
        self.discovery, self.frames, self.auto_load = True, True, True
        self.requests, self.props = [], {}
        self.writer_enabled, self.writer_stops = True, True
        self.writer_calls = []
        self.loading = None
        self.client = IndiClient()
        self.client.connection = Wire(self)
        self.text(
            "nn",
            "model_status",
            status="disabled",
            loaded="a.trt",
            requested="a.trt",
            active="",
            error="",
        )
        self.switch("nn", "enable", toggle=C.SwitchState.OFF)
        catalog = self.switch("nn", "model", a=C.SwitchState.ON, b=C.SwitchState.OFF)
        catalog._elements["a"].label, catalog._elements["b"].label = "a.trt", "b.trt"
        self.text(
            "po4ao",
            "model_status",
            instance="runtime-1",
            reset_request="1",
            reset_applied="1",
            control_request="1",
            control_applied="1",
            control_target="off",
            active_id="0" * 16,
            effective_id="0" * 16,
            effective_controller="integrator",
            reload="idle",
            reload_error="",
            sequence="1",
        )
        for name in (
            "control_enable",
            "reload_engine",
            "reset_model",
            "recollect_warmup",
        ):
            self.switch("po4ao", name, toggle=C.SwitchState.OFF)
        self.text(
            "po4ao",
            "collection_status",
            state="ready",
            requested_generation="1",
            generation="1",
            frames="96",
            target_frames="96",
        )
        for name, value in (
            ("modState", 4.0),
            ("modFrequency", 2000.0),
            ("modRadius", 3.0),
        ):
            prop = P.NumberVector(device="mod", name=name, state=C.PropertyState.OK)
            for key in ("current", "target"):
                prop.add_element(
                    M.DefNumber(
                        name=key, _value=value, min=0, max=10000, step=1, format="%g"
                    )
                )
            self.props["mod", name] = prop
        writer = SimpleNamespace(
            running=lambda: self.writer_enabled, command=self.writer_command
        )
        self.runtime = Runtime(self.client, writer, "nn", "po4ao", "mod", timeout)

    def text(self, device, name, **values):
        """Create real text definitions with string-valued runtime fields."""
        prop = P.TextVector(device=device, name=name, state=C.PropertyState.OK)
        for key, value in values.items():
            prop.add_element(M.DefText(name=key, _value=value))
        self.props[device, name] = prop
        return prop

    def switch(self, device, name, **values):
        """Create real switch definitions used by the production adapter."""
        prop = P.SwitchVector(
            device=device,
            name=name,
            state=C.PropertyState.OK,
            rule=C.SwitchRule.AT_MOST_ONE,
        )
        for key, value in values.items():
            prop.add_element(M.DefSwitch(name=key, _value=value))
        self.props[device, name] = prop
        return prop

    def emit(self, prop):
        """Deliver independent remote evidence through the actual client's parser/cache path."""
        self.client.handle_message(deepcopy(prop))

    def update(self, device, prop, **values):
        """Change simulated remote fields and publish their readback."""
        vector = self.props[device, prop]
        for key, value in values.items():
            vector[key] = value
        self.emit(vector)

    def writer_command(self, enabled):
        """Keep request delivery separate from verified FPS RUN completion."""
        self.writer_calls.append(enabled)
        if enabled or self.writer_stops:
            self.writer_enabled = enabled

    def command(self, message):
        """Simulate the donor's fault/quiescence and C++ request/after-frame semantics."""
        device, name = message.device, message.name
        if device == "nn":
            status = self.props["nn", "model_status"]
            if name == "enable":
                enabled = message["toggle"] == C.SwitchState.ON
                self.update("nn", "enable", toggle=message["toggle"])
                self.update(
                    "nn",
                    "model_status",
                    active=status["loaded"] if enabled else "",
                    status=(
                        "active"
                        if enabled
                        else "fault: reload required" if status["error"] else "disabled"
                    ),
                )
            elif name == "model":
                selected = next(
                    key for key in message if message[key] == C.SwitchState.ON
                )
                filename = self.props["nn", "model"]._elements[selected].label
                for prop in ("model", "enable"):
                    self.props["nn", prop].state = C.PropertyState.OK
                    self.emit(self.props["nn", prop])
                self.update(
                    "nn",
                    "model_status",
                    requested=filename,
                    loaded=filename,
                    active="",
                    status="disabled",
                    error="",
                )
        elif device == "mod":
            self.update(
                device, name, current=message["target"], target=message["target"]
            )
        elif device == "po4ao":
            status = self.props["po4ao", "model_status"]
            if name == "control_enable":
                target = "on" if message["toggle"] == C.SwitchState.ON else "off"
                if target != status["control_target"]:
                    self.update(
                        "po4ao",
                        "model_status",
                        control_request=str(int(status["control_request"]) + 1),
                        control_target=target,
                    )
                    self.props["po4ao", "control_enable"].state = C.PropertyState.BUSY
                    self.emit(self.props["po4ao", "control_enable"])
            elif name == "reset_model":
                self.update(
                    "po4ao",
                    "model_status",
                    reset_request=str(int(status["reset_request"]) + 1),
                )
            elif name == "reload_engine":
                self.loading = (
                    self.runtime.reload_ticket["model_id"],
                    status["reset_request"],
                )
                self.update("po4ao", "model_status", reload="loading")
                self.props["po4ao", "reload_engine"].state = C.PropertyState.BUSY
                self.emit(self.props["po4ao", "reload_engine"])
                if self.auto_load:
                    self.complete_load()
            if self.frames:
                self.apply_frames()

    def apply_frames(self):
        """Apply already admitted reset/selection requests at a simulated completed frame."""
        status = self.props["po4ao", "model_status"]
        if status["reset_request"] != status["reset_applied"]:
            self.update(
                "po4ao",
                "model_status",
                reset_applied=status["reset_request"],
                active_id="0" * 16,
            )
        enabled = status["control_target"] == "on"
        self.update(
            "po4ao",
            "model_status",
            control_applied=status["control_request"],
            effective_controller=(
                "po4ao" if enabled and status["active_id"] != "0" * 16 else "integrator"
            ),
            effective_id=status["active_id"] if enabled else "0" * 16,
        )
        self.props["po4ao", "control_enable"].state = C.PropertyState.OK
        self.update(
            "po4ao",
            "control_enable",
            toggle=C.SwitchState.ON if enabled else C.SwitchState.OFF,
        )

    def complete_load(self):
        """A reset generation rejects an old delayed candidate exactly as the runtime does."""
        model, generation = self.loading
        if generation == self.props["po4ao", "model_status"]["reset_request"]:
            self.update("po4ao", "model_status", active_id=model)
        self.update("po4ao", "model_status", reload="idle")
        self.props["po4ao", "reload_engine"].state = C.PropertyState.OK
        self.emit(self.props["po4ao", "reload_engine"])
        self.loading = None

    def close(self):
        """Detach local callbacks without constructing or closing any socket."""
        self.runtime.close()


class NumericalWorker:
    """Use real session/reference bookkeeping behind an immediate control-pipe boundary."""

    def __init__(self, config):
        """Initialize once on CPU with a genuinely completed synthetic numerical iteration."""
        self.session = TrainingSession(config)
        rng = np.random.default_rng(42)
        self.data = rng.normal(size=(6, 96, 2)).astype(np.float32)
        self.warmup = ReplaySnapshot(self.data, 1, 96, 3, "0" * 16)
        self.session.step(self.warmup, self.warmup, active_model_id="0" * 16)
        # Publication-gate parity is tested separately; this test isolates delayed activation bookkeeping.
        self.session.last_result["record"]["publication_status"] = "ready"
        self.state, self.closed, self.error = "running", False, ""
        self.revision, self.iteration = revision(config), self.session.iteration
        self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)
        self.messages = []

    def send(self, message):
        """Exercise the same staging boundary as the real worker, using actual numerical objects."""
        self.messages.append(message)
        if "run" in message:
            self.state = "running" if message["run"] else "paused"
        if message.get("candidate_id"):
            self.session.stage_publication(message["candidate_id"])
        if message.get("candidate_outcome") in ("failed", "invalidated"):
            self.session.policy_update.pending = None

    def events(self):
        """No spontaneous exports; the test explicitly supplies its verified candidate."""
        return []

    def stop(self):
        """Record owned-session termination without touching other processes."""
        self.closed = True


def bundle(supervisor, name="candidate", session=None):
    """Create a self-consistent analysis snapshot and a fictional engine, never submitted to a GPU."""
    path = supervisor.root / name
    path.mkdir()
    (path / "policy.trt").write_bytes(b"CPU-only fictional engine " + name.encode())
    model = engine_id(path / "policy.trt")
    config = supervisor.desired
    session = session or TrainingSession(config)
    metadata = dict(
        generation=supervisor.record["generation"],
        active_model_id=model,
        artifact_type="tensorrt",
        config=config,
        config_revision=revision(config),
        session_id=session.session_id,
    )
    (path / "policy.trt.json").write_text(json.dumps(metadata))
    save_training_snapshot(
        path / "policy.trt.training.npz",
        session.policy_copy,
        session.dynamics,
        model,
        metadata,
    )
    return path, model


def wait_loading(supervisor, rig):
    """Pump actual supervisor ticks until a load is issued, without completing that load."""
    deadline = time.monotonic() + 2
    while rig.loading is None:
        supervisor.tick()
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_reset_without_frames_survives_restart_and_cannot_resurrect_policy(config):
    """Requests precede waits; an incomplete reset remains a persistent eligibility barrier."""
    rig = Rig(0.04)
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    path, model = bundle(supervisor, session=worker.session)
    supervisor.publish(path, supervisor.record["generation"], threading.Event())
    rig.frames = False
    supervisor.submit("reset", {"request": True})
    assert finish(supervisor)[-1][1] == "Alert"
    assert any(name == "reset_model" for _, name, _ in rig.requests)
    assert worker.closed and supervisor.record["eligible"] is None
    assert supervisor.record["reset_pending"]
    supervisor.close()
    resumed = Supervisor(rig.runtime, config)
    try:
        with pytest.raises(RuntimeError, match="Reset is still pending"):
            resumed.submit("po4ao_enable", {"toggle": True})
        rig.frames = True
        rig.apply_frames()
        resumed.tick()
        assert resumed.record["reset_pending"] is None
        assert rig.props["po4ao", "model_status"]["active_id"] == "0" * 16
        before = len([r for r in rig.requests if r[1] == "reload_engine"])
        resumed.submit("po4ao_enable", {"toggle": True})
        assert finish(resumed)[-1][1] == "Ok"
        assert len([r for r in rig.requests if r[1] == "reload_engine"]) == before
    finally:
        resumed.close()
        rig.close()


def test_external_disable_pauses_and_external_reset_invalidates_session(config):
    """Actual external INDI requests affect owned state without notebook process discovery."""
    rig = Rig()
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    path, model = bundle(supervisor, session=worker.session)
    supervisor.publish(path, supervisor.record["generation"], threading.Event())
    rig.runtime.control(True, threading.Event())
    supervisor.tick()
    rig.command(
        rig.props["po4ao", "control_enable"].make_new_property(toggle=C.SwitchState.OFF)
    )
    events = supervisor.tick() + finish(supervisor)
    assert events[-1][1] == "Ok"
    assert (
        worker.state == "paused" and supervisor.record["eligible"]["model_id"] == model
    )
    rig.command(
        rig.props["po4ao", "reset_model"].make_new_property(toggle=C.SwitchState.ON)
    )
    supervisor.tick()
    finish(supervisor)
    assert worker.closed and supervisor.record["eligible"] is None
    assert supervisor.worker is None
    supervisor.close()
    rig.close()


def test_external_disable_supersedes_an_enable_wait(config):
    """An external Off request cancels an in-flight Enable, not just a publication/recollection."""
    rig = Rig(0.5)
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    path, model = bundle(supervisor, session=worker.session)
    supervisor.publish(path, supervisor.record["generation"], threading.Event())
    rig.runtime.control(True, threading.Event())
    supervisor.tick()
    rig.auto_load = False
    supervisor.submit("po4ao_enable", {"toggle": True})
    wait_loading(supervisor, rig)
    rig.command(
        rig.props["po4ao", "control_enable"].make_new_property(toggle=C.SwitchState.OFF)
    )
    events = supervisor.tick()
    rig.complete_load()
    events += finish(supervisor)
    assert events[-1][1] == "Ok"
    assert worker.state == "paused"
    assert rig.runtime.runtime_state()["control_target"] == "off"
    supervisor.close()
    rig.close()


def test_legacy_eligibility_without_runtime_identity_requires_reset(config):
    """An old on-disk record cannot establish whether the runtime was reset while the app was absent."""
    rig = Rig()
    (Path(config.artifact_root) / "session.json").write_text(
        json.dumps(
            {
                "generation": "old",
                "eligible": {"path": "old", "model_id": "a" * 16},
                "worker_started": False,
            }
        )
    )
    supervisor = Supervisor(rig.runtime, config)
    assert supervisor.record["eligible"] is None
    with pytest.raises(RuntimeError, match="Reset is still pending"):
        supervisor.submit("po4ao_enable", {"toggle": True})
    supervisor.submit("reset", {"request": True})
    assert finish(supervisor)[-1][1] == "Ok"
    assert supervisor.record["reset_pending"] is None
    supervisor.close()
    rig.close()


def test_cancelled_reload_settles_before_disable_and_resume_keeps_known_reference(
    config,
):
    """Cross Runtime -> Supervisor -> numerical bookkeeping with delayed C++ activation."""
    rig = Rig(0.5)
    rig.auto_load = False
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    supervisor.applied = deepcopy(supervisor.desired)
    path, model = bundle(supervisor, session=worker.session)
    supervisor.submit("publish", {"path": str(path)})
    wait_loading(supervisor, rig)
    supervisor.submit("po4ao_enable", {"toggle": False})
    for _ in range(5):
        assert not any(
            command == "po4ao_enable" and state == "Ok"
            for command, state, _ in supervisor.tick()
        )
        time.sleep(0.005)
    assert worker.session.policy_update.pending[0] == model
    assert supervisor.record["pending"] is not None
    rig.complete_load()
    assert finish(supervisor)[-1][1] == "Ok"
    assert (
        worker.state == "paused" and supervisor.record["eligible"]["model_id"] == model
    )
    supervisor.submit("learning", {"run": True})
    assert finish(supervisor)[-1][1] == "Ok"
    recent = ReplaySnapshot(worker.data * 1.01, 2, 96, 4, model)
    assert worker.session.step(recent, worker.warmup, active_model_id=model) is not None
    assert worker.session.policy_update.reference_id == model
    supervisor.close()
    rig.close()


def test_timed_out_load_cannot_acknowledge_pause_until_eventual_completion(config):
    """A bounded failure is not a successful Pause while a late activation remains possible."""
    rig = Rig(0.03)
    rig.auto_load = False
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    path, model = bundle(supervisor, session=worker.session)
    supervisor.submit("publish", {"path": str(path)})
    wait_loading(supervisor, rig)
    finish(supervisor)
    supervisor.submit("learning", {"run": False})
    assert finish(supervisor)[-1][1] == "Alert"
    assert supervisor.record["pending"]
    rig.complete_load()
    supervisor.tick()
    supervisor.submit("learning", {"run": False})
    assert finish(supervisor)[-1][1] == "Ok"
    assert supervisor.record["eligible"]["model_id"] == model
    assert worker.session.policy_update.pending[0] == model
    supervisor.close()
    rig.close()


def test_real_client_cache_is_not_current_after_reconnect_or_device_deletion():
    """Use the actual retained cache rather than a fixture that clears it on reconnect."""
    rig = Rig()
    assert rig.runtime.source() == "cacao"
    wire = rig.client.connection
    wire.status = C.ConnectionStatus.STOPPED
    wire.dispatch_callbacks(C.TransportEvent.disconnection, wire.status)
    wire.status = C.ConnectionStatus.CONNECTED
    wire.dispatch_callbacks(C.TransportEvent.connection, wire.status)
    assert (
        rig.client["nn.model_status.loaded"] == "a.trt"
    )  # The actual client retained its cache.
    with pytest.raises(ConnectionError):
        rig.runtime.source()
    rig.runtime.observations.refresh()
    assert rig.runtime.source() == "cacao"
    rig.client.handle_message(M.DelProperty(device="nn"))
    with pytest.raises(ConnectionError):
        rig.runtime.source()
    rig.close()


def test_silent_device_expires_but_unchanged_fresh_definitions_renew_it(monkeypatch):
    """Unchanged values are allowed; only received current-lifetime evidence renews freshness."""
    rig = Rig()
    clock = [time.monotonic()]
    monkeypatch.setattr(
        "xapp.po4aoTrainer.observations.time.monotonic", lambda: clock[0]
    )
    clock[0] += 6
    with pytest.raises(ConnectionError):
        rig.runtime.source()
    rig.runtime.observations.refresh()
    assert rig.runtime.source() == "cacao"
    rig.close()


def test_runtime_heartbeat_detects_stalled_applogic_despite_fresh_definitions(
    monkeypatch,
):
    """Re-serving a frozen C++ status does not prove the control loop is being observed."""
    rig = Rig()
    clock = [time.monotonic()]
    monkeypatch.setattr(
        "xapp.po4aoTrainer.observations.time.monotonic", lambda: clock[0]
    )
    clock[0] += 6
    rig.runtime.observations.refresh()
    with pytest.raises(ConnectionError):
        rig.runtime.runtime_state()
    rig.update("po4ao", "model_status", sequence="2")
    assert rig.runtime.runtime_state()["instance"] == "runtime-1"
    rig.close()


def test_runtime_restart_invalidates_eligible_engine_before_any_enable(config):
    """Process identity prevents reset-counter reuse from resurrecting a previous runtime's policy."""
    rig = Rig()
    supervisor = Supervisor(rig.runtime, config)
    path, model = bundle(supervisor)
    supervisor.publish(path, supervisor.record["generation"], threading.Event())
    rig.update("po4ao", "model_status", instance="runtime-2", active_id="0" * 16)
    with pytest.raises(RuntimeError, match="identity/reset changed"):
        supervisor.submit("po4ao_enable", {"toggle": True})
    supervisor.tick()
    finish(supervisor)
    assert supervisor.record["eligible"] is None
    assert not supervisor.record["reset_pending"]
    supervisor.close()
    rig.close()


def test_reset_invalidates_already_issued_load_and_exact_reference(config):
    """A delayed candidate completing after Reset can never re-enter current eligibility."""
    rig = Rig(0.5)
    rig.auto_load = False
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    path, model = bundle(supervisor, session=worker.session)
    supervisor.submit("publish", {"path": str(path)})
    wait_loading(supervisor, rig)
    supervisor.submit("reset", {"request": True})
    assert finish(supervisor)[-1][1] == "Ok"
    rig.complete_load()
    supervisor.tick()
    assert worker.closed and supervisor.worker is None
    assert (
        supervisor.record["eligible"] is None and supervisor.record["pending"] is None
    )
    assert rig.runtime.get("po4ao", "model_status", "active_id") == "0" * 16
    supervisor.close()
    rig.close()


@pytest.mark.parametrize("linked", [False, True])
def test_active_nn_selection_is_noop_and_replacement_keeps_modulation(linked):
    """R1 same-model no-op and R2 NN-to-NN replacement never touch CACAO/modulation."""
    rig = Rig()
    rig.writer_enabled = False
    rig.update("nn", "enable", toggle=C.SwitchState.ON)
    rig.update("nn", "model_status", status="active", active="a.trt")
    rig.requests.clear()
    rig.runtime.switch("nn", threading.Event(), model="a", linked=linked)
    assert not rig.requests and not rig.writer_calls
    rig.runtime.switch("nn", threading.Event(), model="b", linked=linked)
    assert all(device == "nn" for device, _, _ in rig.requests)
    assert not rig.writer_calls
    assert rig.runtime.get("nn", "model_status", "active") == "b.trt"
    rig.close()


@pytest.mark.parametrize("target", ["nn", "cacao"])
def test_confirmed_nn_fault_allows_recovery_without_claiming_activation(target):
    """Fault plus confirmed Off/quiescence permits reload or CACAO recovery."""
    rig = Rig()
    rig.writer_enabled = False
    rig.props["nn", "enable"].state = C.PropertyState.ALERT
    rig.emit(rig.props["nn", "enable"])
    rig.update(
        "nn",
        "model_status",
        status="fault: reload required",
        error="bad engine",
        active="",
    )
    rig.props["nn", "model"].state = C.PropertyState.ALERT
    rig.emit(rig.props["nn", "model"])
    assert rig.runtime.source() == "stopped"
    rig.runtime.switch(
        target,
        threading.Event(),
        model="b" if target == "nn" else None,
        expected="stopped",
    )
    assert rig.runtime.source() == target
    rig.close()


def test_unverified_writer_stop_never_enables_nn():
    """The complete NN preparation path still requires actual CACAO RUN exit."""
    rig = Rig(0.02)
    rig.writer_stops = False
    with pytest.raises(TimeoutError):
        rig.runtime.switch("nn", threading.Event(), model="b")
    assert not any(
        name == "enable" and values["toggle"] == C.SwitchState.ON
        for _, name, values in rig.requests
    )
    rig.close()


def test_linked_handover_uses_configured_modulation_states():
    """Hardware state numbers are mappings; the agreed CACAO frequency/radius remain fixed."""
    rig = Rig()
    rig.runtime.unmodulated_state, rig.runtime.modulated_state = 7.0, 8.0
    rig.runtime.switch("nn", threading.Event(), linked=True, model="b")
    rig.runtime.switch("cacao", threading.Event(), linked=True)
    assert [
        values["target"]
        for device, name, values in rig.requests
        if device == "mod" and name == "modState"
    ] == [7.0, 8.0]
    assert rig.runtime.get("mod", "modFrequency", "current") == 2000.0
    assert rig.runtime.get("mod", "modRadius", "current") == 3.0
    rig.close()


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("nonlinear", "channels", 1),
        ("nonlinear", "shared_layers", 1),
        ("nonlinear", "negative_slope", 0.0),
        ("nonlinear", "negative_slope", 1.0),
        ("normalization", "rms_floor_fraction", 1.1),
    ],
)
def test_whole_validation_rejects_model_initialization_failures(
    config, group, key, value
):
    """Do not accept a desired configuration that the model/normalizer necessarily rejects."""
    import tomllib
    import tomli_w

    rig = Rig()
    supervisor = Supervisor(rig.runtime, config)
    valid = deepcopy(supervisor.desired)
    candidate = tomllib.loads(Path(config.training_config).read_text())
    candidate[group][key] = value
    Path(config.training_config).write_text(tomli_w.dumps(candidate))
    supervisor.poll_config()
    assert supervisor.desired == valid and supervisor.config_error
    supervisor.close()
    rig.close()


def test_snapshot_contents_and_identity_are_verified(config):
    """Existing empty or mismatched NPZ files never establish a complete candidate."""
    rig = Rig()
    supervisor = Supervisor(rig.runtime, config)
    path, model = bundle(supervisor)
    verify_candidate(path, supervisor.root, supervisor.record["generation"])
    snapshot = path / "policy.trt.training.npz"
    snapshot.write_bytes(b"")
    with pytest.raises(ValueError):
        verify_candidate(path, supervisor.root, supervisor.record["generation"])
    np.savez(
        snapshot,
        metadata_json=json.dumps({"model_id": model, "session_id": "wrong"}),
        policy_state__x=np.zeros(1),
        dynamics_0_state__x=np.zeros(1),
    )
    with pytest.raises(ValueError, match="identity"):
        verify_candidate(path, supervisor.root, supervisor.record["generation"])
    metadata = json.loads((path / "policy.trt.json").read_text())
    metadata["model_id"] = model
    np.savez(
        snapshot,
        metadata_json=json.dumps(metadata),
        policy_state__x=np.array([np.nan]),
        dynamics_0_state__x=np.zeros(1),
    )
    with pytest.raises(ValueError, match="finite"):
        verify_candidate(path, supervisor.root, supervisor.record["generation"])
    supervisor.close()
    rig.close()
