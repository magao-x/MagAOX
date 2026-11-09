## \file test_supervisor.py
# \brief Configuration, startup, INDI freshness and acknowledged supervisor recovery.
# \ingroup po4aoTrainer_unit_test
"""Offline regressions using synthetic data and explicit instrument boundaries."""

import json
import logging
import subprocess
import sys
import threading
import time
import pytest
import tomli_w
import tomllib
import numpy as np
import torch
import os
import pwd
import stat
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from purepyindi2 import constants as C, properties as P, transports, messages as M
from purepyindi2.messages import DefText, DefSwitch, DefNumber
from xapp.po4aoTrainer.configuration import (
    load_training,
    revision,
    needs_reset,
    apply_hot,
)
from xapp.po4aoTrainer.supervisor import Supervisor
from xapp.po4aoTrainer.switching import Runtime
from xapp.po4aoTrainer.library import engine_id, verify_candidate
from xapp.po4aoTrainer.app import po4aoTrainer, TrainerConfig
from xapp.po4aoTrainer.training import TrainingSession
from contextlib import ExitStack
from purepyindi2.client import IndiClient
from purepyindi2.transports import IndiConnection
from xapp.po4aoTrainer.process import atomic_json
from xapp.po4aoTrainer.po4ao_runtime import ReplaySnapshot, save_training_snapshot


@pytest.fixture
def config(tmp_path):
    """Resolve the existing synthetic numerical fixture through the deployment parser."""
    original = json.loads(
        (Path(__file__).parent / "fixtures/session_config.json").read_text()
    )
    training = deepcopy(original)
    for group in ("runtime", "integrator", "safety"):
        training.pop(group)
    training["MDP"].pop("history_capacity")
    training["RL"].pop("episode_length")
    training["replay_buffers"].pop("capacity")
    training["compute"] = dict(training_device=1, build_device=1)
    (tmp_path / "training.conf").write_text(tomli_w.dumps(training))
    (tmp_path / "runtime.conf").write_text(f"""[parameters]
config_version=2
Nmodes=6
Nmodes_controlled=4
max_Nhist=8
iterations_per_ep=32
warmup_episodes=3
replay_buffer_size=96
integrator_gain=0.3
max_command_abs=100
cuda_device=0
buffer_channel=replay
buffer_metadata_channel=replaymeta
warmup_buffer_channel=warmup
warmup_metadata_channel=warmupmeta
engineDirs={tmp_path}/current
engineName=policy.trt
""")
    return TrainerConfig(
        artifact_root=str(tmp_path),
        training_config=str(tmp_path / "training.conf"),
        runtime_config=str(tmp_path / "runtime.conf"),
        worker_timeout=0.5,
        operation_timeout=0.5,
    )


class FakeRuntime:
    """Observe/wait boundary double; operation ordering stays in Supervisor."""

    po4ao = "po4ao"

    def __init__(self):
        """Record operations and allow the reset test to hold a recollection."""
        self.calls = []
        self.hold = False
        self.reload_ticket = None
        self.observations = SimpleNamespace(refresh=lambda: None)
        self.observed = dict(
            instance="test-runtime",
            reset_request="1",
            reset_applied="1",
            control_request="1",
            control_target="off",
            active_id="0" * 16,
            reload="idle",
        )

    def get(self, device, prop, element):
        """Supply observed device state without inferring it from pending requests."""
        return {
            "effective_controller": "integrator",
            "toggle": C.SwitchState.OFF,
            "effective_id": "0" * 16,
            "active_id": "0" * 16,
            "state": "ready",
            "requested_generation": "1",
            "current": "3",
            "frames": "96",
            "target_frames": "96",
        }[element]

    def wait(self, predicate, cancel, label, timeout=None):
        """Use the production bounded/cancellable wait."""
        return Runtime.wait(self, predicate, cancel, label, timeout or 0.5)

    def runtime_state(self):
        """Return the fixture's independent observed generations."""
        return dict(self.observed)

    def request_control(self, enabled):
        """Admit an immediate controller request at this test boundary."""
        self.calls.append(("control", enabled))
        self.observed["control_request"] = str(
            int(self.observed["control_request"]) + 1
        )
        self.observed["control_target"] = "on" if enabled else "off"

    def control_complete(self, enabled):
        """The simple fixture acknowledges immediate control completion."""
        return True

    def control(self, enabled, cancel):
        """Use the same request/readback entry point as the runtime adapter."""
        self.request_control(enabled)

    def request_reset(self):
        """Record both requests before any frame/worker completion wait."""
        ticket = dict(
            instance=self.observed["instance"],
            after=int(self.observed["reset_request"]),
        )
        self.request_control(False)
        self.calls.append(("reset",))
        self.observed["reset_request"] = self.observed["reset_applied"] = str(
            ticket["after"] + 1
        )
        self.observed["active_id"] = "0" * 16
        return ticket

    def reset_complete(self, ticket, external=False):
        """Compare independently mutable applied reset identity."""
        return int(self.observed["reset_applied"]) > ticket["after"]

    def begin_reload(self, model, issued=None):
        """Record the model while allowing delayed-outcome subclasses to override acknowledgement."""
        self.calls.append(("reload", model))
        self.reload_ticket = {"model_id": model, "outcome": "loaded"}
        if issued:
            issued(self.reload_ticket)
        self.observed["active_id"] = model

    def finish_reload(self, cancel, ticket=None):
        """Use the production wait with the captured ticket identity."""
        return Runtime.finish_reload(self, cancel, ticket)

    def reload_outcome(self, ticket=None):
        """Return a load outcome independent of delivery and later tickets."""
        ticket = self.reload_ticket if ticket is None else ticket
        return ticket["outcome"] if ticket else None

    def recollect(self, cancel, timeout):
        """Hold collection until Reset cancellation when requested by a test."""
        self.calls.append(("recollect",))
        self.wait(lambda: not self.hold, cancel, "collection", timeout)


def finish(supervisor, domain="training"):
    """Pump the real supervisor while its bounded operation executes."""
    deadline = time.monotonic() + 2
    events = []
    while supervisor.domains[domain]["future"] is not None:
        events.extend(supervisor.tick())
        assert time.monotonic() < deadline
        time.sleep(0.005)
    return events


def test_config_types_unknowns_and_reset_boundary(config):
    """Reject partial/unknown edits and distinguish hot fields from architecture changes."""
    initial = load_training(config.training_config, config.runtime_config)
    hot = deepcopy(initial)
    hot["training"]["policy_learning_rate"] *= 2
    assert not needs_reset(initial, hot)
    hot["nonlinear"]["channels"] *= 2
    assert needs_reset(initial, hot)
    path = Path(config.training_config)
    path.write_text(path.read_text() + "\nunknown_key=2\n")
    with pytest.raises(Exception):
        load_training(config.training_config, config.runtime_config)


def test_hot_learning_rate_preserves_adam(config):
    """Change both optimizer rates without replacing model or optimizer objects."""
    from xapp.po4aoTrainer.training import TrainingSession

    initial = load_training(config.training_config, config.runtime_config)
    session = TrainingSession(initial)
    optimizer = session.policy_optimizer
    optimizer.state["sentinel"] = {"step": 7}
    candidate = deepcopy(initial)
    candidate["training"]["policy_learning_rate"] = 0.0002
    apply_hot(session, candidate)
    assert session.policy_optimizer is optimizer
    assert optimizer.state["sentinel"]["step"] == 7
    assert optimizer.param_groups[0]["lr"] == 0.0002
    assert session.config_revision == revision(candidate)


def test_invalid_edit_retains_last_good_configuration(config):
    """An invalid save does not silently replace an active resolved snapshot."""
    supervisor = Supervisor(FakeRuntime(), config)
    previous = deepcopy(supervisor.desired)
    Path(config.training_config).write_text("incomplete = [")
    supervisor.poll_config()
    assert supervisor.desired == previous
    assert supervisor.config_error
    supervisor.close()


def test_reset_cancels_collection(config):
    """Reset supersedes a held warmup."""
    runtime = FakeRuntime()
    runtime.hold = True
    supervisor = Supervisor(runtime, config)
    supervisor.submit("recollect", {"request": True})
    supervisor.submit("reset", {"request": True})
    events = finish(supervisor)
    assert any(name == "reset" and status == "Ok" for name, status, _ in events)
    assert runtime.calls[-2:] == [("control", False), ("reset",)]
    supervisor.close()


def test_bundle_verification_and_reset_eligibility(config):
    """Commit complete matching bundles; Reset prevents reactivation of old files."""
    runtime = FakeRuntime()
    supervisor = Supervisor(runtime, config)
    candidate = Path(config.artifact_root) / "candidate"
    candidate.mkdir()
    (candidate / "policy.trt").write_bytes(
        b"fictional CPU fixture; never loaded by a GPU"
    )
    model = engine_id(candidate / "policy.trt")
    generation = supervisor.record["generation"]
    metadata = dict(
        generation=generation,
        active_model_id=model,
        artifact_type="tensorrt",
        config=supervisor.desired,
        config_revision=revision(supervisor.desired),
        session_id="test",
    )
    (candidate / "policy.trt.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        verify_candidate(candidate, supervisor.root, generation)
    import numpy as np

    np.savez(
        candidate / "policy.trt.training.npz",
        metadata_json=json.dumps(dict(metadata, model_id=model)),
        policy_state__weight=np.zeros(1),
        dynamics_0_state__weight=np.zeros(1),
    )
    assert supervisor.publish(candidate, generation, threading.Event()) == model
    assert (supervisor.root / "current").resolve() == candidate
    supervisor.submit("reset", {"request": True})
    finish(supervisor)
    with pytest.raises(ValueError):
        supervisor.publish(
            candidate, supervisor.record["generation"], threading.Event()
        )
    assert supervisor.record["eligible"] is None
    supervisor.close()


class Client:
    """Actual INDI vector storage with synchronous remote-device responses."""

    status = C.ConnectionStatus.CONNECTED

    def __init__(self):
        """Create an acknowledged disabled NN fixture and catalog."""
        self.props = {}
        self.sent = []
        self.callbacks = []
        self.connection = SimpleNamespace(
            send=self.send,
            add_callback=lambda *a: None,
            remove_callback=lambda *a: None,
        )

    def get_properties(self, *args):
        """Send fresh simulated definitions through the observer callback boundary."""
        for key, prop in self.props.items():
            prop.device = key.split(".")[0]
            self.receive(prop)

    def register_callback(self, callback, device_name=None, property_name=None):
        """Attach the same scoped property callbacks as the production client."""
        self.callbacks.append((callback, device_name, property_name))

    def unregister_callback(self, callback, device_name=None, property_name=None):
        """Detach a scoped observer."""
        self.callbacks.remove((callback, device_name, property_name))

    def receive(self, message):
        """Dispatch a simulated server update, distinct from command delivery."""
        for callback, device, prop in list(self.callbacks):
            if (device is None or device == message.device) and (
                prop is None or prop == message.name
            ):
                callback(message)

    def __getitem__(self, key):
        """Match the client property/element indexing contract."""
        parts = key.split(".")
        prop = self.props[".".join(parts[:2])]
        return prop if len(parts) == 2 else prop[parts[2]]

    def send(self, message):
        """Capture one typed command without contacting a device."""
        self.sent.append(message)


def test_real_xdevice_gui_contract_without_network(config):
    """Build real property definitions and verify Busy then completion on the GUI contract."""
    app = object.__new__(po4aoTrainer)
    app.config, app.name, app.client = config, "po4aoTrainer", Client()
    app.properties, app.callbacks = {}, {}
    from collections import defaultdict

    app.callbacks = defaultdict(list)
    app.log = logging.getLogger("offline-po4ao")
    emitted = []
    app.update_property = lambda prop: emitted.append((prop.name, prop.state))
    app.setup()
    assert (
        not {"nn_activate", "reconstructor", "link_modulation"} & app.properties.keys()
    )
    assert app.supervisor.runtime.observations.devices == ("po4ao",)
    assert set(app.properties["devices"]) == {"po4ao"}
    for name in ("nn_activate", "reconstructor", "link_modulation"):
        with pytest.raises(ValueError, match="Unknown PO4AO"):
            app.supervisor.submit(name, {})
    app.loop()
    assert set(app.properties["status"]) == set(app.supervisor.status)
    assert all(
        p.perm == C.PropertyPerm.READ_ONLY
        for n, p in app.properties.items()
        if n in ("status", "devices", "progress", "training_config")
    )
    app.teardown()
    app._telem_file = SimpleNamespace(close=lambda: None)


def test_duplicate_reset_does_not_cancel_accepted_reset(config):
    """Reject duplicate admission without cancelling the operation already accepted."""
    supervisor = Supervisor(FakeRuntime(), config)
    supervisor.submit("reset", {"request": True})
    cancel = supervisor.domains["training"]["cancel"]
    with pytest.raises(RuntimeError):
        supervisor.submit("reset", {"request": True})
    assert not cancel.is_set()
    assert finish(supervisor)[-1][1] == "Ok"
    supervisor.close()


def test_reload_requires_fresh_ack_even_for_same_model():
    """Old cached outcomes cannot settle a request; timeouts preserve its eventual acknowledgement."""
    client = Client()
    status = P.TextVector(device="po4ao", name="model_status")
    for key, value in dict(
        instance="runtime",
        sequence="1",
        reset_request="1",
        active_id="model-a",
        reload="idle",
        reload_error="old",
    ).items():
        status.add_element(DefText(name=key, _value=value))
    client.props["po4ao.model_status"] = status
    reload = P.SwitchVector(
        device="po4ao", name="reload_engine", rule=C.SwitchRule.AT_MOST_ONE
    )
    reload.add_element(DefSwitch(name="toggle", _value=C.SwitchState.OFF))
    client.props["po4ao.reload_engine"] = reload
    runtime = Runtime(client, "po4ao", 0.04)
    with pytest.raises(TimeoutError):
        runtime.reload("model-a", threading.Event())
    assert runtime.reload_ticket is not None
    with pytest.raises(RuntimeError, match="unresolved"):
        runtime.begin_reload("another-model")
    for state in (C.PropertyState.BUSY, C.PropertyState.OK):
        reload.state = state
        client.receive(reload)
    assert runtime.finish_reload(threading.Event()) == "loaded"
    runtime.close()
    assert not client.callbacks


def test_cancelled_publication_releases_worker_candidate_wait(config):
    """Disable/Reset cancellation cannot strand a worker waiting for a candidate decision."""
    supervisor = Supervisor(FakeRuntime(), config)
    messages = []
    supervisor.worker = SimpleNamespace(
        state="running", closed=False, send=messages.append, stop=lambda: None
    )

    def cancelled(*args, **kwargs):
        """Reproduce cancellation after the candidate reached the training executor."""
        raise InterruptedError("Publication cancelled")

    supervisor.publish = cancelled
    with pytest.raises(InterruptedError):
        supervisor.execute(
            "publish",
            {"path": "candidate"},
            threading.Event(),
            supervisor.record["generation"],
        )
    assert messages == [{"candidate_path": "candidate", "candidate_id": ""}]
    supervisor.close()


def test_disable_recollect_and_reset_preserve_the_expected_worker_lifecycle(config):
    """Only Reset replaces the session; Disable pauses and recollection resumes conditionally."""
    runtime = FakeRuntime()
    workers = []

    class WorkerFixture:
        """Immediate pipe acknowledgements around the real compound operations."""

        def __init__(self, run, launch):
            """Record one owned session, without creating a numerical process."""
            self.run, self.state, self.closed = run, "running", False
            self.revision, self.iteration, self.error = (
                revision(launch["config"]),
                1,
                "",
            )
            self.config_ticket = launch["config_ticket"]
            self.process = SimpleNamespace(poll=lambda: None)
            workers.append(self)

        def send(self, message):
            """Acknowledge a worker's run/pause transition."""
            if "run" in message:
                self.state = "running" if message["run"] else "paused"
                runtime.calls.append(("worker", self.state))

        def events(self):
            """No unsolicited publication from this worker fixture."""
            return []

        def stop(self):
            """Record termination before the reset request is sent to the runtime."""
            self.closed = True
            runtime.calls.append(("stop",))

    supervisor = Supervisor(runtime, config, worker_factory=WorkerFixture)
    try:

        def operation(command, values):
            """Run a compound request to its observed terminal acknowledgement."""
            supervisor.submit(command, values)
            assert finish(supervisor)[-1][1] == "Ok"

        operation("learning", {"run": True})
        operation("po4ao_enable", {"toggle": False})
        assert runtime.calls[-2:] == [("control", False), ("worker", "paused")]
        operation("po4ao_enable", {"toggle": True})
        assert workers[0].state == "paused"
        operation("recollect", {"request": True})
        assert workers[0].state == "paused"
        operation("learning", {"run": True})
        operation("recollect", {"request": True})
        assert workers[0].state == "running"
        assert len(workers) == 1
        operation("reset", {"request": True})
        assert runtime.calls[-3:] == [("control", False), ("reset",), ("stop",)]
        assert supervisor.worker is None and workers[0].closed
    finally:
        supervisor.close()


def test_disable_still_selects_integrator_after_worker_failure(config):
    """A failed training worker cannot prevent the operator from disabling PO4AO."""
    runtime = FakeRuntime()
    supervisor = Supervisor(runtime, config)
    supervisor.worker = SimpleNamespace(
        state="error",
        error="build failed",
        closed=True,
        stop=lambda: runtime.calls.append(("stop",)),
    )
    supervisor.execute(
        "po4ao_enable",
        {"toggle": False},
        threading.Event(),
        supervisor.record["generation"],
    )
    assert runtime.calls == [("control", False), ("stop",)]
    supervisor.close()


def test_worker_protocol_failure_is_reported_without_stopping_companion(config):
    """Malformed/closed worker pipes fail that owned session, not reconstruction control."""
    supervisor = Supervisor(FakeRuntime(), config)

    def malformed():
        """Inject a bad worker record at the process boundary."""
        raise ValueError("malformed worker record")

    stopped = []
    supervisor.worker = SimpleNamespace(
        state="running",
        error="",
        events=malformed,
        revision="",
        process=SimpleNamespace(poll=lambda: 1),
        stop=lambda: stopped.append(True),
    )
    supervisor.tick()
    assert supervisor.status["learning"] == "error"
    assert supervisor.status["error"] == "malformed worker record"
    assert stopped
    supervisor.close()


def test_live_parameter_and_catalog_indi_contract(config):
    """The actual XDevice sends Busy/current/target, rejects forged current, and redefines catalogs."""
    from collections import defaultdict

    app = object.__new__(po4aoTrainer)
    app.config, app.name, app.client = config, "po4aoTrainer", Client()
    app.connection = app.client.connection
    app.properties, app.callbacks = {}, defaultdict(list)
    app.log = logging.getLogger("offline-po4ao")
    emitted = []
    app.update_property = lambda prop: emitted.append(
        (prop.name, prop.state, {key: prop[key] for key in prop})
    )
    app.setup()
    try:
        prop = app.properties["command_regularization"]
        app.handle_request(prop, {"target": 0.025})
        states = [state for name, state, values in emitted if name == prop.name]
        assert states[-2:] == [C.PropertyState.BUSY, C.PropertyState.OK]
        assert prop["current"] == prop["target"] == 0.025
        app.handle_request(prop, {"current": 1, "target": 1})
        app.loop()
        assert prop.state == C.PropertyState.ALERT
        assert prop["target"] == 0.025
        app.handle_request(prop, {"target": 0.01})
        assert prop.state == C.PropertyState.OK
        app.supervisor.catalog = {
            "saved_id": dict(
                label="internal", created_utc="2026-11-03T15:22:00+00:00", iteration=300
            )
        }
        app.publish_catalog()
        assert set(app.properties["policy_catalog"]) == {"saved_id"}
        assert app.properties["policy_catalog"].perm == C.PropertyPerm.READ_ONLY
    finally:
        app.teardown()
        app._telem_file = SimpleNamespace(close=lambda: None)


def test_supervisor_and_notebook_import_boundaries():
    """Check fresh-process imports without inheriting this test runner's torch imports."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from xapp.po4aoTrainer.app import main
from xapp.po4aoTrainer import supervisor, worker
assert "torch" not in sys.modules
assert "matplotlib" not in sys.modules
assert "ImageStreamIOWrap" not in sys.modules
from xapp.po4aoTrainer.training import TrainingSession
from xapp.po4aoTrainer.po4ao_runtime import ReplaySnapshot, publish_tensorrt_engine
from xapp.po4aoTrainer.observer import save_diagnostics, load_diagnostics
from xapp.po4aoTrainer import po4ao_diagnostics
import torch
assert not torch.cuda.is_initialized()
assert "ImageStreamIOWrap" not in sys.modules
""",
        ],
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("penalty", [0.0, 0.5, [0.1, 0.2, 0.3, 0.4]])
def test_penalty_spelling_preserves_resolved_config(config, penalty):
    """Renaming an installed key alone preserves values, revision and Reset eligibility."""
    path = Path(config.training_config)
    settings = tomllib.loads(path.read_text())
    settings["policy_update"]["weight"] = penalty
    path.write_text(tomli_w.dumps(settings))
    previous = load_training(path, config.runtime_config)
    settings["policy_update"]["penalty"] = settings["policy_update"].pop("weight")
    path.write_text(tomli_w.dumps(settings))
    current = load_training(path, config.runtime_config)
    assert current == previous
    assert revision(current) == revision(previous)
    assert current["policy_update"]["weight"] == penalty
    assert not needs_reset(previous, current)


@pytest.mark.parametrize("penalty", [0.5, 1.0])
def test_penalty_rejects_both_spellings(config, penalty):
    """Do not silently choose either an equal or conflicting duplicate setting."""
    path = Path(config.training_config)
    settings = tomllib.loads(path.read_text())
    settings["policy_update"]["penalty"] = penalty
    path.write_text(tomli_w.dumps(settings))
    with pytest.raises(ValueError, match="not both"):
        load_training(path, config.runtime_config)


@pytest.mark.parametrize(
    "penalty", [-0.5, [0.1, -0.2, 0.3, 0.4], [0.1, 0.2], float("nan")]
)
def test_penalty_keeps_validation(config, penalty):
    """The new spelling still rejects invalid scalars, modal values and modal counts."""
    path = Path(config.training_config)
    settings = tomllib.loads(path.read_text())
    settings["policy_update"].pop("weight")
    settings["policy_update"]["penalty"] = penalty
    path.write_text(tomli_w.dumps(settings))
    with pytest.raises(ValueError):
        load_training(path, config.runtime_config)


def test_penalty_hot_update_reaches_existing_regularizer(config):
    """Hot editing the public key changes the real penalty while retaining the policy and Adam."""
    path = Path(config.training_config)
    previous = load_training(path, config.runtime_config)
    session = TrainingSession(previous, device="cpu")
    warmup = np.ones((96, session.controlled_nmodes), dtype=np.float32)
    session._initialize_normalization((warmup, 0.3 * warmup))
    anchors = torch.ones(8, 2 * session.n_history - 1, 1, session.controlled_nmodes)
    session.policy_update.set_anchors(anchors)
    original_cost = session.policy_update.penalty(session.policy).detach()
    assert original_cost > 0
    policy, optimizer = session.policy, session.policy_optimizer
    parameters = deepcopy(policy.state_dict())
    normalization = deepcopy(session.normalization_metadata)
    optimizer.state["sentinel"] = {"step": 7}

    settings = tomllib.loads(path.read_text())
    settings["policy_update"].pop("weight")
    settings["policy_update"]["penalty"] = 0.5
    settings["RL"]["loss_function_penalty"] = 0.01
    path.write_text(tomli_w.dumps(settings))
    candidate = load_training(path, config.runtime_config)
    assert not needs_reset(previous, candidate)
    apply_hot(session, candidate)
    assert session.policy is policy and session.policy_optimizer is optimizer
    assert optimizer.state["sentinel"]["step"] == 7
    assert session.normalization_metadata == normalization
    assert session.config["RL"]["loss_function_penalty"] == 0.01
    assert session.config_revision == revision(candidate)
    for name, value in parameters.items():
        torch.testing.assert_close(
            session.policy.state_dict()[name], value, rtol=0, atol=0
        )
    assert session.policy_update.anchors is None
    session.policy_update.set_anchors(anchors)
    torch.testing.assert_close(
        session.policy_update.penalty(session.policy).detach(), 0.5 * original_cost
    )


@pytest.mark.parametrize("server_fifo_exists", [False, True])
def test_framework_fifo_constructor(tmp_path, monkeypatch, server_fifo_exists):
    """Accept XDevice's default control path whether or not indiserver is present."""
    monkeypatch.setattr(
        transports.IndiFifoConnection, "MAGAOX_USER", pwd.getpwuid(os.getuid()).pw_name
    )
    if server_fifo_exists:
        os.mkfifo(tmp_path / "indiserver.ctrl")
    make_fifo = transports.make_fifo_and_open
    with ExitStack() as handles:

        def open_fifo(path, mode):
            """Close all fixture handles even if the dependency constructor raises."""
            return handles.enter_context(make_fifo(path, mode))

        monkeypatch.setattr(transports, "make_fifo_and_open", open_fifo)
        connection = transports.IndiFifoConnection(
            name="po4aoTrainer", fifos_root=str(tmp_path)
        )
        for suffix in ("in", "out", "ctrl"):
            assert stat.S_ISFIFO((tmp_path / f"po4aoTrainer.{suffix}").stat().st_mode)
        assert connection._reader is None
        assert connection._writer is None


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
        self.loading = None
        self.client = IndiClient()
        self.client.connection = Wire(self)
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
            run_phase="unverified",
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
        self.runtime = Runtime(self.client, "po4ao", timeout)

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

    def command(self, message):
        """Simulate PO4AO request/after-frame semantics."""
        device, name = message.device, message.name
        if device == "po4ao":
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
        self.config_ticket = (
            1  # The initial supervisor configuration has been acknowledged.
        )
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
    assert rig.runtime.get("po4ao", "control_enable", "toggle") == C.SwitchState.OFF
    wire = rig.client.connection
    wire.status = C.ConnectionStatus.STOPPED
    wire.dispatch_callbacks(C.TransportEvent.disconnection, wire.status)
    wire.status = C.ConnectionStatus.CONNECTED
    wire.dispatch_callbacks(C.TransportEvent.connection, wire.status)
    assert (
        rig.client["po4ao.control_enable.toggle"] == C.SwitchState.OFF
    )  # The actual client retained its cache.
    with pytest.raises(ConnectionError):
        rig.runtime.get("po4ao", "control_enable", "toggle")
    rig.runtime.observations.refresh()
    assert rig.runtime.get("po4ao", "control_enable", "toggle") == C.SwitchState.OFF
    rig.client.handle_message(M.DelProperty(device="po4ao"))
    with pytest.raises(ConnectionError):
        rig.runtime.get("po4ao", "control_enable", "toggle")
    rig.close()


def test_silent_device_expires_but_unchanged_fresh_definitions_renew_it(monkeypatch):
    """Unchanged values are allowed; only received current-lifetime evidence renews freshness."""
    rig = Rig()
    clock = [time.monotonic()]
    monkeypatch.setattr(
        "xapp.po4aoTrainer.switching.time.monotonic", lambda: clock[0]
    )
    clock[0] += 6
    with pytest.raises(ConnectionError):
        rig.runtime.get("po4ao", "control_enable", "toggle")
    rig.props["po4ao", "model_status"]["sequence"] = "2"
    rig.runtime.observations.refresh()
    assert rig.runtime.get("po4ao", "control_enable", "toggle") == C.SwitchState.OFF
    rig.close()


def test_runtime_heartbeat_detects_stalled_applogic_despite_fresh_definitions(
    monkeypatch,
):
    """Re-serving a frozen C++ status does not prove the control loop is being observed."""
    rig = Rig()
    clock = [time.monotonic()]
    monkeypatch.setattr(
        "xapp.po4aoTrainer.switching.time.monotonic", lambda: clock[0]
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


def test_publication_completion_cannot_promote_a_replacement_candidate(
    config, monkeypatch
):
    """Hold the status thread's A result while publication replaces it with a still-loading B."""
    rig = Rig(0.03)
    rig.auto_load = False
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    a, model_a = bundle(supervisor, "a", worker.session)
    b, model_b = bundle(supervisor, "b", worker.session)
    generation = supervisor.record["generation"]
    captured, release = threading.Event(), threading.Event()
    original = supervisor.finish_publication
    errors = []

    def delayed_completion(*args):
        """Pause after obtaining A's result but before committing it under the supervisor lock."""
        if threading.current_thread() is thread:
            captured.set()
            assert release.wait(2)
        return original(*args)

    def status_tick():
        """Surface assertion/thread failures in the calling test."""
        try:
            supervisor.tick()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=status_tick)
    try:
        with pytest.raises(TimeoutError):
            supervisor.publish(a, generation, threading.Event())
        rig.complete_load()
        monkeypatch.setattr(supervisor, "finish_publication", delayed_completion)
        thread.start()
        assert captured.wait(2)
        with pytest.raises(TimeoutError):
            supervisor.publish(b, generation, threading.Event())
        assert rig.loading[0] == model_b
        release.set()
        thread.join(2)
        assert not thread.is_alive() and not errors
        assert rig.runtime.runtime_state()["active_id"] == model_a
        saved = json.loads(supervisor.record_path.read_text())
        assert saved["pending"]["model_id"] == model_b
        assert saved["eligible"]["model_id"] == model_a
        assert not any(
            m.get("candidate_outcome_path") == str(b) for m in worker.messages
        )
        monkeypatch.setattr(supervisor, "finish_publication", original)
        rig.complete_load()
        supervisor.tick()
        assert supervisor.record["eligible"]["model_id"] == model_b
        assert supervisor.record["pending"] is None
        assert (
            sum(m.get("candidate_outcome_path") == str(b) for m in worker.messages) == 1
        )
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(2)
        supervisor.close()
        rig.close()


@pytest.mark.parametrize("after_ticket", [False, True])
def test_failed_reload_issuance_never_reuses_the_previous_ticket(
    config, monkeypatch, after_ticket
):
    """A pre-issuance failure rejects B; an uncertain send retains only B's own ticket/reference."""
    rig = Rig(0.03)
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    a, model_a = bundle(supervisor, "a", worker.session)
    b, model_b = bundle(supervisor, "b", worker.session)
    generation = supervisor.record["generation"]
    supervisor.publish(a, generation, threading.Event())
    rig.auto_load = False
    original = rig.runtime.begin_reload

    def fail(model, issued=None):
        """Inject failure on either side of ticket installation/possible remote delivery."""
        if after_ticket:
            original(model, issued=issued)
        raise OSError("reload issuance failed")

    monkeypatch.setattr(rig.runtime, "begin_reload", fail)
    try:
        with pytest.raises(OSError, match="issuance failed"):
            supervisor.execute(
                "publish", {"path": str(b)}, threading.Event(), generation
            )
        supervisor.tick()
        saved = json.loads(supervisor.record_path.read_text())
        assert saved["eligible"]["model_id"] == model_a
        assert not any(
            m.get("candidate_outcome_path") == str(b) for m in worker.messages
        )
        if after_ticket:
            assert saved["pending"]["model_id"] == model_b
            assert {
                "candidate_path": str(b),
                "candidate_id": model_b,
            } in worker.messages
            rig.complete_load()
            supervisor.tick()
            assert supervisor.record["eligible"]["model_id"] == model_b
        else:
            assert saved["pending"] is None
            assert {"candidate_path": str(b), "candidate_id": ""} in worker.messages
            assert rig.loading is None
    finally:
        supervisor.close()
        rig.close()


def test_candidate_reference_is_bound_before_reload_delivery(config, monkeypatch):
    """Even an immediate load cannot reach the runtime before the worker knows its exact candidate."""
    rig = Rig()
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    path, model = bundle(supervisor, session=worker.session)
    command = rig.command

    def observe(message):
        """Check the production issuance boundary before acknowledging any engine load."""
        if message.device == "po4ao" and message.name == "reload_engine":
            pending, ticket = supervisor.publication
            assert pending["model_id"] == ticket["model_id"] == model
            assert ticket is rig.runtime.reload_ticket
            assert worker.session.policy_update.pending[0] == model
        command(message)

    monkeypatch.setattr(rig, "command", observe)
    try:
        supervisor.execute(
            "publish",
            {"path": str(path)},
            threading.Event(),
            supervisor.record["generation"],
        )
        assert supervisor.record["eligible"]["model_id"] == model
    finally:
        supervisor.close()
        rig.close()


def test_new_publication_cannot_replace_an_unresolved_ticket(config):
    """Reject B before changing the link or persisted pending A when A may still activate."""
    rig = Rig(0.03)
    rig.auto_load = False
    supervisor = Supervisor(rig.runtime, config)
    a, model_a = bundle(supervisor, "a")
    b, _ = bundle(supervisor, "b")
    try:
        with pytest.raises(TimeoutError):
            supervisor.publish(a, supervisor.record["generation"], threading.Event())
        with pytest.raises(RuntimeError, match="reload is still unresolved"):
            supervisor.publish(b, supervisor.record["generation"], threading.Event())
        assert supervisor.record["pending"]["model_id"] == model_a
        assert (supervisor.root / "current").resolve() == a
        rig.complete_load()
        supervisor.tick()
        assert supervisor.record["eligible"]["model_id"] == model_a
    finally:
        supervisor.close()
        rig.close()


def test_whole_app_routing_pauses_learning_without_resuming_it(config):
    """Native routing preserves the managed model/session and never restarts numerical work."""
    rig = Rig()
    supervisor = Supervisor(rig.runtime, config)
    worker = NumericalWorker(supervisor.desired)
    supervisor.worker = worker
    try:
        supervisor.reconcile_runtime()
        rig.update("po4ao", "model_status", run_phase="switching_input")
        supervisor.reconcile_runtime()
        assert worker.state == "paused"
        assert not worker.closed
        with pytest.raises(RuntimeError, match="Start PO4AO"):
            supervisor.submit("learning", {"run": True})
        rig.update("po4ao", "model_status", run_phase="stopped")
        supervisor.reconcile_runtime()
        assert worker.state == "paused"
        # A new policy-selection operation while stopped is not cancelled each tick.
        operation = supervisor.domains["training"]
        operation["cancel"] = threading.Event()
        operation["command"] = "set_policy"
        supervisor.reconcile_runtime()
        assert not operation["cancel"].is_set()
        rig.update("po4ao", "model_status", run_phase="running")
        supervisor.reconcile_runtime()
        assert worker.state == "paused"
        assert supervisor.worker is worker
        assert not any(message.get("run") is True for message in worker.messages)
    finally:
        supervisor.close()
        rig.close()
