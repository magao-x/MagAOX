## \file test_supervisor.py
# \brief Offline companion contract, configuration and reset/publication regression tests.
# \ingroup po4aoTrainer_unit_test
"""Use real PurePyINDI2 vectors and the production supervisor; no network connection."""

import json
import logging
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomli_w
from purepyindi2 import constants as C, properties as P
from purepyindi2.messages import DefText, DefSwitch, DefNumber

from xapp.po4aoTrainer.configuration import (
    load_training,
    revision,
    needs_reset,
    apply_hot,
)
from xapp.po4aoTrainer.supervisor import Supervisor
from xapp.po4aoTrainer.switching import Runtime
from xapp.po4aoTrainer.artifacts import engine_id, verify_candidate
from xapp.po4aoTrainer.app import po4aoTrainer, TrainerConfig


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
