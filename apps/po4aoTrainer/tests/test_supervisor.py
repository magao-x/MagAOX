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

    nn, po4ao, modulation = "nn", "po4ao", "mod"

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

    def source(self):
        """Return an observed NN source for expected-source tests."""
        return "nn"

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

    def switch(self, target, cancel, **kwargs):
        """Record independent reconstruction work."""
        self.calls.append(("switch", target))


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


def test_reset_cancels_collection_but_reconstruction_is_independent(config):
    """A held warmup never locks reconstruction and Reset supersedes it."""
    runtime = FakeRuntime()
    runtime.hold = True
    supervisor = Supervisor(runtime, config)
    supervisor.submit("recollect", {"request": True})
    supervisor.submit("reconstructor", {"target": "cacao", "expected": "nn"})
    assert finish(supervisor, "reconstruction")[0][1] == "Ok"
    supervisor.submit("reset", {"request": True})
    events = finish(supervisor)
    assert any(name == "reset" and status == "Ok" for name, status, _ in events)
    assert ("switch", "cacao") in runtime.calls
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
    prop = app.properties["link_modulation"]
    message = prop.make_new_property(toggle=C.SwitchState.ON)
    app.handle_request(prop, message)
    assert emitted[-1] == ("link_modulation", C.PropertyState.BUSY)
    time.sleep(0.02)
    app.loop()
    assert emitted[-1] == ("link_modulation", C.PropertyState.OK)
    assert prop["toggle"] == C.SwitchState.ON
    assert set(app.properties["status"]) == set(app.supervisor.status)
    assert all(
        p.perm == C.PropertyPerm.READ_ONLY
        for n, p in app.properties.items()
        if n in ("status", "devices", "progress", "training_config")
    )
    app.teardown()
    app._telem_file = SimpleNamespace(close=lambda: None)


@pytest.mark.parametrize(
    "target,linked", [("nn", False), ("nn", True), ("cacao", False), ("cacao", True)]
)
def test_switching_orders_actual_writers_and_optional_modulation(target, linked):
    """Use real INDI vectors to confirm ordering, zero unlinked writes, and no loop/PO4AO access."""
    client = Client()

    def vector(device, name, values, switch=False):
        """Build a real readback definition for the remote simulator."""
        prop = (
            P.SwitchVector
            if switch
            else P.NumberVector if device == "mod" else P.TextVector
        )(
            device=device,
            name=name,
            state=C.PropertyState.OK,
            **({"rule": C.SwitchRule.AT_MOST_ONE} if switch else {}),
        )
        for key, value in values.items():
            prop.add_element(
                DefSwitch(name=key, _value=value)
                if switch
                else (
                    DefNumber(
                        name=key,
                        _value=float(value),
                        format="%g",
                        min=0,
                        max=10000,
                        step=1,
                    )
                    if device == "mod"
                    else DefText(name=key, _value=value)
                )
            )
        client.props[f"{device}.{name}"] = prop

    vector(
        "nn",
        "enable",
        {"toggle": C.SwitchState.ON if target == "cacao" else C.SwitchState.OFF},
        True,
    )
    vector(
        "nn",
        "model_status",
        {
            "status": "active" if target == "cacao" else "disabled",
            "loaded": "a.trt",
            "active": "a.trt" if target == "cacao" else "",
            "requested": "a.trt",
        },
    )
    vector("nn", "model", {"model_0000": C.SwitchState.ON}, True)
    client.props["nn.model"]._elements["model_0000"].label = "a.trt"
    for prop, value in [
        ("modState", 4 if target == "nn" else 3),
        ("modRadius", 3),
        ("modFrequency", 2000),
    ]:
        vector("mod", prop, {"current": str(value), "target": str(value)})
    events = []

    class Writer:
        """Writer completion double, preserving command/observation ordering."""

        enabled = target == "nn"

        def command(self, enabled):
            """Only this fixture acknowledges synchronous process exit/start."""
            events.append(("cacao", enabled))
            self.enabled = enabled

        def running(self):
            """Record completion evidence separate from FIFO delivery."""
            events.append(("observe", self.enabled))
            return self.enabled

    writer = Writer()

    def response(message):
        """Acknowledge NN outputs and modulation readbacks without loop properties."""
        client.sent.append(message)
        events.append((message.device, message.name))
        prop = client.props[f"{message.device}.{message.name}"]
        for name in message:
            prop[name] = message[name]
        if message.name == "enable":
            client.props["nn.model_status"]["status"] = (
                "active" if message["toggle"] == C.SwitchState.ON else "disabled"
            )
            client.props["nn.model_status"]["active"] = (
                "a.trt" if message["toggle"] == C.SwitchState.ON else ""
            )
            if message["toggle"] == C.SwitchState.ON:
                assert not writer.enabled
        elif message.device == "mod":
            prop["current"] = message["target"]
        client.receive(prop)
        if message.device == "nn":
            client.receive(client.props["nn.model_status"])

    client.connection.send = response
    runtime = Runtime(client, writer, "nn", "po4ao", "mod", 0.1)
    runtime.switch(target, threading.Event(), linked=linked)
    assert runtime.source() == target
    assert all(msg.device in ("nn", "mod") for msg in client.sent)
    assert any(msg.device == "mod" for msg in client.sent) == linked
    if target == "cacao":
        assert events.index(("nn", "enable")) < events.index(("cacao", True))
    else:
        assert events.index(("cacao", False)) < len(events) - 1


def test_missing_writer_evidence_never_enables_nn():
    """Missing current definitions cannot authorize a writer handover."""
    client = Client()
    writer = SimpleNamespace(command=lambda enabled: None, running=lambda: True)
    runtime = Runtime(client, writer, "nn", "po4ao", "mod", 0.01)
    with pytest.raises(ConnectionError):
        runtime.switch("nn", threading.Event())
    assert not client.sent
    runtime.close()


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


def test_milk_writer_requires_live_identity_and_detects_reused_pid(monkeypatch):
    """Installed binding fields are verified independently of command flags."""
    from xapp.po4aoTrainer.switching import MilkWriter

    md = SimpleNamespace(
        runpid=123, runpidstarttime=SimpleNamespace(tv_sec=1, tv_nsec=2)
    )
    fps = SimpleNamespace(md=lambda: md, RUNrunning=1)
    birth = [10.0]
    process = SimpleNamespace(
        status=lambda: "running",
        cmdline=lambda: ["milk", "wfs2cmodeval-1"],
        create_time=lambda: birth[0],
    )
    monkeypatch.setattr(
        "xapp.po4aoTrainer.switching.psutil.Process", lambda pid: process
    )
    writer = MilkWriter("wfs2cmodeval-1", "unused", "wfs2cmodeval-1", fps=fps)
    assert writer.running()
    birth[0] = 20
    with pytest.raises(RuntimeError, match="reused"):
        writer.running()
    fps.RUNrunning = -1
    with pytest.raises(RuntimeError, match="unverified"):
        writer.running()
    fps.RUNrunning = 0
    assert not writer.running()


def test_fifo_delivery_is_bounded_and_separate_from_writer_evidence(tmp_path):
    """Opening an absent reader never blocks, and an ordinary file is never treated as a FIFO."""
    import os
    from xapp.po4aoTrainer.switching import MilkWriter

    path = tmp_path / "controller.fifo"
    os.mkfifo(path)
    writer = MilkWriter("wfs2cmodeval-1", path, "wfs2cmodeval-1", fps=object())
    with pytest.raises(OSError):
        writer.command(False)
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        writer.command(False)
        assert os.read(fd, 100) == b"runstop wfs2cmodeval-1\n"
    finally:
        os.close(fd)
    path.unlink()
    path.write_text("unchanged")
    with pytest.raises(RuntimeError, match="not a FIFO"):
        writer.command(True)
    assert path.read_text() == "unchanged"


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
    runtime = Runtime(client, None, "nn", "po4ao", "mod", 0.04)
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
        process=SimpleNamespace(poll=lambda: 1),
        stop=lambda: stopped.append(True),
    )
    supervisor.tick()
    assert supervisor.status["learning"] == "error"
    assert supervisor.status["error"] == "malformed worker record"
    assert stopped
    supervisor.submit("reconstructor", {"target": "cacao", "expected": "nn"})
    assert finish(supervisor, "reconstruction")[-1][1] == "Ok"
    supervisor.close()
