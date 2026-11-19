## \file test_library.py
# \brief Offline live-parameter, saved-pair and restored-session regression tests.
# \ingroup po4aoTrainer_unit_test
"""Use real CPU networks and publication metadata; never build engines or contact INDI."""

from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import tomli_w
import torch

from test_supervisor import config, FakeRuntime, finish
from test_training import inputs, make_session
from xapp.po4aoTrainer.configuration import load_training, revision, apply_hot
from xapp.po4aoTrainer.library import PolicyLibrary, engine_id
from xapp.po4aoTrainer.po4ao_runtime import save_training_snapshot
from xapp.po4aoTrainer.supervisor import Supervisor
from xapp.po4aoTrainer.training import TrainingSession


def candidate(directory, session, generation):
    """Write the real portable format beside an explicitly fictional CPU engine."""
    directory.mkdir()
    (directory / "policy.trt").write_bytes(
        b"CPU test engine " + str(directory).encode()
    )
    identity = engine_id(directory / "policy.trt")
    metadata = dict(
        artifact_type="tensorrt",
        generation=generation,
        active_model_id=identity,
        config=session.config,
        config_revision=session.config_revision,
        session_id=session.session_id,
        iteration=session.iteration,
        normalization=session.normalization_metadata,
        policy_update_action_scale=session.update_scale.tolist(),
    )
    (directory / "policy.trt.json").write_text(json.dumps(metadata))
    save_training_snapshot(
        directory / "policy.trt.training.npz",
        session.policy_copy,
        session.dynamics,
        identity,
        metadata,
    )
    return identity


@pytest.fixture
def trained(config, inputs):
    """Train a tiny normalized CPU pair using the actual configuration parser."""
    settings = load_training(config.training_config, config.runtime_config)
    session = make_session(settings)
    session.step(inputs[1], inputs[2], active_model_id="0")
    return session


def test_parameters_merge_pause_ack_and_file_precedence(config):
    """Successive edits merge, remain pending while paused and yield only to same-field valid file edits."""
    supervisor = Supervisor(FakeRuntime(), config)
    base = deepcopy(supervisor.desired)
    supervisor.applied = base
    supervisor.worker = SimpleNamespace(stop=lambda: None)
    supervisor.set_parameter("command_regularization", 0.27)
    supervisor.set_parameter("policy_learning_rate", 1e-7)
    supervisor.set_parameter("delta_policy_regularization", 0.8)
    supervisor.poll_config()
    values = supervisor.parameters()
    assert values["command_regularization"] == dict(
        current=base["RL"]["loss_function_penalty"], target=0.27, state="Busy"
    )
    assert values["policy_learning_rate"]["target"] == 1e-7
    assert supervisor.config_history[revision(supervisor.desired)] == supervisor.desired
    supervisor.applied = deepcopy(supervisor.desired)
    supervisor.worker.config_ticket = supervisor.config_ticket
    assert supervisor.parameters()["policy_learning_rate"]["state"] == "Ok"
    path = Path(config.training_config)
    import tomllib

    file_values = tomllib.loads(path.read_text())
    file_values["training"]["diagnostics_interval"] += 1
    path.write_text(tomli_w.dumps(file_values))
    supervisor.poll_config()
    assert supervisor.desired["RL"]["loss_function_penalty"] == 0.27
    file_values["RL"]["loss_function_penalty"] = 0.04
    path.write_text(tomli_w.dumps(file_values))
    supervisor.poll_config()
    assert supervisor.desired["RL"]["loss_function_penalty"] == 0.04
    assert supervisor.desired["training"]["policy_learning_rate"] == 1e-7
    for name, value in [
        ("policy_learning_rate", 0),
        ("command_regularization", -1),
        ("delta_policy_regularization", float("inf")),
        ("policy_learning_rate", float("nan")),
    ]:
        with pytest.raises(ValueError):
            supervisor.set_parameter(name, value)
    supervisor.close()
    restarted = Supervisor(FakeRuntime(), config)
    assert (
        restarted.desired["training"]["policy_learning_rate"]
        == base["training"]["policy_learning_rate"]
    )
    restarted.close()


def test_per_mode_parameter_and_reset_preserve_overrides(config):
    """Nonuniform weights never become a scalar, while Reset preserves valid session settings."""
    path = Path(config.training_config)
    import tomllib

    settings = tomllib.loads(path.read_text())
    settings["policy_update"]["weight"] = [0.1, 0.2, 0.3, 0.4]
    path.write_text(tomli_w.dumps(settings))
    supervisor = Supervisor(FakeRuntime(), config)
    assert supervisor.status["delta_policy_mode"] == "per_mode"
    with pytest.raises(ValueError, match="Per-mode"):
        supervisor.set_parameter("delta_policy_regularization", 1)
    supervisor.set_parameter("command_regularization", 0.9)
    supervisor.submit("reset", {"request": True})
    assert finish(supervisor) == [("reset", "Ok", "")]
    assert supervisor.desired["RL"]["loss_function_penalty"] == 0.9
    assert supervisor.desired["policy_update"]["weight"] == [0.1, 0.2, 0.3, 0.4]
    supervisor.close()


def test_round_trip_models_normalization_fresh_state_and_current_reference(
    config, trained, inputs, monkeypatch
):
    """Restore exact outputs and buffers with fresh Adam/replay; follow each newly activated policy."""
    source = Path(config.artifact_root) / "candidate"
    model = candidate(source, trained, "old")
    library = PolicyLibrary(Path(config.artifact_root) / "saved_engines")
    first = library.save(source, config.artifact_root, "old", "internal bright")
    second = library.save(source, config.artifact_root, "old", "internal bright")
    assert first["id"] != second["id"]
    settings = deepcopy(trained.config)
    settings["training"].update(
        policy_learning_rate=1e-7,
        first_model_publish_iteration=1000000,
        model_publish_interval=1,
        diagnostics_interval=1,
    )
    restored = TrainingSession(settings)
    path, manifest = library.load(first["id"], settings)
    restored.restore(path / "policy.trt.training.npz")
    restored.acknowledge_restore(model)
    probe = torch.randn(5, 2 * settings["MDP"]["n_history"] - 1, 1, 4)
    torch.testing.assert_close(
        restored.policy(probe), trained.policy_copy(probe), rtol=0, atol=0
    )
    for previous, loaded in zip(trained.dynamics.models, restored.dynamics.models):
        for key, value in previous.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
        dprobe = torch.randn(5, 2 * settings["MDP"]["n_history"], 1, 4)
        torch.testing.assert_close(loaded(dprobe), previous(dprobe), rtol=0, atol=0)
    assert restored.policy_optimizer.state == restored.dynamics_optimizer.state == {}
    assert restored.policy_optimizer.param_groups[0]["lr"] == 1e-7
    assert restored.iteration == 0 and restored.metrics_history == []
    assert restored.buffer_snapshot is None and restored.warmup_snapshot is None
    assert restored.dynamics_pretrained and restored.policy_update.anchors is None
    restored.policy_update.set_anchors(probe)
    torch.testing.assert_close(
        restored.policy_update.penalty(restored.policy), torch.tensor(0.0)
    )
    normalization = deepcopy(restored.normalization_metadata)
    scale = restored.observation_scale.copy()
    active = "1" * 16
    new = deepcopy(restored.policy)
    with torch.no_grad():
        next(new.parameters()).add_(0.3)
    restored.policy_update.stage_publication(new, active)
    restored.sync_active_policy(active)
    assert restored.policy_update.reference_id == active
    torch.testing.assert_close(restored.policy_update.reference_actions, new(probe))
    restored.sync_active_policy("0")
    assert restored.policy_update.reference is None
    restored.sync_active_policy(active)
    torch.testing.assert_close(restored.policy_update.reference_actions, new(probe))
    calls = []
    from xapp.po4aoTrainer import training

    original = training.train_nonlinear_dynamics

    def dynamics(*args, **kwargs):
        """Observe warm-start step count while retaining the real numerical calculation."""
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "train_nonlinear_dynamics", dynamics)
    result = restored.step(inputs[1], inputs[2], active_model_id=active)
    assert not any(
        "first publication" in reason
        for reason in result["record"]["publication_reasons"]
    )
    assert restored.normalization_metadata == normalization
    np.testing.assert_array_equal(restored.observation_scale, scale)
    assert calls[0]["steps"] == settings["training"]["dynamics_grad_steps"]
    assert restored.origin["iteration"] == trained.iteration
    assert manifest["model_id"] == model


def test_library_incomplete_corrupt_incompatible_and_interrupted_save(
    config, trained, monkeypatch
):
    """Incomplete writes remain invisible and neither corrupt nor incompatible bundles can load."""
    root = Path(config.artifact_root)
    source = root / "candidate"
    candidate(source, trained, "test")
    library = PolicyLibrary(root / "saved_engines")
    from xapp.po4aoTrainer import library as module

    original = module.shutil.copyfile

    def fail(source, target):
        """Simulate a failed write part-way through a bundle."""
        if str(source).endswith(".npz"):
            raise OSError("interrupted")
        return original(source, target)

    monkeypatch.setattr(module.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="interrupted"):
        library.save(source, root, "test", "attempt")
    assert library.catalog() == {} and list(library.root.iterdir()) == []
    monkeypatch.setattr(module.shutil, "copyfile", original)
    entry = library.save(source, root, "test", "valid")
    settings = deepcopy(trained.config)
    settings["MDP"]["n_history"] += 1
    with pytest.raises(ValueError, match="matching MDP.n_history"):
        library.load(entry["id"], settings)
    (library.root / entry["id"] / "policy.trt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        library.load(entry["id"], trained.config)


class LibraryRuntime(FakeRuntime):
    """Expose effective control and fresh warmup generations from the fake runtime."""

    def __init__(self):
        """Start with a retained-capable disabled runtime."""
        super().__init__()
        self.collection_generation = 1

    def get(self, device, prop, element):
        """Use acknowledged loaded identity when enabled, and the explicit generation."""
        from purepyindi2.constants import SwitchState

        enabled = self.observed["control_target"] == "on"
        values = dict(
            requested_generation=str(self.collection_generation),
            effective_controller="po4ao" if enabled else "integrator",
            effective_id=self.observed["active_id"] if enabled else "0" * 16,
            toggle=SwitchState.ON if enabled else SwitchState.OFF,
        )
        return (
            values[element] if element in values else super().get(device, prop, element)
        )

    def recollect(self, cancel, timeout):
        """Finish a distinct warmup without changing controller selection."""
        super().recollect(cancel, timeout)
        self.collection_generation += 1


class PreparedWorker:
    """CPU restore/export boundary with production supervisor admission and acknowledgements."""

    def __init__(self, run, launch):
        """Prepare both real networks but use a fictional engine for reload protocol tests."""
        self.run = run
        run.mkdir()
        self.session = TrainingSession(launch["config"])
        self.session.restore(launch["initial_snapshot"])
        self.prepared_path = str(run / "candidate")
        candidate(Path(self.prepared_path), self.session, launch["generation"])
        self.state, self.error, self.closed = "paused", "", False
        self.revision, self.iteration = revision(launch["config"]), 0
        self.config_ticket = launch.get("config_ticket", 0)
        self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)

    def send(self, message):
        """Acknowledge resume/pause and the exact restored activation."""
        if "restore_id" in message:
            self.session.acknowledge_restore(message["restore_id"])
        if "run" in message:
            self.state = "running" if message["run"] else "paused"

    def events(self):
        """No background events are needed at this numerical boundary."""
        return []

    def stop(self):
        """Record cleanup without touching any system process."""
        self.closed = True


@pytest.mark.parametrize("enabled", [False, True])
def test_set_policy_requires_fresh_warmup_preserves_control_and_library(
    config, trained, enabled
):
    """Restored sessions remain paused, keep current settings, and require a newer collection."""
    runtime = LibraryRuntime()
    runtime.request_control(enabled)
    supervisor = Supervisor(runtime, config, worker_factory=PreparedWorker)
    source = supervisor.root / "candidate"
    original = candidate(source, trained, supervisor.record["generation"])
    supervisor.publish(source, supervisor.record["generation"], threading.Event())
    supervisor.submit("save_policy", {"label": "internal"})
    assert finish(supervisor, "library") == [("save_policy", "Ok", "")]
    saved = next(iter(supervisor.catalog))
    assert supervisor.status["message"] == "save_policy complete: " + saved
    supervisor.set_parameter("policy_learning_rate", 1e-7)
    supervisor.submit("set_policy", {"id": saved})
    assert finish(supervisor) == [("set_policy", "Ok", "")]
    assert supervisor.worker.state == "paused"
    assert runtime.observed["control_target"] == ("on" if enabled else "off")
    assert supervisor.worker.session.policy_update.reference_id != original
    assert supervisor.worker.session.policy_optimizer.param_groups[0]["lr"] == 1e-7
    assert supervisor.status["warmup_required"] == "yes"
    with pytest.raises(RuntimeError, match="Fresh warmup"):
        supervisor.submit("learning", {"run": True})
    supervisor.submit("recollect", {"request": True})
    finish(supervisor)
    assert supervisor.status["warmup_required"] == "no"
    assert supervisor.worker.state == "paused"
    supervisor.submit("reset", {"request": True})
    finish(supervisor)
    assert saved in supervisor.library.catalog()
    assert supervisor.record.get("selected_policy", "") == ""
    assert not {"selected_policy", "saved_policy", "library_phase"} & supervisor.status.keys()
    assert supervisor.desired["training"]["policy_learning_rate"] == 1e-7
    supervisor.close()


def test_save_pins_acknowledged_pair_while_publication_advances(
    config, trained, monkeypatch
):
    """A concurrent activation cannot make Save combine files from different publications."""
    supervisor = Supervisor(LibraryRuntime(), config)
    root, generation = supervisor.root, supervisor.record["generation"]
    first = candidate(root / "first", trained, generation)
    second = candidate(root / "second", trained, generation)
    supervisor.publish(root / "first", generation, threading.Event())
    entered, release = threading.Event(), threading.Event()
    original = supervisor.library.save

    def held(source, artifact_root, reset_generation, label):
        """Hold the library operation after its source identity has been captured."""
        entered.set()
        assert release.wait(2)
        return original(source, artifact_root, reset_generation, label)

    monkeypatch.setattr(supervisor.library, "save", held)
    supervisor.submit("save_policy", {"label": "pinned"})
    assert entered.wait(2)
    supervisor.publish(root / "second", generation, threading.Event())
    release.set()
    finish(supervisor, "library")
    assert supervisor.record["eligible"]["model_id"] == second
    saved = next(iter(supervisor.catalog.values()))
    assert saved["model_id"] == first
    supervisor.close()
    restarted = Supervisor(LibraryRuntime(), config)
    assert saved["id"] in restarted.catalog
    restarted.close()


@pytest.mark.parametrize("failure", ["preparation", "reload", "reset"])
def test_failed_or_reset_load_does_not_replace_session(config, trained, failure):
    """Only exact successful activation admits a restored session; cancellation cleans up its worker."""
    runtime = LibraryRuntime()
    supervisor = Supervisor(runtime, config)
    path = supervisor.root / "original"
    previous = candidate(path, trained, supervisor.record["generation"])
    supervisor.publish(path, supervisor.record["generation"], threading.Event())
    entry = supervisor.library.save(
        path, supervisor.root, supervisor.record["generation"], "source"
    )
    workers = []
    entered = threading.Event()

    class FailingWorker(PreparedWorker):
        """Inject failures at the process/preparation boundary, leaving protocol identity real."""

        def __init__(self, run, launch):
            """Create a prepared session and inject the selected boundary failure."""
            super().__init__(run, launch)
            workers.append(self)
            if failure == "preparation":
                self.state, self.error = "error", "bad export"
            elif failure == "reset":
                self.prepared_path = None
            entered.set()

    supervisor.worker_factory = FailingWorker
    if failure == "reload":
        original_reload = runtime.begin_reload

        def reject(model, issued=None):
            """Reject the exact requested load while retaining the previous inference policy."""
            original_reload(model, issued)
            runtime.reload_ticket["outcome"] = "error"
            runtime.observed["active_id"] = previous

        runtime.begin_reload = reject
    supervisor.submit("set_policy", {"id": entry["id"]})
    assert entered.wait(2)
    if failure == "reset":
        supervisor.submit("reset", {"request": True})
    events = finish(supervisor)
    assert supervisor.worker is None
    assert workers[0].closed
    if failure == "reset":
        assert supervisor.record["eligible"] is None
        assert ("reset", "Ok", "") in events
    else:
        assert supervisor.record["eligible"]["model_id"] == previous
        assert any(
            name == "set_policy" and state == "Alert" for name, state, _ in events
        )
    assert entry["id"] in supervisor.library.catalog()
    supervisor.close()


def test_active_reference_changes_within_gradient_iteration(trained, inputs):
    """A callable readback updates anchors before the next gradient, without waiting an outer iteration."""
    active = "0"
    candidate_policy = deepcopy(trained.policy_copy)
    with torch.no_grad():
        next(candidate_policy.parameters()).add_(0.1)
    trained.policy_update.stage_publication(candidate_policy, "beef")
    checks = 0

    def checkpoint():
        """Acknowledge activation only after this iteration has already entered its update loop."""
        nonlocal checks, active
        checks += 1
        if checks == 2:
            active = "beef"
        if checks > 2:
            assert trained.policy_update.reference_id == "000000000000beef"

    trained.step(
        inputs[1], inputs[2], active_model_id=lambda: active, checkpoint=checkpoint
    )
    assert checks > 2
    assert trained.policy_update.reference_id == "000000000000beef"


def test_restored_shapes_missing_and_nonfinite_state_fail(config, trained):
    """Strict model loading rejects malformed tensors before any runtime activation."""
    path = Path(config.artifact_root) / "candidate"
    candidate(path, trained, "test")
    snapshot = path / "policy.trt.training.npz"
    with np.load(snapshot, allow_pickle=False) as stored:
        original = {key: stored[key].copy() for key in stored.files}
    key = next(
        key
        for key in original
        if key.startswith("policy_state__") and original[key].ndim
    )
    for replacement in (
        np.zeros((1, 2, 3, 4, 5)),
        np.full_like(original[key], np.nan),
        None,
    ):
        arrays = dict(original)
        if replacement is None:
            arrays.pop(key)
        else:
            arrays[key] = replacement
        np.savez(snapshot, **arrays)
        with pytest.raises((ValueError, KeyError)):
            TrainingSession(trained.config).restore(snapshot)


def test_reverting_paused_target_waits_for_new_acknowledgement(config):
    """Returning to the previous value cannot acknowledge an older in-flight configuration ticket."""
    supervisor = Supervisor(FakeRuntime(), config)
    supervisor.applied = deepcopy(supervisor.desired)
    supervisor.worker = SimpleNamespace(
        config_ticket=supervisor.config_ticket, stop=lambda: None
    )
    previous = supervisor.desired["training"]["policy_learning_rate"]
    supervisor.set_parameter("policy_learning_rate", previous * 2)
    in_flight = supervisor.config_ticket
    supervisor.set_parameter("policy_learning_rate", previous)
    assert supervisor.parameters()["policy_learning_rate"]["state"] == "Busy"
    supervisor.worker.config_ticket = in_flight
    assert supervisor.parameters()["policy_learning_rate"]["state"] == "Busy"
    supervisor.worker.config_ticket = supervisor.config_ticket
    assert supervisor.parameters()["policy_learning_rate"]["state"] == "Ok"
    supervisor.close()


def test_interrupted_activated_load_requires_explicit_initialization(
    config, trained, monkeypatch
):
    """Cancellation after activation cannot resume old networks; a fresh Set policy clears the barrier."""
    supervisor = Supervisor(LibraryRuntime(), config, worker_factory=PreparedWorker)
    path = supervisor.root / "original"
    generation = supervisor.record["generation"]
    candidate(path, trained, generation)
    saved = supervisor.library.save(path, supervisor.root, generation, "source")
    original = supervisor.publish

    def interrupted(directory, reset_generation, cancel, issued=None):
        """Inject cancellation after exact acknowledgement but before session admission."""
        identity = original(directory, reset_generation, cancel, issued)
        cancel.set()
        return identity

    monkeypatch.setattr(supervisor, "publish", interrupted)
    with pytest.raises(InterruptedError):
        supervisor.load_policy(saved["id"], threading.Event(), generation)
    assert supervisor.worker is None
    assert supervisor.status["initialization_required"] == "yes"
    with pytest.raises(RuntimeError, match="Set policy was interrupted"):
        supervisor.submit("learning", {"run": True})
    monkeypatch.setattr(supervisor, "publish", original)
    supervisor.load_policy(saved["id"], threading.Event(), generation)
    assert supervisor.status["initialization_required"] == "no"
    assert supervisor.worker.state == "paused"
    supervisor.close()


def test_hot_gui_parameters_preserve_optimizer_and_replay(trained):
    """All three controls change the intended settings while retaining numerical session state."""
    policy_optimizer, dynamics_optimizer = (
        trained.policy_optimizer,
        trained.dynamics_optimizer,
    )
    replay, normalization = trained.buffer_snapshot, trained.normalization_metadata
    dynamics_rate = dynamics_optimizer.param_groups[0]["lr"]
    settings = trained.config
    settings["RL"]["loss_function_penalty"] = 0.04
    settings["policy_update"]["weight"] = 0.7
    settings["training"]["policy_learning_rate"] = 2e-6
    apply_hot(trained, settings)
    assert trained.config["RL"]["loss_function_penalty"] == 0.04
    torch.testing.assert_close(trained.policy_update.weight, torch.tensor([0.7]))
    assert trained.policy_optimizer is policy_optimizer and policy_optimizer.state
    assert trained.dynamics_optimizer is dynamics_optimizer and dynamics_optimizer.state
    assert trained.policy_optimizer.param_groups[0]["lr"] == 2e-6
    assert trained.dynamics_optimizer.param_groups[0]["lr"] == dynamics_rate
    assert (
        trained.buffer_snapshot is replay
        and trained.normalization_metadata is normalization
    )


def test_catalog_skips_incomplete_manifests(tmp_path):
    """A partial externally copied entry cannot crash discovery or expose a load option."""
    library = PolicyLibrary(tmp_path / "saved_engines")
    directory = library.root / "incomplete"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(dict(format_version=1, id="incomplete"))
    )
    assert library.catalog() == {}


@pytest.mark.parametrize(
    "payload, expected",
    [(b"", "cbf29ce484222325"), (b"hello", "a430d84680aabd0b")],
)
def test_engine_id_keeps_fnv1a_identity(tmp_path, payload, expected):
    """Keep fixed FNV-1a vectors independent of the shared writer/reader helper."""
    path = tmp_path / "policy.trt"
    path.write_bytes(payload)
    assert engine_id(path) == expected
