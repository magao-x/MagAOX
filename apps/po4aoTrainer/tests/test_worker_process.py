## \file test_worker_process.py
# \brief Real managed worker/pipe lifecycle with synthetic CPU numerical boundaries.
# \ingroup po4aoTrainer_unit_test
"""Exercise production worker control without instrument streams, GPU allocation or model publication."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

from xapp.po4aoTrainer.process import Worker
from xapp.po4aoTrainer.configuration import revision

BOOTSTRAP = r"""
import sys, types, numpy as np
from pathlib import Path
import xapp.po4aoTrainer.training as training
import xapp.po4aoTrainer.po4ao_runtime as runtime
original=training.TrainingSession
training.TrainingSession=lambda config, device: original(config, device="cpu")
runtime.configure_compute_resources=lambda *args:None
class Image:
    def open(self,name):
        self.name=name
        return 0
    @property
    def md(self):
        counter = int(Path(sys.argv[1]).parent.joinpath("warmup-counter").read_text()) if "warm" in self.name else 1
        return types.SimpleNamespace(cnt0=counter, write=False)
sys.modules["ImageStreamIOWrap"]=types.SimpleNamespace(Image=Image)
rng=np.random.default_rng(42)
data=rng.normal(size=(6,96,2)).astype(np.float32)
def snapshot(image,metadata,**kwargs):
    with Path(sys.argv[1]).joinpath("copies.log").open("a") as output:
        output.write(image.name + "\n")
    return runtime.ReplaySnapshot(data, image.md.cnt0,96,3,"0000000000000000")
runtime.stable_replay_snapshot=snapshot
from xapp.po4aoTrainer.worker import run
run(sys.argv[1])
"""


def test_owned_worker_pauses_resumes_recollects_and_exits_on_eof(tmp_path, monkeypatch):
    """Pause preserves PID/session, hot config is acknowledged, and EOF stops the owned worker."""
    config = json.loads(
        (Path(__file__).parent / "fixtures/session_config.json").read_text()
    )
    config["compute"] = dict(inference_device=0, training_device=1, build_device=1)
    config["training"].update(
        training_iterations=10000,
        first_model_publish_iteration=20000,
        diagnostics_interval=50,
    )
    original = subprocess.Popen

    def launch(command, **kwargs):
        """Replace only GPU/shared-memory boundaries around the production worker module."""
        return original([sys.executable, "-c", BOOTSTRAP, command[-1]], **kwargs)

    monkeypatch.setattr("xapp.po4aoTrainer.process.subprocess.Popen", launch)
    worker = Worker(
        tmp_path / "run",
        dict(
            config=config,
            generation="test",
            streams=["replay", "meta", "warmup", "warmmeta"],
            heartbeat_timeout=5,
            build_timeout=5,
            trtexec="unused",
        ),
    )
    (tmp_path / "warmup-counter").write_text("1")
    pid = worker.process.pid

    def wait(predicate, message=None):
        """Keep the real heartbeat flowing while pumping protocol acknowledgements."""
        end = time.monotonic() + 20
        while not predicate():
            worker.send(
                dict(
                    active_id="0" * 16,
                    collection="ready",
                    generation="1",
                    **(message or {})
                )
            )
            worker.events()
            assert worker.process.poll() is None, worker.run.joinpath(
                "worker.log"
            ).read_text()
            assert time.monotonic() < end, worker.run.joinpath("worker.log").read_text()
            time.sleep(0.03)

    try:
        wait(lambda: worker.iteration > 1)
        worker.send({"run": False})
        wait(lambda: worker.state == "paused")
        iteration = worker.iteration
        time.sleep(0.1)
        worker.events()
        assert worker.iteration == iteration
        assert worker.process.pid == pid
        assert (worker.run / "copies.log").read_text().splitlines() == [
            "replay",
            "warmup",
        ]
        config["training"]["policy_learning_rate"] *= 2
        changed = tmp_path / "hot.json"
        changed.write_text(json.dumps(config))
        worker.send({"config_path": str(changed), "run": True})
        wait(
            lambda: worker.revision == revision(config) and worker.iteration > iteration
        )
        worker.send({"run": False})
        wait(lambda: worker.state == "paused")
        iteration = worker.iteration
        # Resume after recollection unwinds stale replay without losing Adam/session state.
        (tmp_path / "warmup-counter").write_text("2")
        worker.send({"generation": "2", "run": True})
        end = time.monotonic() + 10
        while worker.iteration <= iteration:
            worker.send(
                {"active_id": "0" * 16, "collection": "ready", "generation": "2"}
            )
            worker.events()
            assert time.monotonic() < end, worker.run.joinpath("worker.log").read_text()
            time.sleep(0.03)
        assert worker.process.pid == pid
        worker.process.stdin.close()
        worker.process.wait(timeout=10)
        assert worker.process.returncode == 1  # SystemExit on control-channel loss.
    finally:
        worker.stop()


def test_reset_stops_owned_child_group_and_preserves_unrelated_process(
    tmp_path, monkeypatch
):
    """Reset kills a worker's child, never an unrelated process or notebook kernel."""
    import psutil

    original = subprocess.Popen
    bootstrap = """import subprocess,sys,time,json
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
print(json.dumps(dict(event='build',pid=child.pid)),flush=True)
time.sleep(60)
"""

    def launch(command, **kwargs):
        """Place an actual child in the production worker's owned process group."""
        return original([sys.executable, "-c", bootstrap], **kwargs)

    unrelated = original([sys.executable, "-c", "import time;time.sleep(60)"])
    monkeypatch.setattr("xapp.po4aoTrainer.process.subprocess.Popen", launch)
    worker = Worker(tmp_path / "owned", {})
    try:
        end = time.monotonic() + 5
        while not worker.children:
            worker.events()
            assert time.monotonic() < end
            time.sleep(0.02)
        child = worker.children[0]
        worker.stop(timeout=1)
        assert worker.process.poll() is not None
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
    finally:
        worker.stop(timeout=1)
        unrelated.terminate()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("outcome", [None, "loaded", "failed"])
def test_export_pause_and_publication_outcome(tmp_path, monkeypatch, outcome):
    """Exercise real worker cancellation, delayed issuance and persisted final publication outcomes."""
    import psutil

    config = json.loads(
        (Path(__file__).parent / "fixtures/session_config.json").read_text()
    )
    config["compute"] = dict(inference_device=0, training_device=1, build_device=1)
    config["training"].update(training_iterations=1000, diagnostics_interval=1)
    export_boundary = """
import xapp.po4aoTrainer.worker as managed
import xapp.po4aoTrainer.observer as observer
actual_save=observer.save_diagnostics
def save(*args, **kwargs):
    result=actual_save(*args, **kwargs)
    with Path(sys.argv[1]).joinpath("snapshots.log").open("a") as output:
        output.write(str(result) + "\\n")
    return result
observer.save_diagnostics=save
actual_popen=managed.subprocess.Popen
def fake_export(command, **kwargs):
    return actual_popen([sys.executable, '-c', 'import time;time.sleep(60)'], **kwargs)
managed.subprocess.Popen=fake_export
actual_session=training.TrainingSession
def ready_session(config, device):
    session=actual_session(config, device)
    actual_step=session.step
    def step(*args, **kwargs):
        result=actual_step(*args, **kwargs)
        if result is not None and result['record'] is not None:
            result['record']['publication_status']='ready'
        return result
    session.step=step
    return session
training.TrainingSession=ready_session
managed.run(sys.argv[1])
"""
    bootstrap = BOOTSTRAP.replace(
        "from xapp.po4aoTrainer.worker import run\nrun(sys.argv[1])", export_boundary
    )
    if outcome is not None:
        bootstrap = bootstrap.replace("import time;time.sleep(60)", "pass")
    original = subprocess.Popen

    def launch(command, **kwargs):
        """Use production control/build sequencing with only numerical/export boundaries replaced."""
        return original([sys.executable, "-c", bootstrap, command[-1]], **kwargs)

    monkeypatch.setattr("xapp.po4aoTrainer.process.subprocess.Popen", launch)
    (tmp_path / "warmup-counter").write_text("1")
    worker = Worker(
        tmp_path / "run",
        dict(
            config=config,
            generation="test",
            streams=["replay", "meta", "warmup", "warmmeta"],
            heartbeat_timeout=10,
            build_timeout=30,
            trtexec="unused",
        ),
    )
    try:
        deadline = time.monotonic() + 20
        candidates = []
        while not worker.children and not (outcome is not None and candidates):
            worker.send(dict(collection="ready", generation="1"))
            candidates.extend(
                event for event in worker.events() if event["event"] == "candidate"
            )
            assert worker.process.poll() is None, (
                worker.run / "worker.log"
            ).read_text()
            assert time.monotonic() < deadline
            time.sleep(0.03)
        build = worker.children[0] if worker.children else None
        if outcome is not None:
            while not candidates:
                worker.send(dict(collection="ready", generation="1"))
                candidates.extend(
                    event for event in worker.events() if event["event"] == "candidate"
                )
                assert not worker.error, (worker.run / "worker.log").read_text()
                assert time.monotonic() < deadline
                time.sleep(0.03)
            path = candidates[0]["path"]
            worker.send({"candidate_path": path, "candidate_id": "a" * 16})
        worker.send({"run": False})
        while worker.state != "paused":
            worker.events()
            assert not worker.error, (worker.run / "worker.log").read_text()
            assert time.monotonic() < deadline
            time.sleep(0.03)
        assert worker.process.poll() is None
        assert (
            build is None
            or not build.is_running()
            or build.status() == psutil.STATUS_ZOMBIE
        )
        if outcome is not None:
            worker.send({"candidate_outcome_path": path, "candidate_outcome": outcome})
            records = []
            while not any(
                record.get("publication_outcome") == outcome for record in records
            ):
                records = [
                    json.loads(line)
                    for line in (worker.run / "metrics.jsonl").read_text().splitlines()
                ]
                worker.events()
                assert not worker.error, (worker.run / "worker.log").read_text()
                assert time.monotonic() < deadline
                time.sleep(0.03)
            from xapp.po4aoTrainer.observer import load_diagnostics

            matching = next(
                record
                for record in records
                if record.get("publication_outcome") == outcome
            )
            # The outcome log is visible before compression of the final snapshot completes.
            while not (worker.run / "snapshots.log").exists():
                worker.events()
                assert not worker.error
                assert time.monotonic() < deadline
                time.sleep(0.03)
            snapshot = load_diagnostics(
                next((worker.run / "diagnostics").glob("*.npz"))
            )
            assert matching["published_model_id"] == "a" * 16
            assert snapshot.metadata["history"][-1] == matching
            assert len((worker.run / "snapshots.log").read_text().splitlines()) == 1
            assert matching["export_seconds"] >= 0
            assert matching["activation_seconds"] >= 0
            assert worker.process.poll() is None and worker.state == "paused"
    finally:
        worker.stop()


def test_export_restores_models_without_allocating_a_training_session(
    tmp_path, monkeypatch
):
    """Check exported weights, device and build arguments without a GPU or extra Adam/policy copy."""
    import torch
    from unittest.mock import Mock
    from xapp.po4aoTrainer.worker import export_candidate
    from xapp.po4aoTrainer import po4ao_runtime, po4ao_nonlinear_models, training

    config = json.loads(
        (Path(__file__).parent / "fixtures/session_config.json").read_text()
    )
    config["compute"] = dict(inference_device=0, training_device=1, build_device=1)
    metadata = dict(config=config, trtexec="test-trtexec")
    (tmp_path / "request.json").write_text(json.dumps(metadata))
    weights = dict(policy={"policy": 1}, dynamics={"dynamics": 2})
    dynamics, policy = Mock(), Mock()
    policy.eval.return_value = policy
    publisher = Mock()
    session = Mock(
        side_effect=AssertionError("Export must not create a TrainingSession")
    )
    loader = Mock(return_value=weights)
    zeros = Mock(return_value="sample")
    monkeypatch.setattr(training, "TrainingSession", session)
    monkeypatch.setattr(
        po4ao_nonlinear_models,
        "build_nonlinear_models",
        lambda config: (dynamics, policy),
    )
    monkeypatch.setattr(
        po4ao_runtime, "configure_compute_resources", lambda *args: None
    )
    monkeypatch.setattr(po4ao_runtime, "publish_tensorrt_engine", publisher)
    monkeypatch.setattr(torch, "load", loader)
    monkeypatch.setattr(torch, "zeros", zeros)
    export_candidate(tmp_path)
    session.assert_not_called()
    policy.load_state_dict.assert_called_once_with(weights["policy"])
    dynamics.load_state_dict.assert_called_once_with(weights["dynamics"])
    policy.to.assert_called_once_with(torch.device("cuda:1"))
    dynamics.to.assert_called_once_with(torch.device("cuda:1"))
    loader.assert_called_once_with(
        tmp_path / "weights.pt", weights_only=True, map_location=torch.device("cuda:1")
    )
    zeros.assert_called_once_with(
        (
            1,
            2 * config["MDP"]["n_history"] - 1,
            1,
            config["runtime"]["n_modes_controlled"],
        ),
        device=torch.device("cuda:1"),
    )
    assert publisher.call_args.args == (policy, "sample", tmp_path / "policy.trt")
    assert publisher.call_args.kwargs["dynamics"] is dynamics
    assert publisher.call_args.kwargs["build_device"] == 1
    assert publisher.call_args.kwargs["metadata"] == metadata
