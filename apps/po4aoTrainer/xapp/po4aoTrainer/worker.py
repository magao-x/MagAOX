## \file worker.py
# \brief Managed numerical session with cooperative pause, config boundaries and staged export.
"""The worker never publishes to the live engine path or sends instrument commands."""

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import uuid


def export_candidate(directory):
    """Build in a private immutable candidate directory using the original export helper."""
    import torch
    from .training import TrainingSession
    from .po4ao_runtime import publish_tensorrt_engine, configure_compute_resources

    directory = Path(directory)
    metadata = json.loads((directory / "request.json").read_text())
    config = metadata["config"]
    c = config["compute"]
    configure_compute_resources(
        c["inference_device"], c["training_device"], c["build_device"]
    )
    session = TrainingSession(config, device=f"cuda:{c['training_device']}")
    stored = torch.load(
        directory / "weights.pt", weights_only=True, map_location=session.device
    )
    session.policy.load_state_dict(stored["policy"])
    session.dynamics.load_state_dict(stored["dynamics"])
    sample = torch.zeros(
        (
            1,
            2 * config["MDP"]["n_history"] - 1,
            1,
            config["runtime"]["n_modes_controlled"],
        ),
        device=session.device,
    )
    publish_tensorrt_engine(
        session.policy.eval(),
        sample,
        directory / "policy.trt",
        build_device=c["build_device"],
        trtexec=metadata["trtexec"],
        timing_cache=str(directory.parent / "timing.cache"),
        log_path=str(directory / "build.log"),
        metadata=metadata,
        dynamics=session.dynamics,
    )


def run(directory):
    """Use one session and optimizer set until Reset, EOF, heartbeat loss or terminal failure."""
    directory = Path(directory)
    launch = json.loads((directory / "launch.json").read_text())
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def emit(event, **values):
        """Keep library logging out of the dedicated protocol stream."""
        protocol.write(json.dumps(dict(event=event, **values), allow_nan=False) + "\n")
        protocol.flush()

    control = {
        "run": True,
        "active_id": "0000000000000000",
        "collection": "unknown",
        "generation": "0",
    }
    last_heartbeat = time.monotonic()
    buffer = b""
    session = None
    last_state = None
    build = None
    build_identity = None

    def state(value):
        """Acknowledge only a state the worker has actually entered."""
        nonlocal last_state
        record = (
            value,
            0 if session is None else session.iteration,
            "" if session is None else session.config_revision,
        )
        if record != last_state:
            emit("state", state=record[0], iteration=record[1], revision=record[2])
            last_state = record

    def pump():
        """Read available control records and stop on loss of the supervisor."""
        nonlocal buffer, last_heartbeat
        while select.select([sys.stdin], [], [], 0)[0]:
            data = os.read(sys.stdin.fileno(), 65536)
            if not data:
                raise SystemExit("Supervisor pipe closed")
            buffer += data
            if len(buffer) > 1048576:
                raise RuntimeError("Control input exceeded bound")
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                command = json.loads(line)
                last_heartbeat = time.monotonic()
                control.update(command)
        if time.monotonic() - last_heartbeat > launch.get("heartbeat_timeout", 10):
            raise SystemExit("Supervisor heartbeat lost")

    class ReplayChanged(Exception):
        """Unwind an old-data iteration while retaining completed optimizer updates."""

    def checkpoint(training=False):
        """Pause between updates without unwinding the numerical iteration stack."""
        pump()
        if training and control["generation"] != generation:
            raise ReplayChanged()
        while not control["run"] or (training and control["collection"] != "ready"):
            state("paused" if not control["run"] else "waiting")
            time.sleep(0.05)
            pump()
            if training and control["generation"] != generation:
                raise ReplayChanged()
        state(
            "completed"
            if session is not None
            and session.iteration >= session._config["training"]["training_iterations"]
            else "running"
        )

    def stop_build():
        """Stop the owned export subtree on Pause; the worker/Adam state survives."""
        nonlocal build
        if build is None:
            return
        import psutil

        try:
            parent = build_identity
            if not parent.is_running():
                build.wait(timeout=3)
                build = None
                return
            parent.suspend()  # Prevent another child being spawned while enumerating/killing.
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            parent.kill()
            psutil.wait_procs(children + [parent], timeout=3)
        except psutil.NoSuchProcess:
            pass
        build.wait(timeout=3)
        build = None

    def terminate(signum, frame):
        """Allow normal reset to clean up an outstanding export child."""
        raise SystemExit("Worker terminated")

    signal.signal(signal.SIGTERM, terminate)
    try:
        import torch
        import ImageStreamIOWrap as shmio
        from .training import TrainingSession
        from .configuration import apply_hot
        from .po4ao_runtime import configure_compute_resources, stable_replay_snapshot
        from .observer import save_diagnostics

        config = launch["config"]
        c = config["compute"]
        configure_compute_resources(
            c["inference_device"], c["training_device"], c["build_device"]
        )
        session = TrainingSession(config, device=f"cuda:{c['training_device']}")
        streams = []
        for name in launch["streams"]:
            image = shmio.Image()
            if image.open(name) != 0:
                raise RuntimeError("Could not open replay stream " + name)
            streams.append(image)
        generation = None
        while True:
            checkpoint()
            if control.get("config_path"):
                candidate = json.loads(Path(control.pop("config_path")).read_text())
                apply_hot(session, candidate)
                state("running")
            if control["generation"] != generation:
                session.invalidate_replay()
                generation = control["generation"]
            if control["collection"] != "ready":
                state("waiting")
                time.sleep(0.1)
                continue
            before = generation
            try:
                replay = stable_replay_snapshot(*streams[:2], attempts=5)
                warmup = stable_replay_snapshot(*streams[2:], attempts=5)
            except TimeoutError:
                state("waiting")
                time.sleep(0.1)
                continue
            pump()
            if before != control["generation"] or control["collection"] != "ready":
                continue
            try:
                result = session.step(
                    replay,
                    warmup,
                    active_model_id=control["active_id"],
                    checkpoint=lambda: checkpoint(training=True),
                )
            except ReplayChanged:
                session.invalidate_replay()
                continue
            if result is None:
                state("completed" if session.state == "completed" else "waiting")
                time.sleep(0.1)
                continue
            record = result["record"]
            if record is None:
                continue
            record["source_revision"] = launch.get("source_revision", "unrecorded")
            with (directory / "metrics.jsonl").open("a") as metrics:
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
            save_diagnostics(session, directory / "diagnostics")
            if record["publication_status"] != "ready":
                continue
            candidate = directory / ("candidate-" + uuid.uuid4().hex)
            candidate.mkdir()
            metadata = dict(
                config=session.config,
                config_revision=session.config_revision,
                generation=launch["generation"],
                session_id=session.session_id,
                iteration=session.iteration,
                policy_architecture=session.config["NN_models"]["policy_architecture"],
                source_revision=launch.get("source_revision", "unrecorded"),
                trtexec=launch["trtexec"],
            )
            (candidate / "request.json").write_text(json.dumps(metadata))
            torch.save(
                {
                    "policy": session.policy_copy.state_dict(),
                    "dynamics": session.dynamics.state_dict(),
                },
                candidate / "weights.pt",
            )
            with (candidate / "export.log").open("wb") as log:
                build = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "xapp.po4aoTrainer.worker",
                        "--export",
                        str(candidate),
                    ],
                    stdout=log,
                    stderr=log,
                )
                import psutil

                build_identity = psutil.Process(build.pid)
                build_identity.create_time()
                emit("build", pid=build.pid)
                deadline = time.monotonic() + launch["build_timeout"]
                while build.poll() is None:
                    pump()
                    if (
                        not control["run"]
                        or control["generation"] != generation
                        or control["collection"] != "ready"
                    ):
                        stop_build()
                        break
                    if time.monotonic() > deadline:
                        stop_build()
                        raise TimeoutError("Engine export exceeded build_timeout")
                    time.sleep(0.1)
                if build is None:
                    continue
                code = build.wait()
                build = None
            if code != 0:
                raise RuntimeError(
                    "Engine export failed; see " + str(candidate / "export.log")
                )
            checkpoint()
            if control["generation"] != generation or control["collection"] != "ready":
                continue
            emit("candidate", path=str(candidate), generation=launch["generation"])
            # Continue pumping acknowledgements while paused; no optimizer/model changes occur here.
            while control.get("candidate_path") != str(candidate):
                pump()
                state("paused" if not control["run"] else "waiting")
                time.sleep(0.05)
            model_id = control.pop("candidate_id", "")
            if model_id:
                session.stage_publication(model_id)
                with (directory / "metrics.jsonl").open("a") as metrics:
                    metrics.write(json.dumps(record, allow_nan=False) + "\n")
                save_diagnostics(session, directory / "diagnostics")
                while control.get("candidate_outcome_path") != str(candidate):
                    pump()
                    state("paused" if not control["run"] else "waiting")
                    time.sleep(0.05)
                outcome = control.pop("candidate_outcome")
                control.pop("candidate_outcome_path", None)
                if outcome != "loaded":
                    session.policy_update.pending = None
                record["publication_status"] = (
                    "activated" if outcome == "loaded" else outcome
                )
                record["publication_outcome"] = outcome
            else:
                record["publication_status"] = "rejected before reload"
                record["publication_outcome"] = "rejected"
            # JSONL is append-only: later records for the same session/iteration replace the earlier status.
            with (directory / "metrics.jsonl").open("a") as metrics:
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
            save_diagnostics(session, directory / "diagnostics")
            control.pop("candidate_path", None)
    except Exception as error:
        emit("error", error=str(error))
        raise
    finally:
        stop_build()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--export":
        export_candidate(sys.argv[2])
    else:
        run(sys.argv[1])
