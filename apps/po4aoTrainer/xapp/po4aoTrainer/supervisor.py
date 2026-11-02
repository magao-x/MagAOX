## \file supervisor.py
# \brief Two independent operation queues and one owned training session.
"""Compound operations live here; GUI disconnects do not own the worker lifecycle."""

from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .configuration import load_training, needs_reset, revision
from .process import Worker, atomic_json


class Supervisor:
    """Serialize reconstruction and training separately, with reset-generation publication eligibility."""

    def __init__(self, runtime, config, worker_factory=Worker):
        """Load persistent eligibility without automatically starting or reloading anything."""
        self.runtime, self.config, self.worker_factory = runtime, config, worker_factory
        self.lock = threading.RLock()
        self.root = Path(config.artifact_root)
        if not self.root.is_absolute() or not self.root.is_dir():
            raise ValueError(
                "Provision an absolute shared artifact_root before starting the app"
            )
        self.record_path = self.root / "session.json"
        self.record = (
            json.loads(self.record_path.read_text())
            if self.record_path.exists()
            else {
                "generation": uuid.uuid4().hex,
                "eligible": None,
                "worker_started": False,
            }
        )
        self.worker = None
        self.desired = self.applied = None
        self.config_error = ""
        self.pending_reset = ""
        self.linked = False
        self.events = []
        self.domains = {
            name: dict(
                pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix=name),
                future=None,
                cancel=threading.Event(),
                command=None,
            )
            for name in ("reconstruction", "training")
        }
        self.status = dict(
            reconstructor="unknown",
            reconstruction_phase="idle",
            controller="unknown",
            learning="error" if self.record["worker_started"] else "idle",
            training_phase="idle",
            collection="unknown",
            active_policy="",
            retained_policy=(self.record["eligible"] or {}).get("model_id", ""),
            modulation="Unknown",
            message="",
            error="",
        )
        self.poll_config()
        parser = ConfigParser(inline_comment_prefixes=("#", ";"))
        parser.read(config.runtime_config)
        p = parser["parameters"]
        self.streams = [
            p[k]
            for k in (
                "buffer_channel",
                "buffer_metadata_channel",
                "warmup_buffer_channel",
                "warmup_metadata_channel",
            )
        ]
        expected = self.root / "current" / "policy.trt"
        if Path(p["engineDirs"]) / p["engineName"] != expected:
            raise ValueError(
                f"Configure C++ engineDirs/engineName as {expected}; publication is one atomic directory link"
            )
        self.last_heartbeat = 0.0
        self.collection_generation = "0"

    def poll_config(self):
        """Retain the last good snapshot after an incomplete/invalid edit."""
        try:
            candidate = load_training(
                self.config.training_config, self.config.runtime_config
            )
            self.desired = candidate
            self.config_error = ""
            self.pending_reset = (
                revision(candidate)
                if self.applied and needs_reset(self.applied, candidate)
                else ""
            )
        except Exception as error:
            self.config_error = str(error)

    def submit(self, command, values):
        """Admit a request quickly; callbacks never wait for processes or device transitions."""
        domain = (
            "reconstruction"
            if command in ("nn_activate", "reconstructor", "link_modulation")
            else "training"
        )
        with self.lock:
            operation = self.domains[domain]
            busy = operation["future"] is not None
            if command == "reset" and operation["command"] == "reset":
                raise RuntimeError("Reset is already in progress")
            supersede = command == "reset" or (
                command == "po4ao_enable"
                and not values["toggle"]
                and operation["command"] != "reset"
            )
            if busy and not supersede:
                raise RuntimeError(domain + " operation is already in progress")
            if busy and supersede:
                operation["cancel"].set()
                self.events.extend(
                    []
                    if operation["command"] == "publish"
                    else [(operation["command"], "Alert", "Superseded by " + command)]
                )
            if command == "reset":
                if operation["command"] == "reset":
                    raise RuntimeError("Reset is already in progress")
                operation["cancel"].set()
                self.record = dict(
                    generation=uuid.uuid4().hex, eligible=None, worker_started=False
                )
                atomic_json(self.record_path, self.record)
                self.status["retained_policy"] = ""
            cancel = threading.Event()
            operation.update(cancel=cancel, command=command)
            self.status[domain + "_phase"] = {
                "reset": "resetting",
                "recollect": "recollecting",
                "po4ao_enable": (
                    "enabling" if values.get("toggle", False) else "disabling"
                ),
            }.get(command, command)
            generation = self.record["generation"]
            operation["future"] = operation["pool"].submit(
                self.execute, command, values, cancel, generation
            )

    def pause(self, cancel):
        """Wait for cooperative pause while retaining the process, models and optimizer state."""
        if self.worker is None:
            return
        if self.worker.state == "error":
            self.worker.stop()
            return
        self.worker.send({"run": False})

        def paused():
            """A failed worker must exit before Disable/recollection can proceed."""
            if self.worker.state == "error":
                self.worker.stop()
                return True
            return self.worker.state == "paused"

        self.runtime.wait(
            paused,
            cancel,
            "learning pause",
            self.config.worker_timeout,
        )

    def execute(self, command, values, cancel, generation):
        """Run a single admitted compound operation in its independent domain."""
        if cancel.is_set() and command != "publish":
            raise InterruptedError("Operation cancelled")
        r = self.runtime
        if command == "link_modulation":
            self.linked = values["toggle"]
        elif command == "nn_activate":
            r.switch("nn", cancel, linked=self.linked, model=values["model"])
        elif command == "reconstructor":
            r.switch(
                values["target"],
                cancel,
                linked=self.linked,
                expected=values["expected"],
            )
        elif command == "reset":
            if self.worker:
                self.worker.stop()
                self.worker = None
            with self.lock:
                self.record["worker_started"] = False
                atomic_json(self.record_path, self.record)
            self.applied = None
            self.pending_reset = ""
            self.status["learning"] = "idle"
            r.control(False, cancel)
            r.reset(cancel)
        elif command == "po4ao_enable":
            if not values["toggle"]:
                self.pause(cancel)
            else:
                eligible = self.record["eligible"]
                if eligible:
                    self.publish(eligible["path"], generation, cancel, remember=False)
            r.control(values["toggle"], cancel)
        elif command == "recollect":
            running = self.worker is not None and self.worker.state in (
                "starting",
                "running",
                "waiting",
            )
            self.pause(cancel)
            r.recollect(cancel, self.config.collection_timeout)
            if running:
                self.worker.send({"run": True})
        elif command == "learning":
            if not values["run"]:
                self.pause(cancel)
            else:
                self.poll_config()
                if self.config_error:
                    raise ValueError(self.config_error)
                if self.worker is None:
                    if self.record["worker_started"]:
                        raise RuntimeError(
                            "Previous worker session was lost; Reset is required"
                        )
                    run = self.root / ("run-" + uuid.uuid4().hex)
                    self.applied = deepcopy(self.desired)
                    with self.lock:
                        if generation != self.record["generation"] or cancel.is_set():
                            raise InterruptedError("Start superseded by Reset")
                        self.record["worker_started"] = True
                        atomic_json(self.record_path, self.record)
                    self.worker = self.worker_factory(
                        run,
                        dict(
                            config=self.applied,
                            generation=generation,
                            streams=self.streams,
                            trtexec=self.config.trtexec,
                            build_timeout=self.config.build_timeout,
                            heartbeat_timeout=self.config.heartbeat_timeout,
                            source_revision=self.config.source_revision,
                        ),
                    )
                else:
                    if self.worker.state == "error":
                        raise RuntimeError("Worker failed; Reset is required")
                    self.worker.send({"run": True})

                def started():
                    """Report a terminal worker failure immediately instead of waiting out startup."""
                    if self.worker.state == "error":
                        raise RuntimeError(self.worker.error)
                    return self.worker.state in ("running", "waiting", "completed")

                r.wait(
                    started,
                    cancel,
                    "learning start/resume",
                    self.config.worker_timeout,
                )
        elif command == "publish":
            worker = self.worker
            model_id = ""
            try:
                if cancel.is_set():
                    raise InterruptedError("Publication cancelled")
                if worker is None or worker.state not in (
                    "running",
                    "waiting",
                    "completed",
                ):
                    raise RuntimeError("Candidate is no longer publishable")
                model_id = self.publish(values["path"], generation, cancel)
            finally:
                # A cancelled/failed candidate must release the worker's publication wait too.
                if worker is not None and not worker.closed:
                    worker.send(
                        {"candidate_path": values["path"], "candidate_id": model_id}
                    )
        else:
            raise ValueError("Unknown command " + command)

    def publish(self, directory, generation, cancel, remember=True):
        """Verify a complete bundle and atomically switch one directory link before requesting reload."""
        from .artifacts import verify_candidate

        candidate, model_id = verify_candidate(directory, self.root, generation)
        with self.lock:
            if generation != self.record["generation"] or cancel.is_set():
                raise InterruptedError("Candidate predates Reset")
            link = self.root / ("current-" + uuid.uuid4().hex)
            link.symlink_to(candidate, target_is_directory=True)
            os.replace(link, self.root / "current")
            if remember:
                self.record["eligible"] = {"path": str(candidate), "model_id": model_id}
                atomic_json(self.record_path, self.record)
            self.status["retained_policy"] = model_id
        if cancel.is_set():
            raise InterruptedError("Publication cancelled")
        self.runtime.reload(model_id, cancel)
        return model_id

    def readback(self):
        """Refresh observed status independently of operation/button history."""
        r = self.runtime
        try:
            self.status["reconstructor"] = r.source()
        except Exception:
            self.status["reconstructor"] = "unknown"
        try:
            effective = r.get(r.po4ao, "model_status", "effective_controller")
            from purepyindi2.constants import SwitchState

            enabled = r.get(r.po4ao, "control_enable", "toggle") == SwitchState.ON
            self.status["controller"] = (
                ("po4ao" if effective == "po4ao" else "fallback")
                if enabled
                else ("integrator" if effective == "integrator" else "unknown")
            )
            self.status["active_policy"] = r.get(
                r.po4ao, "model_status", "effective_id"
            )
            state = r.get(r.po4ao, "collection_status", "state")
            self.status["collection"] = (
                "collecting"
                if state in ("pending", "collecting", "publishing")
                else state if state in ("ready", "interrupted") else "unknown"
            )
        except Exception:
            self.status.update(
                controller="unknown", active_policy="", collection="unknown"
            )
        try:
            self.status["modulation"] = str(r.get(r.modulation, "modState", "current"))
        except Exception:
            self.status["modulation"] = "Unknown"

    def tick(self):
        """Drain worker events and completed operations; drive heartbeat/config without blocking callbacks."""
        self.poll_config()
        self.readback()
        worker = self.worker
        if worker:
            try:
                worker_events = worker.events()
            except (OSError, ValueError, RuntimeError) as error:
                worker.state, worker.error = "error", str(error)
                worker_events = []
            for event in worker_events:
                if event["event"] == "candidate":
                    with self.lock:
                        operation = self.domains["training"]
                        if (
                            operation["future"] is None
                            and event["generation"] == self.record["generation"]
                        ):
                            cancel = threading.Event()
                            operation.update(cancel=cancel, command="publish")
                            operation["future"] = operation["pool"].submit(
                                self.execute,
                                "publish",
                                event,
                                cancel,
                                event["generation"],
                            )
                            self.status["training_phase"] = "publishing"
                        else:
                            worker.send(
                                {"candidate_path": event["path"], "candidate_id": ""}
                            )
            self.status["learning"] = worker.state
            if worker.error:
                self.status["error"] = worker.error
                worker.stop()
            if (
                time.monotonic() - self.last_heartbeat > 0.5
                and worker.process.poll() is None
            ):
                message = {
                    "active_id": self.status["active_policy"] or "0000000000000000",
                    "collection": self.status["collection"],
                    "generation": self.collection_generation,
                }
                try:
                    message["active_id"] = self.runtime.get(
                        self.runtime.po4ao, "model_status", "active_id"
                    )
                    self.collection_generation = self.runtime.get(
                        self.runtime.po4ao, "collection_status", "requested_generation"
                    )
                    message["generation"] = self.collection_generation
                except Exception:
                    message["collection"] = "unknown"
                if (
                    self.desired
                    and not self.pending_reset
                    and revision(self.desired) != worker.revision
                ):
                    path = worker.run / (revision(self.desired) + ".json")
                    if not path.exists():
                        atomic_json(path, self.desired)
                    message["config_path"] = str(path)
                try:
                    worker.send(message)
                    self.last_heartbeat = time.monotonic()
                except BlockingIOError:
                    pass
                except OSError as error:
                    worker.state, worker.error = "error", str(error)
                    self.status["learning"], self.status["error"] = "error", str(error)
                    worker.stop()
                if self.desired and worker.revision == revision(self.desired):
                    self.applied = deepcopy(self.desired)
        with self.lock:
            for name, operation in self.domains.items():
                future = operation["future"]
                if future is None or not future.done():
                    continue
                error = future.exception()
                self.readback()
                if operation["command"] == "reset" and self.worker is None:
                    self.status["learning"] = "idle"
                self.status[name + "_phase"] = "error" if error else "idle"
                self.status["error"] = str(error) if error else ""
                self.status["message"] = operation["command"] + (
                    " failed" if error else " complete"
                )
                if operation["command"] != "publish":
                    self.events.append(
                        (
                            operation["command"],
                            "Alert" if error else "Ok",
                            str(error) if error else "",
                        )
                    )
                elif error and worker:
                    worker.send({"run": False})
                operation.update(future=None, command=None)
            events, self.events = self.events, []
        return events

    def close(self):
        """Cancel app-owned operations and stop only the owned worker group."""
        for operation in self.domains.values():
            operation["cancel"].set()
        for operation in self.domains.values():
            operation["pool"].shutdown(wait=True, cancel_futures=True)
        if self.worker:
            self.worker.stop()
