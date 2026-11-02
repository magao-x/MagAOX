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
        for key in ("runtime", "pending", "reset_pending"):
            self.record.setdefault(key, None)
        if self.record["runtime"] is None and (
            self.record["eligible"]
            or self.record["worker_started"]
            or self.record["pending"]
        ):
            # Legacy records cannot establish whether the runtime was externally reset while absent.
            self.record.update(
                generation=uuid.uuid4().hex,
                eligible=None,
                pending=None,
                reset_pending={"ticket": None, "external": False},
            )
            atomic_json(self.record_path, self.record)
        self.last_control = None
        self.foreign_model = ""
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
                values={},
            )
            for name in ("reconstruction", "training")
        }
        self.status = dict(
            heartbeat="",
            collection_frames="0",
            collection_target="0",
            reconstructor="unknown",
            reconstruction_phase="idle",
            controller="unknown",
            learning=(
                "resetting"
                if self.record["reset_pending"]
                else "error" if self.record["worker_started"] else "idle"
            ),
            training_phase="reset_pending" if self.record["reset_pending"] else "idle",
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
        self.reconcile_runtime()

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
            if domain == "training" and command != "reset":
                if self.record["reset_pending"]:
                    raise RuntimeError(
                        "Runtime Reset is still pending; retry Reset or wait for frames/readback"
                    )
                observed = self.runtime.runtime_state()
                if self.record["runtime"] != {
                    "instance": observed["instance"],
                    "reset": observed["reset_request"],
                }:
                    raise RuntimeError(
                        "Runtime identity/reset changed; waiting for reconciliation"
                    )
                if command == "learning" and values.get("run") and self.foreign_model:
                    raise RuntimeError(
                        "External policy is not part of this managed session; restore an eligible policy or Reset"
                    )
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
                    generation=uuid.uuid4().hex,
                    eligible=None,
                    worker_started=False,
                    runtime=self.record["runtime"],
                    pending=None,
                    reset_pending={
                        "ticket": values.get("external"),
                        "external": bool(values.get("external")),
                    },
                )
                self.foreign_model = ""
                self.status["learning"] = "resetting"
                atomic_json(self.record_path, self.record)
                self.status["retained_policy"] = ""
            cancel = threading.Event()
            operation.update(cancel=cancel, command=command, values=dict(values))
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
        """Stop numerical work and settle every already-issued load before acknowledging pause."""
        worker = self.worker
        if worker is not None:
            if worker.state == "error":
                worker.stop()
            else:
                worker.send({"run": False})
        self.settle_publication(cancel)
        if worker is None or worker.closed:
            return

        def paused():
            """A failed worker must exit before Disable/recollection can proceed."""
            if worker.state == "error":
                worker.stop()
                return True
            return worker.state == "paused"

        self.runtime.wait(paused, cancel, "learning pause", self.config.worker_timeout)

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
            try:
                if not values.get("external"):
                    ticket = (
                        r.request_reset()
                    )  # Fallback and invalidation precede any blocking cleanup.
                    with self.lock:
                        self.record["reset_pending"]["ticket"] = ticket
                        atomic_json(self.record_path, self.record)
            finally:
                if self.worker:
                    self.worker.stop()
                    self.worker = None
                self.applied = None
                self.pending_reset = ""
            pending = self.record["reset_pending"]
            r.wait(
                lambda: r.reset_complete(pending["ticket"], pending["external"]),
                cancel,
                "PO4AO reset",
            )
            self.complete_reset()
        elif command == "po4ao_enable":
            if not values["toggle"]:
                r.request_control(False)
                self.pause(cancel)
                r.wait(lambda: r.control_complete(False), cancel, "PO4AO control")
            else:
                self.settle_publication(cancel)
                eligible = self.record["eligible"]
                if eligible:
                    self.publish(eligible["path"], generation, cancel)
                r.control(True, cancel)
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
            issued = False

            def accept(model_id):
                """Register the exact policy reference as soon as its load is issued."""
                nonlocal issued
                issued = True
                worker.send(
                    {"candidate_path": values["path"], "candidate_id": model_id}
                )

            try:
                if cancel.is_set():
                    raise InterruptedError("Publication cancelled")
                if worker is None or worker.state not in (
                    "running",
                    "waiting",
                    "completed",
                ):
                    raise RuntimeError("Candidate is no longer publishable")
                self.publish(values["path"], generation, cancel, issued=accept)
            finally:
                if not issued and worker is not None and not worker.closed:
                    worker.send({"candidate_path": values["path"], "candidate_id": ""})
        else:
            raise ValueError("Unknown command " + command)

    def publish(self, directory, generation, cancel, issued=None):
        """Commit a verified bundle; retain unresolved loads separately from eligible policies."""
        from .artifacts import verify_candidate

        candidate, model_id = verify_candidate(directory, self.root, generation)
        with self.lock:
            if (
                generation != self.record["generation"]
                or cancel.is_set()
                or self.record["reset_pending"]
            ):
                raise InterruptedError("Candidate predates Reset")
            current = self.runtime.runtime_state()
            if self.record["runtime"] != {
                "instance": current["instance"],
                "reset": current["reset_request"],
            }:
                raise InterruptedError("Runtime reset changed before publication")
            link = self.root / ("current-" + uuid.uuid4().hex)
            link.symlink_to(candidate, target_is_directory=True)
            os.replace(link, self.root / "current")
            pending = {
                "path": str(candidate),
                "model_id": model_id,
                "generation": generation,
            }
            self.record["pending"] = pending
            atomic_json(self.record_path, self.record)
            if issued:
                issued(model_id)
            self.runtime.begin_reload(model_id)
        outcome = self.settle_publication(cancel)
        if outcome != "loaded":
            raise RuntimeError("PO4AO reload failed or was invalidated")
        return model_id

    def finish_publication(self, outcome):
        """Persist a settled result once and report it to the worker's exact candidate wait."""
        with self.lock:
            pending = self.record["pending"]
            if pending is None:
                return
            if (
                outcome == "loaded"
                and pending["generation"] == self.record["generation"]
            ):
                self.record["eligible"] = {
                    "path": pending["path"],
                    "model_id": pending["model_id"],
                }
                self.status["retained_policy"] = pending["model_id"]
            self.record["pending"] = None
            atomic_json(self.record_path, self.record)
            if self.worker:
                self.worker.send(
                    {
                        "candidate_outcome_path": pending["path"],
                        "candidate_outcome": outcome,
                    }
                )

    def settle_publication(self, cancel):
        """Retain timed-out/cancelled loads until they complete or a runtime reset invalidates them."""
        if self.record["pending"] is None:
            return (
                self.runtime.reload_outcome()
                if self.runtime.reload_ticket is not None
                else None
            )
        if self.runtime.reload_ticket is None:
            raise RuntimeError(
                "A previous reload was unresolved at restart; Reset is required"
            )
        outcome = self.runtime.finish_reload(cancel)
        self.finish_publication(outcome)
        return outcome

    def complete_reset(self):
        """Clear the persistent barrier only after worker exit and actual runtime completion."""
        with self.lock:
            observed = self.runtime.runtime_state()
            self.record["runtime"] = {
                "instance": observed["instance"],
                "reset": observed["reset_request"],
            }
            self.last_control = (observed["instance"], observed["control_request"])
            self.record["reset_pending"] = None
            self.record["worker_started"] = False
            atomic_json(self.record_path, self.record)
            self.status["learning"] = "idle"
            self.status["training_phase"] = "idle"

    def reconcile_runtime(self):
        """Invalidate external resets/restarts and pause this worker on external disable/unknown policy."""
        try:
            observed = self.runtime.runtime_state()
        except (ConnectionError, KeyError):
            return
        epoch = {"instance": observed["instance"], "reset": observed["reset_request"]}
        with self.lock:
            previous = self.record["runtime"]
            changed = previous != epoch
            pending = self.record["reset_pending"]
            if changed and previous is not None and pending is None:
                self.submit(
                    "reset",
                    {
                        "request": True,
                        "external": {
                            "instance": observed["instance"],
                            "after": int(observed["reset_request"]) - 1,
                        },
                    },
                )
                pending = self.record["reset_pending"]
            if (
                pending
                and pending["ticket"] is None
                and changed
                and previous is not None
            ):
                pending["ticket"] = {
                    "instance": observed["instance"],
                    "after": int(observed["reset_request"]) - 1,
                }
            if changed:
                self.record["runtime"] = epoch
                atomic_json(self.record_path, self.record)
            control = (observed["instance"], observed["control_request"])
            operation = self.domains["training"]
            active_worker = self.worker is not None and self.worker.state in (
                "starting",
                "running",
                "waiting",
            )
            if (
                self.last_control is not None
                and control != self.last_control
                and observed["control_target"] == "off"
                and not self.record["reset_pending"]
                and not (
                    operation["command"] == "po4ao_enable"
                    and operation["values"].get("toggle") is False
                )
                and (active_worker or self.record["pending"])
            ):
                self.submit("po4ao_enable", {"toggle": False})
            self.last_control = control
            known = {"0000000000000000"}
            for record in (self.record["eligible"], self.record["pending"]):
                if record:
                    known.add(record["model_id"])
            self.foreign_model = (
                observed["active_id"]
                if observed["active_id"] not in known and self.worker
                else ""
            )
            if self.foreign_model and not self.record["reset_pending"]:
                self.worker.send({"run": False})
                operation["cancel"].set()
                self.status["error"] = (
                    "External policy is outside the managed session; restore an eligible policy or Reset"
                )
            pending = self.record["reset_pending"]
            if (
                pending
                and pending["ticket"]
                and self.worker is None
                and operation["future"] is None
            ):
                if self.runtime.reset_complete(pending["ticket"], pending["external"]):
                    self.complete_reset()
                    self.events.append(("reset", "Ok", ""))

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
            self.status["collection_frames"] = str(
                r.get(r.po4ao, "collection_status", "frames")
            )
            self.status["collection_target"] = str(
                r.get(r.po4ao, "collection_status", "target_frames")
            )
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
        self.reconcile_runtime()
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
                    if self.foreign_model or self.record["reset_pending"]:
                        message["collection"] = "unknown"
                        message.pop("active_id", None)
                    else:
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
        if self.record["pending"] and self.runtime.reload_ticket is not None:
            try:
                outcome = self.runtime.reload_outcome()
                if outcome is not None:
                    self.finish_publication(outcome)
            except (ConnectionError, KeyError):
                pass
        with self.lock:
            for name, operation in self.domains.items():
                future = operation["future"]
                if future is None or not future.done():
                    continue
                error = future.exception()
                self.readback()
                if operation["command"] == "reset":
                    self.status["learning"] = (
                        "resetting" if self.record["reset_pending"] else "idle"
                    )
                self.status[name + "_phase"] = (
                    "reset_pending"
                    if self.record["reset_pending"] and name == "training"
                    else "error" if error else "idle"
                )
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
