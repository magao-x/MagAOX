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

from .configuration import (
    PARAMETERS,
    load_training,
    needs_reset,
    parameter_value,
    revision,
    validate_training,
)
from .library import FILES, PolicyLibrary, verify_candidate
from .process import Worker, atomic_json


class Supervisor:
    """Serialize managed training with reset-generation publication eligibility."""

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
        self.publication = (
            None  # In-memory (persisted candidate, exact runtime ticket) binding.
        )
        self.last_control = None
        self.foreign_model = ""
        self.worker = None
        self.preparing_worker = None
        self.library = PolicyLibrary(
            getattr(config, "saved_engines", "") or self.root / "saved_engines"
        )
        self.catalog = self.library.catalog()
        self.file_config = None
        self.overrides = {}
        self.config_history = {}
        self.config_ticket = 0
        self.parameter_errors = {}
        self.desired = self.applied = None
        self.config_error = ""
        self.pending_reset = ""
        self.events = []
        self.domains = {
            name: dict(
                pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix=name),
                future=None,
                cancel=threading.Event(),
                command=None,
                values={},
            )
            for name in ("training", "library")
        }
        self.status = dict(
            heartbeat="",
            initialization_required=(
                "yes" if self.record.get("initialization_required") else "no"
            ),
            save_available="no",
            warmup_required=(
                "yes" if self.record.get("warmup_required") is not None else "no"
            ),
            delta_policy_mode="scalar",
            collection_frames="0",
            collection_target="0",
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
        """Merge session overrides; only valid edits to the same file field supersede them."""
        with self.lock:
            try:
                candidate = load_training(
                    self.config.training_config, self.config.runtime_config
                )
                if self.file_config is not None:
                    for name, (group, key) in PARAMETERS.items():
                        if candidate[group][key] != self.file_config[group][key]:
                            self.overrides.pop(name, None)
                            self.parameter_errors.pop(name, None)
                self.file_config = deepcopy(candidate)
                for name, value in self.overrides.items():
                    group, key = PARAMETERS[name]
                    candidate[group][key] = value
                self.set_desired(candidate)
                self.config_error = ""
            except Exception as error:
                self.config_error = str(error)

    def set_desired(self, candidate):
        """Keep the desired snapshot and each sent revision for accurate worker readback."""
        validate_training(candidate)
        identity = revision(candidate)
        if self.desired is None or revision(self.desired) != identity:
            self.config_ticket += 1
        self.desired = candidate
        self.config_history[identity] = deepcopy(candidate)
        self.pending_reset = (
            identity if self.applied and needs_reset(self.applied, candidate) else ""
        )
        self.status["delta_policy_mode"] = (
            "per_mode"
            if parameter_value(candidate, "delta_policy_regularization") is None
            else "scalar"
        )

    def set_parameter(self, name, value):
        """Merge one scalar override without recreating the numerical session or writing system config."""
        with self.lock:
            if name not in PARAMETERS or self.desired is None:
                raise ValueError("Training configuration is unavailable")
            if parameter_value(self.desired, name) is None:
                raise ValueError("Per-mode: configured in file")
            candidate = deepcopy(self.desired)
            group, key = PARAMETERS[name]
            candidate[group][key] = float(value)
            validate_training(candidate)
            self.overrides[name] = candidate[group][key]
            previous_error = self.parameter_errors.pop(name, None)
            if previous_error and self.status["error"] == previous_error:
                self.status["error"] = ""
            self.set_desired(candidate)

    def parameters(self):
        """Return current/target and state; a paused worker keeps edits pending until its acknowledgement."""
        with self.lock:
            values = {}
            current = self.applied if self.worker and self.applied else self.desired
            for name in PARAMETERS:
                actual = parameter_value(current, name) if current else None
                target = parameter_value(self.desired, name) if self.desired else None
                values[name] = dict(
                    current=actual if actual is not None else 0,
                    target=target if target is not None else 0,
                    state=(
                        "Alert"
                        if name in self.parameter_errors or self.desired is None
                        else (
                            "Busy"
                            if actual != target
                            or (
                                self.worker is not None
                                and getattr(self.worker, "config_ticket", None)
                                != self.config_ticket
                            )
                            else "Ok"
                        )
                    ),
                )
            return values

    def save_source(self):
        """Pin the complete acknowledged pair, including the retained pair while disabled."""
        eligible = self.record["eligible"]
        if eligible is None or self.record["reset_pending"]:
            raise RuntimeError("No acknowledged policy/dynamics pair is available")
        if not all((Path(eligible["path"]) / name).is_file() for name in FILES):
            raise RuntimeError("The acknowledged publication is missing model files")
        observed = self.runtime.runtime_state()
        if observed["active_id"] != eligible["model_id"]:
            raise RuntimeError(
                "The runtime policy is not the acknowledged managed pair"
            )
        return dict(path=eligible["path"], generation=self.record["generation"])

    def submit(self, command, values):
        """Admit a request quickly; callbacks never wait for processes or device transitions."""
        if command not in (
            "reset",
            "po4ao_enable",
            "recollect",
            "learning",
            "publish",
            "save_policy",
            "set_policy",
        ):
            raise ValueError("Unknown PO4AO training command: " + command)
        domain = "library" if command == "save_policy" else "training"
        with self.lock:
            if command == "save_policy":
                values = dict(values, **self.save_source())
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
            if (
                command == "learning"
                and values.get("run")
                and self.record.get("initialization_required")
            ):
                raise RuntimeError(
                    "Set policy was interrupted after reload issuance; Set policy again or Reset before learning"
                )
            busy = operation["future"] is not None
            if command == "reset" and operation["command"] == "reset":
                raise RuntimeError("Reset is already in progress")
            if (
                command == "learning"
                and values.get("run")
                and self.record.get("warmup_required") is not None
            ):
                raise RuntimeError(
                    "Fresh warmup required: enable PO4AO, recollect warmup, then Start learning"
                )
            supersede = command == "reset" or (
                command == "po4ao_enable"
                and not values["toggle"]
                and operation["command"] != "reset"
            )
            if busy and not supersede:
                raise RuntimeError(domain + " operation is already in progress")
            if busy and supersede:
                operation["cancel"].set()
                if operation["command"] != "publish":
                    self.events.append(
                        (operation["command"], "Alert", "Superseded by " + command)
                    )
            if command == "reset":
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
                self.publication = None
                self.foreign_model = ""
                self.status["learning"] = "resetting"
                atomic_json(self.record_path, self.record)
                self.status.update(
                    retained_policy="",
                    warmup_required="no",
                    initialization_required="no",
                )
            cancel = threading.Event()
            operation.update(cancel=cancel, command=command, values=dict(values))
            if domain == "training":
                self.status["training_phase"] = {
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
        if command == "reset":
            try:
                if not values.get("external"):
                    ticket = (
                        r.request_reset()
                    )  # Fallback and invalidation precede any blocking cleanup.
                    with self.lock:
                        self.record["reset_pending"]["ticket"] = ticket
                        atomic_json(self.record_path, self.record)
            finally:
                if self.preparing_worker:
                    self.preparing_worker.stop()
                    self.preparing_worker = None
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
                    with self.lock:
                        self.applied = deepcopy(self.desired)
                        config_ticket = self.config_ticket
                        if generation != self.record["generation"] or cancel.is_set():
                            raise InterruptedError("Start superseded by Reset")
                        self.record["worker_started"] = True
                        atomic_json(self.record_path, self.record)
                    self.worker = self.worker_factory(
                        run,
                        dict(
                            config=self.applied,
                            config_ticket=config_ticket,
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
        elif command == "save_policy":
            saved = self.library.save(
                values["path"], self.root, values["generation"], values["label"]
            )
            with self.lock:
                self.catalog = self.library.catalog()
            return saved["id"]
        elif command == "set_policy":
            self.load_policy(values["id"], cancel, generation)
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

    def load_policy(self, identity, cancel, generation):
        """Prepare a separate paused session; replace the old session only after its exact runtime load."""
        with self.lock:
            if self.config_error or self.desired is None:
                raise ValueError(
                    self.config_error or "Training configuration is unavailable"
                )
            config = deepcopy(self.desired)
            config_ticket = self.config_ticket
        directory, manifest = self.library.load(identity, config)
        self.pause(cancel)
        # Retain the paused old session until preparation and activation have both succeeded.
        old = self.worker
        staged = self.worker_factory(
            self.root / ("run-" + uuid.uuid4().hex),
            dict(
                config=config,
                config_ticket=config_ticket,
                generation=generation,
                streams=self.streams,
                trtexec=self.config.trtexec,
                build_timeout=self.config.build_timeout,
                heartbeat_timeout=self.config.heartbeat_timeout,
                source_revision=self.config.source_revision,
                initial_snapshot=str(directory / "policy.trt.training.npz"),
                source_bundle=manifest["id"],
            ),
        )
        self.preparing_worker = staged
        last_heartbeat = 0.0
        try:

            def prepared():
                """Drain only the staged worker and keep its supervision alive during export."""
                nonlocal last_heartbeat
                staged.events()
                if staged.state == "error":
                    raise RuntimeError(staged.error)
                if time.monotonic() - last_heartbeat > 0.5:
                    staged.send({"run": False})
                    last_heartbeat = time.monotonic()
                return staged.prepared_path is not None and staged.state == "paused"

            self.runtime.wait(
                prepared,
                cancel,
                "saved policy preparation",
                self.config.build_timeout + self.config.worker_timeout,
            )
            collection_generation = int(
                self.runtime.get(
                    self.runtime.po4ao, "collection_status", "requested_generation"
                )
            )
            model_id = self.publish(staged.prepared_path, generation, cancel)
            with self.lock:
                if cancel.is_set() or generation != self.record["generation"]:
                    raise InterruptedError("Set policy superseded")
                staged.send(
                    {"restore_id": model_id, "active_id": model_id, "run": False}
                )
                self.worker = staged
                self.applied = config
                self.record.update(
                    worker_started=True,
                    initialization_required=False,
                    selected_policy=identity,
                    warmup_required=collection_generation,
                )
                atomic_json(self.record_path, self.record)
                self.status.update(
                    warmup_required="yes",
                    learning="paused",
                    initialization_required="no",
                )
            if old is not None:
                old.stop()
        finally:
            if self.worker is not staged:
                with self.lock:
                    # An issued load can finish after cancellation. Never resume the old networks against it.
                    if generation == self.record["generation"] and any(
                        record and record["path"] == staged.prepared_path
                        for record in (self.record["pending"], self.record["eligible"])
                    ):
                        self.record["initialization_required"] = True
                        self.status["initialization_required"] = "yes"
                        atomic_json(self.record_path, self.record)
                staged.stop()
            self.preparing_worker = None

    def publish(self, directory, generation, cancel, issued=None):
        """Commit a verified bundle; retain unresolved loads separately from eligible policies."""
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
            if self.record["pending"] is not None:
                if self.publication is None:
                    raise RuntimeError(
                        "A previous reload was unresolved at restart; Reset is required"
                    )
                outcome = self.runtime.reload_outcome(self.publication[1])
                if outcome is None:
                    raise RuntimeError("A runtime reload is still unresolved")
                self.finish_publication(self.publication, outcome)
            link = self.root / ("current-" + uuid.uuid4().hex)
            link.symlink_to(candidate, target_is_directory=True)
            os.replace(link, self.root / "current")
            pending = {
                "path": str(candidate),
                "model_id": model_id,
                "generation": generation,
            }
            self.publication = None
            self.record["pending"] = pending
            atomic_json(self.record_path, self.record)
            previous_ticket = self.runtime.reload_ticket

            def bind(ticket):
                """Register the exact candidate/reference before its load can reach C++."""
                self.publication = (pending, ticket)
                if issued:
                    issued(model_id)

            try:
                self.runtime.begin_reload(model_id, issued=bind)
            finally:
                ticket = self.runtime.reload_ticket
                if (
                    ticket is not None
                    and ticket is not previous_ticket
                    and ticket["model_id"] == model_id
                ):
                    # A send failure after ticket installation may still have reached C++.
                    if self.publication is None:
                        bind(ticket)
                else:
                    self.record["pending"] = None
                    atomic_json(self.record_path, self.record)
            publication = self.publication
        outcome = self.settle_publication(cancel, publication)
        if outcome != "loaded":
            raise RuntimeError("PO4AO reload failed or was invalidated")
        return model_id

    def finish_publication(self, publication, outcome):
        """Commit only the captured candidate/ticket pair; discard stale status-thread results."""
        with self.lock:
            pending, ticket = publication
            if (
                self.publication is not publication
                or self.record["pending"] is not pending
                or self.runtime.reload_ticket is not ticket
                or ticket["model_id"] != pending["model_id"]
                or pending["generation"] != self.record["generation"]
            ):
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
            self.publication = None
            atomic_json(self.record_path, self.record)
            if self.worker:
                message = dict(
                    candidate_outcome_path=pending["path"], candidate_outcome=outcome
                )
                try:
                    message["active_id"] = self.runtime.get(
                        self.runtime.po4ao, "model_status", "effective_id"
                    )
                except (ConnectionError, KeyError):
                    message["collection"] = "unknown"
                self.worker.send(message)

    def settle_publication(self, cancel, publication=None):
        """Wait on one captured ticket; cancellation or a send failure cannot replace its identity."""
        with self.lock:
            if publication is None:
                if self.record["pending"] is None:
                    return None
                publication = self.publication
            if publication is None:
                raise RuntimeError(
                    "A previous reload was unresolved at restart; Reset is required"
                )
        pending, ticket = publication
        outcome = self.runtime.finish_reload(cancel, ticket)
        with self.lock:
            if (
                self.runtime.reload_ticket is not ticket
                or pending["generation"] != self.record["generation"]
            ):
                raise InterruptedError("Publication was superseded")
            self.finish_publication(publication, outcome)
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

    def tick(self):
        """Drain worker events and completed operations; drive heartbeat/config without blocking callbacks."""
        self.poll_config()
        self.reconcile_runtime()
        self.readback()
        try:
            with self.lock:
                self.save_source()
                self.status["save_available"] = "yes"
        except (ConnectionError, KeyError, RuntimeError):
            self.status["save_available"] = "no"
        with self.lock:
            required = self.record.get("warmup_required")
            if required is not None and self.status["collection"] == "ready":
                try:
                    observed = int(
                        self.runtime.get(
                            self.runtime.po4ao,
                            "collection_status",
                            "requested_generation",
                        )
                    )
                    if observed > required:
                        self.record.pop("warmup_required")
                        atomic_json(self.record_path, self.record)
                        self.status["warmup_required"] = "no"
                except (ConnectionError, KeyError, ValueError):
                    pass
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
                            worker is self.worker
                            and operation["future"] is None
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
                    elif self.status["controller"] == "unknown":
                        message["collection"] = "unknown"
                        message.pop("active_id", None)
                    self.collection_generation = self.runtime.get(
                        self.runtime.po4ao, "collection_status", "requested_generation"
                    )
                    message["generation"] = self.collection_generation
                except Exception:
                    message["collection"] = "unknown"
                with self.lock:
                    desired = deepcopy(self.desired)
                    config_ticket = self.config_ticket
                    pending_reset = self.pending_reset
                if (
                    desired
                    and not pending_reset
                    and (
                        revision(desired) != worker.revision
                        or getattr(worker, "config_ticket", None) != config_ticket
                    )
                ):
                    path = worker.run / (revision(desired) + ".json")
                    if not path.exists():
                        atomic_json(path, desired)
                    message["config_path"] = str(path)
                    message["config_ticket"] = config_ticket
                try:
                    with self.lock:
                        # Order effective-policy readback with publication outcomes, not an earlier tick snapshot.
                        if not self.foreign_model and not self.record["reset_pending"]:
                            try:
                                message["active_id"] = self.runtime.get(
                                    self.runtime.po4ao, "model_status", "effective_id"
                                )
                            except (ConnectionError, KeyError):
                                message.pop("active_id", None)
                                message["collection"] = "unknown"
                        worker.send(message)
                        self.last_heartbeat = time.monotonic()
                except BlockingIOError:
                    pass
                except OSError as error:
                    worker.state, worker.error = "error", str(error)
                    self.status["learning"], self.status["error"] = "error", str(error)
                    worker.stop()
            with self.lock:
                if worker is self.worker and worker.revision in self.config_history:
                    self.applied = deepcopy(self.config_history[worker.revision])
        with self.lock:
            publication = self.publication
        if publication is not None:
            try:
                outcome = self.runtime.reload_outcome(publication[1])
                if outcome is not None:
                    self.finish_publication(publication, outcome)
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
                if name == "training":
                    self.status["training_phase"] = (
                        "reset_pending"
                        if self.record["reset_pending"]
                        else "error" if error else "idle"
                    )
                self.status["error"] = str(error) if error else ""
                self.status["message"] = operation["command"] + (
                    " failed" if error else " complete"
                )
                if operation["command"] == "save_policy" and not error:
                    self.status["message"] += ": " + future.result()
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
