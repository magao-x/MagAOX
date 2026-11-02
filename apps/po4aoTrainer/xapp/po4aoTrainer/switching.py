## \file switching.py
# \brief Completion-aware NN/CACAO routing using the existing INDI and milk interfaces.
"""No loop-state access and no coupling to PO4AO/training state."""

import math
import os
from pathlib import Path
import stat
import threading
import time
import psutil
from purepyindi2.constants import PropertyState, SwitchState
from .observations import Observations


class MilkWriter:
    """Observe the installed FPS RUN PID and deliver bounded FIFO requests."""

    def __init__(self, name, fifo, process_match, fps=None):
        """Use configured writer identity; injectable FPS supports offline tests."""
        if not name or any(c.isspace() for c in name):
            raise ValueError("FPS name must be a single token")
        self.name, self.fifo, self.process_match = name, Path(fifo), process_match
        self.fps = fps
        self.identity = None
        self.file_identity = None
        self.lock = (
            threading.Lock()
        )  # App status polling shares this binding with switching.

    def running(self):
        """Return verified RUN state; unknown/PID-reuse evidence raises rather than claiming stopped."""
        with self.lock:
            if self.file_identity is not None:
                info = Path("/milk/shm", self.name + ".fps.shm").stat()
                if (info.st_dev, info.st_ino) != self.file_identity:
                    self.fps = None
                    self.identity = None
            if self.fps is None:
                if not Path("/milk/shm", self.name + ".fps.shm").is_file():
                    raise RuntimeError("FPS is absent: " + self.name)
                import CacaoProcessTools

                self.fps = CacaoProcessTools.fps(self.name)
                info = Path("/milk/shm", self.name + ".fps.shm").stat()
                self.file_identity = (info.st_dev, info.st_ino)
            md = self.fps.md()
            pid, stamp = int(md.runpid), (
                md.runpidstarttime.tv_sec,
                md.runpidstarttime.tv_nsec,
            )
            running = self.fps.RUNrunning
            if running == 0:
                self.identity = None
                return False
            if running != 1 or pid <= 0:
                raise RuntimeError("CACAO writer RUN state is unverified")
            process = psutil.Process(pid)
            if (
                process.status() == psutil.STATUS_ZOMBIE
                or not self.process_match
                or self.process_match not in " ".join(process.cmdline())
            ):
                raise RuntimeError(
                    "FPS RUN PID does not identify the configured writer"
                )
            identity = (pid, stamp, process.create_time())
            if (
                self.identity
                and self.identity[:2] == identity[:2]
                and self.identity[2] != identity[2]
            ):
                raise RuntimeError("FPS RUN PID was reused")
            self.identity = identity
            return True

    def command(self, enabled):
        """Write one atomic FIFO command without blocking for an absent FPS controller."""
        fd = os.open(self.fifo, os.O_WRONLY | os.O_NONBLOCK)
        try:
            if not stat.S_ISFIFO(os.fstat(fd).st_mode):
                raise RuntimeError("Configured FPS path is not a FIFO")
            data = f"{'runstart' if enabled else 'runstop'} {self.name}\n".encode()
            if os.write(fd, data) != len(data):
                raise RuntimeError("Incomplete FPS FIFO delivery")
        finally:
            os.close(fd)


class Runtime:
    """Narrow transport adapter shared by supervisor operations and offline fixtures."""

    def __init__(
        self,
        client,
        writer,
        nn,
        po4ao,
        modulation,
        timeout=30,
        *,
        freshness_timeout=5.0,
        unmodulated_state=3.0,
        modulated_state=4.0,
    ):
        """Bind configured device identities; do not infer them from filenames."""
        if (
            not all(
                math.isfinite(value) for value in (unmodulated_state, modulated_state)
            )
            or unmodulated_state == modulated_state
        ):
            raise ValueError("Modulation states must be finite and distinct")
        self.client, self.writer = client, writer
        self.nn, self.po4ao, self.modulation, self.timeout = (
            nn,
            po4ao,
            modulation,
            timeout,
        )

        self.observations = Observations(
            client, (nn, po4ao, modulation), freshness_timeout, heartbeat_device=po4ao
        )
        self.unmodulated_state, self.modulated_state = (
            unmodulated_state,
            modulated_state,
        )
        self.reload_ticket = None
        self.observations.refresh()

    def get(self, device, prop, element):
        """Read a current-lifetime, non-expired INDI observation."""
        return self.observations.vector(device, prop)[element]

    def send(self, device, prop, values):
        """Send through a freshly discovered typed property."""
        vector = self.observations.vector(device, prop)
        self.client.connection.send(vector.make_new_property(**values))

    def close(self):
        """Release observation/reload callbacks without changing device state."""
        if self.reload_ticket is not None:
            self.client.unregister_callback(
                self.reload_ticket["callback"], self.po4ao, "reload_engine"
            )
        self.observations.close()

    def wait(self, predicate, cancel, label, timeout=None):
        """Wait boundedly for observed completion, with cooperative cancellation."""
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            self.observations.refresh()
            if predicate():
                break
            if cancel.wait(0.05):
                raise InterruptedError(label + " cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(label + " did not complete")
        if cancel.is_set():
            raise InterruptedError(label + " cancelled")

    def nn_quiescent(self):
        """Recognize the donor's completed disable, including a latched model fault after quiescence."""
        return (
            self.get(self.nn, "enable", "toggle") == SwitchState.OFF
            and self.get(self.nn, "model_status", "status")
            in ("disabled", "fault: reload required")
            and not self.get(self.nn, "model_status", "active")
        )

    def nn_enabled(self, enabled, cancel):
        """Allow fault recovery only after confirmed quiescence; activation still requires healthy output."""
        self.send(
            self.nn,
            "enable",
            {"toggle": SwitchState.ON if enabled else SwitchState.OFF},
        )
        self.wait(
            (
                (
                    lambda: self.get(self.nn, "enable", "toggle") == SwitchState.ON
                    and self.observations.vector(self.nn, "enable").state
                    == PropertyState.OK
                    and self.get(self.nn, "model_status", "status") == "active"
                    and bool(self.get(self.nn, "model_status", "active"))
                )
                if enabled
                else self.nn_quiescent
            ),
            cancel,
            "NN enable/disable",
        )

    def source(self):
        """Distinguish verified stopped writers from unavailable observations and in-flight NN work."""
        phase = self.get(self.nn, "model_status", "status")
        active = (
            self.get(self.nn, "enable", "toggle") == SwitchState.ON
            and self.observations.vector(self.nn, "enable").state == PropertyState.OK
            and phase == "active"
            and bool(self.get(self.nn, "model_status", "active"))
        )
        if not active and not self.nn_quiescent():
            return "transition"
        cacao = self.writer.running()
        return (
            "conflict"
            if active and cacao
            else "nn" if active else "cacao" if cacao else "stopped"
        )

    def modulation_target(self, target, cancel):
        """Apply only the agreed linked comparison preset and wait for actual current values."""
        if target == "cacao":
            self.send(self.modulation, "modFrequency", {"target": 2000.0})
            self.send(self.modulation, "modRadius", {"target": 3.0})
        self.send(
            self.modulation,
            "modState",
            {
                "target": (
                    self.unmodulated_state if target == "nn" else self.modulated_state
                )
            },
        )
        self.wait(
            lambda: float(self.get(self.modulation, "modState", "current"))
            == (self.unmodulated_state if target == "nn" else self.modulated_state),
            cancel,
            "modulation state",
        )
        if target == "cacao":
            self.wait(
                lambda: abs(
                    float(self.get(self.modulation, "modFrequency", "current")) - 2000
                )
                < 0.01
                and abs(float(self.get(self.modulation, "modRadius", "current")) - 3)
                < 0.01,
                cancel,
                "modulation preset",
            )

    def switch(self, target, cancel, *, linked=False, model=None, expected=None):
        """Hand over verified writers, or recover from both stopped, without changing unrelated control state."""
        if target not in ("nn", "cacao"):
            raise ValueError("Unknown reconstructor target")
        source = self.source()
        if expected is not None and source != expected:
            raise ValueError("Reconstructor changed since the request")
        if source not in ("nn", "cacao", "stopped"):
            raise RuntimeError("Reconstructor state is not verified")
        if target == "nn":
            catalog = self.observations.vector(self.nn, "model")
            if model is not None and model not in catalog:
                raise ValueError("Unknown NN catalog element")
            filename = (
                catalog._elements[model].label
                if model is not None
                else self.get(self.nn, "model_status", "loaded")
            )
            if not filename:
                raise ValueError("No selected/loaded NN model")
            if (
                source == "nn"
                and self.get(self.nn, "model_status", "active") == filename
            ):
                return
            self.nn_enabled(False, cancel)
            if model is not None:
                already_loaded = (
                    self.get(self.nn, "model_status", "loaded") == filename
                    and self.get(self.nn, "model_status", "requested") == filename
                    and self.observations.vector(self.nn, "model").state
                    == PropertyState.OK
                )
                if not already_loaded:
                    self.send(
                        self.nn,
                        "model",
                        {
                            name: SwitchState.ON if name == model else SwitchState.OFF
                            for name in catalog
                        },
                    )
                self.wait(
                    lambda: self.get(self.nn, "model_status", "loaded") == filename
                    and self.get(self.nn, "model_status", "requested") == filename
                    and self.observations.vector(self.nn, "model").state
                    == PropertyState.OK,
                    cancel,
                    "NN model load",
                )
            if source == "cacao":
                self.writer.command(False)
                self.wait(
                    lambda: not self.writer.running(), cancel, "CACAO writer stop"
                )
                if linked:
                    self.modulation_target(target, cancel)
            self.nn_enabled(True, cancel)
        else:
            if source == "cacao":
                return
            self.nn_enabled(False, cancel)
            if linked:
                self.modulation_target(target, cancel)
            self.writer.command(True)
            self.wait(self.writer.running, cancel, "CACAO writer start")

    def runtime_state(self):
        """Read the identity and requested/applied control generations used for reconciliation."""
        vector = self.observations.vector(self.po4ao, "model_status")
        state = {
            name: vector[name]
            for name in (
                "instance",
                "reset_request",
                "reset_applied",
                "control_request",
                "control_target",
                "active_id",
                "reload",
            )
        }
        if (
            not state["instance"]
            or state["control_target"] not in ("on", "off")
            or any(
                not str(state[name]).isdigit()
                for name in ("reset_request", "reset_applied", "control_request")
            )
            or len(state["active_id"]) != 16
        ):
            raise ConnectionError("Runtime status is not fully initialized")
        return state

    def request_control(self, enabled):
        """Queue controller selection without waiting for an input frame."""
        self.send(
            self.po4ao,
            "control_enable",
            {"toggle": SwitchState.ON if enabled else SwitchState.OFF},
        )

    def control_complete(self, enabled):
        """Require the requested selection to have reached an emitted frame."""
        return (
            self.get(self.po4ao, "control_enable", "toggle")
            == (SwitchState.ON if enabled else SwitchState.OFF)
            and self.observations.vector(self.po4ao, "control_enable").state
            == PropertyState.OK
        )

    def control(self, enabled, cancel):
        """Request selection and wait for post-frame acknowledgement."""
        if cancel.is_set():
            raise InterruptedError("PO4AO control cancelled")
        self.request_control(enabled)
        self.wait(lambda: self.control_complete(enabled), cancel, "PO4AO control")

    def request_reset(self):
        """Send fallback and generation invalidation before waiting for any frame or worker."""
        ticket = {
            "instance": self.get(self.po4ao, "model_status", "instance"),
            "after": int(self.get(self.po4ao, "model_status", "reset_request")),
        }
        self.request_control(False)
        self.send(self.po4ao, "reset_model", {"toggle": SwitchState.ON})
        return ticket

    def reset_complete(self, ticket, external=False):
        """Require post-frame reset completion in the intended runtime lifetime."""
        same = self.get(self.po4ao, "model_status", "instance") == ticket["instance"]
        reset = int(self.get(self.po4ao, "model_status", "reset_applied"))
        return (
            same
            and reset > ticket["after"]
            and (
                external
                or (
                    self.get(self.po4ao, "control_enable", "toggle") == SwitchState.OFF
                    and self.observations.vector(self.po4ao, "control_enable").state
                    == PropertyState.OK
                )
            )
        )

    def begin_reload(self, model_id, issued=None):
        """Keep a ticket after issuance; cancelling a caller's wait cannot cancel C++ loading."""
        if self.reload_ticket is not None:
            if self.reload_outcome() is None:
                raise RuntimeError("A runtime reload is still unresolved")
            self.client.unregister_callback(
                self.reload_ticket["callback"], self.po4ao, "reload_engine"
            )
        ticket = dict(
            model_id=model_id,
            instance=self.get(self.po4ao, "model_status", "instance"),
            reset=int(self.get(self.po4ao, "model_status", "reset_request")),
            epoch=self.observations.epoch,
            busy=False,
            terminal=False,
            outcome=None,
        )

        def acknowledge(message):
            """Remember Busy/terminal updates even when they occur between polls."""
            state = getattr(message, "state", None)
            if state == PropertyState.BUSY:
                ticket["busy"] = True
            elif ticket["busy"] and state in (PropertyState.OK, PropertyState.ALERT):
                ticket["terminal"] = True

        ticket["callback"] = acknowledge
        self.reload_ticket = ticket
        self.client.register_callback(acknowledge, self.po4ao, "reload_engine")
        if issued:
            issued(ticket)
        self.send(self.po4ao, "reload_engine", {"toggle": SwitchState.ON})

    def reload_outcome(self, ticket=None):
        """Return a settled load/reset outcome, retaining unresolved tickets through timeouts."""
        ticket = self.reload_ticket if ticket is None else ticket
        if ticket is None:
            return None
        if ticket["outcome"] is not None:
            return ticket["outcome"]
        if (
            self.get(self.po4ao, "model_status", "instance") != ticket["instance"]
            or int(self.get(self.po4ao, "model_status", "reset_request"))
            != ticket["reset"]
        ):
            ticket["outcome"] = "invalidated"
        elif ticket["terminal"] or (
            ticket["busy"] and self.observations.epoch != ticket["epoch"]
        ):
            state = self.get(self.po4ao, "model_status", "reload")
            if state == "failed":
                ticket["outcome"] = "failed"
            elif (
                state == "idle"
                and self.get(self.po4ao, "model_status", "active_id")
                == ticket["model_id"]
            ):
                ticket["outcome"] = "loaded"
        return ticket["outcome"]

    def finish_reload(self, cancel, ticket=None):
        """Wait on the captured load even if a later request replaces the runtime's current ticket."""
        ticket = self.reload_ticket if ticket is None else ticket
        self.wait(
            lambda: self.reload_outcome(ticket) is not None,
            cancel,
            "PO4AO engine reload",
        )
        return ticket["outcome"]

    def reload(self, model_id, cancel):
        """Convenience operation for callers that do not own publication bookkeeping."""
        self.begin_reload(model_id)
        if self.finish_reload(cancel) != "loaded":
            raise RuntimeError("PO4AO reload failed or was invalidated")

    def recollect(self, cancel, timeout):
        """Require a newer complete frozen snapshot; interrupted collection never resumes learning."""
        old = int(self.get(self.po4ao, "collection_status", "requested_generation"))
        self.send(self.po4ao, "recollect_warmup", {"toggle": SwitchState.ON})

        def complete():
            """Pair a complete snapshot with this newer requested collection generation."""
            generation = int(self.get(self.po4ao, "collection_status", "generation"))
            state = self.get(self.po4ao, "collection_status", "state")
            if generation > old and state == "interrupted":
                raise RuntimeError(
                    "Warmup collection interrupted; learning remains paused"
                )
            return (
                generation > old
                and state == "ready"
                and generation
                == int(
                    self.get(self.po4ao, "collection_status", "requested_generation")
                )
            )

        self.wait(complete, cancel, "warmup snapshot", timeout)


def main():
    """Operator-only manual entry point using the same bounded switching implementation."""
    import argparse
    import threading
    from purepyindi2.client import IndiClient
    from .app import TrainerConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("nn", "cacao"))
    parser.add_argument("--config", default="/opt/MagAOX/config/po4aoTrainer.conf")
    parser.add_argument(
        "--linked",
        action="store_true",
        help="Apply the 2 kHz / radius 3 comparison preset",
    )
    parser.add_argument(
        "--model", help="NN catalog element identifier (only for target nn)"
    )
    args = parser.parse_args()
    config = TrainerConfig.from_config(config_path_or_paths=args.config)
    client = IndiClient()
    client.connect()
    try:
        names = [
            config.nn_device + "." + prop
            for prop in ("enable", "model", "model_status")
        ]
        if args.linked:
            names += [
                config.modulation_device + "." + prop
                for prop in ("modState", "modFrequency", "modRadius")
            ]
        runtime = Runtime(
            client,
            MilkWriter(config.fps_name, config.fps_fifo, config.writer_process_match),
            config.nn_device,
            config.po4ao_device,
            config.modulation_device,
            config.operation_timeout,
            freshness_timeout=config.freshness_timeout,
            unmodulated_state=config.unmodulated_state,
            modulated_state=config.modulated_state,
        )

        def discovered():
            """Wait for definitions from this connection instead of the client's retained cache."""
            try:
                for name in names:
                    device, prop = name.split(".", 1)
                    runtime.observations.vector(device, prop)
                return bool(runtime.source())
            except (ConnectionError, KeyError):
                runtime.observations.refresh()
                return False

        runtime.wait(discovered, threading.Event(), "fresh discovery")
        runtime.switch(
            args.target, threading.Event(), linked=args.linked, model=args.model
        )
    finally:
        if "runtime" in locals():
            runtime.close()
        client.connection.stop()


if __name__ == "__main__":
    main()
