## \file switching.py
# \brief Completion-aware NN/CACAO routing using the existing INDI and milk interfaces.
"""No loop-state access and no coupling to PO4AO/training state."""

import os
from pathlib import Path
import stat
import threading
import time
import psutil
from purepyindi2.constants import ConnectionStatus, PropertyState, SwitchState


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

    def __init__(self, client, writer, nn, po4ao, modulation, timeout=30):
        """Bind configured device identities; do not infer them from filenames."""
        self.client, self.writer = client, writer
        self.nn, self.po4ao, self.modulation, self.timeout = (
            nn,
            po4ao,
            modulation,
            timeout,
        )

    def get(self, device, prop, element):
        """Read only from a connected INDI cache; missing definitions fail explicitly."""
        if self.client.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("INDI is disconnected")
        return self.client[f"{device}.{prop}.{element}"]

    def send(self, device, prop, values):
        """Send one typed request, preserving multi-element target/selection semantics."""
        vector = self.client[f"{device}.{prop}"]
        if self.client.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("INDI is disconnected")
        self.client.connection.send(vector.make_new_property(**values))

    def wait(self, predicate, cancel, label, timeout=None):
        """Wait boundedly for observed completion, with cooperative cancellation."""
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while not predicate():
            if cancel.wait(0.05):
                raise InterruptedError(label + " cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError(label + " did not complete")
        if cancel.is_set():
            raise InterruptedError(label + " cancelled")

    def nn_enabled(self, enabled, cancel):
        """Wait for the donor NN app's post-publication enable/disable acknowledgement."""
        target = SwitchState.ON if enabled else SwitchState.OFF
        self.send(self.nn, "enable", {"toggle": target})
        self.wait(
            lambda: self.get(self.nn, "enable", "toggle") == target
            and self.client[f"{self.nn}.enable"].state == PropertyState.OK
            and self.get(self.nn, "model_status", "status")
            == ("active" if enabled else "disabled"),
            cancel,
            "NN enable/disable",
        )

    def source(self):
        """Combine actual NN acknowledgement with observed CACAO writer state."""
        nn = self.get(self.nn, "enable", "toggle") == SwitchState.ON
        if self.client[f"{self.nn}.enable"].state != PropertyState.OK:
            return "transition"
        cacao = self.writer.running()
        return (
            "conflict"
            if nn and cacao
            else "nn" if nn else "cacao" if cacao else "unknown"
        )

    def modulation_target(self, target, cancel):
        """Apply only the agreed linked comparison preset and wait for actual current values."""
        if target == "cacao":
            self.send(self.modulation, "modFrequency", {"target": 2000.0})
            self.send(self.modulation, "modRadius", {"target": 3.0})
        self.send(
            self.modulation, "modState", {"target": 3.0 if target == "nn" else 4.0}
        )
        self.wait(
            lambda: float(self.get(self.modulation, "modState", "current"))
            == (3 if target == "nn" else 4),
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
        """Stop the old writer before starting the target; never issue loop/PO4AO/training commands."""
        if target not in ("nn", "cacao"):
            raise ValueError("Unknown reconstructor target")
        if expected is not None and self.source() != expected:
            raise ValueError("Reconstructor changed since the request")
        if target == "nn":
            catalog = self.client[f"{self.nn}.model"]
            if model is not None and model not in catalog:
                raise ValueError("Unknown NN catalog element")
            self.writer.command(False)
            self.wait(lambda: not self.writer.running(), cancel, "CACAO writer stop")
            self.nn_enabled(False, cancel)
            if model is not None:
                filename = catalog._elements[model].label
                if not filename:
                    raise ValueError("NN model catalog lacks filename labels")
                already_loaded = (
                    self.get(self.nn, "model_status", "loaded") == filename
                    and self.get(self.nn, "model_status", "requested") == filename
                    and catalog.state == PropertyState.OK
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
                    and self.client[f"{self.nn}.model"].state == PropertyState.OK,
                    cancel,
                    "NN model load",
                )
            if linked:
                self.modulation_target(target, cancel)
            self.nn_enabled(True, cancel)
        else:
            self.nn_enabled(False, cancel)
            if linked:
                self.modulation_target(target, cancel)
            self.writer.command(True)
            self.wait(self.writer.running, cancel, "CACAO writer start")

    def control(self, enabled, cancel):
        """Wait for an emitted frame carrying the requested PO4AO selection."""
        target = SwitchState.ON if enabled else SwitchState.OFF
        self.send(self.po4ao, "control_enable", {"toggle": target})
        self.wait(
            lambda: self.get(self.po4ao, "control_enable", "toggle") == target
            and self.client[f"{self.po4ao}.control_enable"].state == PropertyState.OK,
            cancel,
            "PO4AO control",
        )

    def reset(self, cancel):
        """Wait for a newer reset generation to reach an emitted frame."""
        old = int(self.get(self.po4ao, "model_status", "reset_request"))
        self.send(self.po4ao, "reset_model", {"toggle": SwitchState.ON})
        self.wait(
            lambda: int(self.get(self.po4ao, "model_status", "reset_applied")) > old,
            cancel,
            "PO4AO reset",
        )

    def reload(self, model_id, cancel):
        """Require a fresh Busy/terminal acknowledgement, even when reloading the same engine."""
        busy, terminal = threading.Event(), threading.Event()

        def acknowledge(message):
            """Capture transitions that can occur between cache polls."""
            state = getattr(message, "state", None)
            if state == PropertyState.BUSY:
                busy.set()
            elif busy.is_set() and state in (PropertyState.OK, PropertyState.ALERT):
                terminal.set()

        def loaded():
            """Read final model/error evidence only after this reload was acknowledged."""
            if not terminal.is_set():
                return False
            if self.get(self.po4ao, "model_status", "reload") == "failed":
                raise RuntimeError(self.get(self.po4ao, "model_status", "reload_error"))
            return (
                self.get(self.po4ao, "model_status", "active_id") == model_id
                and self.get(self.po4ao, "model_status", "reload") == "idle"
            )

        self.client.register_callback(acknowledge, self.po4ao, "reload_engine")
        try:
            self.send(self.po4ao, "reload_engine", {"toggle": SwitchState.ON})
            self.wait(loaded, cancel, "PO4AO engine reload")
        finally:
            self.client.unregister_callback(acknowledge, self.po4ao, "reload_engine")

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
        client.get_properties_and_wait(names, timeout_sec=config.operation_timeout)
        runtime = Runtime(
            client,
            MilkWriter(config.fps_name, config.fps_fifo, config.writer_process_match),
            config.nn_device,
            config.po4ao_device,
            config.modulation_device,
            config.operation_timeout,
        )
        runtime.switch(
            args.target, threading.Event(), linked=args.linked, model=args.model
        )
    finally:
        client.connection.stop()


if __name__ == "__main__":
    main()
