## \file process.py
# \brief Owned child-process lifecycle and bounded JSON-line control pipes.
"""No process discovery, notebook adoption or cross-trainer locks."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import psutil


def atomic_json(path, value):
    """Replace one resolved configuration or eligibility record on its own filesystem."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, sort_keys=True, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class Worker:
    """Own one worker process group; reset stops only that group and its export children."""

    def __init__(self, run, launch):
        """Start the installed module with logs separated from its protocol."""
        self.lock = threading.RLock()
        self.run = Path(run)
        self.run.mkdir(parents=True, exist_ok=False)
        atomic_json(self.run / "launch.json", launch)
        self.log = (self.run / "worker.log").open("ab", buffering=0)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "xapp.po4aoTrainer.worker", str(self.run)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            start_new_session=True,
            bufsize=0,
        )
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.identity = psutil.Process(self.process.pid)
        self.identity.create_time()
        self.children = []
        self.buffer = b""
        self.state = "starting"
        self.revision = ""
        self.iteration = 0
        self.error = ""
        self.closed = False
        self.prepared_path = None
        self.config_ticket = None

    def send(self, message):
        """Send one atomic short control record, without blocking the app loop."""
        data = (json.dumps(message) + "\n").encode()
        if len(data) > 4096:
            raise ValueError("Control record exceeds atomic pipe bound")
        with self.lock:
            if self.closed:
                return  # Reset can close the worker while a cancelled operation finishes.
            count = os.write(self.process.stdin.fileno(), data)
            if count != len(data):
                raise RuntimeError("Incomplete worker control write")

    def events(self):
        """Drain bounded protocol output and retain worker-reported completion state."""
        with self.lock:
            if self.closed:
                return []
            try:
                self.buffer += os.read(self.process.stdout.fileno(), 65536)
            except BlockingIOError:
                pass
            if len(self.buffer) > 1048576:
                raise RuntimeError("Worker protocol record exceeded bound")
            records = []
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                event = json.loads(line)
                records.append(event)
                if event["event"] == "configuration":
                    self.config_ticket = event["ticket"]
                    self.revision = event["revision"]
                if event["event"] == "prepared":
                    self.prepared_path = event["path"]
                if event["event"] == "build":
                    try:
                        child = psutil.Process(event["pid"])
                        child.create_time()
                        self.children.append(child)
                    except psutil.NoSuchProcess:
                        pass
                if event["event"] == "state":
                    self.state = event["state"]
                    self.iteration = event.get("iteration", self.iteration)
                    self.revision = event.get("revision", self.revision)
                elif event["event"] == "error":
                    self.error, self.state = event["error"], "error"
            if self.process.poll() is not None and self.state != "error":
                self.state, self.error = (
                    "error",
                    "Owned worker exited; Reset is required",
                )
            return records

    def stop(self, timeout=5):
        """Terminate the owned group, bound cleanup, and close descriptors exactly once."""
        with self.lock:
            if self.closed:
                return
            try:
                if self.identity.is_running():
                    self.children.extend(self.identity.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if self.identity.is_running():
                    try:
                        os.killpg(self.process.pid, sig)
                    except ProcessLookupError:
                        pass
                try:
                    self.process.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
            for child in self.children:
                try:
                    child.kill()  # psutil checks the cached process creation identity.
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(self.children, timeout=timeout)
            if self.process.poll() is None:
                raise TimeoutError("Owned worker did not exit after SIGKILL")
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()
            self.closed = True
