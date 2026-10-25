#!/usr/bin/env python3
## \file check-edt-probes.py
# \brief Operator-run disabled EDT kernel-probe registration check.

"""Register and remove disabled probes; never enable events or collect a trace.

Run manually by the instrument operator. Only this invocation's unique event
names may be removed. Failed cleanup prints their names for operator inspection.
"""

import argparse
import os
from pathlib import Path
import platform
import re
import signal
import struct
import uuid


KERNEL = "5.14.0-362.8.1.el9_3.x86_64"
VERSION = "5.6.8.0"
SRCVERSION = "A26E0B4FD875E93168B83BC"
BUILD_ID = "97857cff4fc9d75717d3b58e9c5bc3359a7204f0"
FUNCTIONS = ("edt_DpcForIsr", "edt_wake_up", "edt_start_intr_critical",
             "edt_increment_done", "edt_test_intr", "EdtTraceTimeStamp",
             "pdv_DeviceIntPdv", "edt_wake_up_interruptible", "edt_start_buffers",
             "edt_end_intr_critical")


def build_id(note):
    """Decode the GNU build-ID descriptor from this x86-64 ELF note byte string."""
    offset = 0
    while offset + 12 <= len(note):
        namesz, descsz, kind = struct.unpack_from("<III", note, offset)
        offset += 12
        name_end = offset + namesz
        desc_start = offset + ((namesz + 3) & ~3)
        desc_end = desc_start + descsz
        if name_end > len(note) or desc_end > len(note):
            raise ValueError("Truncated EDT build-ID note")
        if kind == 3 and note[offset:name_end] == b"GNU\0":
            return note[desc_start:desc_end].hex()
        offset = desc_start + ((descsz + 3) & ~3)
    raise ValueError("EDT GNU build-ID note was not found")


def check_identity(module=Path("/sys/module/edt")):
    """Require the operator-verified kernel and actual loaded EDT module build."""
    actual = (platform.release(), (module / "version").read_text().strip(),
              (module / "srcversion").read_text().strip(),
              build_id((module / "notes/.note.gnu.build-id").read_bytes()))
    if actual != (KERNEL, VERSION, SRCVERSION, BUILD_ID):
        raise RuntimeError(f"Loaded kernel/EDT identity differs from the verified build: {actual}")


def preflight(tracefs):
    """Require root, a mounted tracefs root, its existing control, and matched EDT."""
    if os.geteuid() != 0:
        raise RuntimeError("Run this registration check as root")
    mounted = False
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields, filesystem = line.split(" - ", 1)
        mountpoint = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)),
                            fields.split()[4])
        if Path(mountpoint).resolve() == tracefs.resolve() and filesystem.split()[0] == "tracefs":
            mounted = True
    if not mounted or not (tracefs / "kprobe_events").is_file():
        raise RuntimeError(f"No mounted tracefs root with kprobe_events at {tracefs}")
    check_identity()


class ProbeCheck:
    """Own attempted definitions in a unique event group, with exact reconciliation."""

    def __init__(self, tracefs, group=None):
        """Keep the tracefs root and a fresh invocation-specific event group."""
        self.tracefs = tracefs
        self.group = group or f"ocam_edt_check_{os.getpid()}_{uuid.uuid4().hex}"
        self.attempted = []
        self.stopped = 0
        self.errors = []

    def report(self, message):
        """Report a result without allowing output failure to interrupt cleanup."""
        try:
            print(message)
        except (OSError, ValueError) as error:
            self.errors.append(f"Output failed: {error}")

    def definitions(self):
        """Read all probe definitions, retaining duplicate lines for collision checks."""
        result = {}
        for line in (self.tracefs / "kprobe_events").read_text().splitlines():
            fields = line.split()
            if fields and ":" in fields[0]:
                name = fields[0].split(":", 1)[1]
                # The kernel may render an entry's implicit zero offset explicitly.
                if len(fields) == 2:
                    fields[1] = re.sub(r"\+(?:0|0x0)$", "", fields[1])
                result.setdefault(name, []).append(" ".join(fields))
        return result

    def guard_group(self):
        """Refuse a pre-existing event group rather than adopting its definitions."""
        if (self.tracefs / "events" / self.group).exists() or any(
                name.startswith(self.group + "/") for name in self.definitions()):
            raise RuntimeError(f"Event group already exists: {self.group}")

    def write_command(self, command):
        """Append one complete command without creating or truncating any file."""
        data = (command + "\n").encode("ascii")
        descriptor = os.open(self.tracefs / "kprobe_events", os.O_WRONLY | os.O_APPEND)
        try:
            if os.write(descriptor, data) != len(data):
                raise OSError("Short kprobe_events command write; registration state is uncertain")
        finally:
            os.close(descriptor)

    def interrupt(self, signum, frame):
        """Request shutdown between operations without interrupting ownership bookkeeping."""
        self.stopped = signum

    def register(self, function, kind):
        """Attempt one disabled entry or return probe and verify its exact definition."""
        name = f"{self.group}/{kind}_{function}"
        # Pin the unused return-instance reservation for deterministic readback.
        probe = "p" if kind == "p" else "r1"
        definition = f"{probe}:{name} edt:{function}"
        if name in self.definitions():
            raise RuntimeError(f"Refusing existing event: {name}")
        # Record before writing so a failed/short write is reconciled during cleanup.
        self.attempted.append((name, definition))
        self.write_command(definition)
        if self.definitions().get(name) != [definition]:
            raise RuntimeError(f"Registered definition does not match: {name}")
        roots = [self.tracefs] + list((self.tracefs / "instances").glob("*"))
        for root in roots:
            if (root / "events" / name / "enable").read_text().strip() != "0":
                raise RuntimeError(f"Event is enabled in {root}: {name}; no enable state was changed")
        self.report(f"Registered disabled: {name} -> edt:{function}")

    def cleanup(self):
        """Remove only exact attempted definitions, continuing after cleanup failures."""
        for name, definition in reversed(self.attempted):
            try:
                current = self.definitions().get(name)
                if current is None:
                    continue
                if current != [definition]:
                    raise RuntimeError("Definition changed; refusing to remove it")
                self.write_command(f"-:{name} {definition.split()[1]}")
                if name in self.definitions():
                    raise RuntimeError("Definition remains after removal")
                self.report(f"Removed: {name}")
            except (OSError, RuntimeError) as error:
                message = f"Cleanup failed for {name}: {error}"
                self.errors.append(message)
                self.report(message)


def main(arguments=None):
    """Run the operator-only check and return nonzero for any rejection or cleanup failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracefs", type=Path, default=Path("/sys/kernel/tracing"))
    options = parser.parse_args(arguments)
    check = ProbeCheck(options.tracefs)
    handlers = {}
    try:
        preflight(options.tracefs)
        check.guard_group()
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, check.interrupt)
        check.report(f"Verified EDT {VERSION}, build ID {BUILD_ID}; group {check.group}")
        for function in FUNCTIONS:
            for kind in ("p", "r"):
                if check.stopped:
                    break
                try:
                    check.register(function, kind)
                except (OSError, RuntimeError) as error:
                    message = f"Rejected {kind} edt:{function}: {error}"
                    check.errors.append(message)
                    check.report(message)
            if check.stopped:
                break
        # A module-qualified probe can otherwise remain deferred after module removal.
        check_identity()
    except (OSError, RuntimeError, ValueError) as error:
        check.errors.append(str(error))
        check.report(f"Check failed: {error}")
    finally:
        check.cleanup()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    if check.stopped:
        check.report(f"Interrupted by signal {check.stopped}; cleanup was attempted")
    check.report(f"Check finished: {len(check.attempted)} attempted, {len(check.errors)} errors")
    return int(bool(check.errors or check.stopped))


if __name__ == "__main__":
    raise SystemExit(main())
