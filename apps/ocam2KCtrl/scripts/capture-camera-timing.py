#!/usr/bin/env python3
## \file capture-camera-timing.py
# \brief Operator-run private scheduler/IRQ and EDT-wrapper capture for camera correlation.

"""Capture a bounded camera-CPU scheduler/IRQ trace in a private tracefs instance.

Run manually by the instrument operator. This script never changes scheduling,
CPU affinity, IRQ affinity, camera settings, or any pre-existing trace buffer.
Optional EDT probes use invocation-specific global definitions, enabled only
in the private instance; their activation includes setup and settling.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time
import uuid


def cpu_list(value):
    """Expand a Linux CPU list, rejecting malformed or excessively large input."""
    cpus = set()
    for item in value.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", item):
            raise ValueError("CPU list must look like 45 or 45,61 or 45-46")
        ends = [int(number) for number in item.split("-")]
        first, last = ends[0], ends[-1]
        if first > last or last > 65535:
            raise ValueError("Invalid CPU range")
        cpus.update(range(first, last + 1))
    return sorted(cpus)


def cpu_mask(cpus):
    """Encode a CPU list as tracefs's comma-separated 32-bit hexadecimal mask."""
    mask = sum(1 << cpu for cpu in cpus)
    groups = []
    while mask:
        groups.append(f"{mask & 0xffffffff:08x}")
        mask >>= 32
    return ",".join(reversed(groups))


def anchor():
    """Bracket a realtime reading with CLOCK_MONOTONIC nanoseconds."""
    before = time.monotonic_ns()
    realtime = time.time_ns()
    return {"monotonic_before_ns": before, "realtime_ns": realtime,
            "monotonic_after_ns": time.monotonic_ns()}


def read_if_present(path):
    """Read a diagnostic file without making unavailable metadata fatal."""
    try:
        return path.read_text()
    except OSError as error:
        return f"Unavailable: {error}\n"


def command_output(arguments):
    """Collect a read-only command's output without using a shell."""
    try:
        result = subprocess.run(arguments, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=5, check=False)
        return result.stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Unavailable: {error}\n"


def edt_probe_helper():
    """Load the shared ownership and exact-build checks only for requested EDT probes."""
    path = Path(__file__).resolve().with_name("check-edt-probes.py")
    spec = importlib.util.spec_from_file_location("ocam_edt_probes", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


class TraceCapture:
    """Own exactly one new tracefs instance and its output directory."""

    def __init__(self, tracefs, output, cpus, extra_events=False, edt_helper=None):
        """Own one private instance, its native events, and any requested EDT definitions."""
        self.tracefs = tracefs
        self.output = output
        self.cpus = cpus
        self.extra_events = extra_events
        self.instance = tracefs / "instances" / f"ocam-timing-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.created = False
        self.events = []
        self.event_groups = {}
        # Clock anchors bracket the stop write, including interrupt/error cleanup.
        self.stop_anchors = {}
        self.edt_helper = edt_helper
        self.probes = None
        self.edt = {"requested": edt_helper is not None}
        if edt_helper is not None:
            nonce = uuid.uuid4().hex
            self.probes = edt_helper.ProbeCheck(
                tracefs, group=f"ocam_edt_capture_{os.getpid()}_{nonce}",
                event_prefix=f"capture_{nonce[:16]}_")
            self.edt.update({"group": self.probes.group, "definitions": {}, "formats": {},
                             "activation": {}, "deactivation": {}, "safety_errors": [],
                             "remaining_definitions": [], "window_includes_settle": True,
                             "return_maxactive": 64, "return_pool_misses_verified": False,
                             "verified_identity": {"kernel": edt_helper.KERNEL,
                                                   "version": edt_helper.VERSION,
                                                   "srcversion": edt_helper.SRCVERSION,
                                                   "build_id": edt_helper.BUILD_ID}})

    def create_instance(self):
        """Create a unique kernel-owned instance; an existing name is an error."""
        # Defer ordinary termination until the successful mkdir is owned.
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGINT, signal.SIGTERM))
        try:
            self.instance.mkdir()
            self.created = True
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)

    def remove_instance(self):
        """Remove only this instance, without recursively deleting any files."""
        self.instance.rmdir()
        self.created = False

    def write(self, relative, value):
        """Write a control belonging exclusively to this instance."""
        (self.instance / relative).write_text(f"{value}\n")

    def setup(self):
        """Configure bounded per-CPU event buffers without touching trace options."""
        self.create_instance()
        self.write("tracing_on", 0)
        clocks = (self.instance / "trace_clock").read_text().replace("[", "").replace("]", "").split()
        if "mono" not in clocks:
            raise RuntimeError("This kernel lacks the mono trace clock; no substitute is selected")
        self.write("trace_clock", "mono")
        tracer = self.instance / "current_tracer"
        if tracer.exists() and tracer.read_text().strip() != "nop":
            raise RuntimeError("New private instance unexpectedly has an active function tracer")
        for name in ("set_event_pid", "set_event_notrace_pid"):
            if (self.instance / name).exists():
                self.write(name, "")
        self.write("tracing_cpumask", cpu_mask(self.cpus))
        self.write("buffer_size_kb", 64)
        for cpu in self.cpus:
            self.write(f"per_cpu/cpu{cpu}/buffer_size_kb", 32768 if self.probes else 8192)

        # Enable all scheduler events on the selected CPUs, including competitors.
        # IRQ events must not be filtered by the camera PID: they may interrupt idle
        # or a different task and still explain the camera's delayed dispatch.
        for required in ("sched/sched_switch", "sched/sched_wakeup",
                         "irq/irq_handler_entry", "irq/irq_handler_exit",
                         "irq/softirq_entry", "irq/softirq_exit"):
            if not (self.instance / "events" / required / "enable").is_file():
                raise RuntimeError(f"Required kernel event unavailable: {required}")
        groups = ["sched", "irq", "irq_vectors"]
        if self.extra_events:
            groups += ["timer", "workqueue"]
        for group in ("sched", "irq", "irq_vectors", "timer", "workqueue"):
            available = sorted((self.instance / "events" / group).glob("*/enable"))
            self.event_groups[group] = {"available": bool(available),
                                        "enabled": bool(available) and group in groups}
            if group in groups:
                for event in available:
                    self.write(str(event.relative_to(self.instance)), 1)
                    self.events.append(str(event.parent.relative_to(self.instance / "events")))
        if self.probes is not None:
            self.setup_edt()
        self.write("trace", "")

    def profile(self):
        """Retain the global profile and only this invocation's uniquely named rows."""
        raw = (self.tracefs / "kprobe_profile").read_text()
        events = {name.split("/", 1)[1] for name, _ in self.probes.attempted}
        rows = [line for line in raw.splitlines()
                if line.split() and line.split()[0].split("/")[-1] in events]
        return {"raw": raw, "owned_rows": rows, "anchor": anchor()}

    def setup_edt(self):
        """Register eight disabled wrapper probes, then enable only their private-instance controls."""
        options = (self.instance / "trace_options").read_text().split()
        if not {"context-info", "irq-info"}.issubset(options):
            raise RuntimeError("EDT pairing requires existing context-info and irq-info trace options")
        self.probes.guard_group()
        for function, argument in (("edt_start_intr_critical", "lock"),
                                   ("edt_end_intr_critical", "lock"),
                                   ("edt_wake_up", "queue"),
                                   ("edt_wake_up_interruptible", "queue")):
            self.probes.register(function, "p", fetchargs=f"{argument}=%di:x64")
            self.probes.register(function, "r", maxactive=64)
        self.edt["definitions"] = dict(self.probes.attempted)
        for name, _ in self.probes.attempted:
            self.edt["formats"][name] = (self.instance / "events" / name / "format").read_text()
        self.edt["profile_before"] = self.profile()
        self.edt["activation"]["begin"] = anchor()
        for name, _ in self.probes.attempted:
            self.write(f"events/{name}/enable", 1)
            if (self.instance / "events" / name / "enable").read_text().strip() != "1":
                raise RuntimeError(f"Could not confirm private EDT event enable: {name}")
            self.events.append(name)
        self.edt["activation"]["end"] = anchor()

    def close_edt(self):
        """Disable and verify every owned event before removing any global definition."""
        errors = []
        self.edt["definitions"] = dict(self.probes.attempted)
        self.edt["deactivation"]["begin"] = anchor()
        try:
            definitions = self.probes.definitions()
        except (OSError, RuntimeError) as error:
            errors.append(f"Could not reconcile EDT definitions: {error}")
            definitions = None
        for name, definition in self.probes.attempted:
            if definitions is None:
                continue
            if name not in definitions:
                continue
            try:
                if definitions[name] != [definition]:
                    raise RuntimeError("Definition changed; refusing to change its enable state")
                self.write(f"events/{name}/enable", 0)
                if (self.instance / "events" / name / "enable").read_text().strip() != "0":
                    raise RuntimeError("Private disable readback is not zero")
            except (OSError, RuntimeError) as error:
                errors.append(f"Could not confirm EDT event disabled {name}: {error}")
            if name not in self.edt["formats"]:
                try:
                    self.edt["formats"][name] = (self.instance / "events" / name / "format").read_text()
                except OSError as error:
                    errors.append(f"Could not retain EDT event format {name}: {error}")
        if not errors:
            self.edt["deactivation"]["end"] = anchor()
            try:
                self.edt["profile_after"] = self.profile()
            except OSError as error:
                errors.append(f"Could not read final EDT profile: {error}")
        try:
            self.edt_helper.check_identity()
            self.edt["identity_after_stop_verified"] = True
        except (OSError, RuntimeError, ValueError) as error:
            self.edt["identity_after_stop_verified"] = False
            errors.append(f"EDT identity after stop failed: {error}")
        self.edt["safety_errors"].extend(errors)
        return errors

    def remove_edt(self, disabled):
        """Remove reconciled definitions only after their private disable has been confirmed."""
        errors = []
        if disabled:
            self.probes.cleanup()
            errors.extend(self.probes.errors)
        try:
            definitions = self.probes.definitions()
            self.edt["remaining_definitions"] = [name for name, _ in self.probes.attempted
                                                 if name in definitions]
        except (OSError, RuntimeError) as error:
            errors.append(f"Could not list remaining EDT definitions: {error}")
            self.edt["remaining_definitions"] = [name for name, _ in self.probes.attempted]
        if self.edt["remaining_definitions"]:
            errors.append("Retained EDT definitions: " + ", ".join(self.edt["remaining_definitions"]))
        self.edt["safety_errors"].extend(errors)
        return errors

    def save(self):
        """Save stopped buffers, loss counters, and exact event/control definitions."""
        for name in ("trace_clock", "tracing_cpumask", "buffer_size_kb", "buffer_total_size_kb",
                     "trace_options", "set_event_pid", "set_event_notrace_pid"):
            (self.output / f"{name}.txt").write_text(read_if_present(self.instance / name))
        for cpu in self.cpus:
            (self.output / f"cpu{cpu}-stats.txt").write_text(
                read_if_present(self.instance / "per_cpu" / f"cpu{cpu}" / "stats"))
        (self.output / "event-groups.json").write_text(json.dumps(self.event_groups, indent=2) + "\n")
        formats = self.output / "event-formats"
        formats.mkdir(exist_ok=True)
        for event in self.events:
            (formats / f"{event.replace('/', '-')}.txt").write_text(
                read_if_present(self.instance / "events" / event / "format"))
        # A stopped trace is read once after collection; no live reader or disk
        # writer is competing with the camera during the measurement interval.
        (self.output / "trace.txt").write_text((self.instance / "trace").read_text())

    def close(self):
        """Stop private tracing and always reconcile probes, even when saving fails."""
        errors = []
        if not self.created and self.probes is None:
            return errors
        disabled = False
        saved = False
        if self.created:
            try:
                self.stop_anchors = {"begin": anchor()}
                self.write("tracing_on", 0)
                self.stop_anchors["end"] = anchor()
            except OSError as error:
                errors.append(f"Could not stop {self.instance}: {error}")
        if self.probes is not None:
            probe_errors = self.close_edt()
            errors.extend(probe_errors)
            disabled = "end" in self.edt["deactivation"]
        if self.created:
            if self.probes is None:
                native_controls = ["events/enable"]
            else:
                native_controls = [f"events/{event}/enable" for event in self.events
                                   if not event.startswith(self.probes.group + "/")]
            for control in native_controls:
                try:
                    self.write(control, 0)
                except OSError as error:
                    errors.append(f"Could not disable {control} in {self.instance}: {error}")
            try:
                self.save()
                saved = True
            except OSError as error:
                errors.append(f"Could not save trace; preserving {self.instance}: {error}")
        if self.probes is not None:
            # Definitions provide the decoder for retained dynamic-event records.
            # Keep them with any buffer whose stopped contents were not saved.
            removable = disabled and (not self.created or (saved and "end" in self.stop_anchors))
            errors.extend(self.remove_edt(removable))
        if errors:
            if self.created:
                errors.append(f"Preserving {self.instance}: cleanup/save was not confirmed")
            return errors
        if self.created:
            try:
                self.remove_instance()
            except OSError as error:
                errors.append(f"Could not remove stopped instance {self.instance}: {error}")
        return errors


def metadata(tid, cpus):
    """Record identity, scheduler placement, kernel, topology, and interrupt counts."""
    proc = Path("/proc")
    status = read_if_present(proc / str(tid) / "status")
    match = re.search(r"^Tgid:\s+(\d+)", status, re.MULTILINE)
    pid = match.group(1) if match else str(tid)
    result = {"tid": tid, "pid": int(pid), "cpus": cpus,
              "kernel": platform.uname()._asdict(), "anchor": anchor(),
              "boot_id": read_if_present(proc / "sys/kernel/random/boot_id").strip(),
              "thread_status": status,
              "threads": command_output(["ps", "-L", "-p", pid, "-o",
                                         "pid,tid,stat,cls,rtprio,pri,psr,wchan:32,comm"]),
              "interrupts": read_if_present(proc / "interrupts"),
              "softirqs": read_if_present(proc / "softirqs"),
              "affinity": command_output(["taskset", "-apc", pid]),
              "irq_affinities": {}, "cpu_topology": {}}
    for irq in sorted((proc / "irq").glob("[0-9]*")):
        result["irq_affinities"][irq.name] = {
            name: read_if_present(irq / name).strip()
            for name in ("smp_affinity_list", "effective_affinity_list")}
    for cpu in cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        result["cpu_topology"][str(cpu)] = {
            name: read_if_present(topology / name).strip()
            for name in ("physical_package_id", "core_id", "thread_siblings_list", "core_siblings_list")}
    return result


def interrupted(signum, frame):
    """Let ordinary cleanup stop and save the private trace on Ctrl-C or TERM."""
    raise KeyboardInterrupt


def main():
    """Collect operator-requested metadata and a short private kernel trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tid", type=int, required=True, help="Current camwfs frameGrabber TID")
    parser.add_argument("--cpus", help="CPU list; default is the thread's allowed CPU list (maximum four)")
    parser.add_argument("--seconds", type=float, default=5, help="Capture duration, 1–30 seconds (default 5)")
    parser.add_argument("--settle-seconds", type=float, default=5,
                        help="Wait after metadata/setup with tracing off, 0–60 seconds (default 5)")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not exist")
    parser.add_argument("--extra-events", action="store_true", help="Also enable available timer/workqueue events")
    parser.add_argument("--edt-wrappers", action="store_true",
                        help="Capture the verified EDT wrapper probes (maximum 5 seconds)")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Run manually with sudo; tracefs instance creation requires root")
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 30:
        parser.error("--seconds must be finite and between 1 and 30")
    if args.edt_wrappers and args.seconds > 5:
        parser.error("--edt-wrappers limits --seconds to 1 through 5")
    if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 60:
        parser.error("--settle-seconds must be finite and between 0 and 60")
    status = read_if_present(Path(f"/proc/{args.tid}/status"))
    match = re.search(r"^Cpus_allowed_list:\s+(\S+)", status, re.MULTILINE)
    if not match:
        parser.error("Camera TID does not exist or its CPU affinity is unavailable")
    try:
        cpus = cpu_list(args.cpus or match.group(1))
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= len(cpus) <= 4:
        parser.error("Select one to four CPUs with --cpus for this bounded capture")
    if not set(cpu_list(match.group(1))).issubset(cpus):
        parser.error("Capture CPUs must include every CPU allowed for the camera TID")
    tracefs = next((path for path in (Path("/sys/kernel/tracing"), Path("/sys/kernel/debug/tracing"))
                    if (path / "instances").is_dir()), None)
    if tracefs is None:
        parser.error("No mounted tracefs instance directory; see the operator guide")

    if any(not Path(f"/sys/devices/system/cpu/cpu{cpu}").is_dir() for cpu in cpus):
        parser.error("One or more selected CPUs do not exist")

    helper = None
    if args.edt_wrappers:
        helper = edt_probe_helper()
        try:
            helper.preflight(tracefs)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    args.output.mkdir(mode=0o755)
    capture = TraceCapture(tracefs, args.output, cpus, args.extra_events, helper)
    details = {"duration_requested_s": args.seconds, "settle_requested_s": args.settle_seconds,
               "tracefs": str(tracefs), "instance": str(capture.instance),
               "edt_wrappers": capture.edt, "capture_started": False,
               "capture_completed": False, "phase_anchors": {}}
    phases = details["phase_anchors"]
    exit_status = 0
    measurement_complete = False
    phase = "metadata_before"
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        phases[phase] = {"begin": anchor()}
        details["before"] = metadata(args.tid, cpus)
        phases[phase]["end"] = anchor()
        (args.output / "metadata.json").write_text(json.dumps(details, indent=2) + "\n")
        print(f"Private instance: {capture.instance}", flush=True)

        phase = "setup"
        phases[phase] = {"begin": anchor()}
        capture.setup()
        phases[phase]["end"] = anchor()
        details["enabled_events"] = capture.events
        details["event_groups"] = capture.event_groups
        print(f"Settling {args.settle_seconds:g} s with tracing off, then capturing CPUs {cpus} "
              f"for {args.seconds:g} s; output: {args.output}", flush=True)
        (args.output / "metadata.json").write_text(json.dumps(details, indent=2) + "\n")

        phase = "settle"
        phases[phase] = {"begin": anchor()}
        time.sleep(args.settle_seconds)
        phases[phase]["end"] = anchor()

        # Defer console, metadata and output-file I/O until capture has stopped.
        # Missing phase ends mean incomplete operations; enable/disable anchors
        # bracket the actual control writes.
        phase = "capture_enable"
        phases[phase] = {"begin": anchor()}
        details["capture_begin"] = phases[phase]["begin"]
        capture.write("tracing_on", 1)
        phases[phase]["end"] = anchor()
        details["capture_started"] = True
        phase = "capture"
        time.sleep(args.seconds)
        measurement_complete = True
    except KeyboardInterrupt:
        details["interrupted"] = True
        details["interrupted_phase"] = phase
        exit_status = 130
    except (OSError, RuntimeError) as error:
        details["error"] = str(error)
        exit_status = 1
    finally:
        # Further ordinary signals must not interrupt stop/save/remove cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        details["cleanup_errors"] = capture.close()
        if capture.stop_anchors:
            phases["capture_disable"] = capture.stop_anchors
            if "capture_begin" in details:
                details["capture_end"] = capture.stop_anchors["begin"]
        details["capture_completed"] = measurement_complete and "end" in capture.stop_anchors
        if "error" in details:
            print(f"Capture failed: {details['error']}", flush=True)
        phases["metadata_after"] = {"begin": anchor()}
        details["after"] = metadata(args.tid, cpus)
        phases["metadata_after"]["end"] = anchor()
        final_affinity = re.search(r"^Cpus_allowed_list:\s+(\S+)",
                                   details["after"]["thread_status"], re.MULTILINE)
        details["camera_affinity_still_covered"] = bool(
            final_affinity and set(cpu_list(final_affinity.group(1))).issubset(cpus))
        (args.output / "metadata.json").write_text(json.dumps(details, indent=2) + "\n")
        if details["cleanup_errors"]:
            print("\n".join(details["cleanup_errors"]), flush=True)
            exit_status = 1
    print(f"Saved {args.output}; inspect CPU stats for trace loss before interpreting it", flush=True)
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
