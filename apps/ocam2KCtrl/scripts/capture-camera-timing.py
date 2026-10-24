#!/usr/bin/env python3
## \file capture-camera-timing.py
# \brief Operator-run private scheduler/IRQ capture for camera excursion correlation.

"""Capture a bounded camera-CPU scheduler/IRQ trace in a private tracefs instance.

Run manually by the instrument operator. This script never changes scheduling,
CPU affinity, IRQ affinity, camera settings, or any pre-existing trace buffer.
"""

import argparse
import json
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


class TraceCapture:
    """Own exactly one new tracefs instance and its output directory."""

    def __init__(self, tracefs, output, cpus, extra_events=False):
        """Own the new instance below tracefs, its output path, CPU set and optional event groups."""
        self.tracefs = tracefs
        self.output = output
        self.cpus = cpus
        self.extra_events = extra_events
        self.instance = tracefs / "instances" / f"ocam-timing-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.created = False
        self.events = []
        self.event_groups = {}

    def create_instance(self):
        """Create a unique kernel-owned instance; an existing name is an error."""
        self.instance.mkdir()
        self.created = True

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
            self.write(f"per_cpu/cpu{cpu}/buffer_size_kb", 8192)

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
        self.write("trace", "")

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
        """Stop and disable this instance, preserving its buffer if saving fails."""
        errors = []
        if not self.created:
            return errors
        try:
            self.write("tracing_on", 0)
        except OSError as error:
            errors.append(f"Could not stop {self.instance}: {error}")
        try:
            self.write("events/enable", 0)
        except OSError as error:
            errors.append(f"Could not disable events in {self.instance}: {error}")
        if errors:
            errors.append(f"Preserving {self.instance}: stop/event disable was not confirmed")
        try:
            self.save()
        except OSError as error:
            errors.append(f"Could not save trace; preserving {self.instance}: {error}")
            return errors
        if errors:
            return errors
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
    parser.add_argument("--seconds", type=float, default=10, help="Capture duration, 1–30 seconds (default 10)")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not exist")
    parser.add_argument("--extra-events", action="store_true", help="Also enable available timer/workqueue events")
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("Run manually with sudo; tracefs instance creation requires root")
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be between 1 and 30")
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

    args.output.mkdir(mode=0o755)
    capture = TraceCapture(tracefs, args.output, cpus, args.extra_events)
    details = {"duration_requested_s": args.seconds, "tracefs": str(tracefs),
               "instance": str(capture.instance), "before": metadata(args.tid, cpus)}
    (args.output / "metadata.json").write_text(json.dumps(details, indent=2) + "\n")
    print(f"Private instance: {capture.instance}", flush=True)
    exit_status = 0
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        capture.setup()
        details["enabled_events"] = capture.events
        details["event_groups"] = capture.event_groups
        details["capture_begin"] = anchor()
        print(f"Capturing CPUs {cpus} for {args.seconds:g} s; output: {args.output}", flush=True)
        capture.write("tracing_on", 1)
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        details["interrupted"] = True
        exit_status = 130
    except (OSError, RuntimeError) as error:
        details["error"] = str(error)
        print(f"Capture failed: {error}", flush=True)
        exit_status = 1
    finally:
        # Further ordinary signals must not interrupt stop/save/remove cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        details["capture_end"] = anchor()
        details["cleanup_errors"] = capture.close()
        details["after"] = metadata(args.tid, cpus)
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
