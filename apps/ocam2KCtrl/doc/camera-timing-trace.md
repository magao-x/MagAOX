# Camera excursion and scheduler/IRQ capture

This procedure pairs chronological `[fgtrace]` excursion records with a short
kernel event trace. It is intended for the instrument operator to run manually;
agents must not run it on instrument computers or through an INDI tunnel.
Keep the camera rate, loop settings, CPU placement, and workload unchanged
between the comparison captures.

The camera records stage endpoints with `CLOCK_MONOTONIC`. The helper selects
tracefs `mono`, the matching monotonic clock; it refuses a different clock.
Match the absolute endpoint nanoseconds in `[fgtrace]` to kernel timestamps in
seconds (`endpoint_ns / 1e9`). Use the record's camera TID and session/epoch to
avoid associating a restarted process with an old trace. The telemetry log's
realtime timestamp is when the ordinary app thread emits the record, not when
the camera excursion happened.

## 1. Install and enable chronological records

Rebuild/install `ocam2KCtrl` and `logdump`, including their usual library/schema
dependencies. Add these settings to the active `camwfs` configuration:

```ini
[framegrabber]
stageTiming=true
stageBracketUs=1
stageTraceMax=16
```

Preserve the other framegrabber settings, including `stageExcursionUs` (20 µs
unless already changed). These settings take effect after restarting `camwfs`.
`stageTraceMax` bounds shared ready-plus-pending record retention; it is zero by
default. At most that many records can be emitted in one ordinary update, but
pending neighbor records survive updates rather than receiving a reset quota. A nonzero value requires stage timing and positive timing-window,
capacity, and excursion-threshold settings. The bracket criterion qualifies the
diagnostic attribution; it does not set a scheduler margin or change frame delivery.

Restart `camwfs` using the normal operator procedure. Confirm the current camera
thread and its CPU placement:

```bash
getINDI camwfs.th-framegrabber
```

Copy the reported `pid` value (this is the framegrabber **TID**) into the next
command. The numeric value below is an example; replace it after every restart.

```bash
CAMERA_TID=123456
```

```bash
sudo ps -eLo pid,tid,stat,cls,rtprio,pri,psr,wchan:32,comm | awk -v camera_tid="$CAMERA_TID" 'NR == 1 || $2 == camera_tid'
```

```bash
taskset -pc "$CAMERA_TID"
```

Use the allowed CPU list reported by `taskset` for the capture. If the EDT IRQ
runs on another CPU, include that CPU too. The helper accepts one to four CPUs,
requires every allowed camera CPU to be included, and changes no affinity or
scheduling policy. Its metadata includes all threads in the camera process,
actual priorities, allowed CPUs, selected CPU topology/siblings, and before/after
interrupt counters. A changed affinity, exited TID, or migration outside the
captured set invalidates full CPU coverage; inspect the saved before/after
thread status and `camera_affinity_still_covered` field.

## 2. Capture the telemetry baseline without the kernel helper

Collect about a minute with the chronological records enabled and the kernel
capture helper absent:

```bash
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-trace-off.txt"
```

The timeout normally exits with status 124. Keep the whole file; it contains
`[fgtimes]`, `[fgstage]`, and `[fgtrace]` plus any lifecycle messages.
This baseline includes the cost of the new camera records but excludes this
helper's kernel tracing. Compare its interval/stage statistics and diagnostic
loss counters with the previous `stageTraceMax=0` capture too.

## 3. Take a short kernel capture with overlapping telemetry

Start the same telemetry command in one terminal, using a new output file:

```bash
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-trace-on.txt"
```

In a second terminal at the MagAOX repository root, run the helper. Replace `45`
with the actual CPU list from step 1; retain the current `CAMERA_TID` assignment
in this terminal too. The following invocation captures ten seconds:

```bash
sudo python3 apps/ocam2KCtrl/scripts/capture-camera-timing.py --tid "$CAMERA_TID" --cpus 45 --seconds 10 --output "$HOME/camwfs-kernel-$(date -u +%Y%m%dT%H%M%SZ)"
```

The helper creates a uniquely named tracefs **instance**, uses an 8 MiB buffer
for each selected CPU (small buffers elsewhere), and enables the available
scheduler and IRQ/softirq events on those CPUs. It also enables available x86
`irq_vectors` events for local APIC timer/IPI entry and exit. Availability and
enablement of each event group are saved, including an explicit unavailable
state for kernels without those events. All competing tasks are retained;
there is no PID event filter. IRQs are not necessarily executed in the camera's
PID context. A wakeup emitted on an unselected remote CPU may be absent even when
its target is the camera CPU; the local schedule-in/IRQ history remains available.

Collection is memory-only during the interval. The helper stops its own instance
before saving text, event formats, per-CPU loss statistics, clock/control settings,
and `metadata.json`. It does not change global tracing, trace options, other
instances, CPU/IRQ affinity, or scheduler policy. If a later capture needs timer
callbacks or worker execution, add `--extra-events`; available `timer` and
`workqueue` events then supplement the first capture. Keep that as a separate
comparison because tracing itself can perturb timing.

If the helper reports no mounted tracefs directory, inspect the existing mount:

```bash
findmnt -t tracefs
```

When tracefs is not mounted, the operator can mount the standard location and
rerun the helper:

```bash
sudo mount -t tracefs nodev /sys/kernel/tracing
```

Do not replace an existing tracing mount. Missing required tracepoints or the
`mono` clock are reported as setup failures. The helper does not silently select
another clock or enable a broad function tracer.

## 4. Check completeness and share the capture

Keep the entire new output directory plus both telemetry files. Check:

- `metadata.json`: no capture/cleanup errors, expected TID/boot ID, same CPU
  coverage and settings before/after, and the enabled event list.
- `cpu*-stats.txt`: `overrun`, `commit overrun`, and `dropped events` must all be
  zero. Nonzero values mean the trace is incomplete; shorten the next capture or
  increase buffers deliberately before drawing a negative conclusion.
- `trace.txt`: timestamps cover the excursion endpoints, and both IRQ entry/exit
  and camera schedule-in/out records are present. Check `event-groups.json` for
  APIC vector coverage; device IRQ events alone do not cover every CPU interrupt.
  Events straddling the capture
  boundaries cannot have their full duration reconstructed.
- `[fgstage]`/`[fgtrace]`: examine qualification and all record/drop/omission
  counters. A bounded record list can omit additional candidates; absence from
  that list does not prove that no excursion occurred.

An event can be linked directly by its absolute monotonic timestamp. The before
and after metadata also bracket realtime readings with monotonic readings, record
kernel/boot identity, and allow a coarse association with the telemetry log times.
Trace text may use microsecond display precision; do not infer nanosecond causal
ordering from rounded text timestamps. Keep the recorded trace options and
original endpoint nanoseconds.

The trigger baseline is the causal rolling median of at least three strictly
preceding intervals, rather than the aggregate window median; each record stores
the exact baseline and its sample count. For each excursion, examine its
previous/current/next frame endpoints in order.
An overlapping scheduler switch away from the framegrabber identifies off-CPU
wall time. An IRQ/softirq interval on its CPU identifies interrupt execution.
Neither observation alone establishes an upstream root cause. A long wall-time
segment without these events does not prove extra camera work: missing trace
coverage, driver waits, shared-resource stalls, NMIs/SMIs, and tracing overhead
remain possible. Compare trace-off/on statistics before treating the trace as a
representative baseline.

The helper stops and saves on Ctrl-C or TERM and removes only its own instance.
If saving fails, it leaves that instance stopped and prints its exact path.
SIGKILL, a machine failure, or failure to write a trace control can prevent normal
cleanup. The instance path is printed and saved in the initial `metadata.json`.
For that **exact private instance only**, replace the example suffix below with
the reported name, stop its trace/events, save its buffer, then remove the empty
virtual instance directory:

```bash
sudo sh -c 'printf "0\n" > /sys/kernel/tracing/instances/ocam-timing-123-example/tracing_on; printf "0\n" > /sys/kernel/tracing/instances/ocam-timing-123-example/events/enable'
```

```bash
sudo cat /sys/kernel/tracing/instances/ocam-timing-123-example/trace > "$HOME/camwfs-recovered-trace.txt"
```

```bash
sudo rmdir /sys/kernel/tracing/instances/ocam-timing-123-example
```

Do not clear the global trace or remove another session's instance. After the
measurement, restore `stageTraceMax=0` and restart `camwfs` when convenient;
`stageTiming` can remain enabled for the existing aggregate measurements.

## Implementation checks and references

The helper has only been exercised on an ordinary temporary fake tracefs tree,
including partial setup failure, unavailable `mono`, preservation of unrelated
traces, and preserving a stopped instance when trace saving fails. No instrument
or live kernel trace was accessed by an agent. Run its local checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 apps/ocam2KCtrl/scripts/tests/capture_camera_timing_test.py
```

The Linux kernel's [ftrace documentation](https://docs.kernel.org/trace/ftrace.html)
defines `mono` as `CLOCK_MONOTONIC`, per-CPU masks/buffers and loss statistics, and
separate instance event buffers. It also documents that trace options can affect
other buffers on older kernels, which is why this helper leaves them untouched.
[Event Tracing](https://docs.kernel.org/trace/events.html) describes available
tracepoints, event format files, and PID filtering.

The kernel's [x86 IRQ-vector tracepoint definitions](https://github.com/torvalds/linux/blob/master/arch/x86/include/asm/trace/irq_vectors.h)
provide separate local-timer, reschedule, and function-call vector events when
configured; these supplement device IRQ handlers and do not imply NMI/SMI coverage.
