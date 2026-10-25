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

For eligible multi-buffer OCAM acquisition, rearm now runs at the end of
`frameGrabberPostPublish`, after both the main image and sync notifications.
The added `timeoutEnterNs`/`timeoutReturnNs` endpoints bracket the timeout query
that still runs before publication. `rearmEnterNs`/`rearmReturnNs` bracket
queue replenishment independently. `syncPostDoneNs` is the hook-return marker
after deferred rearm; the `syncPost` population excludes that rearm time.
Single-buffer or single-frame/serial-triggered modes retain immediate rearm.
Existing nine-marker captures remain readable with the updated `logdump`.

For the capture-helper update alone, use the updated Python script and this
procedure; no camera or reader rebuild/restart is needed if the above endpoints
are already present. The helper now defaults to five seconds of settling with
tracing off, followed by five seconds of kernel capture. Metadata and trace setup
finish before settling. This separates setup from the timed interval while the
overlapping telemetry retains events in all phases.

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
pending neighbor records survive updates rather than receiving a reset quota.
A nonzero value requires stage timing and positive timing-window, capacity, and
excursion-threshold settings. The bracket criterion qualifies the diagnostic
attribution; it does not set a scheduler margin or change frame delivery.

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

Let telemetry run for about ten seconds, then use a second terminal at the
MagAOX repository root to run the helper. Keep the telemetry reader running to
its sixty-second timeout, preserving records before setup and after capture.
Set `CAMERA_TID` in this terminal to the current value from step 1 too.
The example below uses the observed acquisition CPU 33 and IRQ CPU 61; replace
the list if their placement has changed. Include every allowed acquisition CPU
and the EDT IRQ CPU.

```bash
sudo python3 apps/ocam2KCtrl/scripts/capture-camera-timing.py --tid "$CAMERA_TID" --cpus 33,61 --settle-seconds 5 --seconds 5 --output "$HOME/camwfs-kernel-$(date -u +%Y%m%dT%H%M%SZ)"
```

Both durations are explicit here and are also the defaults. `--settle-seconds`
accepts 0–60 seconds; `--seconds` accepts 1–30 seconds. Five seconds of capture
fits the previously observed camera-CPU event rate in the current ring, but
always inspect the loss counters afterward. Shorten the capture if necessary.
The settling interval does not consume kernel ring storage because tracing
remains off.

The sequence is:

```text
before metadata -> private trace setup -> settle 5 s -> trace ON for 5 s
                                                       |
after metadata <- save stopped trace <- trace OFF <-----+
```

Clock anchors distinguish metadata collection, setup, settling, trace enable,
trace disable and after-metadata collection in the saved `metadata.json`.
No metadata queries, progress output or trace-file saving occur between the
completed enable and the start of disable. Setup/settling and trace-on/off
boundaries can still perturb timing; compare telemetry around those boundaries
with the interior of the capture. A five-second wait is an experimental control,
not proof that every possible setup effect has disappeared.

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
  coverage and settings before/after, the enabled event list, and completed
  setup/settling/capture phases. `capture_started` and `capture_completed` must
  both be true for an ordinary full capture. Keep setup and boundary events
  separate from the steady capture when comparing timing.
- `cpu*-stats.txt`: `overrun`, `commit overrun`, and `dropped events` must all be
  zero. Nonzero values mean the trace is incomplete; shorten the next capture or
  increase buffers deliberately before drawing a negative conclusion.
- `trace.txt`: timestamps cover the excursion endpoints, and both IRQ entry/exit
  and camera schedule-in/out records are present. Check `event-groups.json` for
  APIC vector coverage; device IRQ events alone do not cover every CPU interrupt.
  Events straddling the capture boundaries cannot have their full duration
  reconstructed.
- `[fgstage]`/`[fgtrace]`: examine qualification and all record/drop/omission
  counters. A bounded record list can omit additional candidates; absence from
  that list does not prove that no excursion occurred.

An event can be linked directly by its absolute monotonic timestamp. The before
and after metadata also bracket realtime readings with monotonic readings, record
kernel/boot identity, and allow a coarse association with the telemetry log times.
Trace text may use microsecond display precision; do not infer nanosecond causal
ordering from rounded text timestamps. Keep the recorded trace options and
original endpoint nanoseconds.

`phase_anchors` contains `metadata_before`, `setup`, `settle`, `capture_enable`,
`capture_disable` and `metadata_after` entries as each phase is reached. Each
entry brackets that phase with `begin`/`end` clock anchors; a missing end marks
an incomplete phase. Each anchor includes `monotonic_before_ns`, `realtime_ns`
and `monotonic_after_ns`. Use monotonic bounds directly for `[fgtrace]`
association. `capture_enable` and `capture_disable` bracket the control writes,
rather than treating a progress message as the instant tracing changes.
An excursion wholly between enable's end and disable's begin is inside the
enabled interval, subject to the retained kernel coverage. An excursion within
either switch bracket has uncertain boundary ordering.

The legacy `capture_begin` and `capture_end` anchors remain for existing analysis
tools: they correspond to enable's begin and disable's begin, respectively.
Neither is present if setup/settling stopped before enable was attempted.
`capture_started` requires a successful enable write; `capture_completed`
requires the requested capture wait to finish and a successful stop write.
They do not qualify later save/cleanup errors. `interrupted_phase` identifies
where Ctrl-C/TERM occurred. Prefer the explicit phase brackets for new analysis,
and inspect interruption/error status before assuming a complete capture. Telemetry events
before enable have no kernel coverage in this capture; sharing those events
still helps identify whether the disturbance coincides with metadata, setup or
settling. Keep all chronological records, including events outside the kernel
window and stage-summary windows.

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

## 5. Check EDT driver probe registration

For the next driver investigation, use the separate registration-only helper
after matching the loaded EDT module to the copied build. The current check is
specific to kernel `5.14.0-362.8.1.el9_3.x86_64`, EDT `5.6.8.0`, source version
`A26E0B4FD875E93168B83BC`, and build ID
`97857cff4fc9d75717d3b58e9c5bc3359a7204f0`. It verifies that identity before
making changes and again after the registration attempts.

From the MagAOX repository root, the operator runs:

```bash
sudo python3 apps/ocam2KCtrl/scripts/check-edt-probes.py
```

Share the complete output. The helper tests entry and return definitions for
the ten selected EDT functions, using module-qualified names without instruction
offsets or argument reads. Each event is created under a unique group and
checked to be disabled. Definitions are appended to `kprobe_events` and only
the helper's definitions are selectively removed; the existing tracing controls
and buffers are left alone.

This is a registration check, with no timed collection or active event enabling.
Run it separately from a measurement capture. Success establishes registration
eligibility; actual probe hits, return misses, argument layout and active probe
overhead still need qualification in the subsequent capture. Rejections and
cleanup failures are reported separately. Ctrl-C/TERM requests normal cleanup;
SIGKILL or machine failure can prevent cleanup. If definitions remain, retain
the printed names and errors; do not clear the global `kprobe_events` file.

On the verified exao2 kernel, the observed result is **20 attempted, 12 errors**:
entry/return definitions for the four lock/wakeup wrappers succeed and are
removed; the six proprietary-core functions reject both definitions. Follow-up
kernel warnings identify the `notrace` restriction. The active configuration
has `CONFIG_DYNAMIC_FTRACE=y` and unset `CONFIG_KPROBE_EVENTS_ON_NOTRACE`,
which prevents these uninstrumented functions from being used as tracefs
probe events. Treat this as partial capability discovery; inspect each
registration and cleanup result separately from the aggregate error count.
The [engineering record](../../../agents/plans/2026-10/ocam2k-acquisition-publication-timing.md)
records the accepted targets and the revised measurement scope.

## 6. Capture the accepted EDT lock and wakeup wrappers

Use the updated `capture-camera-timing.py` **and** `check-edt-probes.py` from
the same checkout. This adds Python helper functionality; it needs no camera,
reader or kernel rebuild/restart. Preserve the camera's chronological telemetry
configuration from step 1 and use its current framegrabber TID. The examples
retain acquisition CPU 33 and EDT IRQ CPU 61; adjust them if placement changed.

First collect a native scheduler/IRQ comparison. In one terminal:

```bash
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-edt-native.txt"
```

After about ten seconds, run this in another terminal at the repository root,
with `CAMERA_TID` set as in step 1:

```bash
sudo python3 apps/ocam2KCtrl/scripts/capture-camera-timing.py --tid "$CAMERA_TID" --cpus 33,61 --settle-seconds 5 --seconds 1 --output "$HOME/camwfs-native-$(date -u +%Y%m%dT%H%M%SZ)"
```

Let that telemetry collection finish. Then collect a second telemetry file:

```bash
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-edt-wrappers.txt"
```

Again wait about ten seconds, then request the wrapper capture:

```bash
sudo python3 apps/ocam2KCtrl/scripts/capture-camera-timing.py --tid "$CAMERA_TID" --cpus 33,61 --edt-wrappers --settle-seconds 5 --seconds 1 --output "$HOME/camwfs-edt-$(date -u +%Y%m%dT%H%M%SZ)"
```

Keep both telemetry files and both complete output directories. Include a
helper-free telemetry baseline from step 2 too. Start with one second of active
capture; inspect timing spread and loss before lengthening it. Wrapper mode
accepts 1–5 seconds and allocates 32 MiB per selected CPU, with at most four
CPUs. Native mode retains its 8 MiB buffers and 1–30 second range. The comparison
therefore includes the mode's buffer allocation as well as probe overhead;
phase anchors separate setup from the retained interval.

The mode uses only the four functions whose disabled entry/return registration
already succeeded:

| Wrapper | Entry field | Measurement boundary |
| --- | --- | --- |
| `edt_start_intr_critical` | `lock` pointer | Entry/return bracket lock acquisition |
| `edt_end_intr_critical` | `lock` pointer | Unlock entry/return; compare with the matching acquisition |
| `edt_wake_up` | `queue` pointer | Wakeup call alongside scheduler wake/dispatch events |
| `edt_wake_up_interruptible` | `queue` pointer | Interruptible wakeup call alongside scheduler events |

Entry arguments use the verified x86-64 first-argument register (`%di:x64`),
without dereferencing the pointer. Return probes reserve 64 concurrent instances
each (`r64`). A unique group and unique event names identify each invocation.
Definitions are checked while disabled, then enabled only in the private
instance. Setup rejects missing `context-info` or `irq-info`; trace options are
preserved. Rejected proprietary-core targets are not requested by this mode.

The wrappers become active during setup and remain active through settling,
even though the ring is off until capture begins. They execute for calls on
CPUs outside the recording mask too. `metadata.json`'s `edt_wrappers` object
records definitions, event formats, activation/deactivation anchors, raw and
owned-row profile snapshots, expected build identity, and any retained names
or cleanup errors. Profile snapshots bracket activation through deactivation,
including settling; they do not describe only the one-second ring interval.
No diagnostic or output-file I/O runs between trace enable and stop.

Inspect the ordinary completion/affinity/error fields and per-CPU loss counters
from step 4. Also require confirmed probe deactivation, no safety errors and
no remaining definitions for normal cleanup. Preserve `trace_options.txt` and
event formats: later pairing needs CPU, task, interrupt context, function and
pointer identity. PID alone cannot distinguish calls made from idle/IRQ
context. Match complete call sequences and exclude capture-boundary fragments.

Treat the lock acquisition bracket as instrumented wrapper elapsed time. The
time from acquisition return to unlock entry covers the interior held region,
not its complete duration. Nested calls, missing returns or an owner outside
the selected CPUs limit attribution. Profile miss counters are saved, but
`return_pool_misses_verified` remains false: older kernel implementations omit
return-slot exhaustion from that column, and distro backports differ. Zero
misses alone cannot establish complete entry/return pairing.

On a successful save, the helper disables its events, removes its definitions
and removes its private instance. If saving fails, it retains the stopped
instance and disabled definitions so the dynamic records can still be decoded.
If stopping or disabling is unconfirmed, it retains both and reports the exact
paths/names. Keep these until the trace and formats have been recovered; deleting
dynamic definitions first can remove their formatter. The generic recovery
steps in step 4 must therefore be completed before selectively removing any
retained EDT definitions. Share the errors and retained names with the capture.

## 7. Compare normal polling with a temporary housekeeping pause

The diagnostic housekeeping pause suppresses automatic serial `temp` and `gain`
queries, and `fps` queries when synchronization is off. It leaves acquisition,
FSM processing, timing telemetry and INDI servicing running. With synchronization
on, FPS follows the external timing updates without a serial query.

Rebuild/install `ocam2KCtrl` for this mode and restart `camwfs` using the normal
operator procedure. It requires no new configuration setting or `logdump`
rebuild. A restart begins with normal housekeeping enabled. Let the camera reach
its temperature setpoint before collecting the comparison, and retain the same
camera/loop settings, CPU/IRQ placement and workload throughout.

The runtime interface is:

| Property | Elements and meaning |
| --- | --- |
| `camwfs.housekeeping_pause` | Writable `toggle`: On requests a pause; Off resumes normal polling. Repeating On does not extend an existing request. |
| `camwfs.housekeeping` | Read-only `remainingSec` and fixed `timeoutSec=120`; `pauseStartMonoSec` and `pauseEndMonoSec` retain the latest actual pause boundaries in monotonic seconds. `tempAgeSec`, `gainAgeSec`, `fpsAgeSec` give readback ages (`-1` before a valid reply, after power-off, or after an invalid reply/failed serial attempt). `skippedTemp`, `skippedGain`, `skippedFps` count suppressed polls; `serialTransactions` counts attempted serial transactions. All counters accumulate for the app process lifetime. |
| `camwfs.housekeeping_status` | Read-only `state` and `protectionPolling` are `paused`, `polling`, or `inactive`; the latter applies outside stable powered READY/OPERATING states. `reason` identifies activation or why polling resumed. `temperature`, `gain`, `fps` report `stale`, `current` or `unavailable`; FPS reports `external sync` when synchronization is on. |

The quiet interval starts only after the current serial transaction and its
trailing-byte drain finish. Check `housekeeping_status` for actual activation;
the switch request alone is not the quiet boundary. The callback activates the
pause synchronously once that transaction has finished. The active pause expires
after 120 seconds and resumes polling without replaying skipped queries.
Temperature/gain values remain cached and are explicitly marked stale during
the pause. Resuming does not make cached values current: they remain stale
until successful post-pause queries refresh them. Gain queries can recognize
the camera's `HV` response, so that
software protection-trip polling is paused too. An explicit serial command,
reconfiguration, or power/error transition ends the pause before further serial
I/O; include such an interruption in the capture notes.

First inspect the normal-polling status with no kernel helper running:

```bash
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
```

Require `state=polling`, current temperature/gain readbacks and refreshing ages;
FPS should be current or report `external sync`. Then collect the baseline:

```bash
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-housekeeping-normal-before.txt"
```

Request the pause and inspect its state:

```bash
setINDI -s camwfs.housekeeping_pause.toggle=On
getINDI camwfs.housekeeping_pause
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
```

Once `housekeeping_status.state=paused`, allow about five seconds for effects near
the activation edge to pass. Confirm that at least 65 seconds remain, then take
the 60-second telemetry set:

```bash
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-housekeeping-paused.txt"
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
```

Retain these property outputs with the telemetry. During the interior quiet
interval, `serialTransactions` must remain unchanged while skipped-poll counts
increase. A serial transaction or pause interruption makes that interval a
different comparison; keep its state/reason and timing rather than treating it
as uninterrupted quiet operation. If the remaining time is insufficient, resume
with Off and make a fresh On request before starting another capture.

Resume explicitly after the paused capture and inspect the status:

```bash
setINDI -s camwfs.housekeeping_pause.toggle=Off
getINDI camwfs.housekeeping_status
getINDI camwfs.housekeeping
```

Wait for `state=polling`, successful post-pause temperature/gain queries to mark
their readbacks current, and serial transactions/ages to resume updating. FPS
should be current or report `external sync`. Once refreshed, collect the final
baseline:

```bash
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-housekeeping-normal-after.txt"
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
```

Verify serial transactions resume and readback ages refresh. Keep all three
telemetry files, the property outputs and any lifecycle messages. The mode also
resumes automatically on its deadline if an explicit Off is omitted.

Start with this helper-free comparison. If kernel attribution is needed, repeat
the comparison with the one-second native or wrapper capture from section 6
wholly inside the confirmed active pause, preserving setup/settling time and a
matching normal-polling capture. The probes add timing cost, so keep that
comparison distinct from the helper-free telemetry. A quiet interval does not
establish a safe production polling schedule: serial replies and their EDT
interrupts need not occur in the frame in which a command was issued.

## Implementation checks and references

The helper's local checks use an ordinary temporary fake tracefs tree, including
partial setup failure, unavailable `mono`, phase ordering and interruption,
preservation of unrelated traces, and preserving a stopped instance when trace
saving fails. Agents have accessed no instrument or live kernel trace. Operator
captures and their qualifications are recorded in the
[OCAM engineering record](../../../agents/plans/2026-10/ocam2k-acquisition-publication-timing.md).
Run local checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 apps/ocam2KCtrl/scripts/tests/capture_camera_timing_test.py
```

The EDT registration helper is checked separately with fake controls and
injected registration/cleanup failures, without accessing a live tracefs:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 apps/ocam2KCtrl/scripts/tests/check_edt_probes_test.py
```

Its offline checks cover ownership, append-only writes, short/failed writes,
interruption, output failure, changed definitions, external enablement and
module/mount identity rejection. These do not exercise the actual kernel's
registration path; that result comes from the operator check above.

The Linux 5.14 [Kprobe event documentation](https://www.kernel.org/doc/html/v5.14/trace/kprobetrace.html)
defines module-qualified registration, selective deletion and the default
disabled state of newly defined events.

The Linux kernel's [ftrace documentation](https://docs.kernel.org/trace/ftrace.html)
defines `mono` as `CLOCK_MONOTONIC`, per-CPU masks/buffers and loss statistics, and
separate instance event buffers. It also documents that trace options can affect
other buffers on older kernels, which is why this helper leaves them untouched.
[Event Tracing](https://docs.kernel.org/trace/events.html) describes available
tracepoints, event format files, and PID filtering.

The kernel's [x86 IRQ-vector tracepoint definitions](https://github.com/torvalds/linux/blob/master/arch/x86/include/asm/trace/irq_vectors.h)
provide separate local-timer, reschedule, and function-call vector events when
configured; these supplement device IRQ handlers and do not imply NMI/SMI coverage.
