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

## 7. Configure confirmation-driven housekeeping

`ocam2KCtrl` confirms gain and FPS after startup or an applicable setting change,
then retains their confirmed values without periodic serial queries. Temperature
polling continues, with a faster interval while a setting or thermal transition
is in progress. Acquisition, timing telemetry and INDI servicing continue on
ordinary app-loop passes even when no serial query is due. This policy reduces
serial traffic; it does not schedule serial replies between camera frames.

Rebuild/install `ocam2KCtrl` and restart `camwfs` using the normal operator
procedure. No `logdump` rebuild or new telemetry schema is needed for this policy.
The defaults need no configuration entries. To override them, use:

```ini
[housekeeping]
tempIntervalSec=10
tempChangingIntervalSec=1
confirmationIntervalSec=1
confirmationTimeoutSec=10
tempSettleSamples=2
```

| Setting | Meaning |
| --- | --- |
| `tempIntervalSec` | Idle temperature interval in seconds: default 10 s, or 0.1 Hz. |
| `tempChangingIntervalSec` | Temperature interval during a change or physical convergence: default 1 s. |
| `confirmationIntervalSec` | Interval between pending setting readbacks: default 1 s. |
| `confirmationTimeoutSec` | Fixed deadline for confirming a setting or establishing an initial observation: default 10 s. This is not a physical cooling deadline. |
| `tempSettleSamples` | Consecutive fresh, valid settled readings required before returning to the idle interval: default two. |

Intervals must be finite, positive and representable as monotonic nanoseconds.
The changing interval cannot exceed the idle interval, the confirmation timeout
must be at least the confirmation interval, and the settling count must be
positive. The ordinary app loop and serial response waits determine when due
work is serviced: a one-second setting
does not promise an exact one-Hz readback rate. At most one automatically
scheduled serial query runs per app-loop pass. The oldest due operation wins;
confirmation takes precedence on an initial tie. Queries reschedule from actual
service, with no catch-up bursts. Explicit setters and startup/configuration
sequences can still issue their required ordered commands.

```text
new valid request -> send setting once -> bounded readback confirmation
                                               |
                          +--------------------+--------------------+
                          |                                         |
                     confirmed                                deadline expires
                          |                                         |
            gain/FPS: retain cached value                 unconfirmed/error;
            temp: monitor physical settling               stop request retries
                          |
                 settled fresh samples
                          |
              idle temperature interval
```

Transport success alone does not confirm a setting. A setter reply confirms
only when it contains a valid matching report in the supported numeric
query-reply format; otherwise the application uses bounded getter readbacks.
Setter-specific acknowledgement syntax is not yet hardware qualified. A mismatch
or invalid reply cannot extend a request's monotonic deadline, and readback
retries never resend its setting command.
After expiry, a new explicit request or lifecycle recovery can start a new
confirmation. Current values and requested targets remain distinct; a reported
temperature setpoint cannot overwrite a newer pending target.

For EM gain, confirmation compares the validated integer actually sent to the
camera; existing bounds and protection-reset interlocks still apply. For
internal FPS, `fps=0` remains the automatic/maximum-rate request and confirms
against a positive actual rate. Mode and synchronization setup confirm their
final effective setting, rather than an intermediate `fps 0` command. With
synchronization on, FPS follows the existing external frequency subscription
without serial FPS queries. Numeric FPS comparison allows only floating-point
representation tolerance; it does not assume a firmware rounding rule. A rounded
report outside that tolerance remains unconfirmed and needs hardware/protocol
qualification. External reports confirm the timing-source setting; the camera
acquisition telemetry remains the measurement of delivered rate.

Confirmed gain/FPS values are intentionally cached. Their readback ages can grow
indefinitely without making the confirmation stale. There is **no background
EM-protection-trip polling**: `protectionPolling=confirmation only` describes
this policy. An `HV` response that is actually received still follows the
existing protection handling, and the protection-reset controls are unchanged.

Temperature uses the existing `abs(CCD - SET) < 1 C` settling criterion. A new
setpoint/cooling command or an enabled off-target idle reading enters faster
monitoring. Confirmation of the reported setting is separate from the CCD
reaching its target; confirmed cooling can legitimately remain in the changing
phase for minutes. Invalid/off-target readings reset the consecutive settling
count. Confirmed disabled cooling returns to the idle interval even when the CCD
differs from an inactive stored setpoint. A malformed or unknown report after a
previously confirmed observation begins bounded fast observation checks. A
failed or expired setting request retains slow temperature monitoring; a later
valid observation does not silently acknowledge that expired request.

The existing reply exposes cooling power rather than an explicit control-state
field. The application treats reported cooling power >= 5 as On, and power < 5
with `CCD - SET > 2.99 C` as Off. Other combinations are Unknown; low power near
the setpoint alone does not prove that cooling is enabled or disabled. Unknown
state receives bounded confirmation attempts, then returns to idle polling with
`tempConfirmation=unconfirmed/error` if the deadline expires. These inference
thresholds and the actual firmware responses still require operator
qualification.

Cooling Off always obtains a fresh valid temperature before applying the
existing CCD > 19 C interlock, even if the diagnostic pause was never used.
The query, decision and command remain serialized under the camera mutex.
An invalid refresh cannot authorize cooling Off. Power loss, recovery and
mode/synchronization changes invalidate affected observations; gain is never
reapplied automatically after a reset or trip.

The runtime diagnostics are:

| Property | Elements and meaning |
| --- | --- |
| `camwfs.housekeeping_pause` | Writable `toggle`: On requests the bounded diagnostic pause; Off resumes scheduled housekeeping. Repeating On does not extend an existing pause. |
| `camwfs.housekeeping` | Read-only numeric diagnostics described below. Counts accumulate for the app process lifetime. |
| `camwfs.housekeeping_status` | `state` is `idle`, `changing`, `paused` or `inactive`. `temperature` is `current`, `stale` or `unavailable`; `gain`, `fps` and `tempConfirmation` are `unavailable`, `confirming`, `confirmed` or `unconfirmed/error`. `reason` records the last pause transition/refusal. `protectionPolling` is `confirmation only`, `paused` or `inactive`. |

`housekeeping` contains:

| Elements | Meaning |
| --- | --- |
| `tempAgeSec`, `gainAgeSec`, `fpsAgeSec` | Seconds since the last valid corresponding report; −1 means none since startup/invalidation or the latest failed/invalid attempt. External frequency reports refresh FPS age while synchronized. |
| `tempIntervalSec`, `tempNextSec` | Current nominal temperature interval and seconds until its next scheduled read. Zero next-due means due now, rather than a hard sampling promise. |
| `gainDeadlineSec`, `fpsDeadlineSec`, `tempDeadlineSec` | Remaining confirmation time; zero when no confirmation is pending. |
| `tempQueries`, `gainQueries`, `fpsQueries` | Getter attempts; FPS counts only serial queries. The fresh cooling-Off interlock also increments `tempQueries`. |
| `serialTransactions` | All attempted serial transactions, including explicit setting commands. |
| `remainingSec`, `timeoutSec` | Pause time remaining and its fixed 120-second limit. |
| `pauseStartMonoSec`, `pauseEndMonoSec` | Latest actual pause boundaries in monotonic seconds. |
| `skippedTemp`, `skippedGain`, `skippedFps` | Pause suppression counters retained for compatibility; confirmed gain/FPS have no periodic work to suppress. |

Use the housekeeping confirmation/result fields to distinguish pending or failed
settings from cached values in the standard camera properties. Temperature stays
the latest valid sample between scheduled reads. An overdue reading becomes
stale after its expected interval plus a grace of one ordinary loop opportunity
and the existing serial timeout. Emitting camera telemetry from cached values
does not refresh any readback age.

## 8. Compare scheduled housekeeping with a temporary pause

The diagnostic pause suppresses automatically scheduled serial queries for up
to 120 seconds. It leaves acquisition, FSM processing, timing telemetry and INDI
servicing running. External synchronization reports continue without serial
queries. A new pause is refused while a setting is unavailable/confirming or a
thermal change is pending. Terminal unconfirmed/error states do not themselves
prevent an idle diagnostic pause; retain them in the capture record. Keep
camera/loop settings, CPU/IRQ placement and workload unchanged throughout.

First inspect normal operation with no kernel helper running:

```bash
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
```

Require confirmed gain/FPS and idle, valid temperature monitoring. At steady
state, gain/FPS query counts should stop increasing; temperature query counts
should increase at the configured idle cadence, subject to app-loop service.
Collect the first baseline:

```bash
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-housekeeping-normal-before.txt"
```

On the instrument, `teldump` is an alias for
`logdump --dir=/opt/MagAOX/telem --ext=.bintel`. The timed commands spell out
`logdump` because a shell alias is not expanded as an argument to `timeout`.

Request the pause and inspect its state:

```bash
setINDI -s camwfs.housekeeping_pause.toggle=On
getINDI camwfs.housekeeping_pause
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
```

The quiet interval starts only after any current serial transaction and its
trailing-byte drain finish. Check `housekeeping_status.state=paused`; the switch
request alone is not the quiet boundary. Allow about five seconds for effects
near activation to pass. Confirm that at least 65 seconds remain, then take the
60-second telemetry set:

```bash
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-housekeeping-paused.txt"
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
```

Retain these property outputs with the telemetry. `serialTransactions` must stay
unchanged within the interior quiet interval. No due operations are queued for
later replay. A serial transaction or pause interruption makes that interval a
different comparison; keep its state/reason and timing. An explicit setting
command, reconfiguration, or power/error transition ends the pause before
further serial I/O. If too little time remains, resume with Off and make a fresh
On request before starting another capture.

Resume explicitly after the paused capture:

```bash
setINDI -s camwfs.housekeeping_pause.toggle=Off
getINDI camwfs.housekeeping_status
getINDI camwfs.housekeeping
```

Resume/expiry schedules one temperature refresh; confirmed gain/FPS remain
quiet. Wait for a valid post-pause temperature read, then collect the final
baseline:

```bash
timeout 60s logdump --dir=/opt/MagAOX/telem --ext=.bintel -f camwfs > "$HOME/camwfs-housekeeping-normal-after.txt"
getINDI camwfs.housekeeping
getINDI camwfs.housekeeping_status
```

Verify temperature polling resumes and gain/FPS query counts remain unchanged.
Keep all three telemetry files, property outputs and lifecycle messages. The
pause also ends automatically at its deadline if Off is omitted.

Operator qualification should additionally exercise gain and FPS changes,
automatic FPS, synchronization changes, temperature/cooling transitions and
power/recovery. Verify confirmations finish, retries stop after a deadline,
temperature speeds up during transitions and slows once settled, and pending
work refuses a pause. Validate the firmware's actual reported FPS precision and
cooling-state evidence. Compare timing tails and RMS separately: occasional
serial-associated tails can remain with ten-second temperature polling, and the
separate conversion variation is not resolved by this policy.

Start with these helper-free comparisons. If kernel attribution is needed,
repeat with the one-second native or wrapper capture from section 6 wholly
inside a confirmed pause, preserving setup/settling time and a matching normal
capture. The probes add timing cost, so keep that comparison separate. Serial
replies and their EDT interrupts need not occur in the frame in which a command
was issued; reduced traffic alone does not guarantee deconfliction.

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
