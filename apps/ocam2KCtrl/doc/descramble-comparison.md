# OCAM descramble comparison

The bin2 methods produce the same pixels as the SDK using a mapping prepared
and validated during camera configuration. `parallel` is the default requested
method, using the `camwfs_descramble` worker cpuset and a 100 us post-job sleep.
Other camera modes, unvalidated tables, or unavailable worker/staging setup
use the SDK fallback. SDK and single-thread optimized remain explicitly
selectable.

The development workstation measurements and validation record are in the
[engineering plan](../../../agents/plans/2026-11/ocam_descramble.md).
They do not establish performance on the RTC.

## Configuration and status

Omitting the method setting selects `parallel`; omitting the worker cpuset
selects `camwfs_descramble`. Explicit settings still take precedence, including
an old `sdk`/`optimized` method or `camwfs_sw` worker override. To adopt the
current defaults, remove those overrides or use:

```ini
[camera]
descrambleMethod=parallel

[descramble]
cpuset=camwfs_descramble
sleepUs=100
```

The accepted values are `sdk`, `compact`, `mapped`, `optimized`, and `parallel`;
the default is `parallel`. An unknown value fails configuration. All candidates
require the same validated structured bin2 mapping in the app. Parallel also
requires staged output and verified worker placement as described below. The
offline helper can compare other modes.

`compact` and `mapped` use the same prepared 16-bit index table. `compact`
follows the SDK's alternating-half pixel traversal; `mapped` writes output
pixels sequentially. Their comparison investigates traversal and generated
code without changing the index table. `optimized` uses four fixed-stride
amplifier runs and avoids per-pixel map loads. The compiler can schedule memory
operations differently; these distinctions describe source-level traversal,
not exact machine accesses. The candidates have not established full RTC
performance and jitter qualification.

On 2026-11-18 the operator reported `compact` a few hundred nanoseconds slower
than SDK with the same observed jitter. It remains a diagnostic control; the
local benchmark speedup did not carry over to that RTC test. Details and the
separate `optimized` mean/spread tradeoff are recorded in the engineering plan.

The later four-method capture at approximately 2,000 FPS favors `optimized`
among the candidates: relative to the two confirmed SDK reference captures,
conversion improves by about 3.98 us, with standard deviation increasing by
about 0.09 us. Mapped is both slower and more variable than optimized in that
capture. These short windows support further
SDK/optimized/SDK qualification; they do not establish longer-run tail or
end-to-end loop performance. See the engineering plan for the full statistics.

A subsequent 2,000-FPS comparison with stage timing disabled retains about
3.91 us acquisition-to-write improvement, with standard deviation increasing
by about 0.059 us (5.8%). The smaller spread penalty therefore persists without
stage instrumentation; the cause of the earlier larger penalty remains
unresolved. Final qualification should also use the intended operating
instrumentation setting, fixed throughout each matched comparison.

The later 2,000-FPS capture labeled optimized/parallel averages 33.760 versus
37.423 us acquisition-to-write, with reported standard deviations 1.067 versus
1.119 us. The operator corrected the worker status to `parked`; follow-up
readbacks identified SDK fallback caused by an empty power target despite
observed power On. The active-parallel attribution is withdrawn, so this
capture does not establish worker performance. Repeat it after the power-target
fix, verifying `applied=parallel` and `parallel=running` before and after the
interval. Earlier SDK comparisons motivated the initial optimized default;
that historical decision preceded the corrected worker measurements below.

After the power-target fix, a confirmed continuous-spin parallel capture
averaged 28.826 us acquisition-to-write with 0.799 us reported standard
deviation, versus the earlier optimized 33.760 us and 1.067 us. However,
acquisition/publication interval spread rose to roughly 28 us and the operator
confirmed it disappeared on returning to optimized. Continuous-spin parallel
was therefore not adopted. The RTC reported a 950 ms/s RT budget, motivating
the 100 us post-job sleep to reduce continuous worker CPU occupancy.

The subsequent operator-reported 2 kHz sleep trial averages **29.003 us WMA**
with **0.775 us reported standard deviation**. Acquisition-interval spread
returns to 0.541 us, and publication-interval spread is 1.270 us. Relative to
the earlier optimized capture, WMA improves by **4.757 us (14.1%)** and its
reported spread by **27.4%**, while the large interval-jitter penalty is absent.
This supports the worker option at 2 kHz. These are averages of supplied
rolling-window summaries; the numerical optimized reference comes from an
earlier capture.

A subsequent 3.6 kHz sleep trial averages **28.861 us WMA** with **0.975 us
reported standard deviation**. Acquisition/publication interval spread is
0.591/1.367 us, with no return of the large continuous-spin penalty. WMA spread
is about 0.200 us higher than in the 2 kHz sleep trial, while mean latency
remains near 29 us. This establishes short-run behavior at the stated maximum
rate, but no matched 3.6 kHz optimized reference is supplied. Longer matched
comparisons, frame-delivery/tail checks and complete-loop measurements remain
useful follow-up checks. After these trials, the operator confirmed that
moving `camwfs-sw` to CPU31 had no timing impact and selected parallel as the
default with the dedicated CPU32 worker.

For single-thread operation, explicitly select `optimized` (or `sdk`) and set
`descramble.cpuset` to an empty value before restart to disable worker creation.
Removing the cpuset setting restores its `camwfs_descramble` default. Do not
mix workload or placement changes into an A/B interval.

| INDI property / element | Meaning |
| --- | --- |
| `descramble_method.sdk` / `.compact` / `.mapped` / `.optimized` / `.parallel` | Requested method; exactly one switch is On. |
| `descramble_status.applied` | `sdk`, `compact`, `mapped`, `optimized`, `parallel`, or `none` when no active completed load is reported. |
| `descramble_status.availability` | `ready`, `unsupported mode`, `unavailable table`, or `unconfigured`. |
| `descramble_status.parallel` | `disabled`, `requires staging`, `unverified worker`, `unavailable table`, `parking`, `parked`, or `running`. |
| `descramble_status.caller_cpu` / `.worker_cpu` | Verified logical CPUs, or `-1` before verification. |
| `descramble_status.sleep_us` | Configured worker sleep after job completion, in microseconds; not measured sleep duration. |
| `th-descramble.pid` / `.prio` | Worker thread ID and requested RT priority; actual scheduling is verified before use. |

The request is latched once for an entire frame. `applied` changes after the
load completes and is published by the ordinary app thread. A changed request
is Busy until acknowledged. An unavailable candidate request is Alert and
uses SDK conversion; the request is retained for a later compatible camera
configuration. This also applies to the default parallel request outside
validated bin2 or without an eligible worker: SDK fallback succeeds, while the
unavailable parallel request reports Alert. Configure `sdk` explicitly when
that is the intended mode.
Selecting `sdk` restores the reference path. No method change issues camera
serial commands or restarts acquisition.

`applied` describes completion of conversion and optional digital sampling;
the existing frameGrabber staging copy and stream publication follow it.
Observed power On permits conversion even when the outlet target is empty
(unknown); an explicit Off target still parks the worker and suppresses the
active applied-method readback. Older builds incorrectly used SDK fallback for
parallel requests with an empty target and showed `applied=none`. Upgrade the
app to correct that condition; no outlet command is needed to populate target.
Wait out a full timing window after acknowledgment before collecting data.

## Operator-run RTC comparison

These are instrument operations for the operator to run from their workstation.
Agents must not execute these commands or control the instrument.

1. Build/install the app, configure `framegrabber.stageTiming=true`, and restart
   it using the usual deployment procedure.
   Verify bin2/FPS after restart, then keep FPS, gain, synchronization,
   IRQ/cpuset placement, stage-timing settings and reader workload fixed. Retain
   `framegrabber.staged=true` and the existing `stagedCopyNT` setting. This change
   needs no additional core for the single-thread methods.

2. Inspect preparation and establish the reference selection:

   ```bash
   getINDI camwfs.descramble_status
   setINDI -s camwfs.descramble_method.sdk=On
   getINDI camwfs.descramble_method
   getINDI camwfs.descramble_status
   ```

   Require `availability=ready` before testing a candidate, and wait for
   `applied=sdk` with an Ok selection state. Exclude at least one full
   configured rolling latency window plus warm-up.
   Collect a 60-second reference interval under the usual reader workload.

3. Select the compact control, confirm acknowledgment, exclude the same settling
   interval, and collect another matched 60-second interval:

   ```bash
   setINDI -s camwfs.descramble_method.compact=On
   getINDI camwfs.descramble_method
   getINDI camwfs.descramble_status
   ```

   Require `applied=compact`, `availability=ready`, and an Ok selection state.

4. Select mapped, require `applied=mapped`, `availability=ready`, and an Ok
   selection state, then settle and collect another matched 60-second interval:

   ```bash
   setINDI -s camwfs.descramble_method.mapped=On
   getINDI camwfs.descramble_method
   getINDI camwfs.descramble_status
   ```

5. Return to compact, confirm the same acknowledgment as step 3, settle, and
   collect another 60-second control interval:

   ```bash
   setINDI -s camwfs.descramble_method.compact=On
   getINDI camwfs.descramble_method
   getINDI camwfs.descramble_status
   ```

   This brackets mapped with compact under one fixed setup. Apply the same
   settling interval after every acknowledged selection.

6. Return to the SDK, wait for `applied=sdk` with an Ok selection state, settle,
   and repeat the reference capture:

   ```bash
   setINDI -s camwfs.descramble_method.sdk=On
   getINDI camwfs.descramble_method
   getINDI camwfs.descramble_status
   ```

The existing structured choice remains available through
`descramble_method.optimized=On`. Compare it separately between settled SDK
references, requiring `applied=optimized`, `availability=ready`, and an Ok
selection state. Return to SDK using step 6 afterward.

Record the applied selection with each capture. Compare camera
`fg_timing.delta_aw`, its jitter, dropped/rejected frames and end-to-end loop
latency. Keep instrumentation identical across intervals. Collect the
conversion mean, standard deviation, extrema, sample count and validity:

```bash
getINDI camwfs.fg_stage_timing.conversion_mean \
  camwfs.fg_stage_timing.conversion_stddev \
  camwfs.fg_stage_timing.conversion_min \
  camwfs.fg_stage_timing.conversion_max \
  camwfs.fg_stage_timing.conversion_count \
  camwfs.fg_stage_timing.conversion_valid
```

Timing values are in seconds; count and validity are dimensionless. That
conversion interval includes the metadata write flag, descramble, digital
sampling if configured, and staging copy. `delta_aw` spans the SDK acquisition
timestamp through the stream write timestamp; neither is pure kernel time.

Use the [existing timing guide](camera-timing-trace.md) for capture details.
Do not count overlapping rolling windows as independent samples or derive
percentiles from summary mean/jitter alone. Repeat compact/mapped/compact and
the SDK reference captures if drift changes the controls. Retain a candidate
only if the camera and complete loop show no repeatable regression in mean,
jitter/tails or frame delivery. Leave the SDK selected if qualification is
inconclusive.

## Operator-run parallel comparison

The [RTC cpuset setup](../../../rtSetup/RTC/make_cpusets) now assigns:

| CPU | Cpuset | Use |
| --- | --- | --- |
| 31 | `camwfs_sw` | `camwfs-sw` software framegrabber; formerly `spare31`. |
| 32 | `camwfs_descramble` | OCAM descramble worker; formerly `camwfs_sw`. |
| 33 | `camwfs` | Camera framegrabber, unchanged. |
| 34 | `camwfs_irq` | EDT IRQ, unchanged. |

CPUs32/33 share L3; CPU31 is in the neighboring domain. The operator tested
`camwfs-sw` on CPU31 and reported no timing impact. The application still
verifies worker placement before using it. Keep software-reader placement
and other workloads fixed across reference/parallel intervals.

Before migrating, the operator stops workloads using the dedicated cpusets,
then runs the updated `remove_cpusets` in dry-run mode. It recognizes both
the current and renamed layouts, including retired `spare31`; it preserves
root/system, refuses occupied or unknown groups, and does not move tasks.
Run `remove_cpusets --apply` only after its empty-tree preflight succeeds,
then rebuild using the sibling updated `make_cpusets`. These are operator
steps, not commands for an agent to run on the instrument.

Update any existing camera `[descramble] cpuset=camwfs_sw` to
`camwfs_descramble`, or remove it to use the new default. Restore the
`camwfs-sw` app's `[framegrabber] cpuset=camwfs_sw` if the temporary trial used
`spare31`. The software app now resolves that name to CPU31. The scripts do
not rewrite application configuration. Install the updated cleanup/setup
scripts together before restarting workloads.

For a controlled comparison, merge these settings after the migration; the
explicit SDK selection below establishes a reference instead of using the
parallel method default:

```ini
[camera]
descrambleMethod=sdk

[framegrabber]
staged=true

[descramble]
cpuset=camwfs_descramble
threadPrio=-1
partition=paired30
sleepUs=100
```

`threadPrio=-1` inherits `framegrabber.threadPrio`; an explicit priority must
be 1–99. `partition` accepts `paired30` (default, balanced row pairs),
`paired28` (28/32 pairs), or `halves` (60 output rows each). Changing worker
configuration requires an app restart. Keep the current `stagedCopyNT`, FPS,
gain, synchronization, readers and stage-timing setting fixed during each
comparison. Restart through the normal operator deployment procedure and
verify the actual FPS afterward.

`sleepUs` defaults to **100 us**, accepts integer values from **0 through
200 us**, and requires a restart to change. After publishing completion, the
worker sleeps once, then spins for the next frame. The first job is not
delayed. Zero restores the continuous-spin control. `parallel=running` includes
these short sleeps; `parked` means the worker is inactive between experiments.

At 2 kHz and 3.6 kHz, 100 us is nominally 20% and 36% of a frame period. The
OS can sleep longer than requested, which can delay the next frame; the 200 us
configuration cap does not guarantee an on-time wake. At low FPS or during
acquisition stalls, one fixed sleep followed by spinning can still exhaust
a finite RT budget. This option requires instrument qualification at the
intended cadence. Keep optimized selected if interval jitter persists.

An explicit empty `descramble.cpuset` disables the worker; otherwise its
default is `camwfs_descramble`. A configured worker starts
parked, then verifies both threads' actual priorities/policy, exact single-CPU
cpuset membership, different physical cores, and shared L3. Both framegrabber
and worker need dedicated single-CPU cpusets. Failure leaves `parallel`
unavailable and the existing single-thread methods usable. Placement is
checked again during acquisition configuration. Startup verification does not
detect another task later being assigned to the same core; the operator owns
that resource reservation.

With SDK selected, check:

```bash
getINDI camwfs.descramble_status camwfs.th-descramble
```

Require `availability=ready`, `parallel=parked`, `sleep_us=100`, and the intended
`caller_cpu`/`worker_cpu`. Capture a settled SDK interval. Then test
**optimized → parallel → optimized**, with the same settling and capture
intervals used above:

```bash
setINDI -s camwfs.descramble_method.optimized=On
getINDI camwfs.descramble_status

# After collecting the settled optimized interval:
setINDI -s camwfs.descramble_method.parallel=On
getINDI camwfs.descramble_status

# After collecting the settled parallel interval:
setINDI -s camwfs.descramble_method.optimized=On
getINDI camwfs.descramble_status
```

Require `applied=parallel`, `parallel=running`, `sleep_us=100`, and an Ok
selection state before and after the worker capture. Require `parallel=parked`
in SDK/optimized captures. For an explicit continuous-spin control, configure
`sleepUs=0`, restart, and require `sleep_us=0` instead.
Discard at least one full timing window after each acknowledged transition;
parking/resuming costs can affect the first frame. Finish with a return-to-SDK
capture. The worker completes both image portions before digital sampling,
the existing primary-thread staging copy, publication and raw-buffer reuse.
Unsupported modes/tables, missing staging or unavailable worker placement
fall back to SDK and report the unmet parallel request.

Compare conversion and acquisition-to-write mean, spread and tails, plus frame
quality and full-loop latency. The local experiment found a kernel improvement
whose staged result depended strongly on cadence. The instrument test must
include the final copy and the cost of reserving this core. Keep stage timing
fixed for a crossover; if it is disabled, use ordinary `fgtimes`/`fg_timing`
rather than expecting new conversion-stage samples.

## Hardware-free tests and benchmark

From the repository root:

```bash
make -C tests -f Makefile.one t=../apps/ocam2KCtrl/tests/ocamDescrambler_test.cpp
apps/ocam2KCtrl/tests/ocamDescrambler_test
```

An optional valid table can be checked alongside generated fixtures:

```bash
MAGAOX_OCAM_DESCRAMBLE_TABLE=/opt/MagAOX/config/ocam2k/fli/ocam2_descrambling.txt \
  apps/ocam2KCtrl/tests/ocamDescrambler_test
```

The standalone benchmark opens only the mapping file and ordinary memory:

```bash
make -C apps/ocam2KCtrl/benchmarks
apps/ocam2KCtrl/benchmarks/ocamDescramble_bench \
  /opt/MagAOX/config/ocam2k/fli/ocam2_descrambling.txt \
  bin2 10000 64 -1 7 > /tmp/ocam-descramble.csv
```

Arguments are table, mode (`normal`, `bin2`, `bin1x3`, `bin1x4`), samples per
round, rotating raw-buffer count, CPU (`-1` retains current affinity), and
round count. A chosen CPU pins only this benchmark process. Use an otherwise
available workstation CPU for repeatability; do not run competing benchmarks
on an operating instrument.

Each invocation measures a hot input and the requested rotating-buffer set.
Output includes `sdk`, `mapped` (sequential-output compact map), `compact`
(SDK alternating-half traversal in bin2), `optimized` (structured kernel), and
`memcpy` calibration results, with and without `streamCopy`. All replacement
outputs are checked against the SDK first. Outside bin2, `compact` uses the
sequential-output map; without a validated structure, `optimized` also uses
that fallback. Memcpy is only a timing calibration and does not produce a
usable descrambled image.

The CSV records compiler flags, CPU/affinity, table hash, buffer alignment,
sample count, round means, mean/median/deviation/p95/p99/maximum, and timer
overhead. It uses synthetic data and cannot reproduce DMA or shared-stream
reader traffic. Rebuild with `make -B` when changing compiler options because
make does not track command-line flag changes as dependencies.


## Offline second-thread experiment

The optional benchmark worker uses the same mandatory-job engine as the
application's `parallel` method, with local thread/affinity setup. It uses one persistent worker and the
calling thread. The caller and worker must use distinct physical cores;
choose logical CPU numbers from the workstation's topology, not from their
numeric proximity. The example below uses CPUs 2 and 4, verified as separate
P-cores on mx11. No real-time scheduling or instrument cpuset change is made.

```bash
make -C apps/ocam2KCtrl/benchmarks
make -C apps/ocam2KCtrl/benchmarks check TEST_CPUS='2 4'
apps/ocam2KCtrl/benchmarks/ocamDescramble_bench \
  /opt/MagAOX/config/ocam2k/fli/ocam2_descrambling.txt \
  bin2 10000 64 2 7 4 0 1 0 > /tmp/ocam-two-thread.csv
```

After the original `CPU ROUNDS` arguments, the additional arguments are
`WORKER_CPU OUTPUT_OFFSET_BYTES DIAGNOSTICS PERIOD_US`:

- `WORKER_CPU`: a distinct allowed physical core; omitted or `-1` disables the
  worker and retains the original single-thread comparisons.
- `OUTPUT_OFFSET_BYTES`: even offset 0–62 from a 64-byte-aligned private and
  publication buffer base; default 0. Repeat at 16 to test unaligned ownership
  boundaries. Actual addresses and split boundaries are recorded.
- `DIAGNOSTICS`: 1 adds a separate pass reporting caller/worker kernel durations
  and instrumented totals; default 0. Headline samples omit those extra clocks.
- `PERIOD_US`: default 0 measures a tight loop. Setting 500 schedules sample
  starts approximately every 500 us, with sleeping outside the measured call.
  Lost whole pacing slots are reported. This remains synthetic input with
  ordinary OS scheduling, not a DMA or real-time camera simulation.

`optimized` measures the unchanged production helper. `range_serial` and
`paired_serial` measure the benchmark's mirrored row kernel on one thread,
providing controls for its compiler output and paired traversal. The parallel
choices split output halves or assign matching rows `y`/`119-y` together,
using 28/32 or 30/30 raw-row pairs. The latter splits can differ in cache-line
sharing; even the aligned 30/30 split shares output boundary lines.

Each parallel measurement includes job dispatch and completion waiting. The
primary thread performs `streamCopy` only after all pixel work completes.
A handoff-only calibration performs no pixel work. The worker is blocked and
acknowledged parked during every serial baseline block, and it resumes outside
timed parallel calls. It is created once per invocation, with no per-frame
allocation or thread creation.

All generated raw frames and parallel/serial outputs are checked against the
SDK before timing, including source immutability and output guards. The
standalone checks additionally exercise repeated jobs, input reuse, parking,
startup failures and shutdown after completed calls. The engine has a blocking, single-caller contract; asynchronous park/stop
requests drain admitted work before buffer reuse. Its owner joins the worker
before destroying the engine or its mapping.

Compare complete staged times and tails against `optimized`, not only the
worker's own kernel duration. Local Intel results cannot establish behavior on
the RTC's AMD CCDs or under its shared-stream readers. Keep correctness and
sanitizer builds separate from performance measurements.

The [2026-11-18 execution record](../../../agents/plans/2026-11/ocam_descramble.md#local-second-thread-experiment--2026-11-18)
includes local correctness checks and both timing workloads. Two threads
improved the kernel, but staging erased the gain in tight loops; sleeping
between frames reversed the result under the laptop's ordinary power and
scheduling settings. This establishes cadence sensitivity, not an RTC gain. Those measurements
precede the later shared engine with admission checks; repeat the benchmark
when comparing the current runtime handoff implementation.
