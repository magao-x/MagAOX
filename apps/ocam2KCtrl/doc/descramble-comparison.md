# OCAM single-core descramble comparison

The optional bin2 methods produce the same pixels as the SDK using a mapping
prepared and validated during camera configuration. The SDK remains the
default. Other camera modes and tables that do not match the validated
structure use the SDK even when a candidate is requested.

The development workstation measurements and validation record are in the
[engineering plan](../../../agents/plans/2026-11/ocam_descramble.md).
They do not establish performance on the RTC.

## Configuration and status

Existing configurations need no change. To persist a selection after it has
been qualified, the optional setting is:

```ini
[camera]
descrambleMethod=mapped
```

The accepted values are `sdk`, `compact`, `mapped`, and `optimized`; the default
is `sdk`. An unknown value fails configuration. All three candidates require
the same validated structured bin2 mapping in the app. The offline helper can
compare other modes.

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

| INDI property / element | Meaning |
| --- | --- |
| `descramble_method.sdk` / `.compact` / `.mapped` / `.optimized` | Requested method; exactly one switch is On. |
| `descramble_status.applied` | `sdk`, `compact`, `mapped`, `optimized`, or `none` when no active completed load is reported. |
| `descramble_status.availability` | `ready`, `unsupported mode`, `unavailable table`, or `unconfigured`. |

The request is latched once for an entire frame. `applied` changes after the
load completes and is published by the ordinary app thread. A changed request
is Busy until acknowledged. An unavailable candidate request is Alert and
uses SDK conversion; the request is retained for a later compatible camera
configuration. Selecting `sdk` restores the reference path. No method change
issues camera serial commands or restarts acquisition.

`applied` describes completion of conversion and optional digital sampling;
the existing frameGrabber staging copy and stream publication follow it.
Wait out a full timing window after acknowledgment before collecting data.

## Operator-run RTC comparison

These are instrument operations for the operator to run from their workstation.
Agents must not execute these commands or control the instrument.

1. Build/install the app, configure `framegrabber.stageTiming=true`, and restart
   it using the usual deployment procedure.
   Verify bin2/FPS after restart, then keep FPS, gain, synchronization,
   IRQ/cpuset placement, stage-timing settings and reader workload fixed. Retain
   `framegrabber.staged=true` and the existing `stagedCopyNT` setting. This change
   needs no additional core.

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
