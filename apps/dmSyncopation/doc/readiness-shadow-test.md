# Release-qualified and pipeline shadow tests

This test uses exact `dmtweeter_read_done` records for real main-command reader
readiness, while retaining explicit assumptions for auxiliary commands that
exist only in the model. It writes no DM command and changes no live scheduler
behavior. The design and qualification limits are in the
[engineering plan](../../../agents/plans/2026-10/dm-syncopation-release-readiness-shadow.md).
The opt-in [pipeline resource shadow](#pipeline-resource-shadow) adds independent
preparation, SDK service and three-credit post accounting.

## Setup

Rebuild/install **dmSyncopation and logdump**. The BMC controller publisher is
unchanged; retain both `dmtweeter_read_done` and `dmtweeter_done`, the SDK worker
on spare56, and the release observer at FIFO30 on spare57. Keep operational
synco **Off**, RTC latency tuning On, and camera/loop settings unchanged.

Merge the additive
[`dmtweeter_synco_6khz_shadow.conf`](../dmtweeter_synco_6khz_shadow.conf) into the
existing instance file. It preserves the existing stream mapping and all
delivery, keepout, guard, early-main, copy and snapshot settings. The new keys
are startup-only:

```ini
[simulation]
enabled=true
readerReleaseReady=true
virtualReaderUs=30
backendUs=155
```

The mode requires the existing `readerReleaseMonitor.shmimName` to be configured.
An empty release source, nonfinite/nonpositive durations, or a backend duration
shorter than the virtual-reader duration is rejected. Only simulation accepts
these controls; enabling real scheduling remains prohibited in simulation.

First use `[scheduler] maxSubframes=2`, restart synco through the normal operator
procedure, and compare against the prior two-step capture. Confirm advancing
release qualification and output associations, the expected modeled durations,
and no queue drops or invalid evidence. Then set `maxSubframes=3`, restart and
repeat the same bounded capture.

## What the three-step request means

At 2 kHz main cadence, three requested slots have nominal targets at 0,
166.667 and 333.333 us, so the requested total cadence is 6 kHz. Step zero
remains passive preparation for the real main; the other two steps are
hypothetical auxiliary commands. Actual DM commands still arrive at 2 kHz.

For the real main, a qualified release observed at R plus the existing guard
must precede the committed copy start. The model matches the exact combined
inode/counter/raw writetime and current publisher/session/epoch. It does not
use ordinary timing summaries or stale semaphore posts as readiness. Evidence
arriving after the deadline cannot rescue that occurrence; later targets do
not shift. Attachment/reset/queue loss revokes queued old evidence.

Virtual auxiliaries have no physical release records in this test.
`virtualReaderUs=30` is their assumed trigger-to-last-read duration, followed
by the unchanged guard (20 us by default). `backendUs=155` is an independent
assumed complete backend occupancy, including delivery and callback retirement.
A real main's release cannot clear one of those virtual jobs. Backend work must
finish before the next active trigger and before the earliest modeled main.

**The default conservative backend bound does not support all three slots.**
The final active target at 333.333 us has only about 141.667 us until the
earliest next main at 475 us. A 155 us backend duration therefore skips that
slot even though channel readers are already clear. This is the expected
`backendWrap` result, not a reason to reduce the existing timing reserves.

Explicit smaller `backendUs` values can explore a conditional model, but do
not qualify that bound on hardware. The joined-worker captures described in the
engineering record predate the full pipeline. Current native-entry-to-worker-ready
measurements exclude earlier delivery/preparation and later post retirement;
the pipeline mode below models those resources independently.

## Pipeline resource shadow

Rebuild/install **dmSyncopation and logdump**. Retain the working BMC full
pipeline, SDK worker, early/full completion streams and release observer. Keep
operational synco **Off** and RTC latency tuning On. Merge the additive
[`dmtweeter_synco_pipeline_shadow.conf`](../dmtweeter_synco_pipeline_shadow.conf)
into the existing instance configuration; preserve all stream mappings and
existing timing reserves. This is an update to the instance file, rather than
a complete replacement configuration.

```ini
[scheduler]
maxSubframes=3

[simulation]
enabled=true
readerReleaseReady=true
pipelineBackend=true
virtualReaderUs=60
prepareUs=35
sdkServiceUs=120
postUs=150
```

The new controls are startup-only. Restart synco through the normal operator
procedure after merging them. Actual DM commands continue at the main 2 kHz
rate; the two auxiliary steps remain hypothetical.

| Setting | Meaning |
| --- | --- |
| `pipelineBackend` | Select the separate preparation, SDK and post model; default false preserves the previous modes. Requires simulation and exact reader-release gating. |
| `prepareUs` | Controller input arrival through private-vector readiness. Default 35 us is an explicit preparation/handoff assumption. |
| `sdkServiceUs` | SDK entry through worker readiness. Default 120 us allows about 6.6 us beyond the measured 113.444 us maximum. |
| `postUs` | Independent serialized post service through final credit release, after both SDK readiness and monitor staging. Default 150 us is an unqualified modeling assumption, including final publication/retirement tails. |
| `virtualReaderUs` | Total hypothetical trigger-to-last-read duration. The preset uses 60 us, covering the existing 20 us delivery allowance plus 40 us monitor-reader occupancy. |

Delivery uses the existing `deliveryUs` allowance. Preparation holds the monitor
until its last shared read, while its private vector can become available
earlier. SDK service can overlap the remainder of monitor staging; post work can
overlap the next preparation and SDK service. Exactly three private credits span
all owners, and a credit becomes reusable only after ordered post retirement.
An auxiliary reserves its credit at the fixed copy decision, before its future
input arrival; stage execution is never advanced into the future to make an
admission succeed. The model keeps admitted jobs through retirement and does not
simulate the controller's queued-vector supersession, so unexpected bursts can
produce conservative queue/credit conflicts.

Every stage value is a conditional fixed assumption, including genuine-main
resource work anchored to a coherent combined observation. The actual main's
copy gate still uses its exact observed reader-release record. That record
never clears a hypothetical auxiliary SDK or post job. Native SDK-result/full
completion telemetry remains available for comparison, but does not manufacture
measured endpoints for virtual commands.

Durations must be finite, positive and no greater than one second, and the
reader duration must cover delivery plus preparation. The legacy `backendUs`
value remains a readback/configuration value but is unused in pipeline mode;
there is no need to change it to the SDK duration. All previous copy, guard,
visibility, early-main and cadence settings retain their meanings.

### Zero added main SDK wait

The model grants **no minimum preparation time** before the next main's SDK
entry. A prospective auxiliary must leave monitor readers and the SDK clear by
the earliest next main, and leave at least one credit free there. Its own
preparation and SDK entry must also be available at their modeled times. Fixed
copy/trigger targets never move to accommodate contention.

With the preset and a 500 us period, SDK, post and credit stages individually
fit 166.667 us slots. The last trigger at 333.333 us nevertheless reaches SDK
readiness at **508.333 us** after its 20 + 35 + 120 us path. The earliest next
main is **475 us** with the existing reserves. Therefore that slot is skipped,
and `mainClearanceShortfallUs` records approximately **33.333 us**. This is a
clearance shortfall, not a recommendation to increase the early-main reserve.
The new test does not assert that all three slots fit these assumptions.

An unexpectedly early genuine main can still violate the assumed arrival
envelope after an auxiliary decision has been committed. The shadow reports
predicted added preparation/SDK queue delay and credit misses for that case;
it does not revise the earlier decision or change physical commands. Lost
combined provenance or a model reset conservatively fences uncertain work until
its modeled drain rather than creating an immediately free backend.

### Pipeline readbacks and capture

The operator can inspect:

```bash
timeout -k 2s 8s getINDI dmtweeter-synco.simulation_pipeline
timeout -k 2s 8s getINDI dmtweeter-synco.simulation_main_wait_trace
timeout -k 2s 8s getINDI dmtweeter-synco.simulation_steps
timeout -k 2s 8s getINDI dmtweeter-synco.simulation_status
```

`simulation_pipeline` exposes assumptions, fixed capacity, current/maximum
occupancy, admitted main/auxiliary jobs, retirement/discards and model fences.
`mainSdkWaits` and `maxMainWaitUs` include modeled preparation plus SDK queue
delay at a genuine main's SDK entry. `mainCreditMisses` counts modeled main
admission failures, including uncertain-source quarantine; these are not
measured hardware failures. `mainClearanceShortfallUs` is the largest prospective
clearance overrun that vetoed an auxiliary. Counts/maxima accumulate since model
configuration and survive resource fences; inspect capture-local deltas.

Separate flags identify `sdkCycleFeasible`, `postCycleFeasible`,
`creditCycleFeasible` and `mainProtectionFeasible`. The existing backend-cycle
flag is their conjunction; complete feasibility also requires reader/copy and
cadence geometry. Individual eligible slots are still evaluated when the full
cycle is infeasible. New step counters are `PreparationBusy` and `CreditFull`;
`BackendBusy` identifies SDK contention and `BackendWrap` identifies zero-main-wait
protection in this mode. Earlier modes keep their existing meanings.

Telemetry identifies `simulation_mode=shadow_pipeline` and records the new
assumptions/diagnostics independently of the legacy whole-backend duration.
It retains exact 64-bit counts; INDI numbers remain diagnostic doubles. Raw
copy/reader overlap traces retain their observed timestamps. An observation
during SDK work is not itself an SDK conflict when that main could still prepare
concurrently; the pipeline wait/credit diagnostics report those modeled conflicts.

When `mainSdkWaits` increases, `main_wait` telemetry now identifies the waiting
main, separate preparation/SDK queue delays, and the earlier main or auxiliary
jobs occupying those resources. It records every such SDK-entry event while
FIFO capacity permits, without requiring an overlap trigger. Inspect
`main_wait_quality` / `simulation_main_wait_trace` for pending/lost evidence;
the FIFO holds 64 events and emits at most eight per telemetry record. Event
IDs are process-lifetime; sessions and immutable job IDs preserve provenance
across model resets and credit reuse. Preserve complete `[dmsynco]` lines or
raw binary telemetry, including records after the capture's final main-wait
increment while the queue drains. These events diagnose the model; they do
not establish physical SDK delay. See the
[main-wait field and clock descriptions](dmSyncopation.md).

Rebuild/install both `dmSyncopation` and `logdump` for this trace. Keep the
current pipeline preset and timing bounds unchanged. No controller or camera
rebuild is required for this diagnostic change.

After at least ten seconds of stable cadence, repeat the existing
[bounded simultaneous capture](dmSyncopation.md#capture-a-simulation-run).
Retain all stage feasibility flags, per-step outcomes, model-credit conservation,
main-delay diagnostics and actual camera/controller timing. The simulation uses
fixed storage and generates no native 6 kHz submissions.

## Readbacks and capture

The operator can run bounded checks:

```bash
timeout -k 2s 8s getINDI dmtweeter-synco.simulation_readiness
timeout -k 2s 8s getINDI dmtweeter-synco.simulation_steps
timeout -k 2s 8s getINDI dmtweeter-synco.simulation_status
```

`simulation_readiness` reports:

- `virtualReaderUs` and `backendUs`: explicit assumptions, not measured virtual
  endpoints.
- `readerCycleFeasible` and `backendCycleFeasible`: independent nominal geometry.
  Overall `simulation.feasible` requires both plus the cadence rating. Per-step
  outcomes are still evaluated when only part of the requested cycle fits.
- Source observations/qualified records, invalid/incoherent evidence,
  progression gaps/duplicates, source resets, identity mismatch, SDK failures
  and transport drops. A later successful SDK enrichment is not another release.
- Latest qualified source epoch, publisher session/epoch/sequence, combined
  counter and observation age. INDI integer readbacks are diagnostic doubles;
  telemetry preserves exact evidence.

The per-step counters add `ReleaseMissing`, `ReleaseLate`, `BackendBusy` and
`BackendWrap`. `readerHold` remains a telemetry skip reason for modeled virtual
readers. Warmup, missing output/main and early-main reasons remain distinct.
A missing output at the modeled deadline is not proof of a lost physical DM
command. Review deltas during the capture, not old lifetime totals.

In the legacy reader-release mode, an actual main arriving during modeled virtual
backend occupancy also produces an overlap event with `window:backend` and its
active step's `backendEnd_ns`.
This remains visible if the readers have cleared or a subsequent passive copy
has completed. Reader clearance is not mistaken for backend readiness.

After at least ten seconds of stable cadence, use the existing
[bounded simultaneous capture](dmSyncopation.md#capture-a-simulation-run).
The new optional telemetry readiness section labels real-main release evidence
and virtual assumptions separately. Retain source quality, both feasibility
values and per-step skip reasons with the camera/controller timing. Legacy
records remain readable.

This test does not exercise 6 kHz BMC submission capacity or generate auxiliary
triggers. The later live integration must also provide the intended dmcomb wake
route, exact reader gating and independent backend admission/early-main checks.
