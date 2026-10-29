# Release-qualified three-step shadow test

This test uses exact `dmtweeter_read_done` records for real main-command reader
readiness, while retaining explicit assumptions for auxiliary commands that
exist only in the model. It writes no DM command and changes no live scheduler
behavior. The design and qualification limits are in the
[engineering plan](../../../agents/plans/2026-10/dm-syncopation-release-readiness-shadow.md).

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
not qualify that bound on hardware. The approximately 131 us callback seen at
2 kHz, including observed maxima near 140 us before final retirement, provides
limited margin in that last window. A measured backend-retirement bound or
separate qualification is needed before a live 6 kHz claim.

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

An actual main arriving during modeled virtual backend occupancy also produces
an overlap event with `window:backend` and its active step's `backendEnd_ns`.
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
