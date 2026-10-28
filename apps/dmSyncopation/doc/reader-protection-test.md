# Shadow reader-protection measurement

This test uses the DM controller's completion record to measure the remaining
software reader lifetime at the exact observations used by synco's shadow model.
It does not copy auxiliary commands, post a DM trigger or change any live or
shadow reader hold. Only real main commands have measured completion; virtual
active steps do not create controller records.

## Setup

Rebuild/install the DM controller (`bmcCtrl` for dmtweeter), `dmSyncopation` and
`logdump`. The completion publisher itself does not need a new reader, but this
test adds a new optional `dmsynco` telemetry table, so the reader must be rebuilt.

In the **dmtweeter controller** configuration, enable:

```ini
[dm]
completionStream=dmtweeter_done
```

For the **dmtweeter-synco** instance, use the non-default settings in
[`dmtweeter_synco_reader.conf`](../dmtweeter_synco_reader.conf):

```ini
[shmimMonitor]
shmimName=dm01disp03

[scheduler]
maxSubframes=2

[simulation]
enabled=true

[completionMonitor]
shmimName=dmtweeter_done
```

Restart the affected applications through the normal operator procedure. Do
not enable operational synco or start an auxiliary provider. The new observer
is enabled by a nonempty `completionMonitor.shmimName`, is accepted only with
`simulation.enabled=true`, and starts at ordinary priority zero. Its optional
`completionMonitor.threadPrio` and `.cpuset` follow the standard monitor interface;
retain defaults for the initial comparison. With an empty name the new observer
and its INDI properties are absent. Diagnostics use the existing two-second
`diagnostics.windowSec` and `diagnostics.maxEvents` per-source capacity (20,000
by default). Missing completion streams wait for attachment and yield no matches.

Preserve the camera/loop settings, dmcomb mode/delay and workload used in the
preceding comparison. Retain `sysMonRTC.set_latency.toggle=On`. Record the
settings and any changes with each capture. For the first comparison, keep the
completion publisher enabled in both baseline and observer runs, so the observer
is the only added workload.

## Measurements

```text
combined publication U --> controller start S --> SDK + saturation + delta reads
                     |                           --> completion F --> notice R
                     +--> synco combined observation C (may be later than F)

synco primary observation P --+
                             +--> B = max(P, C) ---- modeled hold ---> B + budget
synco combined observation C -+

Identity join: primary inode/counter -> dmcomb named trace
             combined inode/counter/raw writetime == completion input identity
```

All differences below use the local `CLOCK_MONOTONIC`. Raw producer writetime
and process-trace timestamps establish identity/order in their own clock and
are never subtracted from a monotonic endpoint.

| Metric | Definition | Meaning |
| --- | --- | --- |
| `completionOffset` | `F - B`, signed | Actual remaining software reads relative to the model baseline. Negative means reads finished before synco obtained its later observation. |
| `requiredHold` | `max(0, F - B)` | Nonnegative observed reader requirement, before any safety reserve. |
| `observedHold` | `max(0, R - B)` | Hold needed if synco must receive and inspect this completion notice. |
| `processing` | `F - S` | Controller callback work including the timing handoff, before completion publication. |
| `notification` | `R - F` | Completion publication, wakeup and observer read delay. |
| `modelSlack` | `budget - (F - B)`, signed | Remaining slack in the currently configured model. Negative exceeds that sample's modeled budget. |

The budget is frozen with each combined observation as
`scheduler.deliveryUs + scheduler.keepoutUs + scheduler.guardUs`, currently
20 + 115 + 20 = **155 us** with defaults. The sample retains the actual configured
budget, so a later settings change cannot relabel earlier evidence.

## Readback and capture

The operator can perform bounded readiness checks:

```bash
timeout -k 2s 8s getINDI dmtweeter.dm_completion_stream
timeout -k 2s 8s getINDI dmtweeter-synco.reader_timing_quality
timeout -k 2s 8s getINDI dmtweeter-synco.reader_timing
```

After warming up, `matched` should advance and the metric counts should be
nonzero. An enabled observer with no matches is not a zero reader requirement.
Check publisher status and the quality counters before proceeding.

Use the existing [bounded 120-second capture procedure](dmSyncopation.md#capture-a-simulation-run)
to retain simultaneous full `dmtweeter-synco`, `dmtweeter` and `camwfs` records,
plus before/after INDI snapshots. The procedure invokes `logdump` explicitly;
the instrument's `teldump` alias is equivalent to its telemetry directory/extension
arguments. Name the result `synco-reader-<UTC timestamp>` if desired. Compare a
baseline simulation run without the completion observer against a run with it,
with the publisher and all other settings unchanged. A longer subsequent run
is useful for retaining rare extrema after this short check.

Each updated `dmsynco` line includes a `reader_timing` section, its rolling
populations, cumulative quality and lifetime worst-case provenance. Exact raw
integer identities and timestamps are retained in telemetry. Legacy records
without this table remain readable and report the measurement unavailable.

`reader_timing` INDI has six metrics, each with `_mean`, `_jitter`, `_min`,
`_max`, `_count` and `_span`. Durations and coverage are **seconds**. A negative
offset/slack can be valid; consult `_count` to distinguish it from the `-1`
unavailable convention. Jitter requires at least two observations. The
`reader_timing_quality` property includes window/span/age, queue/pending counts,
matched and rejected populations, source resets, gaps and drops, plus
`maxRequiredUs`, `maxObservedUs`, `overBudget`, `worstValid`, `worstOffsetUs`,
`worstBudgetUs` and `worstSlackUs`. These explicit `Us` fields use microseconds.
Maximum required and observed holds accumulate for the app lifetime; the exact
worst sample is the one with the largest required hold. INDI's numeric identity
representation is not an exact 64-bit matching interface.

## Qualification and interpretation

Each source has its own bounded handoff. Matching and statistics run outside
the callbacks, support either producer arrival order, and use captured historical
identities rather than requiring the live stream to remain on that command.
Pending storage and retention are bounded. Attachment and publisher-session/epoch
changes revoke pending associations. Duplicate/stale notices, busy records,
backend errors/no-send, incoherent metadata, invalid/future/stale timestamps,
identity mismatches, missing peers and queue loss remain visible as quality
counters. Completion freshness is checked at observer capture; ordinary
application queue delay is not counted as notification delay.

Retained primary evidence can support several combined commands and ages out
normally even after a successful join. Expiry/eviction totals alone therefore
do not mean that a completion was missed; inspect missing-peer, sequence-gap
and queue-drop counts as well as the matched population.

Read both rolling statistics and lifetime extrema, together with the change in
quality counts over the capture. In particular, missed completion notifications
can bias the retained tail downward. Do not infer a hard operating bound from a
maximum observed under incomplete coverage. Synthetic initial callbacks and
unqualified controller outcomes never establish completion.

The result describes the controller's observed software endpoint. It does not
measure physical DM settling, qualify an early-arrival reserve, or prove an atomic
multi-stream relationship between command pixels and semaphore tokens. Other
readers and existing copy/early-main protections still apply. A lower measured
requirement is evidence for a subsequent policy change; this test makes no such
change automatically.
