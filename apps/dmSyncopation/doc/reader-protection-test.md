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
| `wake` | `W - F` | Controller endpoint to the timestamp immediately after semaphore return, including publication/posting and scheduling. |
| `capture` | `R - W` | Monitor checks and coherent record read after semaphore return. |

`W` is captured in the monitor's receipt hook immediately after `sem_timedwait`
returns; `R` follows the coherent completion-record read. The new pair requires
`0 < F <= W <= R`. Missing or noncausal receipt evidence leaves the original six
populations and qualified match intact, but contributes neither split metric.
`wake` is not a pure kernel semaphore latency: the controller stamps `F` before
encoding/publishing the record and posting its semaphore.

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

`reader_timing` INDI has eight metrics, each with `_mean`, `_jitter`, `_min`,
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

The quality properties also expose `receiptUnavailable` (no retained wait-return
timestamp), `receiptInvalid` (a nonzero receipt outside `F <= W <= R`) and
`firstEnriched` (an early-release attempt first observed after its SDK result was
added). These counters qualify the smaller split population; they do not turn
a valid read-completion match into a rejected command. Telemetry additionally
retains the exact matched sample with the largest lifetime `R - F`, including
its raw `W`, independently of the worst required-hold sample. This lets one
notification tail be attributed without combining extrema from different events.
Old telemetry has no receipt or split pair and remains readable.

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

## Early channel-release comparison

The worker implementation can publish a separate early endpoint through
`dm.readerReleaseStream=dmtweeter_read_done`. This endpoint means that the
controller finished reading command/channel pixels; its SDK call may still be
running. Keep `dm.completionStream=dmtweeter_done` for the independent version-1
full-completion measurement. Configure worker execution and CPU placement as
specified by the [BMC worker guide](../../bmcCtrl/doc/sdk-worker.md); engineering
decisions are recorded in the
[worker plan](../../../agents/plans/2026-10/bmc-sdk-worker-reader-release.md).

Add this optional observer to the synco instance configuration:

```ini
[readerReleaseMonitor]
shmimName=dmtweeter_read_done
```

The observer defaults to ordinary priority zero and is accepted only in shadow
simulation mode. An empty name disables it. It has its own transport attachment,
bounded queues, source epochs, matching, populations and extrema. It reuses the
same coherent primary/combined observations as the full-completion measurement.
A missing, failed or unqualified full-completion peer does not prevent a genuine
early release from being measured.

`reader_release_timing` and `reader_release_timing_quality` use the same units
and metric definitions as their `reader_timing` counterparts, with `F` now the
last-read endpoint. The optional `reader_release_timing` telemetry table retains
the distinct protocol kind and raw SDK result when available. Both populations
remain observational: the existing **155 us** budget and every live/shadow
scheduler decision stay unchanged.

A later SDK result can enrich the same release record without changing its
identity or last-read timestamp. The observer counts that result once in
`sdkResults`, `sdkSuccess` or `sdkFailed`, without adding a second reader sample.
These count explicit backend outcomes; an SDK exception counts as failed even
though its raw return code remains unavailable.
A failed SDK call does not undo reads that already finished. These are
opportunistic observed-result counts; missed/coalesced notifications can leave
some reader samples without observed SDK-result coverage. Read them together
with completion sequence gaps, duplicates, invalid captures and queue drops.
An unknown SDK result is never counted as successful submission.

If the first observed release record already contains an SDK outcome, its
`notification` still measures `R - F` and `firstEnriched` advances once. That
sample is excluded from both `wake` and `capture`: the later result update
also posts, and the observed semaphore token cannot establish an early-release
wakeup. A later enrichment of an already observed release neither adds a timing
sample nor replaces the first observation's receipt. Full-completion records
with valid receipt timestamps remain eligible for the split normally.

Use the same simultaneous bounded capture after rebuilding the controller,
`dmSyncopation` and `logdump`. Confirm that both independent matched populations
advance when both publishers are configured. The early observer can also be
qualified by itself; zero full-completion matches must not be presented as zero
SDK latency or as a successful SDK result.

## Compare an ordinary and an isolated real-time release waiter

The original observer runs at priority zero with no dedicated cpuset by default.
The primary monitor's `dmtweeter-synco` cpuset does not configure this separate
thread. The observer already blocks directly on the semaphore; there is no
polling interval to shorten.

Rebuild/install **`dmSyncopation` and `logdump`** for the split measurements.
The controller publisher protocol and code are unchanged by this measurement
update, so it does not require another `bmcCtrl` rebuild. First take a fresh
ordinary-observer capture with both new metric populations, using the current
`readerReleaseMonitor` settings. Then change only that monitor's `threadPrio`
to **30** and its `cpuset` to a verified unused dedicated physical core. Keep
the primary monitor, SDK worker, full-completion observer, camera, loop and
all timing reserves unchanged. The existing monitor configuration supports
these two settings; changing them requires restarting synco.

The additive profile
[`dmtweeter_synco_reader_rt.conf`](../dmtweeter_synco_reader_rt.conf) provides
priority 30 and the operator-confirmed **`spare57` / CPU57** assignment. The BMC
SDK worker uses **`spare56` / CPU56**. Merge the section into the existing
instance configuration; it does not replace the rest of that file. No cpuset
creation or instrument configuration change is performed by this update.

Use a different core from the main synco monitor, BMC command monitor and SDK
worker, and retain RTC latency tuning On. Record the selected cpuset and its
actual members. Requested priority in INDI alone does not prove successful
RT scheduling or CPU placement. After restart, obtain the current observer TID
from a bounded query:

```bash
timeout -k 2s 8s getINDI dmtweeter-synco.th-readerReleaseMonitor
```

Replace `CURRENT_TID` below with that numeric TID, then save these read-only
checks with the capture:

```bash
ps -eLo pid,tid,cls,rtprio,psr,comm | awk 'NR==1 || $2==CURRENT_TID'
taskset -pc CURRENT_TID
```

`cls` should report FIFO (`FF`) with priority 30, and the allowed CPU list should
contain exactly the selected core. Keep operational synco Off and the modeled
protection at 155 us. Wait at least ten seconds after restart, confirm advancing
matches and nonzero `wake_count` / `capture_count`, and repeat the bounded
120-second simultaneous capture. The capture block now also retains controller
INDI before/after, including SDK-worker diagnostic state.

Compare the new counters, both populations and worst-notification provenance.
A reduced `wake` tail with similar `capture` supports scheduling/placement as
the dominant source of this observer's earlier delay. A large `capture` tail
places the delay after wait return. Large `wake` still combines producer
publication/posting with wake scheduling; separating those requires a producer
post timestamp or trace. The two settings change scheduling and placement
together; a smaller follow-up experiment can separate their individual effects.
This test changes no live or shadow scheduling decision.
