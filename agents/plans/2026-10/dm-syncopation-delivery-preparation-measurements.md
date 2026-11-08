# Measuring delivery and preparation before reducing shadow bounds

## Objective and starting evidence

The operator wants to keep reducing conservative, unmeasured allowances that
prevent the second auxiliary slot in the 6 kHz shadow. Retain the explicitly
chosen zero-added-main-SDK-wait policy and distinguish genuine 2 kHz measurements
from untested auxiliary-triggered behavior.

The pipeline preset initially assumed delivery 20 us, preparation 35 us, SDK
service 120 us, serialized post 150 us and total virtual-reader duration 60 us.
For a nominal 500 us main period, the final auxiliary trigger is at 333.333 us.
The existing earliest-main cutoff is 475 us; there are only 141.667 us available.
The fixed 175 us delivery/preparation/SDK path therefore misses by 33.333 us.
Individual pipeline stages can sustain a 166.667 us period while this conservative
wraparound test still rejects the final slot.

Optimized genuine-main controller captures provide:

| Measurement | `212117Z` | `234448Z` |
| --- | ---: | ---: |
| Largest paired receipt-to-SDK-entry duration, us | 22.342 | 21.901 |
| Smallest receipt-to-SDK-entry duration, us | 14.999 | 14.697 |
| Largest native-entry-to-worker-ready service duration, us | 113.174 | 109.247 |

A provisional **25 us preparation preset** leaves 2.658 us above the larger
observed prefix maximum. It is a shadow assumption requiring broader and
auxiliary-path qualification, not a proven worst-case execution bound. The
compiled default remains 35 us. SDK service stays 120 us; the prior 113.174 us
tail rules out selecting 110 us from the latest quiet run alone.

With preparation 25 us, the current full path is 165 us and the final slot still
misses by 23.333 us. Even zero delivery leaves 145 us, 3.333 us too long. Thus
preparation reduction is useful progress but cannot alone enable that slot.
Delivery measurement and SDK-bound qualification remain necessary; no implicit
minimum-preparation credit is granted to the next genuine main.

## Scope implemented

1. Add `preparation`, paired receipt to SDK dispatch, using existing monotonic
   endpoints. This separates pre-handoff preparation from `sdkDispatch`
   (dispatch to native entry), while `receiptSdk` retains their full paired span.
   No extra hot-path timestamp is needed for preparation.
2. Add explicitly enabled controller-local delivery measurements with a declared
   producer clock. Retain the original monotonic semaphore-receipt endpoint and
   bracket one producer-clock sample with it and a following monotonic read.
3. Export lower/upper delivery estimates, clock-read bracket, selected clock,
   qualification reasons, coverage and losses through existing DM timing
   telemetry and read-only INDI properties. Aggregate outside the command thread.
4. Reduce only the additive shadow preset preparation value to 25 us and document
   the next operator-run test. Delivery20/SDK120/post150/reader60, genuine-main
   anchoring, live scheduling and next-main protection remain unchanged.

No completion-wire version change, new thread, hardware operation, automatic
bound tuning or producer-clock substitution into monotonic scheduling is added.

## Clock and endpoint contract

MILK dmcomb uses `CLOCK_MILK = CLOCK_ISIO`; locally this resolves to CLOCK_TAI.
Other stream producers, including the MagAO-X framegrabber, explicitly use
CLOCK_REALTIME. A raw `timespec` cannot identify its producer clock. Generic
`dev::dm` therefore defaults `dm.deliveryClock=none` and accepts explicit
`isio`, `realtime` or `tai`. `isio` resolves the controller build's CLOCK_ISIO.
Unsupported clocks fail app configuration. `dm.deliveryBracketUs=1` is the
initial maximum accepted bracket; it is a diagnostic qualification setting.

At a genuine receipt, capture:

```text
M0 = existing CLOCK_MONOTONIC receipt
E  = selected producer-clock sample
M1 = CLOCK_MONOTONIC after that sample
```

For the same command's observed producer writetime P:

```text
upper   = E - P
bracket = M1 - M0
lower   = upper - bracket
```

All producer differences use the declared same clock; monotonic differences
measure uncertainty and retention. Capture never subtracts an absolute producer
writetime from a monotonic receipt. Qualified bounds cover the sample-clock
uncertainty, not all possible metadata/provenance errors. Metadata inside/after
the receipt uncertainty interval is rejected rather than silently clipping a
possibly newer publication to zero latency.

Qualification exposes missing/synthetic captures, invalid/backward clocks,
wide brackets, counter/publication continuity failures, clock-offset jumps,
future/ambiguous publication and publication age beyond the configured timing
window. A large positive TAI/REALTIME mismatch must not become a plausible
37-second delivery measurement. Wrong clock declarations cannot be fully
identified from metadata; explicit producer-clock configuration remains required.
The offset comparison is a clock-continuity check, not a scheduling conversion.

`dmTiming::Event` retains the raw capture through detached pipeline ownership.
Only the ordinary consumer updates delivery-quality counts and window
statistics. Disabled timing, disabled delivery and synthetic callbacks perform
no additional producer-clock reads. Existing `fgtimes` populations and stable
metric IDs retain their behavior; new IDs are appended after the original 24.

Observed inode/counter/writetime consistency is best-effort source association.
In particular, existing dmcomb publication does not provide an atomic pixel/
semaphore identity handshake. These measurements do not claim one. Queue drops
and rejected/missing samples must remain visible when interpreting tails.

## Readbacks and use

`dm_timing_stages` adds `preparation`, `deliveryLower`, `deliveryUpper` and
`deliveryBracket` with the usual mean/jitter/min/max/count/span suffixes in
seconds. `dm_timing_delivery` exposes enabled state, numeric resolved clock,
maximum bracket in microseconds and cumulative qualification accounting.
`dm_timing_state.deliveryClock` names the resolved clock. `telem_dmtimings`
appends an optional structured delivery-quality table; old metric sets remain
readable and the new writer's full 28-row set carries explicit quality even
when delivery capture is disabled.

The downstream measurement is combined publication to BMC receipt. Existing
`dmCombTiming::primaryPublication` measures linked upstream update to combined
publication in the common producer clock. Together they characterize the
**genuine primary path**; independent maxima can form a conservative diagnostic
sum over those observed populations, but do not assert exact per-command pairing
or qualify a hypothetical auxiliary-trigger path.

Do not re-anchor genuine modeled mains using completion-observer receipt times.
Those belong to synco, not BMC. Reader/full completion `startedNs` is callback
capture after receipt; SDK-result `startedNs` is SDK entry. Neither can simply
replace the original controller receipt. Replaying older actual endpoints into
an already-advanced counterfactual model would require separate causal/replay
work and must not retroactively change committed auxiliary decisions.

## Operator test and verification

The concrete procedure is in
[delivery-preparation-test.md](../../../apps/dmSyncopation/doc/delivery-preparation-test.md).
It requires rebuilding/installing the controller and logdump, explicitly enabling
producer-clock delivery capture, restarting through the normal operator process,
merging the 25 us shadow preparation preset and repeating the simultaneous
capture with the same camera/loop/CPU/workload settings. Agents do not run any
instrument/INDI commands. This test leaves actual auxiliary writes disabled.

Verification must cover legacy metric/writer compatibility, paired preparation
endpoints, disabled capture overhead, clock/bracket failures and discontinuities,
synthetic/future/stale/changed metadata, detached-event ownership, queue-loss
coverage, all qualification reason counters, malformed telemetry, INDI/config
readbacks and production controller/reader compilation. Hardware measurements
will determine subsequent delivery/SDK assumptions; neither is reduced here.

## Verification outcome

All focused software suites pass with the project optimization flags:

| Suite | Cases | Assertions |
| --- | ---: | ---: |
| DM timing collector | 30 | 833 |
| DM callback/configuration/INDI integration | 11 | 365 |
| DM timing telemetry | 11 | 2,561 |
| BMC SDK-stub worker/pipeline regression | 30 | 1,351 |

The collector tests verify paired preparation, strict latency/bracket arithmetic,
all disjoint capture outcomes, offset discontinuities, future/stale/reused input,
configuration/epoch/gap recovery, detached ownership and capture failure/errno.
An isolated clock wrapper verifies that disabled/synthetic capture reads no
additional clocks, while enabled capture reads receipt/producer/after endpoints.
Integration tests validate clock parsing, invalid configuration, unavailable
sentinels, delivery quality and the new statistics through a local INDI sink.
Two existing exact floating-point INDI assertions were changed to the established
approximate comparison convention after serialization rounding produced an
otherwise identical printed value.

Telemetry tests cover current 28-row and historical 12/13/18/20/24-row sets,
malformed clock/coverage/population fields and sparse/disabled data. An actual
collector summary with three qualified, one future-publication and one wide-
bracket event round-trips through binary verification, embedded-schema JSON,
text and metadata accessors; expiration removes window statistics while retaining
lifetime quality. Independent review found no blocking issue in clock arithmetic,
causal ordering, default overhead, detached transport or command behavior.

Library/PCH and production logdump builds pass. The full bmcCtrl translation unit
passes syntax validation with the repository's SDK declaration stub, and the BMC
worker/pipeline SDK-stub suite passes. A native vendor-SDK build was attempted
but this workstation lacks `BMCApi.h` under `/opt/Boston Micromachines`; native
linking therefore remains a check on an SDK-equipped machine. No BMC SDK API call
was modified. No instrument commands, processes or hardware were accessed.

C++ formatting, relative documentation links, preset arithmetic and diff
whitespace checks pass. Source and documentation are ready for the operator-run
capture; no hardware latency or 6 kHz qualification result is inferred yet.

## Corrected 2 kHz capture: `synco-sim-20261031T171940Z`

Capture checkout is `2d8c104a5662f867ac64e5e80575b92b42dc413f`. Saved setup is
bin2, 2 kHz, synchronization On, EM gain 1, target −45 C, closed loop at scalar
gain 1 and RTC latency On. The operator reran after correcting camera startup
speed. Modal gains, live dmcomb settings and complete CPU/workload state are not
independently encoded in this archive. All collectors finish at expected timeout
124, all INDI reads succeed, and no reader verification failure is reported.

### Delivery and preparation measurements

There are 121 consecutive valid controller summaries, sequence 1334–1454,
**17:19:39.396043131–17:21:39.659215443 UTC** (120.263172312 s; October 31,
10:19–10:21 Phoenix time). Timed receipts/submissions increase by **240,523**.
Producer-clock delivery capture resolves to **CLOCK_TAI**, maximum bracket 1 us.
Delivery qualification increases by **240,522** of **240,523** attempts, with one
identity rejection. No new wide-bracket, invalid-clock, clock-discontinuity,
future-publication or unavailable captures occur in the text-summary interval.
The largest qualified bracket is **211 ns**, with median rolling mean **47.36 ns**.

| Population | Median of rolling means, us | Largest window maximum, us |
| --- | ---: | ---: |
| Controller preparation: receipt to worker dispatch | 13.489 | 18.044 |
| Receipt to native SDK entry, including handoff | 16.683 | 21.751 |
| Worker dispatch to native entry | 3.203 | 7.464 |
| Combined publication to controller receipt: upper estimate | 4.655 | 6.963 |
| Native entry through SDK-worker readiness | 106.857 | 112.592 |
| Controller reader release | 21.940 | 29.014 |

The new downstream delivery measurement is substantially smaller than 20 us,
but the modeled 20 us is the **whole trigger-to-controller path**. Existing
linked dmcomb `primaryPublication` has median rolling mean **11.378 us** and
maximum **17.984 us**. It starts at the upstream trace update, slightly before
actual primary publication. It is therefore a conservative upstream duration,
not an exact virtual-auxiliary trigger interval.

The independent maxima sum to **24.947 us**; they occur in different reporting
windows and are not a paired per-command full-delivery maximum. Typical rolling
means sum to roughly 16.03 us. These measurements identify the components but do
not justify replacing the whole 20 us assumption with the 6.963 us downstream
maximum. Exact consumed-frame/token association and virtual-auxiliary delivery
remain unqualified.

The 25 us preparation assumption still covers the 21.751 us receipt-to-SDK
maximum, including handoff; the earlier optimized 2 kHz maximum remains 22.342 us.
Do not replace it with pure preparation's 18.044 us maximum without separately
representing worker handoff. SDK service stays below 120 us in this capture.

### Shadow accounting

119 dmsynco summaries span **17:19:39.917241248–17:21:39.253449984 UTC**,
119.336208736 s. The model uses preparation25/delivery20/SDK120/post150/reader60 us.

| Step | Considered | Would run | New skips |
| --- | ---: | ---: | --- |
| 0 | 238,672 | 238,668 | 4 missing release |
| 1 | 238,672 | 238,665 | 4 missing release; 3 SDK busy |
| 2 | 238,672 | 0 | 4 missing release; 238,668 next-main protection |

Every row conserves outcomes/reasons and resource credits. Admissions add
238,672 mains + 238,665 auxiliaries; retirements add 477,337 with unchanged
endpoint occupancy. Maximum modeled occupancy stays two of three credits.
There is no current-session main SDK wait, credit miss, resource reset or discard.
Lifetime main-wait trace detected/drained counts are already 26 before collection
and do not increase; dropped, queued and logFailed remain zero. No blocker event
is exported. Two lock-contention emissions omit a current queue snapshot; the
other 117 and final lifetime accounting retain complete zero-new-event coverage.

Maximum prospective clearance shortfall changes **23.463 → 23.481 us**, consistent
with the nominal 23.333 us wraparound deficit and frozen-grid deviations. This
remains conditional approximately 4 kHz behavior, not a three-slot/6 kHz success.
At 17:19:59.110589250, the four missing-release increments coincide with one new
readiness gap and one early/full-completion observer counter gap. Primary,
combined, upstream and camera observer quality otherwise remains clean.

### Actual post-worker pressure is now the limiting finding

Although handed-off timing events have no new failed/gapped/dropped counts,
actual controller INDI counters show **three new full-credit overloads**:

- pipeline `overloaded`: **411 → 414**;
- worker `rejected`: **412 → 415**;
- worker `coalesced`: **411 → 414**.

The initial counts are historical. Native failures/timeouts, preparation/post
failures, cancellations and supersessions do not increase. The runtime INDI
interval and text interval have different boundaries; INDI admitted/completed/
retired deltas are 238,519, distinct from the 240,523 timed-command text delta.

Post-processing wall-time maxima are:

| Population | Median of rolling means, us | Largest window maximum, us |
| --- | ---: | ---: |
| Delta/difference publication | 8.780 | 536.513 |
| Saturation classification | 3.546 | 2,558.901 |
| Receipt through callback finish | 143.435 | 2,697.954 |

The largest saturation/delta/finish tail is reported in the
**17:19:59.440147055** controller summary. A roughly 2.7 ms held credit at 500 us
main cadence can fill a three-credit pool and reject subsequent inputs; the
observed three overloads are consistent with this mechanism. Aggregate extrema
and endpoint counters do not identify the exact overloaded jobs, so coincidence
is not exact causal reconstruction.

The ordinary post worker reports policy0, priority0 and **60 allowed CPUs**
(the count is not CPU60), whereas the SDK worker is FIFO55 with one allowed CPU.
The saturation interval encloses fixed-size fill/classification and a counter
update; wall time includes scheduling/preemption and is not a measurement of
2.56 ms CPU execution. Post-worker scheduling is a concrete candidate to test.
Existing `sdk.pipelineThreadPrio` and `sdk.pipelineCpuset` can place it at FIFO30
on a dedicated unused core without a code change. Keep the SDK worker, reader
observer, pipeline capacity and all model bounds unchanged for that comparison.
The unused cpuset must be confirmed by the operator before selecting it.

The apparent clean `dmtimes` input-gap count does not negate admission loss:
`claimPipelineJob()` increments overload/rejection before `discardPipelineCommand()`
detaches that receipt without publishing a timing event. Ordinary timing counter
pairs also require contiguous event sequences. Admission counters therefore
remain necessary alongside timing-event quality. The 150 us post assumption is
not qualified by this run; tightening it or treating the current model as a
complete physical loss predictor would be premature.

### Camera baseline and next comparison

114 consecutive valid camera stage summaries cover **239,582** new frames with
no new frame/counter gaps, failures, bracket-wide events or drops. Acquisition
intervals are **480–525 us**, rolling RMS **0.236–0.809 us**; publication intervals
are **477.882–525.873 us**, RMS **0.859–1.250 us**. Eleven qualified excursion traces
(four late, seven early) overlap serial activity; twelve normal serial controls
have no new errors. These populations do not show the controller's millisecond
post tail.

Next compare post-worker placement/priority, then repeat the same simultaneous
capture with the same camera/loop and shadow bounds. No source or bound change
is inferred from this capture. Address actual post-credit pressure before
qualifying a 6 kHz workload or reducing the post-service allowance.

### Selected post-worker comparison

The operator confirmed **spare31 is free**. The next comparison places the
ordered post worker at **FIFO30 / spare31 (CPU31)** using the existing startup
settings `sdk.pipelineThreadPrio=30` and `sdk.pipelineCpuset=spare31` in the
controller's dmtweeter.conf. This is a proposed operator-applied test, not an
agent-performed hardware/configuration change. SDK FIFO55/spare56, early-release
observer FIFO30/spare57, three-credit capacity and all shadow bounds remain as
before. The current executable already supports these keys, so only the normal
controller restart is required. Verify actual policy1/priority30/allowedCpus1,
allow stable reacquisition, and repeat the same simultaneous capture. The
[procedure](../../../apps/dmSyncopation/doc/delivery-preparation-test.md#post-worker-scheduling-comparison-on-spare31)
now includes the exact additive config and checks. This combined policy/placement
test does not isolate those two effects independently.

## FIFO post-worker comparison: `synco-sim-20261031T173634Z`

Capture checkout remains `2d8c104a5662f867ac64e5e80575b92b42dc413f`; this is a
configuration/restart comparison, not a new source build. Camera snapshots match
the corrected baseline: bin2, 2 kHz, synchronization On, EM gain 1, target −45 C,
closed loop scalar gain 1 and RTC latency On. All collectors end at expected
status 124 and all INDI snapshots succeed. The selected operator configuration
is FIFO30/spare31. Runtime readbacks independently confirm **policy 1, priority 30
and one allowed CPU**; the archive does not contain the actual affinity list
needed to identify that CPU numerically as31. SDK remains FIFO55/one CPU.

### Actual controller improvement

121 consecutive valid DM summaries, sequence 67–187, span
**17:36:33.069027171–17:38:33.346033486 UTC** (120.277006315 s; October 31,
10:36–10:38 Phoenix time). Text endpoint totals add **240,554 successful timed
commands**, without new timing failures, counter gaps or drops.

| Maximum duration, us | Ordinary post baseline `171940Z` | FIFO30/single-CPU `173634Z` |
| --- | ---: | ---: |
| Saturation classification | 2,558.901 | **4.018** |
| Delta/difference publication | 536.513 | **14.047** |
| Delta wall span | 2,678.477 | **131.459** |
| Receipt through measured callback finish | 2,697.954 | **150.133** |
| SDK service through readiness | 112.592 | 112.212 |
| Reader release | 29.014 | 29.185 |
| Receipt to worker dispatch | 18.044 | 21.039 |
| Receipt to native SDK entry | 21.751 | 24.205 |

Runtime INDI endpoint deltas show **no new full-credit overload, rejected
admission or coalesced input**: overloaded 0 and coalesced 0 stay zero; rejected 1
is historical and unchanged. Admitted/completed/retired each advance 240,553,
with endpoint in-flight occupancy unchanged at 1. No native failure, timeout,
preparation/post failure, cancellation, supersession or unqualified reader grows.
Different text and INDI endpoints explain the one-command difference.

These results support keeping the dedicated FIFO post-worker configuration.
The combined policy/placement/restart intervention removes the observed
millisecond tails and admission overloads in this short comparison. It cannot
separate priority from CPU-placement effects or guarantee that rare tails are
gone under other loads. In particular, **150.133 us is the whole callback span**,
not a direct measurement of the configured 150 us serialized-post bound. SDK-
ready through final credit retirement still needs its own paired measurement
before that assumption can be tightened mechanically.

### Delivery/preparation envelope

Median rolling means are preparation 13.406 us, complete receipt-to-SDK 16.489 us,
worker handoff 3.102 us, SDK service 106.805 us, reader release 21.913 us,
publication 7.762 us, saturation 2.691 us and receipt-to-finish 139.806 us.
The largest full prefix **24.205 us** leaves only **0.795 us** below the current
25 us preparation/handoff assumption. Pure preparation's 21.039 us maximum
excludes handoff; it is not a substitute for that complete prefix. Keep the25 us
preset provisional and retain 120 us SDK service; older 113.174 us service tails
remain part of the evidence.

Producer-clock delivery capture qualifies **240,512 of 240,554** attempts.
There are 42 new wide-bracket rejections at the 1 us limit, concentrated around
17:37:26–39, with no clock discontinuity, future-publication, invalid, unavailable
or identity rejection. The qualified bracket's median rolling mean is 47.65 ns
and maximum 642 ns. Rejected samples remain part of coverage; reported qualified
maxima do not assert an envelope for those omitted captures.

Combined-publication-to-controller upper delivery has median rolling mean
**4.611 us** and maximum **11.341 us**. Linked upstream primary-update through
combined publication has median rolling mean **11.499 us**, maximum **23.324 us**.
The independent maxima sum 34.665 us, but belong to different commands/windows;
the upstream start also precedes actual primary publication slightly. This is
not a paired end-to-end trigger maximum and does not establish a replacement
for the full 20 us delivery assumption. The useful next delivery diagnostic is
same-command upstream/receipt/SDK-ready timing, preserving lineage and clock
uncertainty, rather than adding unrelated extrema.

### Shadow outcome and camera baseline

120 dmsynco summaries span **17:36:34.132878142–17:38:34.273721635 UTC**,
120.140843493 s. Model assumptions remain preparation 25/delivery 20/SDK 120/
post 150/virtual-reader 60 us, three steps and zero-added-main-SDK-wait protection.

| Step | Considered | Would run | New skips |
| --- | ---: | ---: | --- |
| 0 | 240,282 | 240,282 | 0 |
| 1 | 240,282 | 240,278 | 4 SDK busy |
| 2 | 240,282 | 0 | 240,282 next-main protection |

All rows conserve outcomes and credits. Admissions add 240,282 mains + 240,278
auxiliaries, retirements 480,560 and endpoint occupancy is unchanged. No new
modeled wait, credit miss, resource reset, discard, missing release, readiness
gap or observer-quality error occurs. Readiness qualification advances 240,282.
Lifetime main-wait trace detected/drained 26, resource credit misses 23, resource
resets 21 and maximum clearance shortfall 41.392 us are already present in the
first row and do not grow. No main-wait event is exported and no trace is lost.
The four missing-release skips from the prior baseline do not recur. Three
versus four rare SDK-busy vetoes in different windows is not evidence of a
regression.

Camera telemetry has 114 consecutive valid summaries and 238,515 new frames,
without new frame/counter gaps, failures, bracket-wide events or drops.
Acquisition intervals are 480–525 us, RMS 0.220–0.792 us; publication intervals
478.473–526.304 us, RMS 0.871–1.186 us. Six qualified excursion traces (one late,
five early) overlap serial-query activity.

Retain FIFO30/spare31 for subsequent comparisons. The final auxiliary is still
vetoed by SDK clearance at the protected next-main boundary; improving post
retirement alone does not remove that check. No shadow duration or live-policy
change follows from this capture. Further reduction should use paired total-path
and credit-retirement measurements; physical 6 kHz remains unqualified.
