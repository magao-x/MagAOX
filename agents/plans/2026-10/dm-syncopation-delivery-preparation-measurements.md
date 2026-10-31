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
