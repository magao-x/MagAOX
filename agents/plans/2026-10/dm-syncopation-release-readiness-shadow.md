# Signal-qualified reader readiness and three-step shadow test

## Authorized objective

On 2026-10-29 the operator selected `dmtweeter_read_done` as the reader-readiness
signal and requested evaluation of 6 kHz total cadence at a 2 kHz main rate.
The selected milestone is **shadow at three steps first**, not a live scheduler
revision or permission for agents to operate hardware. This follows the
[worker and observer measurements](bmc-sdk-worker-reader-release.md#16-fifo30--cpu57-capture-20261029t145715z).

The measured release requirement is about 20 us, with a current-run maximum
25.708 us and earlier observed maximum 28.303 us. Release observation on the
verified FIFO30 / CPU57 monitor takes about 3.681 us after the endpoint; its
current maximum is 8.086 us, with an older lifetime maximum 22.172 us. SDK
service remains about 104 us and the full callback about 131 us. Earlier
channel clearance does not make the SDK accept another command sooner.

## Fixed-grid geometry and independent constraints

At a 500 us main period, three total steps have active trigger targets at
166.667 and 333.333 us. With the existing 10 us copy allowance their copy
starts are 156.667 and 323.333 us. Existing early-main 20 us, observation 5 us,
visibility 1 us, passive lead 20 us and preparation reserve 2 us are unchanged.
The next passive copy targets 472 us and starts 462 us.

The last active command therefore has about 128.667 us before the next passive
copy starts, but only **141.667 us before the earliest modeled next main**.
Those are different constraints. An early read release can resolve the first
without resolving whole-backend occupancy before the next main.

The legacy full timing bundle requires a 193 us slot: 155 us reader hold,
10 us copy, 26 us tail and 2 us reserve. It remains unchanged for ordinary
shadow and live scheduling. The new shadow mode separates:

- Actual-main reader clearance: an exact qualified release observed by R,
  plus the existing configured guard, must precede the fixed copy start.
- Virtual auxiliary reader duration: explicitly assumed 30 us after its
  hypothetical trigger, plus the existing 20 us guard.
- Whole backend occupancy: explicitly assumed **155 us**, including delivery
  and callback retirement, independently of reader clearance. Previous backend
  work must fit before an active target, and a new active command must retire
  before the earliest modeled next main.

Keeping the conservative backend assumption at 155 us makes the nominal final
three-step slot fail its backend wraparound test: 333.333 + 155 exceeds 475 us.
The reader/copy geometry can fit while full-cycle backend geometry does not.
Do not silently lower backend occupancy or the undecided early-main reserve to
turn that failure into a successful 6 kHz claim. Explicit smaller backend
assumptions are supported for counterfactual shadow tests, not qualified as
hardware bounds. A same-frame backend retirement measurement or a separately
reviewed bound is needed before interpreting all three slots as deliverable.

## Actual release evidence

The release monitor copies a fixed coherent record with its local observation
time R and attachment generation into a preallocated SPSC handoff. It does not
take the simulation cycle mutex and does not wait for ordinary app logic. All
records, including incoherent/busy/unqualified evidence, are transported so
revocation is not hidden by filtering for successes alone.

The cycle owner drains a bounded queue population before primary, combined or
ordinary advance evaluations, holding one future-R record for a later update.
Each call handles at most 64 records, including a carried lookahead, so a
diagnostic-capacity backlog cannot become an unbounded monitor callback.
Attachment changes, model resets and transport loss fence queued evidence by
generation; a second reset/loss check closes readiness if either races the
drain. Queue allocation occurs only before producer startup, never during a
camera-rate/model reset.

The pure model retains bounded exact combined-input identities and release
history. Qualified readiness requires readers protocol, released state, a
genuine coherent current publisher/session/epoch, exact combined inode/counter/
raw writetime, consistent monotonic endpoints and a release observation available
before the fixed decision. R is used for availability, not the earlier endpoint
F or wait-return W. A late release cannot retrospectively turn an already
skipped occurrence into a successful copy, and skips never move later targets.

Invalid source evidence, unexpected progression or a source reset revokes
clearance. SDK failure is independent: finished reads are still a real endpoint,
but a backend fault closes further hypothetical active operations until recovery.
A later SDK-result enrichment of the same release is not another readiness
event and does not clear a newer or virtual job. A forward publisher restart
can establish a new epoch; backward publisher replay cannot.

Only real main commands produce physical release records during this test.
Virtual auxiliaries have no observed completion, so the explicit virtual-reader
and backend assumptions remain after each hypothetical auxiliary trigger. A
main release must never be reused as acknowledgement of a nonexistent virtual
auxiliary command.

Timestamp-ordered release observations advance already due slots strictly
before their own R, then apply their new readiness or revocation. This preserves
an earlier valid decision when an invalidation arrives later, while an
invalidation before a copy deadline closes that slot. A late event never
reclassifies an already counted occurrence.

Actual main arrivals are also checked against modeled backend ownership after
virtual readers clear. Backend conflicts have a distinct overlap window and
exact backend endpoint. The last active backend retains its own provenance,
because passive step-zero preparation can finish while that earlier operation
is still busy. Reset/silence clears both virtual ownership states. A reader
release or later passive copy cannot hide an auxiliary-backend/main collision.

## Configuration, readbacks and operator sequence

The new startup-only controls are:

| Key | Default | Meaning |
| --- | ---: | --- |
| `simulation.readerReleaseReady` | false | Use exact observed actual-main release in the shadow model. Requires simulation and a configured release monitor. |
| `simulation.virtualReaderUs` | 30 | Assumed virtual auxiliary reader duration before the unchanged guard. |
| `simulation.backendUs` | 155 | Assumed complete backend occupancy; independent of the read release. |

Legacy mode and live behavior remain unchanged. Readiness mode adds separate
reader-cycle and backend-cycle feasibility, raw release evidence and quality,
and per-step `releaseMissing`, `releaseLate`, `backendBusy` and `backendWrap`
skip reasons. Existing `readerHold` covers remaining modeled virtual reader
ownership. Overall feasibility reports the conjunction of the independent
constraints, while per-step outcomes remain evaluable when the whole requested
cycle cannot fit.

The optional telemetry readiness table distinguishes real-main evidence from
virtual assumptions, preserves exact integer identity/time fields and retains
old-record compatibility. The ordinary joined `reader_release_timing` summaries
remain diagnostics and are never the readiness authority.

The operator rebuilds dmSyncopation and logdump, keeps operational synco Off,
and retains the working controller, camera, loop, RTC latency and CPU placement.
The SDK worker remains on spare56 and the RT release monitor on spare57.
The additive three-step profile requests the candidate rate without changing
delivery, keepout, guard, early-main or copy settings. Begin with the new
readiness mode at two steps to compare against the existing model, then request
three and retain per-reason outcomes and both feasibility values. Continue to
use the bounded simultaneous capture with before/after controller diagnostics.

No agent commands are executed on the instrument. Actual auxiliary triggering,
the revised fixed-grid live scheduler and mode-3 primary-semaphore wake routing
remain separate implementation and qualification milestones. A raw 6 kHz
`dmtweeter_read_done` counter rate still represents 2 kHz commands under the
three-update protocol; it is not evidence of 6 kHz physical DM submissions.

## Implementation and verification

Implemented the raw fixed-record transport, per-call limit of 64 including a
carried lookahead, attachment/drop generation fencing, event-time qualification,
independent virtual ownership and backend conflict detection. The live scheduler
and default legacy shadow mode are unchanged. New configuration, readbacks,
backward-compatible telemetry and the additive candidate profile are described
in the [operator procedure](../../../apps/dmSyncopation/doc/readiness-shadow-test.md).

Deterministic tests exercise exact availability plus guard, releases arriving
before combined association or after deadlines, revocation before/after an
already committed decision, source replacement/replay, queue overflow/reset,
nonblocking observer capture, bounded draining including future lookahead,
genuine vs virtual ownership, SDK faults/enrichment and fixed-grid independent
skips. Backend conflicts are tested both after reader clearance and after a
subsequent passive copy has replaced latest-copy provenance; interval ends
remain exclusive. Three-step tests preserve the 155 us final-slot rejection
and separately exercise explicit smaller assumptions and nanosecond boundaries.

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| Pure shadow engine | 53,767 | 49 |
| Synco application integration | 11,347 | 77 |
| Synco telemetry and compatibility | 6,421 | 33 |
| **Total** | **71,535** | **159** |

Fresh library/PCH and generated schema/dispatch, normal dmSyncopation and
logdump builds pass. Telemetry tests cover optional-field legacy compatibility,
exact full-width JSON provenance, exclusive skip accounting and malformed
readiness/backend evidence. Whole changed C/C++ files pass clang-format;
whitespace and operator-guide Bash syntax/local links are checked. These are
workstation checks only; no instrument action or 6 kHz hardware qualification
was performed by the agents.

## First two-step readiness capture: 20261030T024932Z

The operator identifies this as `maxSubframes=2`, with the three-step capture
still in progress. Source: `synco-sim-20261030T024932Z.tgz`, extracted at
`/home/jrmales/tmp/home/jrmales/tmp/synco-sim-20261030T024932Z`.
The 120 synco records span **2026-10-30 02:49:31.435598563 to
02:51:31.647531800 UTC**, **120.211933 seconds**, or October 29,
19:49:31–19:51:31 Phoenix time. All snapshots succeed, all error files are
empty and the bounded collectors return expected timeout status 124. The
saved checkout is `14ad6cd638f3de808b72b5b296d7e9f32638c635`, not independent
running-binary provenance.

### Readiness and two-step outcomes

The records select `shadow_reader_release_ready`, two steps, with valid grid
and both reader-cycle and backend-cycle feasibility throughout. Operational
synco stays Off; no real auxiliary writes or deliveries are indicated. The
model retains virtual reader 30 us plus guard 20 us, independent backend 155 us,
and early reserve 20 us. The release monitor requests priority 30, with the
full-completion observer at priority zero. Current actual thread policy and
affinity are not independently saved in this capture.

| First-to-last telemetry difference | Step 0 | Step 1 |
| --- | ---: | ---: |
| Considered | 240,423 | 240,423 |
| Would run | **240,423** | **240,419** |
| Skipped | 0 | **4** |
| Backend busy | 0 | **4** |
| Reader hold / release missing / release late / backend wrap | 0 | 0 |
| Overlap | 0 | 0 |

The active skip fraction is **0.001663734%**. The four increments first appear
at 02:50:31.023934151, 02:50:52.242586256, 02:51:16.487456311 and
02:51:25.582741774 UTC. These are conflicts with the **assumed** backend
envelope, not observations that the SDK was busy or failed. At the nominal
250 us active target, a 155 us backend bound requires
`max(mainObservation, combinedObservation) <= grid + 95 us`.
The retained primary phase maximum is only 41.709 us, so delayed combined
observation or cycle-lock acquisition is a plausible explanation. Exact
per-skip provenance is not retained, so the responsible event and exact delay
cannot be established from aggregate windows alone.

The raw gate adds **480,846 observations and 240,423 qualified releases**,
exactly two records per command for release and SDK-result enrichment. There
are **no new** invalid/incoherent records, duplicates, progression gaps, source
resets, identity mismatches, SDK failures or transport drops. Queue readbacks
are zero at both boundaries. Source epoch 5, publisher session
2120408464588493, producer epoch 18, PID 788488 and combined inode 4685 stay
stable. Retained input counter and sequence each advance exactly 240,423.
All 120 latest-evidence samples satisfy S <= F <= R <= model evaluation time.
They retain the initial unknown SDK outcome and unavailable raw result after
enrichment, as required for truthful first-observation provenance.

Historical invalid=3 / sourceResets=4 and approximately 61,744 per-step
release-missing skips already exist at the first row and remain unchanged.
They must not be counted as losses during this capture. Before/after INDI
independently adds 238,405 qualified releases, the same number of passive
would-run results, and 238,401 active would-run results with four skips.

### Reader timing and exact paired gain

The table summarizes medians of overlapping two-second means and maxima
retained in those windows, not pooled per-command distributions.

| Metric | Early typical, us | Early maximum, us | Full typical, us | Full maximum, us |
| --- | ---: | ---: | ---: | ---: |
| Required hold | 20.019172 | **29.466** | 131.194969 | **140.676** |
| Processing to endpoint | 19.889490 | 29.376 | 131.262730 | 140.766 |
| Hold until observation | 23.647830 | 33.463 | 135.144990 | 318.341 |
| Notification | 3.632128 | **10.490** | 3.968425 | 189.748 |
| Endpoint to wait return | 3.435597 | 10.289 | 3.730428 | 189.578 |
| Wait return to record read | 0.196750 | 5.370 | 0.230320 | 18.535 |

Both required-hold maxima occur on **the same command**, first visible at
**02:50:05.771936636 UTC**:

```text
Primary inode 4676 / counter 228959219
Combined inode 4685 / counter 259668039
Publisher PID 788488 / epoch 18 / attempt sequence 44697532

Baseline B = 2196444319127300 ns
Early F    = 2196444319156766 ns
Full F     = 2196444319267976 ns

Early required hold =  29.466 us
Full required hold  = 140.676 us
Full F - early F    = 111.210 us
```

This is a genuinely paired early release, with unknown SDK outcome in the
initial early record and successful SDK outcome at full completion. The new
29.466 us maximum is inside the capture and supersedes the earlier 28.303 us
observed maximum; it reinforces using the signal rather than treating 30 us
as an established fixed operating bound. Actual-main decisions use the release
plus guard, while the virtual 30 us duration remains an explicit assumption.

The worst early notification inside the capture is **10.490 us**, exactly
**10.289 us before wait return + 0.201 us afterward**, first reported at
02:49:54.661946965 UTC. The separate 5.370 us post-wait maximum is from another
event and must not be added to the wake maximum. The worst full notification
is **189.748 us = 189.578 us before wait return + 0.170 us afterward**, inside
the capture. Notification tails remain dominated by the pre-return interval.

The early observer adds 240,423 matches, 480,847 notices and 240,424 SDK
successes; full completion adds 240,422 matches from 240,423 primary/combined
observations. One-event differences reflect boundary/in-flight phases. Neither
observer adds invalid, busy, incoherent, gap, missing-peer, queue-drop, receipt
or first-enriched exclusions. Old startup quality losses and one early
first-enriched observation are unchanged history. No new over-budget joins
are reported against the retained 155 us diagnostic comparison.

### Controller, camera and cadence

Controller telemetry adds **238,416 successful submissions** with no new
errors, rejections, counter gaps or drops. SDK-worker snapshots add 238,415
completed commands; fault, failures, timeouts, admission-closed and coalesced
counts stay zero, with rejected receipts unchanged at seven. Actual worker
policy/priority remain FIFO55, with one CPU permitted. Typical SDK time is
103.994118 us and callback duration 130.835140 us; direct entry-to-release
maximum is 29.205 us. The physical stream remains at about 2 kHz, not the
conditional two-step 4 kHz request.

Camera stage telemetry adds **234,505 frames** and 11 successful serial
controls, without new failures, frame/counter gaps or drops. Typical acquisition
interval sigma is 0.297354 us and publication sigma 0.874284 us, similar to
145715Z. Recorded configuration also matches: bin2, 2 kHz, EM gain 1,
synchronization On, shutter SHUT, target -45 C; loop closed at scalar gain 0.05,
RTC latency On. Modal gains are not captured. Large historical camera/controller
telemetry-drop totals predate the first records and remain unchanged, so this
result establishes capture-local coverage rather than lifetime cleanliness.

The ten-second cadence estimate remains full with 64 anchors, no rejects and
period 500000.086178–500000.914713 ns. Sampled phase is -6.169 to +6.432 us.
Primary, combined and pipeline diagnostics add no gaps or invalid evidence.
No overlap/estimator events are triggered or dropped. This validates the
two-step release-qualified baseline; the three-step capture is a separate
comparison and is not inferred from these results. No runtime policy or
instrument setting is changed by the analysis.

## Three-step readiness capture: 20261030T025406Z

The operator identifies this as the three-frame comparison. Source:
`synco-sim-20261030T025406Z.tgz`, extracted at
`/home/jrmales/tmp/home/jrmales/tmp/synco-sim-20261030T025406Z`.
The 120 synco records span **2026-10-30 02:54:05.060525335 to
02:56:05.196148246 UTC**, **120.135623 seconds**, or October 29,
19:54:05–19:56:05 Phoenix time. Snapshot commands succeed, error files are
empty and collectors return expected timeout status 124. The saved checkout
is again `14ad6cd638f3de808b72b5b296d7e9f32638c635`; this does not independently
identify the running binary.

Saved settings match the two-step capture except steps 2 to 3 and the synco
restart. Backend 155 us, virtual reader 30 us, guard 20 us, early reserve and
RTC latency are unchanged. Live synco remains Off. The release observer
requests priority 30 and full completion priority zero; actual observer policy
and affinity are not independently saved. Actual DM commands remain at 2 kHz.

### Three-step outcomes and limiting constraint

| First-to-last telemetry difference | Step 0 | Step 1 | Step 2 |
| --- | ---: | ---: | ---: |
| Considered | 240,271 | 240,271 | 240,271 |
| Would run | **240,271** | **97,901** | **0** |
| Skipped | 0 | **142,370** | **240,271** |
| Backend busy | 0 | **142,370** | 0 |
| Backend wrap | 0 | 0 | **240,271** |
| Reader hold / release missing / release late | 0 | 0 | 0 |
| Overlap | 0 | 0 | 0 |

Step 1 would run **40.746074%** of its opportunities and skip 59.253926%;
step 2 always skips. The grid stays valid, with `readerCycleFeasible=1` and
`backendCycleFeasible=0` throughout. These are conflicts with the explicit
**155 us assumed backend envelope**, not measured SDK-busy events or SDK
failures. Early channel release is working; it does not clear this independent
backend constraint.

The nominal first active target is 166.667 us. Retiring the actual main's
assumed backend before that target requires
`max(mainObservation, combinedObservation) <= grid + 11.667 us`.
This much tighter observation allowance accounts for the large increase in
`backendBusy` skips relative to the two-step target at 250 us. Aggregate
outcomes do not retain exact provenance for each rejected occurrence, so they
do not establish which observation caused each skip.

The final active target is 333.333 us, with only **141.667 us** before the
earliest modeled next main at 475 us. Its assumed completion at 488.333 us
exceeds that boundary by **13.333 us**, explaining the systematic `backendWrap`
rejection. Reader clearance cannot fix this arithmetic. The corresponding
conditional total rate is about **2,814.9 Hz**, including approximately
814.9 Hz of hypothetical auxiliary commands. The two-step run predicted
almost 4 kHz; requesting three steps currently reduces modeled throughput
because the earlier active target rejects many commands and the final target
cannot fit. Neither run delivers real auxiliary commands.

The raw readiness gate adds **480,542 observations and 240,271 qualified
releases**, exactly two records per command. Invalid, incoherent, duplicate,
gap, identity-mismatch, SDK-failure and queue-drop counts remain zero. Source
epoch 5, publisher session 2120408464588493, producer epoch 18, PID 788488 and
combined inode 4685 stay stable. Three startup source resets and 370 historical
release-missing skips per step predate the first record and do not increase.
All 120 sampled evidence records satisfy S <= F <= R <= model evaluation time.

### Reader timing and observer delivery

These are medians of overlapping two-second window means and maxima retained
in those windows, not pooled distributions.

| Metric | Early typical, us | Early maximum, us | Full typical, us | Full maximum, us |
| --- | ---: | ---: | ---: | ---: |
| Required hold | 20.129086 | **24.967** | 131.258332 | **136.307** |
| Processing to endpoint | 19.850574 | 24.697 | 131.173418 | 140.255 |
| Hold until observation | 23.745156 | 30.618 | 135.267750 | 262.045 |
| Notification | 3.612330 | **6.802** | 3.983657 | **131.789** |
| Endpoint to wait return | 3.415841 | 6.602 | 3.748807 | 131.608 |
| Wait return to record read | 0.196470 | 0.712 | 0.226675 | 14.478 |

The early lifetime observed-hold maximum 34.776 us and notification maximum
14.177 us are unchanged pre-capture history, occurring 19.367 seconds before
the first summary. They must not replace this capture's rolling maxima above.
The required-hold maxima increase inside the capture, but their final values
belong to different commands. An intermediate exact paired event, first
visible at **02:54:20.193987456 UTC**, retains:

```text
Primary counter 229467614 / combined counter 260176434
Producer epoch 18 / attempt sequence 45205927

Baseline B = 2196698516866923 ns
Early F    = 2196698516891880 ns
Full F     = 2196698517003140 ns

Early required hold =  24.957 us
Full required hold  = 136.217 us
Full F - early F    = 111.260 us
```

The worst full notification within the capture is
**131.789 us = 131.608 us before wait return + 0.181 us afterward**.
The ordinary full observer therefore still has a substantial delivery tail;
its endpoint is not available to a scheduler before that observer receives it.
Endpoint timestamps also precede final callback retirement, so neither the
136.307 us full required-hold maximum nor the SDK-call duration alone qualifies
a smaller whole-backend bound.

The early observer adds 240,270 matched commands and 480,540 notices, with
240,270 successful SDK results and no failures. Full completion adds 240,269
matches from 240,270 primary/combined observations; the one-event difference
is a boundary/pending phase. There are no new observer exclusions, invalid
records, gaps, missing peers, source resets or drops. Historical full startup
invalid/gap/missing-peer counts stay unchanged. Both diagnostic over-budget
counters remain zero against 155 us.

### Controller, camera and interpretation

Controller telemetry adds **238,425 successful submissions**, with no new
failures, rejections, gaps or drops. SDK-worker snapshots add 240,429 completed
commands; fault/failure/timeout/admission-closed/coalesced counts remain zero
and rejected receipts remain seven. Typical SDK time is **103.991303 us**,
callback duration **130.751478 us**, and receipt-to-callback finish
**131.296963 us**, with retained maximum **140.345 us**. Typical early reader
release is 19.672748 us with maximum 24.516 us. The worker remains FIFO55 with
one permitted CPU. Camera/controller timing sessions and thread IDs match the
two-step run.

Camera telemetry adds **238,515 frames** and 11 successful serial controls,
with no new failures, frame/counter gaps or drops. Typical acquisition interval
sigma is **0.315910 us** and publication sigma **0.937496 us**, versus
0.297354 and 0.874284 us in the two-step capture. Acquisition extrema are
480–525 us; publication extrema 479.786–525.422 us. These small differences
between overlapping short-capture populations do not establish a causal
performance change. Large historical telemetry-drop totals remain unchanged.

The ten-second cadence estimate stays full, with no rejected updates and
period 500000.199850–500000.778255 ns. Sampled phase is -6.196 to +7.226 us;
the lifetime positive phase maximum increases to 27.637 us. Primary, combined
and pipeline quality remain clean. No overlap/estimator events or trace drops
occur; this describes hypothetical jobs against actual main arrivals, not
observed 6 kHz operation.

The result supports the new reader-readiness mechanism and isolates the
remaining backend assumption. The next engineering question is whether a
same-command backend-ready/retirement signal or a separately justified bound
can replace 155 us. Changing reader guard, backdating full-observer availability
or silently shrinking the backend bound would not answer that question. This
analysis changes only the engineering record; runtime policy and instrument
settings remain unchanged.

## Backend-bound sensitivity from the latest two captures

The operator asks what backend bound the latest simulations imply for 6 kHz.
This analysis uses the same `024932Z` and `025406Z` text captures and current
`Simulation::evaluate()` / `Simulation::feasibility()` expressions. It changes
no runtime code or configuration. Local calculations are retained in
`/tmp/synco-backend-bound-20261030.json`; exact retained-command extraction is
also in `/tmp/backend-worst-pairs.py` and `/tmp/backend-worst-pairs.json`.

### Backend duration that fits the configured schedule

For period P, three steps, early-main allowance E and observation allowance O,
the nominal final-slot condition is:

```text
backend <= P / 3 - E - O
        <= 500 / 3 - 20 - 5
        <= 141.667 us approximately
```

This is a scheduling ceiling, not a measured duration. The 20 us early-main
allowance is the currently configured assumption, not a newly qualified
early-arrival limit. The reader/copy check is independent and already fits.

| Assumed backend, us | Latest actual-main/combined observation phase for step 1, us | Nominal final-slot slack, us |
| --- | ---: | ---: |
| 155 | +11.667 | -13.333 |
| 145 | +21.667 | -3.333 |
| 141 | +25.667 | +0.667 |
| 140 | +26.667 | +1.667 |
| 135 | +31.667 | +6.667 |
| 130 | +36.667 | +11.667 |

The phase column follows
`max(mainObservation, combinedObservation) + backend <= grid + P / 3`.
It is a threshold, not a predicted acceptance percentage. The three-step
capture's new primary phase maximum alone reaches +27.637 us, so even 140 us
cannot imply every first active opportunity succeeds. Combined-observation
delay can tighten that constraint further. Occasional skips remain part of
the requested excursion policy.

The runtime wrap check uses the **following committed grid**, which can move
by a phase correction relative to nominal extrapolation. For 118 usable saved
three-step snapshots, the reconstructed final-slot ceiling is approximately
**141.521–141.799 us**, with at most 1 ns uncertainty from the unsaved frozen
fractional remainder. Two snapshots whose `nextGrid` still names the current
early main are excluded. These are sparse samples, not all cycle decisions.
The configured phase slew permits a -1 us following-grid correction, tightening
the nominal ceiling to approximately **140.667 us** before nanosecond rounding.
Thus **140 us is a useful counterfactual shadow test point** with some geometry
headroom; 141 us has too little slack to cover the full configured correction.
Neither value is thereby established as a physical backend bound.

### Measured duration of the existing command path

An important endpoint distinction emerges from the captures. The model applies
the same scalar backend duration after:

- `max(primary observation, combined observation)` for an actual main; and
- the hypothetical auxiliary trigger target for a virtual command.

The virtual command therefore needs delivery and dmcomb time that has largely
already elapsed at the actual-main anchor. A duration measured only from
combined observation cannot automatically be reused as a trigger-to-ready
duration.

| Measured population | Two-step typical / maximum, us | Three-step typical / maximum, us |
| --- | ---: | ---: |
| Primary trace update to combined writetime | 11.651922 / 22.783 | 11.440369 / 21.641 |
| Full endpoint minus matched baseline B | 131.194969 / 140.676 | 131.258332 / 136.307 |
| Controller start to full endpoint | 131.262730 / 140.766 | 131.173418 / 140.255 |

Typical values are medians of overlapping window means; maxima are retained
window extrema. Here B is the later primary/combined observer capture. Adding
the typical delivery and matched-endpoint values gives an **indicative
142.847 us and 142.699 us** for the existing primary path. These are sums of
nearby aggregate populations and separately measured durations, not paired
per-command statistics. They already exceed the nominal 141.667 us ceiling by
about 1.0–1.2 us before final retirement work.

Exact retained command records provide a stronger check without summing
unrelated maxima. P, C and F below are same-host monotonic primary-observer,
combined-observer and full-completion endpoint timestamps:

| Capture / primary counter | C - P, us | F - C, us | F - P, us |
| --- | ---: | ---: | ---: |
| 024932Z / 228923617 | 17.372 | 138.281 | 155.653 |
| 024932Z / 228926231 | 19.217 | 139.773 | **158.990** |
| 024932Z / 228959219 | 14.197 | 140.676 | 154.873 |
| 025406Z / 229467614 | 16.301 | 136.217 | 152.518 |
| 025406Z / 229517991 | 18.846 | 136.307 | **155.153** |

Each becomes newly reported inside its capture. The 158.990 us example first
appears at **2026-10-30 02:49:49.610632855 UTC**, publisher session
2120408464557324, epoch 18, attempt 44664544, combined counter 259635051:

```text
P = 2196427825110151 ns
C = 2196427825129368 ns
S = 2196427825129197 ns
F = 2196427825269141 ns

F - P = 158.990 us
F - S = 139.944 us
```

P is observed **after** actual primary publication, so delayed primary
observation reduces F - P; delayed combined observation cancels out of that
direct difference. These are lower bounds on primary-publication-to-F time,
subject to the existing best-effort metadata association. They include no
completion-observer notification delay. F is stamped before full-record
publication and `retireCommand()`, so actual command-slot availability is later.
Selection retains worst required-hold/notification records, not the largest
F - P across all commands: 158.990 us is an observed example, not an established
global maximum. Auxiliary semaphore triggering could have different delivery
behavior and remains unmeasured here.

### Interpretation and limits of a counterfactual sweep

The schedule needs approximately **140 us or less** for a practical nominal
three-step backend assumption under the current reserves and phase correction.
The existing primary path is approximately **143 us typically**, with selected
same-command examples reaching **159 us before retirement**. The old 155 us is
therefore conservative relative to combined-observation-to-endpoint timing,
but not a demonstrated upper bound on trigger-to-ready time. Simply selecting
140 us would make the geometry fit while understating some measured paths.

A smaller backend assumption can still be explored explicitly in shadow mode
to measure model sensitivity and skip reasons. The saved aggregate summaries
do not contain every primary/combined observation, committed grid or candidate
decision, and no estimator trace was selected in these captures. Exact
counterfactual skip rates at 140/135/130 us cannot be recovered offline. In
particular, extrapolating a 1 Hz grid to a retained worst command cannot recover
its intervening per-cycle phase corrections.

The current reserves are part of the tradeoff: retaining 155 us would require
E <= 6.667 us nominally with O = 5 us; that is arithmetic, not a recommendation
or qualification of a smaller early reserve. Useful next evidence is a
consistent trigger-to-command-retirement measurement, separating real-main
readiness from virtual auxiliary delivery and reporting skips for excursions.
This distinguishes a nominal 6 kHz cadence with allowed skips from a claim that
every three-step cycle fits. No operating bound is changed by this analysis.

## Existing worker-enabled controller timing coverage

The operator proposes returning to `bmcCtrl::commandDM()` to locate backend
costs. Source review confirms existing `[dmtimes]` instrumentation covers the
main worker and monitor stages; a new timing system is not required to begin.
The following uses `025406Z`, with typical values taken as medians of reported
overlapping two-second means. They are not per-command medians.

| Work or interval | Execution context | Telemetry | Typical, us |
| --- | --- | --- | ---: |
| Receipt to callback entry | Monitor | `dispatch` | 0.546 |
| Mapping, scaling, clipping, square root and output-shape post | Monitor | `actuator` | 7.702 |
| Published SDK request to actual worker SDK entry | Monitor/worker handoff | `sdkDispatch` | 2.950 |
| Actual SDK wrapper / `BMCSetArray` call | Worker | `sdk` | 103.991 |
| Private saturation-map preparation | Monitor, overlapping SDK | `saturation` | 3.363 |
| Channel reads / private delta staging | Monitor, overlapping SDK | `staging` | 4.737 |
| Callback entry through early reader-release endpoint | Monitor, cumulative | `readerRelease` | 19.673 |
| Finalization wait for the SDK result | Monitor, overlapping remaining SDK time | `sdkWait` | 98.422 |
| Deferred delta/difference publication after SDK success | Monitor | `publication` | 6.783 |
| Saturation semaphore post | Monitor | `post` | 0.897 |
| Callback entry through timing-finish marker | Monitor, cumulative | `callback` | 130.751 |
| Receipt through timing-finish marker | Monitor, cumulative | `receiptCallback` | 131.297 |

Do not add overlapping SDK, preparation and wait durations. In worker mode,
`delta` spans staging, result wait and publication; its approximately 114.950 us
is not the cost of channel arithmetic alone. `sdkDispatch` excludes the earlier
actuator-vector copy into worker-owned storage. `actuator` includes
`m_outputShape.post()`, so it does not isolate numeric conversion from that
stream's publication cost.

Differences of means within each saved window estimate the main critical-path
positions relative to callback entry:

```text
Request dispatch                         +10.448 us
Actual SDK entry                         +13.395 us
SDK return                               +117.39 us approximately
Deferred publication starts              +123.02 us approximately
Deferred publication ends                +129.80 us approximately
Timing-finish marker                     +130.75 us approximately
Full-completion publication / retirement  later; not separately timed
```

The median per-window SDK-return-to-timing-finish difference is **13.364 us**.
SDK return to publication start accounts for approximately **5.628 us**;
publication itself takes 6.783 us; publication end through callback finish is
approximately 0.955 us, mostly the 0.897 us saturation post. These differences
use nearby metric populations with possible one-event boundary differences;
they are approximate aggregate diagnostics, not paired latency distributions
or worst-case bounds.

Existing coverage gaps are:

- Separate numeric actuator preparation from output-shape publication, and
  separately time the SDK-vector copy/request setup before dispatch.
- Separate SDK result storage/notification, monitor wait-return latency,
  imported timing/result bookkeeping, successful saturation-map copy and
  reader-release SDK-result enrichment. Some of these share the approximately
  5.628 us interval before deferred publication.
- Measure early reader-release record encoding/posting independently of the
  last-read endpoint and wait start.
- Extend coverage past the timing-finish marker through full-completion record
  publication and the point `retireCommand()` clears `m_commandActive`.

The existing worker's `performSdk()` stamps actual entry/return and imports
those immutable timestamps in `finalizeCommand()`, so `sdk` already follows
the worker rather than the monitor's asynchronous `commandDM()` return.
Current telemetry is sufficient to identify the major costs and target a
small marker extension. No implementation is changed by this inventory.

## Fully pipelined controller direction

The operator proposes preparing a subsequent command while the previous SDK
call is running, and moving success-dependent publication to another thread.
This is a viable architectural direction. It is an evaluation, not an
implemented change to the controller or the shadow model.

### Throughput and latency

Current typical stages are roughly 10.45 us from callback entry to request
dispatch, 2.95 us for worker handoff, 104 us SDK service, and 13.36 us from SDK
return to the recorded callback finish. Channel/saturation staging overlaps
SDK execution and releases readers about 19.67 us after callback entry.

With prepared work available and independent finalization, the sustained
SDK-to-SDK initiation interval can approach SDK service plus result/next-job
handoff: approximately **107 us** using current typical service and handoff
figures. Thus **less than 110 us is a plausible throughput target**, not yet a
measured bound. The SDK maximum is 105.680 us in the three-step capture and
**110.900 us in the two-step capture**, already above 110 us before handoff.
Existing handoff tails are also not bounded by their typical 2.950 us. Thus
this is a typical service-interval target with excursion handling, not a hard
less-than-110 us guarantee.

An otherwise idle command still incurs delivery and preparation before its
SDK call, and publication after it. Pipelining does not turn the current
approximately 118 us receipt-to-SDK-return or 131 us receipt-to-timing-finish
latency into 110 us. At 6 kHz, nominal arrivals are 166.667 us apart, so the
main value is independent admission/readiness and handling excursions without
serializing subsequent preparation behind an older callback's entire tail.

### Proposed stages and ownership

```text
Monitor:   prepare job N+1 / snapshot channels / release readers / return
                   | immutable SDK vector ready
SDK owner:         +--> serialized BMCSetArray(N+1) --> keyed result
                                                        |
Finalizer:                 prepared products + SDK result --> publish / recycle
```

Retain early SDK launch as soon as a job's actuator vector is ready; remaining
channel staging may finish concurrently, as it does now. Final publication
requires both completed preparation and a successful matching SDK result.

Use a small preallocated job pool, initially three slots for simultaneous
preparation, SDK execution and finalization. Each slot owns its actuator
vector, clipped output shape needed by delta calculations, saturation map and
count, staged delta/difference, exact source identity, handle/configuration
generation, result/outcome and timestamps. Monitor-only summation scratch can
remain shared within that single owner. A slot cannot be reused until every
owner has finished. Pool credits bound all intermediate backlogs; finalizer
delay must stop admission rather than create an unbounded queue.

Keep all BMC handle calls on the sole SDK worker. Keep external channel reads
on the monitor before its per-job reader release; a publication thread must
never rerun `makeDelta()` against newer channel contents. Publish completed
products in job order through one writer per image. The output-shape stream
currently posts during preparation, before SDK success: preserving that timing
means it can lead the hardware further when a prepared job waits. Moving it
to ordered success publication is a separate observable behavior decision.

### Interfaces that need revision

The current single `m_commandActive` gate intentionally spans SDK return,
publication and whole-callback retirement. It must become separate job-credit,
SDK ownership and finalization states; merely clearing it earlier would expose
shared buffers and callback state. Likewise, the existing single SDK request
and result must become a bounded job/result handoff.

Completion and timing are also currently single-attempt objects:

- An older SDK-result enrichment must not overwrite a newer job's busy/released
  `read_done` record. Use explicit job identities and an ordered result protocol
  or separate result stream, with prompt independent SDK-fault propagation.
- Full completion must refer to the job's frozen input identity. Requiring the
  current combined stream still to contain that old command would reject valid
  completions after the next command arrives.
- Capture timestamps in job-owned records and assemble telemetry through one
  producer. Three threads must not mutate `m_dmTiming`'s current event.
- Saturation needs a payload-preserving handoff or accumulation by the sole
  finalizer. The current counting semaphore plus one `m_instSatMap` can otherwise
  apply a newer mask to an older notification when publications arrive in a
  burst.

Reader clearance, SDK availability, products published and storage retired are
distinct milestones. The shadow model must eventually represent those stages
and any bounded waiting explicitly; lowering the old whole-backend scalar to
110 us would not model this architecture correctly. Decisions still need
defined SDK-start/deadline and freshness behavior for pending work, including
counted skips/coalescing when capacity or an auxiliary deadline is exceeded.

### Faults, control and evaluation scope

The SDK owner must latch an SDK failure/exception/timeout before starting the
next queued command, and cancel unsent jobs with their identities preserved.
Publication failure in an older finalizer can occur after a newer SDK call has
started: stop subsequent admissions and drain committed work, while reporting
SDK success separately from product-publication failure. Preserving the old
guarantee that no newer SDK call starts before all earlier publication succeeds
would reintroduce a finalization barrier and its throughput cost.

Close/reinitialize/map-change/shutdown operations must stop admission and drain
all stages before altering the handle, buffers or generation. A timeout cannot
free a buffer still owned by a live SDK call.

The first implementation tests should demonstrate preparation B progressing
while SDK A is blocked, SDK B progressing while finalizer A is blocked, exact
per-job output/identity association and no overlapping SDK calls. Additional
tests need bounded overload, source replacement, SDK failure with queued work,
late publication failure, saturation payload accounting and full control-drain
ownership. Instrument measurements should then separate SDK initiation spacing,
input-reader clearance, per-command latency and publication/retirement lag.
No runtime code, settings or operating bound is changed by this evaluation.

## Full pipeline implementation record

The operator authorized implementation after the evaluation above. The new
startup-only `sdk.pipelineEnabled` defaults false and requires the existing
SDK worker. The default synchronous and joined-worker paths remain available.
The [controller guide](../../../apps/bmcCtrl/doc/sdk-worker.md#full-preparation--sdk--post-pipeline)
contains the activation settings and changed stream semantics.

### Implemented ownership and admission

- Exactly three preallocated private command credits separate monitor
  preparation, serialized native SDK execution and ordered post work. SDK
  submission can progress while an earlier job is still publishing products.
- Each job owns its actuator vector, clipped shape, saturation mask/count,
  delta/difference, source identity, outcome, generation and timing. Shared
  summation scratch remains monitor-owned; publication never rereads channels.
- Preserve preparation-time output-shape publication. The monitor publishes
  read clearance only after its last channel read; the post worker publishes
  success-dependent products from that job's private storage.
- Announce each latest coherent observed input even when every credit is
  occupied. The SDK owner supersedes older unsent jobs before native entry,
  without dereferencing the monitor's mapping. Disconnect/reattachment also
  cancels unsent old-source jobs. Native calls already running drain truthfully.
- Native failure/exception/timeout fences subsequent SDK work immediately.
  Late post failure can occur after the next native call starts; admissions
  close and committed work drains without pretending to undo submission.
- Stream reallocation, control, reinitialization, release and shutdown drain
  all preparing/native/post owners before changing buffers or handle state.
  Initialization before monitor attachment is supported; job allocation occurs
  once output geometry is available.

SDK-result publication retains the job lease until its publication finishes.
Native-active diagnostics clear when the actual call returns, independently
of that retained lease, so result-publication delay cannot become a false
native timeout. Failure-only logs run outside the handoff mutexes.

### Signals, statistics and compatibility

The monitor remains the only `read_done` writer. It no longer enriches that
record with an older SDK result after a newer command begins. The optional
`sdk.pipelineResultStream` has a distinct SDK-only protocol and publisher;
full completion is separately published in job order using frozen identity,
without requiring the live input still to contain that old command. SDK
return before monitor validation retains actual return evidence with an
unqualified source identity. SDK-result, reader-release and full-completion
protocols cannot substitute for each other.

At an actual 2 kHz rate without skips, `read_done` metadata advances at 4 kHz
with 2 kHz semaphore posts, and SDK/full-result metadata each advance at 2 kHz.
The standard post thread property is `th-pipeline`; `sdk_pipeline` exposes
bounded occupancy, actual SDK/post activity, admission/retirement, overload,
cancellation, supersession and publication failures. Requested placement is
checked against actual thread scheduling and affinity at startup.

Timing events detach from the monitor into jobs and are submitted by the sole
finalizer; no collector has multiple mutable producers. Failed publication
preserves actual successful SDK submission. Admission rejections are counted
by the controller rather than emitted from a second timing-queue producer.
The pipeline has no monitor SDK wait, and saturation classification is folded
into actuator preparation, so `sdkWait` and separate `saturation` populations
are unavailable in this mode. Existing timing endpoints precede full-record
publication and final credit retirement, as documented.

Saturation now has a preallocated, payload-preserving accumulator for the
pipeline: every successful mask and sample count survives delayed/coalesced
notifications. A 64-bit pipeline window is separate from legacy saturation
storage. Reset and geometry changes are synchronized with the reporting
consumer; neither add nor drain allocates. Legacy channel staging reuses the
same arithmetic through a nonowning Eigen reference, avoiding a new temporary
image allocation on the old path.

### Workstation verification

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| BMC controller, including ten pipeline cases | 1,022 | 25 |
| Detached/legacy timing helper | 417 | 21 |
| Completion helper and protocol compatibility | 635 | 19 |
| Completion integration | 293 | 9 |
| Reader timing/qualification | 518 | 23 |
| Saturation payload helper | 64 | 5 |
| Generic pipeline integration | 126 | 5 |
| Generic DM regression | 274 | 5 |
| Timing integration regression | 153 | 8 |
| ALPAO default-path regression | 18 | 1 |
| Iris AO default-path regression | 24 | 1 |
| **Total** | **3,544** | **122** |

Deterministic barriers prove preparation while a previous native call is
held, a subsequent native call while earlier post work is held, exact private
products/identity, latest-input supersession after full-pool rejection,
disconnect handling, source validation, native failures/exceptions, late
publication failure, timeout ownership, startup-before-allocation and control/
reallocation drain. Independent final review finds no blocking ownership or
lock-order defect.

The ten pipeline cases also pass under ThreadSanitizer (**439 assertions**)
without reported races. The library/PCH builds and production controller
object compiles against the real archived BMC 3.5.3 headers. Vendor libraries
are absent locally, so a vendor-linked controller executable and instrument
timing qualification remain operator follow-up. All changed C++ files pass
clang-format; Markdown links/fences and whitespace are checked. No instrument
commands or configuration edits are performed. The operator subsequently
authorized committing and pushing the implementation and documentation.

## First instrument pipeline run: stopped submissions and provenance fix

The operator installed `5aee723de3c10c108d71d2e3a019008fb03e24c3` and reported
that `dmtweeter_sdk_done` stopped updating and `fgtimes` was absent. Startup at
**2026-10-30 04:48:10.516871821 UTC** confirms the requested pipeline and SDK
result name, the monitor on `tweeter` at FIFO55, SDK owner on `spare56` at
FIFO55, and ordinary-priority post worker with inherited affinity. The SDK
reports **4.1.3**, `driverType=1`, `amplifierType=3`; this establishes the
instrument's runtime version separately from the local 3.5.3 source archive.

The supplied snapshots show:

| Evidence | Value |
| --- | ---: |
| Pipeline admitted / retired / in flight | 2 / 2 / 0 |
| Actual native completions / failures / timeouts | 2 / 0 / 0 |
| Timing submitted / failed / no-send | 2 / 1 / 0 |
| Pipeline post failures / cancelled / overloaded | 0 / 0 / 0 |
| Worker fault / admission closed | 1 / 1 |
| Rejected subsequent receipts | 454,742 |

Both native calls succeeded, but one callback failed before post processing.
There are only two events, insufficient to qualify interval statistics, and
the two-second window subsequently becomes stale. Thus absent `fgtimes` and
zero SDK-stream update rate are consequences of stopped command admission,
rather than independent missing-telemetry configuration. A closed AO-loop
flag did not establish continued DM updates. The operator was asked to open
the AO loop while diagnosing the stopped controller; agents executed no
instrument commands.

### Reproduced policy error

`releasePipelineCommand()` previously converted a late live-input identity
change into `preparedStatus=-1`, which permanently faulted admission even
after native submission succeeded. Existing source-advance tests reproduced
that exact sequence and incorrectly expected the latch. The alternative
publisher-side observation of the same change only withheld qualification,
making the outcome depend on which of two nearby checks saw it first.

The original run has no retained per-reason preparation record, so the
specific hardware event cannot be proven from its aggregates. Static staging
geometry loss remains another preparation-side possibility. The code-level
provenance/fault coupling is nevertheless reproduced and corrected, with
separate diagnostics to resolve any recurrence.

The corrected stages now:

- Validate combined-source identity immediately after actuator conversion,
  before shape publication and native queue admission. An unstable initial
  capture or vector observation is superseded/no-send without a hardware
  fault. No SDK call receives that rejected vector.
- Qualify SDK results from their validated immutable vector identity,
  independently of later channel staging or reader/full qualification.
- Withhold reader/full provenance on later input advancement, including loss
  seen only by the publisher's second check, without changing successful
  preparation status or actual SDK timing/outcome.
- Retain the fault latch for actual conversion/staging errors, SDK errors,
  timeouts and publication failures.

`sdk_pipeline.vectorRejected` counts unstable pre-dispatch source observations;
`readerUnqualified` counts late unqualified reader/full identities and can
overlap vector rejection; `preparationFailures` counts real preparation errors.
True negative preparation results now log their failed stage. Invalid capture
tokens no longer manufacture a valid zero input counter in timing records.

### Startup INDI definitions

The startup log also directly reports invalid `sdk_worker` and `sdk_pipeline`
definitions before the first ordinary app update: their property state was
unset. The serializer additionally requires numeric element range, step and
format. Both production definitions now initialize Idle state, complete
numeric metadata, zero/default values and verified worker settings before
registration. A latched fault changes their ordinary readback state to Alert.
The regression serializes and round-trips the actual production properties
through `IndiXmlParser`, before app logic can fill missing attributes.

### Correction verification

The corrected BMC suite passes **1,182 assertions / 28 cases**; generic pipeline
integration passes **203 assertions / 6 cases**. Regressions cover unstable
capture and prequeue vector rejection without native entry, later input
advancement followed by three successful submissions and valid `fgtimes`,
reader publication enabled/disabled, preserved private products, genuine
staging-geometry failure retaining its fault, and valid initial INDI XML.
Independent review finds no blocking ownership or policy regression.

The corrected thirteen pipeline cases also pass ThreadSanitizer (**599
assertions**) without reported races. A fresh library/PCH build and production
controller-object compilation against the local vendor headers pass; formatting,
Markdown links/fences and whitespace are checked. The correction is committed
and pushed under the operator's continuing request to deliver this pipeline
for instrument testing; hardware validation remains operator-owned.

Hardware recovery still requires the operator to install the correction and
reinitialize the controller. At a sustained input rate, check that native
completions continue increasing, fault/admission-closed remain zero,
`preparationFailures` remains zero and timing becomes valid. Provenance-loss
counts can increase without implying a native failure. No instrument settings
are changed by this correction.

## Recovered pipeline baseline: `synco-sim-20261030T052548Z`

The operator reports successful operation after the provenance correction and
supplied this simultaneous capture. Its 120 synco records span **2026-10-30
05:25:47.335581646–05:27:47.697894832 UTC** (120.362313 s; October 29,
22:25:47–22:27:47 Phoenix time). All snapshots succeeded, error files are empty,
and the three collectors exited with the expected timeout status 124. The
saved checkout is `93a8a414fd5385975a5018e2cbe78f13e246fef4`; this records the
checkout rather than independently identifying each running binary.

The operator confirms **dmcomb trigger mode 4 and zero gains**. The saved scalar
loop gain is 1, versus 0.05 in `025406Z`; zero modal gains keep the effective
gain zero. Camera settings remain bin2, 2 kHz, EM gain 1, synchronization On and
target −45 °C. Shutter state is OPEN, versus SHUT in `025406Z`. RTC latency is
On at both boundaries. Thus comparisons below describe the observed changes,
without treating these as an exact experiment with software as the only change.

### Controller health and ownership

The SDK worker reports actual FIFO55 with one permitted CPU. The post worker
reports ordinary scheduling, priority 0 and **60 permitted CPUs**; 60 is a count,
not a CPU number. Exact affinity masks and IRQ placement are not saved.

Between the controller's before/after snapshots:

- Admitted and native-completed commands each increase by **240,418**.
- Retired commands increase by **240,419**, consistent with one job already in
  flight at the first boundary and none remaining at the last.
- Fault, admission-closed, SDK failure/timeout, preparation/post failure,
  vector rejection, cancellation and supersession remain zero.
- Overload stays at its historical value 9, reader-unqualified stays at 2,
  rejected stays at 12 and coalesced stays at 11. None increases during the run.

The 120 valid DM timing summaries add 238,415 events between their first/last
totals, all submitted, with no new failures, no-send events, counter gaps,
invalid records or drops. The timing and INDI boundaries differ, so their
command totals should not be expected to match exactly.

### Preparation, native call and post latency

"Typical" below is the median of the reported two-second window means, in
microseconds. Maxima are retained by those windows in this capture, rather than
unrelated lifetime maxima. The previous column is the joined-worker `025406Z`
baseline.

| DM timing measurement | Previous typical | Pipeline typical | Pipeline maximum |
| --- | ---: | ---: | ---: |
| Actuator preparation | 7.702 | 19.169 | 28.103 |
| SDK enqueue to native entry | 2.950 | 3.202 | 9.167 |
| Native SDK duration | 103.991 | 104.557 | 110.068 |
| Channel staging | 4.737 | 8.393 | 16.862 |
| Callback entry to reader release | 19.673 | 28.848 | 38.412 |
| Receipt to native return | 117.935 | 127.641 | 136.728 |
| Private-product publication | 6.783 | 9.089 | 80.552 |
| Receipt to post timing finish | 131.297 | 146.577 | 252.887 |

Actuator preparation changed meaning: it now includes private-shape preparation,
saturation classification, source validation and shape publication before SDK
dispatch. The previous separate saturation pass averaged 3.363 us. The actuator
increase therefore cannot all be attributed to additional conversion work.
These populations do not isolate the individual private-copy/validation costs.
The reader-release total nevertheless increased by about **9.2 us**, and staging
also takes longer. Separate `saturation` and `sdkWait` populations are empty by
design in pipeline mode.

The longest receipt-to-finish window is at **05:27:19.081575978 UTC**. Its native
duration maximum is 105.569 us and receipt-to-native-return maximum is 130.427 us.
Consequently the 252.887 us finish event contains **at least 122.460 us after
native return**. Publication reaches 80.552 us in the same window, although its
maximum cannot be assigned to that exact command from these aggregates. The
large tail is after the native call; ordinary-priority post scheduling remains
a candidate contributor, alongside publication work.

Pipelining separates resource occupancy from per-command latency. The longer
146.6 us end-to-end mean does not mean that the SDK owner or monitor blocks for
that duration. Conversely, a 104.6 us native mean does not establish the next
native call's earliest start: SDK-result publication and handoff still follow
the return. The finish marker also precedes full-record publication, timing
submission and final job-credit retirement.

### Reader signals and observation delay

The early observer adds 240,724 completions and 240,723 matches; the full observer
adds 240,722 completions and 240,721 matches. Each has one new incoherent capture,
counter gap and missing-combined association. There are no new invalid or
unqualified records, sequence gaps, identity mismatches or queue drops. This
single association loss does not establish a dropped hardware frame.

| Reader measurement | Early typical / maximum (us) | Full typical / maximum (us) |
| --- | ---: | ---: |
| Producer processing, start to release | 29.030 / 38.583 | 146.283 / 253.589 |
| Required hold from observed main baseline | 29.409 / 38.272 | 146.610 / 254.300 |
| Observed hold through notification capture | 33.207 / 42.049 | 148.957 / 289.597 |
| Notification delay after release | 3.799 / 10.990 | 2.321 / 110.158 |

The largest full observed hold first appears in the summary at
**05:27:19.376222804 UTC**. Within printed precision, the maxima in that matched
window force the same-event decomposition: **254.300 us to producer completion
+ 33.794 us to observer receipt + 1.503 us capture = 289.597 us**. The absolute
timestamps and command identity of that event are unavailable. The separate
110.158 us notification maximum is a different excursion; it must not be added
to the producer maximum.

All retained raw lifetime worst-event snapshots predate this capture, including
a roughly 2.7 ms full event from the present controller before collection began.
Those snapshots cannot identify the new window maxima. Signed completion offsets
can also be negative when the combined-stream observer sees a command after the
producer already released it; that is observation ordering, not negative
execution time. The virtual 30 us auxiliary-reader assumption is still an
assumption: the current actual main-command reader maximum exceeds it.

Pipeline `read_done` has one release per command and no SDK-result enrichment.
The raw readiness gate therefore adds **240,724 observations and 240,724 qualified
releases**, with no new quality loss. Zero increments in early SDK-result counts
are expected; the surviving observer retains historical results from the joined
worker. The dedicated `dmtweeter_sdk_done` protocol still has no synco SDK-result
observer.

### Existing shadow result and what to test next

The shadow retains three steps, reader assumption 30 us, backend assumption
155 us, guard 20 us and the existing early reserve. Capture-local decisions are:

| Step | Considered | Would run | Skipped | Skip reasons |
| --- | ---: | ---: | ---: | --- |
| 0 | 240,724 | 240,723 | 1 | 1 missing output |
| 1 | 240,724 | 88,138 | 152,586 | 152,584 backend busy; 2 missing output |
| 2 | 240,724 | 0 | 240,724 | 240,722 backend wrap; 1 backend busy; 1 missing output |

Reader-cycle feasibility is true and backend-cycle feasibility false under that
unchanged single-backend model. These results do **not** model the new separate
preparation, SDK and post owners. No new modeled overlap or transport drop occurs.
The few missing-output decisions are association/availability outcomes, with no
matching new primary/combined observer counter-gap counts.

Cadence retains 64 anchors in its ten-second window, accepts 121 updates with no
rejection, and reports periods **500000.098425–500000.859321 ns**. Sampled phase
is −6.526 to +3.819 us. Historical overlap counts and large phase extrema do not
increase during this capture.

Camera timing remains stable: median acquisition/publication interval means are
500.000500/500.000530 us, with median window deviations 0.298/0.877 us. Observed
interval ranges are 480–519 us and 478.514–520.773 us. The camera adds 238,518
published events between telemetry boundaries and 11 successful serial controls,
with no new failures, frame/counter gaps, invalid records or drops.

This run validates sustained healthy **2 kHz** operation after the fix. The next
shadow evaluation needs separate reader clearance, serialized SDK availability
and three-credit post retirement, with native-return evidence obtained through
the SDK-result protocol. Include publication/handoff after SDK return before
next SDK admission, and completion publication before credit retirement. Preserve
the post tail and actual admission diagnostics in that evaluation. A smaller
scalar `backendUs` alone cannot represent those
independent resources or qualify 6 kHz native operation. Post-thread placement
and preparation cost are useful subsequent controlled measurements. No runtime
settings or code are changed by this analysis.

## SDK-worker readiness measurement

The operator identified that the useful throughput measurement is when the SDK
worker could accept its next prepared command. The 110.068 us maximum above is
native-call duration alone. Measuring consecutive native entries at a 2 kHz
input rate would mostly measure the deliberate 500 us input spacing. Instead,
the operator authorized a worker-ready stamp that excludes that idle interval.

### Endpoint and ownership

After publishing a pipeline command's actual SDK result, the SDK owner retains
its pool lease through the next loop's acquisition of `m_sdkMutex`. Before the
worker waits or selects another request, it commits the previous job's
`sdkDone`/`running` flags, issues the post and result notifications, and captures
`SdkResult::readyNs`. These operations remain under the SDK mutex, so neither
post nor a waiting test/control observer can see completion before the timestamp
exists. A worker-local slot index identifies the still-owned prior job; it
never refers to already recycled storage.

An isolated command reaches this endpoint and can retire without a following
request. The commit also runs before stop handling. Native-active/watchdog
diagnostics still end at actual native return; the retained lease protects
result storage, without extending the native-active duration. Failure and timeout
fencing remains immediate, before subsequent request selection. No extra queue,
thread or per-command allocation is introduced; one monotonic clock read and
one fixed timestamp are added per pipeline native result.

The ordered post owner imports readiness into the command's detached timing
event and remains the sole timing-queue producer. SDK readiness can legitimately
precede delayed monitor staging/reader release, so its causal checks use SDK
entry/return and callback finish independently of staging order. Absent or bad
readiness cannot erase actual native success or stop job retirement. Synchronous,
joined-worker, control and cancelled-before-native paths supply no ready sample.

### Reported measurements and compatibility

Two stable metric IDs are appended without changing prior IDs:

| Metric | ID | Paired command endpoints |
| --- | ---: | --- |
| `sdkTail` | 18 | Native return to worker ready |
| `sdkService` | 19 | Native entry to worker ready |

Both appear in `[dmtimes]` and the existing `dm_timing_stages` INDI property, with
ordinary count, mean, jitter, minimum, maximum and coverage. All durations are
seconds. Per-command pairing provides the combined service distribution directly;
adding independently observed SDK/tail maxima would lose that pairing. Binary
readers accept historical complete 12-, 13- and 18-metric records and the new
20-metric records. Absent historical readiness durations project as unavailable.
The compatible `fgtimes` record and all completion-stream wire formats retain
their previous meanings.

This endpoint includes native execution, result bookkeeping/publication,
completion notifications and next-loop setup/mutex reacquisition. It excludes
the following request-selection, mutex-release and native-entry glue. Worker
readiness also does not establish open fault-free admission or a free job credit
when post work stalls. The SDK-result stream still marks native return, not this
later endpoint. Thus the new service duration is the measured worker occupancy,
with these explicit remaining admission conditions, rather than an asserted
minimum spacing between actual native entries.

### Verification and operator procedure

Deterministic regressions exercise an isolated job, sequential reuse of more
jobs than the three-credit pool, independent monitor/post barriers, later SDK
execution while earlier post work is held, and native error/exception/timeout
drains. Samples survive reuse, exclude later idle waiting and remain valid when
readiness precedes monitor staging. Generic timing tests cover optional and
invalid ready endpoints without losing SDK-success/compatible statistics.
Telemetry tests cover new names, metadata, serialization and historical metric
sets. Independent source review finds no blocking race or lifecycle defect.

| Workstation suite | Assertions | Cases |
| --- | ---: | ---: |
| BMC controller | 1,296 | 29 |
| Generic timing helper | 509 | 23 |
| Timing telemetry and historical compatibility | 1,350 | 7 |
| Generic pipeline integration | 203 | 6 |
| Timing/INDI integration | 157 | 8 |
| **Total** | **3,515** | **73** |

The fourteen BMC pipeline cases also pass ThreadSanitizer (**707 assertions**)
without reported races. The shared library/PCH, rebuilt `logdump`, and production
controller object using the archived vendor headers build successfully. Local
vendor libraries remain absent, so a vendor-linked controller executable and
instrument measurements remain operator follow-up. All changed C++ files pass
clang-format; whitespace and Markdown fences are checked. Only workstation
tests and builds are run.

Rebuild `bmcCtrl` and `logdump`, then repeat the existing bounded simultaneous
120-second capture at actual 2 kHz with the current settings. There are no new
configuration settings, DM commands or operational synco changes. Compare
`sdkService` directly against native `sdk`, and use `sdkTail` to locate the added
worker occupancy. Retain reader/full completion, post latency and admission/
overload diagnostics for the later pipeline shadow model.

## Instrument SDK service capture: `synco-sim-20261030T061730Z`

The new capture contains 120 synco summaries from **2026-10-30
06:17:28.675734762–06:19:28.915794761 UTC** (120.240060 s; October 29,
23:17:28–23:19:28 Phoenix time) and 121 valid controller timing summaries.
The saved checkout is `1ebd7ec5068365a9d5f5819fd0ad1d0ce21e8dae`. All snapshots
succeed, their error files are empty, and collectors return expected status 124.
Both new metrics are populated, establishing that the running controller and
reader support the new measurement independently of the saved checkout hash.

The operator confirms modal gains remain zero and settings, CPU/IRQ placement
and workload are unchanged apart from rebuilding/restarting `bmcCtrl` and
`logdump`. The loop is closed with scalar gain 1 and effective gain zero;
dmcomb remains in mode 4. Camera settings remain bin2, 2 kHz, EM gain 1, shutter
OPEN, synchronization On and target −45 °C. RTC latency is On at both boundaries.
SDK scheduling remains actual FIFO55 with one permitted CPU, and post remains
ordinary priority 0 with 60 permitted CPUs. Exact masks are not saved.

### Measured SDK-worker occupancy

All values below are microseconds. Typical means the median of the reported
two-second window means; each maximum is taken directly from its own population.

| Measurement | Typical | Recorded minimum | Recorded maximum |
| --- | ---: | ---: | ---: |
| Native `sdk` duration | 104.561 | 104.076 | 111.150 |
| `sdkTail`: native return to worker ready | 2.234 | 1.263 | 4.820 |
| `sdkService`: native entry to worker ready | **106.796** | **105.580** | **113.444** |

This answers the worker-occupancy question: the measured maximum through
readiness is **113.444 us**, including result publication and handoff. It is a
direct per-command interval, without adding independent SDK/tail maxima or
including the roughly 500 us idle spacing between input commands. In the 99
summaries with identical populations, the means satisfy service = native + tail
within printed precision. Other windows can retain one additional tail sample
because its later start endpoint falls inside the trailing window; this does
not indicate loss or require equal population counts.

The SDK and service maxima first appear at **06:17:58.941344532 UTC**. Their
shared window does not establish that they are the same command. The separate
tail maximum occurs at **06:19:03.061982175 UTC**, whose service maximum is only
109.387 us. No new raw per-command worst-event record identifies those jobs.

A nominal 6 kHz interval is 166.667 us, leaving **53.223 us** beyond the largest
recorded worker service duration. That is arithmetic SDK-resource headroom;
delivery, next-request selection/native entry, preparation, independent reader
clearance, early-main reserves and three-credit post retirement still need
their own treatment. This 2 kHz measurement does not establish sustained 6 kHz
operation. In particular, the old 155 us single-backend assumption cannot simply
be replaced by the native-entry-to-ready measurement while retaining its old
trigger/observation anchors.

### Controller and completion health

Controller timing totals add **240,452** events/receipts/submissions with no new
failures, no-send events, invalid timestamps, duplicates, counter gaps or drops.
Before/after INDI reports admitted +240,452 and native completed/retired
+240,451, with one job in flight at the last boundary. Fault, admission-closed,
overload, vector rejection, supersession, qualification loss and preparation/post
failures remain zero; rejected stays at its startup value 2 and coalesced at 0.

Typical controller reader release is 29.294 us (maximum 38.042 us), and
receipt-to-native-return is 128.151 us (maximum 135.736 us). Receipt-to-post timing
finish remains 146.842 us typically and reaches **243.891 us**; private-product
publication reaches 100.620 us in that same window. The window's native duration
is at most 109.737 us and SDK service at most 112.072 us. Thus that long post
latency does not represent extended native execution or SDK-worker occupancy.
These different-start populations cannot be subtracted to manufacture a pure
post duration for an unidentified command.

Both reader observers add **240,480** primary/combined/completion matches, with
no new incoherent captures, gaps, missing peers, qualification failures,
sequence loss or drops. Early producer processing is typically 29.560 us,
maximum 38.402 us; full producer processing is 146.511 us, maximum 243.530 us.
Observed holds reach 42.751/245.203 us for early/full respectively. Notification
maxima are 9.728/55.916 us. Current maxima cannot be decomposed into a single
identified event from the independent aggregates. All retained raw lifetime
worst-event snapshots predate this run and refer to older controller PIDs.

### Shadow and camera observations

The existing three-step shadow retains reader 30 us, backend 155 us and guard
20 us; it still does not model the separate pipeline resources. New outcomes:

| Step | Considered | Would run | Skipped | Skip reasons |
| --- | ---: | ---: | ---: | --- |
| 0 | 240,480 | 240,480 | 0 | None |
| 1 | 240,480 | 85,524 | 154,956 | 154,955 backend busy; 1 release late |
| 2 | 240,480 | 0 | 240,480 | 240,480 backend wrap |

There are no new modeled overlaps, output gaps or resets. The single late-release
increment first appears at **06:18:09.095397389 UTC**. The raw readiness gate adds
240,480 observed and qualified releases without any new quality loss. Cadence
accepts 120 updates with no rejection, retains 58–64 anchors and reports periods
499999.862128–500000.749100 ns; sampled phase is −7.386 to +6.800 us.

Camera acquisition/publication interval means remain 500.000250/500.000336 us,
with typical rolling deviations 0.291/0.837 us. Actual camera telemetry adds
238,535 published frames, with no new frame/counter gaps, duplicates, errors or
drops, and 11 successful serial controls. Synco's two camera observers each
count eight duplicates and two counter gaps in two separate bursts; these are
observer quality changes, without corresponding camera publication loss.

An isolated delivery excursion is visible in the camera summary at
**06:18:08.548212317 UTC**: acquisition intervals remain 498–502 us, while SDK
wait-return intervals span 391.449–608.509 us and publication intervals span
392.481–607.667 us. Acquisition-to-write reaches 154.832 us, while conversion
peaks at 31.128 us, rearm at 5.230 us and serial overlap is zero in that window.
The roughly 108 us excursion is therefore already present at camera SDK wait
return. Its window overlaps the late-release decision, but the aggregate records
do not establish exact cross-device command identity or causation. No frame loss
or frame-behind state is demonstrated by these statistics.

The useful next scheduler evaluation is a pipeline-aware shadow using measured
SDK service separately from preparation/reader clearance and post-credit
retirement. This analysis changes only the engineering record; no runtime bound,
settings or code is changed.

## Pipeline-aware simulator implementation

After reviewing the service capture, the operator authorized separating the
shadow's delivery/preparation, SDK-worker service and post-credit retirement.
The operator explicitly chose **zero added main SDK wait**. No minimum future
main preparation interval or allowed main-wait budget is introduced.

### Resource model and decision semantics

The new `syncoPipelineSimulation.hpp` helper owns three fixed private leases and
independent serialized monitor, SDK and ordered-post stages. A coherent genuine
main is modeled from its observed combined command; a hypothetical auxiliary is
modeled from its fixed trigger. Delivery precedes monitor preparation. The private
vector can reach the SDK while monitor staging continues, and a subsequent SDK
service can run while an earlier job still owns a post credit. A job's post stage
requires both staged readers and SDK readiness. Final post retirement frees its
credit.

The helper advances bounded events only through caller time. Auxiliary admission
forecasts use fixed-storage copies of the model, without committing future stage
execution. Rejected forecasts retain the existing resource state; accepted
auxiliaries reserve a credit at the copy decision while their future inputs remain
pending. Input-time ordering handles a genuine main arriving before an already
reserved future auxiliary. Services already started are never preempted. The model
does not implement native queued-vector coalescing/supersession.

An auxiliary is skipped if its preparation or SDK entry would queue, no credit is
available, or its modeled readers/SDK would remain busy at the earliest next main.
At least one credit must also be available for that main. The earliest-main test
uses the existing reserves with **zero** preparation-time credit. An unexpectedly
early main can still violate those assumptions after a decision was committed;
`mainSdkWaits`/`maximumMainWaitNs` retain the predicted added monitor-plus-SDK queue
delay, and `mainCreditMisses` retains model admission/quarantine failures. These
counts describe the hypothetical resource model, not actual controller failures.

All prior fixed copy/trigger targets and exact actual-main release gates remain.
Release evidence cannot retire a virtual SDK or credit. Raw backend-overlap-at-input
logic is retained only by the legacy mode: input arrival during an SDK service does
not prove added SDK wait when preparation can overlap it. New resource diagnostics
describe that conflict at the modeled SDK-entry boundary instead.

Model reset projects existing leases through drain before discarding their
identities, preserving a cooldown fence. Incoherent combined associations add a
conservative unknown-command quarantine. Neither event manufactures an idle SDK or
free replacement credit. Known retirement, discard and in-flight accounting obeys
`mainAdmitted + auxiliaryAdmitted = retired + discarded + inFlight`. Resource
counters/maxima survive fences and schedule resets until fresh configuration;
per-step outcomes remain tied to their existing simulation session.

### Configuration, telemetry and initial test

The startup-only `simulation.pipelineBackend` defaults false. It requires
simulation, exact reader-release gating and the existing release observer. Prior
simulation/operational behavior remains selected by default. The additive
`dmtweeter_synco_pipeline_shadow.conf` requests three steps and enables:

| Assumption | Value | Anchor/ownership |
| --- | ---: | --- |
| Existing delivery | 20 us | Observation/trigger to controller input |
| `simulation.prepareUs` | 35 us | Preparation start to private vector availability |
| `simulation.virtualReaderUs` | 60 us | Total trigger-to-last-read interval, including delivery |
| `simulation.sdkServiceUs` | 120 us | Native entry to worker readiness |
| `simulation.postUs` | 150 us | Serialized post work through credit release |
| Private credits | 3 | Shared by all stages; fixed capacity |

These are explicit conditional assumptions, not qualified hardware bounds. SDK
service includes about 6.6 us above the recorded maximum. Post service includes
unmeasured completion-publication/retirement tails and is an assumption; the model
does not claim that all historical long post excursions fit it. `virtualReaderUs`
must cover delivery plus preparation. The legacy `backendUs` remains a valid
configuration/readback but is unused in this mode.

With the preset, SDK/post/credit geometry each fits nominal 166.667 us slots, but
the last trigger at 333.333 us would clear SDK service at 508.333 us. The earliest
next main is 475 us. Zero added main wait therefore vetoes that slot, recording a
33.333 us `mainClearanceShortfallUs`; the preset deliberately does not assert full
6 kHz feasibility. This diagnostic is a prospective clearance overrun, not a
recommendation to alter the early-main reserve. A short-prefix conditional fixture
exercises all three slots and verifies SDK progress while post work still owns a
credit, without weakening zero-main-wait protection.

The new read-only `simulation_pipeline` property exposes assumptions, occupancy,
admission/retirement/discard conservation, retrospective main-delay diagnostics,
clearance shortfall and separate SDK, post, credit and main-protection feasibility.
New per-step reasons are `PreparationBusy` and `CreditFull`; the existing backend
reasons identify SDK contention and next-main protection in this mode. Optional
appended binary tables/fields retain historical reader compatibility; text labels
the new mode `shadow_pipeline`. Live enable remains prohibited in simulation, and
no new stream writer, semaphore poster or controller command path is introduced.

### Verification and operator follow-up

| Suite | Assertions | Cases |
| --- | ---: | ---: |
| Pure grid/resource integration, including legacy modes | 53,812 | 53 |
| Independent three-stage resource helper | 230 | 12 |
| Application configuration, INDI and write prohibitions | 11,917 | 81 |
| Telemetry, malformed inputs and historical compatibility | 7,529 | 36 |
| **Total** | **73,488** | **182** |

Tests cover overlap, exact resource boundaries, future arrivals, fixed targets,
rejection nonmutation, credit exhaustion, added main queuing, overflow, reset/source
quarantine, independent feasibility and unchanged legacy behavior. Independent
reviews find no blocking helper/integration issue. Shared library/PCH, production
`dmSyncopation` and `logdump` builds pass. Changed C++ files are clang-formatted;
whitespace, Markdown fences and documentation links are checked.

The operator can rebuild/install `dmSyncopation` and `logdump`, merge the additive
preset into the current instance configuration, and repeat the bounded simultaneous
capture with operational synco Off and existing BMC/CPU/camera/loop settings retained.
The [procedure](../../../apps/dmSyncopation/doc/readiness-shadow-test.md#pipeline-resource-shadow)
defines the new assumptions and readbacks. All implementation and verification work
is workstation-only; no instrument configuration or hardware commands are executed.

## Preset not yet applied: `synco-sim-20261030T152231Z`

This capture records checkout `2c5c65b225bb68cb12c5ced72669472f9dd23482`, but
telemetry still identifies `shadow_reader_release_ready`, with 30 us virtual
readers and the legacy 155 us backend. The pipeline property/table is absent.
The operator subsequently confirmed the additive pipeline preset had not been
merged. This run therefore does not test the new resource model.

Actual controller admissions, native completions and retirements increase by
238,455 with no new faults, overloads or submission errors. SDK service is
typically 106.261 us and reaches 109.607 us. Those values supplement the previous
measurements; they do not replace the larger previously observed 113.444 us
maximum. The operator then supplied the corrected pipeline configuration.

## First pipeline resource capture: `synco-sim-20261030T153807Z`

The preset is active in this capture: all 119 synco summaries identify
**`shadow_pipeline`**, with preparation 35 us, SDK service 120 us, post service
150 us and three credits. Reader/delivery assumptions remain 60/20 us. The
recorded checkout is `2c5c65b225bb68cb12c5ced72669472f9dd23482`. All snapshots
succeed, error files are empty and collectors exit with expected status 124.
Synco records span **2026-10-30 15:38:07.021362371–15:40:06.221869856 UTC**
(119.200507 s; October 30, 08:38:07–08:40:06 Phoenix time).

### Fixed-model outcomes and protection

Capture-local differences, rather than lifetime counts, are:

| Step | Considered | Would run | Skipped | Skip reasons |
| --- | ---: | ---: | ---: | --- |
| 0 | 238,401 | 238,389 | 12 | 12 missing output |
| 1 | 238,401 | 238,367 | 34 | 13 missing output; 16 SDK busy; 2 preparation busy; 3 credit/quarantine |
| 2 | 238,401 | 0 | 238,401 | 238,385 next-main protection; 13 missing output; 3 credit/quarantine |

Step 1 is eligible on **99.985738%** of occurrences. Step 2 retains the planned
zero-added-main-wait veto: maximum prospective clearance shortfall increases
from 33.711 to **33.763 us**, consistent with the 33.333 us nominal calculation
plus fractional-grid/phase corrections. SDK, post and credit cycle flags remain
true; main-protection, aggregate backend and complete-cycle feasibility remain
false. No fixed targets are moved to accommodate the late slot.

Modeled main admissions increase by 238,383, auxiliary admissions by 238,367,
and retirements by 476,750; discarded stays zero and in-flight stays one at both
telemetry boundaries. Exact admission/retirement/credit conservation holds in
every summary. Maximum occupancy rises from two to three, but **modeled added
main SDK-entry wait remains zero** throughout.

The six modeled main-credit misses coincide with three unknown-source fences.
At **15:39:47.023577620**, **15:39:49.045626620** and
**15:39:50.057525856 UTC**, each summary adds one combined-observer counter gap,
one simulation output gap, one resource reset and two modeled main-credit misses.
Each also adds one credit skip on each active step. The configured unknown-work
quarantine is 1.05 ms, naturally spanning two 500 us main periods. Discarded work
does not increase. This is strong aggregate evidence of intentional quarantine
rejections, rather than demonstrated persistent pool exhaustion; individual
per-miss reason timestamps are not exported.

The observation/admission totals reconcile exactly: 238,401 genuine primary
observations contain 238,389 coherent associations; 238,383 admitted mains plus
six model credit misses account for those associations. The other twelve
observations lack usable association. These are observation/model outcomes,
not proof of twelve dropped physical commands.

### Actual controller timing and assumption excursions

Actual before/after controller admissions, native completions and retirements
each increase by **240,442**, with no jobs in flight at either boundary. There
are no new native failures, faults, timeouts, overloads, rejections, supersessions,
vector rejections, preparation/post failures or reader-qualification losses.
The controller and telemetry intervals have different boundaries; their counts
are not expected to match the synco first/last-summary interval exactly.

Typical below means median of overlapping two-second window means, in us:

| Controller metric | Typical | Current recorded maximum |
| --- | ---: | ---: |
| Native SDK duration | 104.094 | **147.939** |
| SDK return to worker ready | 2.245 | 4.408 |
| SDK entry to worker ready | **106.341** | **150.083** |
| Controller reader release | 28.799 | 36.879 |
| Receipt to native return | 127.303 | 172.445 |
| Private-product publication | 8.566 | 146.647 |
| Receipt to post timing finish | 145.881 | **521.715** |

The native/SDK-service excursion first appears in the summary at
**15:40:01.569085854 UTC**, and remains in the next overlapping summary. This
establishes at least one command above the model's **120 us SDK-service
assumption**; two reporting windows must not be counted as two distinct commands.
The native-call wall interval accounts for almost all the excursion. It does not
identify vendor work versus OS/interrupt scheduling within that interval. The
150.083 us service maximum leaves only 16.584 us below a 166.667 us slot before
other requirements; the fixed-model zero-wait outcome does not cover this larger
real service interval.

A separate post-completion stall occurs in the window at
**15:38:55.448509752 UTC**. That window's receipt-to-finish maximum is 521.715 us,
receipt-to-SDK-return maximum 132.550 us, and return-to-ready maximum 3.496 us.
Thus its longest finish event includes **at least 385.669 us after SDK-worker
readiness**, while every SDK service in that window is at most 110.369 us.
This post/queue/scheduling latency is distinct from the later native excursion.
The model's fixed post service does not capture arbitrary scheduling stalls;
the lower bound alone cannot identify one job's serialized post service versus
waiting behind earlier post work. Actual private-credit retirement is later
than the reported timing-finish endpoint.

The full completion observer retains a current raw command for the post stall:
sequence 3,202,212, combined counter 284,905,756, primary 254,196,829. Its producer
processing is **521.515 us**, required hold 520.313 us and observed hold
523.108 us; notification contributes only 2.795 us. A separate command's full
notification delay reaches 285.739 us. These maxima belong to different events
and must not be added.

### Observation quality, settings and interpretation

Raw readiness adds 238,401 observed and qualified releases with no new invalid
records, gaps, source resets or drops. Early/full joins each add 238,393 matches.
Their nine duplicate and three counter-gap observations, followed by nine
missing-combined associations, align with the combined-observer episodes above.
No new reader identity mismatch, incoherent/invalid/sequence-loss or transport
drop occurs. Current early processing peaks at 37.180 us and observed early
hold at 41.428 us, remaining separate from the long full-completion stall.

Cadence remains full, accepts 119 updates with no rejection, and reports periods
499999.937–500000.785 ns. Primary, camera and upstream-stage observation quality
has no new local fault. Actual camera telemetry adds 238,569 published frames
and eleven successful serial controls, without new frame/counter gaps, failures
or drops. Median rolling acquisition/publication standard deviations are 0.304/0.882 us;
publication intervals remain 478.814–525.943 us, without the earlier roughly
108 us camera-return excursion.

Saved settings match the previous capture: bin2, 2 kHz, EM gain 1, synchronization
On, shutter OPEN, target −45 °C, closed loop/scalar gain 1, and RTC latency On.
The operator confirms modal gains remain zero and camera/loop settings, CPU/IRQ
placement and workload were unchanged apart from merging the pipeline preset
and restarting synco. Controller/SDK/post
TIDs remain unchanged; SDK reports FIFO55 on one permitted CPU and post ordinary
priority 0 with 60 permitted CPUs. Post scheduling is a candidate contributor to
the long completion tail, not an established cause.

The new simulator is operating and its zero-main-wait/credit safeguards behave
as designed under the fixed assumptions. The capture also exposes real variable
SDK and post latency beyond that nominal model. Subsequent qualification should
exercise such excursions and the resulting skip/credit behavior explicitly,
rather than interpreting zero modeled waits as a measured 6 kHz guarantee or
automatically turning a rare maximum into every command's service duration.
This analysis changes only the engineering record; no runtime bounds or
instrument configuration are changed.

## Pre-SDK preparation investigation after `153807Z`

The operator recalled roughly 10 us before the SDK call and requested an
investigation of the larger pipeline preparation interval. Historical controller
telemetry confirms a real increase, with two endpoints to distinguish. All
values below are in us; typical means the median of rolling-window means.

| Path and capture | Actuator/conversion-publication typical | Actuator/conversion-publication maximum | Receipt to native SDK entry, typical |
| --- | ---: | ---: | ---: |
| Synchronous, `full-loop-20261026T023744Z` | 7.263 | 12.544 | 7.541 |
| Joined SDK worker, `synco-sim-20261030T025406Z` | 7.702 | 12.203 | 13.943 |
| Full pipeline, `synco-sim-20261030T052548Z` | 19.168 | 28.103 | 23.064 |
| Full pipeline, `synco-sim-20261030T061730Z` | 19.286 | 26.821 | 23.457 |
| Full pipeline, `synco-sim-20261030T153807Z` | 19.002 | 26.760 | 23.189 |

Receipt-to-SDK-entry means are calculated per window as receipt-to-SDK-return
mean minus SDK-duration mean, requiring matching sample populations, then
summarized across windows. They exclude native SDK execution. The old conversion
was typically 7–8 us, with recorded maxima around 12–14 us; the current
pipeline's roughly 23 us prefix is not merely a renamed 10 us interval.
The simulator's 35 us preparation setting remains an assumption, not a direct
measurement of the maximum prefix.

### Work moved into the prefix

The joined worker path in `bmcCtrl::commandDM()` converts and clips actuator
values while writing the shared output shape directly. It posts that shape,
copies the actuator vector into SDK-worker storage, dispatches the SDK request,
then classifies saturation while the SDK can run. Its separate saturation pass
typically costs 3.363 us in `025406Z`.

`bmcCtrl::processPipeline()` instead clears a private shape and saturation mask,
converts/clips the input while classifying/counting saturation, revalidates source
identity, and copies/publishes the private shape before queueing the SDK job.
These operations all fall within its current actuator timer. The actuator vector
is already private to the leased job, so the joined path's subsequent vector
copy disappears. This difference in placement explains why the actuator timer
increases by more than the complete receipt-to-SDK prefix does.

There are also compiler-visible differences: byte-mask stores may alias other
storage, and the generated conversion loop repeatedly reloads member/vector
pointers and the actuator count. The new clamp/square-root formulation executes
the square root for clipped values too; the old branch formulation skips it on
out-of-range values. Source validation and shape publication must remain after
conversion and before SDK admission unless their failure semantics are
deliberately changed.

Channel staging is **after SDK dispatch** in both worker paths. Its typical
increase from 4.737 to 8.257 us affects reader-release timing but must not be
added to preparation before native SDK entry. SDK dispatch-to-entry timing
changes only from approximately 2.950 to 3.167 us; it is not the main measured
increase.

### What the workstation checks establish

Production-object inspection on workstation mx11 finds scalar hardware square
root instructions in both paths, without a loss of SIMD vectorization or new
per-actuator library calls. Representative warm-cache loops compiled with the
production optimization flags take approximately 2.915 us for the old conversion
and 3.264 us for the new fused conversion/classification on this workstation.
Caching pointers/counts locally reduces the latter to about 3.038 us. These
numbers describe an Intel workstation microbenchmark, not RTC qualification.

The shared-shape assignment uses an existing Eigen map: it does not allocate or
transpose the image on each command. A separate local ImageStreamIO benchmark,
using a process-private temporary shared-memory directory and no instrument or
reader connections, likewise does not reproduce the multi-microsecond increase
from adding the shape copy. It excludes RTC scheduling, cross-core cache traffic
and real consumer wakeups. Neither benchmark accounts for the full measured
roughly 9.25 us increase in the receipt-to-SDK prefix. Assigning all of that
increase to saturation, clearing or the shape copy would therefore be premature.

### Proposed next change

Move private saturation classification out of the pre-SDK conversion path,
retaining dimension checks before dispatch. It can run after dispatch, as in the
joined implementation, or in the ordered post worker from the immutable job
values before saturation publication. The latter requires keeping each leased
job intact until post retirement and restoring explicit saturation timing.

Cache conversion pointers and counts locally. Consider eliminating redundant
whole-grid clears only after verifying unmapped pixels remain initialized and
mapped pixels are fully overwritten on every slot reuse, including mapping
changes and drained reallocation. Preserve prequeue coherent-source validation
and shared-shape publication so rejected vectors do not produce an SDK command.

Split preparation timing into conversion, source validation and shape
publication to attribute the remaining cost. Also derive receipt-to-SDK-entry
statistics directly from the existing receipt/SDK-start timestamps, without
adding another hot-path timestamp. Current marginal maxima cannot supply that
per-command maximum by subtraction: the latest capture establishes a prefix of
at least 30.087 us in one window, but a separate loose 68.719 us upper bound is
not evidence of a 69 us preparation event.

This investigation updates the engineering record only. No preparation
optimization, telemetry schema, simulator bound or instrument setting has been
changed yet.

## Pre-SDK reduction and preparation telemetry implemented

Following operator approval, the full BMC pipeline now performs saturation-mask
clearing, classification and counting in its ordered post worker. It reads the
leased job's private normalized actuator vector only after monitor staging and
SDK completion. Native failure/exception, cancellation, timeout or preparation
failure suppresses this work and its successful-product publication. A later
loss of reader provenance remains nonfatal to a successful private command.
Controls/reallocation still drain all leased jobs before changing mapping or
storage. The SDK owner can accept the next job independently of this post work.

The conversion loop caches source, shape, vector and mapping pointers and the
actuator count locally. Private shape clearing remains before conversion so
unmapped pixels stay initialized across slot reuse. Coherent source revalidation
and shared-shape publication still precede SDK dispatch, preserving rejection
and publication-failure behavior. Synchronous and joined-worker processing are
unchanged.

Two appended markers divide the original `actuator` interval into three
independently reported stages:

| New metric | Endpoints / interpretation |
| --- | --- |
| `conversion` | Actuator start through private shape initialization, storage checks and numeric conversion. |
| `sourceCheck` | Conversion end through coherent source-identity revalidation. |
| `shapePublication` | Source-check end through shared-shape copy/publication and actuator end. |
| `receiptSdk` | Genuine receipt through actual SDK entry for a successfully submitted command. |

The first three use two additional capture timestamps; `receiptSdk` uses
existing receipt and SDK-entry timestamps without another hot-path clock read.
Its mean/extrema are paired per command, including any dispatch/SDK queue wait;
it is not an isolated conversion-service measurement. Deferred classification
now contributes to `saturation` in pipeline mode. Preparation segments remain
unavailable on paths that do not capture them, while `receiptSdk` can qualify on
synchronous/joined paths with the required successful-send evidence.

The metric IDs append 20–23; all existing IDs and compatible `fgtimes` fields
remain unchanged. Rebuilt readers accept complete historical 12/13/18/20-row
records as well as new 24-row records, projecting absent metrics as unavailable.
New stage names are exposed automatically in `dm_timing_stages` and detailed
`dmtimes` telemetry. Rebuild `bmcCtrl` and `logdump`, including the normal shared
library/precompiled-header dependencies. No new configuration is required, and
the simulator's 35 us preparation assumption is unchanged pending measurement.

Workstation-only validation on mx11:

- Full BMC suite: **1,351 assertions / 30 cases** pass.
- Pipeline ThreadSanitizer suite: **790 assertions / 15 cases** pass with no
  reported data races.
- Timing collector: **592 assertions / 25 cases** pass, including exact paired
  prefix extrema, different SDK durations, segment endpoints and independent
  missing/invalid/failed/synthetic qualification.
- Detailed telemetry: **2,078 assertions / 7 cases** pass, including historical
  20-row records, new metric projections and malformed metric sets.
- DM timing/INDI integration: **168 assertions / 8 cases** pass, including new
  stage availability and direct prefix statistics.
- Generic detached pipeline integration: **203 assertions / 6 cases** pass.
- `libMagAOX`/precompiled header and `logdump` builds pass. The production BMC
  translation unit compiles against the available vendor headers; native BMC
  SDK libraries are absent on this workstation, so a production executable link
  and hardware run are not claimed.
- Changed C++ files pass `clang-format` verification and the diff whitespace
  check. Independent ownership/causality review found no blocking issue.

The deterministic BMC test holds SDK and post execution separately, verifies
that saturation has not run at SDK dispatch or while post is blocked, changes
the shared source before the earlier job completes, and checks private-vector
classification and slot reuse. It also checks that the three preparation
segments partition `actuator` and that direct prefix means match the retained
SDK-entry/receipt pairs. Hardware timing improvement remains to be measured
with the existing simultaneous capture procedure.
