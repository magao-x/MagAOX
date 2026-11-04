# dmSyncopation counter-publication targets

## Current objective

The user clarifies that the target is the **`cnt0++` publication that requests
dmcomb work**, rather than dmcomb's later combined publication, controller
wakeup or SDK call. At 2 kHz main / 8 kHz total, the desired interleaved source
publication grid is:

```text
|---------------+---------------+---------------+---------------|
0              125             250             375             500 us
main cnt0++    aux cnt0++       aux cnt0++       aux cnt0++      main cnt0++
```

Main and auxiliary are different streams and have their own counters. The
target is their interleaved cadence, not 125 us between every pair of updates
of the auxiliary stream alone. Genuine main commands are not retimed. Fit the
long cadence, retain the existing robust phase estimate and freeze each cycle's
slots; a late actual publication must not become the origin of the next slot.
Excursions still cause individual skips, never catch-up writes.

This supersedes the downstream-target interpretation used in the preceding
[combined-trigger plan](dm-syncopation-combined-trigger-deadlines.md). That file
and the [finite-trial record](../2026-10/dm-syncopation-zero-aux-rate-test.md)
remain the engineering history. Instrument actions remain operator-only.

## Source clock and frozen grid

The finite zero mode fits its clock-only estimator directly from coherent
`dm01disp03` publication timestamps and genuine counters. Callback wake/arrival
timestamps no longer define its phase. The existing ten-second cadence baseline,
Q32 fractional period, robust phase fit and committed-cycle rules remain.

After reading coherent primary metadata, capture monotonic / CLOCK_ISIO /
monotonic. `PublicationClock` qualifies the bracket (maximum 1 us), checks
continuity and retains an explicit generation. It projects the frozen producer
grid to local monotonic time solely for waiting, preserving the fractional
remainder and period. No fixed TAI/UTC offset is assumed. Recent source
publications and future targets must be within the four-millisecond horizon;
uncertainty includes a ceiling-rounded one-ppm drift allowance.

An unqualified pair suppresses current writes. A wide bracket retains the raw
cadence fit and is counted; a clock discontinuity/regression starts a new
mapping epoch and resets that fit. Raw coherent main observations and counter
baselines continue across wide captures, avoiding an artificial ten-second
reset for a single wide pair. Snapshot publication enforces arm and active
expiry even when bad clock captures keep callbacks alive and status fresh.

## Counter target versus dispatch preparation

For each slot distinguish:

- `counterTargetRawNs`: frozen desired source publication in CLOCK_ISIO.
- `counterTargetNs`: that same point projected into local monotonic time.
- `targetNs`: final dispatch point, `counterTargetNs - commitLeadUs`.
- `wakeTargetNs`: full-validation wake, `targetNs - wakeAheadUs`.
- `publicationDeadlineNs`: final dispatch admission limit, `targetNs + lateUs`.

`referenceTargetNs` is retained as an alias of `counterTargetNs` in schema 8.
All full validation must finish by the dispatch point. The quiet final wait,
cheap live identity/readiness/fault/expiry guards and final admission check
remain as implemented in the early-wake revision.

The old `publicationAdvanceUs=5` corrected a downstream relative phase and is
removed from the new preset. Any nonzero value now fails configuration with an
explicit diagnostic. `commitLeadUs` is a different quantity: only the measured
final-guard-to-counter preparation cost, bounded to 0-10 us. It initially stays
**zero**, since that cost has not been qualified on hardware. It does not use
the approximately 0.96 us guard-to-API-return duration, which includes work
after the counter update. Do not invent a lead or subtract dmcomb transport.

Keep `ImageStreamIO_UpdateIm()` intact. It samples the raw writetime immediately
before storing metadata and incrementing `cnt0`, then clears `write`, performs
optional history work, posts semaphores and returns. The raw writetime is a
close **pre-counter proxy**, not an exact timestamp of the store. Splitting or
duplicating the library's publication implementation is unnecessary and risks
breaking its circular-buffer/history/semaphore semantics. Actual publication
error and the clock bracket remain visible; the admission cutoff is not a hard
guarantee against preemption inside the update API.

## Direct publication evidence

`PublicationTiming` retains the fixed main and auxiliary source identities and
checks each stream's counter progression separately. It records coherent main
publications before dmcomb association, including an unmatched main, and every
successful auxiliary publication before downstream acknowledgement. Missing
dmcomb linkage therefore does not erase counter-publication evidence.

Full-rate per-step populations include adjacent publication intervals and
errors, relative-main phase error, raw publication error against the frozen raw
target, mapped guard-to-writetime lead, and complete update-API cost. The last
auxiliary-to-next-main interval is included. Gaps, invalid/duplicate evidence,
clock exclusions and source replacement/regression are explicit. A same-source
fault latches measurement exclusion until reset rather than silently mixing
publishers. Cross-source timestamp inversions exclude the affected pair without
latching that fault, allowing collection to recover after a main/auxiliary race.
Existing `zero_phase` combined-output measurements remain separate.

Schema 8 provides:

- `zero_test.targetEndpoint=1`, identifying the counter-publication contract.
- `zero_publication` and `[zero_test] kind=publication` source statistics.
- `zero_clock` and `kind=clock` mapping bounds, generation and coverage.
- Event source inode/counter/raw timestamp, raw/local counter target, local
  dispatch/wake/validation/API markers and the frozen clock offset/bracket.

Raw source intervals/errors never subtract an absolute monotonic timestamp.
The mapped lead estimate uses the paired offset with its reported uncertainty.
Formatting and INDI I/O remain on the ordinary thread via coherent snapshots.

## Next short instrument test

Rebuild/restart dmSyncopation only and merge these settings into the existing
working zero-trial configuration:

```ini
[zeroTest]
publicationAdvanceUs=0
commitLeadUs=0
wakeAheadUs=10
lateUs=2
```

Keep camera/main 2 kHz, synchronization and RTC latency On, existing controller
pipeline and CPU placement, and dmcomb settings/workload unchanged. Use:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 10 0 10
```

The helper requires schema 8 and endpoint 1, verifies the zero downstream
advance, wake-ahead and commit lead before enabling, and records the contract.
An optional fifth argument verifies a later **measured** commit lead; default
zero cannot silently accept a different configured lead. Older endpoint
schemas are rejected. The finite app timer and all capture-quality checks stay.

First examine direct source intervals, per-step raw publication error, missed
slots, clock exclusions and measured pre-counter lead. Only then choose a small
commit lead and repeat; do not tune it from dmcomb output or complete API cost.
A long run remains conditional on the short result. No new hardware result or
qualified literal counter-store bound is claimed by this implementation.

## 2026-11-04: first counter-target capture

`zero-aux-8000Hz-20261104T151038Z` records the new schema-8 endpoint on revision
`64f38e56bc289f7ec55e2b261299ff52c9cc0702`. The helper completes the selected
10-second active interval, with all control/read statuses successful and its
collectors deliberately stopped afterward. The saved setup is main/camera
2 kHz, synchronization On, scalar loop gain 1, EM gain 1, CCD target -45 C and
RTC latency On. The operator confirms stable loop operation and modal gains
remaining zero. Compared with earlier captures, additional latency optimization
has made the overall loop approximately **15 us faster** (operator estimate),
and dmcomb now uses **mode 4 with a 5 us trigger delay**, versus the earlier
10 us. The operator reports lower jitter with that shorter delay. This is the
current optimized baseline; improvement over earlier captures cannot be
attributed solely to the syncopation changes.

The active span is **10.000009231 s**, containing **20,000 main observations**
and **59,997 auxiliary publications, all acknowledged**. Three auxiliary slots
are skipped on the first cycle because that main's combined identity cannot be
matched. There are no later skips, admission/validation/preparation lateness,
readiness/identity/resource/clock failures or unacknowledged publications.
Including startup, the effective source rate is 7,999.693 Hz and auxiliary
delivery is 59,997 / 60,000 = 99.995%.

Full-rate source-publication measurements, in microseconds:

| Interval | Mean | Standard deviation | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Previous auxiliary -> main | 125.050251 | 1.955400 | 118.764 | 148.831 |
| Main -> auxiliary 1 | 124.950321 | 1.953596 | 101.181 | 131.208 |
| Auxiliary 1 -> auxiliary 2 | 124.999867 | 0.036496 | 120.227 | 125.156 |
| Auxiliary 2 -> auxiliary 3 | 125.000098 | 0.013881 | 124.846 | 125.216 |

| Auxiliary target | Mean raw target error | Standard deviation | Maximum lateness |
| --- | ---: | ---: | ---: |
| +125 us | +0.076507 | 0.035712 | 4.882 |
| +250 us | +0.076244 | 0.010885 | 0.229 |
| +375 us | +0.076211 | 0.011037 | 0.273 |

The roughly 1.95 us variation at the main boundaries belongs to the genuine
main publication, which is not retimed. Its own raw target error has mean
+0.126288 us, standard deviation 1.953326 us and range -6.141 to +23.888 us.
The auxiliary grid remains steady across those excursions.

The largest step-1 proxy lateness is consistent with a tail after the final
admission sample, before the ImageStreamIO writetime sample:
`preCounterLead1MaxUs=4.834`, versus target error 4.882 us. It does not represent
a missing publisher or a rejected admission. The step-2 interval minimum of
120.227 us shows the following slot returning to its frozen target. Healthy
event sampling omitted the offending row, so its exact cycle cannot be joined
from this text capture.

Clock qualification rejects no samples and detects no discontinuity, wide
bracket, stale/future publication or invalid projection. Maximum active bracket
width is 160 ns; typical event brackets are about 40-51 ns. Source measurement
has no invalid samples, timing exclusions, source gaps or source faults. Its
one label gap is the startup cycle without auxiliary writes.

The measured guard-to-writetime proxy cost averages only **25.5-25.7 ns**.
Complete API-return duration averages about **0.952 us**, including post-counter
work. Keep `commitLeadUs=0`: tuning a few tens of nanoseconds from the paired
clock estimate would overstate its precision. Sampled validation finishes
5.6-9.8 us before dispatch and payload staging 61.6-82.5 us before dispatch.
There is no evidence that preparation margin needs increasing.

Diagnostic accounting reconciles 79,996 healthy outcomes into 20,000 retained
rows and 59,996 intentionally sampled out, plus six lifecycle/exception rows.
Event queue drops and log enqueue failures are zero. The queue high-water mark
is 2,004 of 20,000, and final text records show it drained.

Controller before/after counters show no new overload, rejected/coalesced/
superseded/cancelled job, vector/preparation/post failure, reader-unqualified
record, SDK failure/timeout or DM timing loss. Clean fully active controller
windows have mean receipt intervals of 125.000 us and standard deviations of
2.75-3.25 us; source publication and controller wake remain distinct endpoints.
All 19,999 classified main commands have matching SDK and reader records. The
713 timing-evidence invalidations and 56 SDK-observation gaps remain diagnostic
coverage caveats; they do not establish lost commands. Camera counters show no
new publication/frame gaps or drops, and its acquisition/publication interval
spread is comparable to the disabled baseline within this capture. The
controller's separate delivery-clock collector excludes 115 wide brackets
over the complete text collection; that is distinct from the zero-trial source
clock, which rejects none. These observations do not establish zero added main
SDK wait or a physical camera-latency bound.

The next useful measurement is **120 seconds with this exact preset**, keeping
`commitLeadUs=0`, dmcomb mode 4 / 5 us delay, and the current optimized loop,
other settings and workload unchanged:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 120 0 10
```

No code or configuration tuning is needed for that longer comparison. No agent
operated the instrument.

## 2026-11-04: 120-second capture exposes a periodic delay

`zero-aux-8000Hz-20261104T153017Z` uses the same revision, schema-8 endpoint and
zero commit lead, with a 10 us early wake and 2 us admission allowance. The
operator confirms stable loop operation, modal gains zero, unchanged workload
and the same optimized setup, including dmcomb mode 4 / 5 us delay. The helper
finishes normally after **120.000112717 s**; all control/read statuses succeed
and collectors stop intentionally.

| Outcome | Count |
| --- | ---: |
| Genuine main observations | 240,000 |
| Auxiliary opportunities | 720,000 |
| Auxiliary posted and acknowledged | 719,308 |
| Initial unmatched-main dependent skips | 3 |
| Later lateness skips | 689 |
| Unacknowledged publications | 0 |

Auxiliary delivery is **99.903889%** and the observed total source rate across
the full active interval is **7,994.226 Hz**. Every posted
auxiliary is acknowledged. No preparation, reader-clearance, identity,
resource, clock, wait or cancellation failure occurs.

Full-rate raw publication error, in microseconds:

| Step | Count | Mean error | Standard deviation | Maximum lateness |
| --- | ---: | ---: | ---: | ---: |
| Main | 240,000 | +0.138642 | 1.834088 | 23.736 |
| Auxiliary 1 | 239,999 | +0.076293 | 0.011179 | 0.285 |
| Auxiliary 2 | 239,311 | +0.085809 | 0.116607 | 4.674 |
| Auxiliary 3 | 239,998 | +0.076403 | 0.011058 | 0.295 |

The adjacent auxiliary-1-to-2 interval has mean 125.009635 us, standard deviation
0.116933 us and range 124.775-129.615 us; auxiliary-2-to-3 has mean 124.990711 us,
standard deviation 0.116900 us and range 120.387-125.227 us. These qualified
adjacency populations exclude missing intended slots. A skipped slot still
creates a real roughly 250 us delivery gap, visible in controller intervals;
near-125-us means of retained adjacent pairs do not eliminate those gaps.

### The misses cluster in one phase band

**688** of the late skips are step 2 at `stage=publication_guard`. They occur
only **114.588358-118.758360 s after trial start**, a 4.170-second band near the
end. Counts by active second are 43 / 340 / 234 / 64 / 7 for seconds 114-118.
Every affected primary counter is odd: only every other 500 us main cycle is
affected. Their local targets modulo 1 ms lie in **136.695-139.687 us**, a
2.992 us phase band.

For all 688 cases, the payload had been staged **66.358-81.665 us early** and
full validation had finished **7.916-9.236 us early**. Rejection clock samples
are **2.002-5.259 us after target**, with median 2.582 us. Thus the writer,
passive preparation and full early validation met their deadlines. Delay arose
in the final wait/guard interval, exceeding the selected 2 us allowance.

This supports investigating a **1 kHz phase-related disturbance**. IRQ activity,
preemption, clock-read latency or final-guard work are possibilities; existing
markers do not identify which one caused it. It is not evidence that the
publisher was late or that the controller rejected the command. Before this
phase band, the full-rate step-2 target-error maximum is only 0.275 us and its
standard deviation is 0.011000 us. The same band produces posted step-2 tails
up to 4.674 us, with guard-to-writetime proxy cost up to 4.634 us. Subsequent
slots retain their original targets.

The other late skip is step 3 at active **69.871939 s**, `stage=validation`.
The planned early wake overshot by 12.515 us and validation finished 2.765 us
after target, despite payload staging 77.126 us early. It is a separate
isolated wake/validation miss. `validationLate=1` is included in the 689 late
skips, not an additional outcome.

### Coverage and simultaneous controller/camera evidence

Source-clock mapping has zero rejection or discontinuity; its largest active
bracket is 401 ns. Source statistics have zero invalid evidence, source gaps,
source faults or timing exclusions. There are 690 label gaps corresponding to
the missing intended slots/boundary. Typical guard-to-writetime proxy cost
remains about 25.5 ns. **Keep `commitLeadUs=0`**; changing a tens-of-nanoseconds
lead does not address this microsecond-scale disturbance.

Healthy event sampling retains **239,999** rows, intentionally omits 719,308,
and retains all 695 lifecycle/exception rows, for 240,694 event rows. The
959,307 full-rate healthy outcomes reconcile exactly. Queue/log/reader drops
remain zero. Successful stride-4 event sampling is phase-locked here: sampled
auxiliary successes have primary counter modulo 4 equal to zero, while the 688
step-2 skips have odd counters. Consequently sampled successful events miss
the disturbed parity. Use the full-rate populations for tails; event-only
success distributions are not representative of this periodic effect. A future
diagnostic should preserve successful timing outliers or rotate sampling phase
without losing the existing complete skip evidence.

Controller snapshot counters increase by 996,662 receipts, submissions,
admissions, completions and retirements over the wider before/after interval.
There are no new rejection/coalescing, overload, cancellation/supersession,
preparation/post failure, SDK failure/timeout or DM timing gap/drop. Fully active
controller windows show receipt intervals from 96.743 to 262.456 us; windows
containing missing slots have elevated mean/spread. These tails do not have
accompanying controller admission failures.

All **239,999 classified main commands** have SDK and reader matches; the only
unmatched main is the initial cycle. Main observer-receipt-to-SDK has mean
29.186 us and maximum 43.302 us, versus 29.007 / 38.202 us in the short run.
The read-done-start-to-SDK prefix is 16.011 / 26.270 us. No directly isolated
main SDK-wait population exists, so these measurements do not prove zero added
main wait. Native main SDK service is 103.843 / 106.871 us. Diagnostic exclusions
remain explicit: 8,928 invalid timing observations and 988 SDK-observation gaps
in the after snapshot (989 in later text as polling continues), plus 119 wide
brackets in the controller's separate delivery-clock collector.

Camera before/after counters add 279,847 frames without new frame/counter gaps,
failed/rejected frames or drops. Fully active acquisition interval standard
deviations are 0.214-1.635 us and publication deviations 1.020-1.734 us. Camera
frame-counter continuity alone still cannot exclude a persistent pipeline lag.

The next investigation should localize the periodic delay on synco's timing
thread/CPU, with IRQ/scheduling evidence and final-wait/guard timing that is not
aliased by healthy stride-4 sampling. Retain the current 2 us admission rule and
other settings during that investigation. No implementation or instrument
setting is changed by this analysis, and no agent operated the instrument.

## 2026-11-04: unchanged repeat moves the affected stage

`zero-aux-8000Hz-20261104T155241Z` repeats the 120-second trial with no changes,
as confirmed by the operator. Revision and preset match the previous capture.
The helper finishes normally after **120.000001813 s**, with no new event/log/
reader queue drops or controller admission/execution failure.

| Outcome | Previous 153017Z | Repeat 155241Z |
| --- | ---: | ---: |
| Main cycles excluded by clock qualification | 0 | 113 |
| Reported main observations | 240,000 | 239,887 |
| Unmatched main combinations | 1 | 226 |
| Explicit auxiliary skips | 692 | 678 |
| Auxiliary publications | 719,308 | 718,983 |
| Auxiliary acknowledgements observed | 719,308 | 718,982 |
| Auxiliary lateness skips | 689 | 0 |

The denominator remains **240,000 genuine main cycles / 720,000 auxiliary
opportunities**. Clock qualification rejects 113 whole cycles before per-slot
bookkeeping, omitting 339 opportunities from the explicit skip counters.
Accounting is therefore `720000 - 339 - 678 = 718983` auxiliary publications,
or **99.858750%**. There are **1,017 absent auxiliary publications**, not just
678. Including all genuine mains, the observed source rate is 7,991.525 Hz;
using the smaller reported `mainObserved` would incorrectly omit genuine mains.

The final unacknowledged publication is step 3 of the last cycle, with
`reason=duration_expired` in acknowledgement. Its target is 3.662 us before the
finite cutoff and API return 2.676 us before it. The app stops waiting at the
cutoff; this is a posted command with truncated acknowledgement observation,
not an acknowledgement timeout or a demonstrated SDK/publication failure.

All 113 clock exclusions are **wide monotonic/producer/monotonic brackets**
exceeding the 1 us capture limit. There are no discontinuities, malformed,
future or stale timestamps, or mapping/projection errors. Maximum accepted
active bracket is 691 ns and the mapping remains in epoch 6. `readerInvalid`
also reaches 113; without per-record rejection evidence, equal totals alone do
not establish a shared cause. Source statistics report 113 excluded source gaps,
with zero invalid evidence, source faults or timing exclusions among accepted
observations.

### Phase comparison

The main frozen-grid phase modulo 500 us progresses from **84.219 us** near
startup to **206.219 us** near completion: 123.562 us at 40 s, 133.269 us at 50 s
and 144.903 us at 60 s. The new clock/recovery disturbances occur around
**40.133-55.004 s**, as the main observation stage crosses the phase region
associated with the preceding run's +250 us final-guard disturbance.

Every first recovery event is on an even main counter, with target modulo 1 ms
623.938-640.623 us. Subtracting the preceding 500 us main period gives an
**inferred rejected-main target phase of 123.938-140.623 us modulo 1 ms**,
overlapping the prior 136.695-139.687 us auxiliary miss band. Auxiliary target
phases in this repeat avoid that band, consistent with zero final-guard misses.

This strengthens the periodic phase-interference hypothesis while explaining
why identical settings can affect a different stage on a later run. The exact
rejected clock-read brackets/timestamps are not retained as event rows; their
phase attribution is inferred from neighboring identities and the frozen grid.
It remains insufficient to identify a particular interrupt, scheduler action,
clock-read operation or hardware source.

### Conservative rebaselining amplifies the exclusions

The post-start unmatched main events form 113 clusters: 112 adjacent-counter
pairs and one singleton. The singleton is followed two counters later by the
next cluster, consistent with a new excluded clock cycle overlapping recovery.
The current code clears `m_zeroHaveCombined` on clock rejection, then breaks
immediately on its first otherwise coherent combined sample when no baseline
is available.

The first event in every cluster samples a combined publication
**478.714-499.633 us older** than the current main: the preceding excluded main's
output. The following 112 unmatched events have current combined timestamps
**5.179-14.678 us after** their own main source, but the counter is two ahead of
that stale baseline, so another association reset occurs. These are available
main combinations rejected during recovery, not evidence that dmcomb failed
to produce the main output.

Thus a rejected clock sample suppresses its own cycle and usually two more
cycles during association recovery. Including the overlap and startup, there
are 339 main cycles without auxiliary publications. A follow-up implementation
should evaluate waiting for the current main's output before establishing a
new baseline, while retaining identity/readiness and timestamp validation. It
should also report whole-cycle exclusions alongside per-slot skips and retain
the rejected clock bracket as diagnostic evidence. No such change is made in
this analysis, and the 1 us clock-quality limit is not relaxed.

### Timing and downstream evidence

Qualified auxiliary publication timing remains precise:

| Step | Mean raw target error, us | Standard deviation, us | Maximum lateness, us |
| --- | ---: | ---: | ---: |
| Auxiliary 1 | +0.077526 | 0.013014 | 0.444 |
| Auxiliary 2 | +0.077378 | 0.012763 | 0.340 |
| Auxiliary 3 | +0.077335 | 0.012618 | 0.321 |

Auxiliary-to-auxiliary interval standard deviations are 14.49 and 14.16 ns.
Typical guard-to-writetime proxy cost is about 26.6 ns. Keep `commitLeadUs=0`:
these figures do not justify changing the lead to address clock exclusions or
association recovery.

Controller before/after counts add 996,345 receipts/submissions/completions/
retirements and 996,344 admissions, with `inFlight` changing from one to zero at
the snapshot boundary. No new overload, rejection/coalescing, cancellation/
supersession, preparation/post failure, SDK failure/timeout or timing gap/drop
occurs. The largest active receipt interval is 512.367 us during the exclusion
cluster; camera acquisition intervals in that report remain 499-501 us. This
is not a camera-rate interruption.

All 239,661 classified mains have SDK and reader matches. Main observer-to-SDK
mean/maximum is 29.231 / 42.159 us, versus 29.186 / 43.302 us previously; native
main SDK service is 103.948 / 109.367 us. Diagnostic caveats remain: 8,283 invalid
timing observations, 929 SDK-observation gaps in after-INDI (930 in later text),
and 124 wide brackets in the separate controller delivery-clock collector.
Exact added main SDK wait is still not isolated.

Camera snapshots add 278,787 frames with no new frame/counter gap, rejection,
failure or drop. Active timing is comparable to the previous run. A separate
roughly +108/-109 us publication excursion occurs before the trial; PDV
acquisition intervals stay at 500 us and SDK wait-return timing shows a delay
and catch-up. Do not attribute that pretrial excursion to active syncopation.

Healthy outcome sampling reconciles 958,643 full-rate outcomes into 239,660
retained and 718,983 intentionally omitted rows, plus 906 exceptional/lifecycle
rows. Clock-excluded cycles rotate the `mainObserved` sampling phase, so the
previous run's fixed-parity alias is not global in this repeat. No event/log/
reader queue loss is observed. The next diagnostic remains exact clock-reject
evidence and IRQ/scheduling timing, together with review of rebaselining's
avoidable recovery penalty. No agent operated the instrument.

## 2026-11-04: clock-rejection evidence and bounded rebaselining

The user authorizes retaining exact rejected-clock evidence and improving
current-main rebaselining. The implementation is limited to the finite zero
trial and its test/capture workflow. Clock capture remains bounded to **1 us**,
final admission remains **2 us** in the preset, and commit lead remains zero.
No IRQ, scheduler, backend, camera or instrument configuration changes here.

Schema **9** retains every enabled clock rejection in the existing bounded
event queue, independent of healthy-event sampling. Each event includes the
exact monotonic-before / producer / monotonic-after readings, applied bracket
and age limits, primary inode/counter/raw writetime, mapping epoch, raw fitted
target when available and a named rejection cause. Invalid mappings have no
local wait target. Rejection causes distinguish malformed captures, wide
brackets, discontinuity, future/stale publication and unrepresentable mapping.
Formatting and telemetry I/O remain on the ordinary thread.

`clockRejected` remains the armed/warmup-inclusive diagnostic total. New
`clockExcludedCycles` and `clockExcludedSlots` count only observations inside
the already-started finite interval. `mainCycles` combines accepted observations
and those active exclusions; `auxiliaryOpportunities` accumulates the actual
slot count before publication decisions, including excluded clock cycles.
For counted active cycles the partition is `posted + skipped +
clockExcludedSlots`. A disabled settings change cannot retroactively alter the
stored opportunity total. These fields do not infer missing upstream callbacks
or incoherent metadata observations, and qualified callbacks still waiting for
cadence acquisition do not enter the scheduled-cycle population.

When a combined baseline is unavailable, a coherent **stale** output may seed
a local waiting baseline. The existing fixed main deadline, resource/primary
identity and finite/cancellation checks continue to apply. Association requires
exactly one counter increment with a strictly progressing publication timestamp
at or after the current main. Only then is that current output committed as the
trusted identity, with `mainRebaselined` incremented; auxiliary reader-clearance
checks and frozen targets are unchanged. Failure with only a stale output does
not promote it to a trusted baseline for the following cycle.

An already-fresh output with no trustworthy baseline is still conservatively
skipped/rebaselined. A fresh timestamp alone cannot identify the producing main,
and read-done identifies the combined command rather than its original source.
Counter jumps/regressions, altered identity, timeout, cancellation and expiry
remain rejection paths. The observed stale-output/current-output recovery
sequence can now complete within its current cycle rather than introducing the
two additional association skips. Hardware verification is still required.

The capture helper requires schema 9 both before settings changes and in the
configured snapshot before enable, while preserving endpoint, startup lead,
duration, freshness, cleanup and quality checks. The operator guide documents
the new event fields and denominator accounting. Rebuild/restart dmSyncopation
only, keeping the current 5 us dmcomb delay and other settings unchanged, then
repeat:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 120 0 10
```

Workstation verification passes **141 application cases / 5,669,253
assertions**, **52 private fake-service capture-helper tests**, a clean
production dmSyncopation build, clang-format, Bash syntax and diff checks.
Coverage includes exact unsampled bracket evidence, active/warmup/terminal
accounting, consecutive clock exclusions, future/stale/malformed/discontinuous
clock causes, immutable opportunity totals after disabled settings changes,
delayed current-main recovery and unchanged slot targets. Recovery guards cover
missing output, unproven fresh output, extra counter update, replaced source,
disable, changed primary, finite expiry and wait error.

Review also adds a frozen combined-inode check immediately after the coherent
read and a post-read clock check before accepting a main association, so a
replacement or delayed read cannot bypass the stated identity/deadline bounds.
The first suite exposed a safely rejected source replacement reported under
the generic association cause; the explicit inode check now preserves the
resource-unavailable cause as well. No remaining review issue is identified.

Affected runtime files are `dmSyncopation.hpp` and `dmSyncopationZeroTest.hpp`;
the app tests, capture helper/private tests, operator guide and engineering
record accompany them. No common device or backend code changed. The new
hardware comparison remains pending.

## 2026-11-04: schema-9 capture validates clock recovery and exposes a metadata guard regression

`zero-aux-8000Hz-20261104T202704Z` uses revision `656be22d`, schema 9 and the
same 120-second preset. The operator reports no external changes. The helper
finishes normally after **120.000003839 s** with 240,000 counted main cycles and
720,000 auxiliary opportunities. The new denominator fields reconcile directly:

```text
717480 posted + 2517 explicit skips + 3 clock-excluded slots = 720000
```

Thus 717,480 auxiliary publications were made (**99.650000%** of opportunities),
of which 717,479 acknowledgements were observed. The final unacknowledged
publication again has `duration_expired` during acknowledgement; it is not a
demonstrated SDK or publication failure. The 2,517 explicit skips separate into
**1,767 main-association skips** and **750 step-2 final-guard lateness skips**.
There are no preparation-late, validation-late, reader-missing, changed-primary,
resource, wait-error or cancellation outcomes. `readerInvalid=1` remains an
explicit readiness-evidence exclusion, not a reader queue drop.

### Clock recovery works on the one rejected bracket

There is one active clock rejection at approximately **19.888 s**, primary
counter `115541820`. The retained event is `wide_bracket`, with exact readings:

| Field | Nanoseconds |
| --- | ---: |
| Monotonic before | 2691901085839841 |
| Producer clock | 1793824062495714834 |
| Monotonic after | 2691901085841154 |
| Bracket width | 1,313 |
| Acceptance limit | 1,000 |

The local read bracket spans **839.841-841.154 us modulo 1 ms**, away from the
roughly 137 us final-guard disturbance band. It omits exactly one cycle's three
auxiliary opportunities. `mainRebaselined=1` and the neighboring association
evidence show recovery without the old two-cycle stale-output cascade. No
clock discontinuity, malformed/future/stale publication or mapping error is
reported. The other main-association failures therefore cannot be assigned to
clock recovery. Accepted active brackets reach 480 ns, and source statistics
have no invalid evidence, timing exclusions or source fault, with one excluded
source gap corresponding to the rejected cycle.

### The new unchanged-counter guard rejects a normal publication transition

There are **589 unmatched mains**: one startup event plus **196 repeating
three-main sequences** (192 isolated triplets and two adjacent six-main groups).
Those 589 mains account for all 1,767 association-dependent auxiliary skips.
The new stricter unchanged-counter/changed-timestamp check is implicated by the
reconstructed counter sequence. For 43 groups, the directly retained preceding
step-3 sample independently proves the unchanged counter/new writetime at the
first rejection; the repeating unmatched-main identities support the same
pattern throughout:

1. The combined counter still equals the preceding retained counter, but its
   writetime has already changed to the current main publication.
2. The next sampled counter is one higher with **the exact same writetime**.
3. The following current output is two counters ahead of that stale association
   baseline, causing another conservative reset.

ImageStreamIO publishes writetime before incrementing the counter. Treating
that intermediate metadata state as a fatal association change introduces a
regression in otherwise normal publication handling. The improved rejected-clock
recovery is not the cause of these 196 groups. The implementation needs to wait
through the unchanged-counter metadata transition while preserving the frozen
source identity, fixed deadline and exact increment requirement. That correction
and its regression tests are in progress; this capture is not its qualification.

### The periodic final-guard pattern also recurs

All **750 late skips** affect step 2 and **even** primary counters. They occur at
active **74.556324-78.917325 s**, with targets **136.695-139.539 us modulo 1 ms**.
The parity differs from the earlier run, but the absolute phase band agrees.
Rejection samples are **2.001-4.924 us late**, median 2.5965 us. Full validation
had completed **8.072-9.363 us early**, and payload staging **65.550-81.393 us
early**. This remains a final-wait/guard timing disturbance rather than slow
preparation. No particular interrupt or kernel operation is identified.

Full-rate auxiliary raw-target errors, in microseconds:

| Step | Count | Mean error | Standard deviation | Maximum lateness |
| --- | ---: | ---: | ---: | ---: |
| Auxiliary 1 | 239,410 | +0.076070 | 0.013751 | 3.276 |
| Auxiliary 2 | 238,660 | +0.084660 | 0.111166 | 5.150 |
| Auxiliary 3 | 239,410 | +0.076112 | 0.011354 | 0.849 |

Healthy event sampling retains 239,007 of 956,889 healthy outcomes and
intentionally omits 717,882. The 3,109 exceptional/lifecycle rows are retained,
for 242,116 event rows total. There are no event/log or reader queue drops.
Retained healthy successes are still only a subset; use full-rate populations
for timing tails and all retained exceptions for the association/skip patterns.

### Controller and camera remain free of new execution/loss faults

Controller snapshots add 996,838 timing receipts/submissions and 996,837 SDK
completions, admissions and retirements; the cross-property snapshots differ
by one count. There is no increase in overload, rejection/coalescing,
cancellation/supersession, preparation/post failure, SDK failure/timeout,
no-send, timing counter-gap or timing-drop totals. Whole-text timing records add
1,012,874 events/submissions without those failures, with 77 wide-bracket
exclusions in the controller's separate delivery-clock collector.

Fully active controller receipt intervals span **90.441-518.769 us**, with
two-second window means 124.9997-130.4130 us and standard deviations
2.54-28.70 us. Missing auxiliary slots remain visible as real delivery gaps;
there is no accompanying controller admission-failure increase. SDK worker
service means are 106.231-106.601 us, maximum 115.948 us. Preparation maximum
is 24.036 us and reader-release maximum 32.741 us.

All **239,410 classified mains** have SDK and reader matches. Main observer
receipt-to-SDK mean/maximum is **29.338 / 39.875 us**, reader-start-to-SDK is
**15.947 / 23.795 us**, and native main SDK service is **103.873 / 107.984 us**.
The separate diagnostic exclusions are 8,617 invalid timing observations and
1,036 SDK-observation gaps. These records do not isolate added main SDK wait.

Camera snapshots add 276,778 frames, with no new failed/rejected frames,
frame/counter gaps, drops or wide brackets. Four new excursion traces and
13 housekeeping controls are recorded without new control/trace failures.
Active acquisition interval standard deviations are 0.238-0.838 us and
publication deviations 1.060-1.435 us; mean acquisition-to-write is
48.154-48.571 us, maximum 53.681 us. The four retained camera excursions show
serial overlap and acquisition/publication changes of roughly 20-27 us; they
do not establish frame loss or identify the separate 1 kHz synco disturbance.
Contiguous camera frame counters still do not measure persistent frame lag.

No instrument action was performed by an agent. The immediate next step is to
correct and test the normal metadata-transition regression, while retaining the
new clock evidence, complete opportunity accounting and existing timing bounds.

## 2026-11-04: correct the writetime-before-counter transition

The capture exposes a regression introduced in the association hardening: a
new writetime with an unchanged counter was treated as a fatal mismatch. The
ImageStreamIO update sequence stores writetime before incrementing `cnt0`.
Polling can observe that intermediate state consistently across both snapshot
reads, so metadata consistency alone is not proof of completed publication.

The correction keeps the trusted/local baseline unchanged while the observed
counter is unchanged, even if writetime has advanced. It continues the existing
bounded wait; it neither accepts the current command nor copies auxiliary data
on timestamp evidence alone. Matching still requires exactly the expected
counter increment, a progressing current-main timestamp, the frozen source
identity and the original association deadline. Reader clearance and all final
publication guards remain unchanged.

If that pending counter never commits before timeout/cancellation/expiry, the
uncommitted timestamp/counter pair is not promoted into `m_zeroLastCombined`.
The old stable key is retained and `m_zeroHaveCombined` is cleared. This also
avoids turning a timeout during a partial publication into a poisoned baseline
and subsequent counter-reset cascade. The existing conservative fallback for
an already-fresh candidate with no known baseline remains unchanged.

This is a narrowly scoped correction to `dmSyncopationZeroTest.hpp`, with a
private fixture that exposes writetime first and delays the main counter
commit while the write flag remains clear. Schema 9, capture arguments,
configuration and timing thresholds are unchanged. The periodic approximately
137 us modulo-1-ms final-guard disturbance remains a separate open issue; this
correction addresses only the normal-publication association regression.

Workstation verification passes **143 application cases / 6,090,323
assertions**, a clean production dmSyncopation build, clang-format and diff
checks. The new fixture exposes the main timestamp while leaving the old
counter and a clear write flag visible, then advances the counter while
preserving that exact timestamp. Both ordinary association and rejected-clock
recovery wait through a 4 us intermediate state, publish all scheduled slots
on their original targets and avoid subsequent cascades. Assertions explicitly
exclude main acceptance or auxiliary publication before counter completion.
An uncompleted transition remains fail-closed under timeout, cancellation,
finite expiry, counter jump and regression; pending metadata never replaces
the stable baseline.

No capture-helper/schema/configuration changes are needed. Rebuild/restart
dmSyncopation only and repeat the unchanged 120-second command above. The
periodic final-guard delay still needs separate IRQ/scheduling investigation.
The hardware capture does not qualify this correction, and no agent operated
the instrument.
