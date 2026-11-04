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
