# dmSyncopation combined-trigger targets and earlier publication deadlines

## Direction and current status

The user clarified the timing objective: at 2 kHz main / 8 kHz total, dmcomb
should trigger the DM controller at nominal offsets 0, 125, 250 and 375 us.
The user directs placing each auxiliary deadline before its downstream trigger
point so that the publication/combination work can finish in time.

This document records the revised endpoint and the measurements/design needed
to choose the lead. The first experimental differential-advance test is now
implemented, as described below. Default behavior remains the prior unadvanced
auxiliary-publication grid. No qualified combined-trigger tolerance or hard
worst-case transport bound is established by this test.

The [main plan](../2026-10/dm-syncopation.md) and
[finite-trial engineering record](../2026-10/dm-syncopation-zero-aux-rate-test.md)
remain the prior implementation/qualification history. Actual instrument
commands and captures are operator actions; agent work stays on the workstation.

## Separate the timing endpoints

| Endpoint | Meaning |
| --- | --- |
| Main source publication | Genuine AO command counter in `dm01disp03`; supplies cycle identity and cadence. |
| Primary monitor observation | Local monotonic callback time currently used to fit the grid phase; includes observation lag. |
| Auxiliary publication | Synco commits the selected auxiliary counter/timestamp and posts its semaphores. |
| Combined publication | dmcomb commits `dm01disp` after reading/summing the channels. |
| Controller semaphore post | The specific dmcomb semaphore notification consumed by the DM controller; desired trigger endpoint. |
| Controller receipt | First controller timestamp after successful semaphore wait; useful delivered-trigger evidence, with wake latency. |
| Native SDK entry/return | Backend execution after controller preparation/queueing; separate from trigger timing. |

`dmtweeter` fgtimes `acq` measures successive controller receipts; `wrt` measures
successive qualified backend returns. A mean near 125 us is useful delivered
cadence evidence, but it does not show every labelled slot's downstream phase
or separate the four transition populations.

`ImageStreamIO_UpdateIm` writes `CLOCK_ISIO` metadata, increments the counter,
clears `write`, optionally records write history, then posts all semaphores.
The combined `writetime` is therefore an observable publication proxy preceding
the exact selected semaphore post. Synco's existing event `observedNs` is after
its auxiliary update API returns; it is not the publication origin. Literal
semaphore-post timing requires a qualified bound/bracket on that offset, or
instrumentation at the selected dmcomb post. Controller wake adds another
separate delay. Preserve these distinctions in configuration and reports.

## What the existing capture already establishes

Using trial 2 of `zero-aux-8000Hz-20261102T025808Z`, select 59,965 complete
sampled cycles with a common combined inode and four consecutive combined
counters. All differences below use raw combined publication timestamps in
one producer clock; absolute producer timestamps are never subtracted from
local monotonic targets.

| Combined-publication transition | Mean us | Median us | Observed range us |
| --- | ---: | ---: | ---: |
| Main to auxiliary 1 | 129.314 | 128.092 | 103.124-141.928 |
| Auxiliary 1 to 2 | 124.847 | 127.882 | 114.406-140.616 |
| Auxiliary 2 to 3 | 125.744 | 128.763 | 115.428-135.656 |

Mean auxiliary combined-publication offsets from the same cycle's actual main
combination are **129.314 / 254.161 / 379.905 us**. Relative to nominal
125/250/375 offsets, the means are about **4.314 / 4.161 / 4.905 us late**.
This supports investigating a few microseconds of differential phase
correction, not adopting those values as full path delays or hard bounds.
Actual main jitter and delay-mode polling phase contribute to the spread.

The last auxiliary to the next main cannot be reconstructed as an adjacent
pair from this sampled trace: `eventStride=4` omits three intermediate cycles.
Counter gaps 12/13 must not be treated as adjacent commands. Add full-rate
paired summaries for all four transitions, especially the last-to-main gap.
The retained sampled complete cycles remain useful cross-checks.

A common downstream delay largely cancels in adjacent intervals. Moreover,
the current phase is fitted from primary-monitor observations. If `M` is the
main-grid-to-combined offset and `A_j` the auxiliary path offset, its relative
combined phase error is approximately `A_j - M`. Blindly subtracting the full
auxiliary delay from the existing grid would overcorrect by the main offset.
Either fit a qualified combined-main phase first, or explicitly estimate the
differential phase correction; do not confuse the two calibration quantities.

## Desired grid and deadline hierarchy

Maintain the long-baseline cadence and freeze future cycles as already agreed.
Use labelled genuine-main timing to determine a slowly fitted selected-post
phase `H_k` once that endpoint mapping is qualified. Until then, explicitly
label a combined-publication proxy rather than claiming the actual post phase. Desired downstream targets are:

```text
D_0 = H_k                     genuine main, not an auxiliary post
D_1 = H_k + period / 4         nominal +125 us
D_2 = H_k + period / 2         nominal +250 us
D_3 = H_k + 3 * period / 4     nominal +375 us
```

Do not retime committed targets to each individual early/late main. Main
excursions still invalidate or skip conflicting auxiliary occurrences; no
catch-up burst or primary-stream publication is added.

For each step distinguish:

1. `D_j`: desired combined-trigger target, and its separately specified allowed
   late output error `E_j`. Qualify early output error separately.
2. `P_j`: latest permissible auxiliary publication/admission checkpoint, early
   enough to cover its qualified remaining path budget `B_j`. Conceptually,
   `P_j = D_j + E_j - B_j`; a before-target cutoff uses `E_j=0`.
3. An earlier wake/final-validation time, allowing measured wake and check cost
   before `P_j`.
4. An earlier publisher-ready/snapshot/copy cutoff, allowing complete staging
   before the remaining checks/publication window.

Define `B_j` from the precise checkpoint guarded by the implementation through
the selected combined post, including intervening API/metadata/semaphore work.
Do not omit the time between the last clock check and the actual publication.
A nominal compensated publication time can be based on a typical lead, but it
must also fit any chosen latest-publication cutoff. A variable polling delay
can make tight error bounds and high delivery fraction incompatible; report
that feasibility tradeoff rather than promising exact 125 us spacing.

The existing 5 us test grace must not silently allow publication beyond a new
hard cutoff. Wake tolerance, validation cost and publication uncertainty must
fit inside the chosen advance. The main plan's 5 us analysis guideline is not
a new hard combined-post alarm/safety requirement. Output quality tolerances,
statistical leads and conservative admission budgets remain separate settings.

Actual prior-command reader clearance remains mandatory before modifying
shared channels. If the command cannot be staged and admitted by its cutoff,
skip that occurrence. Continue to copy every selected contribution; no HOLD or
pixel-equality comparison is introduced.

## Measurement-first implementation sequence

1. **Add exact paired transport evidence without changing targets.** Retain
   the auxiliary counter and existing raw writetime immediately after a coherent
   successful update, with monotonic timestamps bracketing the update API.
   Existing combined counter/writetime then supplies same-clock
   auxiliary-publication-to-combined-publication latency. Reuse an existing
   final-guard timestamp where it provides the correct bracket boundary.
   Check identity/counter consistency and explicitly count torn/ambiguous/lost
   measurements. Format records off the real-time path with bounded storage.
2. **Measure the requested downstream endpoint.** Reuse combined publication
   and controller delivery-clock brackets. If literal selected semaphore-post
   timing cannot be bounded adequately, instrument that dmcomb post; do not
   label publication or wake timestamps as the post itself. Export the existing
   true controller receipt marker by exact combined identity for mains and
   auxiliaries. Include the already captured producer-clock bracket when an
   external-to-monotonic phase conversion is required. Reject discontinuities,
   wide brackets and incompatible clocks; do not synthesize absolute mappings
   from rolling aggregate statistics.
3. **Produce slow per-step calibration.** Fit the main combined/trigger phase
   against the established cadence with qualified mapping, or explicitly report
   a differential correction against the existing primary grid. Estimate
   nominal and tail path delays over a long window. Any bounded/slewed feedback
   applies only to unadvertised cycles, with no chasing of individual
   excursions. Preserve counts for main/aux classifications, exclusions and
   skipped commands. Mode-4 asynchronous counter polling adds variable phase;
   a fixed lead can reduce a bias but cannot remove that jitter.
4. **Select the numerical lead and cutoffs from the measurements.** Keep the
   desired trigger, scheduled wake/publication and writer-ready cutoff distinct
   in the timing stream and INDI. Version any changed field meaning instead of
   silently reinterpreting `targetNs`. Freeze the entire deadline set by
   epoch/cycle/step. Validate reader and backend overlap constraints without
   inferring SDK idleness from release alone.
5. **Qualify the changed schedule with ideal zero commands first.** Compare
   main-to-1, 1-to-2, 2-to-3 and 3-to-next-main intervals, per-step output phase
   error, true controller receipt spacing, tail/error quantiles and skip rates.
   Keep controller admission/retirement/loss and readiness checks alongside
   phase improvements. Start short, then repeat 120 s with unchanged conditions.
   The previous zero-lead captures are the control; no numerical advance is
   enabled by this document.
6. **Publish provider deadlines after the schedule is qualified.** Providers
   receive coherent future cycle/step targets and earlier ready cutoffs, with
   explicit clock, epoch and expiry. Missing ready data causes a skip; the
   publication deadline is not the external writer's copy-start time. Real
   nonzero provider qualification remains separate from this ideal-zero test.

## Correction to previous exact-main latency comparisons

Source audit shows that `ZeroTiming.controllerToSdk` computes
`SDK.startedNs - read_done.startedNs`. `dmCompletion::begin()` timestamps the
reader record after input-identity capture in `capturePipelineCommand`, later
than the true successful-wait receipt captured by `dmTiming::Marker::receipt`.
The existing field is a reader/callback-begin-to-SDK prefix despite its name.

Aggregate `dmtimes.receiptSdk` uses the earlier true receipt marker. Consequently,
the previously quoted baseline differences of about +0.724, +0.730, +0.570 and
+0.496 us compare different origins and cannot establish those exact added
main latencies. Their raw values and joins remain valid, but the added-latency
interpretation is withdrawn. Primary-observer-to-SDK values and native SDK
service measurements do not have this particular origin mismatch.

Do not repurpose an existing endpoint wire timestamp silently. Expose/join the
actual receipt marker explicitly (with protocol/version treatment if required),
then compare active main-only and disabled populations with identical origins.
Retain baseline main classifications while the trial is disabled, without any
auxiliary staging or posting, so both populations use the same measurement path.
This also supplies the downstream evidence needed for the target/lead change.

## 2026-11-03: first compensated finite test

The user requests a test aiming at the n*125 us combined-output grid. Rather
than infer a full transport delay or an absolute semaphore-post mapping from
the existing trace, the first test applies the measured **differential**
correction and adds the missing downstream evidence. It remains a finite ideal
zero-command trial; no production/provider scheduling contract changes.

`zeroTest.publicationAdvanceUs` is a startup setting, default zero. In the
experimental preset it is 5 us, chosen from the preceding 4.161-4.905 us mean
relative combined-output errors. Each auxiliary publication target becomes the
already frozen primary-observer slot minus this advance. The long cadence fit,
fractional step placement, genuine-main identity, reader readiness and no-catchup
rules remain intact. No phase is adjusted to the current main excursion or to
an individual noisy sample.

The preset also sets `zeroTest.lateUs=2`. The final admission checkpoint is then
at least 3 us before the old unadvanced grid point. Staging must complete before
the earlier publication target; actual reader clearance is still required.
Configuration rejects a positive advance that is not greater than the guard
grace after nanosecond conversion. This cutoff bounds admission, not the
unmeasured remaining API/polling/selected-post work. It does not guarantee every
downstream output is before its target.

Version-6 zero-test records retain `targetNs` as the publication target and add
`referenceTargetNs`, `publicationDeadlineNs`, a monotonic final-guard-to-update-
return bracket and the coherent raw auxiliary publication identity. Combined
and auxiliary raw writetimes share CLOCK_ISIO; their difference measures the
auxiliary-publication-to-combined-publication leg without mixing absolute
producer and monotonic clocks.

The bounded `ZeroPhase` collector records every acknowledged combined command
before success-event sampling. It requires consecutive combined identities and
consecutive slot/cycle labels for nominal interval populations. It reports all
four adjacent transitions, including final auxiliary to next main; per-step
same-main phase errors; auxiliary transport times; and explicit gap, invalid,
duplicate and excluded populations. Ordinary telemetry formats these fixed-size
summaries through the existing coherent snapshot. `zero_phase` INDI and
`[zero_test] kind=phase` expose count/mean/spread/min/max. New trials reset the
populations, and changes to fewer steps clear unused INDI fields.

Use the dedicated overlay after the ordinary zero-test configuration:

```ini
[zeroTest]
publicationAdvanceUs=5
lateUs=2
```

After the operator rebuilds/restarts dmSyncopation, keeping other settings
unchanged, start with:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 10 5
```

The third argument verifies the configured advance before On. The helper also
requires version 6 and `0 <= lateUs < advance`; it never writes these startup
settings. A later 120 s comparison uses `8000 120 5`. Review all four interval
errors, per-step phase bias/spread/tails, skipped opportunities and controller
admission/retirement/loss together. The previous zero-advance captures remain
the control. Fixed compensation can improve mean alignment; asynchronous
mode-4 polling and main excursions can still cause error or skipped slots.

The test does **not** claim to implement the qualified absolute selected-post
phase `H_k` or full deadline hierarchy above. It leaves true controller-receipt
wire export, baseline main classification, exact selected-post qualification,
external-provider ready deadlines and numerical production acceptance criteria
as follow-up work. A mean phase improvement by itself does not qualify them.

Workstation verification and capture results are recorded in the associated
[finite-trial engineering record](../2026-10/dm-syncopation-zero-aux-rate-test.md).
No agent operates or changes instrument hardware.
