# Coordinated zero-auxiliary trial

The main loop stays at **2 kHz**. Its genuine command is step 0, and
dmSyncopation inserts zero-valued auxiliary updates on the established grid:

| Target total cadence | Main command | Auxiliary targets in each nominal 500 us cycle |
| --- | --- | --- |
| 4 kHz: two steps | One genuine main at step 0 | One zero write at +250 us |
| 6 kHz: three steps | One genuine main at step 0 | Two zero writes at +166.667 and +333.333 us |
| 8 kHz: four steps | One genuine main at step 0 | Three zero writes at +125, +250 and +375 us |

There is no auxiliary post at step 0 and no independent 4/6/8 kHz timer added to
the main loop. The zero contribution preserves the main, flat and other retained
DM contributions. This trial uses an ideal constant-zero publisher: its command
is already available, so there is no external provider to wait for or emulate.
All instrument commands below are operator actions.

The target is the **source `cnt0++` publication that requests dmcomb work**.
Main and auxiliary streams have separate counters; their interleaved
publications should follow the table above on average. The grid is fitted from
coherent main-stream publication timestamps over the established ten-second
baseline, then frozen for each cycle. Callback arrival jitter does not define
the phase, and a late publication does not move the following slots.

The [counter-publication plan](../../../agents/plans/2026-11/dm-syncopation-counter-publication-targets.md)
records this endpoint and its clock mapping. Stream `writetime` is sampled just
before the counter update, so it provides a pre-counter timing proxy rather
than an exact timestamp of the store. dmcomb output and controller wakeup occur
later and remain separate diagnostics. The earlier downstream 5 us advance is
superseded; a nonzero `publicationAdvanceUs` is now rejected.

This finite empirical trial measures actual delivery and possible main impact.
It deliberately does not impose the shadow's conservative prediction of zero
added future main SDK wait. Actual reader clearance, identity/lifecycle and
deadline checks remain active. Requested cadence is not proof that all slots
ran or that main timing was unaffected.

## Configure the trial

Build/install the updated dmSyncopation through the normal operator workflow.
The new records use the existing text-log wire format, so installed logdump can
decode them without rebuilding. From any directory in the MagAOX checkout,
force a clean app rebuild and install before restarting synco:

```bash
cd "$(git rev-parse --show-toplevel)" &&
make -C apps/dmSyncopation clean &&
make -C apps/dmSyncopation &&
make -C apps/dmSyncopation install
```

The installed binary must expose `zero_test.labelVersion=9`,
`zero_test.targetEndpoint=1` and `zero_test.idealPublisher=1`; an older
coordinated-trial binary does not have the current progress/reporting interface.
Only dmSyncopation needs rebuilding.
The 8 kHz option also requires the updated app's four-step support and capture
script; the installed logdump and DM controller do not need rebuilding.
Use the capture helper from the same updated checkout. After the duration and
reader-diagnostic update, repeat the ten-second 8 kHz capture below before the
longer run. An existing configuration with `zeroTest.durationSec=120` needs no
change for that shorter trial.
The startup log's `MagAOX GIT` line identifies the launched binary independently
of the capture directory's checkout hash. Merge
[`dmtweeter_synco_zero_test.conf`](../dmtweeter_synco_zero_test.conf)
into the working instance configuration; **do not replace the whole config**.
The essential sections are:

```ini
[simulation]
enabled=false
readerReleaseReady=false
pipelineBackend=false

[zeroTest]
enabled=true
channel=dm01disp07
durationSec=120
lateUs=5
eventStride=4

[scheduler]
maxSubframes=2
```

The three simulation flags must all be false. Turning the old live scheduler
On alone is not this trial. This explicit startup mode remains disabled until
`enable.toggle=On`; Off cancels it. The app independently bounds arm/warmup at
30 seconds and active duration at the selected `settings.durationSec`, without
the capture script. The cadence estimator can warm while disabled and requires
the established ten-second baseline before real writes begin.

The startup setting `zeroTest.durationSec` establishes the immutable duration
ceiling and initial selected duration, positive and no greater than 120 seconds.
It appears as `zero_test.durationLimitSec`. While Off, select a shorter duration
with `settings.durationSec`; `zero_test.durationSec` reports that selected app
time limit. Values above the startup ceiling, nonpositive/nonfinite values,
changes while enabled and duration writes outside zero-test mode are rejected.
The selection persists between trials until changed or the app restarts; the
capture helper sets and verifies it for every run. A shorter selection does not
reduce the startup ceiling, so a subsequent 120-second run remains available
when the configuration ceiling is 120.

Restart **dmSyncopation only** through the normal operator procedure. Keep the
normal main writer, camera, reconstruction, DM controller and dmcomb running,
with gains/workload/settings unchanged:

- Camera/main cadence 2 kHz, camera synchronization On.
- Synco monitor on `dmtweeter-synco` / CPU41.
- Reader-release observer on FIFO30/spare57, reading `dmtweeter_read_done`.
- BMC SDK worker FIFO55/spare56 and ordered post worker FIFO30/spare31.
- The working three-credit pipeline and matching delivery-clock diagnostics.
- `sysMonRTC.set_latency.toggle=On`.
- dmcomb mode 4 / DELAY / 5 us, using auxiliary-counter change detection. This
  is the current optimized baseline; earlier captures used 10 us.

The selected existing auxiliary, here `dm01disp07`, must already contain zeros
and be exclusively available to this trial. Stop its other provider; **keep the
normal `dm01disp03` main writer running**. The app does not silently erase an
unknown nonzero contribution. The selected channel must not be excluded by
astrogrid or drive a ZPO/reference-update role. Mode 3 waiting only on the primary
does not provide the required auxiliary combinations. The SDK-result stream is
`dmtweeter_sdk_done` by default (`zeroTest.sdkStream` can name another).

Keep other auxiliary providers idle during this comparison. Their counter
updates would introduce additional combined commands and ambiguous acknowledgements.

After restarting, check the new mode's readbacks while enable remains Off:

```bash
timeout -k 2s 8s getINDI dmtweeter-synco.zero_test
```

Expect `labelVersion=9`, `targetEndpoint=1`, `idealPublisher=1`, `enabled=1` and a
valid, fresh progress snapshot. `eventStride=4` is the default successful-event sampling
described below; it does not reduce the command rate or timing populations.
No `scheduler.maxCadenceHz=8000` setting is needed for the 8 kHz trial. This
finite zero mode uses the clock-only grid estimator and its explicit total step
count; it does not use the ordinary scheduler/shadow cadence-rating gate. The
ordinary live/shadow maximum-cadence default is unchanged.

## Run and capture

For the new **counter-target 8 kHz** test, merge
[`dmtweeter_synco_zero_phase_test.conf`](../dmtweeter_synco_zero_phase_test.conf)
after the ordinary zero-test settings in the working instance config. Keep the
rest of that config and all camera/loop/controller/dmcomb/CPU settings as they
are. The changed settings are:

```ini
[zeroTest]
publicationAdvanceUs=0
commitLeadUs=0
lateUs=2
wakeAheadUs=10
```

Restart dmSyncopation, leaving enable Off, then start with:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 10 0 10
```

The last two arguments verify zero downstream advance and a 10 us early wake;
they do not change either startup setting. The optional fifth argument verifies
`commitLeadUs` and defaults to zero. After reviewing the short capture,
use `8000 120 0 10` for the
long run. Only dmSyncopation needs rebuilding and restarting; no logdump or DM
controller change is required.

Each auxiliary counter target is the corresponding frozen main-publication
grid point. `commitLeadUs` places the final dispatch slightly before that point
to allow for guard-to-counter work only. It starts at **zero until that work is
measured**; it must not be inferred from dmcomb transport or complete API-return
cost. The setting is finite from 0 through 10 us. It does not move the counter
target, accumulate per-frame feedback or permit catch-up writes. All
reader-clearance checks remain active. Review source publication error,
jitter/tails, skip fractions and controller admission together before tuning.

The early-wake mode addresses the failed first compensated capture, which
skipped about 10% of auxiliary occurrences while sleeping toward the
publication target and then checking a 2 us allowance. With `wakeAheadUs=10`,
the app instead services diagnostics and performs full validation at the earlier
wake point. Those checks must finish by the dispatch point. It then uses a
bounded final spin, with no sleeps, queue draining, timing joins or logging.
Only current identity/ownership, cancellation, finite-duration and readiness/
fault guards remain active in that window. Unsafe reader observations revoke
clearance atomically; healthy duplicate release notifications do not.

`wakeAheadUs` is startup-only, finite from 0 through 50 us; a positive value must
round to at least one nanosecond. Zero retains the legacy waiting path. The
nominal final-spin window is at most that lead; preemption beyond the admission cutoff
still causes a deadline skip when the thread resumes. This remains
an isolated-core experiment; the new capture must establish its delivery and
phase behavior. The earlier wake does not remove dmcomb's polling jitter.

For a **zero-lead, no-early-wake comparison** with the current source-clock grid,
set `publicationAdvanceUs=0`, `commitLeadUs=0`, `lateUs=5`, `wakeAheadUs=0`,
restart synco, and use the two-argument commands below. This does not reproduce
the older callback-clock phase fit. Omitting the third, fourth and fifth helper
arguments expects zero downstream advance, wake-ahead and commit lead,
respectively. All runs require schema 9 and endpoint 1; mismatched presets are
rejected before enable.

Start with `dmtweeter-synco.enable.toggle=Off`. From the repository root, run the
finite **4 kHz total** trial:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 4000 120
```

After reviewing that result, run the **6 kHz total** trial:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 6000 120
```

For the **8 kHz total** extension, keep the same main loop, configuration and CPU
placement. Start with ten active seconds:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 10
```

If that capture shows clean admission/error counters and no material main-timing
regression, repeat for the full interval:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 8000 120
```

This selects four total steps while Off; it does not add 8 kHz on top of the
2 kHz main. Keep the existing 5 us publication guard, exact readiness/identity
checks and three pipeline credits. No CPU or controller setting changes are
part of this extension.

The second argument is active seconds, an integer from 1 through 120 and no
greater than `zero_test.durationLimitSec`. The script starts
synco/controller/camera telemetry collectors first, saves bounded before
snapshots, requires explicit test mode and Off, and checks camera
2 kHz/synchronization and RTC latency. It changes `settings.steps` to
two/three/four and `settings.durationSec` to the requested duration while
disabled, then verifies the step and selected-duration readbacks before On.
The app enforces that duration after the grid qualifies; a shell timer no longer
determines a successful stop. The helper waits for the app's finished state,
sends Off in cleanup, saves after snapshots and stops only its own collectors.
No sudo is required.

## Read the counter-publication timing evidence

`zero_test` version 9 reports `targetEndpoint=1`, `commitLeadUs`, `lateUs` and
`wakeAheadUs`. `publicationAdvanceUs` must be zero. The event fields distinguish
the desired counter point from the earlier dispatch and validation points:

| Field | Meaning |
| --- | --- |
| `counterTargetRawNs` | Frozen desired source publication in the ImageStreamIO producer clock. |
| `counterTargetNs`, `referenceTargetNs` | That same counter point projected into local monotonic time. |
| `targetNs` | Final dispatch point: `counterTargetNs - commitLeadUs`. |
| `publicationDeadlineNs` | Latest final admission checkpoint: `targetNs + lateUs`. |
| `publicationBeginNs`, `publicationEndNs` | Monotonic final-guard-to-update-return bracket; zero when no update was attempted. |
| `wakeTargetNs` | Earlier full-validation wake point: `targetNs - wakeAheadUs`. Zero for the legacy wait. |
| `wokeNs` | Immediate clock sample after the final timed wait, before its diagnostic/check work; entry sample if no sleep was needed. |
| `validatedNs` | Full validation completion, required no later than `targetNs`; zero if validation was not reached. |
| `auxiliaryInode`, `auxiliaryCounter`, `auxiliaryPublicationSec/Nsec` | Coherent auxiliary publication identity and raw producer timestamp. |
| `sourceInode`, `sourceCounter`, `sourceSec/Nsec` | Main or auxiliary source identity, distinct from the combined command. |
| `clockOffsetNs`, `clockBracketNs`, `clockEpoch` | Frozen producer-minus-monotonic offset, sampling bracket and mapping generation. |

`observedNs` remains update-return time for posted commands. The raw auxiliary
and combined publication timestamps use the same producer clock and can be
subtracted from each other; do not subtract their absolute values from the
monotonic target or API bracket without the qualified clock offset. `zero_clock`
and `kind=clock` records expose that mapping's bounds and coverage. A wide
sampling bracket suppresses current writes while preserving the raw cadence
fit; a clock discontinuity resets the fit. `clockRejected` counts excluded
primary observations. Clock qualification failure still respects the app's
finite active and arm timeouts.
Use the integer text-event and `kind=clock` fields for nanosecond arithmetic;
large absolute nanosecond values in INDI number properties have floating-point
rounding. The small duration/error populations are already computed before
that conversion.

Every rejected clock capture or publication mapping while requested On produces
an unsampled event with
`action=clock_rejected`, `reason=clock_rejected` and `stage=clock_qualification`.
It retains the primary source identity, raw fitted target when available, and
these exact fields before a later callback can overwrite them:

| Field | Meaning |
| --- | --- |
| `clockFailure` | `invalid_capture`, `wide_bracket`, `discontinuity`, `future_publication`, `stale_publication` or `invalid_mapping`. |
| `clockBeforeNs`, `clockProducerNs`, `clockAfterNs` | Monotonic / CLOCK_ISIO / monotonic read bracket. The before/after times can be compared modulo 1 ms without a clock conversion. |
| `clockBracketNs`, `clockBracketLimitNs` | Actual bracket width and the unchanged 1 us acceptance limit. |
| `clockMaxAgeNs` | Applied publication age limit. |
| `clockActive`, `clockExcludedSlots` | Whether this main observation falls inside the started finite interval, and its omitted auxiliary opportunities. Warmup and terminal observations have zero excluded slots. |

The local mapped target is zero in a rejected-clock event; it must not be
treated as a qualified wait target. When `clockActive=1`, this event records an
active whole-cycle exclusion; warmup/terminal rows are diagnostic only. It is
not a posted main command or an additional per-slot skip.

`zero_test` and summary records distinguish the complete counted population:

| Field | Meaning |
| --- | --- |
| `clockRejected` | All rejected observations while requested On, including warmup. |
| `clockExcludedCycles`, `clockExcludedSlots` | Rejected-clock observations and auxiliary slots within the started finite interval. These slots are outside `skipped`. |
| `mainCycles` | `mainObserved + clockExcludedCycles`. |
| `auxiliaryOpportunities` | Accumulated active-cycle slots, including clock-excluded slots; a later disabled step change cannot rewrite this total. |
| `mainRebaselined` | Main associations recovered by observing a coherent stale output followed by exactly one current output. |

For counted active cycles, auxiliary opportunities reconcile as
`posted + skipped + clockExcludedSlots`. These counters do not reconstruct
upstream commands whose callbacks or coherent metadata were never observed.
Qualified callbacks that remain in cadence warmup/reacquisition are also outside
the scheduled-cycle population until the grid is ready.

After a clock exclusion, main association may use a coherent stale combined
output as a local waiting baseline. It waits within the original main-output
deadline for exactly one counter increment and a publication timestamp at or
after the current main. It commits that current identity before auxiliary work.
This avoids using the preceding main's stale output as the next cycle's trusted
baseline. Counter jumps, source changes, timeout, primary changes, cancellation
and expiry still veto the cycle. An already-fresh output with no trustworthy
baseline retains the conservative one-cycle skip; its timestamp alone does not
prove which command produced it. Exact reader clearance remains required.

ImageStreamIO updates writetime before incrementing `cnt0`; a polling observer
can therefore see a changed timestamp while the counter still names the prior
output. Main association keeps waiting on the stable counter in that state.
It neither accepts the new command nor replaces its baseline from the timestamp
alone. If the counter never commits before the deadline, auxiliary commands
remain skipped and the uncommitted metadata is not promoted to a trusted key.

The primary timing evidence is `zero_publication` and `[zero_test]
kind=publication endpoint=cnt0_writetime_proxy`. Its full-rate populations are
independent of successful-event sampling and downstream acknowledgement:

| Population | Definition |
| --- | --- |
| `intervalj` | Adjacent interleaved source publications ending at step `j`, with consecutive cycle/slot labels and stream-local counter progression. Step 0 includes the final auxiliary to the next genuine main. |
| `intervalErrorj` | That interval minus the preceding frozen `period / steps`; nominally 125 us at 8 kHz. |
| `phaseErrorj` | Source publication minus the same-cycle actual main publication, less `j * period / steps`. |
| `publicationErrorj` | Raw source publication minus its frozen raw counter target. |
| `preCounterLeadj` | Auxiliary raw writetime proxy minus the final monotonic guard sample mapped with the qualified offset. |
| `apiCostj` | Complete guard-to-update-return duration, including post-counter work. |

Each population exposes count, mean, standard deviation (`RmsUs`), minimum and
maximum. Missing slots are excluded from nominal adjacent intervals. Source
gaps, label gaps, duplicates, invalid timing and clock exclusions are explicit;
same-source replacement/regression latches a measurement fault until reset.
An out-of-order pair across main and auxiliary streams is counted and excluded
without permanently stopping collection. A main
with no dmcomb association and an auxiliary with no later acknowledgement can
still contribute coherent source-publication evidence. Use `preCounterLead`
with its clock uncertainty to choose any later dispatch lead; `apiCost` is not
that lead. Preemption within the publication API remains observable and cannot
be prevented by an earlier admission check.

For early-wake events, compare `wokeNs - wakeTargetNs` for observed wake error,
`validatedNs - wokeNs` for validation/service cost, and
`publicationBeginNs - targetNs` for final admission error. An interrupted wait
can have a wake sample before its planned point; do not reinterpret that field
as a hardware IRQ timestamp. `late` at stage `validation` means the full checks
missed their target, whereas `final_wait` reports a live veto during the quiet
window and `publication_guard` reports the final deadline check. The
`validationLate` summary counter is a **subset of `late`**, not another skip
reason to add to the disjoint outcome totals. The existing 2 us admission grace
is not used to excuse validation that finishes after the target.

The separate `zero_phase` INDI property and `[zero_test] kind=phase` text records
contain **full-rate** measurements, independent of successful-event sampling.
For each destination step `j` they report count, mean, standard deviation
(`RmsUs`), minimum and maximum:

| Population | Definition |
| --- | --- |
| `intervalj` | Adjacent combined publications, with exact consecutive counters and slot labels. Step 0 is the final auxiliary to the next genuine main. |
| `intervalErrorj` | That interval minus the preceding frozen `period / steps`; nominally 125 us at 8 kHz. |
| `phaseErrorj` | Auxiliary combined publication minus its same-cycle actual main publication, less `j * period / steps`. Desired mean is zero. Main timing excursions remain visible in this population. |
| `transportj` | Combined publication minus the matched raw auxiliary publication. |

Missing steps and unmatched identities are counted in `counterGaps`,
`labelGaps`, `phaseExcluded`, `transportExcluded` and `invalid`; a gap is never
reported as a nominal 125 us pair. Counts reset on each new trial. Missing
populations have count zero and extrema/RMS of -1. These endpoints are combined
publication proxies before the selected controller semaphore post. Controller
receipt jitter and SDK service remain separate evidence in dmtweeter telemetry.

Before requesting On, the helper requires reporting version at least 9 and endpoint 1,
`snapshotValid=1`, positive `snapshotSequence`, and reported `snapshotAgeSec`
from zero through two seconds. Polls request an explicit list of fully qualified
`zero_test` elements. They return when all requested fields arrive, avoiding the
whole-property wildcard query's normal two-second wait. During the trial the
helper repeats freshness checks and uses its own local five-second watchdog
for sequence progress. Sequence regression, a stalled sequence, invalid/stale
progress or new event queue loss stops the owned trial and marks the capture
incomplete. The local watchdog matters because a cached INDI response can
repeat an old age value.

The configured `zero_test.trial` identifies the preceding trial. Old active or
finished readbacks from that identity do not count as new progress. The helper
requires exactly the next trial identity and the requested selected duration.
Success requires that new trial to finish with positive main observations,
elapsed time at least the requested duration and requested/active/warming all
Off. A brief trial can finish between polls; observing an intermediate active
readback is not required. Unexpected trial identities, a warmup timeout or a
premature/inconsistent finish produce a nonzero exit status.

The helper allows five seconds of reporting grace after the selected active
duration. Grace permits delayed observation of an app stop; it does not extend
the app's active time limit. An active readback reaching the selected duration
is not itself success. Failure to report completion within the bound and grace
causes an Off request and an incomplete capture.

A pre-existing On trial is refused and left alone. If this script attempted On,
interruption or a control/read failure causes a bounded Off request in cleanup.
The app's selected duration is the independent active bound if the capture
shell exits unexpectedly. The script does not change main-loop, camera, dmcomb,
controller, CPU or persistent configuration settings.

Final quality checks also require no increase in event-queue drops or logged-event
enqueue failures; a missing quality readback is incomplete. Read `status.txt`
alongside the captures rather than inferring success from the directory name.

Results under `$HOME/tmp/zero-aux-<total-rate>Hz-<UTC>` include simultaneous
`dmtweeter-synco.txt`, `dmtweeter.txt`, `camwfs.txt`, before/configured/after INDI,
revision/settings, status history and command exit statuses. The result directory
is printed at completion. Preserve the first attempt before retuning.

Equivalent manual controls, after setup and with telemetry already recording:

```bash
timeout -k 2s 5s setINDI -n dmtweeter-synco.settings.steps=2
timeout -k 2s 5s setINDI -n dmtweeter-synco.settings.durationSec=120
timeout -k 2s 8s getINDI dmtweeter-synco.settings.durationSec dmtweeter-synco.zero_test.durationSec
timeout -k 2s 5s setINDI -s dmtweeter-synco.enable.toggle=On
timeout -k 2s 8s getINDI dmtweeter-synco.zero_test.trial dmtweeter-synco.zero_test.active \
    dmtweeter-synco.zero_test.finished dmtweeter-synco.zero_test.elapsedSec
```

Verify the selected duration readback before requesting On. Use three steps
for 6 kHz or four for 8 kHz, and select the desired duration every time while
Off. Cancel with:

```bash
timeout -k 2s 5s setINDI -s dmtweeter-synco.enable.toggle=Off
```

## Interpret the measurement

```text
genuine 2 kHz main -------------------------------+
                                                 +-> dmcomb -> DM controller
main-labelled grid -> zero auxiliary slot(s) -----+           -> SDK -> ordered post
                         ^
                exact reader-release evidence
```

For each frozen auxiliary target `T`, the app waits for exact reader clearance
and copies the complete zero frame **before** the target. It clears the shared
write flag after this passive staging; counters, writetime and semaphores stay
unchanged. If staging finishes at or before `T`, the app waits for `T`, rechecks
the primary/combined identities, auxiliary ownership and reader clearance, and
issues one normal active publication. It never writes primary pixels or posts
the primary semaphore.

`T` is the command-ready deadline for this ideal test. Missing reader clearance
by the deadline skips the slot. A copy that starts or finishes after `T` is
`preparationLate`, separate from scheduler publication lateness. A completed
command does not get prepared again at the publication wake. The existing
`lateUs=5` remains the publication-admission grace after `T`.

Expired targets are skipped, never replayed in a catch-up burst. Genuine main
excursions can cancel conflicting slots without shifting committed targets to
chase those excursions.

Read `zero_test` and synco telemetry as actual outcomes, separately from shadow
`wouldRun`. Legacy `timing.active`/`delivered` describe the old scheduler;
the trial uses `zero_test` and `[zero_test]` text telemetry. Compare
posted/acknowledged counts, independent skips and lateness,
reader-invalid/drop coverage, exact identity matches, resource faults and
telemetry loss. Two, three and four steps supply one, two and three auxiliary
opportunities per main cycle. Actual combined/controller cadence should approach
4/6/8 kHz only when those opportunities are delivered.

At 8 kHz, nominal spacing is 125 us. The preceding 6 kHz capture measured SDK
service through worker readiness up to 111.841 us: **13.159 us nominal
headroom**, not a hard bound. Actual native-entry spacing and credit use depend
on delivery/preparation and genuine-main excursions as well. Compare main
primary/controller-to-SDK prefixes and exact-match coverage with the same
capture's disabled baseline, alongside SDK service, admission/retirement,
overload/rejection/coalescing, failures and queue loss. Retain upstream
compilation or workload changes when comparing historical 6/8 kHz runs.
This remains a finite ideal-zero empirical test, not a change to ordinary
production scheduling or a qualification of nonzero provider commands.

### Read actual slot outcomes

`skipped` means that the app deliberately **did not issue an active auxiliary
publication** for that slot. It may already have copied zeros before the final
guard rejected publication. A skipped slot does not imply that the genuine main
command arrived early.

Each scheduled auxiliary slot contributes one counted outcome. Healthy event
rows can be intentionally sampled, as described below; skipped and failed rows
are always offered to the diagnostic queue. Two/three/four total steps schedule
one/two/three auxiliary slots per main cycle. When the main combination cannot
be matched, each dependent auxiliary slot gets its own skip outcome.

| Counter | Meaning |
| --- | --- |
| `staged` | Completed zero copies, including a copy that overran its ready deadline or was later cancelled; this overlaps publication/skip populations. |
| `acknowledged` | An auxiliary publication was issued and its expected combined update was observed. |
| `preparationLate` | A copy started or finished after the immutable command-ready target `T`; no active publication is issued. |
| `late` | An admission checkpoint exceeded `T + zeroTest.lateUs`, or full early validation finished after `T`. The event stage distinguishes `validation`, `deadline` and `publication_guard`. |
| `readerMissing` | Exact qualifying reader clearance was unavailable by the command-ready deadline or was revoked before publication. |
| `primaryChanged` | Before publication, the primary counter actually changed from this cycle's main identity. |
| `primaryWriting` | Before publication, the primary write flag was set; this does not assert a changed counter. |
| `resourceUnavailable` | A slot's required stream/resource checks failed. |
| `invalidTarget` | The computed slot target was invalid. |
| `waitError` | The absolute wait failed. |
| `combinedUnclassified` | The main or current combined identity could not be classified before an auxiliary publication. |
| `cancelled` | Disable or shutdown cancelled the pending slot before publication. |
| `durationExpired` | The finite active interval ended before publication. |
| `unacknowledged` | An auxiliary publication was attempted, but the expected combined update was not confirmed. This is a posted attempt, not a skip. |

The skipped population is the sum of its disjoint reason counters. The posted
population separates into acknowledged and unacknowledged attempts. Reader
invalid/drop diagnostics and resource faults outside a scheduled slot describe
coverage/lifecycle conditions; they are not additional skip reasons to add again.
A guard failure during acknowledgement belongs to the posted attempt's named
reason and does not increment a pre-publication skip counter.

New summaries/events carry `labelVersion=9`; summaries/INDI also identify
`idealPublisher=1`. Event text includes named `action`,
`reason` and `stage` fields alongside the retained numeric `outcome`. The stage
distinguishes preparation, reader/combined validation, publication deadline/
`publication_guard`, and acknowledgement work. The former `post_copy_guard`
path is not used by this ideal-publisher mode. `stagedNs` is the actual copy
completion time, including an overrun; zero means that no copy completed.
Compare `targetNs - stagedNs` only when `stagedNs` is positive: a nonnegative
value means the command was staged by its ready deadline. `staged` is not another
skip reason to add to `skipped`.

A late guard event retains the exact rejecting clock sample. A
timestamp/residual after a successful publication is a completion stamp;
it can exceed the configured `lateUs` even though the admission checkpoint
occurred within that limit (5 us for the no-early-wake comparison, 2 us for the
counter-target preset).

For example, `action=skipped reason=preparation_late stage=preparation` records a
missed ready deadline. `action=skipped reason=late stage=publication_guard`
records publication-admission lateness after preparation.
`action=unacknowledged reason=ack_timeout` is a
posted attempt whose expected combination was not confirmed in time. Other
posted failures can name `ack_ambiguous` or `publication_failed`; a `wait_error`
includes the wait's numeric `waitResult`.

Earlier captures used compound labels: their `primaryChanged` and `unclassified`
counts cannot be retrospectively split into the new causes. Use the new named
fields and reason counters for subsequent attribution.

### Progress snapshots and reporting coverage

The ordinary reporting thread consumes a coherent latest snapshot without
waiting for the timed cycle's lock. Intermediate state snapshots can coalesce;
their sequences need not be contiguous. Command execution, outcome counters and
main timing joins remain full-rate.

| `zero_test` field | Interpretation |
| --- | --- |
| `trial` | Monotonic trial identity; settings changes preserve it and a new enable request starts the next trial. |
| `requested` | Captured enable request; it is distinct from actual `warming`, `active` and `finished` state. |
| `durationSec` | Selected app-enforced active time limit, writable through `settings.durationSec` only while Off. |
| `durationLimitSec` | Immutable startup ceiling from `zeroTest.durationSec`; a shorter runtime selection cannot raise it. |
| `snapshotSequence` | Monotonic snapshot publication identity within the app; it must keep advancing. |
| `snapshotTimeNs` | Capture time in the RTC's local monotonic clock. |
| `snapshotAgeSec` | Age computed by ordinary reporting on the RTC. |
| `snapshotValid` | A coherent positive capture time not later than the reporting clock; freshness still requires age and sequence checks. |
| `eventsQueued` | Diagnostic event queue depth captured with that snapshot. |
| `eventStride` | Startup selection stride for healthy event rows. |
| `eventsSampledOut` | Healthy rows deliberately omitted in this trial. |
| `eventsDropped` | Diagnostic queue overflow; an unintentional coverage loss. |

`elapsedSec` is evaluated at the snapshot's capture/stopped time. Rereading a
cached snapshot does not make elapsed time advance. Keep producer-clock image
timestamps separate from `snapshotTimeNs`; the latter is a local monotonic time.

`zeroTest.eventStride` defaults to **4**, with allowed values 1 through 1024.
Successful main and auxiliary rows use the same cycle selector, retaining every
fourth cycle's healthy rows by default. Lifecycle, skipped, unacknowledged and
other exceptional rows are never intentionally sampled. Actual commands, all
outcome counters and timing joins are unchanged by this selection.
Set `eventStride=1` in the startup configuration and restart synco when a full
healthy event stream is needed; this increases reporting traffic. Keep the
default four for the next short 8 kHz reporting test.

With sampling enabled, missing healthy event rows are expected and counted by
`eventsSampledOut`; they are not missing commands. Full per-command contiguous
identity reconstruction cannot use that sampled stream alone. Use the exact
summary populations and controller delivery counts for total throughput, and
retain any `eventsDropped` or logger-failure coverage losses separately.

### Ordinary-loop diagnostics

`logic_timing` and `[synco_logic]` text records measure reporting/housekeeping
service, separately from the main, auxiliary and SDK command timings:

- `sequence`, `loopGapUs` and `loopGapMaxUs` describe start-to-start ordinary-loop
  progress, including time spent servicing diagnostics and the normal loop wait.
- `eventsQueued` and `eventsQueueHighWater` show current and greatest observed
  diagnostic queue depth.
- `eventsDrained` and `eventsDrainedLast` count rows popped and offered to the
  logger, not proof of successful persistence.
- `eventsLogFailed` counts event rows and phase evidence rejected by telemetry enqueue;
  `eventsDropped` counts queue overflow before draining. Both are distinct from
  intentional `eventsSampledOut`.
- `primaryUs`, `combinerUs`, `cameraUs`, `pipelineUs`, `readersUs`, `legacyUs`,
  `statusUs`, `eventsUs` and `telemetryUs`, with corresponding `MaxUs` fields,
  measure each group's last-loop total and process maximum. They include each
  group's INDI work; event service includes formatting/enqueue work.
- `reportPublicationUs` and `reportPublicationMaxUs` measure the last completed
  profile report's own formatting, INDI publication and telemetry enqueue, and
  its process maximum. They describe the previous report because a report
  cannot include the time needed to finish publishing itself.

Use these fields to locate slow ordinary reporting stages and queue pressure.
They do not measure physical DM latency or change any command-ready deadline,
publication guard or reader-clearance rule.

In zero-test mode the legacy `reader_timing` and `reader_release_timing` join
collectors are disabled. Their quality properties expose `enabled=0`, with no
qualified legacy timing population. Their primary/combined inputs are not
populated in this mode, so retaining completion records would perform diagnostic
work without producing joins.
The independent `dmtweeter_read_done` observer remains active: its exact release
records, epoch/loss checks and readiness queue still protect every timed write.
Zero-test SDK/main-impact joins and their coverage counters also remain active.
The `readersUs` profile group can therefore include monitor upkeep even though
the unused legacy joins are disabled.

The general reader collector used by other modes now keeps pending identities
in counter order, allowing bounded oldest-record eviction without repeated
full-map scans. It still drains captured source batches and joins available
peers before expiring old evidence, preserving joins whose matching observation
was already queued.

A production provider would need a coherent command available by its advertised
ready deadline, with a missing command causing a skip. This finite test supplies
that command as an invariant zero frame. It does not implement an external
provider protocol, producer notification or full publisher emulator.

Compare capture-local BMC overload/rejection/coalescing, superseded/cancelled
jobs, SDK failures/timeouts, queue losses, publication/receipt/send intervals and
post-service tails. Clean timed-event statistics alone cannot exclude admission
losses before timing handoff.

Exact main-impact joins provide primary-observer-to-SDK and reader-begin-to-SDK
prefixes and SDK service where identities are classified. The compatibility
field `controllerToSdk` starts at `read_done.startedNs`, captured after input
identity is read; it is later than the true semaphore-wake receipt used by
aggregate `dmtimes.receiptSdk`. Do not subtract that aggregate baseline from the
exact field and describe the result as added main latency. A matched true
receipt is needed for that comparison. These prefixes also do not isolate SDK
queue wait. Aggregate `sdkWait` mixes main and auxiliary jobs. Report matched/unmatched/unclassified
and dropped coverage with timing claims, using simultaneous camera and primary
cadence to check changed excursions. Target FPS alone does not prove an
undisturbed main loop.

`mainReaderMatched`, `mainTimingInvalid`, `mainTimingEvicted` and
`mainTimingSourceResets` expose coverage of the classified main/SDK joins. A zero-count
population is unavailable even when its mean field is zero.

In mode 4, dmcomb may have a blank named-primary process trace. Then exact
main-impact prefixes are unavailable and `mainImpactUnclassified` grows. The
coordinated slot attempts still proceed using exact combined-counter and
reader-release checks. Retain that classification gap; do not substitute a
nearby SDK command or interpret aggregate SDK wait as main-only timing.

Leave the trial Off after each capture. The auxiliary remains zero and the normal
main loop remains running. No production timing bound or shadow admission
assumption is automatically changed by this experiment.
