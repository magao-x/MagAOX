# Coordinated zero-auxiliary trial

The main loop stays at **2 kHz**. Its genuine command is step 0, and
dmSyncopation inserts zero-valued auxiliary updates on the established grid:

| Target total cadence | Main command | Auxiliary targets in each nominal 500 us cycle |
| --- | --- | --- |
| 4 kHz: two steps | One genuine main at step 0 | One zero write at +250 us |
| 6 kHz: three steps | One genuine main at step 0 | Two zero writes at +166.667 and +333.333 us |

There is no auxiliary post at step 0 and no independent 4/6 kHz timer added to
the main loop. The zero contribution preserves the main, flat and other retained
DM contributions. All instrument commands below are operator actions.

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

The installed binary must contain the coordinated-trial implementation
(`45c85771` or later); `2d8c104a` does not recognize `zeroTest` configuration.
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

[scheduler]
maxSubframes=2
```

The three simulation flags must all be false. Turning the old live scheduler
On alone is not this trial. This explicit startup mode remains disabled until
`enable.toggle=On`; Off cancels it. The app independently bounds arm/warmup at
30 seconds and active duration at at most 120 seconds, without the capture script.
The cadence estimator can warm while disabled and requires the established
ten-second baseline before real writes begin.

Restart **dmSyncopation only** through the normal operator procedure. Keep the
normal main writer, camera, reconstruction, DM controller and dmcomb running,
with gains/workload/settings unchanged:

- Camera/main cadence 2 kHz, camera synchronization On.
- Synco monitor on `dmtweeter-synco` / CPU41.
- Reader-release observer on FIFO30/spare57, reading `dmtweeter_read_done`.
- BMC SDK worker FIFO55/spare56 and ordered post worker FIFO30/spare31.
- The working three-credit pipeline and matching delivery-clock diagnostics.
- `sysMonRTC.set_latency.toggle=On`.
- dmcomb mode 4 / DELAY / 10 us, using auxiliary-counter change detection.

The selected existing auxiliary, here `dm01disp07`, must already contain zeros
and be exclusively available to this trial. Stop its other provider; **keep the
normal `dm01disp03` main writer running**. The app does not silently erase an
unknown nonzero contribution. The selected channel must not be excluded by
astrogrid or drive a ZPO/reference-update role. Mode 3 waiting only on the primary
does not provide the required auxiliary combinations. The SDK-result stream is
`dmtweeter_sdk_done` by default (`zeroTest.sdkStream` can name another).

Keep other auxiliary providers idle during this comparison. Their counter
updates would introduce additional combined commands and ambiguous acknowledgements.

## Run and capture

Start with `dmtweeter-synco.enable.toggle=Off`. From the repository root, run the
finite **4 kHz total** trial:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 4000 120
```

After reviewing that result, run the **6 kHz total** trial:

```bash
bash apps/dmSyncopation/scripts/capture-zero-aux-test.sh 6000 120
```

The second argument is active seconds, an integer from 1 through 120. The script
starts synco/controller/camera telemetry collectors first, saves bounded before
snapshots, requires explicit test mode and Off, checks camera 2 kHz/synchronization
and RTC latency, changes `settings.steps` to two/three while disabled and verifies
the readback, then requests On. It allows up to 30 seconds to qualify the grid,
waits for the trial or requested shorter duration, requests Off, saves after
snapshots, and stops only its own collectors. No sudo is required.

A prior finished readback is ignored until new trial progress appears. Completion
requires an observed active phase and the requested active duration; a warmup
timeout or premature finish is recorded as incomplete with a nonzero exit status.

A pre-existing On trial is refused and left alone. If this script attempted On,
interruption or a control/read failure causes a bounded Off request in cleanup.
The app's configured duration is the independent upper bound if the capture shell
exits unexpectedly. The script does not change main-loop, camera, dmcomb,
controller, CPU or persistent configuration settings.

Results under `$HOME/tmp/zero-aux-<total-rate>Hz-<UTC>` include simultaneous
`dmtweeter-synco.txt`, `dmtweeter.txt`, `camwfs.txt`, before/configured/after INDI,
revision/settings, status history and command exit statuses. The result directory
is printed at completion. Preserve the first attempt before retuning.

Equivalent manual controls, after setup and with telemetry already recording:

```bash
timeout -k 2s 5s setINDI -n dmtweeter-synco.settings.steps=2
timeout -k 2s 5s setINDI -s dmtweeter-synco.enable.toggle=On
timeout -k 2s 8s getINDI dmtweeter-synco.zero_test
```

Use three steps for 6 kHz, changing settings only while Off. Cancel with:

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

Each actual slot copies a complete zero frame and performs a normal active
publication. The trial never writes primary pixels or posts its semaphore.
Expired targets are skipped, never replayed in a catch-up burst. Genuine main
excursions can cancel conflicting slots without shifting committed targets to
chase those excursions.

Read `zero_test` and synco telemetry as actual outcomes, separately from shadow
`wouldRun`. Legacy `timing.active`/`delivered` describe the old scheduler;
the trial uses `zero_test` and `[zero_test]` text telemetry. Compare
posted/acknowledged counts, independent skips and lateness,
reader-invalid/drop coverage, exact identity matches, resource faults and
telemetry loss. Two steps supply one auxiliary opportunity per main cycle;
three supply two. Actual combined/controller cadence should approach 4/6 kHz
only when those opportunities are delivered.

### Read actual slot outcomes

`skipped` means that the app deliberately **did not issue an active auxiliary
publication** for that slot. It may already have copied zeros before the final
guard rejected publication. A skipped slot does not imply that the genuine main
command arrived early.

Each scheduled auxiliary slot has one outcome event. Two total steps schedule
one auxiliary slot per main cycle; three schedule two. When the main combination
cannot be matched, each dependent auxiliary slot gets its own skip event.

| Counter | Meaning |
| --- | --- |
| `acknowledged` | An auxiliary publication was issued and its expected combined update was observed. |
| `late` | A pre-publication deadline check exceeded `zeroTest.lateUs`, including the final checks before/after the zero copy. |
| `readerMissing` | The current combined command lacked exact qualifying reader-release evidence. |
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

New summaries/events carry `labelVersion=2`. Event text includes named `action`,
`reason` and `stage` fields alongside the retained numeric `outcome`. The stage
distinguishes an initial deadline check, reader/combined validation,
`final_guard`, `post_copy_guard`, and acknowledgement
work. A late guard event retains the exact rejecting clock sample. A
timestamp/residual after a successful publication is a completion stamp;
it can exceed 5 us even though the admission check occurred within the unchanged
5 us limit.

For example, `action=skipped reason=late stage=final_guard` is an intentional
pre-publication deadline skip. `action=unacknowledged reason=ack_timeout` is a
posted attempt whose expected combination was not confirmed in time. Other
posted failures can name `ack_ambiguous` or `publication_failed`; a `wait_error`
includes the wait's numeric `waitResult`.

Earlier captures used compound labels: their `primaryChanged` and `unclassified`
counts cannot be retrospectively split into the new causes. Use the new named
fields and reason counters for subsequent attribution.

Compare capture-local BMC overload/rejection/coalescing, superseded/cancelled
jobs, SDK failures/timeouts, queue losses, publication/receipt/send intervals and
post-service tails. Clean timed-event statistics alone cannot exclude admission
losses before timing handoff.

Exact main-impact joins provide primary-to-SDK and controller-to-SDK prefixes
and SDK service where identities are classified. Those prefixes include delivery
and preparation; they are not an isolated SDK-wait measurement. Aggregate
`sdkWait` mixes main and auxiliary jobs. Report matched/unmatched/unclassified
and dropped coverage with timing claims, using simultaneous camera and primary
cadence to check changed excursions. Target FPS alone does not prove an
undisturbed main loop.

`mainReaderMatched`, `mainTimingInvalid`, `mainTimingEvicted` and
`mainTimingSourceResets` expose coverage of the receipt/SDK joins. A zero-count
population is unavailable even when its mean field is zero.

In mode 4, dmcomb may have a blank named-primary process trace. Then exact
main-impact prefixes are unavailable and `mainImpactUnclassified` grows. The
coordinated slot attempts still proceed using exact combined-counter and
reader-release checks. Retain that classification gap; do not substitute a
nearby SDK command or interpret aggregate SDK wait as main-only timing.

Leave the trial Off after each capture. The auxiliary remains zero and the normal
main loop remains running. No production timing bound or shadow admission
assumption is automatically changed by this experiment.
