# Fixed production copy budget

## Request and evidence

On November 6, 2026 the operator requested a fixed, tuned parameter instead of
learning the production copy budget from isolated timing excursions. The
[production investigation](dm-syncopation-production-providers.md)
records an 87.685 us copy maximum that raised the reservation to 88.685 us.
With the existing 70/20 us preparation leads and 10 us auxiliary wake lead,
that reservation cannot fit in either the 50 us passive or 60 us auxiliary
window. Two snapshots show 73,425 preparation-late skips and zero auxiliary
posts over 24,475 observed main cycles while enable remains On.

## Implementation decisions

- Reuse `scheduler.copyBudgetUs` as a fixed whole-batch production reservation.
  Do not add a second copy-budget setting or an automatic tuning phase.
- Keep measured copy and snapshot maxima as lifetime diagnostics. Neither
  production measurement changes its configured budget. Production admission
  does not use the snapshot budget; elapsed snapshot work already reduces the
  time remaining before the copy cutoff.
- Preserve pre-copy fit, primary identity, destination ownership, exact reader
  readiness and actual completion checks. A late copy still suppresses the
  current auxiliary trigger. It does not raise the budget for later slots.
- Preserve partial/passive-write semantics: a completed destination write
  cannot be undone when a later check rejects the batch. Fixed admission is
  a tuned scheduling reservation, not a guarantee against descheduling during
  a copy.
- Leave the older non-production scheduler's adaptive budgets unchanged.
- Preserve the compiled 10 us default. Set 30 us explicitly in the production
  example as an initial tuning point based on earlier successful four-channel
  runs with roughly 27 us copy maxima. This is not instrument qualification or
  a deployed configuration change.
- Keep this work separate from existing uncommitted INDI XML and investigation
  changes in the shared checkout.

## Verification

Use the existing private-stream production fixture and deterministic clocks to
verify a copy excursion updates diagnostics, fails the current hard cutoff,
and leaves later admissible work able to proceed. Verify a configured budget
that does not fit still prevents destination writes, and snapshot excursions
do not alter configured budgets. Run the complete scheduler suite and build
the application without starting it. All execution stays on workstation
`mx11`; instrument installation, restarts and qualification remain operator
actions.

Validation completed on `mx11`:

- Focused fixed-budget regressions pass: 3 cases, 61,827 assertions.
- The complete scheduler suite passes: 157 cases, 6,696,184 assertions.
- The injected 90 us copy overrun skips its first auxiliary trigger, then
  permits the remaining two auxiliary slots and the passive copy. All three
  auxiliary slots and the passive copy succeed in the next main cycle. The
  copy maximum retains the excursion while the budget remains 30 us.
- The injected 90 us snapshot excursion leaves its configured budget at 10 us;
  later preparation succeeds. Configuration tests also verify explicit budget
  parsing for one-step and four-step production modes.
- A 30 us copy reservation with only 20 us remaining rejects the batch before
  changing destination pixels, metadata, counters or retained contributions.
- A normal application build succeeds. The application was not started or
  installed, and no instrument state or deployed configuration was changed.
- Both touched C++ headers pass clang-format verification; diff checks pass.

Logs are `/tmp/synco-fixed-budget-test-build.log`,
`/tmp/synco-fixed-budget-regression.log`, `/tmp/synco-fixed-budget-suite.log`
and `/tmp/synco-fixed-budget-app-build.log`.

Operator follow-up: deploy the rebuilt app with an explicitly chosen fixed
`scheduler.copyBudgetUs`, starting from the documented 30 us example, and
qualify the actual rate, skip fraction and per-slot timing. The measured maxima
may exceed the configured reservation; they should no longer change it. This
change does not determine what caused the original instrument excursion.

## First operator run with a fixed 30 us budget

The operator confirms `copyBudgetUs=30` and reports restored near-8 kHz cadence
with elevated jitter. The November 6 attachment
`d010f330-ad0c-4243-bb2f-2413c991b8b6/Pasted text.txt` contains two snapshots of
trial 1. Their coherent zero-test summaries are valid, about 22/26 ms old, and
separated by 15.217494001 s of trial time. Counter differences are:

| Evidence | Increase |
| --- | ---: |
| Main cycles, including clock exclusions | 30,435 |
| Auxiliary opportunities | 91,305 |
| Auxiliary posted and acknowledged | 90,736 each |
| Clock-excluded cycles / slots | 189 / 567 |
| Ordinary auxiliary skips | 2 |
| Preparation late | 1 |
| Validation late, also included in late | 1 |

The count identity holds: 90,736 + 567 + 2 = 91,305. Clock qualification accounts
for 99.65% of the 569 omitted auxiliary opportunities. Main cycles plus posted
auxiliaries imply approximately 7,962.61 Hz over this summary interval, consistent
with the separately sampled controller receipt rates of 7,963.97 and 7,954.96 Hz.
The controller rolling windows and counter-update times are different; their
counts must not be equated directly with this synco interval. SDK worker
coalescing, failure, rejection and timeout totals and pipeline overload,
cancellation, rejection and processing-failure totals do not increase.

Incremental timing populations were reconstructed from the cumulative counts,
means and centered second moments in the two snapshots, all within trial 1:

| Publication relative to its frozen target | Mean error, us | Standard deviation, us |
| --- | ---: | ---: |
| Genuine main | -3.3928 | 17.0525 |
| Auxiliary 1 | 0.2165 | 0.0277 |
| Auxiliary 2 | 0.2162 | 0.0274 |
| Auxiliary 3 | 0.2425 | 0.0987 |

The raw intervals ending at steps 0 and 1 vary by 13.038 and 17.054 us; intervals
ending at steps 2 and 3 vary by only 0.0196 and 0.0970 us. The corresponding
combined-publication deviations are 14.322, 17.955, 3.326 and 3.490 us. Auxiliary
publication-to-combination transport averages 10.47--10.85 us with 1.98--2.16 us
standard deviation. Controller receipt jitter is 14.875 and 15.823 us in its two
rolling windows. The largest variation is already present around the genuine
main relative to the frozen grid; successfully published auxiliary slots meet
their individual targets closely.

These main errors do not distinguish actual upstream publication variation
from movement of the fitted grid. Adjacent slot populations exclude missing
labels, whereas controller receipt intervals include the resulting gaps.
Cumulative minima/maxima cannot recover extrema for just this comparison.
`RmsUs` in these properties is centered standard deviation, not RMS error.

The ordinary `timing` property is identical in both queries, including its
older delivered count, despite advancing zero-test counters. Treat that property
as stale for this comparison. The copy budget shown is 30 us, but its displayed
copy/snapshot maxima cannot establish when any excursion occurred. The fresh
zero-test data establishes that preparation suppression is now rare.

The attachment omits `zero_clock`, so the 189 clock exclusions cannot yet be
classified. Current code rejects clock-read brackets above 1 us, malformed or
discontinuous clock pairs, future/older-than-4-ms primary publications, or
unusable mappings. `zero_publication.maxBracketUs` includes accepted evidence
only and cannot rule out wide rejected captures. Request two current
`zero_clock`, `zero_test`, `primary_timing`, `primary_timing_quality` and camera
`fg_timing` readbacks before choosing a change to clock qualification or phase
tracking. Keep the fixed copy reservation unchanged for that comparison.

Local calculation artifacts are `/tmp/synco-fixed-budget-jitter-readback.json`
and `/tmp/synco-fixed-budget-jitter-increments.json`. Only this engineering
record was updated during the analysis; no further runtime or configuration
changes or instrument actions were performed.

## Clock classification and upstream jitter follow-up

The operator's subsequent inline readbacks cover trial 4, independently of
trial 1 above. The summaries advance by 21.296523352 s and 42,593 main cycles.
Clock attempts increase by 42,593, accepted captures by 42,409 and wide brackets
by 184. Clock-excluded cycles also increase by exactly 184, excluding 552
auxiliary slots. Other clock failure counters remain zero and the epoch stays
at 4. The current accepted bracket is 0.05 us; this does not give the widths of
the rejected captures. The acceptance limit in the source is 1 us.

Auxiliary opportunities increase by 127,779, posted/acknowledged by 127,223 and
ordinary skips by four, all validation-late. Preparation-late remains unchanged.
The identity 127,223 + 552 + 4 = 127,779 holds. Wide clock captures account for
99.28% of omitted auxiliary slots in this interval. Main cycles plus auxiliary
posts imply about 7,973.88 Hz; this is a count-derived rate, not a matched
controller measurement.

Fresh primary-publication interval summaries show 18.469 and 16.899 us standard
deviation about a 500 us mean. Their respective ranges are 357.494--649.176 and
366.291--657.221 us. Primary receipt interval deviations are similarly elevated;
the primary observer reports no increase in counter gaps, duplicates, rejected,
invalid or dropped events. Camera publication interval deviations are only
0.979 and 1.028 us, with acquisition deviations of 0.287 and 0.322 us. These
measurements establish actual publication-interval variation downstream of
the camera, independently of the fitted synco grid. They do not isolate a
particular upstream stage or establish whether synco contributes through load.

The operator reports both GPU MVM stages running significantly slower than
normal. Their displayed host duration must be distinguished from GPU execution
time: local MVM source includes transfers and completion waiting in its MVM
duration. Its optional CUDA-event measurement also includes transfers when
CUDA graphs are enabled. Current stage timings and live placement are needed
to distinguish device execution, host scheduling and transfer/wait variation.
The operator's observation directs the next investigation upstream; no further
scheduler change has been made. Wide-bracket exclusions remain a separate
confirmed source of omitted auxiliary slots.

## Enable-dependent MVM slowdown

The operator subsequently identifies `dmtweeter-synco.enable` as the trigger
for the MVM slowdown. This supersedes treating the upstream variation as an
independent condition: upstream publication is affected by enabled synco
activity, although the interfering resource is not yet identified.

Enabling does not change synco thread placement or create its observers. It
adds production snapshots/copies, provider activity, three auxiliary commands
per main and the associated downstream work. The final busy-wait window is
at most 10 us per auxiliary, approximately 6% of one CPU at a 2 kHz main rate;
other work and wakeups add to that. No direct CUDA work appears in synco.

A specific downstream GPU consumer is `dmtweeter-recon`. BMC publishes the
staged delta for each successfully completed command in
`dm::finishPipelineCommand`. `dmRecon` monitors `dm01disp_delta` and performs
upload, GEMV and download before checking `writeDMf`. Consequently, setting
`writeDMf=Off` does not stop reconstruction. The local configuration selects
GPU1, shared with the modes-to-DM MVM. Synco can increase its input publication
rate from 2 kHz toward 8 kHz; actual processing throughput and the deployed
reconstructor's presence/GPU selection remain to be confirmed by the operator.

Prior operator topology evidence places CPUs40--47 on separate physical cores
in one L3 domain. The current map places controller40, synco41, SDK42, post44
and both MVM hosts45 there. The RTC IRQ script also routes NVIDIA interrupts
to the MVM CPU45. This supplies possible shared-resource paths to both MVMs,
but neither configured affinity nor this code audit proves the live mechanism.
Check the active GPU consumer before changing placement or scheduler timing.
Only this engineering record changed; no instrument actions were performed.

The operator confirms `dmtweeter-recon` is running on GPU1 during this test.
The command callback has no camera-FPS throttle: `m_fps` supplies output FPS
metadata, while each successful input semaphore wait invokes reconstruction
when ready. The monitor does not deduplicate counters or drain accumulated
notifications in its ordinary loop, so a backlog can also reconstruct the
latest buffer repeatedly. Actual GPU throughput remains unmeasured, but the
active competing GPU1 consumer is now confirmed rather than inferred from
workstation configuration.

The requested operator comparison keeps synco enabled at four steps and
isolates the reconstructor by stopping it entirely during a bench test with
its feedback out of the loop. Compare both MVM times and controller fgtimes;
merely switching `writeDMf` Off cannot isolate the computation. The operator
reports **no improvement** with recon stopped, and confirms there is currently
nothing else running on the GPUs. This does not support reconstruction load as
the explanation for the present slowdown. Drop competing GPU workloads as the
working diagnosis; no reconstruction rate or feedback changes were made.

The next requested comparison is synco disabled versus enabled at
`settings.steps=1`, with the other workload unchanged. Change the step count
only while disabled and wait for clock warmup to finish. Production explicitly
supports one step, including the existing integration test. This retains
schedule/provider handling, main matching, passive copies and ordinary waits,
while removing the three auxiliary triggers, their final busy waits and the
extra downstream DM commands. It reduces copy/provider work too, so recovery
would identify the added four-step workload without by itself separating
scheduler overhead from controller/driver load. Persistent slowdown at one
step would instead direct attention to the production path that remains.
The operator reports **one step is normal; four steps slows the MVMs**. The
slowdown therefore depends on the work added by auxiliary slots. This does not
yet distinguish auxiliary scheduling/copies/provider work from the increased
combiner/controller/driver load. No instrument actions or application changes
were performed by the agent.

The user asks about waits in the processing chain. The primary GPU MVM waits
for its input trigger and, in graph mode, explicitly calls
`cudaStreamSynchronize` before using the result. Its `GPU...MVM...us` status
duration starts after the input wait and ends before output publication, so
CUDA completion waiting is included but input-semaphore waiting and output
notification are excluded. Non-graph mode instead retrieves results with a
device-to-host `cudaMemcpy`. These are host elapsed measurements.

Synco waits for a primary notification, the matching combined publication,
reader release before contribution overwrites, and absolute slot deadlines.
The current BMC pipeline stages private work and releases the input without
waiting for SDK completion in its monitor callback. The SDK worker waits for
requests and executes the synchronous SDK call; the post worker waits for both
preparation and SDK completion. Exhausted pipeline credits reject a new
command rather than blocking its producer.

There is no direct reader-acknowledgement dependency from synco back to the
primary GPU MVM: the inspected ImageStreamIO publication uses semaphore posts,
not a wait for consumers. Waking consumers can still add scheduling and shared
resource load. Distinguishing host CUDA-completion delay from actual graph
duration requires the existing MVM `w` and `k` timing components in a matched
off/on comparison; in graph mode `k` includes transfers as well as the MVM.
