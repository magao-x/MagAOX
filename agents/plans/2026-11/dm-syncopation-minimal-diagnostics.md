# Minimal production diagnostics

Date: 2026-11-06.

Request: "well let's do that just, go down to the minimal diagnostics".

## Evidence and scope

The [fixed-budget investigation](dm-syncopation-fixed-copy-budget.md) records
normal MVM timing with synco disabled or enabled at one step, and elevated
MVM timing at four steps. Stopping the GPU reconstructor did not improve the
timing. Optional synco camera and combined-output observers were still enabled;
the latter receives roughly four times as many notifications at four steps.
This change removes optional diagnostic work for a controlled operator test.
It does not establish that diagnostics caused the instrument slowdown.

All work is on workstation `mx11`. No instrument queries, instrument stream
attachments, deployed configuration edits, application starts or hardware
actions are performed by the agent. Tests use isolated fixture streams.

## Implementation

Reuse `diagnostics.windowSec=0` for minimal collection. Production defaults to
zero when the setting is omitted; an explicit positive window restores detailed
diagnostics. Other modes retain the existing two-second default. Startup-only
configuration avoids changing thread topology during a running cycle.

Minimal mode removes:

- Optional camera and combined-output monitor threads.
- Primary cadence and upstream trace collection in the scheduler callback.
- Exact main/reader/SDK timing joins and per-slot phase/publication populations.
- Per-command event construction, queueing and logging, including failure rows.
- Optional population formatting and ordinary-loop profiling.

Retain clock fitting/mapping and qualification, exact combined-output matching,
reader-release transport and validation, SDK fault checks, resource identities,
all preparation/publication deadlines, fixed copy budgets and measured maxima.
Operational counters, health, clock status and low-rate summaries remain.
Reader release is not an optional diagnostic observer.

`diagnostics.maxEvents` must remain positive because it also sizes the required
reader-release queue. The zero-event queue is disabled independently. Production
no longer requires a positive diagnostic window or the optional combiner flag.
The primary receipt timestamp remains captured when the scheduling engine needs
it even if all optional collectors are disabled.

Expose disabled collection explicitly using `zero_test.diagnosticsEnabled`,
`zero_phase.enabled`, `zero_publication.enabled` and `logic_timing.enabled`.
Existing observer quality properties likewise report `enabled=0`.
Do not interpret empty timing populations as zero jitter or zero SDK failures.

## Verification

The normal application build and complete scheduler suite pass: **158 cases,
6,999,307 assertions**. Regressions cover default/explicit diagnostic settings,
one- and four-step production through the real primary receipt hook, absent
optional collectors and event populations, and retained rejection of invalid
clock, reader-release and SDK evidence. A nonproduction configuration case also
verifies the existing two-second diagnostic default is unchanged.

The first focused run found two incorrect expectations in that new nonproduction
case: it retains the legacy 5 us late allowance and zero wake-ahead, rather than
the production 6/10 us values. Correcting the test expectations required no
application change. The final full suite passes after rebuilding.

Independent source review found no required control path gated by diagnostic
collection. All changed C++ headers pass clang-format verification, Doxygen
declarations and test references were reviewed, and `git diff --check` passes.
No runtime performance claim is made before the operator comparison.

Validation logs: `/tmp/synco-minimal-test-build.log`,
`/tmp/synco-minimal-suite.log`, `/tmp/synco-minimal-app-build.log`.

## Operator follow-up

Rebuild and restart synco using the same CPU placement, four steps, fixed 30 us
copy budget, controller configuration and workload. If an existing configuration
explicitly sets a positive diagnostic window, set it to zero for this comparison.
Confirm disabled diagnostic flags and continuing active/readiness/clock counters,
then compare MVM times and controller fgtimes. Restore `windowSec=2` and restart
only when a detailed capture is needed.

## Operator result

The operator reports "no change" in the slowdown and confirms the mean is
`1.25e-4` seconds, or 125 us. Fresh trial-2 readbacks confirm
`diagnosticsEnabled=0`, valid snapshots, active four-step scheduling and no
clock exclusions. Over 13.036513728 s, main cycles advance by 26,073, auxiliary
posts by 78,217 and ordinary skips by two, both preparation-late. The identity
78,217 + 2 = 3 * 26,073 holds. Main cycles plus posts imply 7,999.838 Hz; this
count-derived rate is not an independently matched controller delivery count.

At otherwise ideal 125 us spacing, the two skips would contribute approximately
0.55 us of interval deviation if separate, or 0.77 us if consecutive, across
that whole interval. These current counts do not support omitted slots as the
dominant source of sustained 7--8 us jitter. Historical wide-clock exclusions
must not be substituted for these fresh zero counts. The reported fgtimes
windows and counter interval are not explicitly time-aligned.

The operator supplies disabled fgtimes from 20:59:06--13 UTC and enabled
fgtimes from 20:59:24--32 UTC on 2026-11-06:

| Controller measurement | Disabled | Enabled at four steps |
| --- | ---: | ---: |
| Mean receipt interval | 500 us | approximately 125 us |
| Receipt interval deviation (`acq`) | 3.64--4.26 us | 7.58--8.14 us |
| SDK-return interval deviation (`wrt`) | 3.77--4.18 us | 5.93--6.27 us |
| Mean receipt-to-return latency (`wma`) | 113.60--113.92 us | 113.77--113.82 us |
| Receipt-to-return latency deviation | 0.48--1.22 us | 2.00--2.45 us |

These are centered standard deviations, not maximum excursions. Controller
receipt precedes preparation and SDK dispatch; SDK return is a software
endpoint, not physical settling. Latency includes preparation, dispatch/queue
waiting and native SDK work. Smaller return-interval than receipt-interval
deviation is compatible with serialization smoothing irregular arrivals, but
does not identify its mechanism. Do not subtract these variances as independent
stage noises or combine overlapping rolling windows. The configured defaults
are a two-second window and 20,000-event capacity; the local controller config
does not override them.

The rate is now essentially correct, while timing variation among delivered
commands remains. Keeping minimal diagnostics is useful, but this comparison
does not establish a matched performance improvement attributable to disabling
them. No further deadline, copy-budget or clock-gate change follows from these
readings.

## Shared-resource hypothesis and next comparison

The operator asks whether this is memory contention on `acqu_mvm`. The tracked
RTC layout places the MVM host on CPU45, SDK worker on CPU42, ordered post
worker on CPU44, controller on CPU40 and synco on CPU41. Operator-provided
topology from the cpuset investigation places CPUs40--47 on distinct physical
cores sharing one L3 cache. Their cpusets select cores but do not isolate that
cache or memory bandwidth. NVIDIA IRQs are also configured on CPU45; live
placement and interrupt rates are not supplied by the latest capture.

Shared-cache or memory-system interference is plausible, especially given
normal one-step timing and elevated four-step MVM times. It is not measured
cache contention or established DRAM bandwidth saturation. The resident GPU
matrix and the host's transfers/driver/wait work must not be conflated. Synco
has no direct wait on either MVM consumer; earlier reconstructor-stop and
minimal-diagnostics comparisons did not resolve the slowdown.

The next placement comparison can restore the SDK worker to `spare56` and
post worker to `spare31`, after the operator checks those reserves are unused.
Only the controller's `[sdk] cpuset` and `pipelineCpuset` settings change;
priorities, four steps, 30 us copy budget, five-microsecond combiner delay and
minimal diagnostics remain fixed. Restarting and testing the controller are
operator actions. Compare MVM timings and controller fgtimes with the current
placement, ideally restoring the original placement for a repeat. Recovery
would implicate placement/shared-resource effects, but alone would not prove
which cache, memory or driver mechanism is responsible.

## Successful placement comparison

The operator performed the requested worker-placement comparison and reports
"this fixed it" on 2026-11-06. Treat SDK CPU56 / post CPU31 as the working
placement and the recent CPU42/44 assignment as a placement regression. The
reported recovery, together with the earlier normal one-step and slow four-step
comparison, strongly implicates interference associated with the added DM
workload in the MVM host's CPU/cache group. It does not establish L3 contention
specifically, quantify DRAM bandwidth, or separate the SDK and post workers'
individual contributions. No numerical post-fix capture was supplied, so do
not invent a restored jitter value or claim a specific improvement factor.

Update the production configuration guide, reader-test placement comment and
cpuset migration record to preserve this result. The executable CPU map already
provides `spare56` and `spare31`; no cpuset recreation is required. The separate
workstation `/opt/MagAOX/config` checkout on `onsky` still requests
`dmtweeter-sdk` and `dmtweeter-post` in `dmtweeter.conf`; its persistent config
record must also retain the operator's working assignments. That checkout and
all deployed configuration are unchanged by this documentation update.

Retain minimal diagnostics and the fixed 30 us copy budget. No further scheduler
or timing-margin change is warranted by the successful placement result.
Documentation links and `git diff --check` were verified; no application tests
were rerun for these documentation/comment-only changes.
