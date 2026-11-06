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
