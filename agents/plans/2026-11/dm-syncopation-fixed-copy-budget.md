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
