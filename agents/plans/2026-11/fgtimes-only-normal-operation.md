# Normal operation collects only fgtimes

User request: “for both I would like to normally have only fgtimes being collected and reported”
(camera `ocam2KCtrl`/`camwfs` and DM `bmcCtrl`/`dmtweeter`).

## Decisions

- Reuse camera `framegrabber.stageTiming=false`; add `dm.stageTiming=false` as the
  normal DM application default. A restart applies the capture/reporting mode.
- Keep camera acquisition/publication timestamps and DM receipt/successful SDK
  return endpoints for all three compatible fgtimes populations.
- Gate stage clocks before reading them, including detached pipeline events and
  joined BMC worker wait brackets. Basic DM mode also suppresses delivery-clock
  brackets and publication analysis, regardless of stored delivery settings.
- Retain historical low-level DM collector defaults for direct callers; the
  application explicitly supplies its default-false mode.
- Skip diagnostic-only DM stream metadata observations in basic mode.
- Suppress detailed telemetry even on forced reports. Register/update only the
  basic timing INDI property in normal mode. Operational SDK/pipeline, completion,
  and reader-release properties and protocol signals remain independent.
- Schedule the selected telemetry type. Update ALPAO, IrisAO and PI-335 selectors
  alongside BMC because they share the new DM application default. OCAM schedules
  stage records only in detailed mode.
- Keep SDK operational clocks used for timeout, readiness and command ordering.
  No instrument interaction or runtime latency benchmark is performed here.
- Existing camera diagnostics explicitly enabled in the deployed configuration
  require `stageTiming=false` and `stageTraceMax=0` to return to normal mode.

## Validation

All 149 test cases passed (4239 assertions) across the following ten software-only
suites. Rebuilt each target after the final source changes. The existing shared
Catch object is coverage-instrumented, so these builds link with `TESTLIBS=-lgcov`.
Builds were serialized because each Makefile.one invocation regenerates the same
git-version header.

| Suite | Cases | Assertions |
| --- | ---: | ---: |
| `dmTiming_test` | 31 | 877 |
| `dmTiming_integration_test` | 12 | 383 |
| `frameGrabberTiming_test` | 20 | 323 |
| `frameGrabber_integration_test` | 7 | 124 |
| `frameGrabber_test` | 6 | 74 |
| `bmcCtrl_test` | 31 | 1447 |
| `ocam2KCtrl_test` | 36 | 902 |
| `alpaoCtrl_test` | 1 | 18 |
| `irisaoCtrl_test` | 1 | 24 |
| `pi335Ctrl_test` | 4 | 67 |

Tests count clock reads, compare compatible populations across successful and
failed commands, exercise forced reports, and preserve reader-release, SDK result,
and product publication behavior with profiling disabled. Camera basic mode adds
no diagnostic monotonic reads; DM basic mode takes a receipt read and, for the
synchronous backend, its SDK return read. Worker SDK return endpoints are imported
from existing operational captures. Detailed-mode coverage remains intact.

All 14 changed C++ files match clang-format after restoring the mandated
`{ //mutex scope` spelling. `git diff --check` passes. Existing fields retain their
positions when adding mode flags to event/summary state.

## Affected files

Functional sources:
- `libMagAOX/common/dmTiming.hpp`
- `libMagAOX/app/dev/dmTiming.hpp`
- `libMagAOX/app/dev/dm.hpp`
- `libMagAOX/app/dev/frameGrabber.hpp`
- `libMagAOX/app/dev/frameGrabberTiming.hpp`
- `apps/bmcCtrl/bmcCtrl.hpp`
- `apps/ocam2KCtrl/ocam2KCtrl.hpp`
- `apps/alpaoCtrl/alpaoCtrl.hpp`
- `apps/irisaoCtrl/irisaoCtrl.hpp`
- `apps/pi335Ctrl/pi335Ctrl.hpp`

Regression tests:
- `libMagAOX/app/dev/tests/dmTiming_test.cpp`
- `libMagAOX/app/dev/tests/dmTiming_integration_test.cpp`
- `libMagAOX/app/dev/tests/frameGrabber_integration_test.cpp`
- `apps/bmcCtrl/tests/bmcCtrl_test.cpp`

Documentation:
- `libMagAOX/app/dev/doc/dmTiming.md`
- `libMagAOX/app/dev/doc/frameGrabberTiming.md`
- `apps/bmcCtrl/doc/sdk-worker.md`
- This execution record.

## Deployment and limits

Rebuild/install affected applications and restart them. For the currently enabled
camera diagnostic configuration, set `[framegrabber] stageTiming=false` and
`stageTraceMax=0`. Use `[dm] stageTiming=false` for the DM (also its new default).
Stored delivery-clock settings are inactive in this mode. To opt back into camera
traces, enable stage timing and a positive trace capacity together; enable DM stage
timing before selecting a producer delivery clock.

No measured instrument latency/jitter claim is made; the tests demonstrate removed
profiling clock reads and preserved processing/protocol behavior with software
stubs. Operational SDK, completion and reader-release clocks still execute.
