# frameGrabber reconfiguration request race

Implemented on 2026-10-29 on workstation `mx11`, following the explicit request
"ok got it. yes please proceed". No hardware commands were run.

This follows the camera temperature work recorded in
[camera-temperature.md](camera-temperature.md). Changes are recorded on
`jrmales/camera-emgain-temperature`.

## Problem

The shared worker used a plain `bool` for cross-thread requests and cleared it
after `configureAcquisition()` and output-stream setup. A newer callback request
could be overwritten by that late clear. Making the flag atomic alone would
remove the data race but retain the lost notification.

The initial evaluation proposed consuming before configuration. Final inspection
showed that EDT cameras apply `m_nextMode` in `reconfig()`, before configuration.
Consuming at the next configuration entry could still swallow a mode request
queued after that cleanup. The implementation therefore consumes at the cleanup
boundary instead.

## Implemented protocol

- `m_reconfig` is `std::atomic<bool>`, using default sequentially consistent
  operations. Existing application assignments and boolean reads remain valid.
- Remove the unconditional clear after output setup.
- At the existing loop-tail cleanup boundary, first check shutdown, then consume
  with `m_reconfig.exchange(false)` immediately before `derived().reconfig()`.
- A request arriving after the exchange survives cleanup, the next configuration,
  stream setup, and acquisition startup, and triggers another cleanup/pass.
- If `reconfig()` returns an error, set the request again so failed cleanup does
  not discard it. This preserves concurrent requests as well.
- Retain existing readiness, configuration/start retries, and cleanup ordering.
  A request already pending at startup can cause one additional pass.

Requests coalesce: the latest settings may already be observed by an earlier
pass, so repeated application must be safe. The atomic flag protects the request,
not mutable camera settings; applications retain their existing locking rules.

## OCAM adjustments

- `setFPS()` and `setSynchro()` take a defaulted `requestReconfig` argument.
  Ordinary callbacks retain their behavior. `configureAcquisition()` passes
  `false` both directly and through the nested synchro-to-FPS path, avoiding an
  endless self-request loop and preserving an already queued mode/request.
- `reconfig()` holds `m_indiMutex` before `m_cameraMutex`, matching configuration
  lock order and serializing the requested mode with INDI callbacks.
- If `m_nextMode` is empty, cleanup reloads `m_modeName`. A repeated or coalesced
  request therefore cannot pass an empty mode to EDT.
- The sync-frequency callback now holds the INDI mutex while updating state and
  queuing its reset, so it uses the same ownership rule.

No generic outer INDI lock was added: existing derived configure methods already
acquire that mutex, and wrapping them would deadlock.

## Affected files

- `libMagAOX/app/dev/frameGrabber.hpp`
- `apps/ocam2KCtrl/ocam2KCtrl.hpp`
- New `libMagAOX/app/dev/tests/frameGrabber_test.cpp`
- New `apps/ocam2KCtrl/tests/ocam2KCtrl_reconfig_test.cpp`
- `tests/tests.list`, `tests/groups.dox`, `tests/Makefile.one`
- `apps/picamCtrl/tests/picamCtrl_test.cpp`: update the existing pending-temperature
  test's wording to describe request consumption before temperature application.
- This record and the linked camera-temperature record.

Touched C++ files received their documentation pass and clang-format. Required
`//mutex scope` spelling was retained after formatting. No configuration changes
were needed for this fix.

## Offline verification

The new worker suite invokes the real `fgThreadExec()` with private ImageStreamIO
streams, atomic fake shutdown/state/power, and bounded condition-variable pauses.
It covers requests during configuration, the original late-clear window, cleanup,
and the gap between cleanup and the next configuration; coalescing; transient
configuration/start/cleanup failure; acquisition-time requests; and readiness and
power waits. Workers are joined on assertion failure.

An isolated `/tmp` build using the original HEAD framegrabber header fails both
intended mid-configuration and late-clear regression sections. The working tree
was not reverted for that negative control.

Completed suites:

| Suite | Assertions | Cases | Result |
| --- | ---: | ---: | --- |
| frameGrabber worker | 74 | 6 | Passed |
| OCAM reconfiguration | 187 | 4 | Passed |
| Existing OCAM | 474 | 22 | Passed |
| PICam | 8762 | 6 | Passed |
| PVCAM | 331 | 10 | Passed |
| shmimIntegrator | 2003 | 20 | Passed |

MCP3208 compiled with the atomic flag but has six failing cases (29 of 35 cases
pass; 259 of 265 assertions pass). An isolated build against the original HEAD
framegrabber header reproduces exactly the same six failures, including values
and expressions: configuration channel count, three timing diagnostic checks,
synchronized trigger timestamp, and delay-controller timing. No MCP3208 source
was changed. These baseline failures remain a separate test-maintenance item.

CRED2 also compiled, but its full suite aborted in the existing appLogic
mutex-held test: line 2321 expects READY and receives ERROR, then assertion
unwinding destroys an unjoined test thread. Running that section against both
the current and original HEAD framegrabber headers reproduces the same values,
assertion counts, and exit code 134. Remaining cases after that abort were not
verified by the full run. No CRED2 source was changed.

Tests use normal `-O3 -ffast-math` and `TESTLIBS=-lgcov` for the existing
coverage-instrumented dependencies. Builds target the exact executable rather
than the default target to avoid concurrent regeneration of the shared version
header. Template consumers were explicitly rebuilt because the makefile does not
track all included headers.

## Boundaries

- A persistent `configureAcquisition()` or `startAcquisition()` failure can still
  prevent the loop from reaching cleanup. Pending requests remain set, but this
  change does not redesign controller-specific failure recovery.
- Existing ordinary OCAM latency/FPS requests can replace a queued mode with the
  current mode. That setting-coalescing policy is separate from notification
  loss and is preserved here.
- Mutable-setting synchronization in other controllers, particularly PVCAM, is
  unchanged. Atomic notification does not certify those other accesses.
- Keep PICam's independent pending-temperature guard; request consumption is
  earlier than successful SDK target application.
- Deploy/rebuild affected applications through the usual user-controlled process.
  No instrument service was started or restarted during this work.

## Suggested PR text

Title:

```text
Preserve framegrabber reconfiguration requests across worker passes
```

Description:

```md
This work was performed by GPT-6 (Codex) in response to the prompt: "can we fix the lost request bug?", followed by explicit approval: "ok got it. yes please proceed". The implementation is recorded in agents/plans/2026-10/frameGrabber-reconfiguration-race.md.

Make framegrabber reset requests atomic and consume them at the cleanup boundary, preserving requests arriving during cleanup/configuration/output setup and requeuing failed cleanup. Adjust OCAM's internal FPS/synchro application and mode locking so repeated passes remain safe.

Deterministic worker tests fail with the original header and pass with the fix. OCAM, PICam, PVCAM, and shmimIntegrator regressions pass. MCP3208's six failures and a CRED2 test abort reproduce unchanged against the original header. No hardware was operated.
```
