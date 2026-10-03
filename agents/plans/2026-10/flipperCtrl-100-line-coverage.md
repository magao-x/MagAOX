# prompt

as a follow-on, please propose a plan to bring flipperCtrl up to 100% line coverage following the established conventions for MagAO-X.  Let's add it as a new planning doc.

# plan

## Objective and scope

Bring the executable lines in both `apps/flipperCtrl/flipperCtrl.hpp` and `apps/flipperCtrl/flipperCtrl.cpp` to 100% coverage. Require `LH == LF` for each file in an LCOV trace, with both source records present and nonzero denominators. A rounded HTML percentage is insufficient.

Preserve the parking behavior implemented in the [parking plan](%23%20flipper-parking.md), including durable invalidation before motion, conservative restart recovery, power-off reporting, and the once-per-power-cycle mismatch WARNING. Extend tests around production controller code; do not duplicate its implementations in a harness or exclude reachable error handling to meet the target.

This is an app-specific line-coverage goal. Shared framework, tty, logger, INDI, and standard-library coverage remain separate concerns. Branch and function coverage are useful diagnostics, but 100% line coverage does not imply every branch or compiler-generated function has executed. Exercise meaningful alternate outcomes even when they share a covered source line.

## Measured baseline

Measured on 2026-10-03 at commit `9e12d7cb`, using GCC/gcov 15.2.0, LCOV/genhtml 2.0, the existing `COVERAGE=1` build flags, and the repository's HTML options. No production source or test implementation was changed during this planning pass.

- The existing suite passes all 13 test cases and 648 assertions under coverage instrumentation.
- The suite alone covers **294 of 377 header lines: 78.0%**, leaving **83 uncovered lines**. Raw gcov JSON, LCOV, and genhtml agree on that denominator and hit count.
- Compiling the real app and merging an initial LCOV capture exposes four entrypoint lines with no hits from the unit suite. A normal runtime-only capture can omit an app translation unit that has never run; require its source record explicitly rather than silently accepting that omission.
- A manual, bounded `flipperCtrl --help` smoke check in a private temporary working directory and MagAO-X base path returns the documented help status, **1**, prints the flipper options, and creates no position record. It covers all **4 of 4 entrypoint lines** and the header's 12 previously uncovered `setupConfig()` lines. Combined with the suite, the header reaches **306/377 (81.2%)**, with **71 lines remaining**. This smoke check is not yet registered as a test.
- Coverage capture reports a GCC/LCOV inconsistency warning in a standard-library header. It still succeeds, and the flipper source counts agree with gcov. Treat reporting-tool issues separately from app coverage; do not suppress uncovered flipper lines.
- Audit traces, gcov JSON, and the focused HTML report are retained under `/tmp/flipperCtrl-coverage-plan-20261003`. The normal app/test/Catch2-main builds were restored afterward, and only the profile files created by this audit were removed from the repository.

The following inventory refers to the unit-suite-only baseline. Line numbers identify the current revision and will move as implementation changes.

| Area | Current source range in `flipperCtrl.hpp` | Uncovered lines | Main missing behavior |
| --- | --- | ---: | --- |
| Configuration | 227–268 | 30 | Setup, defaults/overrides, load failures and shutdown request |
| State persistence | 270–378 | 8 | Non-missing read failure, directory/temp-file errors, directory-close error, failed external-change invalidation |
| INDI publication | 387–412 | 4 | Timestamp and real driver send paths for position and parked state |
| Startup | 414–432 | 2 | Selection creation/registration and read-only registration failures |
| Position query | 464–558 | 3 | Missing descriptor, oversized input, exhausted packet assembly |
| Device FSM | 560–633 | 32 | Power guards, discovery, connection/reconnection, discovery failures and post-query power change |
| Motion | 635–677 | 1 | Power changing after durable invalidation but before the serial command |
| Callback dispatch | 679–696 | 3 | No-selection return and static callback entrypoint |
| **Total** | | **83** | |

Status decoding, retained/unknown position resolution, power-off hooks, telemetry recording/scheduling, and shutdown already have their executable header lines covered. Preserve those tests and add failure contracts where the line metric alone does not distinguish success from failure.

## Established conventions

Use `AGENTS.md`, `tests/groups.dox`, `tests/testXWC.hpp`, `tests/Makefile.one`, `Make/common.mk`, and `tests/coverage/update_coverage` as the governing conventions. The OCAM lifecycle tests show how to register a separate lifecycle suite; ADC tracker tests provide examples of configuration and explicit fault outcomes. Follow the current grouping rules in `AGENTS.md` rather than older planning documents that use `app_unit_test` for applications.

- Keep the app header-only, with non-trivial definitions outside the class declaration and the `.cpp` limited to the entrypoint.
- Retain `libXWCTest::flipperCtrlTest`, the `flipperCtrl_unit_test` group under `application_unit_test`, and a Doxygen brief/group block for every case. Additional test files should use `\addtogroup` for the existing app test group.
- Include `tests/testXWC.hpp`; preserve explicit real-API references in `FLIPPERCTRL_TEST_DOXYGEN_REF` blocks bracketed by `// clang-format off` and `// clang-format on`. Hide harness-only declarations with `\cond ... \endcond` and explicitly link the entrypoint smoke test to `flipperCtrl.cpp`.
- Document the full changed C++ file: file blocks, members, function declarations, inline parameter descriptions, and return semantics. Keep `m_` names and definitions outside class declarations; run `clang-format` on touched C++ files.
- Retain the real callback validator in behavioral tests. `tests/testMacrosINDI.hpp` defines `XWCTEST_INDI_CALLBACK_VALIDATION`, which returns success immediately on a property match; enabling it in this suite would bypass the behavior being tested.
- Use fresh temporary directories, RAII cleanup, harmless descriptors/local sockets, and deterministic fault injection. Destroy one app before constructing another because `MagAOXApp` permits only one instance per process.
- Preserve `TELEMETER_*` interfaces and their actual return contracts. Keep lock ownership as implemented; annotate any new block used solely for mutex lifetime with `{ //mutex scope`.

## Implementation sequence

### 1. Make the coverage run reproducible

Force recompilation when changing headers, harness substitutions, or `COVERAGE`. `Makefile.one` does not supply complete header dependencies, and switching flags alone does not rebuild existing objects. An app build with only `COVERAGE=1` also reused the existing normal-build object during this audit; `-W flipperCtrl.cpp` forced the required app recompilation without rebuilding the whole repository.

Reset only the flipper test, entrypoint test, app, and shared Catch2-main counters used by a focused run. Do not run the repository-wide coverage cleanup merely to measure this app. Capture initial `.gcno` data as well as executed `.gcda` data, merge them, and extract both production source paths. Keep an uncovered-line inventory by function as work proceeds.

Use the standard `COVERAGE=1`/`-O0` configuration for the acceptance metric. Do not change optimization, combine stale counters, or remove source records to increase the reported percentage.

### 2. Extend the existing harness at dependency boundaries

Keep the current app/log capture base, telemetry sink, serial wrappers, and filesystem wrappers. Add narrowly scoped test-only USB and I/O bases that inherit the real helpers, preserve their fields, and delegate real configuration setup/loading unless a particular result is injected. Substitute those base names only while including the app header, after shared declarations have been included, and immediately undefine the substitutions.

Supply queued discovery/connect results, deterministic harmless connection descriptors, configurable telemetry return values, and specific property-creation/registration failure overloads on the existing test app base. This lets production `setupConfig()`, `loadConfigImpl()`, `loadConfig()`, `appStartup()`, and `appLogic()` execute unchanged.

For telemetry configuration, extend the sink to delegate real `telemeter::setupConfig()` and `loadConfig()` through the corresponding base, while retaining threadless lifecycle and captured-record behavior. The current sink's configuration methods are no-ops; assertions about telemetry defaults/options must not be satisfied by recreating that configuration in test code.

The qualified calls to `tty::usbDevice` and `dev::ioDevice` are non-virtual; merely adding same-named methods to the existing derived `Controller` cannot intercept them. Avoid broad token substitutions such as globally redefining `loadConfig` or `connect`. Prefer the existing test-local base substitution pattern; introduce a small documented production seam only if a concrete dependency cannot be tested this way, preserving its default delegation and behavior.

### 3. Cover configuration and startup contracts

Call the real app configuration methods using isolated configurators and temporary config files. Assert the registered USB, timeout, reversal, and telemetry options; default endpoint mapping and baud rate; explicit `flipper.reverse=false/true`; USB values; and read/write timeout overrides. Test the actual configuration path for reversal rather than relying solely on the harness's direct `reverse()` setter.

Inject USB load success, the two tolerated absent-device results, and an unexpected USB error. Verify the unexpected result is logged while the remaining configuration loads, matching the current behavior. Inject I/O and telemetry load failures to verify `loadConfigImpl()` returns failure and `loadConfig()` logs the fatal configuration error and requests shutdown. Cover both successful and failing telemeter setup, including the setup macro's shutdown flag.

For startup, fail selection creation, new-property registration, read-only parked-property registration, and telemetry startup separately. Assert the return/log result and that later steps do not run after a fatal failure. Keep the existing absent/corrupt state cases to confirm those remain recoverable startup conditions.

Exercise telemetry error returns through powered-on `appLogic()`, `whilePowerOff()`, and `appShutdown()`. In particular, `TELEMETER_APP_SHUTDOWN` logs a helper failure but does not propagate it: the app still returns zero. Test that contract instead of changing it to make a test pass.

### 4. Cover the complete discovery and connection FSM

Drive the real `appLogic()` from `POWERON`, `NODEVICE`, `NOTCONNECTED`, `CONNECTED`, `READY`, and `OPERATING`, with the following assertions:

- Power-off or a non-on observed/target power state returns before discovery, serial I/O, and persistence.
- `POWERON` clears live validity and enters discovery. Both absent-device codes remain in `NODEVICE`; an unexpected discovery error enters `FAILURE`, logs a critical error, and returns failure.
- Successful discovery records the found device and attempts connection. Connection success enters `CONNECTED`, then uses a fresh reply to reach `READY` or `OPERATING`.
- On connect failure, rediscovery can report disappearance (`NODEVICE`), another discovery error (`FAILURE`), or a still-present device (remain `NOTCONNECTED` and retry). Assert no false `CONNECTED` or `READY` and no premature position query.
- Reconnect after an earlier query failure succeeds without manufacturing parking from a target. Exercise both logged and unlogged state variants to verify discovery diagnostics are not repeated unnecessarily.
- A power change injected during the query reaches the post-query guard without overwriting the framework's off state or sending another command.

Assert FSM state, return value, log priority/message, dependency call order, descriptor ownership, and serial command counts. Include successful rediscovery after disappearance so the harness demonstrates recovery rather than only terminating paths.

### 5. Exercise actual INDI sends

Attach a real `indiDriver<MagAOXApp<true>>` using three private input/output/control FIFOs and the existing base's path members. Do not activate its processing thread or connect to an INDI server. Read outgoing messages with bounded local I/O, and let the app/base ownership release the driver and descriptors before removing the directory.

`sendSetProperty()` is non-virtual; a subclass with a same-named method would not capture these calls. A real local driver exercises the currently uncovered timestamp/send lines without changing production publication code.

Verify unknown, parked, busy, completed, reversed, and powered-off snapshots. Assert one coherent position message contains both switch values and the correct property state, the parked message carries the correct numeric flag, and unchanged publication emits no duplicate messages. Confirm the timestamped messages actually arrive rather than only inspecting in-memory properties. Do not use fake non-null driver pointers.

### 6. Finish persistence, framing, and power-transition guards

Extend the current filesystem wrappers with selected temp-file creation, permission-setting, and zero-write failures, plus a hook at a known persistence operation. Check cleanup, descriptor release, preserved errno diagnostics, installed file contents, and whether a command was allowed.

- Cause a non-`ENOENT` state-file open failure using a deterministic invalid path or symlink loop. Permission-only fixtures are unreliable when tests or writes have elevated privileges.
- Fail directory open and `mkstemp`; verify the directory descriptor is closed on temp-file failure. Inject a directory-close failure after successful rename and sync; even if the unparked replacement is installed, the move must be rejected.
- Fail `fchmod` and a zero-byte write as additional behavioral checks, even though their error paths share already covered lines with other faults.
- Observe an external endpoint change while an old parked record exists, then fail the initial invalidation save. Exercise the missing `saveState()` error return, diagnostic, bounded retry, and later recovery. Test `force=true` bypassing both the unchanged check and the retry deadline without sleeping.
- Query with power on and no descriptor: reject it without a serial write. Inject a reply over 4096 bytes and enough fragments or completion notifications to exhaust the eight-read limit; these must fail without a false `READY` or new parked certification.
- Add both accepted completion IDs and malformed completion headers as meaningful framing variants. Preserve the existing real-tty socket test and full status-bit decoder tests.
- Change observed or target power after durable invalidation and before the command guard. Assert no serial move, an unparked backing record, and conservative reporting after the subsequent power-off hook.

Trigger power changes through existing transport/filesystem test hooks or narrow new hooks at the required operation. Mutate the harness's power fields during the synchronous hook; call lock-owning power-off methods only after the tested method returns. Calling `onPowerOff()` reentrantly while `m_indiMutex` is held would deadlock and would not model the framework transition correctly. Use injected events rather than timing races or arbitrary sleeps.

### 7. Cover callback dispatch and the entrypoint

Call `st_newCallBack_m_indiP_position()` directly with a real controller pointer and valid, invalid, and no-selection properties. Cover both logical endpoints, reversal, absent elements, both switches off, conflicting selections, wrong device/name, and existing motion/power guards. Assert return values and hardware effects; do not stop after a validation-only success.

Add `apps/flipperCtrl/tests/flipperCtrl_entrypoint_test.cpp` as a separate Catch2 smoke suite, following the existing additional-lifecycle-suite pattern. Register it in `tests/tests.list` and add a narrowly targeted dependency in `tests/Makefile.one` to build the real flipper executable with the same coverage mode. This must work from a fresh checkout and from either the repository root or `tests/` working directory; do not silently skip a missing executable.

The test should run the instrumented executable with `--help` in a child process, a private working/base directory, and private relative config/log/sys paths. Bound execution, capture its help, assert the documented status of 1 and the flipper options, and confirm no position record is created. Do not start the device loop. Normal child exit flushes the app's gcov counters, covering the real entrypoint without adding a production-only main wrapper, renaming `main` through a broad macro, or excluding the `.cpp`.

## Verification and acceptance

During implementation, run the focused suites after each meaningful group of tests and inspect the actual remaining zero-count lines. After they pass, run the existing coverage-report fast path to check integration with the project report. A final combined trace must contain both app source records and satisfy exact `LH == LF` for each. Report the final numerator/denominator, uncovered-line list (empty), tool versions, and commands; do not infer coverage from assertion counts.

Representative final workflow from the repository root, once the new entrypoint test exists:

```sh
clang-format -i apps/flipperCtrl/tests/flipperCtrl_test.cpp apps/flipperCtrl/tests/flipperCtrl_entrypoint_test.cpp
make -C tests -B -f Makefile.one t=../apps/flipperCtrl/tests/flipperCtrl_test.cpp COVERAGE=1
make -C tests -B -f Makefile.one t=../apps/flipperCtrl/tests/flipperCtrl_entrypoint_test.cpp COVERAGE=1
rm -f apps/flipperCtrl/tests/flipperCtrl_test.gcda apps/flipperCtrl/tests/flipperCtrl_entrypoint_test.gcda apps/flipperCtrl/flipperCtrl.gcda tests/testMain.gcda
apps/flipperCtrl/tests/flipperCtrl_test
apps/flipperCtrl/tests/flipperCtrl_entrypoint_test
lcov --capture --initial --directory apps/flipperCtrl --output-file /tmp/flipperCtrl-initial.info
lcov --capture --directory apps/flipperCtrl --output-file /tmp/flipperCtrl-executed.info
lcov --add-tracefile /tmp/flipperCtrl-initial.info --add-tracefile /tmp/flipperCtrl-executed.info --output-file /tmp/flipperCtrl-combined.info
lcov --extract /tmp/flipperCtrl-combined.info "$PWD/apps/flipperCtrl/flipperCtrl.hpp" "$PWD/apps/flipperCtrl/flipperCtrl.cpp" --output-file /tmp/flipperCtrl-app.info
lcov --summary /tmp/flipperCtrl-app.info
tests/coverage/update_coverage --fast
git diff --check
```

Use a small trace check to require the two expected `SF` paths, positive `LF` values, equal `LH`/`LF`, and no zero-count `DA` entries. The summary alone can round a nearly complete result to 100%. Keep any app-specific permanent CI gate focused; the existing coverage workflow already builds/runs registered tests and produces the report, so no shared build/report overhaul is required.

Run `clang-format --dry-run --Werror` on changed C++ files and build the real app normally after coverage verification, forcing recompilation when switching modes. Rebuild the shared Catch2 main normally as well before reusing it with non-coverage test targets. If a production seam is needed, include that header in formatting and verify its default behavior with the existing tests and normal app build.

## Expected affected files and commit discipline

- `apps/flipperCtrl/tests/flipperCtrl_test.cpp`: existing harness extensions and additional production-path cases.
- `apps/flipperCtrl/tests/flipperCtrl_entrypoint_test.cpp`: bounded real-entrypoint smoke test and its documentation.
- `tests/tests.list`: register the additional suite.
- `tests/Makefile.one`: only the additional suite's real-app build dependency, preserving the current generic test flow.
- `apps/flipperCtrl/flipperCtrl.hpp`: only if a demonstrated dependency requires a narrow production seam; keep existing behavior and the full-file documentation standard. No functional rewrite is planned.
- This plan: record implementation decisions and the final measured coverage results as the follow-on work progresses.

Continue on a namespaced feature branch. Keep any necessary functional testability change and its engineering notes first, test/documentation additions in clear commits, and formatting-only cleanup last when needed. Append `Co-authored by GPT-6.1 Sol Codex` to every commit message. Leave `AGENTS.md` unchanged, as requested for the parking work.

Status: planning complete; coverage implementation has not started. The existing instrumented tests, app build, baseline capture, and manual entrypoint feasibility check passed. Hardware validation of parking remains a separate follow-up; line coverage cannot verify physical movement, filesystem crash durability on every storage device, or every real power-transition timing.
