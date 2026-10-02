# prompt

We need to add parking logic to the flipperCtrl such that it reports position when in state POWEROFF. Will need to implement this with our own backing store for position. See the zaberLowLevel, zaberLowLevelBinary, zaberCtrl, and stdMotionStage for the basic layout of parking.  In this case the device does not provide parking.  We will store position on disk as is done in, for instance, picoMotorCtrl, so that we can recover state after a software restart with power off.  

Analyze the problem and formulate a plan, documenting it below.  Do not alter this prompt above "# plan".  Review AGENTS.md.

# plan

## Objective and scope

Add software parking to `flipperCtrl`: retain a confirmed physical endpoint through power-off, publish its logical `in`/`out` position while the FSM remains `POWEROFF`, and recover that information when the app starts with the device powered off. Parking is bookkeeping; it does not command a special physical position or issue a device parking command.

Keep the existing `presetName` property, its `in` and `out` elements, and `flipper.reverse`. Store physical endpoint numbers, not logical names, so reversal is applied consistently after recovery. Add a read-only `parked.current` number (`0` or `1`), following `zaberCtrl`, to expose whether a confirmed endpoint is available for retention.

Implement this within the existing app rather than introducing `dev::stdMotionStage` inheritance. That helper supplies useful power-off and telemetry conventions, but adopting it would also introduce numeric presets, home/stop properties, and configuration unrelated to this change.

## Findings from the current code

- `MagAOXApp::execute()` calls `appStartup()` before determining initial power state. With power management enabled, it calls `onPowerOff()` both on initial startup with power off and on later power-off transitions, then calls `whilePowerOff()` instead of `appLogic()`. Adding a `POWEROFF` branch only to `flipperCtrl::appLogic()` would therefore miss the required behavior.
- `flipperCtrl` currently overrides neither power-off hook. Its switches and telemetry are updated only in `READY`/`OPERATING`, and its initial `m_pos = 1` is a default rather than a device observation.
- `getPos()` ignores the request write result, logs a failed read without returning an error, accesses `response[16]` without validating its length, and maps every value other than `1` to endpoint `2`. Both connection initialization and normal polling ignore its return value. These paths must be corrected before their results can be persisted.
- `moveTo()` ignores the serial write result. The position callback changes `m_tgt` before taking `m_indiMutex`, has no power/connection guard, and can set the FSM to `OPERATING` even while power is off. These changes must be serialized with polling and power-off handling.
- The Zaber low-level controllers load position/parked state at startup, park settled stages during normal operation, and write their state then. Their power-off hooks preserve that snapshot without querying hardware. `zaberCtrl` preserves a powered-off preset only when its retained parked flag is true.
- `stdMotionStage` uses `telem_stage.moving = -2` for power off. The existing flipper telemetry derives moving solely from `m_tgt != m_pos`, so it cannot currently represent power off or an unknown position correctly.
- `picoMotorCtrl` uses `m_sysPath + "/" + m_configName` for position storage and `elevatedPrivileges` for writing. `MagAOXApp::lockPID()` creates that directory before `appStartup()`. Reuse this location and privilege pattern, with stronger parsing and checked writes.
- `apps/flipperCtrl/tests/flipperCtrl_test.cpp` contains only a construction test. It is already registered in `tests/tests.list` and has an application unit-test Doxygen group.

## Position and parking semantics

Use explicit position validity rather than treating the constructor default or a command target as a measurement. Keep `m_pos` as the last confirmed physical endpoint, with `0` meaning no endpoint has ever been confirmed, and add `m_parked` for a settled endpoint that can be retained. Keep pending-motion state separate from position: recovery must not infer completion merely because a remembered position equals a target.

| Situation | `parked.current` | `presetName` | Stage telemetry |
| --- | --- | --- | --- |
| Powered on, valid settled endpoint | `1` | Confirmed logical endpoint selected; `INDI_IDLE` | Existing physical endpoint value and mapped name; moving `0` |
| App-commanded move pending | `0` | Preserve existing busy selection behavior using the last confirmed endpoint; do not select the target as a measurement | Moving `1`; no new confirmed endpoint until the device reports completion |
| `POWEROFF`, retained parked endpoint | `1` | Retained logical endpoint selected; `INDI_IDLE` | Retained endpoint/name; moving `-2` |
| `POWEROFF`, unparked or no valid snapshot | `0` | Both switches off; `INDI_ALERT` | Preset `0`, empty name; moving `-2` |
| Powered on, invalid/unrecognized status | `0` | Both switches off; `INDI_ALERT`; do not enter `READY` | Preset `0`, empty name; do not claim stopped at an endpoint |

The selection helper already initializes both elements off, so the unknown representation needs no extra `none` element. Preserve the existing physical `1`/`2` telemetry numbering, including when reversed; apply reversal to the name and switch selection. Introduce one helper for publishing switches, parked status, and the corresponding telemetry values so startup, normal polling, and power-off agree.

## Backing store

1. Use one app-specific file: `m_sysPath + "/" + m_configName + "/position"`. No new required configuration is needed.
2. Store two newline-separated integers: the last confirmed physical endpoint (`0`, `1`, or `2`) and its parked flag (`0` or `1`). For example, `2\n1\n` records a confirmed endpoint `2`; `2\n0\n` retains the previous observation for diagnostics but does not certify a recoverable position. A parked flag of `1` requires endpoint `1` or `2`.
3. Read into temporary values and validate the entire record before assigning members. Missing files are a normal first-run condition. Empty, truncated, malformed, out-of-range, or inconsistent records produce an unknown/unparked state and a diagnostic, without shutting down the controller or guessing an endpoint. Ignore abandoned temporary files.
4. Load once during `appStartup()`, before initializing/registering the position and parked properties. A valid parked record sets `m_pos` and `m_tgt` to the stored endpoint. An unparked record must not restore a pending target or become parked merely because the two values match.
5. Save a parked record after a successful live query confirms a settled endpoint, including first connection, move completion, and an observed change made outside the app. Avoid writing on every unchanged FSM loop; track successful saves and retry failed saves at a bounded rate.
6. Before sending any command that can move the flipper, persist an unparked record. This prevents a crash or software restart during motion from recovering the previous endpoint as parked. Do not wait until `onPowerOff()` or shutdown to invalidate or save state.
7. Use a temporary file in the same directory, check write/flush/close results, then atomically rename it over `position`. Sync the file and directory where needed to make invalidation durable before sending a move. Scope `elevatedPrivileges` to the required filesystem operations and clean up failed temporary writes.
8. Proposed failure policy: if pre-move invalidation cannot be persisted, reject the move before touching hardware. A failed save of a newly confirmed endpoint is logged and retried; live reporting remains available. For an app-commanded move, the previously installed unparked record prevents stale recovery until the save succeeds. If a changed endpoint is observed without an app command, attempt to invalidate any stale record before saving the replacement; an inability to write either record must be logged as a recovery risk. This is a deliberate command-policy change needed to prevent stale recovery for moves controlled by the app.

## Implementation sequence

### 1. Make device observations trustworthy

- Separate response validation/decoding from serial I/O sufficiently to test full, short, malformed, and transitional replies. Check request writes and response reads before examining bytes. Validate the expected reply framing and endpoint/status indication, rather than treating any other value as endpoint `2`.
- Verify the endpoint and moving-status representation for the installed flipper before coding that decoder. The repository contains the request bytes and the current byte-16 test, but no protocol documentation establishing how transitional status is encoded. Do not assume equality to a single byte proves a settled endpoint.
- Return read/write failures to callers. A failed connection or initial position query must not transition to `READY`; retain or re-enter the appropriate disconnected/error state and retry. A failed powered-on query must not manufacture or persist a position. Treat failures during a known power-off transition through the power-off path, without attempting further device communication.
- On reconnect, use a fresh live observation as authoritative even if it disagrees with the disk snapshot. Reconcile the snapshot without moving the device automatically. As added during plan approval, issue a WARNING log once when the first settled live endpoint after power-on differs from the inferred parked endpoint.

### 2. Add state persistence and serialize move handling

- Add documented state-file read/write helpers and the small amount of per-instance state needed for parking, pending moves, and save-change/retry tracking.
- Make `moveTo()` the single lock-owning move entry point. Take `m_indiMutex` there before changing target, parked validity, or pending-motion state; require a connected `READY`/`OPERATING` state and power permitting motion, persist invalidation, then send the move. The callback validates the selection and calls this path without taking the lock again. Retain the existing ability to replace an active target while powered on.
- Reject requests selecting both `in` and `out`. Put persistence and power checks inside `moveTo()` so another caller cannot bypass them. Requests for the already confirmed settled endpoint can remain a no-op. Document which publication/persistence helpers require the caller to hold the mutex to avoid recursive locking.
- A serial write failure leaves the snapshot unparked because a partial command may have reached the device. Re-park only after a fresh settled observation. Likewise, an observed transitional/invalid device status must not preserve a parked certification; invalidate the backing record when previously parked state becomes uncertain.
- Publish both position switches coherently after updating their values and property state. Avoid separate messages that can transiently select both endpoints or retain a busy state after power off.

### 3. Add the power-off lifecycle

- Implement `onPowerOff()` under the same state lock. Close/reset the USB descriptor using its existing sentinel convention, clear pending command state, and retain only an already confirmed parked endpoint. Preserve the raw last observation for diagnostics when unparked, but publish unknown position in that case. Do not turn an unfinished command target into a retained endpoint.
- Update `presetName` and `parked.current`, then force an OFF `telem_stage` record. Keep the FSM in `POWEROFF`; perform no serial queries or moves.
- Implement `whilePowerOff()` to maintain the retained/unknown publication and run telemetry scheduling. Perform no repeated disk writes or device I/O. Keep power-off telemetry using moving `-2` even if a remembered target differs from the last observation.
- Keep the existing power-on discovery/connect sequence, with fresh observation required before accepting commands. Ensure an arriving INDI request cannot overwrite `POWEROFF` during either hook.
- Include telemetry shutdown cleanup in `appShutdown()`. Prefer the existing `TELEMETER_*` lifecycle macros where applicable, as required by `AGENTS.md`, and account for their return/error behavior. No telemetry schema or shared stage-helper change is required.

## Validation

Extend the existing test file with a harness using temporary app sys directories and controllable serial replies/writes. Use the existing INDI test support; test the real response-decoding, persistence, move, publication, and power-off paths without attached hardware.

- Round-trip both endpoints and both parked flags; verify default and reversed logical mappings. Missing/bad records must yield unknown, and records from different `m_configName` directories must stay isolated.
- Restart a second controller instance with power off after a valid save: assert the retained switch, parked flag, FSM, and OFF telemetry. Repeat with an unparked record and after simulated interruption between invalidation and completion.
- Verify invalidation is installed before the move write; failed invalidation sends no command. Exercise serial write failure, failed confirmation save, retry, failed rename/flush where injectable, and an abandoned temporary file.
- Exercise valid endpoint replies, transitional/ambiguous status, malformed framing, short reads/timeouts, and request-write failures. Assert no false endpoint `2`, no false `READY`, and no newly parked record on errors.
- Cover power off while idle, during a move, before any successful query, and at initial startup. Assert both power-off hooks perform no hardware I/O, telemetry remains scheduled, and repeated off loops do not rewrite state.
- Verify callbacks are rejected while off/disconnected or while power is turning off, without changing position/target/FSM or writing hardware. Cover conflicting selections, same-endpoint requests, and replacing an active target.
- Reconnect with a live endpoint different from the retained record: emit one WARNING for the power-on mismatch, let the device observation win, and update the stored record after confirmation. Matching endpoints and ordinary powered-on endpoint changes must not produce this warning.
- Verify complete INDI publication states and decode captured `telem_stage` messages, including moving `-2`, unknown preset `0`/empty name, and change detection across power transitions. Prefer per-instance telemetry change tracking over the current function-static values if needed for reliable restart-instance tests.

Run `clang-format` on the changed C++ files and build/run the targeted suite from the repository root:

```sh
clang-format -i apps/flipperCtrl/flipperCtrl.hpp apps/flipperCtrl/tests/flipperCtrl_test.cpp
make -C tests -B -f Makefile.one t=../apps/flipperCtrl/tests/flipperCtrl_test.cpp
apps/flipperCtrl/tests/flipperCtrl_test
make -C apps/flipperCtrl
git diff --check
```

Also format `flipperCtrl.cpp` if its documentation is updated. Hardware validation should cover both endpoints, reversal, idle power-off plus app restart, interrupted motion plus restart, and subsequent power-on recovery. This planning pass does not change C++ or run implementation tests.

## Affected files and documentation discipline

- `apps/flipperCtrl/flipperCtrl.hpp`: functional state, persistence, device-result handling, power hooks, INDI publication, and telemetry. Preserve the app's header implementation pattern, with non-trivial definitions outside the class declaration; rule 22 supplies the app-specific guidance for rules 6 and 13.
- `apps/flipperCtrl/tests/flipperCtrl_test.cpp`: behavioral tests and a Doxygen-hidden harness. Keep `application_unit_test` grouping, document every test case, and preserve explicit real-API references with `FLIPPERCTRL_TEST_DOXYGEN_REF` blocks where harness indirection prevents links.
- `apps/flipperCtrl/flipperCtrl.cpp`: documentation-only cleanup of the placeholder main-program brief if included; retain its main-entrypoint-only structure.
- This plan: update implementation decisions and verification results as work progresses, preserving the prompt above `# plan`.

Perform the required full-file documentation pass on each changed C++ file, replacing placeholder app/file descriptions and documenting all declarations, parameters, members, and return semantics. Keep include guards/order, `m_` naming, declaration grouping, and exact `{ //mutex scope` annotations for lock-lifetime-only blocks. No new test registration or shared Doxygen group is needed.

Work is already on `jrmales/flipper-parking`. If commits are requested, keep functional changes and this engineering plan together, then documentation-only cleanup, then any formatting-only cleanup. Follow the required model/prompt attribution for any eventual PR description.

## Assumptions and limits to review

- A confirmed idle flipper stays physically at its endpoint without power. Disk recovery reports that retained observation; software cannot verify manual movement while the device or controller is off. Motion outside the app between polls is also not guaranteed to be observed before power loss, and external motion combined with storage failure can leave a stale record despite a diagnostic.
- The app configuration name continues to identify the same physical device. A device replacement or reassignment requires clearing its stored `position` file; automatic identity binding can be added separately if that workflow is needed.
- Power loss before a move is confirmed yields unknown, even if the mechanism actually reached an endpoint. Recovery requires a later live query; the target alone is insufficient evidence.
- Verify the installed device's status framing and completion indication before implementing the decoder. No local documentation resolves those details yet.
- The proposed policy rejects moves when the app cannot durably invalidate the backing record. Review this behavior alongside the unknown-state representation and new read-only parked property before implementation.

## Implementation record

- Implemented in the existing header-only app, with no changes to shared stage helpers or telemetry schemas. Position records contain the last confirmed physical endpoint and parked flag; writes use checked short-write/EINTR handling, file sync, atomic rename, and directory sync. File mode is explicitly `0644` so a record written with elevated privileges is readable on the next unprivileged startup.
- Motion commands own the INDI mutex and refuse to run until invalidation succeeds. The old endpoint cannot certify completion of an outstanding move. State-save failures retry after five seconds; unchanged records and powered-off loops produce no disk writes.
- Added `parked.current`, coherent in/out publication, OFF telemetry (`moving=-2`), unknown telemetry (`preset=0`, empty name), per-instance telemetry change tracking, and the power-off hooks. Cached snapshots cannot authorize moves until a live reply has been received.
- Added the requested `LOG_WARNING` message when the first settled power-on endpoint differs from the inference. The comparison survives a transitional status or initial read failure, is consumed once by a settled observation, and is renewed by a later power cycle. It does not warn for ordinary endpoint changes while powered on.
- Checked standard APT framing, full 32-bit status flags, motion masks, and unsolicited completion-message behavior against the [Thorlabs APT protocol, issue 15, pages 59 and 95–101](https://www.thorlabs.us/software/apt/APT_Communications_Protocol_Rev_15.pdf). Preserved the installed app's channel-zero requests and physical endpoint mapping; the decoder accepts channel zero or one. No endpoint switch active is treated conservatively as transit, and simultaneous endpoint switches are rejected. Installed-hardware validation remains outstanding.
- Replaced the construction-only test with fault-injected app/telemetry bases and serial/filesystem wrappers that exercise production controller code. The harness captures real log formats and serialized telemetry, and one test uses the real tty utilities over a local socket. Restart tests destroy the first app before constructing the second because the framework permits only one app instance at a time.
- Verified 13 test cases with 648 assertions. Coverage includes missing/corrupt state, both endpoints and reversal, app-name isolation, interrupted/completed moves, command guards, short/interrupted file writes, write/sync/close/rename failures, bounded retries, malformed/fragmented/coalesced status, live reconciliation warnings, OFF scheduling, and the real tty path.
- Commit messages use the requested final line `Co-authored by GPT-6.1 Sol Codex`. `AGENTS.md` is unchanged because that standing instruction already exists on a branch awaiting merge.

Status: functional implementation and behavioral tests complete; final documentation, formatting, and build verification in progress. Hardware validation remains a follow-up.
