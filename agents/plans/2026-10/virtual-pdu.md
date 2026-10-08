# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
Create a virtual power distribution unit (PDU) that combines channels from other PDUs into a single channel.  

## Discussion
Several devices require two power switches, usually one on a `trippLitePDU` or an `xt1121DCDU` and one on an `acronameUsbHub`.  One has to turn both on or both off to manage the device.  This results in the same device, e.g. `fwtelsim`, showing up in two disparate places on the power controller and otherwise causes operational headaches since the operator has to know and remember.

We will crate a virtual PDU (vPDU), that serves to combine such device power switches into a single switch that operates both.

This is in principle quite simple, as dev::outletController already has multi-outlet semantics, with ordering, timing, etc.  We should be able to derive from that and expose the necessary configuration front end, then manage the INDI client connections to the actual PDUs.

pwrGUI shouldn't need a change, as it's just a configuration change.

To clear a pending issue/to-do item, we will also take this opportunity to make dev::outletController a telemeter, and also add telemetry to 

## Requirements

### Reqs for vPDU:

1. Define and implement a standard MagAO-X configuration language for combining outlets from dev::outletController devices.

2. Implement a standard outletController INDI interface 

3. Must work with MagAO-X FSM power control logic

### Reqs for Telemetry:

1. Define a telemetry schema for outletController
  - handle arbitrary number of outlets
  - use a small integer to represent the states 

2. Define the telemetry type class with full support for messages and FITS headers, etc.

3. Update all existing outletController derived classes to support telemetry.

4. Define telemetry for derived classes as needed.  E.g. for voltage/current/frequency on trippLitePDUs.

5. Define the type class for any such base classes with full support for messages and FITS headers, etc.

## Tests and Metrics

Bring all outletController derived classes up to 100% line coverage.  Do not bother with outletController itself as that is done as separate effort waiting to be merged.

Testing should verify that the vPDU single outlet control works by at least verifying creation of proper INDI outgoing traffic.  It is not required that the behavior of the controlled/slaved devices be tested.

## Scope and Caveats

N/A.

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Execution status: the user reviewed this plan and answered all questions on 2026-10-07. GPT-6 (Codex) is proceeding with the decisions recorded below. Only the Agent Findings and Plan section is edited.

## Task Summary

Add a `virtualPDU` application that presents the normal `dev::outletController` interface while treating named channels on other outlet controllers as its individual outlets. A virtual channel such as `fwtelsim` can then operate both an AC/DC power channel and a USB-hub channel using the existing ordering and delay rules. Device applications can monitor that one virtual channel through their existing `[power]` configuration.

Add a common outlet-state telemetry record, integrate it into every existing outlet-controller application and the new application, and add Tripp Lite electrical telemetry. Replace the existing placeholder tests and add the missing USB-hub suite, targeting 100% executable line coverage of the four controller classes. Testing and execution will use workstation-only substitutes and private local transports.

### Repository findings

| Area | Current implementation | Consequence for this work |
| --- | --- | --- |
| Outlet-controller applications | `apps/trippLitePDU/trippLitePDU.hpp`, `apps/xt1121DCDU/xt1121DCDU.hpp`, and `apps/acronameUsbHub/acronameUsbHub.hpp` are the three existing derived apps. None inherits `dev::telemeter`. | All three need telemetry lifecycle integration; the new app is the fourth coverage target. |
| Channel configuration | `libMagAOX/app/dev/outletController.hpp` parses unused channel sections containing numeric `outlet`/`outlets`, with zero-based indices in `onOrder`/`offOrder` and millisecond delays. | Map numeric virtual outlets to remote channels, then reuse the base parser and sequence implementation. |
| State representation | The base uses `Unknown=-1`, `Off=0`, `Intermediate=1`, `On=2`; INDI exposes `Unk`, `Off`, `Int`, `On`. Mixed outlet states aggregate to `Int`. | Preserve these values and the existing aggregate-state semantics. Telemetry needs a signed byte to retain `-1`. |
| INDI interface | The base publishes each channel as Text `state`/`target`, plus `outlet`, `stateTimes`, `channelOutlets`, `channelOnDelays`, and `channelOffDelays`. `setupINDI()` is deprecated in favor of `appStartup()`. | Reuse that interface and migrate the two deprecated startup calls in touched apps. |
| Power FSM and GUI | `MagAOXApp` monitors Text `state`/`target`; `On` and `Off` are recognized, other values are unknown. `gui/widgets/pwr/pwrDevice.hpp` consumes the standard properties and numeric outlet lists. | No production power-FSM code change is needed. The user approved a GUI follow-up on 2026-10-08 to disable sliders for Unk or unrecognized states. Configuration must select the virtual device/channel and hide duplicate physical entries as desired. |
| INDI subscriptions | `registerIndiPropertySet()` registers stable property pointers keyed by device/property; Def and Set messages share the same callback path. Duplicate registrations fail. Existing subscription retries cover initial missing definitions. | Register each remote channel and each remote device's `fsm` once, keep storage stable, and reuse discovery/retry behavior. |
| Sequence completion | Base ordering/delays operate on successful calls to `turnOutletOn/Off`; sending an INDI command does not confirm the remote switch changed. New-property callbacks execute synchronously in the driver's dispatch path. | Define dispatch versus confirmed-state sequencing explicitly. Waiting for incoming confirmation inside the same callback would block its delivery. |
| Source disappearance | The MagAOX driver wrapper does not forward `delProperty` to apps; a received subscription definition does not establish ongoing liveness. | Do not use an unchanged channel value's age alone to infer disconnects. Periodic explicit refreshes can establish bounded freshness without changing the shared driver. |
| Telemetry reuse | Helpers such as `stdMotionStage` and `frameGrabber` record through the derived app's `telem()`; the app owns `dev::telemeter`, scheduling, and lifecycle calls. | Follow this pattern: the outlet helper supplies generic recording, while each app owns exactly one telemeter. |
| Logger/FITS integration | Schemas and log types are generated from `logger/logCodes.dat`; `telem.cpp` defines `lastRecord`. `logger/logMeta.cpp::verifyLogEntry()` has a separate explicit event-code switch. Integer/string vector metadata enums exist, but formatting currently implements only boolean and float vectors. | Update the verifier as well as registry/build dependencies. Use a String metadata accessor for comma-separated byte states instead of broadening vector support for this task. |
| Tests and coverage | Tripp Lite and DCDU have construction-only tests. USB hub has no registered test suite. `tests/Makefile.one`, `COVERAGE=1`, and `tests/coverage/update_coverage` provide the build/report path. | Coverage work includes lifecycle, configuration, transport failures, parsers, callbacks, and telemetry, rather than just the new virtual app. No baseline percentage has been measured in this planning pass. |
| Local dependencies | The BrainStem headers and static archive are present; `clang-format`, `gcov`, and `lcov` are available. Existing flipper tests demonstrate private FIFO drivers and dependency fault injection. | Prefer those established harness patterns. Availability does not yet establish that all required library/toolchain dependencies build successfully. |

The current outlet base's configuration checks do not fully validate outlet bounds or order permutations, and overlapping channels have independent channel mutexes. The virtual app must validate its configuration before invoking the base. Changes to the outlet helper are authorized here; PR #401 remains responsible for its coverage. Preserve existing aggregation and synchronization behavior unless a scoped correction is needed.

## Key Assumptions

- Proposed executable/class name: `virtualPDU`; INDI instance names remain configurable through the normal `-n` mechanism. One instance may expose multiple combined device channels.
- A virtual outlet references a remote **named channel**, not a raw physical outlet. This preserves the remote controller's own ordering, delays, and channel semantics. The initial protocol supports the standard Text `state`/`target` interface only.
- Virtual configuration uses contiguous one-based outlet numbers. The base still stores zero-based indices internally, and sequence order arrays remain zero-based. Physical PDU and USB-hub numbering conventions remain unchanged.
- Generic outlet telemetry describes observed states, not command acceptance. It records the first snapshot, changes, and forced interval snapshots, with caches owned by each app/helper instance rather than function-static caches.
- The outlet helper provides telemetry recording functions, while each app directly inherits `dev::telemeter<app>`, as established by other device helpers. This avoids introducing a second telemeter through helper inheritance.
- The derived-specific telemetry scope is Tripp Lite frequency, voltage, and total current. USB-hub temperature/current monitoring and DCDU electrical measurements would require additional acquisition work and are not assumed from the unfinished discussion sentence above.
- Real installation, instrument commands, production configuration migration, and hardware validation are user-run steps. This agent will not execute on instrument computers or use an INDI tunnel to them.
- The current feature branch is already `jrmales/virtual-pdu`, and the worktree was clean at the start of review.

## Requirements

1. Define and document the virtual outlet mapping language, with complete examples and strict local validation of mappings and sequence arrays.
2. Expose the existing outlet-controller INDI properties, and generate correct outgoing Text commands to the mapped device/channel with `target=On` or `target=Off`.
3. Derive observed virtual outlet states from incoming channel `state` values; never substitute the requested target for observed power. Preserve the normal MagAOXApp power-FSM interface.
4. Define handling for unavailable sources, partial updates, source restarts, send failures, sequence overlap, and shutdown. Never automatically roll back already issued power commands after a later failure.
5. Define a variable-length outlet-state telemetry schema with small integer states, complete message creation/verification/formatting, typed accessors, metadata/FITS accessors, event registration, and interval support.
6. Integrate common telemetry into Tripp Lite, DCDU, USB hub, and virtual PDU. Add Tripp Lite electrical telemetry with explicit sample validity.
7. Achieve 100% executable line coverage for each of the four controller classes, including new telemetry and error paths. Retain and run the outlet-base regression suite, without taking on its separate coverage effort.
8. Follow `AGENTS.md`: header-only app implementations with definitions outside classes, complete changed-file Doxygen passes, helper macros where available, application test groups and explicit real-symbol links, repository formatting, and separate functional/documentation/formatting commits when applicable.

## Questions and Points of Clarification

All questions below have been answered. Agreed decisions: work against this checkout without merging PR #401; retain command-dispatch ordering; base observed states only on received INDI state values; use per-channel availability and one-based mappings; add only the existing Tripp Lite sensor measurements. Shared endpoints are deferred as a known use-case.

1. **Pending outlet-controller work:** What branch/commit or PR contains that effort, and should it be integrated before this implementation? Recommendation: settle the base interface first; do not duplicate its fixes or tests. If implementation should proceed against this checkout instead, explicitly record that decision and reconcile when the other work lands.

Answer: This is under https://github.com/magao-x/MagAOX/pull/401.  That is confined exclusively to libMagAOX, and is focused only on test coverage.  It nominally brings libMagAOX up to 100% with fairly automated test generation.  So we do not need to put work into dev::outletController coverage, but it is fine to change dev::outletController here and we will update from that end to preserve coverage.

2. **Sequence semantics:** Should delays separate outgoing commands, or should each remote channel reach its requested state before proceeding? Recommendation for the initial version: command dispatch ordering with existing fixed delays, preserving the base's semantics. Confirmed-state sequencing would require asynchronous execution, per-step timeouts, and cancellation rules; it cannot block the INDI receive callback waiting for its own updates.

Answer: Delays separate outgoing commands, and we do not wait for outlets to change state.  As you note, this is how the outletController was designed to perform so it is consistent. I think it is captured, but to be explicit: while the commands do not wait for the state to change on each outlet before going on to the next command, the reported state must be based only on states received from INDI.


3. **Unavailable dependencies:** Is per-channel availability acceptable, so a missing source blocks only virtual channels that use it? Recommendation: keep the virtual app service ready, reject commands for unavailable channels before issuing any traffic, and publish affected observed states as unknown. Use source `fsm=READY` plus channel definitions and periodic refreshes, initially proposing a 5-second refresh and 15-second stale limit, both configurable. The alternative is to take the whole virtual PDU out of READY whenever any dependency is unavailable.

Answer: Yes per-channel.

4. **Telemetry scope and configuration:** Does the proposed one-based mapping syntax and outlet-state plus Tripp Lite electrical telemetry scope meet the intent? In particular, is any additional USB-hub/DCDU sensor telemetry intended by the incomplete sentence “also add telemetry to”? Recommendation: keep additional sensor acquisition outside this task.

Answer: Yes to one-based mapping. I did mean to add any sensor telemetry that is already in those other apps, though I don't know of any off the top of my head. xt1121DCDU is itself virtual so there shouldn't be anything. acronanmUsbHub also doesn't show anything.  So it's only trippLitePDU.

Other proposed limits for review: reject direct self-reference, duplicate endpoint aliases, and endpoints shared between virtual channels in the first version. Cross-instance dependency cycles cannot be detected from one config file and must be excluded in deployment configuration. These limits avoid adding a new shared sequencing framework while the outlet base is being revised.

Answer: Agree with recommendation.  But shared-endpoints is a already known use-case so keep it in to-do for later.

## Tests

### Harness and acceptance metric

- Use Catch2, `tests/testXWC.hpp`, and `libXWCTest::<appName>Test` namespaces; place each app group under `application_unit_test`. Document every test and preserve real-API Doxygen links with the app-specific `*_TEST_DOXYGEN_REF` blocks. Hide harness-only declarations from Doxygen.
- Reuse existing dependency substitution and fault-injection patterns. Execute real controller methods and real configuration parsing; substitute only hardware/transport/log-thread boundaries. Prefer test-local seams to new production abstractions.
- For virtual and DCDU outgoing traffic, capture a real INDI message through private FIFOs or a loopback-only test server, then parse/assert device, property, type, element, and value. A `MagAOXApp<false>` no-op send is not evidence of correct outgoing traffic. No connection to a production INDI server is permitted.
- Use deterministic fake time where freshness/sequence timing requires it, short bounded timing checks where exercising the existing base delays, RAII cleanup, and temporary directories. Construct MagAOXApp instances sequentially because it permits only one instance per process.
- Force recompilation after header, harness, or coverage flag changes: the single-test makefile does not track every header dependency. Use `COVERAGE=1` with the repository's normal `-O0` setting and reset only counters needed for the focused run.
- Merge initial `.gcno` coverage with executed `.gcda` coverage and extract the actual controller implementation paths. Report per-file executable lines hit/total and uncovered lines. Measure production and simulator conditionals separately and combine coverage of the same production source as appropriate; do not drop production branches or use exclusions/optimization to manufacture 100%.
- Acceptance is 100% executable line coverage of each derived controller's implementation, not inherited library code. Entrypoint smoke tests should verify the new app and any touched entrypoint locally; standalone simulator helper coverage is reported separately. Generated/vendor code is not a target.

### Virtual PDU

1. Configuration defaults and explicit mappings; one and many outlets/channels; zero-based sequence orders with one-based outlet numbers; malformed/empty mappings, gaps, invalid endpoint keys, reserved property names, self-reference, duplicate aliases, shared endpoints, invalid bounds, non-permutation orders, and mismatched delay lengths.
2. Startup creates every standard read-only metadata property and read/write channel property. Dynamic registrations have stable backing storage and unique device/property keys. Inject each relevant registration and telemetry lifecycle failure.
3. Correct routing for On/Off commands, `state` fallback, case normalization, no-op requests already at the desired observed state, invalid channel/target, and not-ready/unavailable rejection. Test both direct outlet methods and the real channel callback/delegation path.
4. Multi-source command order and delays, reversal for off, failure on the first or later send, no remaining sends after failure, and no rollback. Test the selected command-dispatch or confirmation contract explicitly.
5. Incoming Def/Set messages route by full device/property identity, including two devices with the same channel name. Handle unknown/intermediate states, target-only updates, irrelevant messages, missing elements, and wrong property types without manufacturing an observed state.
6. Partial refreshes merge without erasing previously valid elements; source FSM departure from READY invalidates its outlets. Exercise startup absence, refresh timeout, reconnection, repeated source definitions, recovery, and unaffected-channel operation under the approved readiness policy.
7. Aggregate state and target updates are accepted by the real MagAOXApp power callback: On maps to powered, Off to unpowered, and Int/Unk to unknown. Test the callback in a separate sequential consumer fixture and run the existing power-FSM regression tests.
8. Initial/change/forced telemetry, independent caches across sequential instances, lock contention, concurrent requests, and shutdown cleanup. A fake receiver may inject source reports; testing the actual slaved devices' behavior is not required.

### Tripp Lite PDU

- Real configuration registration/loading, all limits and protocol-version settings, startup registration failures, telemetry lifecycle, and shutdown.
- FSM paths for connection success/failure, login success/timeout/fatal failure, both PowerAlert versions, post-login behavior, status retries, lock contention, unexpected states, and recovery.
- Replace telnet I/O with a scripted local transport for the production path; additionally exercise the existing simulator. Assert one-based `loadctl` wire commands and error propagation.
- `devstatus` fixtures for voltage/frequency/current/outlet parsing, all-off and mixed outlets, whitespace and final-line cases, ignored lines, every parser error return, malformed short lines, invalid numeric data, and telemetry sample validity after a failed/partial response.
- Every warning/alert/emergency threshold, both low/high frequency and voltage, normal readings, and current thresholds, including boundary equality. Check actual emitted priorities/messages.

### xt1121DCDU

- Configuration defaults/overrides, exactly eight backing channels, invalid channel count, startup registration failures, deprecated-call migration, and telemetry lifecycle.
- POWERON/READY/unexpected-state logic, lock contention, update errors, and shutdown.
- All outlet/property and xt-channel-name mappings, including currently supported channel number 16, invalid indices, missing/invalid `current`, and all eight callbacks.
- Numeric INDI `target=1/0` traffic, failures, observed-state updates, and generic telemetry. Preserve existing mapping/protocol behavior unless a separately documented defect requires correction.

### Acroname USB hub

- Add a test-only BrainStem substitute matching the API used by the app; it must never enumerate or connect to USB hardware. Exercise the controller code rather than just the substitute.
- Configuration and serial number, startup failure injection, POWERON/disconnected/connected/READY transitions, connection failure/retry, dropped connection, model/version/serial reporting, and destructor disconnect.
- Every port-state read and enable/disable path, timeout and connection errors, other current error handling, power-off state/target resets, repeated powered-off hooks, shutdown, lock behavior, and generic telemetry.
- Keep the real app build using its existing BrainStem library. Add test-specific include/link rules only where necessary; add the app to `all_buildable_apps` so standard offline coverage workflows include its production source.

### Telemetry/logging/FITS

- Outlet messages with empty, one-element, mixed-state, and large vectors (including more than 255 outlets); retain `-1/0/1/2` exactly. Check numbering, typed accessors, numeric message strings, empty/unknown accessor results, and buffer verification failures.
- Electrical records with finite readings, sample-invalid records, units, accessor types, and all fields. Never present unmeasured startup zeroes or a failed partial parse as a valid electrical sample.
- Test event dispatch through the generated accessor registry and the real `verifyLogEntry()` path, then construct actual FITS header cards using `logMeta` with synthetic log timestamps. Verify outlet lists serialize as decimal numbers and use state selection rather than interpolation; electrical values follow their documented sample/validity policy.
- Test common telemetry initialization, change suppression, forced interval records, acquisition failure/invalidation, power-off hooks, and lifecycle error behavior. Run the existing logger accessor/metadata and outlet-controller regression suites after integration.

## Implementation Plan

### 1. Resolve approval and the shared-base dependency

The user identified https://github.com/magao-x/MagAOX/pull/401 as separate library coverage work and authorized changes to the outlet helper here. Continue against this checkout without merging that PR; its coverage will be reconciled from that effort. The remaining decisions are recorded above. Establish a local baseline for the focused suites and keep the work on the namespaced feature branch.

### 2. Define the virtual configuration and INDI contract

Proposed configuration, using illustrative device names:

```ini
[device]
pollInterval=5        # seconds between explicit source refreshes
staleTimeout=15       # seconds without a valid refresh before invalidation

[outlet1]
device=pdu0
channel=fwtelsim

[outlet2]
device=usbhub0
channel=fwtelsim

[fwtelsim]
outlets=1,2
onOrder=0,1           # command AC/DC power before the USB channel
offOrder=1,0          # command the USB channel off first
onDelays=0,500        # milliseconds; the first entry is ignored
offDelays=0,500
```

The app first consumes and validates `[device]` and `[outletN]` sections, allocates the contiguous outlet vector, sets `m_firstOne=true`, then delegates remaining channel sections to `outletController::loadConfig()`. Both `outlet` and `outlets` retain their existing base meaning. Validate references, order permutations, and delay vector lengths before the base can index them. Reject collisions with standard published property names. The physical controller configurations need no syntax changes.

A consumer app changes only its normal power configuration:

```ini
[power]
device=vpdu0
channel=fwtelsim
```

Document source-channel delays separately from virtual delays: virtual totals in the existing GUI metadata cannot include the remote devices' complete transition/boot time. Keep numeric `channelOutlets` for GUI compatibility; the mapping is documented in the virtual app configuration.

### 3. Add telemetry schemas and complete logger support

Proposed generic type `telem_outlet`:

- `first_outlet:uint8`: 0 or 1, matching the controller's numbering convention; it is not an outlet count.
- `states:[int8]`: arbitrary-length observed state vector in internal outlet order, values `-1`, `0`, `1`, `2`.

Provide a typed state-vector accessor for software use, a comma-separated decimal String metadata accessor for FITS, and an unsigned numbering accessor. Suggested HIERARCH keywords are `OUTLET STATES` and `OUTLET FIRST`; both use state metadata semantics. Outlet identity is `vector index + first_outlet`, and channel mappings are defined by the logged/documented controller configuration. Channel targets and names are not required in this generic snapshot.

Proposed derived type `telem_pdu` contains `frequency`, `voltage`, `current` floats in Hz/V/A and a `valid` boolean. Build a complete telemetry snapshot only after all required electrical fields of a status response have been parsed successfully, invalidate it on acquisition failure/disconnect, and distinguish unavailable startup data. Keep valid sample values independent of partially updated parser members. Use the existing per-field state metadata selection for FITS; values and validity are serialized together in each binary record. The existing FITS interval selector can choose each field's first changed value, so it does not imply a simultaneous grouped measurement. Do not interpolate across invalid samples, and include the validity field alongside electrical values.

Add the schema/type headers, unique event codes, `lastRecord` definitions, `libMagAOX/Makefile` dependencies, and explicit `verifyLogEntry()` cases. Regenerate through the existing build rather than editing generated files, which are ignored by git. Build the affected logger consumers (`logdump` and `xrif2fits`) locally and verify actual FITS cards in tests; new types need no hardcoded instrument header configuration in `xrif2fits`.

### 4. Integrate common recording and app lifecycle

Add generic recording to the outlet helper following the existing helper-to-derived `telem()` pattern. Store the previous snapshot and initial-record flag per instance. Keep recording independent of whether an INDI driver is attached; placing telemetry behind `updateINDI()`'s no-driver early return would suppress it in otherwise valid contexts.

Each of the four apps owns a single `dev::telemeter<app>` with the normal friend/type declarations, `checkRecordTimes()`, and `recordTelem(const telem_outlet *)`. Tripp Lite also schedules `telem_pdu`. Prefer `TELEMETER_*` lifecycle macros, and expose generic recording to derived apps without copying state serialization.

Wire recording after observed state updates and invalidation/power-off changes, plus forced periodic records. Ensure early-return FSM paths and `whilePowerOff()` do not inadvertently skip required snapshots. Snapshot state under the existing synchronization policy, release any lifetime-only lock scope before operations needing the same lock, and avoid holding the INDI mutex while waiting for external events. Keep legacy command/event logs and existing healthy-device behavior intact.

Replace `setupINDI()` in DCDU/USB hub with `outletController::appStartup()` and propagate configuration failures instead of ignoring them. Resolve helper/app overloads explicitly when adding the common telemetry hooks.

### 5. Implement the virtual app

Create the standard header-only app and entrypoint. Use a stable vector/map of remote endpoint properties and one dispatcher for dynamic source callbacks; register one `fsm` property per remote device. Reuse `MagAOXApp` for subscriptions, property publication, discovery retries, and sending commands rather than creating another INDI client framework.

Under the proposed dispatch policy, `turnOutletOn/Off()` builds a minimal Text property with the configured device/channel and a single `target` element. A successful return means the command was sent. Source callbacks alone update observed states; remote targets do not. Merge partial Def/Set values by element and validate types and state strings.

Under the proposed availability policy, a validated app is READY as a service, and its channel callback preflights all required endpoint availability before delegating sequencing to the base. Keep independent channels usable. Subscribe to source FSM changes, periodically request fresh channel/FSM definitions, and invalidate snapshots when refreshes fail or expire; use a monotonic clock and existing INDI get-property facilities. Check availability again immediately before each send, so a detected mid-sequence failure stops later traffic. Report the failure and retain the partial observed state without rollback or automatic replay.

Use the selected base's sequencing lock contract, reject conflicting endpoint sharing for this initial version, and publish normal aggregate states through `updateINDI()`. Do not mark a switch On/Off simply because outgoing traffic succeeded. Confirmed-state sequencing is outside the approved scope.

Register the app in the appropriate root build lists (proposed `apps_aoc`, `apps_sim`, and `all_buildable_apps`) and ignore its executable. Add local example configuration and application documentation. Deployment placement and actual remote device names remain installation choices for the user.

### 6. Complete controller tests and focused verification

Build the behavioral suites listed above, inventory uncovered lines by real function, and add meaningful failure cases until all four controller classes meet the metric. Exercise Tripp Lite production transport code as well as simulation; isolate BrainStem USB access completely. Keep any necessary production testing seams narrow and defaulting to the existing implementation.

Document discovered defects separately. For example, Tripp Lite's parser indexes fixed character positions without first checking short-line length; length checks/error handling may be needed for deterministic malformed-input tests. Preserve valid-device behavior, avoid unrelated parser/authentication/threshold changes, and coordinate generic outlet fixes with the other effort.

Run app builds, focused app suites, existing outlet/power/logger regression suites, and focused coverage. Run `clang-format` on all touched C++ files with the repository configuration, check the full changed-file documentation, and review `git diff --check`. Update this plan with actual commands, results, coverage totals, and remaining concerns.

### 7. Commit and hand off

Keep functional changes and their updated engineering notes in clean commits, follow with documentation-only changes, and use a separate final formatting-only commit if cleanup remains. Use brief messages with the required `Co-authored by  GPT-6` attribution. Provide the required copyable PR title and attributed Markdown description, summarize affected files and validation, and list user-run configuration/hardware follow-up. Do not deploy or operate instrument hardware.

### Expected implementation files

- New `apps/virtualPDU/{virtualPDU.hpp,virtualPDU.cpp,Makefile}`, example configuration, application documentation, and controller tests.
- `libMagAOX/app/dev/outletController.hpp` for common recording, documentation, and the approved derived contract; `outletController.cpp` only if related support genuinely requires it.
- The three existing controller headers; their tests, plus a new `apps/acronameUsbHub/tests/acronameUsbHub_test.cpp`. Touched entrypoints or simulator files receive the same complete documentation pass.
- New `libMagAOX/logger/types/{telem_outlet.hpp,telem_pdu.hpp}` and matching `types/schemas/*.fbs`; `logger/logCodes.dat`, `logger/types/telem.cpp`, `logger/logMeta.cpp`, `libMagAOX/Makefile`, and focused logger tests.
- Root `Makefile`, `.gitignore`, `tests/tests.list`, and `tests/Makefile.one` as required for new build/test dependencies. `tests/groups.dox` only if an additional group declaration is needed; `application_unit_test` already exists.
- This plan, plus `gui/widgets/pwr/{pwrChannel.hpp,pwrDevice.hpp}` and offline GUI tests for the subsequently approved unknown-state correction. No actual instrument configuration is present or edited by this plan.

## Follow-up and Edge Cases

- Coordinate new outlet-helper lines with the coverage effort in PR #401; merging that PR is not a prerequisite.
- TODO: support shared endpoints between virtual channels, an explicitly identified future use-case. Initial configuration rejects sharing to avoid ambiguous concurrent sequencing.
- Fixed delays specify outgoing command spacing. A remote channel may itself contain multiple outlets and delays, so dispatch success and virtual delay metadata cannot guarantee completion. A sequence that powers the controller responsible for its next step needs confirmation/wait semantics or a deployment design that keeps that controller available.

Comment: a vPDU powering another PDU is tricky, but is not envisioned.

- A source can disappear after preflight or during a delay. Refresh-based availability has a bounded detection interval and cannot make a distributed power operation atomic. Previously sent commands remain in effect; operators see actual reported partial/unknown states.
- Def/Set updates normally suppress unchanged values. Explicit refreshes, rather than age of unsolicited changes, establish freshness; malformed or target-only updates do not restore observed-state validity.
- Direct self-reference is rejected. Cross-instance cycles, chains of virtual PDUs, shared physical channels configured in separate instances, and external clients issuing conflicting commands require deployment review; initial configuration limits do not provide global exclusivity.
- Current mixed-known/unknown aggregation is preserved (`Int`), which the power FSM treats as unknown. Existing consumer behavior during unknown power remains unchanged and is verified through its tests rather than redesigned here.
- Removing duplicate buttons and changing consumers to virtual `[power]` channels are configuration rollout tasks. The user-approved GUI correction disables unknown-state sliders; GUI metadata still cannot show complete remote timing.
- Tripp Lite authentication cleanup, new USB-hub sensors, global integer-vector FITS support, outlet-base 100% coverage, and changes to warning thresholds are separate work unless specifically requested.
- FITS vectors use the existing string-card convention. Test large values and long-card handling; practical record/file limits still apply despite avoiding a fixed outlet count in the schema.
- The exact 100% coverage denominator and uncovered-line inventory will be recorded as the offline suites build. No instrument operations are performed by the agent.


## Execution Notes

- Implemented `virtualPDU` with the approved one-based mappings, dispatch-only ordering, per-channel readiness, and source refresh/expiry. Added the real power-consumer callback test and a known shared-endpoint TODO.
- Added signed-byte `telem_outlet` and complete-sample/validity `telem_pdu` schemas, accessors, message/verification support, FITS cards, event registration, and scheduling. Source state writes and aggregate reads now use a common outlet-state mutex.
- All three existing controllers own one telemeter, use its interface macros, and propagate app configuration failures. DCDU now invalidates observations and maintains periodic telemetry through its power-off hooks; USB-hub hooks retain their existing power-off semantics.
- Tripp Lite electrical telemetry retains only complete successful samples and invalidates failed/deferred/partial measurements. The parser now checks short lines and rejects malformed numeric values; valid protocol responses and warning thresholds are preserved.
- Renamed DCDU source-property members to `m_indiP_chN`, updating every callback and reference. The external numeric channel protocol is unchanged.
- Added shared, threadless test boundaries, private-FIFO INDI serialization/capture, scripted production telnet I/O, a BrainStem substitute that cannot enumerate USB, and production simulator coverage. No test uses an external server or hardware.
- The initial local build exposed missing unbuilt dependencies and a generated-schema/object race. Built the local flatlogs/INDI/telnet dependencies and added the library object dependency on completed generated schemas, so parallel compilation waits for generation.
- Focused suites passed before the final documentation/formatting pass: virtual PDU 8 cases, DCDU 4 cases, USB hub 4 cases, Tripp Lite production 6 cases, simulator 1 case, logger/FITS 3 cases. Pre-final coverage was 100% of virtual PDU 188/188, DCDU 195/195, USB hub 119/119, and Tripp Lite 359/359 executable lines, merging production/simulator paths and constructor/destructor aliases by source line. Final totals and regression/build results will be recorded after the power-off addition and formatting.

- Availability uses a READY source with a fresh valid state element, including the standard Unk value. Unk therefore permits command dispatch while preserving unknown observed power; missing, malformed, stale, and non-READY observations remain unavailable. This follows the base controller's existing ability to command unknown states. Added pre-startup/failed-startup bounds and FSM guards so no request indexes uninitialized source storage.

- Corrected the outlet helper's one-past-end bounds check and removed its signed-conversion overflow path. The derived config wrappers now reject those invalid physical configurations; added app-level rejection tests. The library coverage follow-up remains with PR #401.

- Virtual and DCDU callbacks now record observed changes immediately, so transitions between main-loop samples are not lost. Malformed channel/FSM definitions invalidate their corresponding availability immediately; valid partial target-only channel updates preserve the last observation.

- Source callbacks also publish observed INDI state changes immediately. This preserves fast power transitions for the consumer FSM instead of adding another main-loop sampling delay. Test command traffic still traverses real private FIFOs; non-command publications are captured in a transport substitute and parsed by the same production XML parser.

- Current focused coverage: virtual PDU 199/199, DCDU 188/188, USB hub 119/119, Tripp Lite 359/359 executable lines (100% each). Focused app/simulator/logger suites pass, including immediate publication and callback recording. The existing outlet-controller regression suite passes (591 assertions).
- Broader regression finding: the unchanged `libMagAOX/app/tests/MagAOXApp_test.cpp:716` expects `powerOnWait()==0` after missing required power configuration; unchanged `MagAOXApp.hpp` requests shutdown but retains the default 55. The full suite reports 11/12 cases passed. `git diff --exit-code` confirms both files are untouched. This existing library test/implementation mismatch is outside this feature and can be reconciled with the library coverage work.
- GUI finding and resolution: the pwr GUI ignored Text Unk state updates, and its slider treated unrecognized enum states as Off. On 2026-10-08 the user requested that both cases disable the slider. This correction is implemented and verified below; the virtual INDI state and consumer FSM continue to report unknown correctly.

- The existing `MagAOXAppExecute_test.cpp:464` also fails an injected appLogic-failure expectation (`-1` expected, `0` returned). Its source, harness, and MagAOXApp header are unchanged. This second library regression mismatch is recorded rather than folded into the outlet feature.
- Completed the full changed-file documentation pass, including source-state ownership, telemetry roles, inline parameter docs, and removal of author tags. DCDU macro-generated callback declarations/wrappers are expanded with identical names and behavior to document every parameter at the declaration site; registration macros remain in use. Added the app's Doxygen page and example configuration.

- All standalone controller builds succeeded: `make -C apps/virtualPDU`, `apps/trippLitePDU`, `apps/xt1121DCDU`, and `apps/acronameUsbHub`. `utils/logdump` and `utils/xrif2fits` also build with the new generated telemetry types. No installation or device execution was performed.
- Entrypoint smoke: `MAGAOX_PATH=/tmp/virtual-pdu-smoke apps/virtualPDU/virtualPDU -n virtual-pdu-smoke --help` prints the new refresh/expiry and telemetry options with isolated writable paths. This framework uses a nonzero help exit; the help output was checked directly.
- Logger accessor regression: 392 assertions passed. Logger metadata regression: 22 assertions passed. Focused suites after the documentation pass: virtual PDU 9 cases/13,768 assertions; DCDU 7/17,467; USB hub 5/166; Tripp Lite production 8/291; simulator 1/16; new logger/FITS 3/77. Coverage remains 100%: virtual PDU 199/199, DCDU 196/196, USB hub 119/119, Tripp Lite 359/359.

## GUI Unknown-State Correction (2026-10-08)

- User direction: "Ok I think in both cases (Unk and unrecognized) the GUI should just disable that slider."
- `pwrDevice` maps Text Unk and every unrecognized state string to `pwrChState::Unk`. `pwrChannel` disables its slider for Unk or any unrecognized enum value without moving it to Off or emitting target completion. The last displayed position is retained while disabled.
- Unknown observations cancel pending command timeouts and replace the saved state with Unk, so timeout callbacks cannot re-enable a channel from its previous known value. Disabled slider releases cannot send a command. Initial and disconnected channels wait for a recognized observation too.
- On, Off, and Int observations restore control subject to the existing pending-command/target wait. Unaffected channels remain usable, and target-only updates cannot restore an unknown channel.
- Added a separate Qt/Qwt Catch test project in `gui/widgets/pwr/tests` because the core test runner does not initialize QApplication. The tests use real widgets, timers, INDI properties, and Qt signals without opening an INDI connection. Before the fix, all four initial cases failed (58 failing assertions); after the fix, all five cases pass (146 assertions), including ordinary command completion.
- Local test build: from `gui/widgets/pwr/tests`, run `qmake pwr_test.pro`, `make -f makefile.pwr_test -j4`, then `QT_QPA_PLATFORM=offscreen ./pwr_test`. The production `pwrGUI` also builds successfully. This workstation's Qwt is built in `/home/jrmales/Source/libs/qwt-6.3.0`; qmake was supplied that tree's `src` include directory, `lib` library directory, `-lqwt`, and `QMAKE_RPATHDIR` rather than installing anything.
- Existing header-only widget layout is preserved. Non-trivial circular-buffer definitions are moved below their declarations without changing their bodies, as required by AGENTS.md. The full-file documentation pass covers declarations, parameters, state/ownership, slots, and signals. The app's Doxygen page now describes GUI unknown-state behavior too. Documentation and final formatting are separated from the functional commit.
- Final GUI verification after formatting: 146 assertions in five cases pass; the production GUI rebuild succeeds. `clang-format --dry-run --Werror`, full-file documentation checks, preservation of the original task text, and `git diff --check` pass. The GUI build reports an existing ignored-QFile-open-result warning in untouched `gui/widgets/xWidgets/app.hpp:120`; no GUI connection or instrument operation was performed.

## Configuration Diagnostics Correction (2026-10-08)

- User reported that copied channel sections both containing `outlets=1,2` failed configuration without explaining the cause. Every virtual-PDU validation rejection now reports its failed condition and the relevant section, keyword, value, or valid range.
- Shared-outlet messages name both channels and the one-based virtual outlet, with channel names sorted so diagnostics do not depend on unordered-map iteration. Repeated outlets within one channel have a distinct explanation. Endpoint aliases name both `[outletN]` sections and the duplicated remote device/channel.
- Added diagnostics for poll/stale interval constraints, invalid or missing mapping sections, missing device/channel fields, self-reference, reserved source/property names, malformed/empty numeric values, outlet bounds, delay overflow, and invalid order permutations. The public loader preserves the specific diagnostic rather than appending its former generic failure banner.
- The shared outlet helper now logs its previously silent no-outlets/no-channel error codes, retaining those return codes. Its bounds messages show the configured range, and all four order/delay length checks show actual and expected entry counts. Existing numeric/base bounds validation continues to protect virtual channel assignment; the later ownership scan tracks channel names to explain conflicts.
- Extended the malformed-configuration table to 41 cases that assert the actual emitted diagnostic, and added a public-loader regression for two copied `outlets=1,2` channels. All 10 virtual-PDU cases pass (13,977 assertions). The existing DCDU (17,467), USB hub (166), Tripp Lite production (291)/simulator (16), and outlet-helper (591) assertions pass.
- Final controller line coverage after formatting remains 100%: virtualPDU 278/278, DCDU 215/215, USB hub 128/128, Tripp Lite production/simulator union 494/494. The virtual suite still passes 13,977 assertions in 10 cases. Full-file documentation checks, `clang-format --dry-run --Werror`, `git diff --check`, and preservation of the original task text pass.
- All four standalone applications rebuild successfully after the final formatting pass using sequential `-j1` builds. Functionality, documentation, and formatting are kept in separate feature-branch commits.
- The existing standalone Makefiles allow both cross-app version-header writer collisions and object compilation to race `magaox_git_version.h` generation under `-j4`. Regenerated the ignored header and used sequential, single-job (`-j1`) app builds for final validation. This diagnostics change does not alter the build scripts.
- No shared-endpoint support or configuration acceptance rules are added. Workstation-only fixtures continue to use private local transports and hardware substitutes.

## Zaber Power-Target Error Suppression (2026-10-08)

- User approved adding the ASCII `zaberLowLevel` correction to this feature because combined virtual-PDU actions increase the interval between an Off target and observed Off state. Turning off the monitored `pdu2.stagezaber` first still produced communication errors, so the subscription mismatch does not explain this case.
- Offline review reproduced a drain failure reporting an error and setting ERROR even with observed On and target Off already received. Discovery logged before its caller's power check; connection paths and parent command callbacks also reported expected failures, including failures whose stage helper had deliberately suppressed its own log.
- Added two local predicates: explicit observed/target Off pauses new communication, while only a fully On observed/target pair treats a transport failure as unexpected. Existing conservative unknown-state error handling is retained, but an unknown initial target does not block initial connection. The shared MagAOXApp observed state, target, power FSM, and boot-delay semantics remain separate.
- Connection, discovery, and polling stop starting work for an Off target. Failing drain/send/read/connect phases recheck power before logging or entering ERROR. Disconnect cleanup and missing-stage discovery diagnostics use the same expected-power policy. All seven stage command callbacks reject new requests during known power-off and avoid reporting a stage failure again when power is no longer expected On. Unexpected On/On failures retain their diagnostics, return codes, and recovery behavior.
- Added `zaberLowLevel_power_test.cpp`, registered in `tests/tests.list`. It uses the existing captured-log/isolated-directory harness, real power callbacks and stage command methods, and scripted connect/disconnect/drain/send/receive functions. No fake can open or close a real serial descriptor. One-shot target-only Off callbacks are injected immediately before failures return, preserving observed On.
- The new suite has six cases and 315 assertions, including dispatch through the actual registered callbacks, covering already-received Off, each connection/discovery transport failure phase, all seven parent command callbacks, cleanup, On/On controls, and initial unknown-target connection. The initial 294-assertion version against the pre-fix header fails five cases/33 assertions; the corrected suite, now also checking registered callback dispatch, passes all 315 assertions. The existing Zaber controller (55 assertions/5 cases), stage helper (144/2), and parser (8/2) regressions pass, as does the virtual-PDU power callback regression (42/1). The standalone zaberLowLevel app builds with `-j1`.
- Existing Zaber controller regressions could not compile because two const harness methods called non-const production accessors. Corrected those harness qualifiers without changing production APIs. Its power-off snapshot fixture also called appStartup while still UNINITIALIZED; the fixture now initializes the FSM as the real framework does before startup. The snapshot assertion now uses the Switch-state accessor instead of comparing its stored numeric enum to Text Off, and the fixture creates its FSM property before attaching a driver. Existing harness definitions are moved below their declarations for the required full-file documentation pass. Shared version-header generation also requires sequential test builds, as with app builds.
- Completed the full changed-file API/documentation pass: removed author tags, documented declaration parameters and callback wrappers, kept non-trivial test-harness definitions outside their class declaration, and preserved explicit Doxygen references for real methods under test. The virtual-PDU app page describes Zaber target-aware behavior and the need to command the configured virtual channel.
- Final verification after formatting: all 522 assertions in 15 Zaber cases pass, and the standalone zaberLowLevel rebuild succeeds. The virtual-PDU power callback regression passes all 42 assertions; coverage for all four outlet-controller apps remains 100%, with the same line totals reported below. Formatting dry runs, full-file documentation checks, preservation of the original task text, and `git diff --check` pass. No installation or instrument operation was performed.
- This correction is scoped to ASCII Zaber reporting and command gating. The broader review's MagAOXApp power-field synchronization issue and reversed siglentSDG conditions are separate follow-ups. Installation configuration is not changed. Applications following a virtual power channel receive its early target when the request is sent to that virtual channel; changing an underlying physical channel directly does not change the virtual target.

## Affected Files

- `.gitignore`
- `Makefile`
- `agents/plans/2026-10/virtual-pdu.md`
- `apps/acronameUsbHub/acronameUsbHub.hpp`
- `apps/acronameUsbHub/tests/acronameUsbHub_test.cpp`
- `apps/trippLitePDU/tests/trippLitePDU_sim_test.cpp`
- `apps/trippLitePDU/tests/trippLitePDU_test.cpp`
- `apps/trippLitePDU/trippLitePDU.hpp`
- `apps/virtualPDU/Makefile`
- `apps/virtualPDU/config/example.conf`
- `apps/virtualPDU/doc/virtualPDU.dox`
- `apps/virtualPDU/tests/virtualPDU_test.cpp`
- `apps/virtualPDU/virtualPDU.cpp`
- `apps/virtualPDU/virtualPDU.hpp`
- `apps/xt1121DCDU/tests/xt1121DCDU_test.cpp`
- `apps/xt1121DCDU/xt1121DCDU.hpp`
- `apps/zaberLowLevel/zaberLowLevel.hpp`
- `apps/zaberLowLevel/tests/zaberLowLevel_power_test.cpp`
- `apps/zaberLowLevel/tests/zaberLowLevel_test.cpp`
- `gui/widgets/pwr/pwrChannel.hpp`
- `gui/widgets/pwr/pwrDevice.hpp`
- `gui/widgets/pwr/tests/pwr_test.cpp`
- `gui/widgets/pwr/tests/pwr_test.pro`
- `libMagAOX/Makefile`
- `libMagAOX/app/dev/outletController.hpp`
- `libMagAOX/logger/logCodes.dat`
- `libMagAOX/logger/logMeta.cpp`
- `libMagAOX/logger/tests/pduTelemetry_test.cpp`
- `libMagAOX/logger/types/schemas/telem_outlet.fbs`
- `libMagAOX/logger/types/schemas/telem_pdu.fbs`
- `libMagAOX/logger/types/telem.cpp`
- `libMagAOX/logger/types/telem_outlet.hpp`
- `libMagAOX/logger/types/telem_pdu.hpp`
- `tests/Makefile.one`
- `tests/outletAppTest.hpp`
- `tests/tests.list`


## Final Verification

- Ran repository `clang-format` on all 23 changed C++ files, including the subsequent GUI and Zaber corrections; dry-run formatting, top file/brief/no-author checks, and `git diff --check` pass. The original task text above Agent Findings and Plan remains byte-for-byte unchanged.
- Initial formatting split the empty-numeric-token guard onto its own executable line, exposing a missing error case. Added malformed order/delay arrays with empty CSV tokens and reran the virtual suite: 9 cases/13,774 assertions pass. This validates rejection of a real malformed configuration rather than relying on multiple branches sharing one coverage line.
- Final standard `COVERAGE=1`/`-O0` controller line coverage, merged by source line across production/simulator and compiler aliases:

| Controller | Executable lines hit/total | Line coverage |
| --- | --- | --- |
| virtualPDU | 278/278 | 100% |
| xt1121DCDU | 215/215 | 100% |
| acronameUsbHub | 128/128 | 100% |
| trippLitePDU | 494/494 | 100% |

- Final focused app/simulator/FITS suites all pass. The outlet-controller regression (591 assertions), logger accessor regression (392), and logger metadata regression (22) pass. All four apps and logdump/xrif2fits rebuild after formatting. The isolated entrypoint help check prints every new option without filesystem errors.
- Two unchanged library regressions remain documented above: the missing-power-config wait expectation and injected appLogic-failure return expectation. Shared endpoints remain a follow-up item. The GUI Unk display issue is resolved by the user-approved correction described above; actual installation configuration was not changed.
- Local feature-branch commits separate functionality, documentation, the final coverage case/engineering record, and formatting. No network server, device, instrument command, installation, or deployment was used for validation.
