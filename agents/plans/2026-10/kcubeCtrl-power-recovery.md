# kcubeCtrl / ttmperi power-transition and recovery analysis

Date: 2026-10-17. Status: implemented on `jrmales/kcube-power-recovery`; offline validation complete, operator validation pending.

Prompt: `ttmperi` occasionally shuts down after a USB error and sometimes crashes
on power-on. Analyze the failures, consider recent `filterWheelCtrl` and other
power-target handling work, and document findings in a new plan.

## Scope and evidence

- Final baseline: local `dev`, `8c652ec5b3ef43db0cbdbc3833c5c8238ebf6495`.
  The user switched to `dev` during investigation. `kcubeCtrl.hpp` is identical
  between that baseline and the initially selected `jrmales/dm-poke-sync` branch.
- Reviewed the application, the base power FSM and callback dispatch, outlet
  target publication, filter-wheel recovery, and ASCII/binary Zaber power tests.
- The local workstation configuration `/opt/MagAOX/config/ttmperi.conf` uses
  `pdu0.ttmperi` and two axis serial numbers. Local `pdu0.conf` maps that channel
  to outlet 7. These are local configuration copies, not verification of the
  incident configuration on exao2.
- The local workstation does not have the app's expected
  `/opt/MagAOX/source/tmcController/` dependency. Reviewed upstream
  [tmcController.hpp at 6ba83aef](https://github.com/jaredmales/tmcController/blob/6ba83aefadaf3ac2120e45ad7425aa6bce3aff80/tmcController.hpp)
  through GitHub instead; its blob is `060c3a3a64aa04ee58caaf7420ac5d1ac7b93b56`.
  Its lines 1556 and 1050 match both dependency locations in the supplied log.
  This supports the comparison but does not establish the deployed revision.
- The initial investigation was source analysis only. The implementation record
  below describes subsequent local builds and offline tests. No production app,
  instrument command, or hardware connection was executed. The reported memory
  crash has not been reproduced.

## Findings at the original baseline

### F1. The supplied log matches a deterministic fatal return on power-off

The sequence on 2026-10-15 is:

1. `rest()` finishes at 22:40:36.081452518 after disabling both piezos.
2. At 22:40:57.090616050 a voltage poll fails in
   `tmcController::pz_req_outputvolts` with `usb bulk write failed`.
3. At 22:40:58.090697423 the base application logs `error from appLogic()`.
4. Two controller-close diagnostics follow during teardown.

The approximately 1.000081-second interval between the two errors is consistent
with the explicit `sleep(1)` in each voltage-poll error branch:

- `apps/kcubeCtrl/kcubeCtrl.hpp:430-438` and `:462-470` read the voltage,
  sleep on error, then execute `if(m_powerState == 0) return -1;`.
- `libMagAOX/app/MagAOXApp.hpp:2054-2058` treats any negative `appLogic()`
  return as an instruction to set `m_shutdown = 1`.
- If power still appears On after the sleep, those branches instead log an
  axis-specific error, enter `ERROR`, and return zero. That extra error is absent
  from the supplied sequence.

Thus the strongest explanation is that power changed to Off during the failed
poll or its one-second grace period, and the app deliberately terminated. The
power-off check has the wrong outcome for a recoverable device lifecycle event.
The same fatal-on-Off pattern exists in both `connect()` failure branches
(`:371-397`). A failure is not required when power-off is noticed between polls:
the base FSM can take its ordinary `POWEROFF` path first. This timing difference
explains why shutdown is intermittent.

The sample establishes the fatal-return path much more strongly than a memory
crash. It does not identify the underlying USB fault, or prove who switched power.
`rested` means the piezos were disabled; it does not mean the PDU was off, and
`READY` deliberately continues polling disabled axes. The approximately
21-second delay after `rested` is not a timer in `rest()`.

### F2. Power targets are received but never used by kcubeCtrl

`MagAOXApp::setCallBack_m_indiP_powerChannel()` updates actual and target power
independently (`libMagAOX/app/MagAOXApp.hpp:4079-4119`). A target-only message
preserves the previous actual state. `outletController::turnChannelOff()`
publishes `target=Off` before the outlet operation
(`libMagAOX/app/dev/outletController.hpp:729-765`).

The base main loop gates `appLogic()` on actual power only. It intentionally
does not prevent an application from running during `state=On,target=Off`.
`kcubeCtrl` never calls `powerStateTarget()` and checks the raw actual-state
member only after errors. Consequently it can continue opening, connecting,
initializing, polling, enabling, or setting voltage while shutdown is requested.

Replacing only the fatal return in F1 would keep the process alive but would not
prevent new work during this interval. Checking the target only after errors
would also miss successful operations that complete during a transition.

### F3. Startup communication errors are fatal, with no boot-readiness wait

In `CONNECTED`, either `axis1Initialize()` or `axis2Initialize()` returning
negative causes `appLogic()` to return negative (`kcubeCtrl.hpp:403-417`). Each
initialization contains multiple queries and writes; any transient failure is
therefore enough to terminate the process, even with actual and target power On.
Open/connect success is not proof that the piezo firmware can complete this
entire setup sequence.

`kcubeCtrl` does not call `powerOnWaitElapsed()`. The base exposes
`power.powerOnWait` but does not enforce it automatically. Its default is 55 s;
blindly adding that wait would change startup latency substantially. A suitable
KCUBE delay needs an explicit choice, plus recovery from transient initialization
failure regardless of the chosen delay.

This is a confirmed explanation for additional orderly power-on exits. A
segmentation fault or abort during power-on still requires separate evidence.

### F4. Connection ownership and cleanup are incomplete

`kcubeCtrl` has no `onPowerOff()` or `whilePowerOff()` override, and its
`appShutdown()` does nothing. The base implementations of the power-off hooks
also do nothing. Handles, `m_axis1Enabled`, `m_axis2Enabled`, `m_isSet`, and
published property state are not reset at power-off.

The `POWERON`/`NODEVICE`/`ERROR` block unconditionally calls `open(false)` on both
controllers (`kcubeCtrl.hpp:286-358`). It does not close a previously opened axis
before retrying or release the successful axis when only its peer is missing.
One cube enumerating before the other is enough to exercise this path. A normal
On/On poll error likewise enters `ERROR` and reaches repeated open next cycle.

The reviewed dependency makes this significant:

- `tmcController::open()` (`:998-1017`) calls the FTDI open function every time;
  it has no already-open guard. Failure clears `m_opened` but does not clear
  `m_connected`.
- `connect()` (`:1066-1175`) sets `m_connected` on success but does not clear an
  earlier true value when a later connection attempt fails.
- `close()` (`:1042-1063`) returns immediately on a close error, before clearing
  either flag. It also returns immediately when `m_opened` is false.
- Command methods use `TMCC_CHECK_CONNECTED` (`:1235-1247`), which trusts
  `m_connected` and may reconnect implicitly if false. App-level checks must
  account for that hidden transport activity.

In [libftdi 1.5's source](https://www.intra2net.com/en/developer/libftdi/documentation/ftdi_8c_source.html#l00764),
serial-number discovery assigns temporary handles into `ftdi->usb_dev`.
Reopening an occupied context can lose its old handle. Its
[`ftdi_usb_close()` implementation](https://www.intra2net.com/en/developer/libftdi/documentation/ftdi_8c_source.html#l01197)
closes and nulls the handle even when interface release returns an error.
`tmcController` can therefore retain true flags after the underlying handle is
gone. The deployed libftdi version must be checked before relying on these details.

The trailing close messages are consistent with cleanup after USB disappearance,
not evidence that close initiated the shutdown. The dependency destructor calls
`close()` for each axis. Virtual dispatch during that base destructor uses the
base stderr reporter, explaining the different output format. libftdi's close
path does not replace the previous error string on interface-release failure,
so `[-1:all fine]` and the repeated earlier write-error text can be stale
diagnostics rather than descriptions of the close failure.

These are concrete lifecycle defects and plausible contributors to later crashes;
they do not establish a particular invalid-pointer crash in the reported run.

### F5. Callback checks and reconnect operations can race

Most command callbacks test `READY`/`OPERATING` before acquiring `m_indiMutex`
(`kcubeCtrl.hpp:1011-1183`). They do not revalidate state or power after acquiring
it. Polls, connect, and initialization hold that mutex, but `open()` does not.
The dependency explicitly requires callers to serialize access to each instance.

A callback can pass its state check, wait behind a poll, then execute after the
poll has entered `ERROR`. The next FSM iteration can reopen the same controller
without the lock while that callback uses its FTDI state. This is a credible
memory-crash candidate and needs deterministic interleaving tests and a backtrace.

Power callbacks do not acquire `m_indiMutex`, so holding it in the FSM does not
prevent an independently received power update. However, command callbacks and
power updates share INDI receive dispatch. Sleeping inside a command callback
cannot make that receive thread process its next power message. The repeated
one-second sleeps in helper methods and `set()`/`rest()` can delay those updates
and compound delays while holding the mutex.

The base actual/target members are ordinary integers shared with receive code.
That pre-existing synchronization issue should be recorded when designing a
power snapshot or interruption latch; it is not a confirmed cause of this crash.
Avoid claiming that an atomic interruption flag alone synchronizes those fields.

### F6. Dependency I/O can stall recovery

`TMCC_READ_RESPONSE` (`tmcController.hpp:1345-1374`) repeatedly reads until the
expected byte count arrives, without an overall deadline. Repeated zero-byte
reads can keep the application inside the call indefinitely. A per-call USB
timeout does not bound that loop. This predicts a hang, not necessarily a crash,
and means app-level pre/post power checks alone cannot guarantee timely recovery.

The write macros accept nonnegative short writes, the command macro ignores a
flush error, and fixed-size replies are decoded without validating the complete
APT message identity. These are additional failure-handling gaps to test at the
dependency boundary, not established explanations of the sample.

### F7. Tests and diagnostics do not cover these paths

- `apps/kcubeCtrl/tests/kcubeCtrl_test.cpp` is a stale validator stub referring
  to `m_indiP_identify` and its callback, neither of which exists in this app.
  The real properties are `axis1_identify` and `axis2_identify`. It does not
  exercise the FSM or transport and is absent from `tests/tests.list`.
- Both voltage callbacks ignore `indiTargetUpdate()`'s result and can use an
  uninitialized local target when the message lacks `target` and `current`.
  Validate finite numeric input as well. This is an independent defect, not a
  demonstrated power-on crash trigger.
- The axis-2 open error prints `rv1` (`:353`), and its voltage-poll error says
  axis1 (`:468`). Dependency errors omit axis identity altogether.
- `handleNewProperty()` ignores callback return codes
  (`MagAOXApp.hpp:3594-3617`). Returning negative from a command callback is not
  the same shutdown path as returning negative from `appLogic()`; callback
  failures need an explicit recovery-state transition.

## Existing solutions to reuse

| Reference | Applicable behavior |
| --- | --- |
| `0be9b679`, filter-wheel communication recovery | Central connection reset; recoverable I/O failures keep the app alive; lock transport access; gate commands and reconnects on actual and target power; explicit power-off and shutdown cleanup. |
| `a2b55547`, filter-wheel READY recovery | Replace the already-registered base power callback with a wrapper that still invokes base handling; latch interruptions/generations so an Off-to-On update between FSM iterations is remembered. Reuse the pattern, not wheel-specific homing or retained-position policy. |
| `ee656700`, ASCII Zaber power handling | Separate known power-off from confirmed On/On; suppress expected shutdown communication errors while retaining genuine powered error reporting. |
| `4fc13f6a`, binary Zaber power handling | Apply those distinctions in lower-level helpers and callbacks, with injected target-only updates during transport failure. |
| `tests/outletAppTest.hpp` and the filter-wheel/Zaber power tests | Reuse isolated application state, private local INDI transport, captured logs, and scripted transport boundaries. |

Do not globally change the base FSM to treat target Off as actual Off as part of
this fix. Applications have different cleanup semantics, and the existing fixes
handle this at the app/transport boundaries. Small common power predicates or a
power-interruption helper may merit a separate libMagAOX proposal after the KCUBE
requirements are tested; a new general controller base class is unnecessary here.

## Proposed behavior

1. Start a hardware operation only with confirmed actual and target On, a valid
   connection for that operation, and the appropriate FSM state. Use the public
   power accessors so disabled power management retains its defined behavior.
2. Known actual or target Off cancels the active sequence, invalidates old
   connection/command state, and returns nonfatally from `appLogic()`. Unknown
   power also prevents new I/O but must not be presented as a confirmed Off.
3. An On/On communication or initialization error logs the first useful failure,
   releases both connections, and retries discovery at a bounded rate. Persistent
   absence, one missing axis, and late firmware readiness do not terminate the app.
4. Recover through full initialization of both axes. Preserve existing startup
   semantics: both piezos disabled and `m_isSet` false before READY. Never replay
   an interrupted `set`, enable, or voltage command, or automatically apply 75 V.
   A transport reset cannot prove that an already accepted hardware command was
   undone; expose uncertain status until fresh initialization/readback succeeds.
5. Power-off and shutdown cleanup are idempotent and send no piezo commands to an
   unavailable controller. Resource-release failure must not preserve a usable
   connection flag or turn an expected power-off into application termination.

## Implementation plan

### 1. Establish offline regression coverage before changing behavior

Replace the stale validator test with real callback and FSM coverage using the
existing isolated harness. Script both FTDI contexts independently and fail any
attempt to use real USB. Prefer exercising the real `tmcController` against a
stubbed FTDI boundary where its flags and retry behavior matter; a mock that
always makes `open()` and `close()` safe would hide F4. Pin the reviewed dependency
revision in the test/build setup instead of silently testing a substitute API.

### 2. Make transport ownership explicit

Add small shared reset/recovery helpers used by discovery failure, connect/setup
failure, poll failure, command failure, power-off, and shutdown. Hold the same
mutex for open/connect/I/O/close. Recheck state, power, and connection validity
after locking. On a partial two-axis attempt, unwind both axes before retrying.
Set a noncommandable state before releasing handles during shutdown, since the
base stops INDI after calling `appShutdown()`.

Address dependency flag cleanup and safe reopening in `tmcController`, preferably
upstream with tests. If a local `tmcCon` adapter is needed first, make its cleanup
contract explicit and verify it against the deployed library. Do not reset an
owning controller by assignment from a temporary: it owns a raw FTDI pointer and
does not define safe copy semantics. Preserve serial/configuration on any context
recreation and handle allocation failure.

### 3. Handle power transitions before and after each operation

Introduce concise app predicates for known Off and confirmed On/On. Gate the FSM,
all seven command callbacks, and multi-operation initialization/`set()`/`rest()`
sequences. A helper must still report cancellation/failure to its caller so the
caller stops the sequence; convert recoverable results to zero at the FSM
boundary. Do not mechanically replace every `return -1` with `return 0`.

Use the existing filter-wheel callback replacement pattern if latching transitions:
retain base partial-update semantics and record an interruption without waiting
for the hardware mutex. Consume the latch under the app lock, including a rapid
Off/On pair with no intervening FSM iteration. Define synchronized observation of
power and interruption state explicitly; avoid introducing a callback lock cycle.

### 4. Recover initialization and constrain blocking

Honor a documented boot-readiness policy using the existing configurable wait
where appropriate; establish its KCUBE default from observed startup behavior.
Keep retries nonfatal and rate limited even after the delay expires. Stop between
axes and between setup operations when power changes.

Bound dependency read completion with a monotonic deadline and correctly handle
short writes, incomplete replies, and flush failures. Implement this in the
existing dependency where possible, rather than duplicating the APT protocol in
the app. A new thread around an unbounded read is not an adequate cancellation
solution. Retain finite individual USB timeouts as part of the overall bound.

### 5. Publish truthful status and useful errors

Clear stale enable/set/request state on interruption and mark voltage readbacks
idle or invalid according to existing INDI conventions; do not publish a newly
measured zero when no read occurred. Validate callbacks before issuing I/O.

Put axis/serial, operation, return code, FSM state, and actual/target power in
recovery diagnostics. Preserve the underlying error before close/reopen can
replace it. `tmcCon` currently logs errors before app-level power classification:
use the dependency's error-control parameter or an owner-aware reporter so known
power-off failures can be quiet without hiding genuine On/On faults. Remove the
one-second sleep-as-classification pattern once transitions are handled directly.

### 6. Documentation, verification, and commits

Keep the app's header-only pattern. Apply the full changed-file documentation pass,
inline parameter documentation, mutex-scope comments, and application test/Doxygen
link conventions. Register the suite in `tests/tests.list` and its group under
`application_unit_test`. Run `clang-format` on changed C++/Doxygen files and
`git diff --check`.

Build the production app and run focused offline tests, then sanitizer checks for
the ownership and concurrency cases where supported. Keep this plan current.
Before any implementation commit, use a feature branch such as
`jrmales/kcube-power-recovery`; do not commit directly to `dev`. Separate functional,
documentation, and formatting changes as required by AGENTS.md.

## Required regression cases

| Scenario | Required result |
| --- | --- |
| Supplied failure: voltage write fails and actual power becomes Off before return | `appLogic()` returns zero; no later axis query or shutdown request; cleanup is safe. Test both axes. |
| Target-only Off while actual remains On | No new open/connect/setup/poll/command work, including READY and OPERATING; base actual state remains unchanged. |
| Off arrives during successful or failed transport | Stop before the next operation and do not publish stale success; distinguish expected Off from a genuine On/On failure. |
| Rapid Off/On between FSM iterations | Old handles and command state are invalidated; a fresh initialization is required. |
| Startup with one axis absent or appearing late | Surviving axis is released or deliberately retained without repeated open; bounded retry and eventual recovery. Test both enumeration orders. |
| Every connect/setup operation fails once, then succeeds | Process stays alive; both axes reinitialize; READY appears only after success. No automatic `set()`. |
| Close returns an error after releasing its handle | No stale connection flags, double ownership, unsafe retry, or destructor crash. |
| Callback passes its initial state check while a poll fails | Revalidation after locking prevents command/reopen overlap; exercise shutdown ordering as well. |
| Partial `set()` or `rest()` failure | Remaining steps are not run, success is not published, and the interrupted command is not replayed on reconnect. |
| Unknown/partial power messages, repeated On, unmanaged power | Defined gating and partial-update semantics; harmless repeated updates do not cause reconnect loops. |
| Zero-byte reads, short writes, truncated/unexpected replies | Bounded completion and coherent recovery without corrupting last valid readbacks. |
| Invalid/missing/nonfinite voltage and missing switch elements | Rejected without transport activity or uninitialized values; actual seven callbacks are covered. |

## Follow-up evidence and open decisions

The analysis supports fixing F1–F5 without waiting for a crash dump. Determining
which defect causes the reported memory crash still needs the following evidence,
collected by a human operator on the instrument and brought back for local review:

- Exact termination type: normal exit, SIGSEGV, SIGABRT, or hung process, plus a
  core backtrace with all threads for an actual crash.
- Whether the failure occurs on first app start with already powered cubes or
  only on an off/on cycle while the app remains running.
- Deployed MagAOX and `tmcController` revisions, local dependency modifications,
  libftdi/libusb versions, and actual power configuration.
- App/PDU state and target logs around the event, and kernel USB disconnect/reset
  messages. These distinguish commanded removal, USB instability, and software
  races. A write failure alone does not establish a bad cable or device.
- Measured cube enumeration/firmware readiness timing to choose a startup wait,
  and confirmation that recovery should retain the current disabled-before-READY
  behavior rather than restore prior output commands.

All instrument reproduction, power cycling, and INDI queries must be performed
by the operator under AGENTS.md rule 0. The implementation and regression suite
should remain entirely runnable on a workstation without instrument access.

## Analysis validation

Source locations were checked against the final `dev` baseline and the pinned
upstream dependency. The reported error locations and one-second interval agree
with the identified path. Whitespace verification passed using
`git diff --no-index --check /dev/null agents/plans/2026-10/kcubeCtrl-power-recovery.md`
to include the new, untracked file. clang-format does not apply to this
Markdown-only change. No executable verification or confirmed memory-crash
reproduction is claimed.


## Implementation record

The user approved implementation and selected a companion `tmcController` branch,
retaining MagAOX's external dependency. Both repositories use
`jrmales/kcube-power-recovery`; MagAOX is based on the dev revision above, and the
companion on upstream `6ba83aefadaf3ac2120e45ad7425aa6bce3aff80`. Its functional
change is commit `8421603` (followed by documentation and formatting commits).

### Application decisions

- Wrap the registered power callback using the established filter-wheel pattern.
  After the base callback handles partial messages, publish an atomic actual/target
  snapshot with an interruption generation. Only its callback thread writes the
  snapshot after startup. Each connection attempt retains its generation, so even
  Off/On entirely between iterations invalidates the old connection. The transport
  hook can read that snapshot while the hardware mutex is held, without deadlock.
  Broader base-class raw power-field synchronization is outside this change.
- Serialize open, connect, initialization, polling, commands, recovery, and shutdown
  with `m_indiMutex`. Commands validate admission after locking. The main loop uses
  try-lock so an active callback is allowed to finish. A failed callback immediately
  enters recovery, since the INDI dispatcher does not use its return code to drive
  the application FSM.
- Recover transport/setup failures by releasing both axes, clearing commanded enable
  and set flags, and returning zero from `appLogic()`. Retry at a monotonic deadline.
  Expected power cancellation suppresses USB error noise; genuine On/On failures
  retain axis, serial, operation, result and state context, deduplicated while retrying.
  A missing device at open is informational.
- Add `connection.retryInterval` and `timeouts.io`, both defaulting to 1000 ms and
  bounded to 1..60000 ms. Require distinct nonempty axis serials. Invalid config
  and INDI registration remain fatal startup errors.
- Set the app's `power.powerOnWait` default to zero to preserve its existing immediate
  startup attempts. Honor an explicitly configured wait using a monotonic deadline,
  including startup into already-On power and generation changes. This replaces
  reliance on the base loop counter, which cannot represent target-only interruptions.
  A deployment may choose a larger wait after measuring firmware readiness.
- Reinitialize with both outputs disabled. Never replay set/rest/enable/voltage
  commands after an interruption. Retain last measured voltages with Idle/Alert
  status until fresh readings are available, rather than fabricate zero measurements.
- Validate all seven real command callbacks, including numeric parsing and nonfinite
  values. Finite voltage requests still clamp to 0..150 V and publish that clamped
  target only after the write succeeds.
- Keep the app header-only. Require `TMCC_IO_RECOVERY_VERSION` from the companion,
  and add `TMC_CONTROLLER_PATH`, `FTDI_CFLAGS`, and `FTDI_LIBS` build overrides.
  Default headers still come from `/opt/MagAOX/source/tmcController`; pkg-config
  supplies FTDI flags. No dependency sources are vendored in MagAOX.

### Companion decisions

- Give each context unique, noncopyable ownership, handle allocation failure, make
  open idempotent, unwind every failed/canceled connect, and clear connection flags
  even if closing reports a USB interface-release error. Changing identity releases
  any previous device. Destruction closes silently before freeing the context.
- Add a virtual `communicationAllowed()` hook (default true), finite USB timeouts,
  and an overall steady-clock reply deadline. Read in at most 100 ms slices, check
  cancellation between steps, and check settling waits in 10 ms slices. A synchronous
  USB operation already in progress may take its configured timeout to return;
  cancellation cannot undo a hardware command already sent.
- Treat short writes and flush failures as errors; never retry a partially written
  command. Validate reply ID, address, payload length and channel before decoding.
  Return -300 for response deadline, -301 for short write, -302 for a mismatched
  reply, and -666 for cancellation. Existing FTDI operation offsets remain.
- Replace unaligned integer aliasing with explicit little-endian encoding/decoding
  and initialize reserved MMI bytes. Reject invalid voltage settings, including
  nonfinite floats under fast-math. These changes remove undefined behavior exposed
  by sanitizer-oriented review while preserving supported wire messages.
- Provide a reusable scripted FTDI boundary in the companion's tests directory.
  The MagAOX suite compiles the real controller and its protocol against that boundary;
  no libftdi functions or actual USB devices are used by either test executable.

### Verification

On workstation `mx11`, using development headers extracted under `/tmp/kcube-sdk`
(no system installation) and the local companion checkout `/tmp/kcube-tmcController`:

```sh
make -C apps/kcubeCtrl COVERAGE=1 TMC_CONTROLLER_PATH=/tmp/kcube-tmcController \
  FTDI_CFLAGS='-I/tmp/kcube-sdk/usr/include/libftdi1 -I/tmp/kcube-sdk/usr/include/libusb-1.0' \
  FTDI_LIBS=/usr/lib/x86_64-linux-gnu/libftdi1.so.2
make -C tests -f Makefile.one t=../apps/kcubeCtrl/tests/kcubeCtrl_test.cpp COVERAGE=1 \
  TMC_CONTROLLER_PATH=/tmp/kcube-tmcController \
  FTDI_CFLAGS='-I/tmp/kcube-sdk/usr/include/libftdi1 -I/tmp/kcube-sdk/usr/include/libusb-1.0'
apps/kcubeCtrl/tests/kcubeCtrl_test '[kcubeCtrl]'
make -C /tmp/kcube-tmcController/tests check \
  FTDI_CFLAGS='-I/tmp/kcube-sdk/usr/include/libftdi1 -I/tmp/kcube-sdk/usr/include/libusb-1.0'
```

- Production app compilation/linking passed against the actual libftdi runtime;
  the executable was not run.
- App suite: 15 cases, 5420 assertions passed. Covers every startup FTDI boundary
  failing once, target-Off and rapid Off/On at every boundary, both missing-axis
  orders, poll failures, all seven command failures, malformed inputs, bounded
  incomplete replies, partial set/rest, queued command versus actual polling failure,
  shutdown, repeated cleanup, partial registration and invalid configuration.
- App suite also passed all 5420 assertions with AddressSanitizer and
  UndefinedBehaviorSanitizer instrumentation of the application/test translation unit.
  Shared MagAOX libraries were not rebuilt with sanitizer instrumentation.
- Dependency suite: 154 checks passed with AddressSanitizer and UndefinedBehaviorSanitizer,
  including `-ffast-math`. Checks allocation/open/close ownership, all connection
  failures and cancellation points, fragment decoding, supported messages,
  malformed replies, short writes, zero-byte reads and nonfinite commands.
- LeakSanitizer cannot run under this environment's tracing/ptrace setup. Sanitized
  runs use `ASAN_OPTIONS=detect_leaks=0`; fake-context ownership is checked explicitly.
  No claim of leak-sanitizer or whole-process thread-sanitizer coverage is made.

### Remaining operator validation

Deploy the companion revision before rebuilding the app. Confirm real KPZ101
firmware replies satisfy the stricter frame checks, measure startup timing and
perform supervised power-cycle/USB-fault checks. If the memory crash persists,
collect the core and all-thread backtrace along with revisions and power/USB logs
listed above. Offline tests establish the repaired lifecycle behavior, not the
specific physical cause of the reported USB failure or a confirmed crash diagnosis.
