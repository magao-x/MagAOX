# filterWheelCtrl error recovery

Issue: https://github.com/magao-x/MagAOX/issues/359

Prompt: Review the issue, implement communication recovery without automatic
homing, commit that work, then implement READY recovery when an error interrupted
a confirmed stationary READY state and the position is unchanged after reconnect.

## Decisions

- Communication failures keep the application running, release the serial port,
  clear the interrupted homing/preset sequence, and restart USB discovery.
- A successful communication recovery defaults to NOTHOMED without issuing home
  or move commands, including when stage.powerOnHome is enabled. The READY
  exception below applies only after validating the retained position and history.
- Startup and observed power-on retain the existing stage.powerOnHome behavior.
  A pending automatic home is canceled by a communication failure, so a failed
  initialization or homing attempt cannot cause repeated automatic homing.
- Observed power-off releases the port and clears sequence state. Commands and
  reconnect attempts are suppressed when observed or requested power is off.
- Invalid integer replies fail without replacing the last valid position with
  zero. Transport failures from callbacks follow the same recovery path as polls.
- Existing manual movement in NOTHOMED remains available. This change does not
  introduce a new motion policy or resume an interrupted command automatically.
  Recovery does not send a stop command; an already accepted controller movement
  can still finish, while the host abandons its interrupted sequence.

## READY recovery

The initial recovery implementation was committed as `0be9b679` before starting
this extension.

- Retain a snapshot only after a complete successful READY poll reports zero
  motor speed, no active homing/preset sequence, and uninterrupted power/command
  history. Preserve raw counts separately from subsequent serial reads.
- An error originating from that confirmed READY state can retain the snapshot
  across repeated discovery, open, setup, drain, or reply failures.
- Restore READY after setup and a fresh successful status poll only if speed is
  exactly zero and raw counts exactly match. Nonzero speed invalidates the
  exception even below the existing normal-motion threshold. A valid changed
  count or nonzero speed permanently discards that interruption's snapshot.
  Keep a candidate in CONNECTED during verification, then transition directly to
  READY on success so no intermediate NOTHOMED transition is logged.
- Raw, numeric preset, named preset, home, and stop requests invalidate the
  snapshot, including rejected requests while disconnected. An atomic generation
  also detects stop requests or power updates during polling.
- Replace the base application's already-registered power callback during
  startup with an app-specific wrapper that preserves base power handling and
  records observed/requested Off or unknown states. Repeated On updates are benign.
  An Off-to-On update between FSM iterations still invalidates READY recovery.
- Latch power interruptions so later ordinary READY polling cannot erase the
  lost-reference evidence. An accepted new homing sequence starts a new reference;
  rejected home requests leave the interruption latched.
- Startup/power-on, unconfirmed READY, interrupted HOMING/OPERATING, and failed
  READY validation retain the default NOTHOMED behavior. Recovery issues no motion
  command, whether it restores READY or defaults to NOTHOMED.
- Matching counts cannot identify an unreported controller reset that happens to
  reproduce the retained count. Controller reset/homed status would be needed to
  distinguish that case if the protocol makes it available.

## Verification

Use the shared offline application-test harness, scripted USB/serial boundaries,
and a private INDI driver. Exercise the real FSM and callbacks without opening an
instrument connection. Check recovery from polling, setup, homing, preset and raw
movement failures; repeated failures and USB rediscovery; malformed replies;
power-off during recovery; manual homing; and both power-on-home settings.

Initial recovery verification on the local workstation:

- `make -f Makefile.one t=../apps/filterWheelCtrl/tests/filterWheelCtrl_test.cpp COVERAGE=1`
  from `tests`, followed by the test executable: 9 cases, 2,902 assertions passed.
- `make -C apps/filterWheelCtrl COVERAGE=1`: production application compiled and
  linked successfully. Coverage was selected to match the existing shared library.
- `clang-format` applied to the changed C++ and Doxygen files; its dry-run check and
  `git diff --check` passed.
- All runtime verification used scripted transport and private local FIFOs. No
  application was launched against instrument hardware.

READY recovery verification completed on the local workstation:

- The same test build and executable passed 17 cases with 7,245 assertions.
  Coverage includes unchanged/changed counts, exact stationary speed, repeated
  failed retries, accepted/rejected motion requests, partial power updates, rapid
  Off-to-On updates, interruption during snapshot and verification polls, rejected
  home requests, and renewed eligibility after manual homing. Successful recovery
  is verified to remain CONNECTED during status validation and then enter READY.
- The production application compiled and linked with `COVERAGE=1`.
- Formatting and `git diff --check` passed. All runtime tests remained offline.

Status: Initial recovery and the READY extension are implemented and verified.

Instrument validation must be performed by a human operator.
