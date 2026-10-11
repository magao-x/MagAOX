# filterWheelCtrl error recovery

Issue: https://github.com/magao-x/MagAOX/issues/359

Prompt: Review the issue, then implement communication recovery without automatic
homing. A later change may restore READY when an error interrupted a confirmed
stationary READY state and the position is unchanged after reconnect.

## Decisions

- Communication failures keep the application running, release the serial port,
  clear the interrupted homing/preset sequence, and restart USB discovery.
- A successful communication recovery enters NOTHOMED without issuing home or
  move commands, including when stage.powerOnHome is enabled.
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
- The READY-position exception is deferred. It needs a retained valid position,
  fresh stationary/position replies, and invalidation on power changes or commands.

## Verification

Use the shared offline application-test harness, scripted USB/serial boundaries,
and a private INDI driver. Exercise the real FSM and callbacks without opening an
instrument connection. Check recovery from polling, setup, homing, preset and raw
movement failures; repeated failures and USB rediscovery; malformed replies;
power-off during recovery; manual homing; and both power-on-home settings.

Completed verification on the local workstation:

- `make -f Makefile.one t=../apps/filterWheelCtrl/tests/filterWheelCtrl_test.cpp COVERAGE=1`
  from `tests`, followed by the test executable: 9 cases, 2,902 assertions passed.
- `make -C apps/filterWheelCtrl COVERAGE=1`: production application compiled and
  linked successfully. Coverage was selected to match the existing shared library.
- `clang-format` applied to the changed C++ and Doxygen files; its dry-run check and
  `git diff --check` passed.
- All runtime verification used scripted transport and private local FIFOs. No
  application was launched against instrument hardware.

Status: Initial recovery implementation complete. READY-position preservation is
deferred, and instrument validation must be performed by a human operator.
