Prompt: apps/zaberLowLevel is crashing frequently. The problem seems to be that a command is rejected, but the command is different which is rejected. Review AGENTS.md and analyze please. Let's document this as a plan under agents/plans, and please lay out a plan to address these issues.

## Analysis

The current `apps/zaberLowLevel` failure pattern is most consistent with ASCII protocol reply desynchronization, not with one specific invalid Zaber command.

The reported symptoms line up as follows:

- one poll cycle logs a rejected command such as `get driver.temperature` or `get system.led.enable`
- the specific rejected command varies from run to run
- the same cycle then logs an unknown warning token such as `6` or `5`
- warning parsing then fails with `parsing incomplete warning response`
- `zaberLowLevel` transitions from `READY` to `ERROR` to `FAILURE`

### Observed protocol-handling problems

#### 1. `sendCommand()` does not distinguish replies from alerts/info

`apps/zaberLowLevel/zaberStage.hpp` uses `za_receive()` and `za_decode()` to read one ASCII message at a time after each command is sent.

The bundled Zaber ASCII transport explicitly supports three decoded message types:

- `@` normal reply
- `!` alert
- `#` info

However, the stage code currently accepts the first decoded message whose `device_address` matches the stage and hands it to `getResponse()` as though it were the reply for the command in flight.

That is unsafe because unsolicited `!` alerts and `#` info messages can arrive between command send and command reply. If one of those messages is consumed first:

- `getResponse()` interprets the message as the command reply
- `m_commandStatus` can be set false because the message does not carry normal `"OK"` reply flags
- the command currently being issued gets blamed as "Rejected"

This explains the most important observed symptom: the rejected command name changes from one crash to the next.

The likely architecture bug is therefore:

- command/response correlation is based only on device address
- message type is ignored
- asynchronous same-device traffic is misclassified as the synchronous reply

#### 2. `parseWarnings()` accepts truncated payloads too far

The warning parsing code expects a response shaped like:

- `"00"`
- `"01 WR"`
- `"05 FD FQ FS FT FB"`

But the guard inside the parsing loop currently checks:

- `response.size() < 3 + n * 3`

That bound is too weak for extracting a two-character warning token at `substr( 3 + n * 3, 2 )`.

As a result, partially truncated warning payloads can:

- produce a one-character token such as `"6"` or `"5"`
- get logged as `unknown stage warning: 6`
- continue to the next iteration
- then finally fail with `parsing incomplete warning response`

That matches the log sequence you captured.

The correct minimum length for warning `n` is effectively:

- `5 + n * 3`

because the parser needs room for the separating space plus the two-character warning token.

#### 3. The `READY` loop masks earlier failures and makes `getWarnings()` the fatal edge

Inside `zaberLowLevel::appLogic()` in the `READY` state, the app polls:

- knob state
- LED state
- parked state
- position
- temperature
- warnings

The return values from `getKnob()`, `getLED()`, `getParked()`, and `updateTemp()` are currently ignored by the caller. Those methods log their own failures, but the app keeps advancing through the poll sequence. The first poll method that is treated as fatal is `getWarnings()`, which moves the app to `ERROR`.

Operationally this means:

- the first visible log may be a rejected `get ...`
- the state transition often happens only when warnings are polled afterward
- the warning-query failure may be secondary fallout from a desynchronized receive stream

So the current crash signature is likely showing:

1. protocol desynchronization
2. misattributed "rejected" command log
3. malformed warning parse on a later read
4. state-machine escalation

#### 4. Alerts and stale input are not drained or modeled explicitly

The ASCII path does not currently define clear handling for unsolicited traffic:

- alerts from devices
- info messages
- stale replies left on the serial stream from earlier activity

Without an explicit policy, the next synchronous query may consume the wrong message and shift the receive stream out of phase.

## Likely Root Cause

The most likely root cause is that the ASCII controller is not correlating replies robustly. An unsolicited `!` alert, `#` info message, or stale same-device message can be consumed as though it were the reply to the current command, causing:

- a false `command Rejected` log on whichever query is in flight
- stale or malformed response payloads to be handed to later queries
- warning parsing failures
- transition to `ERROR` and then `FAILURE`

The warning parser off-by-one issue likely amplifies and clarifies the problem in logs, but it does not appear to be the primary source of the desynchronization.

## Remediation Plan

### 1. Harden reply correlation in `zaberStage`

Update the ASCII receive path so that only a true `@` reply is accepted as the response to a command in flight.

Implementation goals:

- decode every received message before acting on it
- ignore or separately process `!` alerts and `#` info messages during synchronous command waits
- continue reading until the expected `@` reply for the addressed stage is found or timeout occurs
- keep stage state updates from alerts explicit rather than letting them masquerade as command replies

This is the highest-priority fix because it addresses the likely root cause behind the varying rejected command names.

### 2. Decide and document alert/info handling policy

Before changing code broadly, define what the app should do with unsolicited traffic while waiting for a reply.

Recommended policy:

- `@` reply for the target device: consume as the command reply
- `!` alert for any device: log or update warning/state bookkeeping, then continue waiting
- `#` info for any device: log at low priority or ignore, then continue waiting
- reply/alert for a different configured device: do not treat it as the current command reply

This policy should be reflected in code comments near the receive loop so the protocol assumptions remain explicit.

### 3. Fix `parseWarnings()` bounds checking

Tighten the warning parser so it rejects truncated payloads before producing one-character warning tokens.

Expected outcomes:

- no more `unknown stage warning: 5` or `6` from truncated responses
- malformed warning strings fail immediately and deterministically
- logs better distinguish "stream desync / truncated response" from "legitimate unknown warning code"

Add or extend unit tests for:

- `"00"`
- valid multi-warning responses
- truncated responses such as `"01 "`, `"02 WR"`, and `"06 6"`

### 4. Fail the `READY` poll loop more coherently

Revisit how `zaberLowLevel::appLogic()` handles poll failures in `READY`.

Recommended change:

- if any core query fails because reply parsing or command status is invalid, stop the cycle immediately
- surface the first failing operation rather than continuing into later queries

That should make field logs easier to interpret and reduce the amount of secondary corruption after the first protocol error.

Potential scope:

- `getKnob()`
- `getLED()`
- `getParked()`
- `updatePos()`
- `updateTemp()`
- `getWarnings()`

### 5. Improve diagnostic logging around received message type

While hardening this path, add enough logging to confirm whether unsolicited alerts/info are present during failures.

Useful diagnostics:

- decoded message type (`@`, `!`, `#`)
- device address
- axis number
- reply flags / warning flags
- the raw command being waited on when a non-reply message arrives

This logging should be concise and ideally guarded to avoid flooding normal operation, but it will be valuable during verification.

### 6. Add targeted tests for protocol desynchronization behavior

The existing tests focus on warning parsing. Add tests that exercise the new reply-selection logic.

Targets:

- a normal `@` reply path
- an unsolicited `!` alert arriving before the matching `@` reply
- an unsolicited `#` info message arriving before the matching `@` reply
- malformed warning payload rejection

If the current code structure makes direct unit tests difficult, factor the message-selection logic into a small helper that can be tested independently.

### 7. Verify operational compatibility with actual stage firmware

After code hardening, verify on hardware that the queried settings are in fact supported by the deployed controllers and firmware.

This step is still worth doing even if desynchronization is the main bug, because a true command rejection remains possible.

Verification targets:

- `get driver.temperature`
- `get system.led.enable`
- `get knob.enable`
- `warnings`

Expected result:

- if those commands are valid, the prior "Rejected" logs should disappear once reply correlation is fixed
- if any command is truly unsupported on some devices, the app can then be updated to degrade gracefully per capability

### 8. Stage the implementation in small, reviewable steps

Recommended change order:

1. add this plan and capture the root-cause hypothesis
2. fix reply selection to require `@` replies
3. fix warning-parser bounds and add tests
4. tighten `READY` poll failure handling
5. add temporary diagnostics if needed for on-hardware verification
6. remove or reduce extra diagnostics once behavior is confirmed

This sequencing keeps the highest-value protocol fix isolated and makes it easier to correlate behavior changes with logs from the next deployment.

## Risks and Edge Cases

- Some alerts may carry useful state transitions, so ignoring them entirely could lose information. The fix should skip treating them as command replies, but it may still be worth updating warning bookkeeping when they arrive.
- If multiple devices on the daisy chain emit asynchronous messages, same-address filtering alone is definitely insufficient; the new code should remain strict about message type.
- If there is already stale unread traffic in the serial buffer from earlier code paths, the first hardened receive may expose that more clearly. In that case, an explicit drain or recovery strategy may also be needed.
- A true unsupported command may still exist on some hardware variants. The reply-correlation fix should not assume every rejection is false, only that the current logs are not reliable enough to prove a command-level incompatibility.

## Expected Outcome

After these changes, `apps/zaberLowLevel` should:

- stop attributing random commands as rejected
- stop producing misleading one-character warning codes from truncated payloads
- fail closer to the first real protocol error, with better logs
- become much easier to validate against the actual Zaber hardware behavior
