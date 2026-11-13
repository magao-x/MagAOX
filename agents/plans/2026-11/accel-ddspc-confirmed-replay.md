# Confirmed-input shadow replay and bounded ADC forwarding

2026-11-13, `parkertjohnson1/accel_ddspc`, starting at `c18b81c5`, personal
workstation `ASTR-LT-428.local`. The operator reports validated accelSim injection
and lossless 8-kHz ADC observation with producer depth 128. Preserve those paths;
do not develop another disturbance generator or AO application.

## Authorized implementation

1. Add a bounded, caller-owned cursor-copy API to the actual capture/reader.
   Preserve decoding, capture behavior, existing retention API and calibration.
   Report source/segment changes, unavailable counters and remaining eligible data.
2. Use that API only in an isolated offline forwarding adapter for this stage.
   Release capture locks before calling the application. Retain original sample
   times/identities; record frontend availability separately. Never infer native
   timing qualification from ADC transport health.
3. Reuse `accelSimDetail::model` from the simulator worktree, read-only. Generate
   explicit synthetic WFS exposures and independent actually consumed, limited
   input intervals before any shadow callback. No candidate enters the recording.
4. Feed the real app's `processShadowAdc/Wfs/Applied` seams with its existing
   shared pipeline and independently controlled training switches. Test source
   loss, bounded forwarding, recovery, readiness, unverified input refusal, and
   assisted feature influence with standard-model invariance.
5. Keep native startup/input callbacks and all DDSPC mathematical algorithms
   unchanged. Compile the optional replay target with `DDSPC_SHADOW_ONLY`; no
   native app, stream, INDI server, instrument, service, install or deployment.

## Live integration remains unqualified

The native WFS stream/producer, exposure-to-frame association, modal calibration,
common-clock mapping and uncertainty must be established. Arrival time is not an
exposure timestamp. `dmRecon` projects commands, not optical WFS residuals.

The simulator's injected `dm10` contribution is not the combined applied input.
Combiner channels, limiting/mapping, driver consumption and optical timing remain
required. BMC publishes its `_shape` before `BMCSetArray`; SDK success alone is
not physical application evidence. Do not relabel either as VerifiedApplied.

The builder requires completed constant-input coverage of lag-adjusted exposures.
Independent replay intervals may close before the following decision. Real holds
ending after that decision cannot be certified in advance; qualified completed
prefixes or a separately reviewed latency contract would be needed.

## Verification record

Implemented the additive cursor API, replay-only bounded dispatcher, real
accelSim-model recording and actual guarded-app tests. The source reader lock is
released before downstream processing. No native startup or callback calls the
new dispatcher; live WFS/applied-input integration remains deferred.

[Application guide](../../../apps/loPredCtrlAcc/tests/CONFIRMED_REPLAY.md) records
the exact native evidence gaps, functions, causal synthetic timeline, commands,
test coverage and limitations. Local verification passed:

- Full offline regression: 116 cases, 2,259,267 assertions.
- Capture/actual-worker API-double ADC tests: 15 cases, 448,098 assertions.
- Dedicated ADC/DDSPC target: 13 ADC, 25 numeric, 21 guarded app and 4 Python tests.
- New confirmed-model replay: 6 cases, 40,049 assertions; combined guarded app and
  replay: 27 cases, 50,374 assertions.
- Address/undefined and thread sanitizers: ADC and combined guarded app/replay
  pass without findings. clang-format run; whitespace check clean.
- 800 WFS records produce 793 updates per learner, ranks 20/20 and 26/26,
 774 eligible candidates (709 assisted), 3,202 ADC samples with zero capture or
 forwarding loss. Callback work mean 16.385 us, p99 19.041 us, max 32.624 us on the
 personal Mac; no native real-time qualification is claimed.

Initial new-test failures were incorrect fixture assumptions (an absent frame
claimed to remain in a temporal cube, and uninterrupted rank readiness after
training pause); corrected without algorithm changes. Review also removed a
default replay-only clamp that could have hidden source rail evidence; a new
rail-rejection/recovery scenario passes. There are no remaining test failures.

Tracked-file hashes confirm the original controller, numerical algorithms,
native app header and archived results are unchanged. The simulator worktree
remains clean. No live app, stream, service, installation, commit or deployment
was performed. Existing ADC-only execution is not connected to the new adapter.
