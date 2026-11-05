# dmSyncopation production providers

## Objective and settled decisions

Promote the qualified counter-publication scheduler to continuous production
operation and support same-host producers through a reusable
`MagAOX::app::dev::dmSyncopationProducer<derivedT>` CRTP helper. Migrate
dmSpeckle as the first producer while preserving its legacy operation.

The [counter-publication record](dm-syncopation-counter-publication-targets.md)
contains the 8 kHz qualification and the decisions preceding this integration.
The final 120-second constant-zero run delivered and acknowledged all 719,997
post-startup auxiliary slots with a 6 us final admission allowance. These
measurements establish the transport/timing path, not nonzero application
correctness. No agent may operate instrument hardware.

Production providers are independent and same-host only. A missing, incomplete,
invalid or late input retains only that producer's last complete destination
contribution; other producers proceed. Always copy every selected mapping.
Explicit zero removes a contribution. A missed input does not move targets or
invalidate a valid later-slot input. Expired slot commands cannot be replayed
implicitly at a later slot.

Step 0 uses the same ready-copy operation within the reader-safe before-main
window, preserving destination counters. Genuine main commands are never gated
by producer readiness. Physical consumption is separate telemetry, not a
per-producer acknowledgement required for passive copying. A completed passive
copy remains visible if a later auxiliary trigger is suppressed. Previous
read-done evidence does not block a new, unexpectedly early main during copying;
copy-window qualification remains necessary.

## Protocol and ownership

Reuse the 41-word provider schedule layout with production version 2. Keep the
older version-1 validation APIs for legacy behavior. A production request
identifies epoch, generation, cycle/step, monotonic target and cutoff.

Retain one 2-D float32 `<destination>_aux` source per producer. Add a
producer-owned signed-int64 `_ready` record and scheduler-owned `_ack` record.
The ready record commits the request identity, exact source inode/counter,
completion timestamp and producer session/sequence/PID. The acknowledgement
echoes the exact command and states copied/discarded. The helper freezes the
single outstanding source payload until that acknowledgement or a validated
schedule reset. This is transport ownership: the acknowledgement occurs after
a private snapshot/discard, without waiting for a physical DM completion.

Source, ready and acknowledgement coherence must be checked across snapshots.
Reject nonfinite pixels, stale/mismatched identities and malformed transports.
A rejected source must not replace its retained contribution or block eligible
independent producers. Revalidate shared reader/clock/primary/output identity
and timing conditions before changing destinations and publishing a trigger.

## Implementation

1. Share the tested finite-zero grid/readiness/wait mechanism with opt-in
   continuous production mode. Preserve finite trial behavior and the older
   scheduler when neither new mode is selected.
2. Publish production schedule targets/cutoffs from the frozen grid. Add
   independent tagged source admission, retained values and complete passive
   copies. Include passive step 0 and continuous disable/reacquire behavior.
3. Put coherent schedule reads, ready/ack transport, deadline checks and
   immutable-buffer handling in the reusable CRTP helper. Provide normal
   base-class integration macros and documented application responsibilities.
4. Replace dmSpeckle's duplicated provider loop/transport with the helper while
   retaining target-time sparkle, arbitrary-cube and tip/tilt generation.
5. Verify workstation-local private streams: independent input misses,
   on-time/late command identity, passive nonzero copies, retention, malformed
   input, restart/replacement, cancellation and timing safety. Re-run existing
   finite zero and legacy application regressions.
6. Publish the [production guide](../../../apps/dmSyncopation/doc/production-providers.md)
   and operator configuration. Instrument qualification remains operator-only.

Initial production preparation settings are `readyLeadUs=50` and
`step0LeadUs=20`; final publication admission remains 6 us with 10 us early
validation wake and zero unmeasured commit lead. Input preparation deadlines
remain separate from publication admission. Changed geometry/mapping counts
require measured copy-budget qualification.

## Verification record

The initial shared producer helper suite passes **10 cases / 269 assertions** under the normal
optimized build flags. Its private-stream cases cover exact acknowledgement,
single-buffer ownership, cooperative writer exclusion, invalid transports,
epoch reset, stale identities, nonfinite pixels, publication after a missed
cutoff, changed source/record extents and predecessor ownership after a producer
restart. A replacement helper adopts an intact unacknowledged commitment before
it may reuse the source; later publications use its new session. The build and run logs are `/tmp/dmSyncopationProducer-build.log` and
`/tmp/dmSyncopationProducer-run.log`.

The scheduler suite passes **152 cases / 6,514,030 assertions**, including the
existing finite-zero and legacy regressions and the new production cases.
Production coverage includes independent source misses in both directions,
exact transport acknowledgements, stale/late identities, nonfinite/busy
sources, malformed record metadata and future commands, valid/invalid startup
configurations, and nonzero common-engine operation at both passive-only
`N=1` and four-step cadence. Continuous mode ignores finite-trial expiry in
both the timed callback and ordinary status path. Build and run logs are
`/tmp/dmSyncopation-production-final-build.log` and
`/tmp/dmSyncopation-production-final-run.log`.

The final dmSpeckle migration suite passes **6 cases / 85 assertions** against
the final helper, including predecessor ownership and geometry hardening.
Both applications compile and link successfully. Logs are
`/tmp/dmSyncopation-production-app-build.log`,
`/tmp/dmSpeckle-production-final-test-build.log`,
`/tmp/dmSpeckle-production-final-test-run.log`, and
`/tmp/dmSpeckle-production-final-app-build.log`. clang-format verification and
`git diff --check` pass for all touched C++ files.

Shared record utilities validate mapped pointers and exact element/byte extents.
The helper freezes source geometry and identity and revalidates them before
access, including after publication waits. Pixel checks use mxlib's bitwise
finite-value classifier under the normal fast-math flags. The independent
consumer rejects incompatible input metadata before reading payloads. Source
or sidecar replacement retains that producer's contribution until explicit
resource reacquisition and any necessary producer reconnection.

Standard production timing/health reflects the active grid and confirmed
publications. The mode exposes only steps/cadence in the disabled-only settings
property; static reader-hold knobs remain specific to the legacy scheduler.
Main silence invalidates the provider schedule and starts reacquisition. Rate
reporting resets its baseline on rearm and guards against unsigned differences
when session totals restart. dmSpeckle backs off unavailable transport retries
by 100 ms while retaining short waits for an attached, idle protocol.

Two initial offline failures were fixture issues: imported ImageStreamIO
handles leave `used=0`, so the fixture must explicitly claim its private
files before destroying them; otherwise old sidecars survive between cases.
NaN fault injection now writes IEEE bits directly so finite-math optimization
cannot remove the intended malformed payload. Neither issue identified a
runtime admission failure. The subsequent full scheduler suite passes.

No hardware actions or changes to instrument configuration are part of this
work. Changing nonzero commands and the selected mapping/producer setup remain
operator qualification work.

## 2026-11-04: correct production INDI startup registration

The first operator startup failed while inserting `dmtweeter-synco.production`.
`setupProductionINDI()` used `REG_INDI_NEWPROP_NOCB`, which already registers
the property, and then called `registerIndiPropertyReadOnly()` on the same
property. The second insertion failed and aborted app startup. This was a
production integration error; the prior offline tests did not exercise this
property's startup registration.

Create the read-only numeric property without registering it, populate all
elements with initial numeric values, then register it once. The new regression
runs the production property setup with INDI enabled but no driver or monitor
threads. It verifies one registry entry, the property pointer and null callback,
read-only/idle metadata, initialized elements and serializable definition XML.
It also verifies that disabled production mode registers no property. No
instrument or external INDI connection is involved.

The normal app build also lacked a dependency on the new production header.
Add that header and the shared producer helper to dmSyncopation's declared
dependencies, and the producer helper to dmSpeckle's dependencies, so ordinary
incremental builds pick up their changes. The corrected scheduler regression
suite passes **153 cases / 6,514,054 assertions**; its log is
`/tmp/dmSyncopation-production-indi-run.log`.
The normal incremental dmSyncopation build recompiles and links successfully
with the corrected dependency. Dry-run dependency checks also select a fresh
compile for changes to the production header and the shared producer helper.
Build and dependency logs are `/tmp/dmSyncopation-production-indi-app-build.log`,
`/tmp/dmSyncopation-production-dependency-check.log`, and
`/tmp/dmSpeckle-producer-dependency-check.log`. Formatting and diff checks pass.
