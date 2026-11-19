# accelSim simple DM output restoration

User request: compare `parkertjohnson1/accelSim` with the operational refactoring,
restore scale/on/off disturbance output through the existing
`dmSyncopationProducer` and `dm01disp10_aux`, delete duplicated qualification,
preserve calibration/physics/acquisition/WFS synchronization and producer
ownership/scheduling/acknowledgement/removal, and do not change loPredCtrlAcc or
the producer base. No commit, push, installation or deployment is authorized.

This record supersedes the accelSim-specific output policy in
`ddspc-operational-output.md`; the controller's policy is outside this change.

The original branch writes a calibrated same-sample disturbance directly to its
DM channel. The current checkout added SyncoOutput, separate authorization and
routing flags, an operational build profile, 8-kHz/WFS-health qualification,
a 250-us cache-age check, revised enable/arm state and fault latches. These are
removed from accelSim. The historical `7f59b4f7` dmSpeckle implementation shows
the direct producer poll/publish pattern; this checkout's dmSpeckle is a legacy
writer and is untouched.

The acquisition post-publication hook still synthesizes the physical pointing
surface into a private cache. A separate worker calls the unchanged producer
base. Cache copies never make acquisition wait; no model prediction, extra
sample, new slot clock, freshness timeout or output interlock was added. Mask,
geometry, finite values, projection rank and physical residual validation remain.
Calibration identity is checked while loading; runtime uses immutable loaded
calibration rather than a calibration-filesystem watchdog.

The app retains only operator intent, transient numerical/transport diagnostics
and receipt bookkeeping for active/zero status. The base retains exclusive
producer ownership and decides when the payload can be reused. Off queues a
zero independently of subsequent ADC samples; discarded zeros are retried until
copied. The previous bounded 100-ms shutdown removal attempt is retained. A
minimal CRTP lifetime holder allows the unchanged producer to finish that zero
after MagAOXApp has set its shutdown flag. It supplies only that flag, with no
wrapper publication or qualification methods.

The old dmInjection staging entrypoint remains a stateless compatibility adapter
because existing loPredCtrlAcc transport tests call it. Its old timing argument
is ignored, and it stores no enable revisions, fault state or output ownership.
The direct-write dmStream wrapper is deleted. Existing model, ADC conversion and
WFS phase-tracking code is unchanged apart from whole-file formatting.

INDI names and elements are retained. Numeric targets are parsed with std::stod
so NaN/Inf text reaches the existing finite-value rejection instead of the legacy
getter's failed numeric extraction becoming zero. dm_status Number elements now
serialize explicit 0/1 values, rather than boolean true/false strings. ready means
calibration prepared; active/zero_acknowledged require actual exact copy receipts.

The simulator's normal build supports output; it no longer has a separate
operational/shadow build profile. acc000.conf and config/example.conf remove the
obsolete settings. The former keeps 8-kHz acquisition and depth 128. A deployed
matching dmmask path and existing timing-stream name must still be supplied;
no deployed names or calibration paths are guessed. The WFS-reference name may
remain empty for timer mode or retain the operator's existing deployed value.

Verification on ASTR-LT-428.local (personal workstation), without hardware:

- Model/synchronization suite: 18 cases, 226432 assertions passed.
- Real app controls and unchanged producer with private ordinary-file API and
  framework doubles: 9 cases, 128143 assertions passed, including 8000 scaled
  virtual slots, numerical recovery, exact/mismatched/discarded receipts,
  exclusive ownership, scheduler reset, disable and post-shutdown zero removal.
- ASan/UBSan run of the same app/producer suite: 9 cases, 128143 assertions
  passed with no sanitizer findings.
- Both configuration files parse, omit the removed keys and start disabled.
  clang-format (preserving the required mutex-scope comment) and git diff --check
  pass. A normalized comparison confirms all 16 acquisition/synchronization/
  diagnostic function bodies are unchanged apart from formatting.
- Native app build stops in the existing framework precompiled header because
  generated/logTypes.hpp is unavailable. Native Linux linkage/runtime and actual
  8-kHz deadline performance are not claimed by the private protocol tests.
- The untouched loPredCtrlAcc operational_config_test.py reports one error and
  one failure: it requires the removed output_authorized key and shadow build
  profile (and contains an old simulator-config mirror). This is a stale
  cross-app expectation; no controller file was edited to change it.

Concurrent unrelated edits appeared in loPredCtrlAcc/shadow_controller.* and
libMagAOX/app/syncoOutput.hpp during this task. They were left untouched and are
not part of this work. No instrument command, live stream access, commit, push,
installation or deployment was performed.

## November 19 quiet-output investigation

Reviewed scheduler and dmSpeckle at bb5a5d89c594bc7949af2949f457a1520b7a3845.
The local producer and timing headers match that revision. The archived config
process list at c673302edbf57a583d0661df240fe9afab8035f6 maps tweeterMod to
dmSpeckle. Its deployed revision/configuration still needs confirmation.

The scheduler sets all producers' step count through maxSubframes. There is
no producer-side registration. At 2 kHz, steps 1/2/3/0 target
125/250/375/500 us, with step zero belonging to the next primary cycle.
The 70-us ready lead gives cutoffs 55/180/305/430 us. writeLeadUs=50 waits
until 5/130/255/380 us when called early. Later copy budgets govern scheduler
work after producer completion.

Poll attaches and locks before validating the schedule. It requires a coherent
version-2 record with valid identity, state, reference and expiry. Outstanding
ownership waits for an exact receipt or valid epoch/generation reset. Poll
selects the earliest future cutoff and suppresses duplicate requests. Zero
returns do not indicate transport faults. Publish waits, revalidates identity,
checks deadline and finite pixels, then publishes payload and ready sidecar.
The scheduler snapshots or discards and echoes the exact commit to release
ownership. A late wake can fail before either producer stream changes.

accelSim uses a plain std::thread and 50-us unsuccessful-poll sleep. dmSpeckle
uses threadStart with priority/cpuset configuration and a 10-us poll sleep.
This difference can affect deadlines. Live thread scheduling remains unmeasured.

A reproduced acknowledgment race is fixed. A receipt arriving between the app's
receipt check and the base poll could release ownership inside poll, after
which the app cleared bookkeeping without recording the disposition. Receipt
observation now follows poll and precedes the next publication. This fixes
active/zero bookkeeping but cannot explain a source counter that never moves.

Added dm_status counters for requests, zero-result polls, commits, failed
publishes observed before/after cutoff, exact copied/discarded receipts and
outstanding ownership. late_returns describes time of observation because
the base's boolean return does not identify its failed check.

Tests now use realistic slot ordering, cycle labels, 70-us cutoffs and expiry.
They reproduce quiet sources with disabled records updated every frame and
with valid requests waking 51 us late against writeLeadUs=50. A candidate
125-us write lead passes all four scaled slots with receipts at cutoff.
Configuration defaults remain pending live evidence.

The receipt regression failed before the fix. Afterward 12 DM tests passed
128276 assertions. Model/synchronization passed 18 tests and 226432 assertions.
clang-format and diff checks passed. Native compilation on this Mac stops at
missing generated/logTypes.hpp in the framework precompiled header.

On a provisioned development workstation, rebuild from the repository root:

```sh
make -C apps/accelSim clean
make -C apps/accelSim -j4
make -f apps/accelSim/tests/Makefile.offline -j4 check
```

The exao2 root cause remains pending runtime enable.toggle, settings.steps,
a coherent timing payload with monotonic time, and producer counter deltas.
production.enabled selects the implementation. Runtime enable controls
scheduling, and a disabled scheduler still updates invalid timing records.
The operator has been asked for these readbacks. No instrument access, commit,
push, install or deployment was performed.

## Confirmed primary outage from operator readbacks

The supplied primary stream rate is zero and scheduler health reports
production main timeout; reacquiring. Producer requests, commits and failures
are all zero while poll_waits grows. This establishes schedule ineligibility
as the cause of the current absent output. Ownership succeeds before eligibility
is evaluated. dmSpeckle uses the identical base and would also wait.
The scheduler timeout code increments generation and publishes an invalid record.
A primary cadence is required for sustained production. A finite outstanding
schedule can remain usable briefly after the last primary update until expiry.

The scheduler does not inspect AO feedback gain or optical loop closure.
It requires actual primary publications and its existing reader-release and
combined-stream prerequisites. cacaoInterface exposes mfilt.loopON and loopgain,
but the available source does not establish a deployed open-loop primary writer.
Do not infer that camera acquisition alone supplies the primary cadence.

The implementation comparison finds identical producer configuration macros,
source ownership, poll eligibility, tagged publish and exact transport receipts.
accelSim uses a lifetime-only CRTP holder for its bounded post-shutdown removal.
It prepares calibration at startup and attaches on demand. dmSpeckle attaches
through its application lifecycle. accelSim automatically queues zero on disable,
where dmSpeckle requires modulation off followed by its separate zero request.
The cached-surface mutex is independent of the acquisition stepping and acquisition
uses try_lock. There is no separate DM clock or cadence estimator.
(Superseded below: the cache and its mutex were removed by the target-time timeline.)
Thread scheduling differs: accelSim uses std::thread and 50-us poll sleeps,
dmSpeckle uses framework threadStart and 10-us poll sleeps. This is not the cause
of zero eligible requests with an invalid timing record.

A remaining substantive difference is target-time synthesis. dmSpeckle evaluates
a deterministic waveform at targetNs. accelSim holds the latest acquired
stochastic sample. Future random innovations cannot be recovered from that
sample. The operator has been asked whether to use a conditional forecast or
buffer a shared future realization. Neither change is silently substituted for
preserving the existing simulation/ADC relationship.

Added outage/reacquisition regression and exercised four slots with the required
50-us lead. The 125-us configuration suggestion is withdrawn for this failure.
The config template now uses the supplied scheduler name. The matching mask
path and deployed WFS phase-reference stream still require deployment values.

Operator build/install commands, to be executed by the operator on the RTC:

```sh
cd apps/accelSim
make clean
make -j4
sudo make install
```

Installation preserves an existing acc000.conf. Merge the supplied template with
the deployed mask and WFS reference. Keep dm.enabled=false and scale=0 initially.
With the normal primary publisher running at 2 kHz, scheduler runtime enable On,
steps=4 and active timing, the operator can use:

```sh
setINDI -n 'acc000.dm_scale.target=0.1'
setINDI -s 'acc000.dm_enable.toggle=On'
getINDI 'acc000.dm_status.*'
```

Expected acc000 publication is 8 kHz throughout. Primary is approximately 2 kHz,
timing records normally follow the primary cadence, and eligible auxiliary
payload/ready/ack rates approach 8 kHz when deadlines and receipts are serviced.
requests/published/copied should grow; active becomes 1 after a copied receipt.
On disable, the operator uses:

```sh
setINDI -s 'acc000.dm_enable.toggle=Off'
getINDI 'acc000.dm_status.*'
```

Keep the scheduler running until zero_acknowledged=1 and active=0. Then auxiliary
publication stops while ADC continues. If the scheduler is inactive, removal
remains pending until it recovers. Native RTC rates are operator verification.

## November 19 target-time timeline (completes the interrupted implementation)

Decision (operator): buffer one shared future realization rather than forecast.
The interrupted checkpoint compiled, but 10 of 13 DM tests failed: none advanced
the timeline, so `physical(targetNs)` found no data. Review found these defects:
linear interpolation (6.2% amplitude and 0.33 deg error at 900 Hz, varying with
sub-sample position, so acc000 and the DM saw different gains); sensor noise taken
from the floor node, which reuses or skips draws when synchronized acquisition
precedes its node; noise split by recomputing the 214-mode sum; a synthesis fault
publishing zeros booked as an active contribution; dropped catch-up discontinuity
counts; shutdown clamps read from unused model counters; a dead pixel cache and mutex;
unbounded catch-up after a stall; an O(n) ring scan under the lock; and a 2-ms
horizon short of the scheduler's 4-ms bound at its 500-Hz minimum main rate.

Implementation:

- `model::step()` stores its existing noise draw in `sample::m_noise`. RNG order
  and arithmetic are unchanged; the 18 model tests and 226432 assertions match.
- `accelSimTimeline.hpp` is an index-addressed 128-node ring on the acquisition
  grid: node k at epoch + k/fps, the nominal time of tick k. After each acquisition
  it generates 5 ms ahead, one model step per acquisition in steady state.
  A gap of more than maxCatchUp ungenerated nodes redraws stationary histories once,
  as before; shorter gaps now need no restart.
- Evaluation uses 4-point cubic Lagrange weights shared by pointing, WFS pointing and
  noise-free acceleration. This preserves a = -(2 pi f)^2 L theta for every mode
  at every time. The error is <1e-4 to 300 Hz and 0.56% at 900 Hz.
- acc000 is the physics at the acquisition time plus the consumed tick's own noise.
  The DM uses noise-free pointing at `request.targetNs`. All clocks are same-host
  CLOCK_MONOTONIC, the clock of the scheduler, producer and acquisition.
- serviceDM skips and counts uncovered targets. A synthesis fault publishes a
  removal zero. New `dm_status.attached` and `timeline_misses` elements are reported.
- Clamp diagnostics count published measurements, not nodes generated ahead.

Measured on this Mac: steady advance 4.36 us per acquisition, the legacy step cost.
The one-time anchor after start or rate change is about 190 us. physical() takes 16 ns.
Cubic look-ahead uses nodes up to two intervals after a measurement. Its worst-case
later innovation is 9.5e-5 m/s^2 rms, 1/86 ADC count and 1/67 sensor-noise rms.

Verification on ASTR-LT-428.local (personal workstation), without hardware:

- Offline app/producer/timeline suite: 22 cases, about 4.7M assertions passed.
  This includes 11 DM output and 7 timeline cases.
- Same suite under ASan+UBSan and under TSan: no findings. Removing the reader lock
  makes TSan report a race in timeline::advance, so concurrency is exercised.
- Mutations killed: evaluation at cutoff, evaluation at poll time (latest-sample hold),
  linear weights, floor-node noise, fault zero booked active, unbounded catch-up.
- Standalone: model 18/226432, sync-regression 8/145840, disturbance 14/12023,
  DM runtime 3/80016, timeline 7 cases.
- Read-only syntax check of loPredCtrlAcc sim_bridge/confirmed_replay passed.
- Native build stops in the framework PCH (generated/logTypes.hpp missing).
  The Linux-only full accelSim_test fixture was updated for m_acquiredSample but
  could not be compiled here.

Not changed: producer base, timing protocol, scheduler, dmSpeckle, loPredCtrlAcc,
configuration keys, the std::thread DM worker and its 50-us retry sleep.
dmSpeckle uses threadStart with priority/cpuset and 10-us waits. Revisit only if RTC
`late_returns` rises once the scheduler is active. No instrument access, commit,
push, installation or deployment was performed.

## DM scale range [0,100]

Operator request: allow `dm_scale` above 1. `accelSimDetail::maxDMScale = 100`
(accelSimDM.hpp) now bounds the INDI property, the runtime callback, startup
`dm.scale` validation and `dmInjection::synthesize`. The scale remains a linear
multiplier of the noise-free pointing surface. Finite-value checks, enable/disable,
the timeline, acc000 and the producer path are unchanged. The tests accept 1.1, 1.5,
2.5, 37.5, 50 and 100, and reject -1, 100.001, NaN and Inf. A published four-slot
surface at scale 2.5 matches the target-time expectation.
