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
