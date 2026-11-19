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
