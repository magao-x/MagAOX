# Upstream-synchronized DDSPC shadow milestone

2026-11-14, personal workstation ASTR-LT-428.local, branch
`parkertjohnson1/accel_ddspc`, starting at committed checkpoint `0df552ab`.
No git staging/commit/push, install, deployment, service, INDI network, producer or
instrument operation is authorized. No source changes to `loPredCtrl`, accelSim,
mcp3208Ctrl or ad7689Ctrl.

Jared's architectural correction supersedes the physical exposure/source-clock
native input gate: acquisition owns RTC semaphore/delay synchronization. DDSPC
consumes mode/lock, epoch/counter grouping, source continuity and local
availability. Do not add a new PLL, delay correction or camera exposure estimator.
Keep the useful older strict adapter and regressions separately available.

Implemented: upstream contract with four distinct ADC pairs per WFS frame;
eight-frame recovery qualification; optional modal reference without reconstructed
camera parent; separate observed-request input domain; complete snapshot/modal
projection/request-sampling factories; native worker packet queues; independent
standard/assisted native RLS/QR; diagnostic forecast generation independent of
synco eligibility; bounded scalar prediction queues and source tags. Configuration
selects the expected upstream convention but never declares a producer locked.

Retained earlier work: deterministic ADC/native qualification replay, rectangular
indexing, independent fits and rank checks, no future-target fitting, matched
conditional error/PSD tools and negative controls. Added primary origin-only
forecasts using held-last confirmed inputs; actual future commands appear only in
auxiliary conditional scoring. No assumed/proposed input enters training.

The replay deliberately omits camera lineage, physical exposures and producer
clock mappings while both models learn and predict. Mode-off, wrong counter
phase, future sample delivery, lock loss/recovery, missing command evidence,
duplicate channels, future target/input changes and queue bounds are tested.

Real native binding remains explicit: actual producer/reference/status and counter
anchors, calibrated full-DM-image to WFS-mode projection, complete command sampling
and input partition are required. No live observer currently supplies that reviewed
bundle automatically. Requested model input is not physical application. Held
2-kHz commands cannot excite independent four-slot responses. Do not waive rank
or causality checks, or call conditional future-input error a prospective forecast.

See `UPSTREAM_SHADOW.md` for source review and numerical validation commands, and
`ACCELSIM_SHADOW_EXPERIMENT.md` for the separately authorized experiment. Existing
disturbance calibration/mechanism is reused, not redeveloped. Synco commit/cutoff/
ack integration and real-time control qualification remain separate; there is no
DM output interface in this milestone. Final validation results are recorded in
the current implementation document after the regression/sanitizer runs.

Validation completed: full regression 155 cases/3,044,306 assertions; native
observation/ADC/application guards 57/14,297; confirmed-input replay 6/40,049;
45 distinct existing Python checks; ASan+UBSan multirate/native-app subset
29/779,787 with no findings; three new Python report checks. All pass. Fifteen
historical loPredCtrl digests match. The positive report covers 48 runs with both
learners, four horizons and independent/unrelated ADC controls, with no input or
trace loss. F=1 primary synthetic MSE improvement averages 12.95% tip/14.28% tilt;
unrelated ADC gives approximately zero improvement. Detailed longer-horizon/error
PSD results and workstation timing limitations are in UPSTREAM_SHADOW.md.

Changes are confined to loPredCtrlAcc and this plan. A fresh Linux native build,
actual producer status/reference binding, reviewed modal DM projection, complete
request sampler/input partition and a native score consumer remain before the
controlled instrument experiment. Exposure/clock reconstruction is no longer a
prerequisite. No installation, instrument command or git mutation was performed.

Affected files:

- Runtime: `loPredCtrlAcc.hpp`, `shadow_controller.hpp/.cpp`,
  `multirate_shadow.hpp/.cpp`, `online_shadow.hpp/.cpp`, new
  `synchronized_inputs.hpp/.cpp`.
- Build/configuration: `Makefile`, `loPredCtrlAcc.conf`,
  `examples/loPredCtrlAcc-online-shadow.conf.example`, `tests/Makefile.offline`,
  `tests/Makefile.replay`.
- Tests/tools: new `tests/qualified_shadow_replay.hpp/.cpp`,
  `tests/ddspc_qualified_replay_test.cpp`, `tests/qualified_shadow_main.cpp`,
  `tests/qualified_shadow_report.py`, `tests/qualified_shadow_report_test.py`;
  updated `tests/native_application_test.cpp`, `tests/native_deployment_test.py`.
- Documentation: new `UPSTREAM_SHADOW.md`, `ACCELSIM_SHADOW_EXPERIMENT.md`;
  updated `ONLINE_SHADOW.md`, `NATIVE_DIAGNOSTICS.md`, `tests/README.md`, this plan.

Paths above are relative to `apps/loPredCtrlAcc` except this plan.

## 2026-11-15 resumed native receiver and syncopation milestone

Completed the interrupted tree without replacing the earlier factories/replays.
This section supersedes the earlier statements that no native bundle is supplied
automatically and that no producer helper is included. No git or instrument
operation was performed. See `apps/loPredCtrlAcc/SYNCO_NATIVE.md` for the complete
current interface, configuration, file inventory and test results.

- `NativeInputPipeline` now consumes original ADC/modal observations, complete
  requested-DM payloads and actual local sampling ticks on the numerical worker.
  It binds reviewed full-image/modal calibration and upstream reference/counter
  anchors, consumes the actual producer's `timingDiag` subscription, and calls
  existing synchronized-input factories and independent RLS/QR models. No PLL,
  exposure reconstruction or physical-application assertion is introduced.
- Read-only native observer hooks automatically supply these records; no subclass
  or manual injection of qualified origins is needed. DM payload matching survives
  metadata loss without shifting subsequent command associations. Incompatible
  WFS/DM layouts explicitly revoke model input, rather than treating the last
  valid command as an indefinitely qualified idle request.
- A bounded ordinary recorder drains origin-only/conditional scalar scores after
  the numerical producer finishes. It exclusively creates a private CSV and
  enforces row/file/queue bounds. Actual sampled input cutoffs stay in in-memory
  qualified records; this CSV is not a complete qualified-input replay archive.
- Reused exact `dmSyncopationProducer.hpp` and `dmSyncopationTiming.hpp` from
  `origin/jrmales/dm-syncopation@bb5a5d89c594bc7949af2949f457a1520b7a3845`.
  Removed only the interrupted formatter's indentation divergence by restoring
  the upstream bytes. The app inherits the actual CRTP base and uses its lifecycle
  macros, but never calls poll/publish or opens auxiliary/ready/ack transport.
- `SyncoShadowProducer` pre-renders private complete images and uses the actual
  Request identity and advertised targets/cutoffs. Copy-only readout rejects late,
  changed, invalid or pre-loss plans, retains pending preceding phases and falls
  back to an eligible standard payload. Four payload slots are explicit; model
  preparation remains at WFS origins. Numerical work is not moved into a 125-us
  callback. This is not per-ADC-sample replanning or real producer transport.
- Future identity is `dm10disp10_aux`, as requested. The inspected upstream example
  maps `dm01disp10`, so the actual live destination/ownership must be reviewed;
  neither stream is opened for DDSPC output. Native inputs remain `dm01disp`.

Found and fixed during completion: unsized native plan output; premature recorder
exit before numerical-producer drain; virtual-replay clock advancement by host CPU
time; input queue/payload mismatch recovery; invalid command streams appearing
idle; and missing pending-plan/fallback/source-loss barriers. These preserve the
existing RLS/QR equations. Tests include actual model-to-prepared-payload flow and
concurrent preparation/readout. Synthetic qualification is explicit throughout.

Final functional verification: full offline 170 cases/3,043,796 assertions;
focused native/multirate 45/781,001; native observation 58/14,309; confirmed replay
6/40,049; shadow-only 22/10,364; timing Python49; report Python3 and producer
Python4. All pass. Focused ASan+UBSan passes45/781,001 with no findings. Fifteen
original loPredCtrl digests match. Application/default/output safety tests remain
enabled; all forbidden output API sentinels are zero. Formatting and whitespace
checks pass; exact upstream headers retain upstream formatting.

Optimized 2500-pixel whole-frame receiver mean90.47us/max183.21us in the focused
run; callback p50 .334us/p99 .833us/max1.917us, excluding actual publication/SDK.
Paced optimized worker accepts nominal8-kHz ADC/2-kHz WFS input and produces1799
complete records (1798 with deliberate metadata loss and recovery). The full-suite
worker has a2691us outlier, exceeding the example500us budget; no deadline or WCET
claim. The batch dataflow fixture uses a10ms job limit; sanitizer replay delivers
four times slower while preserving nominal acquisition timestamps/counters. These
test choices do not weaken production budgets or interlocks. ASan/UBSan does not
qualify thread-race absence or real-time execution.

Remaining before controlled shadow commissioning: fresh Linux GCC14/C++20 build
with actual ImageStreamIO/current mxlib; actual reviewed P and modal indices/units;
producer reference/lock/counter anchors; complete and sufficiently exciting DM
requests; bounded Linux worker timing. Optional R and a reviewed WFS/synco cycle
relation enable private four-slot payload preparation. Neither rates alone nor
requested publication establishes physical application. Physical output requires
later command-domain/response-phase qualification, committed prefixes, limits,
ownership, actual helper transport timing and safe handover. Current build cannot
activate it. Existing ADC-only defaults are preserved. No deploy/install/commit/
push, service restart, telescope access or live test was performed.
