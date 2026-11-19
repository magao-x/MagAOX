# DDSPC operational output engineering record

2026-11-19. User authorized implementation only, with no instrument access,
installation, deployment, service changes, git staging/commit/push or physical
output. Workstation `ASTR-LT-428.local`; branch
`parkertjohnson1/accel_ddspc`; starting controller checkpoint `ef3bc843`.

## Scope and decisions

The checked-out branch did not contain `apps/accelSim`. Imported only that
application directory from locally available `parkertjohnson1/accelSim`,
`9d4c5a99`. No branch switch, merge or other application's source was imported.
The model, ADC encoder, synchronization/delay implementation and physical TT
synthesis numerical headers remain byte-identical to that source. This app's
runtime publishing adapter, build, tests/docs and disabled experiment config are
changed; the complete imported directory will appear new until the user stages it.

The two exact required sources are `dm01disp10_aux` (disturbance) and
`dm10disp10_aux` (correction). No local evidence establishes their two destinations
are combined onto the same tweeter. Authorizations/limits remain disabled/zero in
examples, pending instrument verification. Never substitute a different name.

No new scheduler or synchronization algorithm. `syncoOutput.hpp` factors only
shared output state/finite bounds/zero/receipt handling around the unchanged
`dmSyncopationProducer` base. The contained CRTP instance outlives framework's
shutdown flag long enough to attempt protocol zero. Default builds compile out
all real auxiliary/sidecar I/O. Explicit operational builds still require startup
configuration authorization, verified routing, approved finite positive bounds
and a runtime ON. Both output paths use the actual base's cooperative producer
lock, real Request, ready Commit and matching copied/discarded Ack. Base headers
remain identical to the earlier pinned upstream revision. No direct DM-driver
access, source creation or unscheduled source write is added.

Retain native pipeline/math, individual ADC8k and WFS2k learning. Add opt-in
coefficient/rank-certificate freeze for phase banks, default learning OFF,
reset incarnation barriers, matched explicit model selection and selected-mode
cache retrieval. An operational-capable build cannot enable the old Legacy
FrameGrabber path, even after an invalid loader call. Existing Legacy numerical
algorithms and `apps/loPredCtrl` remain untouched.

The operational request-domain model observes its **own** correction source with
reviewed full-image P/R, excluding the disturbance. Total-command diagnostic fits
remain available but cannot authorize physical output. No publication or receipt
is reclassified as a physical input certificate. Zero corrective history is
unexcited: reuse the original Gaussian count/strength convention only on an
explicit authorized probe request, with separate approved modal and per-pixel
limits. Learning never implicitly starts exploration. Faults do not rearm.

`learn.toggle` trains both independent banks; OFF freezes fit, not prediction.
`predict.toggle` requires qualified selected model; OFF requests own zero.
Reset/mode switch requires output OFF and removed. Physical mode does not silently
fall back during the standard/assisted comparison; invalid selected input closes
output and requires operator recovery. Existing diagnostic shadow fallback is
retained. Queued scores carry model epoch, and old reset-era work is rejected.

Native input recording and scored-origin recording are optional. A zero score
queue capacity disables exporting score rows while retaining INDI RMSE. The new
mock tests use process-private ordinary files/API doubles, not shmims or services.

## Files

- `libMagAOX/app/syncoOutput.hpp`: common existing-base output lifetime, startup
  guard, bounds, own zero, receipt, timeout/reset/exception policies.
- `apps/loPredCtrlAcc/loPredCtrlAcc.hpp`, `_native.hpp`, `Makefile`: existing
  INDI controls, runtime gates, declared correction-input domain, explicit probe,
  native-only build profiles, install/profile dependencies.
- `online_shadow.*`, `phase_shadow.*`, `multirate_shadow.hpp`,
  `shadow_controller.*`, `synco_shadow.*`: operator fit/freeze/reset, epoch scoring,
  fresh scheduler status, strict selected-mode cache, unchanged RLS/QR kernels.
- Imported `apps/accelSim`: runtime/build/docs adapt the existing simulation;
  `accelSimDM_test.cpp` expects private preparation instead of direct app writes.
- `tests/ddspc_output_test.cpp`, `synco_transport_test.cpp`,
  `operational_platform.cpp`, `operational_config_test.py`, `Makefile.offline`,
  existing phase/native app tests: finite/slew/authorization/removal policies,
  unchanged real producer mock, controls, disabled configs and regressions.
  Confirmed replay now defaults to the imported local model, with the existing
  external `MAGAOX_REPO` override retained.
- `examples/operational/{acc000,loPredCtrlAcc}.conf`, simulator `acc000.conf`:
  fail-closed complete parameter templates; no invented instrument values.
- `OPERATIONAL_DDSPC.md`, historical report pointers, simulator README and
  `tests/groups.dox`: current operator/deployment/safety documentation.

## Validation record

Local compiler: Apple Clang on macOS, Eigen 5. No Linux VM available (`limactl
list` reports no instances), no Docker executable, no instrument build attempted.
Actual producer header is compiled against explicit API doubles; installed native
Linux ABI/link and actual accelSim lifecycle remain unqualified.

Initial broader regression passed 187 cases / 9,542,729 assertions. Subsequent
focused tests added explicit finite/slew guard coverage; final counts/results
are appended after completion below. Initial producer protocol passed 3 cases /
5,126 assertions, including both RLS and QR, standard and assisted selections,
simulator scale0/nonzero, independent other-source preservation and own zero.
Portable simulator model/synthesis tests passed 22 cases / 296,584 assertions.
These are synthetic/local software tests, not optical control results.

Testing found and fixed: incorrect mock geometry for a 2500-pixel image; rejected
configuration leaving Legacy entry points callable in an operational profile;
wrong local offline probe link expectation for Linux-only clock_nanosleep; stale
pre-reset status/scores; and mandatory external-model checkout no longer needed
once accelSim is imported. No failures are concealed as native success.

A development-host O2 phase timing run (2400 measured ADC calls, 2500-pixel
private images, H=2/F=2/two modes) reported stride1 median11.75 us,
p99=13.459 us, observed max75.375 us against 125 us. WFS numerical jobs (600)
median136.792 us, p99=149.708 us, observed max238.667 us against the nominal500 us
WFS period. Private producer-copy callback (no actual transport) median0.292 us,
p99=0.833 us, max8.917 us. Unchanged simulator synthesis/preallocated mock mean
2.02735 us, max60.209 us. These observed host timings include OS variability and
are neither WCET nor native scheduler/SDK deadline qualification. Do not assume
125 us minus ~110 us DM write time is the available numerical budget.

## Remaining commissioning requirements

Read OPERATIONAL_DDSPC.md for build profiles, exact commands, ownership/handover,
config values and operator sequence. A separately authorized test must establish
both actual aux routes and additive combination; actual reference/timing names,
source/cycle anchors, calibration P/R and matching TT/mask; approved modal/pixel
limits and transport timeout; native Linux build; actual source acquisition and
complete request-history coverage under8k output; actual producer callback/copy/
ack deadlines; explicit probe excitation if needed; zero/enable/disable/fault/
shutdown behavior of each owner independently. No default configuration bypasses
any of these. Missing consumer/crash removal remains unconfirmed and requires
operator recovery, not invented physical application evidence.

## Final handoff checkpoint

Ownership/concurrency work is complete. The private mock verifies the existing
producer lock prevents both payload and ready-sidecar writes while a competing
owner holds it. Recovery after releasing that lock sends only removal zero,
without automatic reactivation. Startup state/callback installation is serialized
against early control/status calls. Both application destructors stop captured
output callbacks while their application state is still alive, including failed
startup; normal shutdown retains explicit zero/removal diagnostics.

Final completed local runs:

| Suite | Result |
| --- | --- |
| Full offline regression | 189 cases, 9,540,906 assertions passed |
| Full ASan/UBSan regression | 189 cases, 9,545,459 assertions passed; no sanitizer findings |
| Actual producer/private mock, including competing ownership | 4 cases, 5,141 assertions passed in normal and ASan/UBSan builds |
| Operational-profile native controls/output policies, including concurrency | 15 cases, 224 assertions passed in normal and ASan/UBSan builds |
| Native observation/application/ADC | 60 cases, 14,334 assertions passed |
| Multirate/native pipeline/model/safety | 65 cases, 7,278,319 assertions passed |
| Confirmed-input replay | 6 cases, 40,049 assertions passed |
| Strict shadow application | 22 cases, 10,364 assertions passed |
| Simulator model/disturbance/synthesis | 36 cases, 308,607 assertions passed |
| Native timing tools | 49 Python tests passed |
| Disabled operational configurations/build profiles | 2 Python tests passed |

Asynchronous assertion totals vary between runs. Existing source provenance,
report, deployment and build-orchestration Python checks also pass. Source
integrity and git diff whitespace checks pass; nothing is staged or committed.
The complete imported accelSim directory must be included in the user's later
review/staging, together with the new common header and test support files.
No original loPredCtrl, ADC acquisition, dmSpeckle or existing producer/timing
header was modified. No instrument process or stream was accessed.

A later normal-build run under parallel workstation compilation load measured
the 2500-pixel ADC path median13.167 us, p99=16.667 us, max157.125 us; the small
positive-control fixture also observed203.375 us. Thus observed host tails can
exceed125 us despite a low median. Its WFS jobs measured median153.792 us,
p99=247.542 us, max382.375 us. Private prepared-image lookup/copy measured
median0.291 us, p99=0.416 us, max1.792 us. These do not include actual Linux
producer/SDK transport or establish real-time deadlines. The producer currently
uses an ordinary thread; native scheduling/affinity and timing must be qualified
before physical8k commissioning, without assuming15-us slack is a compute budget.

The final configs are deliberately disabled templates. Missing actual reference,
timing source, mask, modal P/R, source/cycle anchors, approved limits/timeout and
route evidence must be supplied by the instrument team, not guessed locally.
Actual Linux application compilation (especially the accelSim native lifecycle),
DM routing/additivity, optical response and controlled enable/removal remain
unverified. The local Linux build helper correctly refuses this macOS host.
No further optimization or development cycle is part of this handoff.

## Simulator producer-default simplification

Follow-up to checkpoint f05f10d4: the user explicitly requested removing accelSim's
mandatory max_abs_um, max_step_um and ack_timeout_us settings. This supersedes the
simulator-specific limit requirements above; loPredCtrlAcc remains unchanged.

Reviewed local bmcCtrl and the locally available syncopation branch bb5a5d89:
BMC conversion clamps the combined normalized squared voltage to [0,1], but no
per-source step limit was established. dmSpeckle uses finite commands and matching
producer acknowledgements, without these three mandatory settings. No local
executable/config mapping establishes the identity of tweeterMod; ttmModulator
controls a different function generator and was not substituted for it. Detailed
source references are in OPERATIONAL_DDSPC.md's simulator policy section.

Minimal change: an application-only producer-default selection in SyncoOutputConfig
(default false) is enabled only by the accelSim constructor. It skips extra
per-source bounds and permits base-only acknowledgement waiting (no guessed timer).
The source buffer cannot be reused before a matching copy/discard receipt. Finite
checks, compile/output authorization, qualification, ownership, exact deadlines,
fault latching and own-channel zero removal are preserved. DDSPC still requires
all three configured positive values, including independently tested missing-field
cases. No INDI/config switch exposes the simulator policy to DDSPC.

Retired simulator keys are consumed as ignored optional strings for existing-file
compatibility. Removed them from both acc000.conf copies and the ADC-only example.
The operational config test's prior requirement that every timingStream be blank
was stale (the controller example now names dmtweeter_synco_timing). It now checks
that a name does not grant output/association authority, without changing the
configured timing stream or the controller's limits.

Focused normal and ASan/UBSan runs both pass the real-producer mock (4 cases,
10,152 assertions; zero/nonzero initial scales) and native controls/output policy
suite (16 cases, 250 assertions), plus 2 Python configuration tests. The mock
publishes a nonzero scaled simulator surface with all three settings zero, removes
it with a matching zero receipt, and preserves the independent DDSPC source.
No native Linux application or instrument was run. Shutdown remains bounded;
missing consumer receipts cannot be reclassified as successful physical removal.
The portable simulator model/disturbance/synthesis regression also passes 36 cases
and 308,607 assertions after this change. Simulator numerical headers, both
acquisition applications, original loPredCtrl, loPredCtrlAcc's production source
and configuration, and the existing producer/timing headers are unchanged.
Only the shared helper's explicit simulator-policy branch changes output policy;
there is no change to the slot protocol or DM channel assignments.
