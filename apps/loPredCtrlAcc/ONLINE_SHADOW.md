# Qualified online multirate DDSPC shadow path

This change adds a bounded native-input fanout and a separate numerical worker using
the existing `PredictiveController`, RLS, QR-RLS and regularized DDSPC solve. It does
not add an EOF predictor, a command publisher, or a way to qualify telemetry through
configuration or INDI. The installed/default build remains `DDSPC_SHADOW_ONLY`.
`apps/loPredCtrl` is unchanged. The default configuration leaves this worker disabled,
preserving continuous ADC/native observation.

**The currently observed native streams are insufficient to admit training.** Their
receipt rates do not establish WFS exposures, camera/modal association, a common
source clock, calibrated total applied modal input, or physical DM hold intervals.
No current producer calls the new qualified-record adapter. Enabling the worker
therefore still reports zero model updates and precise interlocks. This is a prepared,
tested learning path, not a claim that current telescope data qualify it.

## Source audit and reusable implementation

| Component | Relevant implementation | Result |
| --- | --- | --- |
| Operational baseline | `../loPredCtrl/ar_controller.cpp`, `update_system()`, `update_controller()`, `calculate_command()` | Existing historical history ordering, relative penalty and extra feedback remain untouched. |
| Native causal DDSPC | `ar_controller.cpp`, `append_sample()`, `prepare_training()`, `learn_prepared()`, `reference_controller()` | Same existing RLS/QR updates and regularized least-squares control; independent command width now supports a rectangular response matrix. |
| Independent model manager | `shadow_controller.cpp`, `process()`, `step()`, `predict_next()` | Independent learners, normalized empirical rank, finite-value gates, frozen-origin scoring; next-origin prediction never inserts an unconfirmed input. |
| Native assembler | `multirate_shadow.cpp`, `qualify()`, `origin()`, `applied()`, `complete()` | Exact source bindings; individually timestamped ADC features; measured WFS only; certified input coverage. |
| Worker | `online_shadow.cpp`, `Impl::run()` | Numerical work on its own ordinary-priority thread; bounded input queues and soft duty budget. |
| Application | `loPredCtrlAcc.hpp`, `configureShadow()`, `qualifyNativeShadow()`, `processNativeOrigin()`, `processNativeApplied()` | Typed, non-actuating qualified-input adapter seam; `ddspc_mode` requests effective selection without resetting either model. |
| Raw acquisition | `native_observer.cpp`, `Impl::enqueue()` | Nonblocking owned-record SPSC fanout, including all individual ADC samples; no Eigen work in acquisition. |

The pre-existing causal shadow convention is **not** bit-for-bit the operational
Legacy control law: its history corrections and literal command penalty already
differed from Legacy's relative penalty and extra feedback. This change preserves
Legacy and the single-slot defaults, rather than silently declaring those laws
equivalent. Native shadow readbacks cannot be used as an operational controller
comparison until that distinction is accounted for.

## Multirate identification and candidates

Let `m` be the number of measured calibrated modes, `N` the WFS history, `F` the
WFS target horizon and `Ha` the ADC history in WFS periods. At frame origin `i`,
`y_i` is one measured WFS vector. No intermediate WFS measurements are invented.
The four confirmed input vectors `u_i,0 ... u_i,3` correspond to successive nominal
125-us holds. Define `U_i=[u_i,3;u_i,2;u_i,1;u_i,0]` of width `nu=4m`.

Each assisted ADC record is `a_i=[a_3,x;a_3,y;...;a_0,x;a_0,y]`: four distinct
timestamp-selected channel pairs in the preceding frame. Features use the verified
grid phase, not a camera publication marker. `Ha` such blocks preserve `4Ha`
individual ADC samples. Neither averaging nor decimation occurs.

At a matured target `t=i+F`, existing causal DDSPC learns

```text
Yf = A Yp + B Up + D Ap + C Uf
Yf = [y_(i+F); ...; y_(i+1)]               : Fm
Yp = [y_i; ...; y_(i-N+1)]                : Nm
Up = [U_(i-1); ...; U_(i-N+1)]            : (N-1)nu
Ap = [a_i; ...; a_(i-Ha+1)]               : 8Ha (assisted only)
Uf = [U_(i+F-1); ...; U_i]                : Fnu
Theta canonical = [A B D C]               : Fm × ((F+N-1)nu+Nm+8Ha)
Theta internal = [C B D A]
C                                        : Fm × Fnu (rectangular)
```

The unchanged learners update the same multi-output regression. For RLS, with
regressor `phi`, target `Yf`, forgetting factor `gamma` and covariance `P`:
`g=P phi/(gamma+phi^T P phi)`, `Theta <- Theta+(Yf-Theta phi)g^T`,
`P <- (P-g phi^T P)/gamma`. QR-RLS uses its existing corrected information-factor
updates. Prior regularization is not evidence of empirical rank.

The unchanged reference cost is `||M x+C Uf||²+lambda||Uf||²`, `M=[A B D]`.
The solve is `(C^T C+lambda I) Uf=-C^T M x`. Already committed oldest slots are
fixed, separately evidenced inputs: the same solve uses free columns of `C` and
adds `C_fixed U_fixed` to the known residual. Committed prefixes are never fed
back as physically applied training data. Later training requires separate closed
physical-input certificates for every slot.

`online_shadow.standard_slots=1` retains the original single-rate history
dimensions and causal model/law. It admits a frame only when all four confirmed
inputs are exactly identical. A single-rate model cannot represent a partially
committed frame: it can propose an uncommitted whole frame or, with `F>=2` and a
fully committed held current frame, the next whole frame. No input averaging is
used to hide within-frame changes.

An explicit `standard_slots=4` instead supplies a WFS-only predictor in the same
lifted command space as the assisted predictor. This is the appropriate matched
offline comparison when only ADC regressors should differ; it is a different
identification dimension from the operational single-rate baseline. Tests verify
that changing/dropping ADC features leaves the standard coefficients unchanged.

Readiness requires complete contiguous histories, enough actual updates, empirical
full rank and finite coefficients/gains. Normal 2-kHz AO input held across four slots
makes those four command columns dependent. Such telemetry cannot identify all
four slot responses: the assisted model must remain unexcited. No artificial
excitation, regularization-based readiness, fabricated commands or WFS interpolation
is introduced to defeat this gate.

This first path computes a lifted plan at each qualified WFS origin, using all
individual ADC history samples available then. It does **not** yet replan after
each newly acquired ADC sample. Targets remain measured 2-kHz WFS residuals;
there is no measured validation of intra-exposure optical residuals at 8 kHz.
Four future slot values are diagnostic candidates, not 8-kHz deadline qualification.

## Timing/provenance gates

Existing detailed evidence is retained in [NATIVE_TIMESTAMP_PROVENANCE.md](NATIVE_TIMESTAMP_PROVENANCE.md).
Key source findings in this checkout:

- `libMagAOX/app/dev/edtCamera.hpp:549-558`: `pdv_wait_last_image_timed()` supplies
  an SDK frame/DMA marker. The wrapper does not establish exposure start/end.
- `libMagAOX/app/dev/frameGrabber.hpp:1018-1036`: REALTIME software publication,
  inherited acquisition marker, independent `cnt0` and circular-plane `cnt1`.
  Publication is not an exposure boundary; `cnt1` is not a modal parent counter.
- `apps/nnReconstructor/nnReconstructor.hpp:702` uses a callback REALTIME marker
  in an alternate reconstruction path. The conventional CACAO/MILK producer and
  its clock/build/host must be identified for the actual run; stream name and
  similar numeric timestamps do not establish them.
- `apps/bmcCtrl/bmcCtrl.hpp:435,509`: `BMCSetArray` submission/return does not
  measure optical application. `dm01disp` remains a read-only combined **request**.
  Optional software/SDK endpoints are diagnostic and cannot create an applied
  certificate. An idle DM stream remains a healthy attached observation source.

`MultirateContract` must bind exact file/device/image/generation identities,
reviewed clock epoch and uncertainty, calibration/layout, true exposure duration
and grid offset, real ADC phase, camera/modal parent association, and independent
applied-input provenance. Its reference string/booleans are a programming contract
for a reviewed adapter, not authentication or a user-facing override. Current code
deliberately provides no INDI/config flags that manufacture these facts.

`OnlineOrigin` must identify an actually retained modal and camera record, bind the
observed calibrated residual, supply explicit exposure boundaries, freeze the
decision time and only include features whose acquisition/availability/receipt
precede that time. ADC acquisition phase must fit the configured jitter plus clock
uncertainty, with unique consecutive samples in the same source segment. Missing,
late, ambiguous, stale or out-of-order data invalidate history/readiness.

`OnlineAppliedSlot` must independently certify total after-limits calibrated modal
input and a closed effective 125-us hold interval. Every frame requires four exact,
non-overlapping intervals; static command counters may be referenced by multiple
independent hold certificates. The adapter must establish completeness of all
other AO/DM channels and the projection/sign/units. This strict first adapter does
not approximate variable transition times, reconstruct unknown optical delays or
integrate ambiguous exposure coverage. Inherited `InputConfig` response-lag and
minimum-command-delay offsets must be zero: boundaries must already describe the
reviewed physical convention, and actual cutoffs carry scheduling delay. Unsupported
nonzero offsets are rejected rather than ignored.

Training waits for actual command coverage; candidate generation uses only a prior
frame already confirmed **by the frozen origin**. Late proofs cannot be backdated
to generate a seemingly causal prediction. Queue/source losses break histories.
Common source identity or clock changes revoke the contract; ADC loss invalidates
only assistance. A new trusted contract resets both fits and requires new training.
Common-input recovery requires a new complete contiguous history and rank check.

An assisted request falls back to standard only when the standard model has a valid
plan for the same timing/schedule convention. Otherwise no candidate is exposed;
there is no held shadow command or actuation safe-command policy. Switching changes
diagnostic selection only; neither fit is reset. There is no slew-limited physical
handoff in this non-actuating path.

### Future synco interface

The read-only review of the newer sibling MagAO-X worktree finds
`apps/dmSyncopation/dmSyncopation.hpp` and
`libMagAOX/app/dev/dmSyncopationProducer.hpp`. Its immutable commit/prefix and
input-ready cutoff are the future interface; none of its publishing code is linked
into this worker. Installed build identity and timing must still be established.

Origin packets carry actual cutoffs for four current slots and, when required,
the next four. They must be ordered and no later than their nominal slot. A plan
completed/read after its earliest free input-ready cutoff is invalid. The roughly
110-us write and 125-us slot leave about 15 us nominal slack, **not** a computation
budget. The current 5-ms soft worker period is deliberately conservative and cannot
guarantee those deadlines. Qualified fast inference/per-ADC replanning and actual
synco schedule telemetry are separate remaining work. No direct or unscheduled DM
writes may be added to this path.

## Resources, diagnostics and safety

The native acquisition thread copies fixed observations into one bounded SPSC queue.
It never calls RLS/QR or takes a numerical-model mutex. Qualified WFS/input records
use a bounded try-lock mailbox; contention/full queues count as loss and break history.
The worker requests `SCHED_OTHER` at its minimum normal policy priority and never
changes other threads/processes or affinity. Defaults: 2,048 raw queue, 4,096 ADC
retention, 32 pending origins, 5,000-us period, 1,000-us soft budget and stride four.
Every history row is retained; adaptation, empirical-rank and controller refresh occur
every fourth record. `gamma` is per actual update; cached rank is at most three rows
old and any input break invalidates readiness immediately. Set stride one for the
original per-record numerical behavior, subject to measured resource limits.

Numerical dimensions are capped (one/two modes, at most 128 regressors, bounded
histories/horizons). Eigen threading is disabled by existing build/test conventions.
Individual matrix solves are nonpreemptible: this is a soft quota, not a hard CPU or
WCET limit. A single numerical job exceeding the quota latches a worker fault and
stops further model jobs until restart. Restart clears provenance/readiness; reload
cannot silently alter a running online session. Shutdown stops the native observer
before joining/destroying its borrowed numerical sink, then stops ADC acquisition.

Compact INDI stays at 11 properties / 60 elements, refreshed at the existing
250-ms default. Existing `ddspc_control`, `ddspc_performance`, `timing` and `safety`
report actual independent states/update counts/scored forecast counts, conditional
RMSE, last numerical latency and precise inhibition reasons. `input_buffers.online_queue`
adds a single text summary of raw/qualified backlog and drops. Until evidence exists,
expect `training=inhibited`, both ready flags false, updates/predictions zero,
`model_interlock=source_host_mapping_unqualified`, and `dm_write_enabled=false`.

RMSE scores frozen coefficients/state against later observed WFS and independently
confirmed future inputs, before training on that target. It is **conditional,
prequential** prediction error: the future AO input can itself depend on later WFS
measurements. It is not an unconditional forecast, proof of causal system
identification under closed-loop excitation, disturbance rejection, or a prediction
of how un-applied shadow candidates would have performed. Correlation/endogeneity,
frequency-specific model bias and optical response require separate validation.

No physical output path, Legacy activation, stream creation or hardware access is
added. The Makefile still forces `DDSPC_SHADOW_ONLY` and strict finite-value checks.
Tests call the real application header with private framework/transport doubles;
output entrypoints refuse writes and sentinels remain unchanged.

## Local verification and next qualification steps

From the repository root, on a personal/development machine only:

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 check-multirate
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 check-native-observation
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 check-confirmed-replay
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 \
  BUILD_DIR=build/online-shadow-sanitized \
  CXXFLAGS='-O1 -g -std=c++20 -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check-multirate
```

The deterministic fixtures exercise actual native DDSPC, rectangular indexing,
both RLS and QR updates, known causal coefficients, independent model selection,
loss/recovery, timestamps/late delivery, full-rank gates, immutable prefixes, missed
cutoffs, common/source/clock restarts, stride, resource bounds and real application
lifecycle/output guards. Synthetic certificates are explicitly accepted only by
isolated engine instances. The production worker rejects them.

`examples/loPredCtrlAcc-online-shadow.conf.example` enables only bounded processing,
not qualification. Do not copy it over operational configuration or start/restart an
application while observers are operating. No deployment or instrument command is
part of this milestone.

Before valid online training: establish the installed writer/host/clock mappings;
obtain true exposure/parent metadata and calibrated residual selection; obtain an
independent complete applied-input/response timing source; implement a reviewed
adapter that supplies those records and immutable synco schedules; verify empirical
slot excitation and full rank; then benchmark actual Linux acquisition/learning and
deadline behavior under an authorized resource budget. If these records cannot be
obtained, retain the interlocks. No config change can substitute for them.

## Validation performed for this change

Local macOS / Clang / Eigen 5.0.1, strict finite-value math, 2026-11-14. No native
application, operational stream, INDI connection or instrument service was run.

| Check | Result |
| --- | --- |
| Full offline numerical/application regression | 145 cases, 2,269,843 assertions passed. |
| Native acquisition, ADC, INDI, lifecycle and compile-time output guards | 57 cases, 14,297 assertions passed. |
| Existing confirmed-input replay using the sibling accelSim numerical header only | 6 cases, 40,049 assertions passed. |
| Native observation tools / timestamp audit / configuration-deployment / Linux-build orchestration tests | 3 + 28 + 5 + 9 = 45 Python tests passed. |
| Focused multirate and real-header lifecycle tests under ASan + UBSan | 19 cases, 5,085 assertions passed; no sanitizer findings. |
| Legacy source preservation | All 15 reference files match byte-for-byte; no changes under `apps/loPredCtrl`. |
| Whitespace and changed C++ formatting | `git diff --check` clean; clang-format applied to changed C++ files. |

The unpaced two-mode QR fixture (`N=5`, `F=3`, `Ha=5`, matched four-slot standard,
stride four, 1,200 records) measured 40.820 us average and 297.125 us maximum wall
time per synthetic WFS record, including raw-record assembly and both models.
Those are whole-frame averages, mixing adaptation/non-adaptation records, not
individual solver WCET or an 8-kHz inference measurement. Sanitizers increased this
to 737.708 us average / 4,162.88 us maximum. Timing is informational and has no
pass threshold; nothing in this run qualifies a real-time deadline or resource
budget on exao2. Threaded test assertion totals can vary with polling iterations.

The actual native Linux/GCC executable was **not** compiled/linked on this Mac:
real Linux ImageStreamIO/MagAO-X runtime dependencies are unavailable here. The
application-header tests use explicit framework/transport doubles and real C++
controller/worker/INDI code. Passing Linux-build orchestration tests is not a real
Linux build result. A later approved development-host build/link and review of
its dependency versions remain required before deployment.

## Affected files

All paths below are relative to `apps/loPredCtrlAcc`, except the plan:

- New `multirate_shadow.hpp`, `multirate_shadow.cpp`: qualified individual ADC and
  physical-input assembly, source/clock/loss gates, independent plans.
- New `online_shadow.hpp`, `online_shadow.cpp`: bounded native fanout consumer and
  ordinary-priority numerical worker.
- `ar_controller.hpp`, `ar_controller.cpp`: independent command/output widths,
  rectangular existing solve and next-origin state construction.
- `shadow_controller.hpp`, `shadow_controller.cpp`: independent input lifts,
  explicit next-origin forecasts, fixed prefix and optional update stride.
- `native_observer.hpp`, `native_observer.cpp`: optional nonblocking owned-record sink.
- `loPredCtrlAcc.hpp`, `loPredCtrlAcc_native.hpp`: startup-only worker option,
  qualified adapter seam, lifecycle, selection and compact status.
- `Makefile`: new objects and forced single-thread Eigen math in shadow builds.
- `loPredCtrlAcc.conf`, new `examples/loPredCtrlAcc-online-shadow.conf.example`:
  disabled installed default and explicit development-only opt-in example.
- New `tests/ddspc_multirate_test.cpp`, `tests/native_application_test.cpp`,
  `tests/native_deployment_test.py`: numerical, safety, actual worker/app and config checks.
- `tests/Makefile.offline`, `tests/Makefile.replay`: link actual worker/assembler,
  dedicated multirate test target, preserve older replay targets.
- New `ONLINE_SHADOW.md`, `NATIVE_DIAGNOSTICS.md`, `NATIVE_APPLICATION.md`,
  `tests/README.md`: architecture, precise limitations, 60-element schema and verification.
- New repository-root `agents/plans/2026-11/accel-ddspc-online-shadow.md`: engineering record.
