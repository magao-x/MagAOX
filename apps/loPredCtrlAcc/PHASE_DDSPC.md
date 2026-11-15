# Native 2-kHz learning and ADC-driven 8-kHz shadow prediction

2026-11-15, following checkpoint `70990414`. This supersedes the WFS-origin-only
native path in `SYNCO_NATIVE.md`. No producer, original `loPredCtrl`, ADC driver,
operational configuration or installed binary is changed. `DDSPC_SHADOW_ONLY`
remains mandatory; neither the auxiliary image nor its sidecars are opened.

## Two owners, one input contract

`PhaseShadow` is the live receiver selected automatically by `OnlineShadowWorker`
when native upstream inputs are enabled. The older `NativeInputPipeline` remains
an explicitly selected historical replay reference (`m_sampleDriven=false` in
its test fixture); that switch is not exposed as an application control.

The native observer first checks complete DM requests, WFS and the existing synco
record, then forwards individual ADC samples through the existing bounded cursor.
Its acquisition owner performs only fixed-history work, cached matrix products,
the calibrated full-image/request projection, and private payload preparation.
It never calls RLS/QR, rank checking or a control factorization. All numeric
adaptation remains on the bounded ordinary-priority learning worker.

Each genuine WFS publication advances measured WFS history **once**. Four phase
contexts retain the individual ADC histories that were available at their own
decisions. A closed WFS/request record goes through a bounded SPSC handoff. The
numerical owner runs the existing native RLS/QR kernels, empirical excitation
checks and gain preparation. Immutable snapshots exchange under a short mutex;
the acquisition owner uses try-lock and skips if unavailable. No learning work is
done under this lock. An unchanged validated fit reuses its cached gains.

The complete-record interface conservatively waits for the record's request
history to close before passing it to the learner. This can add approximately one
WFS interval to adaptation; it does not delay using the newest actual WFS value
for fast inference. No placeholder command or repeated WFS observation is inserted
to advance the learner earlier. `training_stride` applies to real WFS records,
not ADC arrivals; gamma retains its meaning per actual weight update.

The old consumer-side 4:1 estimate from asynchronous `timingDiag` snapshots is
removed. There is no clock mapping, exposure reconstruction, phase estimator,
PLL or acquisition-delay adjustment in this live path. Producer validity,
counter/segment/source continuity, fresh local availability and a reviewed
counter association are required. Acquisition synchronization belongs entirely
to ad7689Ctrl (or the explicitly identified simulator producer).

## Phase convention and regression

Let k be a **real modal WFS counter**, p in {0,1,2,3}, m the selected mode count,
H the WFS history, F the WFS prediction horizon and Ha the configured ADC history.
The reviewed consumer anchors define

```
k = modal_anchor + floor((adc_counter - adc_anchor)/4)
p = (adc_counter - adc_anchor) mod 4
```

`adc_anchor` denotes the first consumer phase following that modal origin; it
need not denote the producer's physical reference-slot zero. The latest actual
modal frame must match k before acceleration can be used. This consumes a
counter contract rather than fitting phase from receipt/publication times.
Producer-side resets or skipped nominal slots invalidate this association;
reviewed fresh anchors/source incarnation are required, not clock reconstruction.

At p, A(k,p) is the newest-first list of the last **4 Ha individual X/Y pairs**,
dimension 8 Ha. It includes the current ADC pair and causally available earlier
pairs, including earlier WFS intervals. It does not wait for later phases of k.
There is no averaging. Newest ADC receipt must be no more than 250 us old and the
latest WFS receipt no more than 1 ms old. Thirty-two consecutive complete samples
are required for assisted recovery. Historical samples in A are history, not new
measurements; continuity and source/segment identity are checked.

For the matched four-slot comparison, q=4m. All stacks below are newest-first:

```
Yp(k) = [y(k), ..., y(k-H+1)]                    dimension Hm
Up(k) = [U(k-1), ..., U(k-H+1)]                  dimension (H-1)q
Uf(k) = [U(k+F-1), ..., U(k)]                    dimension Fq
Yf(k) = [y(k+F), ..., y(k+1)]                    dimension Fm

standard:  Yhat_f = A Yp + B Up + C Uf
assisted:  Yhat_f = A_p Yp + B_p Up + D_p A(k,p) + C_p Uf
```

One WFS-only learner is shared across phases. Four assisted views use the same
RLS/QR framework but their own phase-specific regressors/coefficients; shifting
ADC windows are therefore never applied to coefficients identified at a different
phase. Each assisted view receives the same genuine WFS targets and command
records. All four fits must qualify before assistance is selected. The standard
fit is unaffected by ADC features, mode selection or an ADC-only interruption.
`C` is rectangular, Fm by 4Fm. WFS history is never upsampled.

Training uses only complete original command requests. The four request values
are reconstructed at the **existing synco record's advertised input-ready
cutoffs**, with exact retained counter/hash identity and a coherent command-poll
witness after each cutoff. Idle coherent polls are healthy. Busy/racing/missing
polls do not certify coverage. This replaces the observer's own 125-us sampling
grid. No shared-memory publication or acknowledgement is called physical DM
application. The declared model input remains the observed-request domain.

ADC phase p selects the assisted coefficient view. Independently, j is the number
of original command slots whose advertised cutoffs have closed with coherent
request coverage. Delivery latency can make j differ from p. Their original
observed values form a fixed prefix. Partition C into free and fixed columns and
define z=[Yp; Up; A(k,p); fixed_prefix_j] (omit A for standard). The cached
free-input gain is

```
K_(p,j) = -(C_free' C_free + lambda_eff I)^(-1) C_free' [A_p B_p D_p C_fixed]
U_free = K_(p,j) z
```

The numerical owner prepares prefix gains for j=0,...,4 (j=4 requires F>1).
Fast inference selects the oldest remaining modal command and adds the same
configured current-WFS feedback to both models. It renders only the one eligible
full-image candidate, not all future images. A delayed phase-zero ADC event may
prepare a fresh slot-one candidate using K_(0,1); it cannot reuse or retag an
expired slot-zero command. The actual next Request must agree with the observed
prefix before a payload is prepared. Original observed requests, never these
proposals, enter training.

The primary WFS forecast uses the same frozen coefficients/state: known prefixes
retain their observed values and unknown future requests explicitly hold the
latest available requested value. Actual later inputs do not enter that forecast.
Scores are emitted only when the real target WFS counter arrives. Four phase
forecasts of one target do not create four WFS measurements. Origin, phase, ADC
counter, coefficient version, target, decision/completion times, candidate and
actual slot cycle/step/epoch/generation/cutoff are recorded. ADC phase is distinct
from scheduler step. No conditional future-input metric is computed by this path: CSV marks
`conditional_valid=0` and writes NaN to that column.

## Standard baseline correspondence

Both native banks now use the operational penalty policy
`lambda_eff = regularization * maxCoeff(C' C)`, with the same gamma, prior,
stride and readiness rule, plus the same configured `parameters.gain * y(k)`
feedback. Scaling uses each model's own C: the configured penalty policy is
identical, while its absolute scaled value can differ when the identified C
differs. Givens QR/covariance RLS equations are unchanged. Stable LDLT replaces
an explicit matrix inverse without changing the stated quadratic minimizer.
Historical causal replays retain their literal penalty default.

`standard_slots=1` requires proof that all four observed slot requests were held
equal; a pre-WFS request cannot stand in for a varying command frame. Only its
first slot can prepare a new hypothetical held-frame payload; later phases still
provide WFS forecasts but cannot modify a partially closed single-rate input.
`standard_slots=4` is the matched
lifted WFS-only baseline for an ADC information comparison. The latter is a
multirate extension, not an exact operational loPredCtrl clone.

Even the single-slot shadow model is not bitwise operational closed-loop replay:
causal explicit histories replace Legacy's callback/command offsets; the causal
future block has F inputs rather than Legacy's F-1 block; inputs are independently
observed combined requests rather than loPredCtrl's own proposed commands; and no
output integration, exploration or publication runs. These differences are
explicit. The original Legacy-compatible branch and `apps/loPredCtrl` are unchanged.

## Synco and fault behavior

The unchanged `dmSyncopationProducer` base from Jared's branch remains inherited.
The future identity remains `dm10disp10_aux`; the actual installation mapping
must still be reviewed. Shadow mode never polls/publishes through the base.

Phase ordering follows **absolute advertised targets**, not numerical step labels.
The production order can be 1,2,3,0. Existing exact epoch/generation/cycle/step/
target/cutoff checks remain. The private cache is prepared before mock Request
readout. Missed requests are skipped, not retagged. Model work is never moved into
the producer callback and 110-us writes do not provide a 125-us computation budget.

The single-slot cache uses actual advertised targets/cutoffs and accepts nominal
2-kHz periods of 450--550 us with consistent quarter-frame spacing (integer-ns
rounding allowed). It does not require an exact 500,000-ns software period. A
changed first chronological step makes the current WFS/request window ineligible
and revokes private payloads. Windows can recover when the declared association
returns; a persistent new association needs a reviewed configuration/restart.
The retained historical whole-frame replay cache still uses its original exact
500-us/125-us grid.

Each new valid ADC counter creates an opportunity; duplicates do not. Qualification,
snapshot contention, stale inputs or an expired cutoff can inhibit it. Standard
fallback can also use actual new synco Requests during ADC loss, without adding
an ADC sample or a WFS training record. Such scores have `adc_counter=0`. Input
loss revokes old caches; changed WFS/DM source incarnations require a new reviewed
contract. Numerical budget faults remain latched until worker restart.

## Configuration and commissioning requirements

Default acquisition/diagnostics and `online_shadow.enabled=false` are unchanged.
Use `examples/loPredCtrlAcc-online-shadow.conf.example` only with a reviewed
full-image P calibration, selected WFS modes/units/signs, source reference and
counter anchors. R is optional for forecasts and required for private images.
For four-slot request histories, also supply the existing timing stream and the
reviewed `modal_counter + synco_cycle_offset == advertised_cycle + 1` relation,
then enable `native_inputs.synco_association`. This is not output authorization.

For the real producer's current schema:

```ini
[native_inputs]
sync_device = ad7689Ctrl
sync_property = ddspcTiming
# calibration, reference, epoch and anchors remain explicitly reviewed settings.
```

This reads existing enabled/valid, frame_seq, reference_id, phase_acquisitions and
nominal_slot fields. Changes of nominal-slot/publication offset revoke the old
counter association. `sync_property=timingDiag` remains supported for accelSim's
existing synchronized mode/lock diagnostics. Neither INDI snapshot is promoted
to lossless per-frame telemetry. Timer mode, missing status or missing anchors
cannot enable assistance.

The 60-element INDI schema is unchanged. Its bounded `input_buffers.online_queue`
text adds ADC opportunities/predictions, per-phase counts and fast-path current/
peak microseconds. Existing properties report minimum assisted per-phase updates/
rank, aggregate readiness, origin-only RMSE, numerical-job timing and inhibitions.
Per-model RMSE coverage can differ: use the CSV intersection of origin, target,
phase, channel and horizon for performance comparisons, excluding unqualified or
late origins as specified by the experiment. Never compare unmatched subsets.

At F=2, two modes and two models, four phase origins require up to **64,000 scalar
rows/s**, or 3.84 million for a full minute. The example's five-million-row budget
covers warm-up plus a minute. F=8 can require 256,000 rows/s; the hard ten-million
row cap covers less than 40 seconds and recorder throughput must be qualified.
Reaching the limit or losing diagnostic rows never changes input/model state.
No ADC samples are decimated to reduce recording traffic.

The example uses `training_stride=1` for one adaptation per completed real WFS
record (nominal 2 kHz). Larger explicit strides reduce learning load without
removing source history or ADC prediction opportunities. The default disabled
worker retains its historical stride configuration. The timing fixture measures
both stride one and stride four; neither establishes native OS deadline compliance.

Before an authorized shadow test: make a fresh private GCC14/C++20 Linux build
against actual ImageStreamIO/current mxlib, review P/R/anchors and synco mapping,
check command-history completeness and actual slot excitation, and profile the
whole native observation/learning load. Held 2-kHz commands cannot identify four
independent command columns; the empirical-rank interlock is retained. The
injected command and ADC regressors can also be collinear; receiving nonzero
signals alone does not establish identifiability. The single-rate standard model
can still be used separately. No exposure/clock reconstruction or AD7689 change
is requested.

These results will not authorize physical corrections: requested-input versus
actual response phases, total versus disturbance/corrective input partition,
stroke/slew limits, ownership, actual producer transport deadlines and safe
handover remain separate requirements.

## Reproduce local verification

```sh
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/phase-shadow check-multirate
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/compact-regression check
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/compact-regression check-native-observation check-confirmed-replay check-shadow-only check-native-timing
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/native-pipeline-asan \
  CXXFLAGS='-O1 -g -std=c++20 -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check-multirate
make -C apps/loPredCtrlAcc -f tests/Makefile.offline -j2 BUILD_DIR=build/phase-full-asan \
  CXXFLAGS='-O1 -g -std=c++20 -Wall -Wextra -fno-fast-math -fsanitize=address,undefined -fno-omit-frame-pointer' check
```

These are offline C++ tests using the actual ADC decoder, phase receiver,
DDSPC/RLS/QR and Request/payload logic, with explicit private transport doubles.
Virtual source timestamps are nominal 8/2 kHz; compute benchmarks measure actual
host execution. They are not an OS scheduling, native ABI or instrument test.

## Local numerical and timing results

On the local macOS workstation with Apple Clang/C++20 and Eigen 5, the deterministic
coherent-vibration fixture gave the following paired, origin-only prediction MSE.
Both modes use identical WFS targets, horizons and observed commands; only the
assisted views receive the ADC history. These are prediction errors, not physical
closed-loop residuals or vibration suppression measurements.
Scores use a fixed later evaluation interval while causal online learning
continues; each scored forecast was frozen before its target arrived.

| Native learner | Standard MSE | Assisted MSE | Relative reduction |
| --- | ---: | ---: | ---: |
| Covariance RLS | 0.0351479 | 0.0148544 | 57.74% |
| Givens QR RLS | 0.0351467 | 0.0148540 | 57.74% |

The unrelated-ADC negative control gives assisted/standard MSE=1.05517 (5.52%
worse). The standard coefficients are bitwise equal between the related and
unrelated runs. No target was used before its recorded availability. Stride-one
tests verify exactly one learning update per eligible real WFS target and no
coefficient changes between those updates. All four assisted phase fits and
single-slot/four-slot command indexing are tested separately.

The optimized full-suite timing fixture uses H=2, F=2, two modes, four individual
ADC pairs in each phase history and a 2,500-pixel private image. It times the
whole ADC observe call, including history/context construction, cached inference,
private rendering and mock payload copy; it excludes native transport/decoding,
DM projection on other input events, and numerical learning.

| Work | Samples | Median | 99th percentile | Observed maximum |
| --- | ---: | ---: | ---: | ---: |
| ADC prediction call, stride one | 2,400 | 11.834 us | 14.542 us | 48.417 us |
| WFS learning job, stride one | 600 | 137.250 us | 156.042 us | 234.666 us |
| ADC prediction call, stride four | 2,400 | 11.875 us | 14.416 us | 46.500 us |
| WFS learning job, stride four | 600 | 57.791 us | 84.959 us | 99.125 us |

The latest prediction maximum is below the nominal 125-us interval, but an earlier
development run observed an **832.125-us** host outlier. No measured
maximum establishes WCET, input-ready cutoff compliance or real-time readiness.
Numerical jobs run separately and need not fit a 125-us slot. These small-model
execution times do not establish sustained native throughput for deployment
dimensions. Instrumented ASan/UBSan builds are substantially slower (about 127 us
median per ADC call and 1.94 ms median per stride-one WFS job in the full suite)
and provide correctness checks only. The concurrent two-owner fixture processed 1,399 closed
WFS records and 5,600 ADC opportunities with zero handoff drops, using virtual
source times and bounded producer pacing; it is not a wall-clock 8-kHz stream test.

Final verification on the completed sources:

| Target | Result |
| --- | --- |
| Full offline regression | 183 cases / 8,969,496 assertions passed |
| Focused native/multirate/producer | 59 cases / 6,706,351 assertions passed |
| Native observation/application | 59 cases / 14,316 assertions passed |
| Confirmed-input replay | 6 cases / 40,049 assertions passed |
| Shadow-only application | 22 cases / 10,364 assertions passed |
| Focused ASan + UBSan | 59 cases / 6,706,351 assertions passed; no findings |
| Full ASan + UBSan | 183 cases / 8,973,933 assertions passed; no findings |
| Native timing Python | 49 tests passed |
| Report / producer Python | 3 / 4 tests passed |
| Observation tools / timing audit / deployment / Linux-build tooling Python | 3 / 28 / 5 / 9 tests passed |

Concurrent test assertion counts can vary with the number of available snapshots.
Output API sentinels remain zero. The actual INDI properties are compiled with
explicit MagAO-X/ImageStreamIO transport doubles; this is not a Linux native link
test. The existing INDI `getErrorMsg` temporary-string compiler warning is outside
the changed code. Formatting and whitespace checks pass. ASan/UBSan and bounded
concurrent regressions do not prove thread-race absence or real-time behavior.

## Changed files

Paths below are relative to `apps/loPredCtrlAcc`, except the engineering plan.

- New runtime: `phase_shadow.hpp/.cpp`.
- Runtime integration: `loPredCtrlAcc.hpp`, `native_pipeline.hpp/.cpp`,
  `native_observer.hpp/.cpp`, `online_shadow.hpp/.cpp`, `multirate_shadow.hpp`,
  `shadow_controller.hpp/.cpp`, `synco_shadow.hpp/.cpp`.
- Build/configuration: `Makefile`, `loPredCtrlAcc.conf`,
  `examples/loPredCtrlAcc-online-shadow.conf.example`, `tests/Makefile.offline`,
  `tests/Makefile.replay`.
- Tests: new `tests/ddspc_phase_test.cpp`; updated
  `tests/ddspc_native_pipeline_test.cpp`, `tests/native_application_test.cpp`.
- Documentation: this file; supersession links in `SYNCO_NATIVE.md`,
  `UPSTREAM_SHADOW.md`, `ONLINE_SHADOW.md`, `NATIVE_DIAGNOSTICS.md`,
  `ACCELSIM_SHADOW_EXPERIMENT.md`, `tests/README.md`; milestone in
  `agents/plans/2026-11/accel-ddspc-upstream-shadow.md`.

The syncopation base/timing headers remain byte-identical to the upstream
checkpoint documented in `SYNCO_NATIVE.md`. Original loPredCtrl numerical source
digests remain unchanged. No source producer or tracked result is modified.
