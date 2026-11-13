# Offline ADC–DDSPC infrastructure

2026-11-12, `parkertjohnson1/accel_ddspc`, audited base `116c6b5a`.
Host verified as personal workstation `ASTR-LT-428.local`. The user reports that
the app builds/starts in ADC-only shadow mode on exao2, but accelSim must remain
disabled. That report is not an agent hardware validation. No producer, native
application, server, service or instrument stream is used by this task.

## Audit and smallest proposed plan

The audit/plan was presented before editing. Existing components are already
separated at useful boundaries:

| Component | Existing isolation and coverage | Remaining coupling/gap |
| --- | --- | --- |
| `AdcCapture` in `adc_input.*` | Standard-library-only decoding, metadata brackets, timestamp mapping, retention, gap/restart handling | Its existing test file also includes worker/API fixtures, obscuring the standalone boundary |
| `AdcReader` in `adc_reader.*` | Own worker/mapping and ADC mutex; offline API-double reconnect/shutdown tests | Native ImageStreamIO ABI/producer behavior remain separate runtime qualification |
| `CausalInputBuilder` | Stream-free WFS exposure/applied-interval pairing and extensive synthetic provenance tests | No native WFS/applied-input/ADC feature adapter exists; testing must not imply one |
| `ShadowController` and causal `PredictiveController`/RLS/QR | No publisher or device interface; two independent models, validity/readiness and frozen-origin diagnostics | Learner headers include mxlib image headers even where only Eigen math is used; no production refactor is needed for this task |
| Native `loPredCtrlAcc` header | Existing stream-free lifecycle doubles, real INDI value objects, guarded shadow builds | A header/framework test is not a native-base/driver integration test |
| Older simulation/benchmark tools | Actual numerical implementations and many synthetic plant tests | Some require historical Git snapshots or another accelSim worktree; they do not connect current raw ADC capture directly to the learner |

Plan: split ADC core/transport tests without changing production; add one owned
8-kHz SI raw source and test-only spectral estimators; connect actual decoding to
the existing two-model controller in an explicit 2-kHz synthetic plant; strengthen
output/INDI guard tests; expose a dedicated offline target and reproducible CSV
CLI; document limits and validation. No production configurations or numerical
algorithms are to change. No new live alignment adapter is part of this work.

The original baseline passed 83 offline C++ cases and 9 guarded app cases before
the test separation and additions. These are local synthetic/API-double passes.

## Synthetic timing and plant contract

Each WFS decision has four 125000-ns ADC frames available beforehand. Raw encoding
is generated independently from SI signals around configured zero 2048. Actual
`AdcCapture::drain()` supplies decoded acceleration and its mapped/original times.
Assisted features average only a complete, available four-frame segment. Loss
breaks assisted history; standard WFS/applied-input records remain usable.

The synthetic physical acceleration combines28.5-Hz and 56-Hz tones, independent
Gaussian broadband force, and separate sensor noise. Timestamp jitter affects
actual acquisition times; lost reads emulate a depth-one producer overwriting
unread payloads, preserving observable counter loss semantics. No accelSim header,
binary or worktree is required.

The plant is deliberately synthetic, in arbitrary WFS/input units:
`d_(i+1)=pole*d_i+coupling*M*a_i`, `y_i=d_i+dmGain*u_(i-1)`, with
`M=[[1,.15],[-.1,1]]`. The physical mean `a_i` drives only the plant; controller
features use decoded noisy/quantized ADC measurements. Independent WFS noise is
added only to measured residuals. All three trajectories share exogenous seeds.
Both feedback runs train on identical common calibration inputs; evaluation uses
independent simulated plants and a small declared persistent excitation.

At decision i, `predict_cached()` uses the previous validated fit and the current
causal `[y_p;u_p;a_p]` state. The bounded input is consumed by the synthetic plant
before constructing the actual `ShadowRecord` for learning. Reported forecast
RMSE is the production frozen-origin **conditional** score using later actual
inputs for scoring; it is not a free-running disturbance forecast or control
feature. Readiness/rank and numerical faults come from the real implementation.

## Shadow safety boundary

Source review confirms that shadow startup/lifecycle skips the output
frameGrabber and Legacy sensor monitor, direct acquisition/output entrypoints
reject requests, and Legacy learning/prediction/integration/exploration/load/save
callbacks reject enabling actions. ADC transport opens existing input images only.
The offline core links no ImageStreamIO or INDI implementation. Worker/application
safety tests use explicit API/lifecycle doubles and capture-only INDI transport.

`DDSPC_SHADOW_ONLY` is **not a security sandbox or INDI namespace reservation**.
A native process still advertises its own diagnostic properties and uses its
configured device identity/FIFOs. Reusing a production name/broker is not prevented
by that macro. Native unique-instance/private-broker setup and installed ABI,
framework destructor, syscall/scheduler and producer-concurrency behavior cannot
be proven by these doubles. No native test or operator procedure is run here.

## Execution record

The dedicated checks pass: 11 ADC/synthetic cases, 16 actual-model/input cases,
10 guarded app cases, 2 CLI tests and 2 example-config tests. The broader regression
passes 94 cases plus 8 build-tool tests and the original 15-file Legacy verifier.
ASan/UBSan passes 11 ADC and 16 model/input cases; ThreadSanitizer passes 13 combined
ADC/synthetic/worker API-double cases. No failing test or sanitizer finding remains.
Native runtime execution was neither required nor performed.

An optional `DDSPC_MINIMAL_APP_TESTS=yes` test-build flag drops historical oracle
objects when the dedicated safety target supplies its explicit app source list.
The default full regression build is unchanged. This removes needless Git-history
coupling from the dedicated target. Core/simulation executable symbol inspection
found no unresolved native stream/DM/socket/INDI-command dependencies.

Fresh private artifacts were generated for default, dropout/jitter/noise and
zero-coupling scenarios. Default variance is .24845735 open, .00263336 standard and
.00247875 assisted (5.87% improvement over standard), but the assisted 54–58-Hz PSD
band is 11.8% worse. With 200 lost ADC reads (25 ms), assistance holds input and has
variance .01514936 versus standard .00263776, despite zero gate violations/faults
and eventual readiness recovery. Zero physical coupling gives negligible
assistance; evaluation excitation itself disturbs the quiet plant. The guide
records these negative outcomes and does not infer universal suppression.

The CLI now exports `metrics.csv`, `residuals.csv`, `psd.csv` and `learning.csv` in
an explicitly requested **new** directory, refusing existing artifacts. Learning
evidence includes rank, updates, readiness, conditional RMSE and coefficient norm
every 100 WFS steps, first readiness and the final step. No model/forecast truth
is fed back into the independent source. Outputs stay in ignored build directories
or a requested new directory; tracked historical results are not regenerated.
No commit, merge, push or installation is part of this task.

## Affected files

Paths below are relative to `apps/loPredCtrlAcc/` except this plan:

- `tests/Makefile.adc-ddspc` (new) and `tests/Makefile.offline`.
- `tests/ddspc_adc_test.cpp`, `tests/ddspc_adc_transport_test.cpp` (split out),
  `tests/ddspc_adc_synthetic_test.cpp` (new).
- `tests/offline_adc.hpp/.cpp`, `tests/offline_ddspc.hpp/.cpp`,
  `tests/offline_ddspc_main.cpp` (new).
- `tests/ddspc_adc_ddspc_test.cpp`, `tests/offline_cli_test.py` (new).
- `tests/ddspc_app_test.cpp` (additional guarded safety case).
- `OFFLINE_ADC_DDSPC.md` (new), `tests/README.md`, and this engineering record.

The final default CLI re-export, including learning evidence, exactly matches the
earlier numerical metrics. All final production-source/configuration and tracked
historical-result diffs are empty. Final scope remains test infrastructure only.

## Offline fallback implementation and review record (2026-11-12)

The subsequent user request authorized robust fallback **in test harnesses only**.
Host/branch were rechecked (`ASTR-LT-428.local`, `parkertjohnson1/accel_ddspc`). All
previous uncommitted infrastructure was retained. No production source/config or
existing historical result was edited, and no native runtime or service ran.

Smallest design: add test-only `offline_control.hpp/.cpp` for timestamp windows
and mode/command policy; keep `AdcCapture`, `ShadowController`, causal histories,
QR/RLS, reference solve and native shadow behavior unchanged. Extend the owned
synthetic source with independent ADC oscillator error, variable delivery latency
and repeated losses. Construct one immutable physical/noise/probe/delivery
schedule for all three private feedback plants. The simulated plant keeps its
explicit one-frame ADC lead and static-gain/one-frame DM delay; no mechanical
validity or native readiness is inferred.

At each decision, requested assisted mode requires fresh complete ADC windows/
history, common WFS/input validity and successful current cached prediction.
Otherwise the corresponding ready standard model is used; neither-ready produces
a bounded ramp toward zero without exploration. Recovery requires10 consecutive
fully qualified decisions by default. A common absolute bound3 and all-frame
slew bound.2 apply to calibration, both models, fallback and safe ramp; actual
consumed commands feed both learners. The production selection label is untouched.

The feature window is timestamp-defined, not four-index grouping: causal ZOH
integration over a500-us half-open window with explicit optional lag, bounded
175-us held-sample age/gaps, contiguous source/counter segments and500-us default
effective-age bound. Acquisition centroid and latest contributing receipt remain
separate. A private FIFO delivery model ensures frozen-deadline causality under
latency variation. Unknown source timestamp mapping and native temporal-cube
retention remain outside this timing model.

`ddspc_offline_control_test.cpp` and expanded ADC/DDSPC tests cover analytical
weights/centroids, future/late exclusion, stale/incomplete/gapped windows,
independent cadence drift, invalid proposals, recovery hysteresis, all-frame
bounds,1/5/25/100-ms losses, repeated loss, common WFS failure, strict feature
freshness despite healthy20-ms capture, and identical physical forcing/no-future
prefixes. New CLI diagnostics include every decision/transition reason, consumed
command, timestamp evidence, command RMS, residual band power, fallback/safe/
longest-unavailable time and maximum switching discontinuity. Conditional RMSE
now accumulates across valid model segments instead of retaining only the latest.

Validation passed:11 ADC/synthetic cases (447999 assertions),25 numerical/input/
window/policy cases (38610 assertions),10 guarded application cases (2152
assertions),2 CLI and2 configuration Python tests. ASan/UBSan passed11+25 cases
and CLI tests; TSan passed13 ADC/worker cases (448030 assertions), with no findings.
Clang-format23 and whitespace checks passed. Existing94-case/historical/build-tool
results above refer to the prior infrastructure pass, not a new run here.

`offline_fallback_matrix.py` and the `comparison` target ran24 scenarios/72
trajectories with no numerical-fault or invalid-assisted-gate frames. Full summary
CSV is newly archived as `tests/results/offline_fallback_2026-11-12.csv`; existing
historical result files were preserved. Full private artifacts remain under
`build/offline-fallback-review-20261112/`. `OFFLINE_FALLBACK.md` records policy,
commands, timing assumptions, all comparison outcomes and production review items.

Default variance is.00262831 standard versus.00240344 assisted-requested (8.56%
lower); assisted54–58-Hz power remains11.75% higher. With25-ms ADC loss variance
is.00240753 and the longest unavailable episode45.5 ms. All switching steps remain
<=.2 synthetic units. Normal rolling-rank losses cause88.5 ms total fallback over
the2-s evaluation even without injected loss.100±80-us delivery jitter prevents
assisted requalification under the strict contract, correctly yielding standard
behavior. The explicit375-us lag/750-us age profile permits delayed features.
A25-ms WFS-invalid interval exposes conservative safe lockout for the remaining
1.8 s: zero safe input removes excitation and does not automatically requalify a
rank-deficient model. No unsafe re-excitation or rank bypass was added. These
remaining design limits must be reviewed before any production transfer.

Additional affected files for this task: `tests/offline_control.hpp/.cpp`,
`tests/ddspc_offline_control_test.cpp`, `tests/offline_fallback_matrix.py`,
`tests/results/offline_fallback_2026-11-12.csv`, `OFFLINE_FALLBACK.md`; plus the
existing test-only generator/harness/CLI/tests/Makefile, infrastructure guide,
tests README and this plan. No commit or publication is authorized.
