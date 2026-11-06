# DM integration preparation and shadow validation

Engineering snapshot: 2026-11-06, branch `parkertjohnson1/accelSim`, starting
revision `5e227a47`. This report concerns offline numerical proposals. No instrument
connection, DM attachment, application restart, configuration edit or deployment
was performed. The live observer remains observation-only, with
`input.allowSingleFrameGap=false`; the producer's depth-one configuration is unchanged.

## Evidence and DM command architecture

“Verified source” means the code in this checkout. “Owner evidence” means earlier
user/instrument-owner guidance recorded in the engineering plans. Neither proves
the present configuration, ownership or dynamic response of a running instrument.

| Path and API | Evidence and consequence |
| --- | --- |
| `apps/accelFFCtrl/accelFFCtrlModel.hpp`, `harmonicEstimator::predict` | Verified: `m_pointing` is predicted physical disturbance in radians after the configured sensor rotation, without opposing correction sign. Acceleration phase is at `m_sampleTimeNs`; `m_targetTimeNs` is an explicit target. |
| `apps/accelFFCtrl/accelFFCtrlInput.hpp`, `inputProcessor::drain/predict` | Verified: raw UINT16 decoding, source counters/generations, clock translation, freshness and reset state. Prediction now+horizon includes measurement age once. No DM writer. |
| `apps/accelSim/accelSim.hpp`, `prepareDM` | Verified: configured FITS pair/mask are checked for stable file identity and shape before preparing the simulator's existing DM injection. This is not called by the shadow tools. |
| `apps/accelSim/accelSimDisturbance.hpp`, `tipTiltProjection` | Verified: weighted, column-scaled QR fits `B*K = 0.5e6*[x,y]`. Rows of K correspond to supplied modes; columns to registered pupil radians. Reflection enters once. Full supplied mode images, including pixels outside the fit mask, determine output. |
| `apps/dmMode/dmMode.hpp`, `sendCommand` | Verified: constructs `sum(amp[i]*mode[i])` afresh, copies the complete image, increments its own `cnt0`, clears write and posts semaphores. It does not accumulate that image into the channel. This method does not set an acquisition/target timestamp. |
| `apps/hoPredCtrl/hoPredCtrl.hpp`, mapping load / `send_dm_command` | Verified: configured mapping and shaped full-image contribution, followed by channel overwrite, counter increment and notification. This does not establish the running AO loop's algorithm or settings. |
| `apps/accelSim/accelSimDMStream.hpp`, `dmStream::operator()` | Verified: existing-image 2-D 50x50 float validation, busy rejection, full payload replacement, `md->atime` from the corresponding ADC, then active `ImageStreamIO_UpdateIm`. No consumer acknowledgement or scheduled target-time application is implemented here. The installed library's timing/ABI remains a Linux qualification item. |
| `scripts/cacao/RTC/cacao-startup` / earlier DM injection plan | Source starts `DMch2disp-01`. Owner evidence says changed summed additive `cnt0` values trigger a combination of current channel contents, AveMode=0, DClevel=0, trigger mode 4 with 10-us delay. Deployed combiner source/settings were not independently checked in this phase. That 10 us is historical configuration evidence, not end-to-end latency. |
| `libMagAOX/app/dev/dm.hpp`, `processImage` / `makeDelta` | Verified: consumes the combined stream, calls derived `commandDM`, then derives delta/saturation diagnostics. `makeDelta` subtracts summed non-delta channels from output shape and compares against summed delta channels. The term `deltaChannels` does **not** turn channel writes into time increments. Its classification still affects downstream telemetry/offloading interpretation. |
| `apps/bmcCtrl/bmcCtrl.hpp`, `commandDM` | Verified: actuator address mapping, volume/actuator-gain conversion, clamping and square-root conversion to the BMC SDK's voltage representation. These conversions belong to the driver; do not put them into the feed-forward angular/modal mapping. Physical surface-micron input units are also supported by prior owner evidence. |
| `apps/alpaoCtrl/alpaoCtrl.hpp`, `commandDM` | Verified: separate volume/stroke conversion, mean removal and clipping for the woofer. Tweeter coefficients/calibration cannot be copied to this interface. |
| `apps/t2wOffloader/t2wOffloader.hpp` | Verified: averaged tweeter-to-woofer offloading, with integrated gain/leak state and separately normalized modes. Its gain/leak, actuator mask and `t2wOffMat` are not a calibration from physical pointing to tweeter TT. |
| `apps/w2tcsOffloader`, `apps/ttmModulator` | Other offloading/modulation paths have different calibration and purpose. No source establishes them as an interchangeable 28/56-Hz pointing-correction channel. |
| `apps/cacaoInterface/cacaoInterface.hpp` | Verified: monitoring/control of external CACAO gain/state parameters; not the deployed feedback kernel or its transfer function. `loPredCtrlAcc` is absent. `loPredCtrl`/`hoPredCtrl` are not evidence of which algorithm currently owns the AO contribution. |

The **tweeter/DM01 is the supported candidate** because the approved simulator TT
basis and existing command path refer to it. This is not a measured bandwidth or
headroom comparison with the woofer. Final actuator selection requires owner review
of the measured response and AO allocation. No correction channel is assigned.
In particular, `dm01disp10` was assigned to the simulator disturbance experiment;
that is not permission to overwrite it with correction. Sharing it would erase or
race the disturbance and could create a false suppression result.

The intended eventual sum is a complete per-writer contribution:

```text
h_total_um = h_flat + h_AO + h_disturbance + h_FF + other_owned_channels
h_FF_um(t) = B * u_FF(t)             # absolute current contribution
```

An integrator may integrate **its own control state**, but the image it writes is
still a replacement. Repeated `u_FF += proposed` is incorrect. Stopping one writer
would require a confirmed zero of its own contribution, not `zeroAll` or an AO/flat
reset. Driver clipping of the total does not establish stale-command cleanup or
exclusive writer ownership.

## Physical calibration specification

The unknown physical chain must be separated from the available DM-pupil algebra:

```text
a_sensor = (ADC-zero) * voltsPerBit / sensitivity
p_pred   = -R_est * a_harmonic / (omega^2 * L_eff)    # current estimator, rad
wfs_dist = Q(omega) * p_pred                         # measured WFS units
wfs_DM   = J(omega) * u                              # measured WFS units/coefficient
u_FF     = -g * J(omega)^+ * Q(omega) * p_pred
```

C acts on the estimator's already rotated reported frame; do not apply `R_est`
a second time. Alternatively a validated real static coordinate map C and geometric/modal K give
`u_FF=-g*K*C*p_pred`, with calibrated phase/latency handled consistently by the
prediction target and plant model. Do not apply both an empirical composite map
and its already-absorbed lever-arm/optical scaling again. A single real C/K and
common horizon may not describe two frequencies if the mechanical/optical transfer
is frequency dependent. The prototype does not identify complex transfer functions.

| Quantity | Available evidence | Required before physical correction |
| --- | --- | --- |
| ADC offsets, volts/count and sensitivity | Source/config defaults match simulator: 2048, 0.0083799, 1.02; no factor of 9.81 is inserted. | Actual channel identities, sensor units, polarity and frequency-dependent bench calibration with uncertainties. The effective simulation sensitivity is not proof of a hardware V/g conversion. |
| Effective lever arm | Simulator/default value 10 m. | Measured accelerometer-to-angular transfer at the bands of interest, or independently established mechanical geometry. Local acceleration need not be rigid telescope pointing. |
| Pointing-to-WFS response Q | No representative local live measurement supplied. | Simultaneous timestamped acceleration and calibrated WFS/POL data. Account for AO sensitivity and exposure; a closed-loop residual alone biases an open-loop transfer estimate. |
| Coordinate map C | Simulator axes are increasing DM column/row; estimator rotation is configurable. | Measured axes, cross coupling, parity/reflection and sign from sensor/telescope/WFS to this pupil. A notebook rotation or a FITS mode angle does not provide this registration. |
| B, fitting support and nominal pupil coordinates | Locally archived real TT pair and dmmask; owner-defined DM-pupil coordinates below. | Current approved files/hashes and coherent calibration revision. Confirm mode ordering, full-image layout and actual illuminated geometry. |
| WFS-to-modal scaling/sign J | Geometric K exists conditionally in pupil coordinates. Physical WFS response/sign remains unmeasured. | Authorized signed TT-mode response measurements under a documented AO/WFS state, using the actual interaction/reconstruction matrices and units. No physical poke is authorized here. |
| End-to-end delay/phase | Stream/source conventions and historical combiner settings; no measured correction-path delay distribution. | Acquisition, receipt, query, publication, combination, driver and WFS exposure/response timing, with clock uncertainty. Do not substitute the earlier synthetic 1-ms delay or 1.75-ms compensation. |
| DM bandwidth and cross-axis response | Earlier 50/300-Hz plants are synthetic. | Measured complex transfer versus frequency, operating point and amplitude, including resonances, driver effects and uncertainty. |
| AO interaction and saturation headroom | Code shows summing, driver clipping and offloader state; current settings absent. | Current AO gains/leaks/filters, all additive contributions/flat, saturation diagnostics, `deltaChannels` classification, offloader behavior and exclusive ownership. |

The local archived inputs are **workspace fixtures**, not a freshly certified
operational calibration:

- `scratch/accelSim_dm_calib/CMmodesDM.00.fits`: SHA-256
  `a5bb41b535fe46cb225c17f1086dbf9ac190cf3bd451379e9551ffb77383daf9`.
- `scratch/accelSim_dm_calib/dmmask.fits`: SHA-256
  `0970dab118bcdb73236ae045da93c550288a4d632b70b2748903272f2aa166f8`.
- Owner-defined **DM-pupil** geometry: `x=(column-24.5)*6.5/48`,
  `y=(row-24.5)*6.5/48` metres. This does not identify telescope/WFS axes.

Read-only reanalysis gives 1715 fit pixels, modal surface RMS
0.9999999269 / 0.9999999916 micron, and relative fit residuals
8.46879e-8 / 8.63474e-8. Independent NumPy SVD agrees with the already archived
C++ `tipTiltProjection` output. No piston removal or renormalization is applied:

```text
K [coefficient / DM-pupil radian] =
[[-798911.75116813, -315942.59493194],
 [ 318826.69049347, -792420.64908820]]
```

For an explicitly assumed +1 mas pupil-X disturbance, the opposing numerical
coefficients are approximately `[+0.00387323347, -0.00154571541]` at gain one.
That verifies numerical ordering/sign in the **declared pupil frame**, not the
optically measured accelerometer/WFS correction sign. Neither `dmmap` nor the
2039-pixel offloader mask replaces `dmmask`; their meanings/normalizations differ.
The archived-calibration fixture is opt-in and labelled `archivedPupilOnly`.

## Shadow interface and record semantics

`accelFFCtrlShadow.hpp::shadowCommand` wraps the existing offline `feedForward`
using the unchanged `prediction` type. It is a single-owner numerical object,
not a concurrent logger. It requires separate explicit C and K,
a nonzero manifest calibration ID and a synthetic/archived-pupil-only evidence
label. Gain and finite modal bounds default to zero. A zero-bound validator reuses
the same numerical checks to audit prediction validity even when gain is zero.
No file, clock, stream, channel name or publisher is accepted by this component.

Each `shadowRecord` contains lifetime log ordinal; reset epoch; input revision;
calibration identity/evidence; source generation/counter; integer query,
acquisition, prediction-target and requested-target nanoseconds; qualified summed
and mapped radians; proposal state, common limiting scalar and both coefficients;
a numerical-validity flag and shadow lifecycle fault. `active` means a numerical
proposal only. Neither that flag nor a calibration evidence label enables hardware.

Targets must remain exactly `query+horizon`; age is not added again. All configured
bands must be coherent and ready. Invalid/stale/nonfinite input gives zero. Query
rollback, unacknowledged generation change, counter rollback or changed/backward
acquisition identity latch zero until an explicit reset. Reset advances epoch and
requires an acquisition newer than the reset before qualification. The source
counter may normally advance four frames per 2-kHz query on 8-kHz input: such
query-to-query advances are not dropped-frame measurements. Loss accounting stays
in the production input reader. Every record is an absolute replacement; nothing
is accumulated or scheduled for physical application.

A standalone exporter can write these records and provenance manifests to ordinary
files. This is blocking **offline** diagnostic I/O, not a proposed real-time
logger implementation. Future live capture needs bounded preallocated records,
an asynchronous writer, independently counted logging loss and measured overhead.
The live application does not include the new header or a new logging callback.

There is no general ImageStreamIO field here for “apply this command at target
T.” The simulator assigns acquisition time before active update (its retention by
the installed primitive remains to be verified), `dmMode` updates counters without
assigning acquisition/target time, and drivers consume current payloads. Counter
increments are writer-local publication identities, not hardware acknowledgements.
A combiner may coalesce or delay updates and emits its own identity. Preserve
separate source, proposal, channel, combined, driver and optical-observation IDs
in any future timing study; do not infer a one-to-one delivery relationship.

## Offline AO interaction experiment

A new test-only leaky integrator uses an earlier 2-kHz WFS residual:

```text
u_AO[k] = (1-leak)*u_AO[k-1] - gain_AO*K*e_WFS[k-1]
u_sum   = u_AO + u_FF                       # complete contributions
r       = d + P*u_sum
```

It reuses the previous synthetic delayed first-order `dmResponse` and recorded
causal dual-band FF commands. The nominal synthetic plant is again 1 ms/300 Hz;
AO gains 0.1, 0.3 and 0.8 and modal headroom proxies 0.03/0.003 are **test choices**,
not instrument settings or hardware limits. First WFS feedback is available only
for the next query. Saturation is a common-scalar **modal proxy**, not the BMC
flat-biased per-actuator clipping model. The optional back-calculation weight 0.5
is illustrative and does not describe deployed anti-windup.

For a linear unsaturated common-path plant, `r=S*(d+P*u_FF)`,
`S=(I+P*C_AO)^-1`. An AO integrator does not automatically double-correct an exact
FF contribution: it responds to the resulting residual. Double counting can
instead arise from duplicate FF writers, inconsistent POL reconstruction, or
unmodelled modal/temporal coupling. A POL estimator must account for all actually
applied channels and their dynamics, including FF, rather than AO command alone.

The 60-s record gives these full 8-kHz variance reductions after 4 s:

| Synthetic case | X / Y reduction | Consequence |
| --- | ---: | --- |
| AO only, gain 0.1 | 25.47% / 23.14% | Reference trajectory. |
| Shadow calculation/logging, no FF applied | Identical | Every AO/residual sample and trajectory hash match AO-only exactly. |
| FF only | 94.88% / 94.24% | Matches the previous recorded cancellation study. |
| AO 0.1 plus FF | 95.75% / 95.09% | Consistent feedback can improve residuals. |
| AO 0.3 only / plus FF | 86.79% / 85.66%; 99.07% / 98.89% | Depends on the prescribed loop. Not a recommended live gain. |
| Duplicate FF contribution | 20.17% / 17.22% | Independent duplicate correction can defeat suppression. |
| Wrong sign | -190.69% / -198.65% | Amplification. |
| Incorrect incremental FF semantics | -1873.94% / -1993.77% | 105121 saturated queries; proposals must replace rather than accumulate. |
| Tight combined headroom | 17.15% / 4.68% | 118092 saturated queries; individual FF bounds do not protect total headroom. |
| Illustrative back-calculation | 62.97% / 60.66% | Internal candidate peak drops from 0.06025 to 0.00719 coefficient; not a hardware policy. |
| Actual plant delay 3 ms, AO 0.3 | -1611.13% / -2332.86% | Large response/latency errors can excite clipped oscillation. |
| AO gain 0.8 | -1606.50% / -1997.33% | Output remains clipped while residuals become poor; bounded output is not proof of stability. |

Additional cases disable FF or reset AO halfway through the record to expose
state/settling transients. No real AO loop, resonant DM, WFS exposure or
DM-to-accelerometer mechanical feedback is represented. These counterexamples
motivate measurements; they do not prove a stability margin or a safe live gain.
The diagnostic PSD plot uses uniformly sampled 2-kHz WFS outputs; whole-record
RMS is computed at the 8-kHz simulation rate. It is not a measured WFS spectrum.

## Future authorized observation-only protocol

This is a proposed procedure, not authority to execute it now. Agents must not
run on or connect to instrument hosts. Any future instrument steps are performed
by an explicitly authorized human operator; source analysis remains on a workstation.

1. **Build qualification on a non-instrument Linux development environment.**
   Use the intended GCC/C++20, Eigen, mxlib, ImageStreamIO and MagAO-X dependencies.
   Run the existing `tests/validate_linux.sh` from a matching checkout; it builds
   both applications and configuration/INDI tests without launching them, runs
   standalone tests and native tests in private synthetic namespaces, and records
   versions/header/library paths. Run the new shadow/archived fixtures and AO
   replay too. The runner refuses known instrument hostnames. Native/API/ABI
   failures block further qualification. No install/launch command is supplied here.
2. **Verify native ADC transport.** Preserve production depth one and disabled
   gap policy. Exercise private 4/8-kHz two-channel UINT16 tests, writer busy,
   coherent-copy retries, missing/overwritten frames, bad times, restart and
   generation/counter rollback. Existing deeper-buffer fixtures are private
   historical tests only. Record processing versus known losses and readiness;
   semaphore posts do not prove lossless delivery.
3. **Approve a coherent calibration bundle.** Obtain current B/TT mode indices,
   dmmask, full pupil geometry, WFS interaction/reconstruction/reference/mask
   files, units, signed axis transforms, ADC calibration, band-dependent angular
   transfer, and measured DM/WFS timing/response. Hash the entire bundle and record
   operating point, wavelength/modulation, AO state and uncertainty. Old archived
   B/K plus identity C cannot pass the physical-chain gate.
4. **Qualify capture before an instrument observation.** Develop/review a separate
   bounded capture path on a development host. It must not attach any DM command
   output, share a live consumer semaphore, or perform blocking file I/O in the
   acquisition worker. Measure capture overhead, overflow and missing records.
   Current tools consume existing local files; no live per-query capture hook was
   added in this phase. Five-Hz INDI is insufficient for phase/PSD qualification.
5. **Human collection under a separate authorization.** Observe existing processes;
   do not restart `accelSim` with an unreviewed DM-enabled configuration. Collect
   each accelerometer counter/generation, ADC pair, acquisition/write stamps,
   stable-copy receipt and actual realtime/monotonic anchors with uncertainty;
   all rejected/busy/racing candidates; input dropped/reset/revision/readiness
   events; and every prediction query/target/sample/value plus logging loss.
   Keep default shadow gain zero; any counterfactual nonzero proposal gain must
   be an explicit offline/shadow-only test choice, with no publication path.
6. **Collect WFS/AO context.** Candidate source names in this checkout include
   `aol1_modevalWFS`, `aol1_modevalDMf_mon`, `aol0_modevalDMf_mon` from
   `wooferTweeterRecon`, and `camwfs_sync` notification metadata. These names are
   not proof of current availability or radians. Record actual mode bases/units,
   mode indices, exposure timing and current controller gains/leaks/filters,
   AO applied-command/flat/other-channel and saturation diagnostics, and all
   provenance/clock/generation information. `wooferTweeterRecon` uses DM `atime`
   and WFS `writetime-1/fps+offset`; its configured offsets are not measured
   timing constants to copy. `camwfs_sync` is a notification, not WFS TT data.
7. **Timing and spectral audit offline.** Reconcile ADC versus logger loss; retain
   exact integer timestamps and independent counters. Verify target=query+horizon,
   sample freshness and source identity, generation/reset barriers and zero on
   invalidity. Match predictions to the actual WFS exposure/reference, accounting
   for AO sensitivity or a validated POL reconstruction. Report coherence,
   cross-spectrum/phase, residual PSD and uncertainty by calibration/loop revision.
   Do not concatenate gaps into a uniform PSD or compare predicted open-loop
   disturbance directly with an unmodelled closed-loop WFS residual. Existing
   `analyze_timestamps.py` is a necessary timing audit, not this optical calibration.
8. **Review pass/fail evidence before considering actuation.** Mandatory software
   criteria: all builds/tests/native checks pass; zero nonzero disabled/invalid/
   stale proposals; zero target-time violations; complete provenance and loss/reset
   accounting; repeatable absolute coefficients within explicit numerical bounds;
   zero alteration of existing AO outputs by shadow logging; and default policy/
   depth unchanged. Physical acceptance needs owner-approved coherence, gain/phase
   uncertainty, timing-error and total-headroom budgets at each band. Unset budgets,
   unknown registration/sign/response, unaccounted logger loss or missing native
   evidence mean **not qualified for real commands**. Timing uncertainty must meet
   `omega*delta_t <= allowed_phase_error`; no physical timing threshold is invented.

After all gates, a separate approved design must assign an exclusively owned
correction channel, confirm current combination/trigger/delta semantics and
applied-zero behavior, serialize enable/fault/disable/shutdown, address process
failure cleanup, and preserve AO/flat/offloader ownership. No live writer is
implemented or authorized by this report.

## Reproduction and current results

From the workstation checkout root, using an existing scientific Python environment
(with Astropy only for the optional archived-FITS analysis):

```bash
make -C apps/accelFFCtrl/tests BUILD_DIR=/tmp/accelFFCtrl-shadow-build test benchmark shadow-tools
ACCELFFCTRL_SHADOW_DIR=/tmp/accelFFCtrl-shadow-final-60s/logs \
  /tmp/accelFFCtrl-shadow-build/accelFFCtrlCancellation_export /tmp/accelFFCtrl-shadow-final-60s/cancellation 60
python3 apps/accelFFCtrl/tests/validate_cancellation.py /tmp/accelFFCtrl-shadow-final-60s/cancellation
python3 apps/accelFFCtrl/tests/validate_shadow.py \
  --shadow-directory /tmp/accelFFCtrl-shadow-final-60s/logs --output /tmp/accelFFCtrl-shadow-final-60s/report
/tmp/accelFFCtrl-shadow-build/accelFFCtrlAO_export \
  /tmp/accelFFCtrl-shadow-final-60s/cancellation/ar_2_gain10.bin /tmp/accelFFCtrl-shadow-final-60s/ao
python3 apps/accelFFCtrl/tests/validate_shadow.py --ao-directory /tmp/accelFFCtrl-shadow-final-60s/ao \
  --output /tmp/accelFFCtrl-shadow-final-60s/ao/report
python3 apps/accelFFCtrl/tests/validate_shadow.py --archived-pupil-directory scratch/accelSim_dm_calib \
  --projector-columns scratch/direct_tt_analysis/cpp_mapping_columns.csv --output /tmp/accelFFCtrl-shadow-archive
ACCELFFCTRL_SHADOW_CALIBRATION=/tmp/accelFFCtrl-shadow-archive/calibration_fixture.txt \
  /tmp/accelFFCtrl-shadow-build/accelFFCtrl_test '[accelFFCtrl-shadow-calibration]'
make -C apps/accelFFCtrl/tests PYTHON=python3 validation-test
python3 -m unittest discover -s apps/accelSim/tests -p test_validate_psd.py -v
```

The previous 76-run 60-s cancellation study is rerun with optional shadow records
alongside the original proposer. Each shadow modal/state result must equal that
original proposal; original binary/summary results are checked against the prior
archive. Twenty-two logs contain 2,640,000 query records, including zero-gain,
invalid/stale input, reset and reattachment. All evidence labels are synthetic.
The archived-pupil fixture separately passes signed modal examples and retains
unchanged FITS hashes. AO-only versus shadow-only trajectories match exactly.
Detailed final test counts and sanitizer results are recorded in the engineering plan.
