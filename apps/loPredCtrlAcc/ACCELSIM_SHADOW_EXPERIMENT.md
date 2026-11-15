# Controlled accelSim DDSPC shadow comparison — not yet authorized to execute

This procedure preserves the already validated accelSim disturbance injection.
It does not redevelop or require repeating its DM-to-WFS response test. No DDSPC
correction is applied. Do not start injection, restart applications, change active
configuration or run this experiment while other observers are operating.

## Admission before an authorized window

- Build/review the guarded binary privately; confirm `DDSPC_SHADOW_ONLY`, shadow
  mode and `safety.dm_write_enabled=false`. Default online training remains off.
- Supply the live calibration/reference/anchor evidence in `SYNCO_NATIVE.md`. Observe actual upstream
  semaphore mode, lock and expected 4:1 progression; never set them from an INDI
  `ddspc_mode` request or from FPS alone. Leave AD7689 synchronization to its owner.
- Identify the active WFS modal producer, selected tip/tilt modes, signs/units and
  the complete DM-to-modal calibration. Retain identities/hashes of that evidence.
- Define the **optimized corrective input** separately from the injected disturbance.
  `dm01disp` includes the injected `dm01disp10` contribution. Training/scoring on
  actual future **total** commands can explain away that disturbance through `C`
  and make conditional error appear excellent without ADC predictive information.
  Either use a reviewed corrective-request stream/partition, or explicitly label
  the experiment as total-request plant identification. Do not subtract an
  unobserved channel or fabricate a physically applied corrective input.
- Confirm request-history completeness and a reviewed four-slot sampling phase.
  If inputs are held 2 kHz and four columns are dependent, retain the unexcited
  interlock. Do not enable new excitation or change the DDSPC rank criterion merely
  to obtain readiness. Identifying independent 8-kHz responses may require a later
  authorized data set or a separately reviewed reduced/structured model.
- Native score recording now uses the ordinary recorder thread: configure a finite
  `prediction_capacity`, an exclusive private `prediction_file` and a sufficient
  `prediction_max_rows`. The manual queue consumer is disabled while that recorder
  owns it. Input and diagnostic queue drops are recorded separately.

## Disturbance settings and identity

Use the previously approved accelSim calibration, mode cube, mask, source depth 128
and disturbance scale **S_used**. Its numeric scale and installed calibration are
not provided in this checkout; the operator must copy the already validated value
into the experiment record. Do not choose a new amplitude during this procedure.
`zeroOutput=false` is required for a nonzero ADC comparison. Changing zeroOutput
does not establish or enable injection.

Source `parkertjohnson1/accelSim@9d4c5a99` exposes `dm_enable.toggle`,
`dm_scale.current/target`, and `dm_status.ready/active/fault`. Only the authorized
operator may use that established injection handover. DDSPC never touches it.

The reference numerical population (`accelSimModel.hpp:335-350`) includes:

| Component | Per-axis acceleration RMS before DM scale | Approximate coupled pointing RMS at lever arm 10 m |
| --- | --- | --- |
| 28 Hz | 0.002 m/s² | 1.33 mas × S_used |
| 56 Hz | 0.001 m/s² | 0.167 mas × S_used |
| 54 / 58 Hz | 0.002 m/s² each | Neighbouring components must be resolved separately |

The scale changes DM injection, independently of ADC generation. Preserve that
distinction in calibration/generalization analysis. Use the actual installed
population/configuration if it differs; the synthetic qualification report uses
28.5/56 Hz solely as a known software test signal. No production frequency or
amplitude change is part of this work.

## One common recording, three analyses

Prefer simultaneous standard and assisted shadow fits on the **same realization**:

1. Disturbance-only baseline is the observed WFS record with normal AO operating
   unchanged and no DDSPC output. Record input health and all existing DM activity.
2. Standard DDSPC runs independently without ADC regressors.
3. Assisted DDSPC uses the same measured WFS and corrective-input records, plus all
   individual ADC samples. `ddspc_mode` changes diagnostic selection only; neither
   fit is reset or granted output permission.

Record at least 10 s warm-up plus a common 60-s evaluation interval. Also require
both empirical ranks to be full, minimum update counts met and readiness stable;
time alone is insufficient. Start evaluation at the later readiness time and
retain all unavailable/late origins as missing, not zero-error predictions. Use
F=1 initially, then separately predeclared F=2,4,8 WFS frames with identical paired
intervals. Nominal horizons are 0.5, 1, 2, 4 ms; each future command block has four
individual opportunities. Do not mistake those targets for measured 8-kHz WFS.

For the matched information comparison use `standard_slots=4`, identical WFS
history/penalty/forgetting/prior/update stride and command domain. The single-rate
`standard_slots=1` baseline is retained and should be reported separately when
its held-input convention is satisfied. Operational `loPredCtrl` is never changed.

Capture repeat realizations on separately authorized occasions if possible.
Reusing a seed after changing rates/skipping samples does not prove the same
realized disturbance; simultaneous paired fits and recorded actual samples do.

## Required recording

- Every accepted `acc000` UINT16 X/Y sample, calibrated SI values, original
  timestamps/counters, generation/segment and reader losses/restarts.
- Every selected `aol1_modevalWFS` measurement/counter and local availability.
  `camwfs` / `camwfs_sync` markers remain useful diagnostics, not exposure gates.
- Upstream producer mode/lock/reference configuration, heartbeat, frame IDs,
  dropped/gap/discontinuity counters and consumer counter anchors.
- Full `dm01disp` requested snapshots and counters; the injection channel
  `dm01disp10` and any corrective components needed by the chosen input partition.
  Preserve original identities, publication/receipt and actual sampler cutoffs.
- Available synco schedule/epoch/targets/cutoffs, source-ready/ack and software/SDK
  completion records, with their actual semantics. Missing physical application
  evidence stays missing.
- Model settings, source/calibration IDs, update/rank/readiness traces, origin-only
  and conditional predictions, target counters/horizons, computation/deadline
  status, diagnostic candidates and all queue-loss counters. A compact INDI
  snapshot is not a substitute for these numerical records.

## Metrics and pass/fail

Software/input pass: expected 8 kHz/2 kHz rates within the predeclared cadence tolerance,
no unexplained input loss, invalid samples, restarts or queue overflows; current
upstream lock/status; exact one-to-four source grouping; no future-available
features; independently observed complete command histories; finite fits and
genuine empirical readiness. Missing evidence or unexcited matrices is a valid
**inhibited** result, not permission to override a gate.

Compare tip and tilt separately on the exact common origin/target/horizon set:

- Primary origin-only RMSE and error variance; report the held-last-input
  assumption and its error under future normal-AO updates.
- Auxiliary conditional RMSE with actual future inputs, clearly labelled.
- Measured WFS PSD and origin-only error PSD. Use gap-aware Hann Welch windows,
  preferably 8192 WFS frames (0.244 Hz resolution), identical windows for both fits.
  Integrate 27–29 Hz and 55–57 Hz; report 54/58 Hz neighbours separately and a broader
  54–58 Hz band so a local degradation is not hidden in total variance.
- Command-history RMS and shadow-candidate RMS as different quantities; no
  residual reduction is attributed to unapplied candidates.
- Updates, readiness/rank, finite-value/numerical faults, forecast coverage,
  late-compute fraction, model-job mean/p95/p99/max, queue bounds and losses.

Prediction-quality acceptance: reproducible improvement on paired origin-only
errors with no unexplained 56 Hz worsening is the criterion to advance performance
claims. A software-pipeline pass does not require an ADC benefit; no benefit may
reflect insufficient coupling, an unsuitable input partition, excitation limits
or model/horizon error. Unrelated/time-shifted ADC negative controls and independent
recorded realizations should reject apparent benefits from chance or future-input
conditioning. Quantify uncertainty across realizations; do not tune on the final
evaluation interval.

Synco/real-time pass is separate and remains unqualified. Queued late diagnostics
can validate the algorithms but cannot qualify 125-us writes. No physical DDSPC
control transition follows automatically from any of these results.
