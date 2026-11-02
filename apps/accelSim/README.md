# accelSim

Continuous accelerometer simulation with the `mcp3208Ctrl` raw-count interface.
The default output is **`acc000`**, with frames containing X then Y as two
`uint16_t` values in [0,4095]. Only one producer may own that stream and its INDI
instance at a time. The executable name does not determine the replacement INDI
identity: supply the real instance with `-n` when its configuration is available.

The approved specification and engineering record are in
[`acc-sim-app-creation.md`](../../agents/plans/2026-10/acc-sim-app-creation.md).
Optional same-sample tweeter disturbance injection is implemented separately from
the model and remains disabled by default. The default configuration has no DM
channel and performs no DM access. No instrument startup configuration is supplied.

## Optional runtime DM injection

The intended experiment uses **dm01disp10**, a pre-existing 50x50 float32 additive
image in physical DM surface microns, at **8 kHz**. Jared Males assigned channel 10
for the experiment; tweeterModulator also uses it. The operator must establish
exclusive handover, including its final zero and any bridge/automatic writers,
**before configuring accelSim to attach**. accelSim does not stop other applications
or acquire an ownership lease. Its startup zero would overwrite another owner's
contribution if handover had not occurred.

Configuration keys in `[dm]`:

| Key | Default | Meaning |
| --- | --- | --- |
| `channelName` | empty | Existing owned channel; intended deployment value `dm01disp10`. Never created/replaced. |
| `modeCube` | empty | Approved TT FITS cube; first two planes in native order. |
| `mask` | empty | Matching deployed dmmask fitting support. |
| `enabled` | false | Must remain false; startup enable is rejected. |
| `scale` | 0 | Independent experiment/commissioning amplitude factor in [0,1], applied only to the DM contribution. |
| `maxRelativeResidual` | 1e-6 | Maximum physical projection residual; reviewed runtime acceptance setting. |

Jared Males confirmed that the existing MagAO-X/DM system provides sufficient
command protection for this injection. accelSim applies the requested `dm.scale`
without additional actuator or P2V limiting. Scale adjusts experiment amplitude;
it is not a hardware safety mechanism. The physical convention is +X increasing
column, +Y increasing row, center (24.5,24.5), 48 pitches across 6.5 entrance-pupil metres. No sky/Az/El,
K-mirror, AEG, or offloader calibration/gain is used.

At startup, with an explicitly configured channel and valid calibration, accelSim
loads/validates the calibration, precomputes B*K, opens the existing image, validates
2-D 50x50 float32 geometry, and publishes zero. Any preparation failure leaves
injection unavailable while accelerometer acquisition continues. There is no
automatic reconnect/reload: stream replacement or changed calibration metadata
requires operator review and restart/revalidation.

Only `dm_enable.toggle` arms injection, and only at an active ADC rate of 8000 Hz.
`dm_scale.current/target` controls commissioning amplitude; begin at zero and
raise it deliberately. `dm_status` reports `ready`, `active`, and numeric `fault`:
0 = no fault, 2 = calibration, 3 = stream, 4 = numerical, 5 = rate/synchronization.
Enable/disable reports Busy until processed at a completed sample boundary, then
Ok or Alert. These properties update at the ordinary application cadence.

The existing `frameGrabberPostPublish` hook consumes the **same** completed
`sample::m_pointing` after ADC publication. It converts arcsec to radians using
`pointingDisturbance`, synthesizes from two precomputed angular responses, applies
the requested scale, and copies the entire local command into the channel. Active
publication uses `ImageStreamIO_UpdateIm`, the primitive used by mxlib's active
`milkImage::post`. It does not manually reproduce counters or semaphore posting.
The user-inspected deployed combiner recomputes when the sum of additive `cnt0`
values changes (`AveMode=0`, `DClevel=0`, process trigger mode 4, 10-us delay).
Passive publication would not satisfy this contract.

No additional model/RNG step, acceleration integration, `m_wfsPointing` attenuation,
FITS read, solve, heap allocation, consumer wait, or routine logging/INDI is added
to the sample path. There is no DM worker/queue. The approximately 2-kHz WFS stream
remains the existing ADC phase reference; it does not decimate DM publication.
The two output publications have a measurable skew, not atomic simultaneous
delivery. Coalescing in the combiner/driver can still prevent distinct delivery
of every producer frame and must be measured.

Disable requests stop nonzero output and actively publish zero before acknowledgement.
Faults zero/inhibit and latch until an explicit new control request; they never
stop ADC acquisition. Source loss zeros after the existing synchronization timeout;
ADC holdover continues. A rate change away from 8 kHz inhibits injection. On normal
shutdown acquisition joins before a final zero/close, preventing a late nonzero
write. Only the configured channel is touched. An unset channel touches no DM.

Calibration/backing-file supervision runs outside acquisition and is not an
instantaneous content-hash guarantee. A vanished stream or failed zero is reported
as a fault; zero cannot be guaranteed if the combiner/driver is down. A crash can
leave a held/partial command or write flag. There is no crash watchdog/expiry lease:
the operator/supervisor must provide recovery. Disabled does not transfer ownership
back to tweeterModulator, whose zero callback must not race accelSim.

The local numerical/mock publisher costs a few microseconds per frame, but the
125-us budget includes native publication, scheduling, and model computation.
Full Linux native-stream/load testing remains required before physical commissioning;
the local macOS app build is blocked by existing platform dependencies. No physical
DM stream is used during development tests.

## Independent DM disturbance calculations

[`accelSimDisturbance.hpp`](accelSimDisturbance.hpp) implements the physical
conversion from the cleaned notebook using the confirmed tweeter surface units.
The numerical helper itself has no stream access. Runtime integration is described
above; its engineering record and commissioning requirements are recorded in
[`accelSim-dm-injection.md`](../../agents/plans/2026-10/accelSim-dm-injection.md).

`accelSimDetail::pointingDisturbance` copies only `sample::m_pointing`, converting
arcsec to radians in the simulator's X/Y axes. It does not use `m_wfsPointing`,
sensor noise, or ADC values, and does not advance the model or draw random numbers.
For supplied entrance-pupil coordinates in those same axes, it evaluates:

- `opd(x,y)`: `theta_x*x + theta_y*y`, in metres.
- `surface(x,y)`: half that OPD, converted to microns for a reflective surface.
- `surfaceP2V(D)`: `D/2 * hypot(theta_x,theta_y) * 1e6`, in microns over a circular
  pupil; the default entrance-pupil diameter is 6.5 m.

These are ideal physical quantities, not actuator commands or a synthetic WFS.
The current simulation convention explicitly defines **+X as increasing DM image
column and +Y as increasing DM image row**. These are DM-pupil coordinates, not
telescope Az/El, sky, or WFS coordinates. The notebook's example 30-degree rotation
is not used. A 900-nm OPD P2V across
6.5 m corresponds to 28.5597 mas pointing and 0.450 micron reflective surface P2V.
The existing stationary C++ mode population predicts approximately 2.939 mas radial
RMS and 0.04631 micron surface P2V RMS, consistent with the notebook's nominal
scale. Individual finite records need not equal this ensemble result.

The deployed tweeter additive streams `dm01dispXX` contain **microns of physical
DM surface displacement**, not OPD or voltage. The user-confirmed TT calibration is:

```text
/opt/MagAOX/cacao/tweeter-vispyr/tweeter-vispyr-calibs/default/CMmodesDM/CMmodesDM.00.fits
```

This `(2,50,50)` cube contains an approximately orthogonal rotated TT pair. Each
mode is RMS-normalized to **1 micron of DM surface over the deployed dmmask** at
coefficient one. **P2V = 4 * RMS** is the nominal pure-tilt relation for a uniformly
weighted continuous circular pupil, not an exact requirement on the sampled
actuator grid. The actual masked P2V values are approximately **3.8492 and 3.8128
microns**. For an ideal orthogonal pair and pupil diameter D,
`D * hypot(theta_x,theta_y) * 1e6 / 8` is only an ideal-basis coefficient sanity
check. At 2.9 mas and D=6.5 m, that gives approximately 0.01142, with ideal physical
surface P2V of 0.04569 micron. The simulation is never empirically rescaled to make
a finite realization match these checks, and the supplied modes are never
renormalized by the code.

Read-only validation used local copies from the active calibration
`fklcombo_z10_2400_smoothed_co0.27_pb0.5_mod3_20241113_181929_1563modes`.
Its `dmmask.fits` is binary with 1,715 included pixels and is appropriate fitting
support: the measured mode RMS values are 0.999999927 and 0.999999992 microns.
The first two planes of the supplied full `CMmodesDM.fits` cube are bit-for-bit
identical to this pair. The 2,039-pixel `bmc_2k_actuator_mask_F32.fits` used by
`t2wOffloader` is a different actuator normalization mask; it does not replace
`dmmask` for this fit. The telescope AEG/offloading chain is not used to determine
disturbance amplitude. Neither offloader's 0.05 controller gain is used.
`dmmap.fits` is a nonnegative scalar image with undocumented units/provenance;
current evidence does **not** establish it as a coordinate map or fitting weights.
Do not use it for either purpose without further calibration evidence.

`accelSimDetail::tipTiltProjection` derives the mapping from two supplied mode
arrays, explicit physical pupil X/Y coordinate arrays in metres, and nonnegative
pupil fitting weights. All arrays share native pixel order `column + width*row`.
For the current simulation convention, the expert-supplied pupil center is
`(24.5,24.5)`, the nominal pupil diameter is **48 actuator pitches**, and the
entrance pupil diameter is **6.5 m**. Construct the supplied maps as:

```text
metres_per_pitch = 6.5 / 48
x[column + 50*row] = (column - 24.5) * metres_per_pitch
y[column + 50*row] = (row    - 24.5) * metres_per_pitch
```

These values specify a nominal linear, isotropic pupil scale. They are supplied
geometry, not inferred from cube dimensions, mode RMS, or the mask centroid.
The existing generic projector already accepts these maps; no new numerical
helper, hardcoded calibration arrays, or production mapping constant is needed.
No external instrument-axis registration is implied or required for this
intentional DM-pupil simulation convention.

Let B contain the two surface-micron mode columns, W the pupil weights, and
`H = 0.5e6 * [x,y]` the required surface per radian. The helper solves

```text
K = argmin ||sqrt(W) * (B*K - H)||
coefficients = K * theta_rad
surface_um = B * coefficients
```

It uses Eigen's column-pivoted Householder QR, with internal column scaling for
numerical conditioning and restoration of the original units. This supports
rotations, reversed signs, swapped modes, and nonorthogonal pairs without a
hard-coded angle or a direct X/Y correspondence. Both mode and coordinate matrices
must have rank two (relative pivot threshold 1e-10 after column scaling).
`relativeResidual()` reports weighted residual norm divided by target norm for
each axis. Construction requires an explicit maximum allowed residual in [0,1);
there is no assumed deployment tolerance. Non-TT components can cause a residual
and are not silently removed through piston subtraction or mode renormalization.

Zero weights exclude samples from the fit; coordinates there may be nonfinite.
Every mode pixel must be finite because the **entire supplied modes**, including
outside-pupil pixels, define the output. The helper does not impose an output mask.
Local calibration analysis found rank two, condition number 1.000489585, and
normalized weighted mode dot product -0.0004894653. Pixel X/Y ramps centered at
column/row (24.5,24.5) reconstruct with relative residuals 8.47e-8 and 8.63e-8.
With the explicitly supplied nominal geometry above, this now also determines
the physical mapping for the chosen simulation axes. There are 244 nonzero
pixels outside the mask in each mode; retaining
the full supplied mode shapes is intentional. Unit tests use synthetic modes
and explicitly registered coordinates independently of these local FITS files.

For these exact local modes, nominal geometry, and `dmmask`, the measured mapping
is shown below for review; it is **not hardcoded** into the helper:

```text
K [coefficient/radian] = [[-798911.751168, -315942.594932],
                         [ 318826.690493, -792420.649088]]
K [coefficient/mas]    = [[-0.003873233470, -0.001531732925],
                         [ 0.001545715415, -0.003841763719]]
```

The condition number is 1.00832628. Positive mode gradients are at approximately
-158.082845 and -68.422878 degrees, measured from increasing column toward
increasing row. Their 89.659968-degree separation and unequal slopes are retained;
the projector does not replace the pair with a perfect rotation.

| Pointing | c0 | c1 | Sampled surface P2V [micron] | Ideal circular-pupil P2V [micron] |
| --- | --- | --- | --- | --- |
| +1 mas X | -0.003873233470 | 0.001545715415 | 0.01542818566 | 0.01575644464 |
| +1 mas Y | -0.001531732925 | -0.003841763719 | 0.01542818690 | 0.01575644464 |
| +2.9 mas X | -0.011232377062 | 0.004482574702 | 0.04474173842 | 0.04569368944 |
| +2.9 mas Y | -0.004442025482 | -0.011141114784 | 0.04474174201 | 0.04569368944 |

For these axis-aligned cases, fitting samples span indices 1 through 48, a
47-pitch separation. Their P2V is therefore 47/48 of the nominal 48-pitch ideal,
within the measured calibration residual. The full supplied-mode images have
the same extrema in these cases. Do not force sampled P2V to the ideal by
renormalizing the modes, adjusting vibration amplitude, or changing the diameter.

At +2.9 mas, masked double-precision surface residuals are 1.02437e-9 micron RMS
and 4.71251e-9 maximum for X, and 1.03548e-9 RMS and 4.27759e-9 maximum for Y.
The relative residuals remain 8.47e-8 and 8.63e-8; float command synthesis keeps
both below 8.9e-8. Fitted center piston is below 3.3e-11 micron for these cases.
The asymmetric mask gives a small nonzero target mean, which is preserved rather
than subtracted. The basis is sufficiently planar for this numerical conversion;
runtime acceptance uses the configured `dm.maxRelativeResidual` tolerance.

An equivalent slope calculation fits the two mode gradients into G in surface
microns per entrance-pupil metre. Then `theta_rad = 2e-6 * G * coefficients` and
`K = inverse(2e-6 * G)`. Its relative agreement with the coordinate projection is
2.1e-12. Converting micron-per-pixel slopes into G requires the same `6.5/48`
metre scale: a sampled surface slope alone does not determine entrance-pupil
angle. An empirical angular calibration could absorb that scale, but the
telescope-secondary AEG offload path is not used here.

`projection.response()` returns the two full-image angular-response columns B*K,
in **surface microns per radian of simulated X/Y pointing**. Supply these to
`tipTiltBasis(width, height, response[0], response[1])` to reuse its preallocated
float-image synthesis. `projection.coefficients(pointing)` exposes the matching
modal amplitudes. No additional reflection factor, pupil scaling, or driver gain
is applied after projection. Neither helper opens a stream or chooses a channel.

Calibration storage is copied/validated at construction; synthesis reuses a
preallocated float image and replaces the whole contribution without accumulation.
Nonfinite data, invalid dimensions, and output overflow are rejected. An overflow
zeros the entire local output before throwing. `zero()` also clears only local
memory: neither action promises that an applied DM command was cleared. Returned
image references are owned by the helper and change on its next synthesis/zero
call; one thread must own that mutable state. Numerical checks do not establish
that calibration planes are physically correct, independent, or safe to apply.

To test these helpers with the existing Eigen dependency, without MagAO-X runtime
dependencies or shared memory:

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra -DCATCH_CONFIG_MAIN \
  $(pkg-config --cflags eigen3) \
  apps/accelSim/tests/accelSimDisturbance_test.cpp \
  -o /tmp/accelSim_disturbance_test
/tmp/accelSim_disturbance_test '[disturbance]'
```

Do not compile the numerical checks with fast-math: finite-input validation and
the near-unit-root vibration model require ordinary floating-point semantics.
The suite is also registered in `tests/tests.list` with the usual application
unit-test group. All bases in its tests are synthetic, explicitly defined fixtures.

Runtime integration uses these unchanged numerical helpers with the explicit
channel, calibration, enable, and scale controls described above. Exclusive
channel handover and native publication/timing/zero verification remain required
before commissioning. External sky/AzEl registration would be additional work
only if that interpretation is requested later.

## Model and timing

The model preserves the notebook's 14 prominent modes and 200-mode population
over 100–900 Hz, acceleration RMS envelope, direct pole radius `r`, and separate
X/Y WFS couplings. Each mode has independent X/Y histories, stationary Gaussian
initialization, and analytically scaled innovations. Pointing is in arcsec and
each mode contributes `-(2*pi*f)^2 * leverArm * theta_rad` to acceleration.
Independent accelerometer noise is added after summing the modes. WFS-coupled
pointing is retained internally without sensor noise or a separate output stream.

The ADC calculation is `zero + acceleration*sensitivity/voltsPerBit`, followed
by `std::round`, floating-point clamping, and conversion to `uint16_t`. Defaults
are 2048, 1.02, and 0.0083799 respectively. The three values and the sensor PSD
are configurable in [config/example.conf](config/example.conf).

Default sampling is 6000 Hz. The physics uses the configured nominal rate even
when a synchronization source has scheduling jitter. Set the requested accelerometer
rate through `sim.fps`, `fps.target`, or the external FPS subscription. Invalid,
nonfinite, nonpositive, above-10000, and aliased rates are rejected. An accepted
rate change takes effect at an acquisition boundary, recalculates the model,
and redraws stationary histories without reseeding. This introduces an explicit
counted discontinuity. The high-frequency parameter list is retained.

Timer acquisition uses an absolute monotonic schedule with fractional-period
accumulation and interruptible sleeps. Up to **eight skipped ticks** are advanced
before generating the current pair; only the newest pair is published. A larger
gap redraws stationary histories instead of performing unbounded catch-up.
Eight ticks represent 1.333 ms at 6 kHz. OS wake-up lateness is counted separately
from skipped publication opportunities. There is no catch-up publication burst.

Setting `synchro.shmimName` enables a **WFS phase reference for a continuous
accelerometer clock**. The worker waits for either the next accelerometer deadline
or a WFS semaphore. A semaphore updates the phase reference; it never publishes a
sample or moves the already pending deadline. Sampling starts even if the reference
stream is absent, and continues through silence, missing notifications, or source
replacement. These events clear lock/cadence diagnostics and enter holdover at the
last applied phase; they do not recreate acc000 or restart the physical model.

The measured WFS period is elapsed time divided by source-counter progress,
initialized from the first measured interval and then smoothed by `alpha`. The
phase remains `wrap(0.5*T_wfs - (dtTransfer + wfsProcess + dtF + wfsRead), T_wfs)`,
with the signed `postDelay` added and wrapped. The default latency sum is 340.6 us,
so a measured 500-us period gives 409.4 us without a hard-coded phase constant.

The clock subtracts elapsed **fractional WFS cycles** before projecting the phase
reference onto the accelerometer time grid. Thus a 2-kHz reference works with
8 kHz, 6 kHz, 5.5 kHz, and other valid sampling rates; no integer samples-per-WFS
counter or per-trigger burst is used. With a non-integer ratio, not every WFS
reference can coincide with an accelerometer sample while preserving the requested
rate. The relative phase advances continuously across WFS cycles instead.

Three synchronization options control this behavior:

- `synchro.referenceFps=0`: learn the nominal WFS phase-tracking cadence from the
  first measured interval after acquisition/reconnection. Set the known nominal
  WFS rate (for example 2000) when available to avoid basing the phase grid on a
  noisy first interval. This affects fractional-cycle accounting, not accelerometer
  FPS; the latency formula still uses measured/smoothed cadence. The learned value
  is held until timing is invalidated. A changed WFS nominal rate requires matching
  this setting/reacquiring the reference. A wrong nominal reference can produce
  repeated correction transients; it cannot force the output rate to the WFS rate.
- `synchro.maxSlew=0.05`: limit the phase adjustment of each following interval to
  5% of an accelerometer period (valid range greater than zero through 0.25).
  `cadenceGuard_ns` and measured computation time further limit available correction
  headroom. Phase offset stays within half an accelerometer period of the nominal
  epoch, so corrections cannot accumulate a long-term rate change. Lock acquisition
  takes several samples. After an overrun, the next publication is at least half
  an accelerometer period later, preventing an immediate recovery burst.
- `synchro.timeout=0.01`: reference silence timeout and source-file check/retry
  interval in seconds. Increase this above the expected source period for slow
  references. Timeout clears lock while sampling continues; a replacement source
  needs fresh timing before relocking.

Source-counter gaps count missed **WFS references** only. Actual missed
accelerometer deadlines count `skipped_publications` and advance numerical ticks
using the existing eight-skipped-tick bound; larger gaps restart stationarity once.
Counter receipt is accounted immediately, even if a subsequent wait is canceled.
Duplicates cannot create extra samples. The code never retunes AR coefficients
from WFS jitter. Keep `fps.device` empty unless that subscription supplies the
requested **accelerometer** FPS; subscribing it to a 2-kHz WFS FPS still explicitly
requests a 2-kHz accelerometer rate.

For the 8-kHz / 2-kHz example, set `sim.fps=8000`,
`synchro.shmimName=camwfs_sync`, `synchro.referenceFps=2000`, and leave
`fps.device` empty. The free-running 6000-Hz defaults and timer path are unchanged.

Acquisition timestamps use `CLOCK_REALTIME` immediately before sample generation,
including recovery computation. They represent the current acquisition event,
not reconstructed timestamps for missed frames. `frameGrabber` owns write time,
`cnt0`, `cnt1`, per-plane metadata, circular buffering, and all output semaphore
posts. The image has three axes `[2,1,depth]` even at depth one. Wall-clock
corrections can change epoch timestamps; pacing remains monotonic.

## Controls and diagnostics

The app uses `MagAOXApp<true>`, `frameGrabber`, and the required `telemeter` mixin.
It exposes the normal `fg_*`, thread, FSM, logging, and telemetry properties, plus
the hardware application's `fps`, `numChannels`, `alpha`, `synchroDelay`, and all
23 `timingDiag` elements. `fps.current` is the active nominal rate; `target` may
briefly reflect a pending update. FPS requests report Busy until the acquisition
worker acknowledges that request, then Ok; invalid requests retain Alert until a
valid request. Superseding requests cannot inherit an earlier acknowledgment.
Measured publication rate is in `fg_timing`.
`numChannels` accepts only 2, including on its external subscription; rejected
updates leave the shape and accepted values intact and set Alert. There is no
additional camera-style start/stop control.

`channel_readout_us` measures simulation acquisition computation, not fictitious
SPI hardware timing. In timer mode synchronization-only measurements are zero.
`alpha` controls timing smoothing, never AR damping.

Each `timingDiag` update sends one coherent property, including source-loss
invalidation. Names and units are retained, but the synchronized meanings now
reflect a phase-referenced sampling clock rather than one ADC read per semaphore:

- `delay_model_us` / `synchro_delay_target_us`: measured-cadence WFS model and
  offset-adjusted reference phase. `trigger_time_us` / `trigger_interval_us` describe
  those WFS predictions, not accelerometer intervals.
- `delay_applied_us` / `synchro_delay_us`: reference phase including the remaining
  offset error of the scheduled accelerometer clock, wrapped over a WFS period.
  `delay_phase_error_us` retains the positive half-period boundary. `delay_lock`
  requires a fresh reference and the existing absolute/fractional phase thresholds;
  excessive acquisition lateness also clears it. Lock is not an independent
  measurement of hardware/WFS latency.
- `delay_budget_us` / `delay_capped`: available per-accelerometer-interval phase
  correction and whether correction remains beyond that budget.
- `avg_read_latency_us`: smoothed arrival-to-sample latency for selected WFS
  references, measured at the first sample at/after their predicted phase.
  `read_latency_error_us` is that sample's lateness relative to its retained
  reference prediction. A pending measurement survives an intervening WFS event;
  not every reference necessarily contributes a measurement.
- `sync_frames_received` counts semaphore receipts; source ID/gap/drop fields
  describe WFS counters. `sync_frames_written` counts accelerometer publications
  in synchronized mode, including holdover, and can exceed received references.
- `sim_diag.source_period_ratio` is measured WFS period / accelerometer period:
  approximately 4 for 8/2 kHz and 3 for 6/2 kHz, not an error requiring equal rates.

The additional read-only `sim_diag` contains `skipped_publications`,
`discontinuities`, `late_samples`, `max_lateness_us`, `source_period_ratio`, and
`clip_x_low`, `clip_x_high`, `clip_y_low`, `clip_y_high`. Clipping counts actual
clamps after rounding, including numerical samples advanced during catch-up;
values rounding naturally onto a rail are not clamp events. Counters are retained
over FPS changes. A shutdown log summarizes timing and clipping totals. A
nonblocking snapshot transfers normal acquisition diagnostics to the ordinary app
loop, without INDI calls, logging, or allocations in normal numerical stepping.
Lifecycle/outage transfers require delivery and may wait only for the app loop's
short snapshot copy; that mutex is released before any INDI work.

Seeds default to 42 (samples) and 123 (mode parameters). Repeatability is within a
fixed C++ standard library/build and fixed numerical stepping schedule, not across
NumPy RNGs or different OS timing. Stationary RMS is an ensemble property: a
60-second record of an `r=0.9999999` mode is not forced to the configured RMS.
PSD calculation stays in Python.

## Build and focused tests

From the repository root, on a configured MagAO-X development host:

```sh
make -C apps/accelSim
make -C tests -f Makefile.one t=../apps/accelSim/tests/accelSim_test.cpp
./apps/accelSim/tests/accelSim_test '[accelSim]'
./apps/accelSim/tests/accelSim_test '[indi-regression]'
./apps/accelSim/tests/accelSim_test '[accelSim-streams]'
```

The last test creates a private temporary ImageStreamIO namespace and exercises
real frameGrabber publication and a synthetic semaphore source. It does not
connect to a live INDI server or instrument. Run it separately from the live
benchmark because ImageStreamIO can cache its shared-memory directory.

The numerical portion can run without MagAO-X dependencies (including on macOS):

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra \
  -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY \
  apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_model_test
/tmp/accelSim_model_test '[model]'
/tmp/accelSim_model_test '[sync-regression]'
```

The hardware-independent DM runtime tests additionally require Eigen:

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra -DACCELSIM_MODEL_ONLY \
  -DCATCH_CONFIG_MAIN $(pkg-config --cflags eigen3) \
  apps/accelSim/tests/accelSimDM_test.cpp -o /tmp/accelSim_dm_test
/tmp/accelSim_dm_test '[dm-runtime]'
```

On a configured non-instrument Linux development host, build the same test with
`tests/Makefile.one`, then run `[dm-controls]` and, in a separate process,
`[accelSim-dm-streams]`. The latter creates synthetic images with private names
under a fresh `/tmp` ImageStreamIO namespace and rejects a cached different path
before creating anything. It tests native active counters/semaphores, geometry/type
rejection, and the app's same-sample post-publication/zero path. It never names a
deployed DM stream. Sanitizers should also cover the numerical/runtime suites.

The standalone tests include the production synchronization accounting and
status-invalidation logic, integer/fractional rate ratios, correction bounds,
jitter/phase shifts, holdover, half-period phase boundaries, and trigger startup.
The INDI regressions capture real XML serialization without a server, checking
coherent timing snapshots and FPS Busy/Ok/Alert transitions. Their local driver
fixture uses Linux `/proc/self/fd` to release its unactivated input descriptor.
Full app tests additionally check snapshot delivery,
continued sampling with a missing or duplicate synthetic reference, stale-frame
rejection, and source disappearance/reconnection; those require the normal dependencies.

The app and normal test Makefiles disable fast-math locally, since nonfinite
validation and near-unit-root calculations require ordinary floating-point
semantics. Do not override this with fast-math when validating.

## Local launch and inspection

Use a configured **non-instrument Linux development host**. The following creates
private config/log/FIFO/shared-memory paths and uses a local test instance, not
an invented deployment identity. The example must be used alongside the host's
normal MagAO-X runtime prerequisites and user/group permissions. No INDI tunnel
to an instrument should be used for these checks.

```sh
accel_sim_root=$(mktemp -d /tmp/accelSim-local.XXXXXX)
export MAGAOX_PATH="$accel_sim_root"
export MILK_SHM_DIR="$accel_sim_root/shm"
mkdir -p "$MILK_SHM_DIR" "$MAGAOX_PATH/config" "$MAGAOX_PATH/logs" \
  "$MAGAOX_PATH/telem" "$MAGAOX_PATH/sys" "$MAGAOX_PATH/calib" \
  "$MAGAOX_PATH/drivers/fifos"
./apps/accelSim/accelSim -n accelSim-local \
  -c apps/accelSim/config/example.conf --framegrabber.threadPrio=0 \
  >"$accel_sim_root/app.log" 2>&1 &
accel_sim_pid=$!
```

Priority zero is for an unprivileged smoke test. The approved performance test
uses the normal framework priority/cpuset settings on a suitable development
host. Supply those settings when evaluating performance. A local INDI server,
if used, must attach to this private FIFO directory. The exact deployment
instance, source stream, circular depth, and downstream configuration remain
operator-supplied settings.

In the same environment, an existing reader can inspect the unchanged stream:

```sh
shmimLat -n acc000
```

For a ten-minute read-only benchmark of the running producer:

```sh
ACCELSIM_LOCAL_BENCHMARK=1 ./apps/accelSim/tests/accelSim_test '[accelSim-live]'
```

For a producer configured at 8 kHz, add `ACCELSIM_EXPECTED_FPS=8000` to the
benchmark command. This changes only the reader's expected-rate assertion; it
does not configure the producer. The default remains 6000 Hz.


It checks ADC ranges, counter progression, and mean publication rate within 1%
of 6000 Hz, and reports reader counter gaps and write-interval mean, standard
deviation, maximum, and histogram. Histogram upper bounds are 100, 150, 175,
200, 250, 500, and 1000 microseconds, followed by an overflow bin. Intervals are
computed only for consecutive observed frames; a slow reader cannot measure
every producer interval. Reader gaps must not be confused with producer skips.
Collect the producer's `sim_diag` values and acquisition/write timing properties
through the isolated local INDI session as well. Record RSS during the run, for
example from a second local shell with the captured PID:

```sh
ps -p "$accel_sim_pid" -o pid,etime,rss,pcpu
```

Take repeated RSS measurements after startup to assess stable memory. There is
no approved hard jitter/miss threshold yet; report the distribution and every
counted miss. Stop this local producer and inspect its summary:

```sh
kill -TERM "$accel_sim_pid"
wait "$accel_sim_pid"
cat "$accel_sim_root/app.log"
```

## Offline visual validation on macOS

This workflow calls the same C++ `model::step()` as the application, with no
MagAO-X runtime dependencies, shared-memory streams, INDI, or wall-clock pacing.
The finite record is a diagnostic export only; production remains continuous.
The export uses the normal model defaults: **6000 Hz, 60 seconds, seed 42,
modeSeed 123**, producing 360000 X/Y pairs from 214 resolved modes.

From the repository root, build and export:

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra \
  -DCATCH_CONFIG_MAIN -DACCELSIM_MODEL_ONLY \
  apps/accelSim/tests/accelSim_test.cpp -o /tmp/accelSim_model_test
mkdir -p /tmp/accelSim-visual-validation
ACCELSIM_EXPORT_DIR=/tmp/accelSim-visual-validation \
  /tmp/accelSim_model_test '[accelSim-export]'
```

The normal full test binary also supports `[accelSim-export]`. Exported files:

- `samples.csv`: nominal time [s]; physical and WFS pointing X/Y [arcsec];
  acceleration X/Y including sensor noise [m/s²]; raw ADC X/Y [counts].
- `modes.csv`: resolved frequencies, per-axis acceleration RMS, pole radii,
  and X/Y WFS couplings used by this C++ realization.
- `metadata.json`: sample rate, duration, both seeds, lever arm, ADC calibration,
  noise PSD, and the model's actual lower/upper clipping counts for each axis.

Prepare an offline Python environment once, then generate the figures:

```sh
python3 -m venv /tmp/accelsim-validation-env
/tmp/accelsim-validation-env/bin/python -m pip install numpy scipy matplotlib
/tmp/accelsim-validation-env/bin/python apps/accelSim/tests/validate_psd.py \
  /tmp/accelSim-visual-validation
```

Add `--show` to also open interactive Matplotlib windows (the plot files are
saved first). `--time-start 5 --time-seconds 0.5` selects a different short time
trace; PSDs and RMS always use the entire exported record. Defaults show the
first second. Without `--show`, plotting uses the headless Agg backend.

Results are printed and saved in `validation.json`; all measured and predicted
spectra are saved in `psd.csv`. Figures are saved under the export directory:

| File in `plots/` | Meaning |
| --- | --- |
| `pointing_timeseries.png` | Physical pointing X/Y in mas, first second by default. |
| `pointing_psd.png` | Physical pointing PSD in mas²/Hz; narrow low-frequency modes tend to dominate the pointing variance. |
| `wfs_psd.png` | WFS-coupled pointing PSD in mas²/Hz; each mode's power is multiplied by its axis coupling squared, without accelerometer sensor noise. |
| `accel_psd.png` | Acceleration PSD in (m/s²)²/Hz, including the 100–900 Hz population and sensor noise. |
| `adc_psd.png` | Inverse-calibrated ADC acceleration PSD, overlaid with the same pre-ADC realization to expose quantization/clipping effects. |

All PSD figures mark the 14 prominent frequencies. Welch uses SciPy with
`window="hann"`, `nperseg=10*fs` (limited to record length for shorter inputs),
50% overlap, `detrend="constant"`, and `scaling="density"`.

Dashed curves are stationary ensemble AR(2) predictions, including the **same
finite Hann window and constant detrending** as the estimator. They are computed
from the analytic lag covariance, not fitted to the record. This avoids comparing
an unresolved high-r resonance's infinite-record peak height directly with a
10-second-window estimate. Physical/WFS theory uses the lever-arm conversion and
squared axis couplings; acceleration theory adds the configured white sensor noise.
The ADC plot's theory is explicitly **pre-ADC**: it does not model quantization
or clipping. The noise-dominated tail should approach 1e-8 (m/s²)²/Hz before
quantization; default ADC quantization adds roughly 1.87e-9 (m/s²)²/Hz under the
usual white quantization-error approximation.

Reported RMS is `sqrt(mean(x²))`, without subtracting the record mean. X/Y
realizations are independent. Very high-r modes can have different realized
amplitudes across axes/seeds and differ from the ensemble RMS in a 60-second
record; this is expected, not a failed normalization check. Clipping totals count
actual model clamps, separately from valid samples that happen to equal 0 or 4095.
Older exports without metadata use documented default calibration and reconstruct
clamp counts from their acceleration values; regenerate data to obtain exact C++
counters. Analysis flags such as `--fps`, `--lever-arm`, and `--volts-per-bit`
override metadata for analysis only; they do not regenerate or alter the simulation.

Optional offline checks:

```sh
/tmp/accelSim_model_test '[model]'
/tmp/accelsim-validation-env/bin/python -m unittest discover \
  -s apps/accelSim/tests -p 'test_validate_psd.py' -v
```

## Current verification limits

The macOS numerical tests (also with address/undefined-behavior sanitizers) and
offline export/PSD comparison have passed. The
normal app and full test builds are blocked before app compilation by existing
mxlib/platform dependencies (`mx/math/floatUtils.hpp` missing and `timegm`
undeclared). No Linux stream, INDI runtime, or ten-minute performance result is
claimed. The real deployment configuration and raw-ADC DDSPC consumer are absent
from this checkout, so final downstream substitution still requires those inputs.
See the planning document for exact results and remaining coverage.
