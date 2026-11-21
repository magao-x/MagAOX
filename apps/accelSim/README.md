# accelSim

Continuous accelerometer simulation with the `mcp3208Ctrl` raw-count interface.
The default output is **`acc000`**, with frames containing X then Y as two
`uint16_t` values in [0,4095]. Only one producer may own that stream and its INDI
instance at a time. The executable name does not determine the replacement INDI
identity: supply the real instance with `-n` when its configuration is available.

The approved specification and engineering record are in
[`acc-sim-app-creation.md`](../../agents/plans/2026-10/acc-sim-app-creation.md).
Optional tweeter disturbance injection of the same simulated realization, evaluated at
the scheduler's target times, is implemented separately from the model and remains
disabled by default. The default configuration has no DM
channel and performs no DM access. No instrument startup configuration is supplied.

## Zero-acceleration ADC test output

Set this startup-only option to publish constant zero-acceleration counts:

```ini
[sim]
zeroOutput=true
```

The default is `false`, which preserves normal vibration/noise ADC output.
With the current `adc.zero=2048` calibration, every output pair is **2048,2048**,
not raw0,0. The override reuses the existing ADC encoder at acceleration0, including
rounding/clipping of a custom configured zero point. It applies only when copying
the pair into the frameGrabber destination. The underlying vibration model, RNG,
sensor noise, clipping diagnostics and DM disturbance injection continue unchanged.
In particular, zero ADC output does not disable or zero an enabled DM disturbance.
ImageStreamIO datatype/geometry, pacing, counters, timestamps and semaphore
publication remain framework-owned and unchanged.

Portable `[zero-output]` tests verify calibration and8,000 paired physical samples
without any stream or device. On a provisioned isolated Linux development host,
run the hidden native publication test in a fresh process:

```sh
./apps/accelSim/tests/accelSim_test '[accelSim-zero-streams]'
```

It checks every observed pair for2048,2048, advancing counters/acquisition times,
and consistent per-plane timestamps at temporal depths1 and4. It verifies its
private ImageStreamIO namespace before opening/creating anything. Native tests
are not run by the portable model-only build; they remain unverified on this Mac.

## Optional runtime DM injection through syncopation

The source is **`dm01disp10_aux`**. This workflow assumes accelSim is its sole producer.
The existing `dmSyncopationProducer` base owns scheduling,
exclusive cooperative producer ownership, finite-payload validation, tagged
`_ready` commits and exact copy/discard `_ack` handling. The implementation follows
the direct poll/publish pattern in the historical syncopation-enabled `dmSpeckle`
implementation (`7f59b4f7`); this checkout's `dmSpeckle` retains its legacy writer.
Only the base writes the auxiliary source. accelSim never writes a DM destination.

The operator workflow is:

1. Set `dm_scale.target` to a finite value in [0,100]. It linearly multiplies the
   calibrated noise-free pointing surface; values above 1 amplify the physical disturbance.
2. Toggle `dm_enable.toggle` on to publish the scaled physical disturbance.
3. Toggle `dm_enable.toggle` off to submit a tagged zero. Wait for
   `dm_status.zero_acknowledged=1` and `active=0` before transferring ownership.

Toggling off alone removes the contribution; a following `dm_scale.target=0` only
sets the scale for the next enable. With the toggle on, `scale=0` publishes zero
surfaces that are still counted as an active (enabled) contribution. This differs
from `dmSpeckle`, whose modulation-off retains its last contribution until a separate
`zero` request: the scheduler retains the last copied contribution of any stopped
producer, so accelSim queues the zero automatically and retries it until copied.

A normal build supports this workflow. There is no simulator operational/shadow
build profile, output authorization flag, routing assertion, exact-rate gate,
WFS-lock gate, 250-us freshness watchdog or fault latch. Remove the obsolete
`dm.output_authorized`, `dm.routing_verified`, `dm.max_abs_um`, `dm.max_step_um`
and `dm.ack_timeout_us` keys from older simulator configurations. loPredCtrlAcc
keeps its own existing configuration and output policy unchanged.

### One realization at acquisition and target times

[`accelSimTimeline.hpp`](accelSimTimeline.hpp) holds a single realization of the
unchanged model on a fixed `CLOCK_MONOTONIC` grid. Node *k* is at `epoch + k/fps`,
the nominal time of accelerometer tick *k*; the epoch is the acquisition schedule's.
The acquisition thread alone steps the model (and its RNG). After each acquisition
it generates nodes 5 ms ahead, normally one `model::step()` per acquisition, as before.

- **acc000** is evaluated at the actual acquisition time (the instant stamped on the
  frame) plus the sensor-noise draw belonging to the consumed tick, then quantized by
  the unchanged ADC encoder. A node-aligned acquisition reproduces the legacy sample.
- **The DM** thread evaluates noise-free physical pointing at each request's
  `targetNs`, exactly as `dmSpeckle` evaluates its waveform at `targetNs`. The
  scheduler, producer and accelSim all use the same-host `CLOCK_MONOTONIC`;
  wall-clock (`CLOCK_REALTIME` frame timestamps) and shared-memory times are not used.

Between nodes, 4-point cubic Lagrange weights are applied identically to pointing,
WFS pointing and noise-free acceleration. Because the weights are linear, every
evaluated time keeps each mode's `a = -(2*pi*f)^2 * leverArm * theta_rad` relation
exactly. At 8 kHz the amplitude error is <1e-4 up to 300 Hz and 0.56% at 900 Hz
(linear interpolation: 6.2% at 900 Hz, varying with sub-sample position). Over the
continuous trajectory, the regression of `leverArm * theta''` on `a` is 1.0002 at 28 Hz
and 0.998 at 120 Hz.

Measurements never leave the acquisition thread before their acquisition time.
The cubic evaluation of a measurement uses nodes up to two grid intervals after it
(weight at most 0.064). Summed over all 214 modes, the worst-case later innovation
reaching it that way is 9.5e-5 m/s^2 rms. That is 1/86 of one ADC count, 1/67 of the
sensor-noise rms and 0.24% of the physical acceleration rms; no node is ever published.
The DM does see the future relative to acc000: it is the disturbance itself.
Controllers must therefore not read `dm01disp10_aux` or `dm01disp10` as inputs.

DM enable state and DM polling only read the timeline, so they cannot change the
realization. The ring holds 128 nodes (about 9 KB). Its mutex covers only short
copies; DM reads take about 16 ns. If a target lies outside the generated horizon,
the slot is skipped and counted (`timeline_misses`); the scheduler keeps the last
copied contribution. The production scheduler publishes targets at most two primary
periods ahead (<= 4 ms at its default 500-Hz minimum main rate), within the horizon.

Producer waits never run on the acquisition thread. Both timer and WFS
phase-reference acquisition work with DM output; neither sample age nor phase
lock qualifies it. A numerically invalid command publishes a zero, counted as a
removal (not active), until the next valid synthesis. Ordinary missed slots,
unavailable transport and discarded receipts are retried without a fault reset.

Calibration requires a finite 50-by-50 TT pair and a rank-two mode-coefficient
calibration: either a physical fit within the residual tolerance, or explicit
coefficients (see "Calibration methods"). File identity
is checked while loading; acquisition uses the validated in-memory calibration.
The source must already be a compatible CPU float32 50-by-50 auxiliary image,
with its scheduler and acknowledgement stream available. The base's cooperative
lock does not stop unrelated direct writers; exclusive source handover remains
the operator's responsibility. No source access occurs while initially disabled.
Startup `enabled=true` is rejected. `scale=0` is the supplied startup default;
a configured finite scale in [0,100] is also accepted while output remains off.

The read-only `dm_status` elements describe the producer handshake. Counters accumulate since process startup.

| Element | Meaning |
| --- | --- |
| `ready` | Calibration was prepared; this is informational and does not depend on acquisition timing. |
| `attached` | The producer currently holds the source lock and scheduler/receipt sidecars. |
| `requested` | Current `dm_enable` operator request. |
| `active` | Last copied payload was an enabled command. This is the scheduler's copy receipt, not confirmation that the DM applied it. |
| `zero_acknowledged` | An exact copied-zero receipt was observed, with no later payload submitted. |
| `fault` | Calibration unavailable or the current sample cannot produce a finite command; no latched code 5. |
| `transport_fault` | Current attachment/transport error; clears on recovery. |
| `poll_waits` | Attached polls returning zero. These include ineligible schedules, repeated slots and waits for a receipt. |
| `requests` | New eligible slots returned by the base. |
| `timeline_misses` | Eligible slots skipped because their target was outside the generated timeline. |
| `published` | Successful payload and ready-record commits by this instance. |
| `late_returns` / `rejected` | Failed publish calls observed at or after the cutoff / before the cutoff. The base returns a boolean. A late return alone cannot identify the internal check that failed. |
| `outstanding` | The base is holding the payload for the scheduler, including a pending commit adopted on startup. |
| `copied` / `discarded` | Exact dispositions observed for this instance's commits. |

The scheduler sets the step count for all mapped destinations. On the 2 kHz
production grid, steps 1, 2, 3 and 0 target 125, 250, 375 and 500 µs. Step 0
belongs to the next primary cycle. With `readyLeadUs=70`, the input cutoffs
are 55, 180, 305 and 430 µs. `step0LeadUs` and `copyBudgetUs` govern later
scheduler work. Retain `syncopation.writeLeadUs=50` for the supplied configuration.

For quiet auxiliary streams, compare counter deltas over the same interval.
If `poll_waits` rises with `requests=0`, check `outstanding` and the current
timing payload. The scheduler can publish invalid records at the primary rate
while its runtime `enable.toggle` is Off. `production.enabled=true` selects
the scheduler implementation. Check the runtime enable and `settings.steps`
separately. Rising `requests` and `late_returns` show failures after request
selection. Once `published` rises, compare `copied` and `discarded` with
scheduler `production.accepted3`, `missing3` and `rejected3` for the supplied
destination ordering.

The states are distinguished as follows. `attached=1` with rising `poll_waits` and
constant `requests` means the producer is waiting for eligible scheduler requests.
Rising `published` means payloads were committed. Rising `copied` means the scheduler
copied them, while `active=1` shows the last copy was an enabled command.
`zero_acknowledged=1` with `active=0` means removal was copied. Only
`transport_fault=1` (or `attached=0` while enabled) indicates a transport failure.

### DM worker and 8-kHz publication timing

The DM worker follows dmSpeckle's producer pattern. It is started by the framework
`threadStart` with `dm.threadPrio` and `dm.cpuset`, the same keys as dmSpeckle's
`[modulator]`. The supplied `acc000.conf` uses the established producer settings
`threadPrio=20` and a dedicated cpuset (`spare31`), separate from `dmtweeter-synco`
(priority 30) and the acquisition core. Defaults are 0 (`SCHED_OTHER`) and empty.
After an unsuccessful poll the worker sleeps 10 us while output is active, 1 ms when
idle and 100 ms after a transport failure. It polls again at once after a commit.
The DM worker no longer shares a mutex with INDI property I/O, so app-loop updates
cannot delay it.

**Timing contract** (from the producer base and the dev scheduler):

- The scheduler publishes one record when it observes each primary (main) update. The record holds
  the targets of steps 1, 2, 3 and 0 (the next main) at main + 125/250/375/500 us. Each cutoff is
  `readyLeadUs=70` earlier; the scheduler snapshots, copies and acknowledges at the cutoff.
- `syncopationPoll()` returns the earliest step whose cutoff is still ahead, and only after the
  previous commit's receipt (or a plan reset). One command can be outstanding at a time, and a
  step becomes eligible only after the previous slot's cutoff.
- `syncopationPublish()` waits until `cutoff - writeLeadUs`, publishing at once if that has passed,
  and must complete by the cutoff. `writeLeadUs` only delays early commits.
- Steps 2, 3 and 0 are therefore eligible for about 120 us before their cutoffs. Step 1 exists only
  from its record's publication until main + 55 us, so its margin is bounded by the scheduler
  (`readyLeadUs`, record latency), not by accelSim.

**Advance preparation.** While the worker waits for a receipt, it reads each new record. It then
prepares the surfaces of all remaining steps privately, in four preallocated buffers, from the
timeline at each exact `targetNs`. A staged surface is used only if its slot identity (epoch,
generation, cycle, step, target, cutoff), `dm_scale` and timeline generation match. Disabling
clears the buffers. Otherwise the surface is synthesized on observation. The base still owns the
single source buffer, so nothing is published before eligibility or over an unconsumed command.
On a development Mac a staged publication takes 1.3 us (p99 1.6 us), against 3.0 us when
synthesized on observation. Staging a record (up to four surfaces) takes about 6 us, off the
critical path.

**writeLeadUs.** With surfaces prepared ahead, waiting only shortens the margin. `acc000.conf` sets
`writeLeadUs=125` (one subframe), so each request commits as soon as it becomes eligible. In the
simulated scheduler with prompt wakes, the mean margin rises from 45 to 95 us. Steps 2, 3 and 0
keep more than 70 us, and step 1 keeps 30-45 us; all four steps are still published.

`dm_steps` reports cumulative `advertised_N` (slots seen in scheduler records),
`requests_N`, `published_N` and `copied_N` for each subframe N (0-3). It also reports
`skipped_slots` and `prepared_hits`/`prepared_misses` (staged versus synthesized surfaces).
Use the differences to classify losses:
- Cycle count × steps minus `advertised` were never requested.
- `advertised` minus `requests` were not observed in time.
- `requests` minus `published` were not published in time; see `late_returns`, `rejected`
  and `timeline_misses`.
- `published` minus `copied` were published but not copied (`discarded`). Skipped slots are scheduler slots of one plan
that no poll returned before their cutoffs. `dm_timing` reports statistics for each
app-loop report interval (about 1 s): `samples` (commits), `poll_gap_mean_us` and
`poll_gap_max_us` (time between polls: sleep plus wake latency), and `slack_min_us`
and `slack_mean_us` (cutoff minus first observation of a request). It also reports
`prepare_*` (timeline plus synthesis), `publish_*` (the producer call, including its
intentional wait to `cutoff - writeLeadUs`) and `margin_*` (cutoff minus commit time:
minimum, mean, RMS and the 1% and 0.1% lower quantiles). `lead_*` gives target minus commit
time, `receipt_*` when the copy receipt was observed after the cutoff, and `slack_rms_us` the
observation jitter.

**Primary cadence requirement.** Production syncopation needs the main publisher
(`dm01disp03`) to update at or above the scheduler's `tracking.minMainRateHz`
(default 500 Hz). The main publisher controls step 0. Without updates, the scheduler
reports `production main timeout; reacquiring` and publishes invalid records.
The producer then attaches and waits, with `requests=0` and rising `poll_waits`;
`dmSpeckle` would do the same. Output resumes automatically when valid records
return. A running WFS camera alone does not update the primary. The scheduler
checks primary publications, not optical loop closure or gain. The repository
documents no supported open-loop primary writer, so how to keep `dm01disp03`
publishing without closing the loop remains an operational question.

An acknowledgement records the scheduler's private snapshot or discard. Disable
removal does not require another accelerometer sample. Normal shutdown retains
the existing bounded 100-ms tagged-zero attempt and reports an unacknowledged
removal. A tiny CRTP lifetime holder permits that final zero after MagAOXApp sets
its shutdown flag; it adds no scheduling, ownership or qualification policy.

The current implementation decisions and test limits are recorded in
[accelSim-simple-dm-output.md](../../agents/plans/2026-11/accelSim-simple-dm-output.md).
The complete configuration template is [acc000.conf](acc000.conf). It names the
deployed scheduler stream `dmtweeter_synco_timing`; fill in the absolute
`dm.modeCube` path and the calibration (`dm.calibration`, `dm.coefficients` or
`dm.mask`). The timeline adds no configuration keys.
Retain the deployed `synchro.shmimName` if WFS phase-reference acquisition is
wanted; leaving it empty selects timer acquisition. The template preserves the
8-kHz rate, seeds, ADC calibration, two-channel format and source depth 128.
`config/example.conf` remains an ADC-only example with no DM channel configured.
The older simulator configuration copied under `loPredCtrlAcc/examples` is
unchanged; use the simulator's own template and instructions for this workflow.

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

### Calibration methods (`dm.calibration`, `dm.mask`, `dm.coefficients`)

The DM command is always `c0*mode0 + c1*mode1` over the full first two planes of
`dm.modeCube`, with `c = K * theta` for simulated pointing `theta` in radians. No
actuator value is masked, zeroed or renormalized. Calibration only determines the
2x2 matrix `K` (rows modes, columns simulated X/Y):

- **`dm.calibration=fit` (default):** fit `K` to the physical tilt surface as above.
  With `dm.mask` set, its values are the fitting weights (unchanged behavior: the file
  is stat'ed, read as one 50x50 plane, and checked for changes while loading).
  Without `dm.mask`, every pixel has weight 1.0.
- **`dm.calibration=coefficients`:** use `dm.coefficients = K00,K01,K10,K11` in mode
  coefficient per radian. No tilt surface is fitted and no mask is read. `K` and the
  two modes must be finite and rank two. `dm.mask` is rejected with this method.

**Why the deployed cube fails without a mask.** The fit must reproduce the tilt
surface within `dm.maxRelativeResidual` (1e-6) over the weighted pixels. The deployed
planes are the tilt only over `dmmask` (1,715 pixels, residual about 8.5e-8). There
are 244 nonzero pixels outside `dmmask`, and the remaining 541 pixels are zero, while
the tilt target is nonzero there. With uniform weights (or any support beyond
`dmmask`, such as the nominal 48-pitch pupil of 1,804 pixels), the zero and
outside-pupil pixels enter the fit, and the relative residual cannot be expected to
meet 1e-6. Preparation then fails, `dm_status.ready` stays 0 and `dm_enable` is
refused. For a synthetic cube built that way, the uniform-weight residual is 0.58; the
startup log and `dm_calibration.status` now report the deployed values.
The residual check is correct: it reports that the whole grid is not a tilt plane,
which is exactly what uniform weights ask the fit to assume. Raising the tolerance would
accept a `K` biased by the outside-pupil values. The supported mask-free operation is
`dm.calibration=coefficients` with the `K` measured by the `dmmask` fit (below).
`acc000.conf` uses that `K`, so the commands, orientation, units and amplitude are
the same as with `dm.mask`. If the cube changes, start once with
`calibration=fit` and `mask` set, then copy `K` from the startup log.

**Diagnostics.** The startup log reports the method, support, `K`, fit pixel count
and residuals, or "DM injection unavailable: calibration ...: <reason>". The read-only
`dm_calibration` property carries `status` (`ready`, or the same reason), `method`,
`support`, `k00`..`k11` [coefficient/rad], `residual_x`, `residual_y` and `fit_pixels`.

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

### Optical coupling and independent optical disturbance (`[optical]`)

`sample::m_pointing`, the pointing injected on the DM, is

```text
theta_DM = sum_k g * c_k * theta_k + a * theta_indep
```

where `theta_k` is mode k's pointing, `c_k` its configured optical coupling in
[0,1], `g` the common runtime coupling `optical_coupling` and `a` the runtime
`optical_amplitude`.
Acceleration and ADC counts still use every `theta_k` in full, so they are
bitwise unchanged by these settings. `optical.coupling` is empty (all 1), one
value for all modes, or 15 values: the 14 prominent modes in notebook order
(11, 14, 17, 20, 25, 28, 54, 56, 58, 75, 85, 120, 220 and 300 Hz), then one
value shared by the 200 weak modes. `theta_indep` is a second realization of
the same 214-mode population, drawn from its own generator seeded by
`optical.seed`, so the accelerometer does not see it. `sample::m_physicalPointing`
keeps the uncoupled mechanical sum sensed by the accelerometer, and
`sample::m_wfsPointing` applies the per-mode WFS couplings to the optical pointing
(each mode's `c_k * theta_k` plus its independent term). Couplings below 1 or a
nonzero amplitude make ACC/WFS coherence realistically less than 1; with
`optical.amplitude=1` the measured ACC/WFS magnitude-squared coherence at 120,
220 and 300 Hz falls from 0.61–0.83 to 0.33–0.46, about half (`[optical]` tests).
The baseline is below 1 because the weak modes are fully sensed but 1% WFS
coupled. The defaults (couplings 1, amplitude 0) reproduce the previous sequence
exactly. None of these settings change synchronization, DM calibration, or
`dm.scale`, which still scales the whole DM disturbance.

`optical.coupling` and `optical.seed` are startup-only. Two writable INDI numbers
adjust the rest at runtime, in the same `current`/`target` style as `dm_scale`:

- `optical_coupling` (`g`, in [0,1], initialized from `optical.commonCoupling`,
  default 1) **multiplies** every per-mode coupling `c_k`; it does not replace
  them. With `optical.coupling` set per mode, `g` scales that whole profile, so
  the relative coupling between modes is preserved. With a single configured
  value `c`, the effective coupling of every mode is `g * c`.
- `optical_amplitude` (`a`, in [0,100], initialized from `optical.amplitude`)
  scales the independent realization.

Out-of-range or nonfinite targets are rejected and keep the last accepted value.
The acquisition thread copies both values into the model before generating new
timeline nodes, so a change applies only to nodes generated afterwards: samples
already in the 5-ms lookahead keep their values, and the DM sees the change
within about 5 ms. No oscillator is reset, and the main generator's draws,
acceleration, ADC counts, 8-kHz acquisition timing and DM publication are
unchanged. The independent realization is allocated and stationary-initialized
at startup from its own generator, and advances only while `a > 0`. Setting
`a = 0` freezes it, and raising `a` later resumes from its frozen state, which
is still a valid stationary sample. With `g = 1` and `a = 0` restored, DM
pointing returns exactly to the uncontrolled sequence.

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
accumulation and interruptible sleeps. The realization advances on its fixed time
grid, so skipped ticks need no separate catch-up while the gap stays within the
5-ms generated horizon. Only the newest pair is published. If a stall leaves more
than **eight** ungenerated nodes before the current time, the model redraws
stationary histories once (a counted discontinuity) instead of performing
unbounded catch-up. OS wake-up lateness is counted separately from skipped
publication opportunities. There is no catch-up publication burst.

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
accelerometer deadlines count `skipped_publications`; the realization continues on
its time grid, with the bounded-catch-up restart described above.
Counter receipt is accounted immediately, even if a subsequent wait is canceled.
Duplicates cannot create extra samples. The code never retunes AR coefficients
from WFS jitter. Keep `fps.device` empty unless that subscription supplies the
requested **accelerometer** FPS; subscribing it to a 2-kHz WFS FPS still explicitly
requests a 2-kHz accelerometer rate.

For the 8-kHz / 2-kHz example, set `sim.fps=8000`,
`synchro.shmimName=camwfs_sync`, `synchro.referenceFps=2000`, and leave
`fps.device` empty. The free-running 6000-Hz defaults and timer path are unchanged.

Acquisition timestamps use `CLOCK_REALTIME` immediately after the monotonic
acquisition time at which the sample is evaluated. They represent the current acquisition event,
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
The writable `optical_coupling` and `optical_amplitude` numbers adjust the DM
optical disturbance at runtime; see "Optical coupling and independent optical
disturbance" above.

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
  requires a fresh reference, and the existing absolute/fractional thresholds must
  hold for both the phase error and the largest acquisition lateness over the
  current and previous WFS periods. It is evaluated the same way at every reference
  and acquisition, so a reference cannot re-assert lock over late samples, and one
  complete on-time WFS period restores it. Lock is not an independent measurement
  of hardware/WFS latency.
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
`clip_x_low`, `clip_x_high`, `clip_y_low`, `clip_y_high`. It also explains
`delay_lock`: `lateness_last_us`, `lateness_window_max_us` (the value the lock
uses), `lock_phase_ok`, `lock_timing_ok`, and the acquisition thread's
`acq_timer_slack_ns`, `acq_sched_policy` (1 is SCHED_FIFO) and `acq_priority`.

Synchronized acquisition reaches each deadline through a `sem_timedwait` timeout.
Futex timeouts carry the thread's timer slack, 50 us by default, and older kernels
apply it to SCHED_FIFO threads too, so every sample was taken about 50 us late
(`read_latency_error_us` about 52 us), just over the 50-us lock threshold. The
acquisition thread now sets its timer slack to 1 ns at startup;
`acq_timer_slack_ns` should read 1. Clipping counts actual
clamps after rounding in published measurements, not nodes generated ahead;
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

Rebuild on a provisioned workstation or development host. These commands only
build; they do not install, launch or deploy anything:

```sh
cd ~/source/MagAOX-DDSPC
make -C apps/accelSim clean
make -C apps/accelSim -j4
```

No `DDSPC_OPERATIONAL_BUILD` flag is needed for accelSim. Before a later operator
launch, retain the actual WFS reference and scheduler names and matching mask in
[acc000.conf](acc000.conf), with `dm.channelName=dm01disp10_aux`, `enabled=false`
and the desired startup scale (the supplied value is zero). The existing
scheduler advertises the slots; for the usual 2-kHz/four-slot setup these are
125 us apart. accelSim does not create that schedule.

Run the actual app's control and producer regressions without native streams or
hardware using the shared framework/API doubles:

```sh
make -f apps/accelSim/tests/Makefile.offline -j4 check
```

This executable includes the real accelSim header, real INDI properties and the
unchanged producer base. Its source/sidecar registry uses ordinary private files
under a fresh temporary directory, with a virtual monotonic clock shared by the
producer and the app's acquisition seam. It checks 8000 scaled target-time slots,
four-slot 2-kHz publication with `writeLeadUs=50`, finite-command rejection and
recovery, and exact, mismatched and discarded receipts with zero retry. It also
checks source ownership and attachment status, scheduler outage and reset, targets
outside the timeline, and removal after the app shutdown flag is set. acc000 values
are shown bitwise identical with DM output disabled, active or polled faster.
The `[timeline]` cases check legacy equivalence, reproducibility under acquisition
jitter, noise separation, the pointing/acceleration relation, spectral accuracy,
buffer edges, bounded catch-up and concurrent reads. It never connects to a native INDI service or
ImageStreamIO runtime. It requires Eigen, mxlib and CFITSIO development libraries;
unchanged INDI implementation files are built as C++14 for their legacy standard
library calls. This is protocol coverage, not a native Linux throughput test.

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

The shared-timeline tests need only the standard library:

```sh
c++ -std=c++20 -O3 -fno-fast-math -Wall -Wextra -DCATCH_CONFIG_MAIN \
  apps/accelSim/tests/accelSimTimeline_test.cpp -o /tmp/accelSim_timeline_test
/tmp/accelSim_timeline_test '[timeline]'
```

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

- `samples.csv`: nominal time [s]; physical (accelerometer), optical (DM) and
  WFS pointing X/Y [arcsec];
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
offline export/PSD comparison have passed. The offline app/producer/timeline suite
also passes under address/undefined-behavior and thread sanitizers. The normal app
build on macOS stops before app compilation in the framework precompiled header
(`generated/logTypes.hpp` missing). Native 8-kHz DM publication has not yet been
observed on the RTC, whose scheduler had no primary cadence at last report. No Linux stream, INDI runtime, or ten-minute performance result is
claimed. The real deployment configuration and raw-ADC DDSPC consumer are absent
from this checkout, so final downstream substitution still requires those inputs.
See the planning document for exact results and remaining coverage.

## Offline controller/DM cancellation validation

The accelFFCtrl [offline cancellation study](../accelFFCtrl/README.md#offline-feed-forward-dm-cancellation-2026-11-06)
uses the unchanged `accelSimModel.hpp` model to record isolated 28-Hz, 56-Hz and
combined modes, then feeds only raw UINT16 measurements into the production
controller numerics. Synthetic TT calibration, delayed DM response and residual
WFS calculations are separate offline test components; they do not invoke
`dmInjection`, open a DM stream or modify this application's disturbance,
synchronization, calibration or DM behavior. The independently labelled oracle
uses truth only as a perfect-prediction baseline.

For the documented 60-s noisy/quantized realization, gain 1 reduces physical
pointing variance by 94.9–95.3% (28 Hz), 88.8–91.5% (56 Hz), and 94.2–94.9% (both).
These are simulated results with an explicitly synthetic 300-Hz/ 1-ms DM plant
and unity WFS coupling. The study models source-noise and command holding and
preserves startup/reset/inhibition intervals. It is not an on-instrument test,
physical DM calibration, AO-loop interaction test or deployment authorization.
See the controller README for commands, PSDs, gain sweeps, fault cases and limits.

## Offline DM integration preparation

The [shadow integration report](../accelFFCtrl/offline/INTEGRATION.md) reviews the
existing simulator's DM-pupil projection, active-publication source and additive
channel conventions without invoking them. Locally archived TT FITS/mask copies
support a read-only conditional pupil calibration; they do not establish the
accelerometer/WFS registration or DM timing. Channel 10 remains the historically
assigned simulator disturbance channel, not a proposed shared correction channel.
The controller's shadow prototype uses host memory/files only. Simulator runtime,
RNG, synchronization, configuration and DM behavior are unchanged; no running
instrument was accessed or modified.

The subsequent [Phase 12 telemetry report](../accelFFCtrl/offline/RECORDED_TELEMETRY.md)
adds file-only passive calibration/timing analysis and replay through the existing
controller. No real simultaneous ADC/WFS recording was available. Its analytic
28/56-Hz fixtures are explicitly synthetic and separate from `accelSim`'s existing
disturbance model; no simulator generation, synchronization or configuration was
changed. Their forecast/simulated-suppression results are not instrument measurements.
