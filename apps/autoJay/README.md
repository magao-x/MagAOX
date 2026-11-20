# autoJay

`autoJay` measures NCPC/tweeter registration on a shared camera image. It averages
an illuminated unpoked baseline, NCPC pokes, and tweeter pokes, removing one DM's
poke before applying the other. Each baseline-subtracted response is registered
to its own aligned reference. The reported residual is tweeter displacement
minus NCPC displacement, in detector pixels; aligned references define zero.
A separate `closedLoopIndi` instance commands `ttmperi`.

The operator establishes the optical setup and opens both `holoop` and
`twAlign-camwfs-ctrl`. The sensor checks these conditions; it does not move setup
stages, `ttmpupil`, or `stagecamlensx/y`. Camlens follow-up and deciding when the
alignment is complete remain manual.

## Configure an instance

Use [config/camflowfs.conf](config/camflowfs.conf) for sensor instance
`autoJay-camflowfs` and [config/camflowfs-ctrl.conf](config/camflowfs-ctrl.conf)
for controller instance `autoJay-camflowfs-ctrl`. The example filenames are not
deployment names: keep the actual instance names, controller input device, and
sensor controller setting consistent. These examples are not installed by this
change. The execution host, process list, INDI routes, and camera/DM shared-memory
routes must be configured by the operator.

Run the sensor on ICC with the camera and NCPC. Reserve `dm02disp02` for NCPC
and `dm01disp02` for tweeter exclusively for this sensor. Relay the tweeter
command from ICC to RTC. Both examples set `trigger=` in `[ncpc]` and
`[tweeter]`. AutoJay starts `acquisition.dmSleep` after both local writes
complete, then discards `settleFrames` distinct camera frames before averaging.
Set the delay to cover ICC-to-RTC delivery and both mirrors settling when
applying and removing pokes. The example delay is provisional. Confirm that
the relay delivers commands and zeros with the AO loop open. A local write
can succeed while the relay is offline.

To verify a contribution on the combined stream, set a DM's `trigger` to
`dm02disp` for NCPC or `dm01disp` for tweeter. Omitting the key retains these
defaults. A named trigger must be visible to AutoJay, and its settling delay
starts when the combined update is verified.

The app refuses nonzero owned channels at startup of a sequence. Both DM
combiners must process these active writes with the AO loop open. Use the
reserved channels from one alignment app or camera instance at a time.

In each of `[ncpc]` and `[tweeter]`, set paired comma-separated `pokeX` and
`pokeY` lists and one signed `pokeAmp` in DM command units, as in `dmPokeXCorr`.
Coordinates start at zero and are checked against that DM's stream dimensions.
Each listed actuator receives `pokeAmp`. The rest receive zero. Verify the
candidate coordinates in the flowfs example with the bump mask in and replace
the zero amplitudes with commissioned values before capture.

For a weighted pattern, use `pattern` (a finite, nonzero, full-DM-sized 2-D FITS
map) and a signed scalar `amplitude` in that DM section. Choose one input form
per DM. Capture saves the generated or supplied maps and their amplitudes in the
same calibration bundle.

For `camflowfs`, require `stagelosel.presetName.pupil` On and
`stageflowfs.position.current` at 32.0, initially with a 0.01 tolerance. The local
`stageflowfs` pupil preset is 33.0, so selecting that preset alone does not satisfy
this setup. Also capture `stagelosel.position.current` with a 0.01 tolerance.
Confirm these tolerances and add the actual bump-mask/filter requirements.

The six `setup` arrays have one entry per requirement in matching order. Types
are `0` for a number, `1` for text, and `2` for a switch; switch values are `0`/`1`.
`*` records a property's current value during capture and requires that saved
value during measurement. Numeric tolerances are absolute and use the property's
units. Missing, Busy, or Alert setup properties block acquisition. AutoJay
retains each subscribed value until it receives an update. `monitor.maxAge`
applies to camera frames only. Remove `monitor.refreshInterval` from any copied
configuration.

The local `camflowfs` mapping uses `pvcamCtrl`: the example captures
`exptime.current` and all four `readout_speed` switches (`sensitivity`, `speed`,
`dynamic_range`, `sub_electron`). PVCAM exposes no EM-gain property. Set a suitable
`quality.saturation` threshold for the chosen readout mode; its default is
effectively unbounded. Speed mode's 8-bit pixels are widened in the stream, so
stream datatype alone does not identify the saturation level.

[config/camsci1.conf](config/camsci1.conf) is intentionally incomplete and refuses
loading until its numeric TODOs are resolved. Supply the calibrated `fwscind`,
`stagebs`, and `stagesci1` settings, mask/filter requirements, and pokes. PICam
uses `exptime.current`, `emgain.current`, and the `readout_speed`/`vshift_speed`
selection switches named in the template. Confirm the deployed image orientation;
the local camera configuration uses `flipLR`. `autoJay.orientation` is a configured
identity tag, not a camera command or an automatic measurement of orientation.
Update it and take a separate calibration if stream orientation changes.

## Capture and inspect

Start with `control.enabled=false`. Reference capture and observational
measurements then need no controller acknowledgment. Accepted observational
deltas advance `measurement.counter` but carry `measurement.valid=0` and leave
`ready.toggle` Off; they cannot drive a guarded controller. Settled, current
`ttmperi` voltage readbacks are still required, and the sensor still applies
the configured DM pokes.

After manually establishing the desired aligned state and optical setup, request
`capture.request`. This uses the same three-state acquisition as a measurement,
checks signal quality without needing an older calibration, and saves/installs
one complete bundle. Capture never publishes a correction sample. Use
`measure.request` for one measurement, `continuous.toggle` for repeated measurements,
and `stop.request` to cancel. `status.current` explains progress/rejection;
`status.calibration` identifies the installed bundle.

The default accepted shift limit is 5 detector pixels, configurable with
`quality.maxShift`; it applies to both individual displacements and their
differential. Search-boundary peaks are also rejected. `quality` publishes each
DM's x/y displacement, correlation, and SNR. The SNR, overlap, competing-peak, and
visible-feature thresholds are provisional and require validation with the
selected patterns and mask. Partial grids are acceptable only when the remaining
signal supports a unique registration. An ambiguous original template cannot be
made acceptable by discarding absent features.

The two captured references share one ROI; all three current images share another
ROI. Reference and current ROI sizes/origins may differ. Their binning, orientation,
and optical pixel scale must agree. Physical-coordinate mapping preserves fractional
bin phases, and common ROI changes cancel from the differential. Camera geometry
must remain unchanged throughout a sequence. Rotation/magnification changes need
another calibration.

## Enable corrections after measuring the matrix

Measure the camera/profile-specific mapping from detector-pixel residuals to
the two `ttmperi` voltages with small known user-commanded moves. Verify its sign,
scale, and acceptable correction size. The controller computes
`target = current - gain * diag(gains) * matrix * residual`. Controller references
remain `0,0`, and the example global gain is `0.1`. Do not reuse another camera's
matrix or substitute an identity matrix.

All four `loop.intMatXX` entries are deliberately commented out in the controller
example. `input.guarded=true` requires explicit finite entries and a nonsingular
matrix, so the example refuses startup until calibrated values are supplied.
Leave `upstreamFollowClosed=false`. Enable the sensor's `control.enabled` only
after this controller is configured; changing that setting requires restarting
the sensor. Both correction axes must be enabled and provide settled readbacks.

With the controller open, start continuous sensing and wait for fresh accepted
samples. The first sample establishes the sensor session without commanding.
Then manually close the controller's `loop_state.toggle`; only a newer sample
completed after arming can command. The sensor waits for the matching `processed`
acknowledgment, voltage readback, and quiet interval before the next cycle.
Acknowledgment alone does not prove optical settling. Producer/controller clocks
must agree because guarded timestamps use epoch seconds.

Stop, capture, reload, invalid setup, rejected images, stale input, and a new
session invalidate control and require manual rearming. Recovery never
automatically closes the loop. Open the controller when satisfied with alignment,
stop sensing, and perform the required camlens follow-up. There is no automatic
convergence stopping criterion.

## Calibration, diagnostics, and recovery

Bundles are stored under
`<calibDir>/<sensor-instance>/<profile>/calibration__T<UTC timestamp>/` and contain
`calibration.conf`, both pattern maps, both references, and `unpoked.fits`.
The manifest records geometry, setup, amplitudes, timing, and integrity metadata.
Capture stages a complete bundle and publishes it atomically. Later edits to the
original source pattern files do not change an installed calibration.

Startup and `reload.request` select the newest complete valid bundle in the
configured profile, logging skipped corrupt candidates. Its camera, streams,
trigger settings, and current setup must still match before measurement.
Recapture after changing the relay setup or its settling delay. A setup mismatch
does not cause silent fallback to an older optical setup. Recapture after changing
the required setup or sampling; a bundle missing a newly configured setup binding
is invalid. ROI size/origin changes alone are supported.

`save_diagnostics.request` retains the next completed sequence. Rejected completed
sequences also save diagnostic FITS images under `autoJay.diagnosticsDir`, default
`<basePath>/rawimages/<sensor-instance>`. Suffixes `_0`, `_1`, and `_2` identify
baseline, NCPC, and tweeter means; reference images are retained when available.
Headers identify the reason, calibration, sequence, and frame counters. An
incomplete timed-out sequence may have no completed diagnostic product.

On a normal exit or detected failure the app attempts to clear its owned channels.
It cannot clear an unreachable stream after process death or a lost route. Before
restarting after such a fault, the operator must inspect channel ownership and
state, remove residual pokes through the established instrument procedure, and
restore valid streams/setup. Never clear a combined stream as a substitute.

## Development and offline tests

The app's main thread owns acquisition transitions, DM streams, calibration I/O,
and processing. The image monitor converts each camera frame into a private
float buffer and checks its counters before using it. Registration and
persistence run outside the frame/state lock. The acquisition, calibration,
and registration helpers can be tested without launching the application or
connecting to instrument resources.

Registration uses mxlib FFT cross-products and cubic interpolation of an
overlap-normalized correlation surface. Noise comes from frame-stack variance,
with baseline variance included. The test fixtures cover unequal ROIs, subpixel
shifts, partial and ambiguous patterns, first capture, and rejection paths.
Build individual suites with `tests/Makefile.one`, for example:

```sh
make -C tests -f Makefile.one t=../apps/autoJay/tests/autoJayRegistration_test.cpp
apps/autoJay/tests/autoJayRegistration_test
```

Use consistent coverage flags with `tests/testMain.o` when that object was built
with coverage enabled. Offline tests do not establish instrument latency, optical
accuracy, useful SNR thresholds, channel routes, or the correction matrix. Those
commissioning measurements must be performed by the user.
