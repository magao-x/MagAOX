# Automatic collection observing metadata

The operator requested automatic ND readback from `fwtelsim` and recording of
`flipwfsf`. Work is local on `rlandman/dldata-observation-metadata`; no instrument
commands, deployment, or remote file access are permitted by AGENTS.md rule 0.
Pre-existing notebook/export changes in this checkout are outside this change.

Use the existing named-position interface: `fwtelsim.filterName` and
`flipwfsf.presetName`. An Idle or Ok switch property with exactly one selected
element is a settled reported position. The flipper must report `in` or `out`.
These are controller readbacks, not independent optical measurements.
Fresh fsm.state readback rejects disconnected/faulted controllers. READY and
OPERATING are accepted; POWEROFF requires a retained known named position, as
supported by the flipper's parked-position reporting.

Start refreshes both readbacks and waits at most five seconds. Read-only refresh
requests run once per second in appLogic, never in the image callback. Missing,
moving, ambiguous or stale positions prevent Start. During a run, reported changes
are latched; image acceptance/command publication stop and normal abort handling
preserves the partial run and releases the collector's owned channel.

Append `nd`, `flipwfsf`, property identifiers and local Unix readback receipt times
to the existing immutable `run.conf` context. No image or coefficient format,
distribution, normalization, channel, camera setting or output path changes.
Expose live readbacks as `observation.nd` and `observation.flipwfsf` in cursesINDI.
Directory labels no longer need to supply the authoritative ND metadata.

Verification covers actual INDI property parsing without hardware, invalid and
stale readbacks, immutable run metadata, and the existing capture/file regressions.
The full Linux app build and live INDI integration remain operator-run checks.

The follow-up request adds a dated default to the existing bank notebook updater.
`--dated-paths` converts a recognized banks/YYYYMMDD/name setting into visible
BANK_ROOT/BANK_DATE/BANK_NAME/BANK_DIR parameters, using date.today() on the
notebook computer. It preserves the recipe, exact calibration, raw-data path,
explicit overrides and collision checks, and creates a separate backup. It does
not execute the notebook, move banks or generate coefficients. Conversion is
opt-in so existing access-only callers keep their path-preserving behavior.

The optional home-folder training patch reads ND from run.conf, rejects conflicting
manual labels, and filters flipwfsf through the existing run settings rather than
changing per-sample array formats. Historical runs retain manual ND and unknown
flipper state. The patch is prepared locally; no home or instrument file is edited.

Verification completed on the workstation:

- 15 CPU C++ cases / 188 assertions pass with AddressSanitizer and UndefinedBehaviorSanitizer.
- The actual collector adapter, compiled with real INDI property types and stubbed
  base/hardware interfaces, passes fresh-start, change-latching, capture/publication
  guards, timeout and pending-Stop smoke checks. This is not a full Linux app build.
- All 14 Python cases pass with NN_DATA_PATH pointing to a patched offline copy,
  including automatic ND, conflicting-label rejection, flipper filtering and legacy
  fallback, plus dated paths, idempotence, custom overrides and targeted access setup.
- A copy of the operator's generator notebook has all code cells compiled; the
  exact calibration line, recipe, seed and overwrite assertions remain unchanged.
- clang-format 23.1.3 and git diff --check pass. The standard local build is missing
  generated/development headers; full Linux compilation and live INDI remain unverified.

Affected files: the app Makefile and dlDataCollection.hpp; new dlObservation.hpp;
dlDataCollection_test.cpp; update_bank_notebook.py and test_bank_notebook_access.py;
training_metadata.patch and test_training_metadata.py; README.md and QUICKSTART.md;
this execution record. Pre-existing nnReconstructor changes remain untouched.

## camwfs frame rate and E.M. gain

The follow-up request records `camwfs.fps.current` (Hz) and
`camwfs.emgain.current` (dimensionless gain factor). Reuse the existing read-only
polling, five-second Start deadline and latched change/partial-run handling.
Camera number properties must be Idle/Ok with finite positive current values;
targets are never used as readback. Parse numeric text with checked streams
because IndiElement::get<double>() does not check conversion failure.

Freeze `camwfs_fps`, `camwfs_emgain`, property identifiers and local Unix receipt
times in run.conf. Expose both values under the existing `observation` INDI
property. A reported change or loss of valid readback aborts the run rather than
silently mixing settings. This records camera-reported settings, not measured
saved-sample throughput. No per-frame arrays, training distribution, camera
commands, output paths or notebook changes are needed. Existing training run
settings dictionaries retain these extra run.conf keys without a format change.

Local verification: all 16 CPU C++ cases / 215 assertions pass with AddressSanitizer
and UndefinedBehaviorSanitizer. The collector adapter smoke check uses the actual
app methods with real INDI properties and fake hardware/base interfaces to check
camera/position changes, fresh-start, capture/publication guards, timeout and Stop.
Full Linux compilation and live readback behavior remain operator-run checks.
The user's earlier permission allows source-file transfer to their exao2 home
clone only; builds, installation, restarts and instrument commands remain excluded.

## Correct ND wheel property

The operator's readback shows the camera values and flipper position are valid,
but fwtelsim.presetName is absent and observation.nd is empty. Source inspection
confirms filterWheelCtrl sets m_presetNotation="filter", so stdMotionStage
publishes filterName rather than the generic stage presetName. Correct the
subscription and run.conf provenance label to fwtelsim.filterName; keep
flipwfsf.presetName. Start timeout errors now name only invalid readbacks.
The prior fake adapter smoke check used presetName for both devices and therefore
did not catch the incorrect wheel interface. Update that check to deliver the
actual distinct property names and verify the subscription keys and timeout text.
All 16 CPU cases / 216 assertions pass with ASan/UBSan; the updated adapter
smoke check also passes. These checks do not establish live instrument readiness.

## New-run command counter

The operator reports an immediate Start failure with one collector process.
Source inspection finds that the command counter is captured only on attachment,
so an external zero while idle is treated as an intervening run writer. At a new
Start, validate mapping/layout, reject an active write or any nonzero/nonfinite
command, check that the counter remains stable across the zero scan, then adopt
that counter. This preflight never writes to or zeros the channel. Counter
continuity checks remain strict during preparation, capture and between sets.
The operator's specific counter state remains unobserved; an active external
writer will still be rejected.

The actual-adapter smoke regression, with real INDI properties and stubbed
hardware/base interfaces, passes under ASan/UBSan for idle counter adoption,
strict subsequent counter changes, nonzero Start rejection without modifying the
channel, active-write rejection, observing changes, timeout and pending Stop.
No full Linux build or instrument operation is performed by the agent.
