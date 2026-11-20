# Task Description
<!-- The first section is filled out by the user 
     
     This is intended as a guide for how to prompt a coding agent to solve a problem.
     
     Make a copy of this template with a new name under agents/plans/<YYYY-MM>. Then
     Fill out each subsection below as needed. Feel free to add additional
     subsections, etc.  

     When complete, commit it on your feature branch and then prompt the agent to review this file.
-->

## Problem Statement
We want to automate the "J-test", which is the alignment of the pupil on `dmncpc`

## Discussion
<!-- Put more expository text here.  Fully describe the expected behavior and rationale. Include references to other parts of the software that serve as examples. Provide a link to an API/SDK manual.   -->
The manual J-test puts the a pattern on dmncpc which is then manually aligned using ttmperi to center it on an image of a pupil.

To automate this, what we can do is register `dmncpc` actuators to `dmtweeter` actuators after the tweeter-wfs auto-alignment procedure has completed.    Here we can poke the ncpc actuators in a pattern, measure their location on a camera (a slightly out-of-pupil image), and use that as the fiducial .  We then poke tweeter actuators and measure their location.  We then reference to a calibrated offset between those two positions and adjust ttmperi to move the tweeter actuators w.r.t. the ncpc actuators.  This will be a closed-loop process.

This will be done on camflowfs to enable doing this with bump mask in and minimizing coronagraph disturbances, but we also want to be able to do it on camsci1.  We'll want calibrations for both.  Here we will start with camflowfs.

Additional details:
- when we do this we are moving the image of the tweeter on the lower bench.  It will also move on camwfs.  We will need to follow this with stagecamlensx/y, though not necessarily automatically.  We do NOT move ttmpupil as part of this procedure.
- stagelosel needs to be in pupil
- stageflowfs will need to be at a calibration-specific point.  Start with 32.0
- there is no analog of the response matrix used to calculate a reference in dmPokeXCorr.  We will need to record a calibration image for each DM to use cross-correlation
- we will need to deal with the presence of the pupil.  Probably by subtracting an un-poked image as a dark. I think that dmPokeCenter has this in it.
- dmPokeXCorr is much more up to date than dmPokeCenter, so we should reference its conventions.
- the actual position of the pupil should not need to be fit (no circle fitting), we're just measuring the two actuator grids and moving the tweeter on the ncpc to a pre-recorded calibration offset
- So the sequence is:
  -- take unpoked pupil image
  -- take ncpc poked image
  -- subtract unpoked from ncpc
  -- cross-corr ncpc with its reference
  -- take tweeter poked image
  -- subtract unpoked from tweeter
  -- cross-corr tweeter with its reference
  -- difference the two
  -- publish different
  -- closedLoopIndi process sends correction to ttmperi

Poke patterns: we will need to create new poke patterns to optimize for use with the bump mask in, etc.  We should build in a reference taking mode that saves a reference image for a pattern when we have the system aligned.  Write to calib following other similar practices, enabling auto-recall, etc.  

Calibration: includes the poke patterns, relevant stage positions, references, and aligned offsets.

## Requirements
<!-- List the requirements that must be met.  A numbered list works best -->
- coordinate taking unpoked, ncpc-poke, and tweeter-poked images on a configurable camera
- process the images w.r.t. a reference for each and produce a delta signal
- check stagelosel and stageflowfs or fwscind/stagebs/stagesci1 for positions to match the calibration
- load the most recent calibration
- verify that calibration and setup match
- automatically select the poke patterns and references for a calibration
- reject measurements that are too large or are low SNR
- references and current camera ROIs should not need to match in size

## Tests and Metrics
<!-- List any useful metrics that should be targeted by tests -->
Should have tests of all calibration management (recording, recalling)

Should test for image processing edge cases (different sizes, pokes not completely in pupil such that only N out of M sensed)


## Scope and Caveats
<!-- Add any needed caveats and scope restrictions -->
For now creating of the patterns is not in scope, just the handling of them.

# Instructions to Agent
<!-- Specific instructions for the agent.  Below is our standard, but you can modify it as needed. -->

Analyze the above task and create a plan to implement a solution.  Document your findings below under "Agent Findings and Plan".  The comments under each heading provide guidance.  Keep this document up to date as you work.

Review AGENTS.md.  Do not alter any text above the "Agent Findings and Plan" below.  Do not begin implementation until the user has reviewed the plan and answered any questions.

# Agent Findings and Plan

Status: implemented and validated offline, 2026-10-31, on workstation `mx11` and
branch `jrmales/auto-jay`. The approved plan and the user's answers were committed
unchanged as `2f954d10` before implementation. The proposal below is retained as
the design record; the execution notes at the end document implementation choices
and validation. No instrument connections or commands were used. Local
`/opt/MagAOX/config` files were read only as examples, not as deployed-state evidence.

## Task Summary

Implement a two-DM registration sensor, provisionally named `autoJay`, beginning
with `camflowfs`. After the existing tweeter/WFS alignment, acquire an unpoked
image and separate NCPC and tweeter poke images, subtract the unpoked image,
register each result to its own recorded reference, and publish the change in
their relative registration. A separate `closedLoopIndi` instance will drive
`ttmperi` toward the calibrated alignment.

Reference capture will save a complete calibration containing both patterns,
both measured references, their aligned relationship, and the required camera
and optical setup. Loading a calibration selects its patterns and references
together. The image processing will support different reference/live ROI sizes
and origins without fitting the pupil or generating new poke patterns.

### Source findings and reuse

| Existing code | Finding and planned use |
| --- | --- |
| `apps/dmPokeXCorr/dmPokeXCorr.hpp` | Use its application, INDI, telemetry, diagnostic FITS, and cross-correlation conventions. Its reference comes from a zonal response matrix, and `analyzeSensor()` requires equal image dimensions; neither behavior meets this task unchanged. |
| `libMagAOX/app/dev/dmPokeWFS.hpp` | Provides the current single-DM acquisition conventions, combined-DM trigger, settling, measurement publication, and diagnostic metadata. Its sequence is alternating positive/negative pokes on one channel. It cannot implement the requested three-state acquisition unchanged. |
| `apps/dmPokeCenter/dmPokeCenter.hpp` | Records an unpoked pupil, but also takes a shutter-closed dark and uses positive/negative poke differencing. The new baseline must be an illuminated, unpoked image; do not copy its shutter or pupil-fitting procedure. |
| `apps/shmimIntegrator/shmimIntegrator.hpp` | Follow calibration placement under `m_calibDir/m_configName`, timestamped `__T` names, FITS I/O, newest-file discovery, and generation checks around asynchronous loading. Extend these ideas to complete bundles rather than independently recalling two images. |
| `libMagAOX/common/roiGeometry.hpp`, `libMagAOX/app/dev/stdCamera.hpp` | Reuse ROI coordinate conversion and the coherent applied `roi_region_state` property. Camera `state_string` is optional: PVCAM and CRED2 disable it. Explicit metadata is needed. |
| `mx/improc/imageXCorrFFT.hpp` in installed/local mxlib | Requires equal array dimensions, but supports masks and subpixel registration. The installed version has `mftLimitPeakSearch(true)`; `maxLag()` alone does not constrain its MFT coarse-peak search. Verify the dependency version before using that API. |
| `apps/closedLoopIndi/closedLoopIndi.hpp` | Accepts a two-component measurement and counter, then applies a configured 2-by-2 matrix. A changed counter triggers processing; it does not validate sample age, property state, or finite values. |
| `apps/flipperCtrl/flipperCtrl.hpp` | Provides checked temporary-write/rename persistence to follow for atomic calibration publication. |

The acquisition review also found that semaphore wakes can outnumber distinct
camera images, and existing wait loops can retry timeouts indefinitely. A
combined-DM stream event is not proof that a particular poke has reached the
optics. The new acquisition path needs explicit frame identity, bounded waits,
and cancellation/cleanup tests. The unresolved timing/sign investigation is
recorded in `agents/plans/2026-09/dm-poke-sign-synchronization.md`; this task must
not assume that investigation established the physical cause.

## Key Assumptions

1. The operator establishes the initial alignment and optical configuration.
   This app verifies stage positions; it does not move setup stages, `ttmpupil`,
   or `stagecamlensx/y`. Camlens follow-up remains an operator task initially.
2. `camflowfs` is the first commissioned profile. Camera selection and calibration
   metadata will also support a separate `camsci1` profile; its optical positions
   and control matrix must be supplied before commissioning it.
3. Each DM has an exclusively reserved additive poke channel. Zero in those
   channels means no additional poke, while existing flats and other channels
   remain active. Channel names, ownership, and transport are not yet confirmed.
4. Patterns will be supplied as finite two-dimensional FITS command maps with
   the corresponding DM dimensions and a documented scalar amplitude per DM.
   Save the maps and amplitudes in each calibration. Pattern generation and
   optimization are outside this task.
5. The first version supports ROI size/origin changes with unchanged binning,
   orientation, and pixel scale. Changes in binning or optical magnification
   require a separate calibration; uncalibrated resampling is excluded.
   **Confirmed by the user during plan review:** the NCPC and tweeter images
   share one live ROI, and both reference images share one capture ROI. Only
   the live ROI versus the reference-capture ROI may differ. The unpoked
   baseline also uses that acquisition's ROI; any within-sequence ROI change
   invalidates the whole sequence. Store one geometry per reference pair and
   compute one shared reference-to-live mapping for both correlators.
6. Reference capture defines the current aligned relationship as the desired
   relationship. Any independently specified adjustment to that zero point
   must be explicit, rather than silently changing a reference or controller
   setpoint.
7. Start idle, with corrections disabled. Calibration capture, reload, camera
   changes, invalid setup, and measurement failures require an open correction
   loop. Recovery does not automatically close it.

## Requirements

1. **Acquisition:** snapshot one calibration/setup generation; clear both owned
   poke channels and confirm/settle their removal before averaging an unpoked
   baseline; apply only the NCPC pattern, settle and average; clear NCPC and
   confirm/settle its removal before applying the tweeter pattern; settle and
   average tweeter, then clear it. Correlate the two baseline-subtracted images.
   Ordered writes to independent DM pipelines alone do not establish this
   ordering at the camera. No phase may contain both patterns.
2. **Frame integrity:** copy image data and its source counter/timestamp as one
   accepted sample, count each source frame at most once, discard configured
   settling frames, and bound the whole phase wait. Abort on timeout, stream
   recreation, camera reconfiguration, stop, shutdown, or a setup/calibration
   generation change. Keep slow processing and disk I/O outside image locks.
3. **Cleanup:** all normal, cancelled, failed, and exception exits clear both
   owned additive channels. Validate a channel before writing; never clear a
   combined stream or another writer's channel. Join workers before releasing
   stream/semaphore resources. A failed cleanup is an explicit fault.
4. **Setup validity:** require current, connected, settled device information.
   For `camflowfs`, check `stagelosel` in pupil and the calibrated numeric
   `stageflowfs` position, initially 32.0. Local configuration calls 33.0 the
   `pupil` preset, so that preset alone is insufficient. For `camsci1`, check the
   calibrated `fwscind`, `stagebs`, and `stagesci1` settings. Record any additional
   mask/filter requirements in the selected profile. Recheck before every
   phase and before publication.
5. **Calibration:** capture/load patterns, references, geometry, setup, and
   aligned offset as one versioned unit. Automatically recall the newest
   complete valid bundle for the selected camera/profile, then verify its
   compatibility with the current setup. Report the selected ID and mismatches.
   Do not silently select an older optical setup just because its stages match.
6. **Registration:** use one mapping from the shared reference ROI to the shared
   live ROI, with known detector coordinates and valid-pixel masks. Build
   equal-sized working arrays for mxlib without treating padding as data.
   Require both references to have their declared common geometry. Permit
   partial pattern visibility when enough unambiguous signal remains.
7. **Acceptance:** independently validate both DM registrations and their
   differential. Require finite data, nonzero usable signal, sufficient valid
   overlap, adequate poke SNR, acceptable normalized correlation and peak
   uniqueness, and bounded individual/differential displacement. Reject peaks
   at the search boundary, sign-inverted matches, and ambiguous grid aliases.
8. **Output:** publish one complete `measurement` vector containing
   `delta_x`, `delta_y`, and a new `counter` only for an accepted pair. Publish
   quality/status separately. No successful counter advance for reference
   capture, partial acquisition, or rejection.
9. **Control:** `closedLoopIndi` owns `ttmperi` commands. Its reference values
   are zero because the sensor applies the calibration offset. Confirm the
   camera-specific matrix and signs; do not copy another camera's matrix.
   Hold `ttmperi` stable throughout a baseline/NCPC/tweeter sequence and wait
   for the preceding correction to settle before the next baseline.
   Exercise startup/reconnect/replay behavior before declaring the interface
   safe for closed-loop use.
10. **Observability:** expose calibration identity, setup mismatch, acquisition
    state, per-DM displacement and quality, final residual, and rejection reason.
    Preserve useful FITS diagnostics and measurement/frame/DM counters so a bad
    measurement can be reproduced offline.

## Questions and Points of Clarification

These questions should be resolved at plan review; proposed defaults are given
where they permit a concrete design.

1. **Operating sequence and optics:** confirm setup checking only, with manual
   camlens follow-up. What position tolerances and mask/filter settings must
   match? Is the main AO loop open or closed during these pokes, and should the
   tweeter/WFS alignment loop be paused while `ttmperi` corrections run? Its
   state can affect the measured reference response.

ANSWER: AO loop needs to be off.  tweeter/WFS alignment loop off as well.

2. **DM channels and execution host:** which dedicated NCPC/tweeter channels
   should this app own, and where should it run? Local examples use
   `dm02disp06` and `dm01disp06` for existing alignment apps; these are not
   automatically available. Confirm access to the camera, both poke channels,
   and both combined trigger streams, including any cross-host relays.

ANSWER: use dm0Xdisp02

3. **Pattern and ROI contract:** are full DM-sized FITS maps plus per-DM scalar
   amplitudes acceptable? Is supporting different ROI sizes/origins at the same
   binning sufficient for the first version? Proposed answer: yes to both.

ANSWER: yes to both. 

4. **Calibration selection and zero:** should recall mean newest valid bundle
   within an explicitly selected profile, followed by setup verification, as
   proposed? Does recording an aligned pair define zero residual, or is there
   an externally measured nonzero offset that must be entered separately?

ANSWER: Yes.  Yes let's start with assumption that the aligned references define zero without an additional offset. 

5. **Controller calibration and acceptance targets:** is there a measured
   `camflowfs` residual-to-`ttmperi` matrix? What alignment precision, capture
   range, cycle time, and permissible correction per cycle are required? These
   determine averaging, lag bounds, gains, and final acceptance thresholds.

ANSWER: No, it has not been measured yet.  Start with 5 pixels as the largest shift we will report as valid, but make it configurable.  Anything larger should not be published.  For now it's fine to let the operator decide when to open the loop, not defining a convergence tolerance for automated stopping.  Assume gain=0.1 to start.

No hardware interrogation is needed to begin the offline implementation after
review. Unknown deployment values can remain explicitly required configuration;
instrument measurements and commissioning must be performed by the user.

## Tests

All automated tests use synthetic images, temporary calibration directories,
fake DM/frame interfaces, and captured INDI commands. They must not connect to
instrument streams or devices.

| Area | Cases and expected outcome |
| --- | --- |
| Calibration capture and round trip | Capture the first calibration from supplied patterns and setup with no existing references. Save and reload both maps/amplitudes, references, shared ROI, anchors/offset, setup, and acquisition metadata without changing their association. Reload remains independent of edits to the original pattern files. |
| Recall policy | Multiple timestamps/profiles/cameras; no calibration; corrupt newest bundle; unsupported schema; abandoned staging directory. Select the newest valid complete candidate in the requested profile, report skipped candidates, and block use on setup mismatch. |
| Persistence failures | Inject failure during each component write/read, final validation, and rename. No partial bundle becomes selectable; no partial in-memory swap; previous calibration remains intact. |
| Compatibility and races | Missing/partial/disconnected properties, stage motion and tolerance boundaries, wrong mask, camera orientation/binning changes, and changes during capture/load/processing. Stale generations cannot publish or install. |
| Acquisition order | Assert baseline/NCPC/tweeter command order, acknowledgment/settling of poke removal across independent DM pipelines, requested distinct-frame counts, reuse of the same baseline/ROI, and no simultaneous pokes. Duplicate wakeups cannot duplicate a frame; counters resetting/reallocated streams abort. |
| Cleanup and deadlines | Stop, shutdown, timeout, exception, and stream loss in every phase clear accessible owned channels, report failed cleanup, publish no sample, and terminate without hanging. Unrelated DM channels remain unchanged. |
| Coordinate/sign convention | Synthetic independent shifts, common-mode shifts, zero residual, and deliberate calibrated offset. Common motion cancels; differential sign matches the controller contract. Test a shared live ROI differing from a shared reference ROI in size/origin, rectangular arrays, odd/even extents, and supported nonunit binning. Reject unequal reference-pair geometry and ROI changes within an acquisition. |
| Registration accuracy | Fixed synthetic patterns with known integer and subpixel translations across the accepted lag range. Proposed initial numerical target: no more than 0.1 detector pixel error in noiseless well-supported fixtures; operational accuracy and speed remain commissioning targets. |
| Partial patterns | Remove or obscure N of M features in either DM image, including clipping at ROI/pupil edges. Recover the known shift when sufficient distinctive support remains; reject insufficient or periodic/ambiguous support. No all-pokes-visible requirement. |
| Quality rejection | Baseline-only/no response, zero variance, low SNR, saturation, nonfinite pixels, inverted response, competing peaks, search-edge peaks, and large individual/differential shifts. No accepted output; retain a specific reason. Use finite checks that remain valid under project `-ffast-math` flags. |
| Controller protocol | One valid pair produces one complete update and at most one correction. Rejection/capture never produces a correction. Test zero residual, matrix orientation, target endpoints, loop-open/unready states, duplicate counters, restart/reconnect/retained input, and both readiness/measurement message orders. Delay correction acknowledgment/readback and verify the next baseline waits; unexpected motion during a sequence invalidates it. |

Build new suites through `tests/Makefile.one`, list them in `tests/tests.list`,
and group application tests under `application_unit_test` in `tests/groups.dox`.
Use per-case Doxygen blocks, `libXWCTest::autoJayTest`, and explicit API references
through `tests/testXWC.hpp` when harness indirection prevents normal links. Run
the existing `dmPokeXCorr` and `closedLoopIndi` suites for any shared changes.

Commissioning is a separate user-run validation: capture aligned references,
compare measured shifts with small known `ttmperi` moves, establish matrix sign
and scale, check repeatability/SNR with the bump mask, then close at low gain.
Record results for each camera profile and verify the subsequent camlens step.

## Implementation Plan

### 1. Establish interfaces and isolate reusable components

Create `apps/autoJay/` following the existing header-only application pattern:
class declarations and out-of-class definitions in `.hpp`, entrypoint in `.cpp`.
Use `dev::shmimMonitor`, `dev::telemeter`, `mx::improc::milkImage`, mxlib FITS and
correlation utilities, and the corresponding interface macros where provided.

Keep calibration/compatibility and registration calculations independently
testable. Use a small composable acquisition helper with explicit per-DM channel
contexts, frame identity, deadlines, and transaction cleanup. The app owns the
three-phase sequence. Do not force dual-DM behavior into the existing single-DM
`dmPokeWFS` inheritance hierarchy.

There is a useful future common utility here: frame-stack acquisition and DM
channel cleanup are needed by both sensors. Extract only a narrowly reusable
primitive if it can be shared with regression coverage; otherwise keep the new
helper local and record promotion to `libMagAOX/app/dev` as follow-up. Avoid a
broad rewrite of existing alignment apps as a prerequisite.

### 2. Implement calibration bundles and setup snapshots

Proposed layout beneath `m_calibDir/m_configName/<profile>/`:

```text
calibration__T<timestamp>/
    calibration.conf
    ncpc_pattern.fits
    tweeter_pattern.fits
    ncpc_reference.fits
    tweeter_reference.fits
    unpoked.fits
```

The versioned manifest records camera/stream identity; one shared reference ROI, binning,
coordinate units and orientation; relevant camera mode/exposure/gain; optical
profile and required positions/tolerances; both DM identities/dimensions and
pattern amplitudes; settling/averaging settings; processing version; reference
anchors and aligned offset; timestamp and provenance. Store masks as additional
FITS components when needed. Keep paths relative and validate every component.

Write a hidden sibling staging directory, check/close all files, validate the
complete bundle, then atomically rename it on the same filesystem. Recall ignores
staging directories. Load into a temporary immutable object and install only
after all checks pass and its setup/configuration generation is still current.
Never replace only one DM reference. Automatic recall and explicit selection use
the same validator. A retained older active object cannot enable measurement
unless it still matches the selected setup.

Subscribe to typed camera/stage state, including coherent applied ROI geometry.
Preserve ROI origin/extent as spatial metadata rather than including them in an
exact setup key that would forbid the requested size mismatch. Require matching
binning/orientation and sufficient usable image coverage. Invalidate snapshots
on connection loss, reconfiguration, or motion; a selected preset can be Busy
while its stage is still moving.

### 3. Implement bounded three-state acquisition and reference capture

Use one worker and an explicit state sequence: idle, validating/settling,
baseline, NCPC, tweeter, cleanup, processing, then publish or save. Repeated
measurement returns through correction settling, validation, and a fresh
baseline for every pair. Capture uses the same acquisition code but never
publishes control samples. It must work from supplied patterns and setup
metadata without an existing calibration. Validate capture using signal/noise,
saturation, usable support, template ambiguity, and repeatability checks rather
than requiring correlation against an old reference. The operator-established
alignment defines the desired relationship. Install/save only if the complete
capture and setup snapshot remain valid.

Accumulate frames in the monitor callback or a bounded frame queue, preserving
data/counter/timestamp identity. Do not implement a count of semaphore wakes
against one mutable latest-image buffer. Snapshot all amplitudes and timing
parameters at the start. Wait for combined-DM updates and configured settling
with deadlines; verify commanded contribution where possible and record the
evidence. Physical response latency still needs operator validation.

Use a single cleanup path/guard for both owned channels, including exceptions.
Reject nonzero/unavailable channels at startup rather than claiming another
writer's commands. On stream recreation, invalidate the acquisition and recheck
ownership/geometry before any subsequent writes. Block capture/reload while a
transaction is active, or cancel and join it before changing the active object.

### 4. Implement registration, offset accounting, and quality gates

For each DM, subtract the same averaged illuminated baseline. Build one mapping
between the reference pair's shared ROI and the live sequence's shared ROI,
then use it for both DMs. Map to a common grid with known detector origin,
unchanged sampling, and separate validity masks. Pad/crop for equal FFT array
dimensions without geometrically stretching either image. ROI-origin changes
are common to both DM measurements and must not appear as differential motion;
retain the coordinate mapping for overlap and individual-displacement checks.
Prevent circular-wrap matches with suitable padding/support and a bounded
physical displacement search.

Let `q_N` and `q_T` be fixed reference anchors in detector coordinates, and
`s_N`, `s_T` the measured displacements from those templates. Record the aligned
relationship `C = q_T - q_N`. Then:

```text
p_N = q_N + s_N
p_T = q_T + s_T
delta = (p_T - p_N) - C = s_T - s_N
```

The anchors need no pupil or actuator-center fit; they establish consistent
template coordinates. Thus two aligned capture images produce zero residual
even if their grids occupy different absolute positions. Do not subtract the
physical offset a second time from `s_T - s_N`. Express output and limits in
detector pixels, converting from sampled-pixel shifts using the recorded binning.

Use two mxlib correlators with subpixel peak estimation. Check availability of
the bounded MFT search API, or provide a tested bounded-search alternative.
Evaluate normalized correlation on actual shifted valid overlap; the existing
`peak / full-array-pixel-count` rule is unsuitable once masks/padding differ.
Require minimum supported reference signal, exclude a main-peak neighborhood
when testing competing peaks, and estimate poke SNR from stack/background noise.
Correlation coefficient alone is not a signal-to-noise measurement. Validate
both DM results before forming/accepting the differential.

### 5. Integrate publication and the correction loop

Publish `measurement.delta_x/delta_y/counter` atomically as one property update
and use a dedicated readiness toggle to open `closedLoopIndi` on stop, capture,
invalid setup, or failure. Set `upstreamFollowClosed=false`; the operator closes
the loop after readiness is restored. Publish diagnostics without advancing the
accepted measurement counter.

Define a bounded between-cycle acknowledgment/settling contract: after publishing
a sample in closed-loop operation, wait until its correction (or explicit
no-command outcome) has been processed, the commanded voltages are reported,
and the configured settling interval has elapsed. An unchanged voltage for a
zero correction must not cause an indefinite wait. Observe `ttmperi` command
and current values throughout acquisition; an unexpected change invalidates the
sequence. `OPERATING` alone is not evidence of settled motion. Establish this
contract in offline integration tests, adding optional processed-counter
reporting to `closedLoopIndi` if its current interface cannot supply the needed
acknowledgment.

The existing controller is sufficient for ordinary accepted-counter processing,
but separate readiness and measurement messages are not an atomic validity
contract. Add explicit startup/reconnect/replay tests first. If retained or
late samples can command the actuator, include a narrow opt-in controller input
guard covering validity, session/freshness, and rearming; preserve existing
instances' behavior and test it. Do not claim a separate Off toggle eliminates
that race, or leave failed protocol tests as an implicit deployment assumption.

Provide example sensor/controller configs with zero controller references and
the established `ttmperi.axis1_voltage` / `axis2_voltage` targets. Require an
explicit measured control matrix; local `ncpcAlign-pupil-ctrl.conf` establishes
the endpoint convention only. Keep correction disabled until its sign, scale,
gain, and acceptable step size have been checked for this profile.

### 6. Build integration, documentation, and commissioning handoff

Expected implementation files:

- New `apps/autoJay/autoJay.hpp`, `autoJay.cpp`, `Makefile`, focused helper headers,
  tests, and example configuration files.
- Top-level `Makefile` build lists after the execution-host decision.
- `tests/tests.list` and `tests/groups.dox`; `tests/Makefile.one` only if additional
  test linkage is actually needed.
- `apps/closedLoopIndi/closedLoopIndi.hpp` and its tests only for the protocol guard
  or correction acknowledgment described above, if required by integration tests.
- Shared helper/library files only for a justified extraction with regression
  tests; telemetry schema/registration files if richer binary records are needed
  beyond the existing `telem_pokeloop` summary.
- This plan as the engineering record. Prepare companion deployment configuration
  changes for process lists, INDI subscriptions, reserved channels, and stream
  routes once the host is known; do not edit/install operational `/opt` files as
  part of offline development.

After implementation, build the app and focused suites, run the offline tests,
complete the full-file Doxygen pass, run `clang-format` on touched C++ files, and
check the diff. Keep functional changes and this plan together, follow with
documentation-only changes, and put any formatting-only cleanup last. All commits
belong on the namespaced feature branch. No commit or PR is part of this planning
step.

## Follow-up and Edge Cases

- **Required before commissioning:** resolve channel ownership/transport, camera
  and mask profile, stage tolerances, loop sequencing, control matrix, numerical
  thresholds, and timing on the real system. The agent will not run those tests.
- **Sequential-image drift:** source/seeing/AO changes between baseline and the
  two pokes can bias subtraction. Bound cycle time and detect unusable stacks;
  consider interleaved baselines or signed pokes only if data show the requested
  three-state sequence is inadequate.
- **Partial grids:** fewer visible pokes can be acceptable, but a periodic or
  symmetric remaining subset can make translation ambiguous even at high SNR.
  Pattern design may need to improve identifiability; it remains outside scope.
- **Calibration validity:** translation-only registration does not correct
  rotation, magnification, changed binning, or unmodelled optical configuration.
  These require another calibration or a separately reviewed model extension.
- **Camera metadata timing:** INDI setup and shared-memory frames are separate
  streams. Abort on geometry changes and require settled matching metadata/frame
  dimensions; do not describe a source-counter snapshot as an acknowledgment of
  physical camera or DM response.
- **Process death or disconnected DM:** a cleanup guard cannot clear a channel
  after uncatchable process death or loss of access. Document an operator recovery
  procedure and avoid automatic reuse until channel state is verified.
- **Later work:** commission `camsci1`, optimize patterns, automate camlens
  following if desired, and promote acquisition/persistence helpers to common
  library utilities when a second consumer justifies the extraction.

Initial planning validation (2026-10-30): reviewed the relevant acquisition,
correlation, ROI, calibration, controller, and test/build sources. Only this
Markdown file changed at that checkpoint; no application build or test run was
needed for the planning edit.

## Execution Record — 2026-10-31

### Implemented behavior

- Added the header-only `autoJay` application and independently testable acquisition,
  calibration, and registration helpers. The main application thread owns DM
  transactions and processing; the existing image-monitor thread accumulates
  coherent, distinct frames. A separate acquisition worker was unnecessary.
- Enforced the answered operating conditions: both `holoop` and the tweeter/WFS
  alignment loop must be open. Default owned channels are `dm02disp02` and
  `dm01disp02`. Writes are active because AO is off and cannot trigger passive
  channel updates. The app checks combined-command changes at affected pixels,
  waits for settling, and explicitly removes NCPC before applying tweeter.
- Frame counts, command/frame deadlines, source counter regressions, individual
  raw-frame saturation, stream geometry/identity, channel ownership, and setup
  changes are checked. Stops and exceptions clear accessible owned channels.
  File replacement or another writer causes a fault instead of a write to a
  different channel. Stream-close errors are reported during normal cleanup.
- Implemented first-reference capture, immutable versioned bundles, strict FITS
  dimensions/data validation, per-file integrity checks, atomic directory
  publication, and newest-valid selection within the explicit profile. Failed
  recall removes the previous calibration's eligibility. Loaded patterns and
  references are always associated as one object, independent of later edits to
  source pattern files.
- Kept one shared ROI for each reference pair and one for each live triplet.
  Mapping supports unequal extents/origins, fractional bin phases, and anisotropic
  binning at unchanged sampling. Reference images themselves encode the aligned
  relationship; the published residual is `tweeter shift - NCPC shift`, with no
  additional offset, as answered by the user.
- Registration uses mxlib FFT cross-products followed by overlap-specific normalized
  correlation and mxlib cubic interpolation. Testing demonstrated that selecting
  the original independently normalized FFT/MFT peak and only checking its quality
  afterward could bias cropped/missing-poke measurements. This implementation does
  not require the newer mxlib bounded-MFT API. A single bounded absent-feature
  refinement is allowed only after the original template passes the ambiguity
  check; a reproducer verified that masking must not turn an ambiguous match into
  an accepted one.
- Default displacement limits are 5 detector pixels per component for each DM and
  the differential, configurable through `quality.maxShift`. Search-edge peaks
  are rejected too. SNR comes from measured frame-stack variance, including the
  shared baseline variance; noise-only first captures do not pass self-correlation
  alone. The operator decides convergence and when to open the correction loop.
- Setup rules are generic typed INDI bindings with numeric tolerances or exact
  text/switch matching. Flowfs prerequisites are retained when extra mask/filter
  requirements are configured: `stagelosel` pupil selection and captured numeric
  position, plus `stageflowfs` numeric position (initially 32.0). Science-camera
  profiles must supply `fwscind`, `stagebs`, and `stagesci1` bindings. Loaded
  bundles must contain all configured setup bindings. Camera exposure is captured;
  examples also capture the applicable readout/gain settings.

### Control integration and deliberate choices

- Added opt-in `input.guarded` to `closedLoopIndi`. Existing configurations retain
  the legacy input contract. Guarded measurements include validity, process
  session, counter, and completion time; stale, malformed, replayed, nonfinite,
  or old-session input cannot command. New sessions and faults require manual
  arming, and only newer post-arm samples may command. The controller reports a
  `processed` acknowledgment with the sample identity, outcome, and target voltages.
- The sensor waits for a matching acknowledgment, voltage readback, and a quiet
  interval before the next transaction. Both correction axes must be enabled in
  guarded operation. The default readback tolerance is 0.005 V, allowing one
  150/32767 V controller quantization step. An acknowledgment is not a measurement
  of optical settling; the configured timing still needs commissioning.
- Guarded controller configuration requires all four explicit finite matrix
  entries and a nonsingular matrix. The user has not measured this matrix, so
  `control.enabled=false` is the sensor default: capture and observational
  measurements work without a correction controller. Observational measurements
  advance the accepted counter but publish `valid=0` and readiness Off, preventing
  a guarded controller from using them. Enable corrections only after supplying
  the measured matrix. The example gain is 0.1 and controller references are zero.
- Saved reference pixels describe the setup at acquisition, not at completion of
  disk I/O. If setup changes while a validated historical capture is being saved,
  the complete bundle may remain on disk, but it is not installed and the status
  explicitly says so. Installation and subsequent use require current validation.
- Rotation, magnification, binning changes, pattern creation, automatic camlens
  following, and instrument deployment remain outside this implementation.
  `autoJay.orientation` is an explicit profile identity; it must reflect the
  deployed stream orientation. No camera or setup-stage motion is commanded.

### Offline validation

| Suite | Cases | Assertions | Result |
| --- | ---: | ---: | --- |
| autoJay acquisition | 3 | 107 | Passed |
| autoJay calibration | 9 | 426 | Passed |
| autoJay registration | 9 | 105 | Passed |
| autoJay application harness | 13 | 1127 | Passed |
| closedLoopIndi guarded protocol | 7 | 2955 | Passed |
| closedLoopIndi existing callbacks | 1 | 33 | Passed |
| dmPokeXCorr existing regression | 3 | 13 | Passed |
| **Total** | **45** | **4766** | **Passed** |

New and changed suites were built with the normal `-O3 -ffast-math` settings.
The existing `tests/testMain.o` and local libraries contain coverage instrumentation,
so test links used the coverage runtime (`TESTLIBS=-lgcov` or `--coverage`). Both
applications build at `-O3 -ffast-math` with `EXTRA_LDFLAGS=--coverage` for those
same local libraries. An initial parallel app build collided when both makefiles
removed their shared generated Git-version header; a serial build passed. No
build-system changes were made for that unrelated behavior.

Tests use synthetic images, temporary files, fake DM interfaces, and captured
INDI commands; none start the sensor or connect to instrument resources. The
calibration suite injects component write, validation, and rename failures and
checks corrupt/truncated images and mixed bundles. Registration fixtures test
subpixel shifts, unequal ROIs, missing/clipped features, ambiguous grids, and
quality rejection. Protocol tests capture actual callback-generated commands,
including replay, restart, invalid input, readiness, and partial send failures.
Full-file Doxygen checks and `clang-format` were applied to touched C++ files;
the required literal `{ //mutex scope` annotation is retained. `git diff --check`
passes for the implementation.

### Affected files and handoff

- `apps/autoJay/{autoJay.hpp,autoJay.cpp,Makefile,autoJayTypes.hpp,autoJayAcquisition.hpp,
  autoJayCalibration.hpp,autoJayRegistration.hpp}`.
- `apps/autoJay/tests/{autoJayAcquisition_test.cpp,autoJayCalibration_test.cpp,
  autoJayRegistration_test.cpp,autoJay_test.cpp}`.
- `apps/autoJay/config/{camflowfs.conf,camflowfs-ctrl.conf,camsci1.conf}` and
  `apps/autoJay/README.md`.
- `apps/closedLoopIndi/closedLoopIndi.hpp` and its existing/new protocol tests.
- Top-level `.gitignore` and `Makefile`, `tests/tests.list`, `tests/groups.dox`, and this plan.

The app is included in RTC/ICC build lists; that does not select or configure its
execution host. Examples are not installed. Before using hardware, the operator
must supply patterns/amplitudes, confirm exclusive channel ownership and stream
routes, resolve mask/stage profile settings and tolerances, measure the control
matrix, and validate physical settling and image-quality thresholds. The science
camera example intentionally has unresolved optical settings. Existing unrelated
`tests/tmp.drawio` and `tests/xlog/` content is left untouched.

### Follow-up: actuator list input

For commissioning, use the same `pokeX`, `pokeY`, and `pokeAmp` convention as
`dmPokeXCorr` in each DM section. The operator chooses the coordinates and
amplitudes. AutoJay validates the coordinate lists, checks bounds against the
actual DM streams, and expands them to full command maps at reference capture.
The existing FITS input supports weighted patterns. Both input forms produce the
same calibration bundle format, and recall uses the saved maps and amplitudes.
The flowfs example uses candidate lists from the local NCPC bump-alignment and
tweeter WFS alignment configs. Verify those coordinates in the intended optical
setup and select amplitudes before capture.

On 2026-11-19, the optimized app build and the offline application suite passed
(15 cases, 1,387 assertions). The suite covers capture and bundle reload with
either pattern source, malformed input, and coordinates outside the DM stream.
