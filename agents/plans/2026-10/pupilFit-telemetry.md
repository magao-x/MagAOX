# pupilFit per-fit telemetry

Issue: https://github.com/magao-x/MagAOX/issues/104

Request: inspect pupilFit telemetry, record every fit for every pupil, and reconstruct
averages when reading logs instead of storing redundant averages. The user approved
the audit recommendations and requested this plan followed by implementation.

## Design

- Add `telem_pupilfit`: one record for every processed input image, including repeated
  identical fits. Store the input frame counter and acquisition timestamp, reference
  versus live-image identification, validity, and a compact vector of pupil structs
  containing x, y, diameter, median intensity, and background. Vector order follows
  the existing quadrant numbering. Only the three or four active pupils are stored.
- Do not store across-pupil averages. Provide derived accessors and human-readable
  output through the log type, including averages in ordinary logdump output. Add
  derived averages to this type's JSON formatter without adding serialized fields.
- Add `telem_pupilfit_state`: active setpoints, requested/effective reference mode,
  threshold, pupil/background quantile indices, image dimensions, and averaging
  state/window identifier. Record at initialization, changes, and periodic intervals.
  State changes and fit records are serialized under the app mutex so the preceding
  state record applies to each fit. A new averaging window identifies resets even
  when averaging was already enabled. Means and variances over time remain derived.
- Emit fit telemetry directly from processing, never from periodic callbacks. Record
  failed attempts explicitly and suppress publication/accumulation of failed results.
  Validate finite fit results and usable pupil contrast, and propagate fitter errors.
- Use the existing telemetry, FlatBuffers generation, metadata, and logger dispatch
  infrastructure. Retain frame-grabber timing telemetry and use TELEMETER macros.
- Correct three-pupil averages/offsets to use only active pupils and exact division
  by their count. Preserve the established header-only app convention.
- Source stream metadata is sampled when processing begins; the existing shared
  memory monitor does not provide an atomic image/metadata snapshot. This change
  records every fit performed, not every upstream camera exposure.

## Execution

1. Create the namespaced feature branch and document the plan (this file).
2. Implement the two telemetry types, registration, app recording/state transitions,
   error handling, and three-pupil arithmetic corrections.
3. Test serialization, derived text/JSON output, repeated identical fits, three/four
   pupils, state changes and periodic snapshots, averaging resets, reference fits,
   and invalid fits using local synthetic data only. Build pupilFit and logdump.
4. Commit functional changes and this updated plan. Complete a full-file Doxygen
   pass in a separate commit, then clang-format touched C++ files in a final commit.

## Status

- Source audit complete: only frame-grabber timing telemetry currently exists;
  per-pupil values go to INDI and accumulated statistics go to stderr.
- Implemented `telem_pupilfit` (20071) and `telem_pupilfit_state` (20072), generated
  dispatch integration, live/reference emission, state snapshots and averaging windows.
- Successful records contain all active pupil measurements; failed attempts contain
  the pupil count and `valid=false` with an empty measurement vector. They do not
  publish new INDI/stream values or enter temporal statistics.
- State provenance distinguishes requested mode from active mode, including current-fit
  capture (3) and not-yet-allocated (255). State snapshots precede the fits they describe.
- Text and JSON formatters derive x/y/D/median/background averages. Invalid fits expose
  no text average, JSON `average: null`, and NaN through scalar mean accessors.
- mxlib `math/floatUtils.hpp` supplies finite-value validation under the repository's
  `-ffast-math` build flags. This requires a version of mxlib providing that header.
- The app uses its existing INDI mutex to serialize fits, state changes, cached reference
  pixels, and temporal averaging. Frame-grabber access to the published offsets is also
  protected. The three-pupil output now publishes its average INDI property as well.
- Validation: normal optimized pupilFit tests pass (420 assertions, 7 cases), including
  actual binary telemetry queue/file round-trips, generated text/JSON dispatch, JSON
  parsing, missing/non-finite/corrupted payload rejection, repeated identical fits,
  reference identification, state callbacks, averaging resets and failed-frame guards.
- `make -C apps/pupilFit` and `make -C utils/logdump` passed. No instrument connections
  or commands were used. Documentation and final formatting verification remain.

## Implementation notes

- Per-pupil measurements use a FlatBuffers struct (five floats, 20 bytes). Three/four
  pupils occupy 60/80 measurement bytes; message metadata and the log envelope add overhead.
- Derived offset readers must combine each live fit with its preceding state record.
  The formatters derive across-pupil measurement means directly from each fit; they do
  not maintain cross-record setpoint state. Temporal readers also select the recorded
  averaging window and exclude reference fits and invalid attempts.
- The existing INDI callback declaration macros hide their parameter declarations;
  their parameter documentation belongs beside the macro invocation. Ordinary header
  declarations retain inline parameter documentation.
- Existing upstream frame delivery and telemetry logger buffering behavior are retained.
  This records each attempted fit, rather than promising capture of every camera exposure.
