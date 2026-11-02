# dlDataCollection improvements

## Request and scope

Implement findings 1–10 from the 2026-10-12 review on
`rlandman/dlDataCollection-improvements`, based on `origin/dev` (02363bcf).
The acquisition uses the internal source and the operator changes `fwtelsim`.
ND sequencing is a later change; this work fixes collection and offline preparation.

## Execution boundary

Current local AGENTS.md rule 0 prohibits agent execution on instrument computers.
All changes, builds and tests run on the workstation. One read-only RTC inspection
had already been dispatched while the updated local instructions were being read.
The user subsequently explicitly authorized read-only RTC file/state checks and
prohibited all other remote commands. That session authorization takes precedence
for read-only inspections; installation, remote builds/tests and hardware commands
remain excluded.
The earlier review's remote notebook is replaced by a tracked offline preparation
tool and notebook here; the original remote file is not edited.

## Decisions

- Keep the application header-only, with out-of-class definitions and standard
  shmimMonitor macros. Put CPU-only collection logic in a local helper header so
  the production state machine and file handling can be tested without hardware.
- Use a recorded 50x50 actuator mask and explicit command units. Reject non-finite,
  excluded-actuator, malformed, or checksum-incompatible input before acquisition.
  Never silently clip or rescale commands. Legacy CSV conversion is offline.
- Use little-endian float32 binary banks with exact sizes and SHA-256 checksums.
  Python hashlib and the installed OpenSSL implementation calculate these in C;
  a Python byte-at-a-time checksum would add substantial preparation overhead.
- Preserve the existing four-pupil, column-major crop and two-pixel padding.
- NumFrameSkip is the minimum source-counter advance after publication, consistently
  for every command. Eight retains the old eight-frame sample interval while fixing
  the old seven-frame settling age after the first sample. Zero means the first
  fresh frame, never the image that triggered publication. Delay is not reduced.
- Reject repeated/reset source counters and torn image snapshots; record source
  counters, source timestamps, command publication times, and applied labels.
- Use two bounded set buffers. A normal-priority worker preflights, preloads and
  writes. The capture path never parses or writes files. If no set is ready,
  pause with the owned command channel zeroed rather than mislabeling frames.
- Preserve partial sets on Stop and failure. Exclusive run directories and
  checked no-replace file finalization prevent overwriting old runs. Completion
  markers distinguish complete sets from partial or failed writes.
- Camera reconnect aborts the current run; it never resumes against dataset zero.
  Main-thread shutdown joins workers; image callbacks never call appShutdown.
- Remove unused TensorRT/CUDA dependencies. Preserve old configuration key names;
  document required bank migration and the formerly unused modalNorm setting.
- The common build uses fast-math. An optimized regression exposed removal of
  NaN/Inf validation under those assumptions. The app and its registered test
  retain -O3 but explicitly disable fast-math; input validity must remain observable.
- User requested no overengineering: retain one worker, exactly two buffers,
  and only Start/Stop plus bank/output controls. Remove the extra run-label control;
  ND sequencing, camera-setting management and general pipeline abstractions are outside scope.

## Planned verification

CPU Catch2 checks exercise bank integrity, layouts, mask enforcement, exact set
completion, command/image association under duplicate notifications and frame
jumps, torn reads, cancellation, bounded backpressure, reconnect/timeout aborts,
write failures, and collision protection. Python tests exercise hardware-free
generation, every sample including the last, deterministic seed handling, mask
and piston behavior, and both legacy CSV layouts. Run sanitizer and optimized
fast-math checks where applicable, and clang-format every changed C++ file.

The Linux MagAO-X application build and physical latency/framerate validation
remain operator-run checks if local dependencies are unavailable. They are not
substituted by portable helper tests.

## Execution notes

- Read current local and remote instructions; local instructions are newer.
- Created feature branch from origin/dev; existing untracked user files remain.
- Implemented findings 1–10: exact termination/main-thread shutdown; fresh stable
  frame association; strict offline bank validation; checked exclusive output;
  stream/configuration checks; hardware-free generation; recorded-mask enforcement;
  abort on reconnect; bounded worker I/O; and eligible-frame-only crop/publication.
- Source checks preserve the previous crop orientation and command memory order.
  Stream mapping identity and intervening-writer checks precede publication and
  sample acceptance. Cleanup releases only a channel still owned by this app.
- Local C++ checks pass: 146 assertions in 13 cases under AddressSanitizer plus
  UndefinedBehaviorSanitizer, ThreadSanitizer, and -O3 with fast-math disabled.
  The optimized check initially exposed elided NaN/Inf validation with fast-math;
  corrected the application and registered test flags, then reran successfully.
- Five offline Python tests pass using NumPy and HCIPy 0.7.1. Both CSV layouts,
  seeded generation, the last sample, explicit repairs and provenance are covered.
- Adapter syntax passes with the real ImageStreamIO and INDI declarations and
  lightweight MagAOX base declarations. This is a syntax smoke check, not a full
  application build. The workstation lacks the full Linux MagAOX dependencies.
- No instrument build, install, restart, or command was performed. Native app
  compilation, integration checks, and measured instrument throughput remain
  operator validation. NumFrameSkip has not been tuned down.
- At the current 10000-sample, four 64x64-pupil configuration, two image/command
  buffers use about 1.51 GB plus small timing records. Slow output pauses capture
  at set boundaries with the owned channel zeroed. Failed writes retain buffers,
  block another Start, and never produce a successful completion marker.
- Documented migration, controls, output interpretation, interrupted runs and
  workstation verification in the app README. Completed touched-file Doxygen,
  including explicit production-API links in the fault-injection tests.
  Functional changes, documentation and final clang-format cleanup are separate
  commits. The original remote notebook and instrument config remain untouched.
- clang-format 23 follows the repository configuration with one command-line
  CommentPragmas exception for `^mutex scope$`: the default formatter inserts
  a space after `//`, conflicting with AGENTS.md's exact annotation. Preserve
  `{ //mutex scope` without changing the repository's global formatter settings.

## Notebook workflow handoff (2026-10-12)

- The user explicitly authorized the notebook workflow in
  `/home/rlandman/NNReconstructor_2026B` on exao2 and uploading files there.
  Remote execution, builds, installation, restarts and hardware commands remain
  excluded. Only file reads/checks and the authorized file transfer were used.
- The active workflow is four notebooks: generation, verified packaging, bright
  training and faint training. Shared helpers remain offline; distributions are
  edited directly in the generation notebook rather than through prepare_bank.py.
- User selected fwtelsim ND1 for bright and ND1+ND2+ND3 for faint. Magnitude targets
  were removed from sampling and notebook controls. The old metadata slot remains
  unknown for file compatibility. Exponent, RMS, layer and physical ND are retained.
- User explicitly required the exact existing modal FITS and no additional masking.
  Generation reads the original AOC-mounted FITS and preserves every modal value
  and its ordering. There is no extra circular mask or piston subtraction. The
  required bank validity mask records only the support already present in the basis.
- Bright loss follows smartPWFS_elt LogVariance: average per-frame
  log(masked MSE + epsilon squared). Faint follows MeanRMSE: average per-frame RMSE,
  without target-amplitude normalization or epsilon. This command-data workflow
  evaluates errors in nm surface; the bright RMS floor is explicitly named
  epsilon_nm_surface. Both notebooks default to zero additional L2 regularization.
- Default generation RMS is the explicit actuator OPD proxy. No calibrated DM
  response or physical optical-RMS certification is inferred from a modal basis.
- Packaging verifies complete-bank coverage and finalized run markers before
  import. Repeating a complete, identically labelled import is idempotent;
  interrupted imports remain preserved and require inspection.
- Local verification passes 13 tests and executes all four notebooks with tiny
  synthetic data under both TensorFlow 2.21/Keras 3.15 and TensorFlow 2.18/Keras 3.6.
  Checkpoint reloads, batch-one SavedModel parity and packaging reruns pass.
  A small bank generated with the real 2400-mode FITS preserves its values exactly
  and passes the production C++ bank validator.
- Read-only inspection confirms the old CSV collector is still installed on exao2.
  A source patch restricted to this app's four build/source files, a count override,
  and operator build/install instructions accompany the notebooks. The remote
  source header matches the patch base. No full Linux app build or live framerate,
  acquisition, GPU training, or model commissioning was performed.
- TOMORROW.md documents the three manual physical ND collections, shared paths,
  independent training filters/loss controls, original calibration provenance,
  complete-run checks, resource estimate and the collector installation prerequisite.

## Direct home-source update and notebook simplification (2026-10-15)

- The user declined GitHub publication and explicitly requested a source update in
  their exao2 home checkout, including switching the correct feature branch. A Git
  bundle transferred the four existing commits directly; the checkout is now on
  `rlandman/dlDataCollection-improvements` at d5ca5278. The previous PO4AO branch and
  unrelated untracked files were preserved. No build/install/restart was run.
- The generator now writes one bank per run. One visible parameter cell contains
  the bank save path, suggested collector image path, original FITS, seed, set
  size and a single layer list. ND never silently changes the distribution.
  Operators can reuse a bank at multiple physical NDs or explicitly generate a new
  bank with a new recipe/path. The notebook prints the actual collector settings.
- Packaging has an explicit ND/bank/raw-folder table and visible output path.
  Both training notebooks now show their dataset/model paths directly. The four
  notebooks no longer require hidden path settings or smoke-test branches through
  experiment.py. Original modal values, no additional masks/piston processing,
  metadata, loss definitions and checked file handling are preserved.
- Fifteen unit tests pass. The four actual notebooks pass locally under TensorFlow
  2.18/Keras 3.6 with three tiny banks made from the exact original FITS, different
  distributions, custom paths containing spaces, repeated packaging and both
  training/export profiles. All three banks pass the production C++ validator.
- Previous home-folder files are backed up before the simplified versions replace
  them. Installation remains the operator's normal app-directory `make install`;
  the install target already invokes sudo for the installation step.

## Automatic notebook bank access (2026-10-15)

- Operator readback now confirms that the new collector is running and connected
  to isRTC. Its Start request failed with `missing offline bank manifest`; an
  operator `find` reports permission denied on the generated bank tree for xsup.
- `apps/dlDataCollection/update_bank_notebook.py` adds a permissions cell directly
  after generation in the existing home-folder notebook. It preserves the user's
  paths, distributions, calibration, outputs and other cells, backs up the original,
  and can be reapplied without duplicating the new cell. It never runs the notebook.
- The cell grants named-user xsup traversal on operator-owned ancestor directories
  and recursive read/search access within the selected bank using Linux setfacl.
  It does not grant xsup write access or make other home-directory files readable.
  Missing tools, missing manifests and ACL failures stop with an error. Operators
  can run this cell alone for an existing bank without regenerating coefficients.
- Five focused Python tests pass; notebook schema, all code-cell syntax and exact
  preservation of the original cells/settings pass locally. Actual Linux ACL
  application remains operator-run and has not been exercised on the instrument.
- Prepared the updated notebook in the local staging folder and a standalone updater
  for operator transfer. No exao2 commands or uploads were performed: current
  AGENTS.md rule 0 requires the user to perform instrument-side steps. Apply once
  with `python3 ~/NNReconstructor_2026B/update_bank_notebook.py`, reopen the notebook,
  then use Run All normally. No collector source change or rebuild is needed.

## Trained-model TensorRT handoff (2026-10-15)

- The operator reports training finished and requests a final-cell TensorRT export
  and dlReconstructor test configuration/procedure. The existing notebooks export
  only Keras/SavedModel. The app executable is nnReconstructor; dlReconstructor is
  the configured instance. No deployed runtime/config/loop state was inspected.
- Added an operator-run exporter under apps/nnReconstructor/model_testing. It
  converts the existing fixed batch-one SavedModel with tf2onnx, checks ONNX I/O,
  builds a float32-I/O TensorRT engine using the existing vendor builder, and
  compares actual trtexec output with one held-out best-checkpoint prediction.
  Engine building and inference are user-run on the target GPU, never agent-run.
- The generated dlReconstructor.conf derives the source, crop and imageNorm from
  training.json, verifies the saved original modal-basis checksum, and uses the
  isolated existing output nn2026B_test_modes. Named xsup ACLs cover the engine
  and configuration. No original modal values, masks or normalization are changed.
- The guide covers target-runtime compatibility, export, timing, output-stream
  creation, isolated inference and rollback/conditions for eventual loop testing.
  A bright_002 example config and pasteable final cell accompany it.
  Connecting to aol1_modevalWFS requires verifying the live 2400-mode conversion,
  units/sign, sole writer and normal controller commissioning settings; source
  review still finds unchecked CUDA/TRT errors and the existing output-buffer race.
- Four local tests pass for config fidelity, crop bounds, command arguments,
  read-only access and failed numeric parity. TensorFlow 2.18/Keras 3.6 SavedModel
  conversion to ONNX opset17 passes with tf2onnx1.17/ONNX1.17; ONNX Runtime1.20.1
  agrees on all seven available held-out synthetic images (real original 2400-mode
  basis) with maximum coefficient difference 1.61e-10. GPU commands are mocked in
  tests; no real TensorRT engine build, live inference or closed-loop test occurred.
- The operator supplied the actual trained model directory
  `/home/rlandman/NNReconstructor_2026B/models/20261015/bright_002`; prepared example
  config and guide paths now match it. Export still uses the notebook's MODEL_DIR
  and derives preprocessing from that model's training record. Current live config
  and downstream modal conversion remain unprovided, so the test stays isolated.
