# nnReconstructor controls and validation

This app retains the existing pupil extraction, padding, imageNorm multiplication, synchronous
`executeV2()` and modal ordering. It accepts exactly one FP32 image input and one FP32 modal output.
Internal TensorRT computation may still use FP16. `modalNorm` remains unapplied and `rebuildEngine`
remains a compatibility setting; the app does not load ONNX or rebuild engines.

Remove `explicit_tt`, `use_fp16` and `dataDirs` from existing configuration/override files before
running this version: these settings and their C++ implementation have been removed. Unknown settings
cause startup to fail. Models are loaded from `engineDirs` plus `engineName`.

## Configuration and cursesINDI

After removing the obsolete settings above, these optional parameters add runtime controls:

```ini
[parameters]
startEnabled=true
transitionTimeout=30
```

At startup the app scans the top level of `engineDirs` for `.trt`, `.engine` and `.plan`
files. It loads `engineName` first and lists only candidates that deserialize, match the
FP32 image/output contract and modal count, and can create an execution context. Rejected
files are logged and skipped; failure to load `engineName` or list the directory fails
startup. Candidates are inspected one at a time before acquisition starts, so large catalogs
increase startup time and require memory for the loaded model plus one candidate.

`models` remains an optional list of extra filenames, including nonstandard extensions;
it is not needed for directory discovery. The startup model stays first; explicit entries
retain their order, followed by sorted discoveries with duplicates removed. New files appear
after an app restart. Selection revalidates the file, since files can change after startup.
`xsup` needs read and execute permission on the catalog directory and read permission on engines.

All choices must share preprocessing, modal basis, ordering, sign and units. Tensor metadata
cannot establish calibration/basis identity; confirm this from model provenance. Engine scanning
does not execute inference, establish numerical parity, or select/activate a different startup model.

The `model` switch elements have readable identifiers such as `model0_bright_trt`; their labels
contain the full filenames. In cursesINDI, use `p` on the desired element to request a load and
`t` on `enable.toggle` to request activation/deactivation. Loading the already-selected model
is the explicit reload operation. It does not activate reconstruction.

Read `model_status.requested`, `.loaded`, `.active`, `.status` and `.error` to distinguish the
request from completion. `active` names the model that published in the current enabled run;
it is not a guarantee of ongoing source frames, downstream consumption or a closed loop. Use the
existing monitor properties to inspect stream connection state. Properties report Busy during
transitions and Alert on failures/rejected requests. The enable readback stays On during ordinary
disable until the zero publication completes. An enabled-but-waiting state is not successful inference.

`startEnabled=true` preserves automatic startup. `false` starts passively: it performs no inference
or unsolicited zero publication. Automatic startup must only be used with the existing exclusive-writer
arrangement. Neither the app nor these controls stop/restart MVM or change the loop state.

## Operator-managed transitions

Open the relevant loop and establish exclusive writer ownership before changing reconstruction.
There is no guaranteed app-local loop interlock and no use of cached loop state as one.

- **Disable:** close new-inference admission and discard unclaimed results. An admitted inference
  may finish, but its result is discarded. An already-claimed publication may finish. The publisher
  then writes exactly one zero vector and acknowledges Off after its post-publication hook. The
  stream remains externally owned; repeating Off after completion does not publish another zero.
- **Select:** allowed only when disabled and both workers are quiescent. A background loader creates
  and validates the candidate while keeping the previous engine available. Failure keeps the previous
  `loaded` model, records the requested filename and leaves a fault. Success never activates it.
- **Enable:** requires a validated WFS attachment, no unfinished transition and no latched fault.
  The first monitor callback anchors the source counter; a later changed counter admits inference.
  This deliberately waits across a fresh frame boundary rather than replaying a pending result.
- **WFS reconnect:** reuse fixed buffers, invalidate unclaimed results and wait across a fresh frame
  boundary. A claimed snapshot remains valid until publication completes. Invalid source geometry,
  dtype or CPU/GPU location latches a fault; a valid reconnect alone does not clear it.
- **Runtime fault:** stop new inference and discard pending results, retaining the last published
  stream value. No unsolicited zero or automatic fallback is sent. If disable was already requested,
  its explicit zero can still finish. After workers stop, load a valid model explicitly (the same
  selection is allowed), then separately enable. Reload can fail again if the underlying CUDA
  context/driver is unhealthy; this may require an operator-supervised process restart.
- **Output unavailable/changed:** do not acknowledge the disable zero on an invalid mapping. Restore
  the externally owned stream; the framegrabber retries attachment. A timed-out disable remains
  incomplete until its actual zero finishes, and the fault then requires explicit reload. If zero
  timestamping fails, retry Off after correcting the clock problem.
- **Shutdown:** stop admitting work and cancel an unclaimed zero. Join the loader and data-path
  threads before releasing resources. Shutdown is not a substitute for a completed disable; request
  Off and verify completion beforehand if a zero publication is needed.

A zero modal measurement does not prove zero DM command: downstream filter history and loop state
are separate. The existing base's hook establishes that the bytes/counter publication path has
completed. The base discards the semaphore-post return value, so the app cannot confirm successful
notification or downstream receipt through that interface. Check the consumer during commissioning.

Timeouts report incomplete work without falsely acknowledging completion. They cannot safely cancel
native CUDA/TensorRT calls. A candidate completing after its deadline is rejected before installation.
A driver stall during inference, loading or resource destruction can still hold shutdown until an
operator/supervisor intervenes. Do not free resources or activate another writer on a timeout alone.

Output metadata is checked before a publication is claimed; filename identity is checked at most
once per second on the publisher. These checks cannot prevent an external replacement race and add
a periodic filesystem call whose latency must be measured. Disable before controlled replacement.

## Local tests

From the repository root:

```sh
make -C apps/nnReconstructor/tests test
make -C apps/nnReconstructor/tests optimized
make -C apps/nnReconstructor/tests asan
make -C apps/nnReconstructor/tests tsan
```

All test outputs stay in the app's ignored `tests/.build` directory. Tests compile the actual app
methods against CPU-only CUDA/TensorRT, INDI and base-class boundary doubles. They exercise the
production preprocessing, validation, state transitions, real C++ loader thread and result handoff.
They do not validate the SDK ABI, GPU inference, network INDI transport, actual ImageStreamIO posting,
or the real base-class thread lifecycle. The non-unit modalNorm fixture verifies its legacy no-op.

The test Doxygen group lives in this directory to honor the app-only scope. No global test/build or
Doxygen-group file is changed. New test cases must exercise production behavior, not a parallel model
of the state machine.

The `optimized` target also runs the suite with the repository's default `-O3 -ffast-math` flags.
Finite-value validation inspects IEEE exponent bits by reference: ordinary `std::isfinite()` and
by-value floating-point arguments can be optimized under assumptions that exclude NaN/infinity.
Tests inject their storage bits and check FP32 output rejection, invalid
normalization/timeouts, and continued acceptance of valid configurations. Override `CXX` with the
production compiler when running this target on the supported system. No arithmetic flags are changed.

## Supported-system checks before deployment

An operator must perform these checks on an authorized supported system; no instrument commands
were run during implementation.

1. Build the app against the existing MagAOX, CUDA and TensorRT versions. Do not upgrade dependencies.
2. Replay identical recorded inputs through baseline and changed apps into an isolated output with
   no control/DM consumer. Compare the FP32 input tensors and modal vectors. Agree the numerical tolerance before comparing GPU results.
3. At the intended rate and representative CPU/GPU contention, measure callback-to-publication and
   source-acquisition-to-publication latency separately: median, p95, p99, maximum, first inference,
   output intervals, source sequence gaps and repeated frames. Match source/result identifiers;
   the existing output `atime` remains callback-entry time, not sensor acquisition time.
4. Exercise disable during inference/publication, one-shot zero counter increments, failed load,
   explicit recovery, missing/replaced output, WFS reconnect, and shutdown. Verify consumer behavior,
   not merely the INDI readback. Do not change frame dropping or draining policy as an optimization.
5. Only after isolated validation, use the established operator-controlled loop-open commissioning
   procedure, verified model basis/sign/scale and a known working rollback configuration.

No GPU numerical-parity or end-to-end latency result is claimed here. Performance changes (pinned
buffers, async enqueue, warm-up or graphs) remain a separate future step after measurement.

## Training notebook export

`model_testing/final_cell.py` replaces the TensorRT export cell in `Train_bright.ipynb`.
It calls `export_catalog_engine.py` to retain internal FP16 computation with FP32 I/O,
build outside the visible catalog, and publish a uniquely named completed engine into
`ENGINE_DIR` (default `/home/rlandman/NNReconstructor_2026B/engines`). The exporter grants
`xsup` directory-listing and engine-read access. Match the app's `engineDirs` to this path;
`engineName` must name an existing startup engine there. Repeated exports do not overwrite
previous engines. The original training artifacts remain in `MODEL_DIR`.

The export cell does not restart/enable the app or validate numerical parity. After an
operator-run build and app restart, select the discovered model while disabled, then enable
separately. `model_testing/export_nn_engine.py` remains the separate isolated FP32 parity workflow.
CPU-only publication tests:

```sh
python -m unittest discover -s apps/nnReconstructor/model_testing -p test_export_catalog_engine.py
```
