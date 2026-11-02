# Test a newly trained 2026B CNN

Run these steps yourself on exao2. The source tree calls the executable
`nnReconstructor`; `dlReconstructor` is its configured device name.
The training notebook exports `best.keras` and `saved_model/`, not a TensorRT engine.
The export below builds with the existing target-GPU TensorRT installation and
generates a configuration that writes to an isolated modal stream.

## 1. Finish training and collection

Finish the notebook's existing final evaluation/export cell. It creates the
SavedModel and the one-image held-out input `x`. `model` is the best-validation
checkpoint returned by the training helper. Confirm the held-out results are
useful, including the large-command and small-command cases.

Wait for any data collection to complete. Open the relevant AO loops and zero the
dedicated collection channel through the normal instrument controls. Verify its
zero state. Keep the optical setup, WFS preprocessing and reference matched to
collection. Select fwtelsim ND1 for a first bright-model test.

Stop other training kernels before benchmarking or operating the reconstructor.
The training kernel can retain GPU memory even after fitting has finished. If
the engine build runs out of GPU memory, preserve its log, shut down that kernel,
and run the build/parity commands from steps 3 and 4 in a terminal. Do not upgrade
the instrument's TensorRT/CUDA installation to resolve an export problem.

## 2. Add the exporter beside the notebook

Copy `export_nn_engine.py` and `requirements-export.txt` to
`/home/rlandman/NNReconstructor_2026B/`. If the converter packages are missing,
install only these optional packages into the current notebook environment:

```python
%pip install --no-deps -r /home/rlandman/NNReconstructor_2026B/requirements-export.txt
```

The pins are tf2onnx 1.17.0 and ONNX 1.17.0; TensorFlow is not changed. Import the
exporter only after installation. A kernel that already imported incompatible
converter dependencies may require restarting and reloading the best checkpoint.

Check that the existing app and selected builder use matching TensorRT versions:

```sh
ldd /opt/MagAOX/bin/nnReconstructor | rg 'nvinfer|cudart'
```

The default builder path below comes from this checkout's app Makefile:
`/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec`. If the installed app uses a
different runtime, identify its matching existing builder first and pass it as
`trtexec='/actual/path/to/trtexec'`. A serialized engine is not a portable model file.

## 3. Add this final notebook cell

```python
from export_nn_engine import export_engine

engine_path = export_engine(MODEL_DIR, x, model(x, training=False).numpy())
```

The exporter uses the existing `MODEL_DIR`, not a second hardcoded model path. It:

- Converts `saved_model/` to ONNX opset 17 using a CPU-only converter subprocess.
- Requires one fixed batch-one float32 input and one float32 modal output.
- Builds `MODEL_DIR/tensorrt_fp32/model_fp32.trt` with float32 linear I/O and TF32
  disabled for an initial numerical baseline.
- Runs TensorRT on the same held-out image and compares its output with the best
  checkpoint. This is a one-image conversion smoke check, not full model validation.
- Generates `MODEL_DIR/tensorrt_fp32/dlReconstructor.conf` from the recorded WFS
  source, image normalization and pupil crop. It verifies the saved modal checksum.
- Grants xsup traversal/read access to the engine and generated configuration.

Existing exports are not overwritten. Preserve/rename an old `tensorrt_fp32`
folder before deliberately rebuilding. Converter errors appear in the notebook;
build and inference errors have `build.log` and `parity.log` in that folder.
No services, shared-memory streams or DM commands are started by the exporter.

If training retained too much GPU memory, stop its kernel and build the already
exported ONNX from a terminal. Set the following path to your actual MODEL_DIR:

```sh
MODEL_TEST_DIR=/home/rlandman/NNReconstructor_2026B/models/20261015/bright_002/tensorrt_fp32
/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec \
  --onnx="$MODEL_TEST_DIR/model.onnx" \
  --saveEngine="$MODEL_TEST_DIR/model_fp32.trt" \
  --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw --noTF32 --skipInference
```

## 4. Check conversion and timing

Do not proceed if conversion/parity failed. For the terminal fallback, complete
the held-out parity check before using the app:

```sh
/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec \
  --loadEngine="$MODEL_TEST_DIR/model_fp32.trt" \
  --loadInputs="input:$MODEL_TEST_DIR/reference_input.bin" \
  --exportOutput="$MODEL_TEST_DIR/engine_output.json" \
  --iterations=1 --duration=0 --warmUp=0
```

Use the same Python environment to run the checker. This also completes config
creation/access if the notebook stopped during an out-of-memory build:

```python
from pathlib import Path
import numpy as np
from export_nn_engine import check_engine_output, test_config

MODEL_DIR = Path('/home/rlandman/NNReconstructor_2026B/models/20261015/bright_002')
out = MODEL_DIR / 'tensorrt_fp32'
config, _, _, modes = test_config(MODEL_DIR, out)
print(check_engine_output(out/'engine_output.json', np.load(out/'reference_output.npy'), modes),
      'nm surface RMS difference')
(out/'dlReconstructor.conf').write_text(config)
```

If using that fallback, grant xsup read access to the engine/config and traversal
on their parent directories with setfacl, as for the collected banks. The normal
final notebook cell does this automatically.

For a timing baseline, with other GPU jobs stopped:

```sh
MODEL_TEST_DIR=/home/rlandman/NNReconstructor_2026B/models/20261015/bright_002/tensorrt_fp32
/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec \
  --loadEngine="$MODEL_TEST_DIR/model_fp32.trt" --warmUp=1000 --duration=10
```

Keep the timing output. This is an engine benchmark; the app also copies and
crops images and publishes SHMIM output. Check actual output rate in step 6.
The first engine uses float32 for comparison, not a claim of maximum framerate.
An internal-FP16 engine can be tested later with float32 I/O retained; the app
requires FP32 I/O regardless of internal engine precision. Recheck accuracy then.

## 5. Create an isolated output stream

As xsup on exao2, confirm `nn2026B_test_modes` has no consumers and no existing
writer. Create it once using the output width printed by the exporter. With the
unchanged original basis that width is 2400:

```sh
cacao <<'EOF'
mk2Dim s>tf32>nn2026B_test_modes 2400 1
quit
EOF
```

Inspect with `milk-shmimmon nn2026B_test_modes`: require 2400 by 1, float32. If a
stream of that name already exists with the wrong shape or another writer, stop
and use a different isolated name consistently in the test config and commands.
The app attaches to an existing output stream; it does not create this one.

## 6. Run live-input inference without connecting to the DM

Keep the relevant loops open. Stop the existing dlReconstructor application using
its established launcher/tmux session before starting another instance of that
device. Do not stop its INDI bridge. Do not start two applications with the same
name or two writers on one output stream.

Inspect the input with `milk-shmimmon aol1_imWFS2` (or the recorded source if
different). Require CPU-backed float32 with the width/height printed in the
generated config comment: normally 120x120. The current app does not validate
input type/crop bounds against changed live geometry. Stop if they differ.

Launch as xsup, using the generated config and your actual MODEL_DIR path:

```sh
/opt/MagAOX/bin/nnReconstructor -n dlReconstructor \
  -c /home/rlandman/NNReconstructor_2026B/models/20261015/bright_002/tensorrt_fp32/dlReconstructor.conf
```

Require the startup log to report one input, one output, input 4x64x64 and output
2400 for the default pupil/original basis. Exactly one FP32 image input and one
FP32 modal output are required. The app validates tensor roles and maps bindings
by role, independent of enumeration order. Remove `dataDirs`, `use_fp16` and
`explicit_tt` from existing base/override configs; they are no longer accepted.

Verify the published output destination, then observe the stream:

```sh
getINDI -h localhost -p 7624 -t 3 'dlReconstructor.fg_shmimName.name'
milk-shmimmon nn2026B_test_modes
```

The destination must be `nn2026B_test_modes`, its counter must advance, and its
values must stay finite. Check output rate against the actual WFS source rate
and observe logs for CUDA, TensorRT, shape and missed-frame errors. A running
process alone does not establish valid inference. Test responses to small known
disturbances using the existing instrument calibration procedure; compare the
reconstructed actuator map using the saved original mode2act, not a different basis.

The app checks CUDA/TensorRT failures and synchronizes the output handoff. These
source-level checks do not establish GPU parity or instrument readiness. Keep
this first test isolated; validate at-rate output before considering loop control.

## 7. Before a closed-loop test

The prepared configuration intentionally stops at isolated inference. Connecting
it to the HO loop requires checking the current live config and downstream CACAO
calibration, which were not available for this handoff. Specifically:

1. Open the loop and retain a working baseline/rollback configuration.
2. Confirm the live mode-to-actuator matrix is exactly the original 2400-mode
   basis saved with this model, in the same order and surface-command units.
   Matching vector length alone is insufficient. Do not truncate/pad coefficients.
3. Confirm reference subtraction, normalization and crop match training, and
   confirm the estimated disturbance has the sign expected by the existing
   negative-feedback controller. The app does not apply modalNorm.
4. Disable the competing reconstructor writer through the normal CACAO procedure
   before targeting `aol1_modevalWFS`; leave downstream control/limits as established.
5. With the loop still open, verify that the actual modal stream has the expected
   dimensions and only the new writer. Test the sign/scale with the normal small
   calibration disturbance and available command limits.
6. Close at the instrument's established commissioning gain, watching residuals
   and DM commands. Open immediately on growth/non-finite output, then restore
   the baseline reconstructor and verify it before continuing.

Do not infer a safe gain, modal conversion, or writer-switch command from this
document. Those depend on the live loop setup. No engine build, app launch,
calibration switch or loop closure was executed by the agent.

References: [tf2onnx SavedModel conversion](https://github.com/onnx/tensorflow-onnx),
[TensorRT 10.0 CLI options](https://github.com/NVIDIA/TensorRT/blob/release/10.0/samples/common/sampleOptions.cpp),
[MagAO-X CACAO setup](https://magao-x.org/docs/handbook/operating/cacao.html).
