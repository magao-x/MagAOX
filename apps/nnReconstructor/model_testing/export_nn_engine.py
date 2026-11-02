r"""\file export_nn_engine.py
\brief Operator-run export and isolated test configuration for a trained 2026B CNN.

Build on the target GPU with its existing TensorRT installation. This module
does not start services, create instrument streams, or send hardware commands.
"""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_config(model_dir, export_dir):
    """Use recorded preprocessing and an isolated modal stream, preserving the original basis."""
    model_dir, export_dir = Path(model_dir).resolve(), Path(export_dir).resolve()
    record = json.loads((model_dir / "training.json").read_text())
    manifest = record["dataset_manifest"]
    preprocessing = manifest["collection_preprocessing"]
    pupil = int(preprocessing["pupil"])
    x = [int(value) for value in preprocessing["pupil_x"].split(",")]
    y = [int(value) for value in preprocessing["pupil_y"].split(",")]
    height, width = [int(value) for value in preprocessing["source_shape"].split(",")]
    scale = float(preprocessing["imageNorm"])
    modes_path = model_dir / record["calibration"]["mode2act"]["file"]
    if hashlib.sha256(modes_path.read_bytes()).hexdigest() != record["calibration"]["mode2act"]["sha256"]:
        raise ValueError("Saved modal basis checksum differs from the training record.")
    modes = np.load(modes_path, allow_pickle=False)
    if (
        record["commands_units"] != "um_surface"
        or manifest["image_axes"] != ["sample", "pupil", "source_column", "source_row"]
        or manifest["image_shape"] != [4, pupil, pupil]
        or modes.ndim != 2
        or modes.shape[1] != 2500
        or not np.isfinite(modes).all()
        or len(x) != 2
        or len(y) != 2
        or pupil <= 4
        or min(x + y) < 0
        or max(x) + pupil - 4 > width
        or max(y) + pupil - 4 > height
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Unsupported training units, input axes, modal basis or crop geometry.")
    source = preprocessing["source"]
    if any("\n" in value or "\r" in value for value in (source, str(export_dir))):
        raise ValueError("Configuration values cannot contain line breaks.")
    config = f"""# Isolated inference test: this output must have no DM/control-loop consumer.
# Input must remain CPU-backed float32, {width} x {height}, matching training.
# The loop's live mode-to-actuator matrix must match the original saved basis
# before any later connection to aol1_modevalWFS. This file does not change it.
[shmimMonitor]
shmimName={source}
cpuset=acqu_mvm
threadPrio=45

[framegrabber]
shmimName=nn2026B_test_modes
cpuset=acqu_mvm
threadPrio=45

[parameters]
engineDirs={export_dir}
engineName=model_fp32.trt
rebuildEngine=false
imageNorm={scale:g}
modalNorm=1
m_pupPix={pupil}
pup_offset1_x={x[0]}
pup_offset1_y={y[0]}
pup_offset2_x={x[1]}
pup_offset2_y={y[1]}
"""
    return config, (1, 4, pupil, pupil), (1, len(modes)), modes


def check_engine_output(path, reference, modes):
    """Compare trtexec's actual output with the best checkpoint on the same recorded image."""
    output = json.loads(Path(path).read_text())
    if len(output) != 1:
        raise ValueError("The app requires exactly one model output.")
    prediction = np.asarray(output[0]["values"], dtype=np.float32).reshape(reference.shape)
    np.testing.assert_allclose(prediction, reference, rtol=1e-4, atol=1e-6)
    active = np.any(modes != 0, axis=0)
    difference = (prediction - reference) @ modes[:, active]
    return float(1000 * np.sqrt(np.mean(np.square(difference, dtype=np.float64))))


def export_engine(model_dir, sample, reference, trtexec=None):
    """Convert SavedModel, build a float32 engine, check one real sample, and write a test config.

    sample and reference come from the existing final evaluation cell, using the
    best checkpoint and a held-out image. The input is already padded/normalized.
    Existing exports are preserved; rename an earlier tensorrt_fp32 folder to repeat.
    """
    import onnx

    model_dir = Path(model_dir).resolve()
    output = model_dir / "tensorrt_fp32"
    config, input_shape, output_shape, modes = test_config(model_dir, output)
    sample, reference = np.asarray(sample, dtype="<f4"), np.asarray(reference, dtype="<f4")
    if sample.shape != input_shape or reference.shape != output_shape:
        raise ValueError(f"Expected input {input_shape} and output {output_shape}.")
    if not np.isfinite(sample).all() or not np.isfinite(reference).all():
        raise ValueError("The reference image/prediction contains non-finite values.")
    saved_model = model_dir / "saved_model"
    if not (saved_model / "saved_model.pb").is_file():
        raise FileNotFoundError("Run the existing final evaluation/SavedModel cell first.")
    trtexec = trtexec or "/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec"
    if not Path(trtexec).is_file() or shutil.which("setfacl") is None:
        raise RuntimeError(
            "Use the target host's existing trtexec and setfacl; do not install another TensorRT."
        )
    output.mkdir(exist_ok=False)
    sample.tofile(output / "reference_input.bin")
    np.save(output / "reference_output.npy", reference, allow_pickle=False)
    onnx_path, engine_path = output / "model.onnx", output / "model_fp32.trt"
    print("Converting SavedModel to ONNX...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(saved_model),
            "--signature_def",
            "serving_default",
            "--opset",
            "17",
            "--output",
            str(onnx_path),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
    )
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("The app requires one image input and one modal output.")
    for tensor, shape in zip([graph.graph.input[0], graph.graph.output[0]], [input_shape, output_shape]):
        actual = tuple(d.dim_value for d in tensor.type.tensor_type.shape.dim)
        if tensor.type.tensor_type.elem_type != onnx.TensorProto.FLOAT or actual != shape:
            raise ValueError(f"Expected static float32 tensor {shape}; got {actual}.")
    print("Building TensorRT engine; log:", output / "build.log", flush=True)
    with (output / "build.log").open("x") as log:
        subprocess.run(
            [
                str(trtexec),
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
                "--inputIOFormats=fp32:chw",
                "--outputIOFormats=fp32:chw",
                "--noTF32",
                "--skipInference",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise RuntimeError("TensorRT produced no engine; inspect build.log.")
    print("Checking TensorRT output; log:", output / "parity.log", flush=True)
    with (output / "parity.log").open("x") as log:
        subprocess.run(
            [
                str(trtexec),
                f"--loadEngine={engine_path}",
                f"--loadInputs={graph.graph.input[0].name}:{output / 'reference_input.bin'}",
                f"--exportOutput={output / 'engine_output.json'}",
                "--iterations=1",
                "--duration=0",
                "--warmUp=0",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    rms = check_engine_output(output / "engine_output.json", reference, modes)
    config_path = output / "dlReconstructor.conf"
    config_path.write_text(config)
    parents = [str(p) for p in output.parents if p.stat().st_uid == os.geteuid()]
    subprocess.run(["setfacl", "-m", "u:xsup:--x", "--", str(output), *parents], check=True)
    subprocess.run(["setfacl", "-m", "u:xsup:r--", "--", str(engine_path), str(config_path)], check=True)
    print(f"TensorRT/reference command difference: {rms:.6g} nm surface RMS")
    print("Engine:", engine_path)
    print("Isolated test configuration:", config_path)
    print("Required test output stream:", output_shape[1], "x 1 float32, nn2026B_test_modes")
    print("After creating the isolated stream and stopping any existing instance, launch as xsup:")
    print(shlex.join(["/opt/MagAOX/bin/nnReconstructor", "-n", "dlReconstructor", "-c", str(config_path)]))
    return engine_path
