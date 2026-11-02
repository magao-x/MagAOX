r"""\file export_catalog_engine.py
\brief Build the notebook's FP16 engine with FP32 I/O and publish it to the app catalog.

Called by the final training cell on the supported GPU host. Does not control the app.
"""

import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def export_catalog_engine(model_dir, engine_dir):
    """Export a SavedModel and atomically publish a uniquely named engine after a successful build.

    model_dir is the training run directory containing saved_model; engine_dir is
    the app's configured engineDirs. Internal FP16 and FP32 I/O match the notebook.
    This builds an engine; it does not establish numerical parity or activate it.
    """
    model_dir, engine_dir = Path(model_dir).resolve(), Path(engine_dir).resolve()
    saved_model = model_dir / "saved_model"
    if not (saved_model / "saved_model.pb").is_file():
        raise FileNotFoundError("Run the notebook's SavedModel export cell first.")
    onnx_path = model_dir / "model.onnx"
    subprocess.run(
        [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(saved_model),
         "--opset", "17", "--output", str(onnx_path)],
        check=True,
    )
    engine_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{model_dir.parent.name}_{model_dir.name}_{stamp}_fp16.trt"
    destination = engine_dir / name
    # A non-recursive startup scan never sees an incomplete build.
    with tempfile.TemporaryDirectory(prefix=".export-", dir=engine_dir) as staging:
        staged = Path(staging) / name
        subprocess.run(
            ["/opt/MagAOX/vendor/TensorRT-10.0.0.6/bin/trtexec", f"--onnx={onnx_path}",
             f"--saveEngine={staged}", "--fp16", "--inputIOFormats=fp32:chw",
             "--outputIOFormats=fp32:chw"],
            check=True,
        )
        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError("TensorRT produced no engine; nothing was published.")
        parents = [str(p) for p in engine_dir.parents if p.stat().st_uid == os.geteuid()]
        if parents:
            subprocess.run(["setfacl", "-m", "u:xsup:--x", "--", *parents], check=True)
        subprocess.run(["setfacl", "-m", "u:xsup:r-x", "--", str(engine_dir)], check=True)
        subprocess.run(["setfacl", "-m", "u:xsup:r--", "--", str(staged)], check=True)
        # Same-filesystem hard link publishes the complete file without overwriting a prior export.
        os.link(staged, destination)
    print("Published engine:", destination)
    print("Restart nnReconstructor to discover it; model selection does not enable reconstruction.")
    return destination
