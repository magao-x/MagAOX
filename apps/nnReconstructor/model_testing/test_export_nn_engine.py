r"""\file test_export_nn_engine.py
\brief Check model-test configuration, conversion commands, and parity failure handling.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx

from export_nn_engine import check_engine_output, export_engine, test_config


class ExportTests(unittest.TestCase):
    """Exercise operator export plumbing locally without CUDA or instrument access."""

    def make_model_record(self, root):
        """Write a small training record with the same schema as the real notebooks."""
        root.mkdir()
        modes = np.zeros((2, 2500), dtype=np.float32)
        modes[0, 0], modes[1, 1] = 1, -2
        np.save(root / "mode2act.npy", modes)
        record = {
            "commands_units": "um_surface",
            "calibration": {
                "mode2act": {
                    "file": "mode2act.npy",
                    "sha256": hashlib.sha256((root / "mode2act.npy").read_bytes()).hexdigest(),
                }
            },
            "dataset_manifest": {
                "image_axes": ["sample", "pupil", "source_column", "source_row"],
                "image_shape": [4, 16, 16],
                "collection_preprocessing": {
                    "pupil": "16",
                    "pupil_x": "0,12",
                    "pupil_y": "0,12",
                    "source_shape": "24,24",
                    "imageNorm": "2",
                    "source": "aol1_imWFS2",
                },
            },
        }
        (root / "training.json").write_text(json.dumps(record))
        (root / "saved_model").mkdir()
        (root / "saved_model/saved_model.pb").write_bytes(b"test fixture")
        return modes

    def test_configuration_preserves_calibration_and_uses_isolated_output(self):
        """A test config derives preprocessing from training and never targets the live modal stream."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "trained model"
            original = self.make_model_record(root)
            config, shape, outputs, modes = test_config(root, root / "tensorrt_fp32")
            self.assertIn("shmimName=nn2026B_test_modes", config)
            self.assertNotIn("shmimName=aol1_modevalWFS", config)
            for removed in ("dataDirs=", "use_fp16=", "explicit_tt="):
                self.assertNotIn(removed, config)
            self.assertIn("imageNorm=2", config)
            self.assertEqual(shape, (1, 4, 16, 16))
            self.assertEqual(outputs, (1, 2))
            np.testing.assert_array_equal(modes, original)
            (root / "mode2act.npy").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                test_config(root, root / "tensorrt_fp32")

    def test_invalid_crop_is_rejected(self):
        """Do not launch the C++ app with a crop outside its source image."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "model"
            self.make_model_record(root)
            record = json.loads((root / "training.json").read_text())
            record["dataset_manifest"]["collection_preprocessing"]["pupil_x"] = "0,24"
            (root / "training.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                test_config(root, root / "tensorrt_fp32")

    def test_parity_rejects_wrong_or_nonfinite_predictions(self):
        """An engine mismatch cannot be interpreted as a successful export."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "output.json"
            modes = np.eye(2, 2500, dtype=np.float32)
            reference = np.array([[1, 2]], dtype=np.float32)
            for values in ([1, 3], [1, float("nan")]):
                path.write_text(json.dumps([{"values": values}]))
                with self.assertRaises(AssertionError):
                    check_engine_output(path, reference, modes)
            path.write_text(json.dumps([{"values": [1, 2]}]))
            self.assertEqual(check_engine_output(path, reference, modes), 0)

    def test_export_uses_fp32_io_cpu_conversion_and_readonly_access(self):
        """Simulate target-only commands while checking the complete export/verification sequence."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "model with spaces"
            self.make_model_record(root)
            tool = Path(folder) / "trtexec"
            tool.touch()
            commands = []

            def run(argv, **kwargs):
                """Write deterministic stand-ins for converter and GPU outputs."""
                commands.append((argv, kwargs))
                if "tf2onnx.convert" in argv:
                    graph = onnx.helper.make_graph(
                        [
                            onnx.helper.make_node(
                                "Constant",
                                [],
                                ["modes"],
                                value=onnx.helper.make_tensor(
                                    "prediction", onnx.TensorProto.FLOAT, [1, 2], [1, 2]
                                ),
                            )
                        ],
                        "fixture",
                        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4, 16, 16])],
                        [onnx.helper.make_tensor_value_info("modes", onnx.TensorProto.FLOAT, [1, 2])],
                    )
                    onnx.save(onnx.helper.make_model(graph), argv[-1])
                elif any(arg.startswith("--saveEngine=") for arg in argv):
                    (root / "tensorrt_fp32/model_fp32.trt").write_bytes(b"GPU stand-in")
                elif any(arg.startswith("--exportOutput=") for arg in argv):
                    (root / "tensorrt_fp32/engine_output.json").write_text('[{"values": [1, 2]}]')

            with patch("shutil.which", return_value="/usr/bin/setfacl"), patch(
                "subprocess.run", side_effect=run
            ):
                engine = export_engine(root, np.ones((1, 4, 16, 16)), np.array([[1, 2]]), tool)
                with self.assertRaises(FileExistsError):
                    export_engine(root, np.ones((1, 4, 16, 16)), np.array([[1, 2]]), tool)
            self.assertTrue(engine.exists())
            self.assertEqual(commands[0][1]["env"]["CUDA_VISIBLE_DEVICES"], "-1")
            self.assertIn("--inputIOFormats=fp32:chw", commands[1][0])
            self.assertIn("--outputIOFormats=fp32:chw", commands[1][0])
            self.assertIn("--noTF32", commands[1][0])
            self.assertNotIn("--fp16", commands[1][0])
            self.assertIn("u:xsup:r--", commands[-1][0])
            self.assertTrue((root / "tensorrt_fp32/dlReconstructor.conf").is_file())


if __name__ == "__main__":
    unittest.main()
