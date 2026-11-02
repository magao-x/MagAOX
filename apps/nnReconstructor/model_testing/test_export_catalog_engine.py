r"""\file test_export_catalog_engine.py
\brief Test catalog export publication, build flags and failure cleanup without a GPU.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from export_catalog_engine import export_catalog_engine


class CatalogExportTests(unittest.TestCase):
    """Exercise the real export function with subprocess-only boundary doubles."""

    def test_publish_and_failures(self):
        """Only a successful nonempty build with readable permissions becomes a catalog entry."""
        for failure in (None, "convert", "build", "empty", "permissions"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                model = root / "20261015" / "bright_003"
                saved = model / "saved_model"
                saved.mkdir(parents=True)
                (saved / "saved_model.pb").write_bytes(b"saved")
                catalog = root / "engines"
                catalog.mkdir()
                existing = catalog / "older.trt"
                existing.write_bytes(b"older")
                commands = []

                def run(command, **kwargs):
                    """Emulate compiler artifacts and fail a chosen external boundary."""
                    commands.append(command)
                    self.assertTrue(kwargs["check"])
                    self.assertEqual(list(catalog.glob("*.trt")), [existing])
                    if "tf2onnx.convert" in command:
                        phase = "convert"
                    elif command[0].endswith("trtexec"):
                        phase = "build"
                        self.assertIn("--fp16", command)
                        self.assertIn("--inputIOFormats=fp32:chw", command)
                        self.assertIn("--outputIOFormats=fp32:chw", command)
                        target = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--saveEngine=")))
                        target.write_bytes(b"" if failure == "empty" else b"engine")
                    else:
                        phase = "permissions"
                    if failure == phase:
                        raise subprocess.CalledProcessError(1, command)

                with patch("export_catalog_engine.subprocess.run", side_effect=run):
                    if failure:
                        with self.assertRaises((subprocess.CalledProcessError, RuntimeError)):
                            export_catalog_engine(model, catalog)
                        self.assertEqual(list(catalog.iterdir()), [existing])
                    else:
                        output = export_catalog_engine(model, catalog)
                        self.assertEqual(output.parent, catalog)
                        self.assertTrue(output.name.startswith("20261015_bright_003_"))
                        self.assertEqual(output.read_bytes(), b"engine")
                        self.assertIn(["setfacl", "-m", "u:xsup:r-x", "--", str(catalog)], commands)
                        self.assertEqual(len(list(catalog.iterdir())), 2)
                self.assertEqual(existing.read_bytes(), b"older")

    def test_missing_saved_model(self):
        """A missing training export never starts a compiler or creates a catalog."""
        with tempfile.TemporaryDirectory() as folder, patch("export_catalog_engine.subprocess.run") as run:
            with self.assertRaises(FileNotFoundError):
                export_catalog_engine(Path(folder) / "model", Path(folder) / "engines")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
