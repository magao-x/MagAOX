r"""\file test_training_metadata.py
\brief Exercise the optional home-folder loader patch against small offline runs.

Set NN_DATA_PATH to the patched nn_data.py; no instrument interfaces are used.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("NN_DATA_PATH"), "Set NN_DATA_PATH to the patched training loader")
class TrainingMetadataTests(unittest.TestCase):
    """Test the actual patched packager and categorical filtering without large files."""

    @classmethod
    def setUpClass(cls):
        """Load the selected offline module without modifying the home-folder original."""
        spec = importlib.util.spec_from_file_location("nn_data_metadata_test", os.environ["NN_DATA_PATH"])
        cls.data = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.data
        spec.loader.exec_module(cls.data)

    def test_recorded_nd_and_flipper_are_used_for_training(self):
        """Read ND from run.conf, reject conflicting labels, and filter recorded flipper states."""
        data = self.data
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bank = data.generate_bank(root / "bank", [data.PowerLaw(n=4)], data.demo_calibration(), samples_per_set=2)
            for nd, flip in [("ND2", "out"), ("ND3", "in")]:
                run = data.demo_run(bank, root / nd)
                config = run / "run.conf"
                config.write_text(config.read_text() + f"nd={nd}\nflipwfsf={flip}\n")
                raw_hash = data.sha256(run / "images_dataset_0.bin")
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    data.package_run(run, bank, root / "dataset", {"nd": "ND1"})
                data.package_run(run, bank, root / "dataset")
                self.assertEqual(raw_hash, data.sha256(run / "images_dataset_0.bin"))
            dataset = data.Dataset(root / "dataset")
            self.assertEqual(len(dataset.select(values={"nd": ["ND2"]})), 4)
            self.assertEqual(len(dataset.select(values={"nd": ["ND3"], "flipwfsf": ["in"]})), 4)
            self.assertEqual(len(dataset.select(values={"nd": ["ND2"], "flipwfsf": ["in"]})), 0)
            self.assertEqual(len(dataset.select(values={"flipwfsf": ["out"]})), 4)
            self.assertEqual(dataset.manifest["runs"][0]["settings"]["flipwfsf"], "out")

    def test_old_runs_keep_manual_nd_and_unknown_flipper(self):
        """Historical acquisitions remain readable without inventing a flipper position."""
        data = self.data
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bank = data.generate_bank(root / "bank", [data.PowerLaw(n=2)], data.demo_calibration(), samples_per_set=2)
            run = data.demo_run(bank, root / "raw")
            data.package_run(run, bank, root / "dataset", {"nd": "ND1"})
            dataset = data.Dataset(root / "dataset")
            self.assertEqual(len(dataset.select(values={"nd": ["ND1"], "flipwfsf": ["unknown"]})), 2)


if __name__ == "__main__":
    unittest.main()
