"""Regression checks of the offline bank generator and legacy-format migration."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("prepare_bank", Path(__file__).parents[1] / "prepare_bank.py")
bank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bank)


class BankTests(unittest.TestCase):
    """Use real temporary payloads without importing any instrument interface."""

    def setUp(self):
        """Create a recorded mask and exclusive scratch directory."""
        self.scratch = tempfile.TemporaryDirectory()
        self.root = Path(self.scratch.name)
        self.mask = np.ones(2500, dtype=np.float32)
        self.mask[:50] = 0
        self.mask[1744] = 0
        np.save(self.root / "mask.npy", self.mask.reshape(50, 50))

    def tearDown(self):
        """Remove only this test's temporary files."""
        self.scratch.cleanup()

    def test_seed_mask_and_last_sample(self):
        """A fixed seed reproduces all samples; excluded positions and active piston are correct."""
        args = (self.mask, 3, 2, 123, 0.01, 0.1, -2.5, -0.5)
        first = list(bank.random_commands(*args))
        second = list(bank.random_commands(*args))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.shape, (3, 2500))
            self.assertTrue(np.any(a[-1] != 0))
            self.assertTrue(np.all(a[:, self.mask == 0] == 0))
            self.assertTrue(np.all(np.abs(a[:, self.mask != 0].mean(axis=1)) < 1e-7))

    def test_both_csv_layouts_and_manifest(self):
        """One-column and 2500-column CSV inputs produce the same canonical float32 payload."""
        values = np.arange(5000, dtype=np.float64).reshape(2, 2500) * self.mask
        for name, payload in [("flat", values.ravel()), ("rows", values)]:
            source = self.root / name
            source.mkdir()
            np.savetxt(source / "modeval_dataset_0.csv", payload, delimiter=",")
            output = self.root / f"bank_{name}"
            bank.main(["convert", "--source", str(source), "--output", str(output),
                       "--mask", str(self.root / "mask.npy"), "--samples", "2", "--sets", "1",
                       "--units", "um_surface"])
            actual = np.fromfile(output / "modeval_dataset_0.bin", dtype="<f4").reshape(2, 2500)
            np.testing.assert_array_equal(actual, values.astype("<f4"))
            fields = dict(line.split("=", 1) for line in (output / "manifest.conf").read_text().splitlines())
            self.assertEqual(fields["set_0_sha256"], bank.checksum(output / "modeval_dataset_0.bin"))
            self.assertEqual(fields["generation_sha256"], bank.checksum(output / "generation.json"))
            record = json.loads((output / "generation.json").read_text())
            self.assertFalse(record["repair_mask"])
            self.assertEqual(len(record["source_files"]), 1)

    def test_repair_is_explicit_and_existing_banks_are_preserved(self):
        """Conversion rejects incompatible masks unless the user explicitly requests a new repaired bank."""
        source = self.root / "legacy"
        source.mkdir()
        values = np.ones((1, 2500))
        np.savetxt(source / "modeval_dataset_0.csv", values, delimiter=",")
        common = ["convert", "--source", str(source), "--mask", str(self.root / "mask.npy"),
                  "--samples", "1", "--sets", "1", "--units", "um_surface"]
        bad = self.root / "bad"
        with self.assertRaises(ValueError):
            bank.main(common + ["--output", str(bad)])
        self.assertFalse((bad / "manifest.conf").exists())
        output = self.root / "repaired"
        bank.main(common + ["--output", str(output), "--repair-mask"])
        self.assertTrue(json.loads((output / "generation.json").read_text())["repair_mask"])
        actual = np.fromfile(output / "modeval_dataset_0.bin", dtype="<f4")
        self.assertTrue(np.all(actual[self.mask == 0] == 0))
        original = (output / "modeval_dataset_0.bin").read_bytes()
        with self.assertRaises(FileExistsError):
            bank.main(common + ["--output", str(output), "--repair-mask"])
        self.assertEqual((output / "modeval_dataset_0.bin").read_bytes(), original)

    def test_validation(self):
        """Wrong dimensions, non-finite values, and nonbinary masks are rejected."""
        for bad in [np.zeros(2499), np.ones(2500) * np.nan, np.ones(2500) * 0.5]:
            np.save(self.root / "bad.npy", bad)
            with self.assertRaises(ValueError):
                bank.load_mask(self.root / "bad.npy")
        with self.assertRaises(ValueError):
            bank.validate_commands(np.zeros(2499), self.mask, 1)
        commands = np.zeros((1, 2500))
        commands[0, 100] = np.inf
        with self.assertRaises(ValueError):
            bank.validate_commands(commands, self.mask, 1)

    def test_generate_cli_records_seed_without_hardware(self):
        """The complete generation command saves provenance and generates the last row."""
        output = self.root / "generated"
        bank.main(["generate", "--output", str(output), "--mask", str(self.root / "mask.npy"),
                   "--samples", "2", "--sets", "1", "--seed", "42", "--units", "um_surface",
                   "--ptv-max", "0.1"])
        values = np.fromfile(output / "modeval_dataset_0.bin", dtype="<f4").reshape(2, 2500)
        self.assertTrue(np.any(values[-1] != 0))
        self.assertEqual(json.loads((output / "generation.json").read_text())["seed"], 42)
        self.assertNotIn("magaox_interface", bank.__dict__)


if __name__ == "__main__":
    unittest.main()
