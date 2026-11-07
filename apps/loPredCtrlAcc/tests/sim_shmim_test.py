r"""\file sim_shmim_test.py
\brief Verify create-only simulation transport contracts using temporary-file views; no real shmim access.
"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from sim_shmim import SimImage, SimImages


def fake_runtime(root):
    """Mimic only the image API for contract tests; no physical simulator or controller is duplicated."""
    records = {}

    class Owner:
        """Temporary-file identity plus shared test storage."""
        def create(self, name, data):
            """Create a fresh object without any native shared-memory library."""
            self.name = name
            (root / (name + ".im.shm")).touch(exist_ok=False)
            self.md = SimpleNamespace(size=[data.shape[0], 1], cnt0=0, write=0)
            records[name] = SimpleNamespace(md=self.md, data=data.copy())
            return 0

        def destroy(self):
            """Remove only the fake object created by this owner."""
            (root / (self.name + ".im.shm")).unlink()
            del records[self.name]

    class Image:
        """Separate fake views, matching magaox.Image's squeezed copy boundary."""
        _milk_shm_prefix = root

        def __init__(self, name):
            """Open test storage only."""
            self.record = records[name]
            self.md = self.record.md

        def write(self, value):
            """Copy and publish one complete object."""
            self.record.data = value.copy()
            self.md.cnt0 += 1

        def get_data(self, wait=False):
            """Return a squeezed copy without waiting or updating a producer."""
            assert not wait
            return self.record.data.T.squeeze().copy()

        def close(self):
            """The fake view has no native resource to release."""

    return SimpleNamespace(Image=Owner), Image, records


class SimShmimTest(unittest.TestCase):
    """Test isolated naming, counter packets, exact word preservation and owned cleanup."""
    def test_packet_roundtrip_and_cleanup(self):
        """Channels 0/1, source timestamps and consumed input survive distinct writer/reader views."""
        with tempfile.TemporaryDirectory() as directory:
            backend, image, records = fake_runtime(Path(directory))
            bank = SimImages(backend, image)
            raw = np.array([[2048, 2000], [2049, 2001], [2050, 2002], [2051, 2003]], np.uint16)
            pointing = np.arange(8, dtype=np.float64).reshape(4, 2)
            adc, d = bank.source_frame(raw, pointing, 12, 1000000000)
            np.testing.assert_array_equal(adc, raw)
            np.testing.assert_array_equal(d, pointing[-1].astype(np.float32))
            tag = bank.images["source_tag"].get_data().ravel()
            self.assertEqual(tag[:5].tolist(), [1, 51, 1000375000, 1000376000, 12])
            u = np.array([.25, -.5], np.float32)
            bank.images["wfs"].write(np.zeros(2, np.float32))
            np.testing.assert_array_equal(bank.consumed(u, np.zeros(2, np.float32), 12, 1000500000), u)
            with self.assertRaises(ValueError):
                bank.images["adc01"].write(raw[-1].astype(np.float32))
            bank.close()
            bank.close()
            self.assertFalse(records)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_refuse_existing_or_non_simulation_image(self):
        """Creation must not attach, replace or erase a pre-existing image, even with a simulation prefix."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend, image, records = fake_runtime(root)
            existing = root / "ddspc_sim_existing.im.shm"
            existing.write_text("preserve")
            with self.assertRaises(FileExistsError):
                SimImage("ddspc_sim_existing", 2, np.float32, backend, image)
            with self.assertRaises(ValueError):
                SimImage("dm01disp10", 2, np.float32, backend, image)
            self.assertEqual(existing.read_text(), "preserve")
            self.assertFalse(records)

    def test_incomplete_publication_is_not_consumed(self):
        """Counter equality alone is insufficient while ImageStreamIO's producer write flag is set."""
        with tempfile.TemporaryDirectory() as directory:
            backend, image, _ = fake_runtime(Path(directory))
            view = SimImage("ddspc_sim_busy", 2, np.float32, backend, image)
            view.reader.md.write = 1
            with self.assertRaises(RuntimeError):
                view.get_data()
            view.reader.md.write = 0
            view.close()

    def test_replacement_is_not_reopened_or_destroyed(self):
        """A replaced identity fails closed and its replacement remains untouched during cleanup."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend, image, _ = fake_runtime(root)
            view = SimImage("ddspc_sim_replaced", 2, np.float32, backend, image)
            replacement = root / "replacement"
            replacement.write_text("other owner")
            replacement.replace(view.path)
            with self.assertRaises(RuntimeError):
                view.get_data()
            view.close()
            self.assertEqual(view.path.read_text(), "other owner")


if __name__ == "__main__":
    unittest.main()
