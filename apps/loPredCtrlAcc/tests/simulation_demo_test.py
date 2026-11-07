r"""\file simulation_demo_test.py
\brief Verify actual aoSim latency/sign, consumed-input provenance, ADC inverse and causal DDSPC boundaries offline.
"""
import ctypes as ct
import os
from pathlib import Path
import unittest

import numpy as np
from simulation_demo import APP, Prediction, aosim_methods, bridge, plant, simulate, source


class SimulationConnectionTest(unittest.TestCase):
    """Use only headless numerical implementations; no simulator startup, ImageStreamIO, INDI or hardware."""
    @classmethod
    def setUpClass(cls):
        """Open the explicit test library and read the other repository's two numerical method bodies."""
        cls.lib = bridge(Path(os.environ.get("DDSPC_SIM_LIB", APP / "build/simulation-c++/libddspc_sim.so")))
        cls.methods = aosim_methods(Path(os.environ.get("MAGAOX_REPO", Path.home() / "source/MagAOX")))

    def test_actual_aosim_delay_and_applied_input_are_distinct(self):
        """A consumed impulse is visible before the delayed WFS response; preserve both identities."""
        for lag in (0, 1):
            p = plant(self.methods, lag=lag)
            measured, consumed = [], []
            for k in range(7):
                p.update_wfs()
                measured.append(p._wfs.data[0, 0])
                p._dm.write(np.array([[1 if k == 0 else 0], [0]], np.float32))
                p.update_dm()
                consumed.append(p._dm.last_read[0, 0])
            self.assertEqual(np.flatnonzero(measured).tolist(), [lag + 2])
            self.assertEqual(np.flatnonzero(consumed).tolist(), [0])

    def test_separate_channels_cancel_with_opposite_sign(self):
        """The real aoSim methods propagate each DM input; equal opposite correction cancels its disturbance."""
        c, d, w = (plant(self.methods) for _ in range(3))
        for k in range(40):
            w._current_dm_state = c._current_dm_state + d._current_dm_state
            w.update_wfs()
            np.testing.assert_array_equal(w._wfs.data, np.zeros((2, 1), np.float32))
            disturbance = np.array([[np.sin(k)], [.5 * np.cos(k)]], np.float32)
            c._dm.write(-disturbance)
            d._dm.write(disturbance)
            c.update_dm(); d.update_dm()
            np.testing.assert_array_equal(c._dm.last_read, -d._dm.last_read)

    def test_adc_inverse_and_timing_protocol(self):
        """Decode actual count units and reject future/repeated records before accepting a consumed input."""
        lib = self.lib
        handle = lib.ddspc_sim_controller_create(1, np.array([.001, .002, 2048, .0083799, 1.02]))
        self.assertTrue(handle)
        try:
            y, u, out = np.zeros(2, np.float32), np.zeros(2, np.float32), np.ones(2, np.float32)
            raw = np.array([[2048, 2049], [2050, 2047], [2052, 2046], [2054, 2045]], np.uint16)
            info = Prediction()
            self.assertEqual(lib.ddspc_sim_accept(handle, u), -1)
            with self.assertRaises(ct.ArgumentError):
                lib.ddspc_sim_query(handle, 0, 1000000000, 999500000, y, raw.ravel(), out, ct.byref(info))
            self.assertEqual(lib.ddspc_sim_query(handle, 0, 1000000000, 999624001, y, raw, out, ct.byref(info)), -1)
            np.testing.assert_array_equal(out, np.zeros(2, np.float32))
            self.assertEqual(lib.ddspc_sim_query(handle, 0, 1000000000, 999500000, y, raw, out, ct.byref(info)), 0)
            expected = ((raw.astype(float) - 2048) * .0083799 / 1.02 / .002).mean(axis=0)
            np.testing.assert_allclose([info.ax, info.ay], expected, rtol=1e-6)
            self.assertFalse(info.valid)
            self.assertEqual(lib.ddspc_sim_accept(handle, u), 0)
            self.assertEqual(lib.ddspc_sim_accept(handle, u), -1)
            self.assertEqual(lib.ddspc_sim_query(handle, 0, 1000500000, 1000000000, y, raw, out, ct.byref(info)), -1)
        finally:
            lib.ddspc_sim_controller_destroy(handle)

    def test_source_is_deterministic_and_quantized(self):
        """Same actual producer configuration returns identical X/Y pointing/ADC pairs without clipped rails."""
        a, d, units = source(self.lib, 100, 1)
        b, e, other_units = source(self.lib, 100, 1)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(d, e)
        np.testing.assert_array_equal(units, other_units)
        self.assertEqual(a.dtype, np.uint16)
        self.assertEqual(a.shape, (100, 4, 2))
        self.assertTrue(((a > 0) & (a < 4095)).all())

    def test_channel_01_encoding_from_actual_producer(self):
        """Audit channel order against the last noisy physical sample; these audit values never enter DDSPC."""
        lib = self.lib
        handle = lib.ddspc_sim_source_create(100, 42)
        self.assertTrue(handle)
        try:
            units = np.empty(5)
            lib.ddspc_sim_source_units(handle, units)
            raw, pointing, noisy_acceleration = np.empty(2, np.uint16), np.empty(2), np.empty(2)
            different = False
            for _ in range(100):
                self.assertEqual(lib.ddspc_sim_source_step(handle, raw, pointing), 0)
                self.assertEqual(lib.ddspc_sim_source_audit(handle, noisy_acceleration), 0)
                expected = np.clip(np.floor(units[2] + noisy_acceleration * units[4] / units[3] + .5), 0, 4095)
                np.testing.assert_array_equal(raw, expected.astype(np.uint16))
                decoded = (raw.astype(float) - units[2]) * units[3] / units[4]
                self.assertTrue((np.abs(decoded - noisy_acceleration) <= .5 * units[3] / units[4]).all())
                different |= raw[0] != raw[1]
            self.assertTrue(different)
        finally:
            lib.ddspc_sim_source_destroy(handle)

    def test_channel_01_decoder_is_not_a_tip_tilt_assignment(self):
        """An offset on raw channel 0 changes only feature 0; no ideal acceleration or angular conversion is used."""
        lib = self.lib
        units = np.array([.001, .002, 2048, .0083799, 1.02])
        handle = lib.ddspc_sim_controller_create(1, units)
        try:
            raw = np.full((4, 2), 2048, np.uint16)
            raw[:, 0] += 4
            info, out = Prediction(), np.zeros(2, np.float32)
            self.assertEqual(lib.ddspc_sim_query(handle, 0, 1000000000, 999500000,
                                                np.zeros(2, np.float32), raw, out, ct.byref(info)), 0)
            self.assertAlmostEqual(info.ax, 4 * units[3] / units[4] / units[1], places=5)
            self.assertEqual(info.ay, 0)
        finally:
            lib.ddspc_sim_controller_destroy(handle)

    def test_standard_is_independent_of_accelerometer_words(self):
        """Even invalid sensor words cannot change the independently trained standard controller's correction."""
        raw, d, units = source(self.lib, 700, 1)
        a = simulate(self.lib, self.methods, raw, d, units, train=400, mode=3)
        b = simulate(self.lib, self.methods, np.zeros_like(raw), d, units, train=400, mode=3)
        for field in ("yx", "yy", "request_x", "request_y", "version", "rank", "valid"):
            np.testing.assert_array_equal(a[field], b[field])

    def test_no_future_pointing_or_adc_can_change_earlier_corrections(self):
        """Private source truth is plant-only, and only already sampled ADC/WFS enter model queries."""
        raw, d, units = source(self.lib, 700, 100)
        a = simulate(self.lib, self.methods, raw, d, units, train=400, mode=4)
        changed_raw, changed_d = raw.copy(), d.copy()
        changed_raw[550:] = 3000
        changed_d[550:] *= -2
        b = simulate(self.lib, self.methods, changed_raw, changed_d, units, train=400, mode=4)
        for field in ("yx", "yy", "request_x", "request_y", "version", "rank", "valid"):
            np.testing.assert_array_equal(a[field][:550], b[field][:550])

    def test_mock_shmim_boundaries_preserve_the_existing_loop(self):
        """The optional real-image adapter changes transport only, including ADC and consumed-input readback."""
        import tempfile
        from sim_shmim import SimImages
        from sim_shmim_test import fake_runtime
        raw, d, units = source(self.lib, 700, 100)
        expected = simulate(self.lib, self.methods, raw, d, units, train=400, mode=4)
        with tempfile.TemporaryDirectory() as directory:
            backend, image, _ = fake_runtime(Path(directory))
            images = SimImages(backend, image)
            try:
                actual = simulate(self.lib, self.methods, raw, d, units, train=400, mode=4, images=images)
            finally:
                images.close()
        np.testing.assert_array_equal(actual, expected)

    def test_calibrated_model_keeps_fresh_evidence_without_retraining(self):
        """A fixed estimator is revalidated/published against current records, never relabeled as a stale snapshot."""
        path = Path(os.environ.get("DDSPC_SIM_FIXED_LIB", APP/"build/simulation-c++-train2400/libddspc_sim.so"))
        if not path.exists():
            self.skipTest("Run simulation_diagnose.py --run to build the explicit training-end experiment")
        lib = bridge(path)
        raw, d, units = source(lib, 2800, 100)
        a = simulate(lib, self.methods, raw, d, units, mode=4)
        active = a[2400:][a["valid"][2400:] != 0]
        self.assertGreater(len(active), 300)
        np.testing.assert_array_equal(active["version"], np.full(len(active), a[2400]["version"]))
        self.assertGreater(active["model_sequence"][-1], active["model_sequence"][0])
        self.assertTrue((active["age_ns"] <= 2000000).all())


if __name__ == "__main__":
    unittest.main()
