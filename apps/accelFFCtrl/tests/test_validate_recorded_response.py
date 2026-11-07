r"""\file test_validate_recorded_response.py
\brief Exact aoSim numerical-method parity and existing-shadow response sign, timing, clipping and training selection.
\ingroup accelFFCtrl_unit_test
"""

import ast
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from validate_real_harmonic import HORIZON
from validate_recorded_response import choose_gain, metrics, response


def fixture(forecast, reference):
    """Create explicit numerical forecast snapshots; no physical predictor or instrument calibration is inferred."""
    n = np.arange(len(forecast))
    query = 1_000_000_000 + n * 500000
    return pd.DataFrame(
        dict(
            query_ns=query,
            sample_ns=query - 100000,
            target_ns=query + HORIZON,
            counter=n + 1,
            generation=1,
            valid=1,
            tip=forecast[:, 0],
            tilt=forecast[:, 1],
            reference_tip=reference[:, 0],
            reference_tilt=reference[:, 1],
        )
    )


def aosim_numerical_response(commands, reference, lag):
    """Execute the actual two aoSim methods in host memory, excluding all imports/setup/streams by AST extraction."""
    path = Path(__file__).resolve().parents[2] / "aoSim/xapp/aoSim/__init__.py"
    source = ast.parse(path.read_text())
    app = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "aoSim")
    methods = [
        node for node in app.body if isinstance(node, ast.FunctionDef) and node.name in ("update_dm", "update_wfs")
    ]
    if len(methods) != 2:
        raise ValueError("aoSim numerical methods changed")
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(path), "exec"), namespace)
    state = SimpleNamespace(
        _dm_command_history=np.zeros((lag + 1, 2, 1), dtype=np.float32),
        _noise=0,
        _nmodes=2,
        _wfs=SimpleNamespace(write=lambda value: None),
    )
    effects, residuals = [], []
    for command, disturbance in zip(commands, reference):
        state._dm = SimpleNamespace(get_data=lambda wait=False: command[:, None])
        state._current_disturbance = disturbance.astype(np.float32)[:, None]
        namespace["update_dm"](state)
        namespace["update_wfs"](state)
        effects.append(state._current_dm_state[:, 0].copy())
        residuals.append(state.err[:, 0].copy())
    return np.array(effects), np.array(residuals)


class TrainingGainTest(unittest.TestCase):
    """Require the predeclared RMS/outside-band guard when selecting only training gain rows."""

    def test_outside_band_guard(self):
        """A lower training RMS cannot override excessive outside-band power, even on only one axis."""
        rows = pd.DataFrame(
            dict(
                gain=np.repeat([0, 0.5, 1], 2),
                nrmse=[1, 1, 0.95, 0.94, 0.8, 0.8],
                outside_power_change_percent=[0, 0, 1, 2, 0, 5.1],
            )
        )
        self.assertEqual(choose_gain(rows), 0.5)
        rows.loc[rows.gain == 0.5, "nrmse"] = 1.02
        self.assertEqual(choose_gain(rows), 0)

    def test_outside_band_power(self):
        """Removing 28 Hz while doubling an 80-Hz amplitude must report fourfold outside-band power."""
        n = np.arange(32000)
        time = n / 2000
        target, unrelated = np.sin(2 * np.pi * 28 * time), 0.5 * np.sin(2 * np.pi * 80 * time)
        reference = np.repeat((target + unrelated)[:, None], 2, axis=1)
        effect = np.repeat((-target + unrelated)[:, None], 2, axis=1)
        rows, _ = metrics(reference, effect, 1000000000 + n * 500000, 1000000000, "analytical")
        for row in rows:
            self.assertAlmostEqual(row["outside_power_change_percent"], 300, places=8)
            self.assertGreater(row["band0_suppression_db"], 200)


@unittest.skipUnless(
    os.environ.get("ACCELFFCTRL_RESPONSE_REPLAY"), "build response-tools for the existing C++ response"
)
class RecordedResponseTest(unittest.TestCase):
    """Check the file bridge against actual aoSim equations and analytical sinusoid/step responses."""

    def setUp(self):
        """Own all fixture inputs/outputs in a temporary directory, independent of shared memory."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.executable = Path(os.environ["ACCELFFCTRL_RESPONSE_REPLAY"])

    def run_response(self, frame, gain=1, steps=2, bandwidth=0, sign=1):
        """Pass declared numerical forecasts through the real C++ shadow and plant, retaining exact timestamps."""
        source = self.root / "forecasts.csv"
        frame.to_csv(source, index=False)
        return response(self.executable, source, self.root / "response.csv", [1, 1], gain, steps, bandwidth, sign)

    def test_actual_aosim_delay_and_additive_sign(self):
        """aoSim lag+1 matches the C++ ideal-delay model, including initial zeros and positive WFS addition."""
        command = np.arange(24).reshape(12, 2) / 32
        reference = np.column_stack((np.ones(12) / 8, -np.ones(12) / 4))
        frame = fixture(-command, reference)
        for lag in (0, 1, 2):
            actual = self.run_response(frame, steps=lag + 1)
            effect, residual = aosim_numerical_response(command, reference, lag)
            np.testing.assert_array_equal(actual[["effect_tip", "effect_tilt"]], effect)
            np.testing.assert_array_equal(actual[["residual_tip", "residual_tilt"]], residual)

    def test_sinusoid_target_delay_wrong_sign_and_future_exclusion(self):
        """A 1-ms-ahead sinusoid cancels after two steps; reversed response doubles it and future rows cannot affect the past."""
        time = np.arange(4000) / 2000
        reference = np.column_stack((0.4 * np.cos(2 * np.pi * 28.5 * time), 0.3 * np.sin(2 * np.pi * 56 * time)))
        future = np.column_stack(
            (0.4 * np.cos(2 * np.pi * 28.5 * (time + 0.001)), 0.3 * np.sin(2 * np.pi * 56 * (time + 0.001)))
        )
        frame = fixture(future, reference)
        actual = self.run_response(frame)
        np.testing.assert_allclose(actual[["residual_tip", "residual_tilt"]].to_numpy()[2:], 0, atol=5e-14)
        reversed_sign = self.run_response(frame, sign=-1)
        np.testing.assert_allclose(
            reversed_sign[["residual_tip", "residual_tilt"]].to_numpy()[2:], 2 * reference[2:], atol=5e-14
        )
        relabeled = frame.copy()
        relabeled[["reference_tip", "reference_tilt"]] += 100
        additive = self.run_response(relabeled)
        pd.testing.assert_frame_equal(
            actual[["mode0", "mode1", "effect_tip", "effect_tilt"]],
            additive[["mode0", "mode1", "effect_tip", "effect_tilt"]],
            check_exact=True,
        )
        changed = frame.copy()
        changed.loc[2500:, ["tip", "tilt", "reference_tip", "reference_tilt"]] = 100
        mutated = self.run_response(changed)
        pd.testing.assert_frame_equal(actual.iloc[:2500], mutated.iloc[:2500], check_exact=True)

    def test_bounds_zero_gain_and_integrity_inhibition(self):
        """Common clipping preserves direction; stale/future/wrong-target/invalid/source-change rows cannot issue held commands."""
        frame = fixture(np.tile([1.0, -8.0], (1000, 1)), np.zeros((1000, 2)))
        zero = self.run_response(frame, gain=0)
        np.testing.assert_array_equal(zero[["mode0", "mode1"]], np.zeros((1000, 2)))
        frame.loc[500, "sample_ns"] = frame.query_ns.iloc[500] - 3000000
        frame.loc[501, "sample_ns"] = frame.query_ns.iloc[501] + 1
        frame.loc[502, "target_ns"] += 1
        frame.loc[503, "valid"] = 0
        frame.loc[600:, "generation"] = 2
        actual = self.run_response(frame)
        np.testing.assert_array_equal(actual.loc[:499, ["mode0", "mode1"]], np.tile([-0.375, 3], (500, 1)))
        np.testing.assert_array_equal(actual.loc[500:503, ["mode0", "mode1"]], np.zeros((4, 2)))
        np.testing.assert_array_equal(actual.loc[600:, ["mode0", "mode1"]], np.zeros((400, 2)))
        self.assertTrue(np.all(actual.loc[:499, "state"] == 3))

    def test_existing_finite_bandwidth_step_response(self):
        """The existing causal pole evolves the previous interval; its extra sample is not silently omitted."""
        frame = fixture(np.tile([0.25, -0.5], (100, 1)), np.zeros((100, 2)))
        actual = self.run_response(frame, steps=1, bandwidth=300)
        pole = np.exp(-2 * np.pi * 300 * 0.0005)
        response_factor = np.zeros(100)
        response_factor[2:] = 1 - pole ** np.arange(1, 99)
        expected = response_factor[:, None] * np.array([-0.25, 0.5])
        np.testing.assert_allclose(actual[["effect_tip", "effect_tilt"]], expected, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
