## \file test_po4ao_diagnostics.py
# \brief Regression tests ported with the numerical implementation.
"""Tests for modal PO4AO replay diagnostics."""

import importlib.util
import unittest

import numpy as np

from xapp.po4aoTrainer.po4ao_diagnostics import (
    _counter_rates,
    _dynamics_raw_weights,
    _plot_coefficient_map,
    estimate_delay_frames,
    evaluate_dynamics_rollout_details,
    lag_correlation,
    measured_replay_cost,
    modal_scale_ratios,
    read_runtime_telemetry,
    replay_buffer_health,
    replay_metrics,
)


class ReplayDiagnosticsTest(unittest.TestCase):
    """Validate delay and replay-health calculations."""

    def test_estimates_integer_frame_delay(self):
        """Verify estimates integer frame delay."""
        generator = np.random.default_rng(4)
        actions = generator.normal(size=(500, 3)).astype(np.float32)
        observations = np.zeros_like(actions)
        observations[3:] = -actions[:-3]

        delay, scores = estimate_delay_frames(observations, actions, max_lag=8)

        self.assertEqual(delay, 3)
        self.assertEqual(scores.shape, (9,))

    def test_reports_nonfinite_and_saturated_samples(self):
        """Verify reports nonfinite and saturated samples."""
        observations = np.array([[0.0, np.nan], [1.0, 0.0]], dtype=np.float32)
        actions = np.array([[0.0, 1.0], [np.inf, -1.0]], dtype=np.float32)

        metrics = replay_metrics(observations, actions, 1.0, delay_max_lag=1)

        self.assertEqual(metrics["nonfinite_samples"], 2)
        self.assertAlmostEqual(metrics["command_clip_fraction"], 2 / 3)
        self.assertTrue(np.isfinite(metrics["observation_rms"]))
        self.assertTrue(np.isfinite(metrics["action_rms"]))

    def test_reads_runtime_indi_values(self):
        """Verify reads runtime indi values."""
        values = {
            "po4ao.runtime_stats.p50_us": 98,
            "po4ao.runtime_stats.p95_us": 115,
            "po4ao.runtime_stats.p99_us": 121,
            "po4ao.runtime_stats.max_us": 466,
            "po4ao.runtime_stats.missed_frames": 0,
            "po4ao.runtime_stats.deadline_misses": 2,
            "po4ao.runtime_stats.inference_failures": 0,
            "po4ao.runtime_stats.replay_drops": 0,
            "po4ao.model_status.active_id": "1cf7778a58831a50",
        }

        telemetry = read_runtime_telemetry(values)

        self.assertEqual(telemetry["runtime_p99_us"], 121.0)
        self.assertEqual(telemetry["runtime_active_model_id"], "1cf7778a58831a50")

    def test_probe_delay_uses_command_minus_baseline(self):
        """Verify probe delay uses command minus baseline."""
        generator = np.random.default_rng(8)
        probe = generator.normal(size=(1000, 2)).astype(np.float32)
        observations = np.zeros_like(probe)
        observations[3:] = -probe[:-3]
        actions = observations + probe

        health = replay_buffer_health(
            observations,
            actions,
            baseline_gain=1.0,
            max_lag=6,
        )

        self.assertEqual(health["response_lag_frames"], 3)
        self.assertEqual(health["control_delay_frames"], 2)
        self.assertAlmostEqual(health["probe_rms"], np.sqrt(np.mean(probe**2)))

        metrics = replay_metrics(
            observations,
            actions,
            10.0,
            delay_max_lag=6,
            baseline_gain=1.0,
        )
        self.assertEqual(metrics["estimated_response_lag_frames"], 3)
        self.assertEqual(metrics["estimated_delay_frames"], 2)

    def test_reports_repeated_replay_frames(self):
        """Verify reports repeated replay frames."""
        observations = np.array(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [2.0, 3.0]],
            dtype=np.float32,
        )
        actions = observations.copy()

        health = replay_buffer_health(observations, actions, max_lag=2)

        self.assertAlmostEqual(health["repeated_observation_fraction"], 2 / 3)
        self.assertEqual(health["longest_repeated_observation_run"], 3)
        self.assertEqual(health["longest_repeated_action_run"], 3)

    def test_lag_correlation_preserves_response_sign(self):
        """Verify lag correlation preserves response sign."""
        reference = np.arange(10, dtype=np.float64)[:, None]
        response = -reference

        correlations = lag_correlation(reference, response, max_lag=0)

        self.assertAlmostEqual(correlations[0, 0], -1.0)

    def test_assumed_delay_skips_correlation_measurement(self):
        """Verify assumed delay skips correlation measurement."""
        observations = np.ones((8, 1), dtype=np.float32)
        actions = np.ones_like(observations)

        metrics = replay_metrics(
            observations,
            actions,
            10.0,
            delay_max_lag=100,
            assumed_control_delay=0,
        )

        self.assertEqual(metrics["estimated_delay_frames"], 0)
        self.assertIsNone(metrics["estimated_response_lag_frames"])
        self.assertEqual(metrics["delay_scores"], [])
        self.assertEqual(metrics["delay_source"], "assumed")

    def test_runtime_counter_rates_handle_restarts(self):
        """Verify runtime counter rates handle restarts."""
        history = [
            {"timestamp_unix_ns": 0, "count": 10},
            {"timestamp_unix_ns": 2_000_000_000, "count": 16},
            {"timestamp_unix_ns": 4_000_000_000, "count": 2},
        ]

        np.testing.assert_array_equal(_counter_rates(history, "count"), [0, 3, 0])

    def test_measured_cost_matches_training_objective(self):
        """Verify measured cost matches training objective."""
        observations = np.full((4, 2), 2.0)
        actions = np.full((4, 2), 3.0)

        cost = measured_replay_cost(
            observations,
            actions,
            observation_scale=[2.0, 2.0],
            action_scale=[3.0, 3.0],
            loss_penalty=0.25,
            episode_length=2,
        )

        self.assertAlmostEqual(cost["observation"], 1.0)
        self.assertAlmostEqual(cost["action_penalty"], 0.25)
        self.assertAlmostEqual(cost["total"], 1.25)

    def test_modal_scale_ratios_use_latest_episode(self):
        """Verify modal scale ratios use latest episode."""
        observations = np.array([[100.0], [100.0], [2.0], [2.0]])
        actions = np.array([[100.0], [100.0], [3.0], [3.0]])

        ratios = modal_scale_ratios(
            observations,
            actions,
            observation_scale=[2.0],
            action_scale=[3.0],
            episode_length=2,
        )

        np.testing.assert_allclose(ratios["observation"], [1.0])
        np.testing.assert_allclose(ratios["action"], [1.0])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_coefficient_map_matches_legacy_dynamics_in_raw_units(self):
        """Verify coefficient map matches legacy dynamics in raw units."""
        from types import SimpleNamespace
        import matplotlib.pyplot as plt
        import torch
        from reference_models import ConvDynamics

        for n_history in (1, 3):
            model = ConvDynamics(
                torch.ones(1, 2), n_history, control_delay=min(2, n_history - 1)
            )
            model.set_modal_scales([2.0, 3.0], [4.0, 5.0])
            with torch.no_grad():
                model.net.weight.fill_(0.25)
            legacy = SimpleNamespace(
                normalizer=model.normalizer,
                net=model.net,
                n_history=model.n_history,
                control_delay=model.control_delay,
            )
            weights = _dynamics_raw_weights(SimpleNamespace(models=[legacy]))
            features = torch.arange(4 * n_history, dtype=torch.float32).reshape(
                1, 2 * n_history, 1, 2
            )
            expected = model(features).detach().numpy()[0, 0, 0]
            np.testing.assert_allclose(
                np.sum(weights * features.numpy()[0, :, 0], axis=0),
                expected,
            )
            figure, axis = plt.subplots()
            try:
                _plot_coefficient_map(axis, weights, np.arange(2 * n_history), "Test")
                np.testing.assert_array_equal(axis.images[0].get_array(), weights.T)
            finally:
                plt.close(figure)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_dynamics_rollout_details_include_trace_and_modal_error(self):
        """Verify dynamics rollout details include trace and modal error."""
        import torch

        class AdditiveDynamics(torch.nn.Module):
            """Known additive predictor for testing measured and predicted trace data."""

            def __init__(self):
                """Initialize the local test double without external resources."""
                super().__init__()
                self.models = torch.nn.ModuleList([torch.nn.Identity()])
                self.models[0].normalizer = type(
                    "Normalizer",
                    (),
                    {"observation_scale": torch.ones(1, 1, 1, 2)},
                )()

            def normalize_observation(self, values):
                """Keep unit scales so expected rollout errors are analytic."""
                return values

            def forward(self, features):
                """Return the known additive next-observation prediction."""
                return features[:, -2:-1] + features[:, -1:]

        actions = np.full((12, 2), 0.5, dtype=np.float32)
        observations = np.arange(12, dtype=np.float32)[:, None] * 0.5
        observations = np.repeat(observations, 2, axis=1)

        details = evaluate_dynamics_rollout_details(
            AdditiveDynamics(),
            observations,
            actions,
            n_history=2,
            episode_length=8,
            horizon=3,
            batch_size=4,
            device="cpu",
            mode_indices=[0, 1],
        )

        np.testing.assert_allclose(details["horizon_mse"], 0.0)
        np.testing.assert_allclose(details["per_mode_nrmse"], 0.0)
        np.testing.assert_allclose(
            details["trace_observed_normalized"],
            details["trace_predicted_normalized"],
        )


if __name__ == "__main__":
    unittest.main()
