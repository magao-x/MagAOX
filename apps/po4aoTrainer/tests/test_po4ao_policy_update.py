## \file test_po4ao_policy_update.py
# \brief Regression tests ported with the numerical implementation.
"""CPU checks for fixed quadratic policy-update losses and reference tracking."""

from copy import deepcopy
import json
import unittest

import numpy as np
import torch

from reference_models import LinearLocallyConnectedPolicy
from xapp.po4aoTrainer.po4ao_nonlinear_models import NonlinearModalModel
from xapp.po4aoTrainer.po4ao_policy_update import (
    PolicyUpdatePenalty,
    policy_history_batch,
)


class PolicyUpdateTest(unittest.TestCase):
    """Check loss gradients and diagnostics without instrument streams or GPUs."""

    def setUp(self):
        """Use reproducible small histories on CPU."""
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.features = torch.ones(16, 5, 1, 4)

    def regularizer(self, architecture="linear_local", weight=100, **kwargs):
        """Use explicit strong weights for analytic checks, independent of defaults."""
        regularizer = PolicyUpdatePenalty(
            architecture=architecture,
            action_scale=np.ones(4),
            baseline_gain=1,
            weight=weight,
            **kwargs,
        )
        regularizer.set_anchors(self.features)
        return regularizer

    def policy(self, architecture):
        """Make a linear or nonlinear policy initially identical to the baseline."""
        if architecture == "linear_local":
            return LinearLocallyConnectedPolicy(3, 4, integrator_gain=1)
        return NonlinearModalModel(3, 4, architecture=architecture)

    def perturb_mode(self, policy, architecture, mode=0, factor=1.15):
        """Change one mode's current-observation gain while preserving other modes."""
        with torch.no_grad():
            if architecture == "linear_local":
                policy.output.weight[0, 0, 0, mode] *= factor
            else:
                policy.output.weight[0, :, 0, mode] *= factor

    def test_equal_weights_match_joint_loss_without_a_bound(self):
        """Equal weights give the same mean-square loss, not a per-mode constraint."""
        for architecture in ["linear_local", "nonlinear_local", "nonlinear_global"]:
            with self.subTest(architecture=architecture):
                policy = self.policy(architecture)
                self.perturb_mode(policy, architecture)
                regularizer = self.regularizer(architecture)
                metrics = regularizer.metrics(policy, self.features)
                coupled = architecture == "nonlinear_global"
                self.assertTrue(metrics["policy_update_finite"])
                self.assertAlmostEqual(
                    float(regularizer.penalty(policy).detach()), 0.5625, places=4
                )
                self.assertNotIn("policy_update_allowed", metrics)
                self.assertAlmostEqual(
                    metrics["policy_update_max_rms"],
                    0.075 if coupled else 0.15,
                    places=4,
                )
                self.assertEqual(len(metrics["policy_update_rms"]), 1 if coupled else 4)

    def test_penalty_gradient_reduces_distance(self):
        """Distance enters backpropagation for linear, nonlinear and coupled policies."""
        for architecture in ["linear_local", "nonlinear_local", "nonlinear_global"]:
            with self.subTest(architecture=architecture):
                policy = self.policy(architecture)
                self.perturb_mode(policy, architecture, factor=1.3)
                regularizer = self.regularizer(architecture)
                optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
                before = regularizer.metrics(policy, self.features)[
                    "policy_update_max_rms"
                ]
                for _ in range(10):
                    optimizer.zero_grad()
                    regularizer.penalty(policy).backward()
                    self.assertTrue(
                        any(p.grad is not None for p in policy.parameters())
                    )
                    optimizer.step()
                after = regularizer.metrics(policy, self.features)[
                    "policy_update_max_rms"
                ]
                self.assertLess(after, before)

    def test_frozen_physical_scales_and_per_mode_weights(self):
        """Unequal scales normalize modes; local weights can emphasize selected modes."""
        scale = np.array([0.001, 0.01, 0.1, 1], dtype=np.float32)
        features = self.features * torch.from_numpy(scale)
        policy = self.policy("linear_local")
        self.perturb_mode(policy, "linear_local")
        regularizer = PolicyUpdatePenalty(
            architecture="linear_local",
            action_scale=scale,
            baseline_gain=1,
            weight=[200, 100, 100, 100],
        )
        regularizer.set_anchors(features)
        scale[:] = 999  # Constructor must not retain a mutable numpy view.
        result = regularizer.metrics(policy, features)
        self.assertTrue(result["policy_update_finite"])
        self.assertAlmostEqual(result["policy_update_max_rms"], 0.15, places=5)
        self.assertAlmostEqual(
            float(regularizer.penalty(policy).detach()), 1.125, places=4
        )

    def test_heldout_distance_is_diagnostic_not_a_bound(self):
        """Report unseen-state changes even if anchors have zero action change."""
        regularizer = self.regularizer()
        regularizer.set_anchors(torch.zeros_like(self.features))
        policy = self.policy("linear_local")
        self.perturb_mode(policy, "linear_local", factor=2)
        result = regularizer.metrics(policy, self.features)
        self.assertTrue(result["policy_update_finite"])
        self.assertEqual(result["policy_update_max_rms"], 1.0)
        self.assertEqual(result["policy_update_anchor_penalty"], 0.0)

    def test_reference_waits_for_acknowledgment_and_pending_is_immutable(self):
        """Publication itself and subsequent training cannot move the active reference."""
        regularizer = self.regularizer()
        policy = self.policy("linear_local")
        self.perturb_mode(policy, "linear_local", factor=1.05)
        exported = deepcopy(policy)
        regularizer.stage_publication(policy, "1234")
        regularizer.observe_active_model("0")
        self.assertEqual(regularizer.reference_id, "0000000000000000")
        self.assertEqual(
            regularizer.metrics(policy, self.features)["policy_update_pending_id"],
            "0000000000001234",
        )
        with self.assertRaises(RuntimeError):
            regularizer.stage_publication(policy, "5678")
        self.perturb_mode(policy, "linear_local", factor=3)
        regularizer.observe_active_model("1234")
        self.assertIsNone(regularizer.anchors)
        regularizer.set_anchors(self.features)
        torch.testing.assert_close(
            regularizer.reference_actions, exported(self.features)
        )
        self.assertTrue(
            all(
                p.grad is None and not p.requires_grad
                for p in regularizer.reference.parameters()
            )
        )
        self.assertGreater(
            regularizer.metrics(policy, self.features)["policy_update_max_rms"], 1
        )
        regularizer.observe_active_model("0")
        regularizer.set_anchors(self.features)
        torch.testing.assert_close(
            regularizer.reference_actions, self.features[:, 0, 0]
        )

    def test_unknown_model_and_nonfinite_values_fail_closed(self):
        """Do not guess previous-run weights or silently accept invalid distances."""
        regularizer = self.regularizer()
        with self.assertRaises(RuntimeError):
            regularizer.observe_active_model("5678")
        policy = self.policy("linear_local")
        with torch.no_grad():
            policy.output.weight.fill_(float("nan"))
        result = regularizer.metrics(policy, self.features)
        self.assertFalse(result["policy_update_finite"])
        self.assertIsNone(result["policy_update_max_rms"])
        json.dumps(result, allow_nan=False)

    def test_invalid_configuration(self):
        """Reject invalid scales/weights and vector weights for coupled models."""
        for changes in [
            {"weight": -1},
            {"weight": float("nan")},
            {"weight": [1, 2]},
            {"action_scale": [1, 1, 0, 1]},
            {"baseline_gain": float("inf")},
            {"architecture": "nonlinear_global", "weight": [100] * 4},
            {"architecture": "unknown"},
        ]:
            settings = dict(
                architecture="linear_local", action_scale=np.ones(4), baseline_gain=1
            )
            settings.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PolicyUpdatePenalty(**settings)

    def test_weight_scales_loss_and_zero_disables_it(self):
        """Strong fixed weights change loss scale; zero contributes no gradient."""
        policy = self.policy("linear_local")
        self.perturb_mode(policy, "linear_local", factor=1.2)
        low = self.regularizer(weight=1).penalty(policy)
        high = self.regularizer(weight=100).penalty(policy)
        torch.testing.assert_close(high, 100 * low)
        self.regularizer(weight=0).penalty(policy).backward()
        self.assertTrue(
            all(torch.count_nonzero(p.grad) == 0 for p in policy.parameters())
        )

    def test_regularizer_changes_optimum_but_has_zero_gradient_at_reference(self):
        """A quadratic toy rollout has gain (target + lambda * reference)/(1 + lambda)."""
        policy = self.policy("linear_local")
        regularizer = self.regularizer(weight=100)
        regularizer.penalty(policy).backward()
        self.assertTrue(
            all(torch.count_nonzero(p.grad) == 0 for p in policy.parameters())
        )
        policy.zero_grad()
        with torch.no_grad():
            policy.output.weight[:, 0] = 102 / 101
        objective = (policy(self.features) - 2).square().mean() + regularizer.penalty(
            policy
        )
        objective.backward()
        self.assertTrue(
            all(torch.all(p.grad.abs() < 2e-5) for p in policy.parameters())
        )

    def test_coupled_metric_aggregates_each_batch_before_taking_maximum(self):
        """Different changed modes in anchor/held-out states must not be double-counted."""
        features = torch.zeros_like(self.features)
        features[:, 0, 0, 0] = 1
        heldout = torch.zeros_like(features)
        heldout[:, 0, 0, 1] = 1
        policy = self.policy("nonlinear_global")
        self.perturb_mode(policy, "nonlinear_global", mode=0, factor=2)
        self.perturb_mode(policy, "nonlinear_global", mode=1, factor=2)
        regularizer = self.regularizer("nonlinear_global")
        regularizer.set_anchors(features)
        self.assertAlmostEqual(
            regularizer.metrics(policy, heldout)["policy_update_max_rms"], 0.5, places=5
        )

    def test_zero_publication_id_rejected_and_identical_export_keeps_reference(self):
        """Fallback is not an exported model; re-exporting identical weights is harmless."""
        regularizer = self.regularizer()
        policy = self.policy("linear_local")
        with self.assertRaises(ValueError):
            regularizer.stage_publication(policy, "0")
        regularizer.stage_publication(policy, "1234")
        regularizer.observe_active_model("1234")
        regularizer.set_anchors(self.features)
        anchors = regularizer.anchors
        regularizer.stage_publication(policy, "1234")
        self.assertIsNone(regularizer.pending)
        self.assertIs(regularizer.anchors, anchors)

    def test_history_order_and_episode_boundaries(self):
        """Anchors use C++ feature ordering and never bridge episodes or held-out slices."""
        values = np.arange(24, dtype=np.float32).reshape(-1, 1)
        features = policy_history_batch(
            values,
            values + 100,
            n_history=3,
            episode_length=8,
            samples=100,
            device="cpu",
        )[:, :, 0, 0]
        torch.testing.assert_close(features[:, 1], features[:, 0] - 2)
        torch.testing.assert_close(features[:, 2], features[:, 0] - 1)
        torch.testing.assert_close(features[:, 3:], features[:, 1:3] + 100)
        self.assertTrue(torch.all(features[:, 0] // 8 == features[:, 1] // 8))
        self.assertEqual(len(features), 18)
        with self.assertRaises(ValueError):
            policy_history_batch(
                values[:-1],
                values[:-1],
                n_history=3,
                episode_length=8,
                samples=4,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
