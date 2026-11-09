## \file test_training.py
# \brief Numerical parity, replay, policy references and persistent optimization.
# \ingroup po4aoTrainer_unit_test
"""Offline regressions using synthetic data and explicit instrument boundaries."""

from copy import deepcopy
from dataclasses import replace
import json
import threading
import numpy as np
import pytest
import torch
import unittest
from pathlib import Path
from xapp.po4aoTrainer.training import TrainingSession
from xapp.po4aoTrainer.po4ao_runtime import ReplaySnapshot
from unittest.mock import patch
from xapp.po4aoTrainer.po4ao_replay import HistoricalReplay, sample_replay_pair
from reference_models import LinearLocallyConnectedPolicy
from xapp.po4aoTrainer.po4ao_nonlinear_models import NonlinearModalModel
from xapp.po4aoTrainer.po4ao_policy_update import (
    PolicyUpdatePenalty,
    policy_history_batch,
)


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def inputs():
    """Use fixed synthetic input with extra stream modes beyond the trained prefix."""
    config = json.loads((FIXTURES / "session_config.json").read_text())
    rng = np.random.default_rng(20261102)
    recent, warmup = [
        ReplaySnapshot(
            rng.normal(size=(6, 96, 2)).astype(np.float32), 1, 96, 3, "0000000000000000"
        )
        for _ in range(2)
    ]
    return config, recent, warmup


def make_session(config):
    """Seed initialization and consume the notebook's explicit identity-probe input."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    session = TrainingSession(config)
    probe = torch.randn(2, 5, 1, 4)
    torch.testing.assert_close(session.policy(probe), probe[:, 0, 0])
    return session


def test_matches_pinned_notebook_iteration(inputs):
    """Match all parameters, Adam moments and numerical diagnostic fields to the source run."""
    config, recent, warmup = inputs
    session = make_session(config)
    result = session.step(recent, warmup, active_model_id="0")
    with np.load(FIXTURES / "notebook_iteration.npz", allow_pickle=False) as reference:
        for name, model in (("policy", session.policy), ("dynamics", session.dynamics)):
            for key, value in model.state_dict().items():
                np.testing.assert_allclose(
                    value.detach().numpy(),
                    reference[f"{name}.{key}"],
                    rtol=1e-5,
                    atol=1e-7,
                )
        for name, optimizer in (
            ("policy", session.policy_optimizer),
            ("dynamics", session.dynamics_optimizer),
        ):
            for index, state in optimizer.state_dict()["state"].items():
                for key, value in state.items():
                    np.testing.assert_allclose(
                        value.numpy(),
                        reference[f"{name}.adam.{index}.{key}"],
                        rtol=1e-5,
                        atol=1e-7,
                    )
        for key in (
            "dynamics_loss",
            "policy_loss",
            "policy_validation_loss",
            "fallback_policy_validation_loss",
            "candidate_cost_ratio",
            "candidate_action_rms_ratio",
            "rollout_loss",
            "dynamics_per_mode_nrmse",
            "measured_cost_per_step",
        ):
            np.testing.assert_allclose(
                result["record"][key], reference[key], rtol=1e-5, atol=1e-7
            )
        assert result["record"]["publication_status"] == str(
            reference["publication_status"]
        )
        assert result["record"]["publication_reasons"] == json.loads(
            str(reference["publication_reasons"])
        )


def test_cached_replay_keeps_training_without_recopies_or_reinitialization(
    inputs, monkeypatch
):
    """Partial data waits; cached data advances Adam without recopying or refreshing history."""
    config, recent, warmup = inputs
    session = make_session(config)
    assert (
        session.step(replace(recent, valid_length=0), warmup, active_model_id="0")
        is None
    )
    assert session.iteration == 0 and session.normalization_metadata is None
    session.step(recent, warmup, active_model_id="0")
    optimizer = session.policy_optimizer
    steps = {key: value["step"].clone() for key, value in optimizer.state.items()}
    parameters = deepcopy(session.policy.state_dict())
    history_count = session.historical.metrics()["historical_episodes"]
    original_pair = session._replay_pair
    copied = []

    def copy_pair(snapshot):
        """Record expensive array conversion while retaining the real validation."""
        copied.append(snapshot)
        return original_pair(snapshot)

    monkeypatch.setattr(session, "_replay_pair", copy_pair)
    assert session.step(recent, warmup, active_model_id="0") is not None
    assert session.iteration == 2 and not copied
    assert session.historical.metrics()["historical_episodes"] == history_count
    assert any(
        not torch.equal(value, session.policy.state_dict()[key])
        for key, value in parameters.items()
    )
    for key, value in steps.items():
        assert (
            optimizer.state[key]["step"]
            == value + config["training"]["policy_grad_steps"]
        )
    new_replay = replace(recent, counter=2, episode_id=4)
    session.step(new_replay, warmup, active_model_id="0")
    assert len(copied) == 1 and copied[0] is new_replay
    assert session.policy_optimizer is optimizer
    assert session.iteration == 3
    assert (
        session.last_result["record"]["dynamics_gradient_steps"]
        == config["training"]["dynamics_grad_steps"]
    )


def test_recollection_preserves_normalization_and_policy(inputs):
    """Invalidate history and anchors immediately, then rebuild on complete fresh warmup."""
    config, recent, warmup = inputs
    session = make_session(config)
    session.step(recent, warmup, active_model_id="0")
    parameters = deepcopy(session.policy.state_dict())
    normalization = deepcopy(session.normalization_metadata)
    session.invalidate_replay()
    assert session.policy_update.anchors is None
    assert session.historical.metrics()["historical_episodes"] == 0
    assert session.step(replace(recent, counter=2), warmup, active_model_id="0") is None
    assert (
        session.step(
            replace(recent, counter=2),
            replace(warmup, counter=2, valid_length=1),
            active_model_id="0",
        )
        is None
    )
    for key, value in parameters.items():
        torch.testing.assert_close(
            value, session.policy.state_dict()[key], rtol=0, atol=0
        )
    session.step(
        replace(recent, counter=2, episode_id=4),
        replace(warmup, counter=2, data=warmup.data * 2),
        active_model_id="0",
    )
    assert session.normalization_metadata == normalization
    assert session.iteration == 2


def test_pause_resumes_same_in_progress_update(inputs):
    """A cooperative pause between optimizer steps resumes bit-identically to uninterrupted work."""
    config, recent, warmup = inputs
    reference = make_session(config)
    reference.step(recent, warmup, active_model_id="0")
    session = make_session(config)
    paused, resume = threading.Event(), threading.Event()
    calls = 0
    errors = []

    def checkpoint():
        """Block after the first dynamics update, before the next optimizer step."""
        nonlocal calls
        calls += 1
        if calls == 3:
            paused.set()
            assert resume.wait(5), "Test failed to resume cooperative checkpoint"

    def run():
        """Capture worker failures so the main test cannot silently pass."""
        try:
            session.step(recent, warmup, checkpoint=checkpoint, active_model_id="0")
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert paused.wait(5)
        assert session.iteration == 0
        assert session.dynamics_optimizer.state
        assert not session.policy_optimizer.state
    finally:
        resume.set()
        worker.join(10)
    assert not worker.is_alive() and not errors
    for model in ("policy", "dynamics"):
        for key, value in getattr(reference, model).state_dict().items():
            torch.testing.assert_close(
                value, getattr(session, model).state_dict()[key], rtol=0, atol=0
            )


def test_publication_waits_for_acknowledgement(inputs):
    """Staging preserves the active reference until a matching runtime ID is observed."""
    config, recent, warmup = inputs
    session = make_session(config)
    session.step(recent, warmup, active_model_id="0")
    assert session.last_result["record"]["publication_status"] == "ready"
    session.stage_publication("1234")
    assert session.policy_update.reference_id == "0000000000000000"
    with pytest.raises(RuntimeError):
        session.stage_publication("5678")
    session.step(
        replace(recent, counter=2, episode_id=4), warmup, active_model_id="1234"
    )
    assert session.policy_update.reference_id == "0000000000001234"
    assert session.policy_update.pending is None


def test_configuration_is_a_snapshot_and_bad_data_cannot_publish(inputs):
    """Caller edits and corrupt replay cannot overwrite the effective settings or train."""
    config, recent, warmup = inputs
    session = make_session(config)
    config["training"]["policy_learning_rate"] = 5
    copy = session.config
    copy["training"]["policy_learning_rate"] = 10
    assert session.config["training"]["policy_learning_rate"] == 0.001
    recent.data[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        session.step(recent, warmup, active_model_id="0")
    assert session.iteration == 0
    with pytest.raises(RuntimeError):
        session.stage_publication("1234")


def test_interruption_invalidates_previous_candidate(inputs):
    """Stopping during an update cannot leave the preceding iteration eligible for export."""
    config, recent, warmup = inputs
    session = make_session(config)
    session.step(recent, warmup, active_model_id="0")

    def stop():
        """Represent the managed worker's terminal cooperative stop request."""
        raise InterruptedError("stop")

    with pytest.raises(InterruptedError):
        session.step(
            replace(recent, counter=2), warmup, checkpoint=stop, active_model_id="0"
        )
    assert session.state == "failed"
    with pytest.raises(RuntimeError):
        session.stage_publication("1234")
    with pytest.raises(RuntimeError):
        session.step(recent, warmup, active_model_id="0")


class HistoricalReplayTest(unittest.TestCase):
    """Use a fake monotonic clock; no waits, GPU, files or instrument streams."""

    def setUp(self):
        """Make three identifiable four-frame episodes and a controllable clock."""
        self.now = 0.0
        clock = patch(
            "xapp.po4aoTrainer.po4ao_replay.time.monotonic",
            side_effect=lambda: self.now,
        )
        clock.start()
        self.addCleanup(clock.stop)
        self.observations = np.repeat([1.0, 2.0, 99.0], 4).reshape(-1, 1)
        self.actions = -self.observations
        self.history = HistoricalReplay(4, seconds=120, max_episodes=20)

    def observe(self, episode_id, history=None):
        """Offer a snapshot with its latest episode reserved for validation."""
        return (self.history if history is None else history).observe(
            self.observations,
            self.actions,
            episode_id=episode_id,
        )

    def test_copies_only_training_episode_and_keeps_frame_order(self):
        """Exclude the validation tail and detach retained data from its snapshot."""
        self.assertTrue(self.observe(3))
        self.observations[:] = 1000
        observation, action = self.history.sample_pair()
        np.testing.assert_array_equal(observation, np.full((4, 1), 2))
        np.testing.assert_array_equal(action, np.full((4, 1), -2))
        self.assertEqual(observation.dtype, np.float32)
        self.assertFalse(observation.flags.writeable)

    def test_spacing_and_duplicates_do_not_refresh_age(self):
        """A thousand training reads do not turn one episode into new experience."""
        self.observe(3)
        for _ in range(1000):
            self.assertFalse(self.observe(3))
        self.now = 5
        self.assertFalse(self.observe(4))
        self.now = 6
        self.assertFalse(self.observe(4))
        self.assertTrue(self.observe(5))
        self.now = 119
        self.assertEqual(self.history.metrics()["historical_episodes"], 2)
        self.now = 120
        self.assertEqual(self.history.metrics()["historical_episodes"], 1)
        self.now = 126
        self.assertIsNone(self.history.sample_pair())
        self.assertFalse(self.observe(5))

    def test_expiry_is_identical_for_fast_and_slow_training(self):
        """Elapsed time alone determines what remains, including stopped producers."""
        fast = HistoricalReplay(4, seconds=120, max_episodes=20)
        slow = HistoricalReplay(4, seconds=120, max_episodes=20)
        self.observe(3, fast)
        self.observe(3, slow)
        for second in range(120):
            self.now = second
            fast.sample_pair()
        self.now = 121
        self.assertIsNone(fast.sample_pair())
        self.assertIsNone(slow.sample_pair())

    def test_capacity_and_whole_episode_concatenation(self):
        """Old data is replaced gradually and batches retain intact episode blocks."""
        for index in range(25):
            self.now = index * 6
            self.observations[4:8] = index
            self.observe(index + 3)
        observations, actions = self.history.pair(3)
        self.assertEqual(self.history.metrics()["historical_episodes"], 20)
        self.assertEqual(observations.shape, (12, 1))
        np.testing.assert_array_equal(observations[:, 0], np.repeat([5, 14, 24], 4))
        self.assertEqual(actions.shape, observations.shape)

    def test_restart_clear_and_episode_counter_regression(self):
        """Recollection or a producer restart cannot mix old and new settings."""
        self.observe(30)
        revision = self.history.revision
        self.now = 1
        self.assertTrue(self.observe(3))
        self.assertEqual(self.history.metrics()["historical_episodes"], 1)
        self.assertGreater(self.history.revision, revision)
        self.history.clear()
        self.assertIsNone(self.history.sample_pair())
        self.assertTrue(self.observe(3))

    def test_expiry_changes_ridge_revision_without_a_new_snapshot(self):
        """An unchanged replay counter does not leave expired history in the fit."""
        self.observe(3)
        revision = self.history.revision
        self.now = 120
        self.assertGreater(self.history.revision, revision)
        self.assertIsNone(self.history.pair(3))

    def test_rejects_invalid_settings_and_data(self):
        """Fail before silently storing partial, malformed or nonfinite episodes."""
        for settings in [dict(seconds=0), dict(seconds=np.nan), dict(max_episodes=0)]:
            with self.assertRaises(ValueError):
                HistoricalReplay(4, **settings)
        with self.assertRaises(ValueError):
            self.history.observe(
                self.observations[:-1], self.actions[:-1], episode_id=3
            )
        with self.assertRaises(ValueError):
            self.history.observe(
                self.observations, self.actions, episode_id=3, validation_episodes=0
            )
        self.observations[4] = np.nan
        with self.assertRaises(ValueError):
            self.observe(3)

    def test_sampler_mixture_and_expired_fallback(self):
        """History supplies 20% of draws when populated, recent data otherwise."""
        self.observe(3)
        recent = (np.full((8, 1), 4), np.full((8, 1), -4))
        warmup = (np.full((8, 1), 5), np.full((8, 1), -5))
        for draw, expected in [(0.1, 5), (0.3, 2), (0.5, 4)]:
            with patch(
                "xapp.po4aoTrainer.po4ao_replay.np.random.random", return_value=draw
            ):
                pair = sample_replay_pair(
                    recent,
                    warmup,
                    warmup_fraction=0.2,
                    historical=self.history,
                    historical_fraction=0.2,
                )
                self.assertTrue((pair[0] == expected).all())
        self.now = 120
        with patch("xapp.po4aoTrainer.po4ao_replay.np.random.random", return_value=0.3):
            self.assertIs(
                sample_replay_pair(
                    recent,
                    warmup,
                    warmup_fraction=0.2,
                    historical=self.history,
                    historical_fraction=0.2,
                ),
                recent,
            )

    def test_invalid_mixture_is_rejected(self):
        """Always leave some weight for current conditions."""
        pair = (self.observations, self.actions)
        with self.assertRaises(ValueError):
            sample_replay_pair(pair, pair, warmup_fraction=0.8, historical_fraction=0.2)


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
