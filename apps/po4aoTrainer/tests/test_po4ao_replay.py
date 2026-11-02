## \file test_po4ao_replay.py
# \brief Regression tests ported with the numerical implementation.
"""CPU-only checks for time-based historical replay and training mixtures."""

import unittest
from unittest.mock import patch

import numpy as np

from xapp.po4aoTrainer.po4ao_replay import (
    HistoricalReplay,
    ridge_replay_pair,
    sample_replay_pair,
)


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

    def test_ridge_mixes_whole_training_episodes(self):
        """Keep the requested 60/20/20 mix when three sources have enough episodes."""
        for index in range(3):
            self.now = index * 6
            self.observe(index + 3)
        recent = (np.full((36, 1), 4), np.full((36, 1), -4))
        warmup = (np.full((36, 1), 5), np.full((36, 1), -5))
        observations, actions = ridge_replay_pair(
            recent,
            warmup,
            episode_length=4,
            warmup_fraction=0.2,
            historical=self.history,
            historical_fraction=0.2,
        )
        self.assertEqual(observations.shape, (60, 1))
        self.assertEqual(int((observations == 4).sum()), 36)
        self.assertEqual(int((observations == 5).sum()), 12)
        self.assertEqual(int((observations == 2).sum()), 12)
        self.assertEqual(actions.shape, observations.shape)
        self.now = 132
        observations, _ = ridge_replay_pair(
            recent,
            warmup,
            episode_length=4,
            warmup_fraction=0.2,
            historical=self.history,
            historical_fraction=0.2,
        )
        self.assertEqual(observations.shape, (48, 1))
        self.assertFalse((observations == 2).any())

    def test_invalid_mixture_is_rejected(self):
        """Always leave some weight for current conditions."""
        pair = (self.observations, self.actions)
        with self.assertRaises(ValueError):
            sample_replay_pair(pair, pair, warmup_fraction=0.8, historical_fraction=0.2)


if __name__ == "__main__":
    unittest.main()
