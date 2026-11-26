## \file po4ao_replay.py
# \brief Packaged replay from the nonlinear notebook.
"""Bounded historical episodes with elapsed-time forgetting, not iteration aging.

Snapshots are already checked for coherent publication by po4ao_runtime. Only
complete training episodes are copied here; the current validation tail is never
archived. Time is measured when Python first retains an episode, not inferred
from nominal frame rate or the number of training updates.
"""

from collections import deque
import time

import numpy as np


class HistoricalReplay:
    """Keep time-spaced whole episodes in RAM and expire them by monotonic time."""

    def __init__(self, episode_length, *, seconds=120.0, max_episodes=20):
        """Retain at most one episode per seconds/max_episodes elapsed seconds."""
        if (
            not isinstance(episode_length, int)
            or episode_length < 1
            or not isinstance(max_episodes, int)
            or max_episodes < 1
            or not np.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(
                "Positive episode length, capacity and finite age are required"
            )
        self.episode_length = episode_length
        self.seconds = float(seconds)
        self.max_episodes = max_episodes
        self.interval_seconds = self.seconds / max_episodes
        self._episodes = deque(maxlen=max_episodes)
        self._last_episode_id = None
        self._last_added = float("-inf")
        self._revision = 0

    def clear(self):
        """Invalidate all historical episodes when the collection is restarted."""
        self._episodes.clear()
        self._last_episode_id = None
        self._last_added = float("-inf")
        self._revision += 1

    def _expire(self, now):
        """Remove expired data even if the producer or training was paused."""
        while self._episodes and now - self._episodes[0][0] >= self.seconds:
            self._episodes.popleft()
            self._revision += 1

    @property
    def revision(self):
        """Identify additions/expiry so ridge fitting can refresh without new data."""
        self._expire(time.monotonic())
        return self._revision

    def observe(self, observations, actions, *, episode_id, validation_episodes=1):
        """Copy the newest eligible episode once, with no histories across gaps.

        Re-reading a cached/overlapping snapshot never refreshes episode age.
        A regressing episode ID clears the archive as an additional restart guard.
        Acquisition remains snapshot-driven: episodes unavailable while Python is
        busy cannot be recovered after C++'s rolling buffer overwrites them.
        """
        now = time.monotonic()
        self._expire(now)
        if not isinstance(episode_id, int) or episode_id < 0:
            raise ValueError("episode_id must be a non-negative integer")
        if not isinstance(validation_episodes, int) or validation_episodes < 1:
            raise ValueError("At least one validation episode must be withheld")
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        if (
            observations.ndim != 2
            or observations.shape != actions.shape
            or observations.shape[1] < 1
            or len(observations) % self.episode_length
            or len(observations) <= validation_episodes * self.episode_length
        ):
            raise ValueError(
                "Matching complete episodes and a training prefix are required"
            )
        if self._last_episode_id is not None and episode_id < self._last_episode_id:
            self.clear()
        if self._last_episode_id == episode_id:
            return False
        self._last_episode_id = episode_id
        if now - self._last_added < self.interval_seconds:
            return False
        end = len(observations) - validation_episodes * self.episode_length
        start = end - self.episode_length
        observation = np.array(observations[start:end], dtype=np.float32, copy=True)
        action = np.array(actions[start:end], dtype=np.float32, copy=True)
        if not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise ValueError("Historical episodes must contain only finite data")
        if self._episodes and observation.shape != self._episodes[-1][1].shape:
            raise ValueError("Historical mode count changed; restart the training cell")
        observation.setflags(write=False)
        action.setflags(write=False)
        self._episodes.append((now, observation, action))
        self._last_added = now
        self._revision += 1
        return True

    def sample_pair(self):
        """Return one randomly chosen, unexpired whole episode or None."""
        self._expire(time.monotonic())
        if not self._episodes:
            return None
        _, observations, actions = self._episodes[
            np.random.randint(len(self._episodes))
        ]
        return observations, actions

    def pair(self, max_episodes):
        """Concatenate evenly time-spaced episodes for a batch ridge fit, or None."""
        self._expire(time.monotonic())
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        if not self._episodes:
            return None
        indices = np.linspace(
            0,
            len(self._episodes) - 1,
            min(max_episodes, len(self._episodes)),
            dtype=int,
        )
        selected = [self._episodes[index] for index in indices]
        return (
            np.concatenate([episode[1] for episode in selected]),
            np.concatenate([episode[2] for episode in selected]),
        )

    def metrics(self):
        """Report bounded occupancy and oldest first-retention age, in seconds."""
        now = time.monotonic()
        self._expire(now)
        return {
            "historical_episodes": len(self._episodes),
            "historical_oldest_seconds": (
                now - self._episodes[0][0] if self._episodes else 0.0
            ),
            "historical_max_seconds": self.seconds,
        }


def _check_fractions(warmup_fraction, historical_fraction):
    """Require non-negative fractions leaving a positive recent-data share."""
    if (
        not np.isfinite([warmup_fraction, historical_fraction]).all()
        or min(warmup_fraction, historical_fraction) < 0
        or warmup_fraction + historical_fraction >= 1
    ):
        raise ValueError(
            "Warm-up/historical fractions must be non-negative and sum to < 1"
        )


def sample_replay_pair(
    replay, warmup, *, warmup_fraction, historical=None, historical_fraction=0.0
):
    """Choose a training source; empty/expired history falls back to recent data."""
    _check_fractions(warmup_fraction, historical_fraction)
    draw = np.random.random()
    if draw < warmup_fraction:
        return warmup
    if historical is not None and draw < warmup_fraction + historical_fraction:
        pair = historical.sample_pair()
        if pair is not None:
            return pair
    return replay
