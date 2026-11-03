## \file test_po4ao_runtime.py
# \brief Regression tests ported with the numerical implementation.
"""Tests for replay metadata, compute isolation and policy publication."""

from types import SimpleNamespace
from itertools import product
import os
import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from xapp.po4aoTrainer.po4ao_runtime import (
    CPU_LINEAR_ARTIFACT_MAGIC,
    cached_replay_snapshot,
    configure_compute_resources,
    LINEAR_POLICY_MAGIC,
    LINEAR_POLICY_VERSION,
    _engine_id,
    _parse_cpu_list,
    _trtexec_command,
    _write_linear_policy,
    modal_rms_scales,
    model_publication_feedback,
    publish_tensorrt_engine,
    save_training_snapshot,
    select_controlled_modes,
    stable_replay_prefix_snapshot,
    stable_replay_snapshot,
)


class ModelPublicationFeedbackTest(unittest.TestCase):
    """Keep publication eligibility unchanged while explaining every veto."""

    def feedback(self, **overrides):
        """Evaluate a ready candidate with optional gate failures."""
        settings = dict(
            iteration=300,
            first_iteration=200,
            last_publish_iteration=200,
            publish_interval=100,
            candidate_clip_fraction=0.0,
            max_clip_fraction=0.02,
            candidate_action_rms_ratio=1.5,
            max_action_rms_ratio=1.5,
            nonfinite_samples=0,
            policy_update_finite=True,
            pending_model_id=None,
            active_model_id="0000000000000000",
            reference_model_id="0000000000000000",
        )
        settings.update(overrides)
        return model_publication_feedback(**settings)

    def test_exact_limits_are_ready(self):
        """Verify exact limits are ready."""
        feedback = self.feedback(candidate_clip_fraction=0.02)
        self.assertEqual(feedback["publication_status"], "ready")
        self.assertEqual(feedback["publication_reasons"], [])

    def test_explains_timing_and_pending_activation(self):
        """Verify explains timing and pending activation."""
        feedback = self.feedback(
            iteration=150, last_publish_iteration=100, pending_model_id="abc"
        )
        self.assertEqual(feedback["publication_status"], "waiting")
        self.assertEqual(len(feedback["publication_reasons"]), 3)
        self.assertIn("iteration 150/200", feedback["publication_reasons"][0])
        self.assertIn("50/100 iterations", feedback["publication_reasons"][1])
        self.assertIn("activation of model abc", feedback["publication_reasons"][2])

    def test_reports_all_safety_failures_even_during_cooldown(self):
        """Verify reports all safety failures even during cooldown."""
        feedback = self.feedback(
            iteration=250,
            candidate_clip_fraction=0.03,
            candidate_action_rms_ratio=1.94,
            nonfinite_samples=2,
            policy_update_finite=False,
            active_model_id="abc",
        )
        self.assertEqual(feedback["publication_status"], "blocked")
        reasons = feedback["publication_reasons"]
        self.assertEqual(len(reasons), 6)
        self.assertIn("3.00% > 2.00%", reasons[0])
        self.assertIn("1.940 > 1.500", reasons[1])
        self.assertIn("2 non-finite samples", reasons[2])
        self.assertIn("50/100 iterations", reasons[-1])

    def test_nonfinite_candidate_values_still_block(self):
        """Verify nonfinite candidate values still block."""
        for field in ("candidate_clip_fraction", "candidate_action_rms_ratio"):
            with self.subTest(field=field):
                self.assertEqual(
                    self.feedback(**{field: float("nan")})["publication_status"],
                    "blocked",
                )

    def test_eligibility_matches_original_conditions(self):
        """Compare all gate combinations, including boundary and NaN values."""
        for iteration, last, clip, rms, bad_samples, finite, pending, active in product(
            (199, 200, 299, 300),
            (None, 200),
            (0.02, 0.03, float("nan")),
            (1.5, 1.94, float("nan")),
            (0, 1),
            (True, False),
            (None, "abc"),
            ("0000000000000000", "abc"),
        ):
            expected = (
                iteration >= 200
                and (last is None or iteration - last >= 100)
                and clip <= 0.02
                and rms <= 1.5
                and bad_samples == 0
                and finite
                and pending is None
                and active == "0000000000000000"
            )
            feedback = self.feedback(
                iteration=iteration,
                last_publish_iteration=last,
                candidate_clip_fraction=clip,
                candidate_action_rms_ratio=rms,
                nonfinite_samples=bad_samples,
                policy_update_finite=finite,
                pending_model_id=pending,
                active_model_id=active,
            )
            self.assertEqual(feedback["publication_status"] == "ready", expected)


class FakeImage:
    """Minimal ImageStreamIO-like object for snapshot tests."""

    def __init__(self, data, counter=3):
        """Initialize the local test double without external resources."""
        self.data = np.asarray(data)
        self.md = SimpleNamespace(write=False, cnt0=counter)

    def __array__(self, dtype=None, copy=None):
        """Expose fake stream data through the NumPy array protocol."""
        return np.array(self.data, dtype=dtype, copy=True)


class StableReplaySnapshotTest(unittest.TestCase):
    """Validate stable-copy metadata and slicing."""

    def test_cache_refreshes_only_after_coherent_publication(self):
        """Reuse unchanged data; mismatched counters and busy writes cannot replace the cache."""
        image = FakeImage(np.ones((2, 6, 2), dtype=np.float32))
        metadata = FakeImage(np.array([6, 7, 0, 0], dtype=np.uint32))
        first = cached_replay_snapshot(image, metadata)
        with patch(
            "xapp.po4aoTrainer.po4ao_runtime.stable_replay_snapshot",
            wraps=stable_replay_snapshot,
        ) as capture:
            self.assertIs(cached_replay_snapshot(image, metadata, first), first)
            capture.assert_not_called()
            image.md.cnt0 = 4
            with self.assertRaises(TimeoutError):
                cached_replay_snapshot(image, metadata, first, attempts=1)
            metadata.md.cnt0 = 4
            image.data *= 2
            second = cached_replay_snapshot(image, metadata, first)
            self.assertIsNot(second, first)
            np.testing.assert_array_equal(first.data, np.ones_like(first.data))
            np.testing.assert_array_equal(second.data, np.full_like(second.data, 2))
            image.md.write = True
            with self.assertRaises(TimeoutError):
                cached_replay_snapshot(image, metadata, second, attempts=1)

    def test_uses_valid_length_and_publication_metadata(self):
        """Verify uses valid length and publication metadata."""
        image = FakeImage(np.arange(24).reshape(2, 6, 2))
        metadata = FakeImage(np.array([4, 7, 0x89ABCDEF, 0x01234567], dtype=np.uint32))
        snapshot = stable_replay_snapshot(image, metadata)
        self.assertEqual(snapshot.data.shape, (2, 4, 2))
        self.assertEqual(snapshot.counter, 3)
        self.assertEqual(snapshot.episode_id, 7)
        self.assertEqual(snapshot.model_id, "0123456789abcdef")

    def test_rejects_invalid_valid_length(self):
        """Verify rejects invalid valid length."""
        image = FakeImage(np.zeros((2, 6, 2)))
        metadata = FakeImage(np.array([7, 7, 0, 0], dtype=np.uint32))
        with self.assertRaises(RuntimeError):
            stable_replay_snapshot(image, metadata)

    def test_copies_only_requested_modal_prefix(self):
        """Verify copies only requested modal prefix."""
        image = FakeImage(np.arange(48).reshape(4, 6, 2))
        metadata = FakeImage(np.array([5, 9, 0, 0], dtype=np.uint32))
        snapshot = stable_replay_prefix_snapshot(image, metadata, modes=2)
        self.assertEqual(snapshot.data.shape, (2, 5, 2))
        np.testing.assert_array_equal(snapshot.data, image.data[:2, :5])


class ControlledModesTest(unittest.TestCase):
    """Validate controlled-prefix selection from full-sized replay data."""

    def test_selects_contiguous_prefix(self):
        """Verify selects contiguous prefix."""
        values = np.arange(24, dtype=np.float32).reshape(4, 6)
        selected = select_controlled_modes(values, 3)
        np.testing.assert_array_equal(selected, values[:, :3])
        self.assertTrue(selected.flags.c_contiguous)

    def test_rejects_invalid_count(self):
        """Verify rejects invalid count."""
        with self.assertRaises(ValueError):
            select_controlled_modes(np.zeros((4, 6), dtype=np.float32), 7)


class ModalNormalizationTest(unittest.TestCase):
    """Validate fixed warm-up-derived modal scales."""

    def test_computes_rms_and_floors_inactive_modes(self):
        """Verify computes rms and floors inactive modes."""
        observations = np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32)
        actions = np.array([[0.0, 6.0], [0.0, 8.0]], dtype=np.float32)

        observation_scale, action_scale = modal_rms_scales(
            observations, actions, floor_fraction=0.1
        )

        self.assertAlmostEqual(observation_scale[0], np.sqrt(12.5))
        self.assertAlmostEqual(observation_scale[1], 0.1 * np.sqrt(12.5))
        self.assertAlmostEqual(action_scale[0], 0.1 * np.sqrt(50.0))
        self.assertAlmostEqual(action_scale[1], np.sqrt(50.0))

    def test_rejects_nonfinite_warmup_data(self):
        """Verify rejects nonfinite warmup data."""
        observations = np.array([[1.0, np.nan]], dtype=np.float32)
        actions = np.ones_like(observations)

        with self.assertRaises(ValueError):
            modal_rms_scales(observations, actions)


class TensorRTCommandTest(unittest.TestCase):
    """Validate explicit GPU selection for engine building."""

    def test_selects_requested_build_device(self):
        """Verify selects requested build device."""
        command = _trtexec_command("trtexec", "model.onnx", "model.trt", "cache", 1)
        self.assertIn("--device=1", command)

    def test_rejects_invalid_build_device(self):
        """Verify rejects invalid build device."""
        with self.assertRaises(ValueError):
            _trtexec_command("trtexec", "model.onnx", "model.trt", "cache", -1)


class ComputeIsolationTest(unittest.TestCase):
    """Validate CPU-set parsing used to isolate notebook work."""

    def test_parses_linux_cpu_list(self):
        """Verify parses linux cpu list."""
        self.assertEqual(_parse_cpu_list("0-2,5,8-9\n"), {0, 1, 2, 5, 8, 9})

    def test_bounds_torch_threads_without_expanding_cpu_affinity(self):
        """Intersect system/allowed CPUs and retain a smaller operator-selected thread count."""
        properties = SimpleNamespace(name="test GPU", major=8, minor=6)
        with tempfile.TemporaryDirectory() as directory:
            cpuset = Path(directory) / "cpuset.cpus"
            cpuset.write_text("1-2,8")
            for threads in (1, 16):
                with self.subTest(threads=threads), patch(
                    "xapp.po4aoTrainer.po4ao_runtime.SYSTEM_CPUSET_FILE", cpuset
                ), patch(
                    "os.sched_getaffinity", side_effect=[{1, 2, 7}, {1, 2}], create=True
                ), patch(
                    "os.sched_setaffinity", create=True
                ) as affinity, patch(
                    "torch.cuda.is_available", return_value=True
                ), patch(
                    "torch.cuda.device_count", return_value=2
                ), patch(
                    "torch.cuda.set_device"
                ), patch(
                    "torch.cuda.get_device_properties", return_value=properties
                ), patch(
                    "torch.get_num_threads", side_effect=[threads, min(threads, 2)]
                ), patch(
                    "torch.set_num_threads"
                ) as limit:
                    result = configure_compute_resources(0, 1, 1)
                    limit.assert_called_once_with(min(threads, 2))
                    self.assertTrue(affinity.called)
                    self.assertTrue(
                        all(call.args[1] == {1, 2} for call in affinity.call_args_list)
                    )
                    self.assertEqual(result["torch_cpu_threads"], min(threads, 2))


class LinearPolicySidecarTest(unittest.TestCase):
    """Validate the versioned CPU linear-policy artifact."""

    def test_writes_header_model_id_and_feature_major_weights(self):
        """Verify writes header model id and feature major weights."""
        weights = np.arange(6, dtype=np.float32).reshape(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.linear")
            _write_linear_policy(path, weights, "0123456789abcdef")
            with open(path, "rb") as sidecar:
                payload = sidecar.read()

        header_size = struct.calcsize("<8sIIIQ")
        magic, version, features, modes, model_id = struct.unpack(
            "<8sIIIQ", payload[:header_size]
        )
        self.assertEqual(magic, LINEAR_POLICY_MAGIC)
        self.assertEqual(version, LINEAR_POLICY_VERSION)
        self.assertEqual((features, modes), weights.shape)
        self.assertEqual(model_id, 0x0123456789ABCDEF)
        np.testing.assert_array_equal(
            np.frombuffer(payload[header_size:], dtype="<f4").reshape(weights.shape),
            weights,
        )

    def test_linear_publication_skips_tensorrt_build(self):
        """Verify linear publication skips tensorrt build."""

        class FakeWeights:
            """Array-backed tensor stand-in for export tests without GPU allocation."""

            def __init__(self, values):
                """Initialize the local test double without external resources."""
                self.values = values

            def detach(self):
                """Return this array-backed stand-in for a detached tensor."""
                return self

            def cpu(self):
                """Return this already CPU-resident tensor stand-in."""
                return self

            def numpy(self):
                """Return the test tensor as a NumPy array."""
                return self.values

        class FakePolicy:
            """Linear policy stand-in for checking the CPU publication path."""

            output = SimpleNamespace(bias=None)

            def raw_linear_weights(self):
                """Expose reference weights in the runtime feature-major ordering."""
                return FakeWeights(np.arange(6, dtype=np.float32).reshape(3, 2))

        with tempfile.TemporaryDirectory() as directory:
            engine_path = os.path.join(directory, "policy.trt")
            model_id = publish_tensorrt_engine(
                FakePolicy(),
                None,
                engine_path,
                build_device=1,
                trtexec="/does/not/exist",
                metadata={"policy_architecture": "linear_local"},
            )
            with open(engine_path, "rb") as artifact:
                self.assertEqual(artifact.read(8), CPU_LINEAR_ARTIFACT_MAGIC)
            self.assertEqual(model_id, _engine_id(engine_path))
            self.assertTrue(os.path.isfile(engine_path + ".linear"))


class TrainingSnapshotTest(unittest.TestCase):
    """Validate portable, compact policy and dynamics snapshots."""

    def test_saves_linear_weights_and_model_identity(self):
        """Verify saves linear weights and model identity."""

        class FakeWeights:
            """Array-backed tensor stand-in for export tests without GPU allocation."""

            def __init__(self, values):
                """Initialize the local test double without external resources."""
                self.values = values

            def detach(self):
                """Return this array-backed stand-in for a detached tensor."""
                return self

            def cpu(self):
                """Return this already CPU-resident tensor stand-in."""
                return self

            def numpy(self):
                """Return the test tensor as a NumPy array."""
                return self.values

        class FakeLinearModel:
            """Minimal raw-weight model for snapshot serialization tests."""

            def __init__(self, values):
                """Initialize the local test double without external resources."""
                self.values = values

            def raw_linear_weights(self):
                """Expose reference weights in the runtime feature-major ordering."""
                return FakeWeights(self.values)

        policy = FakeLinearModel(np.ones((3, 2), dtype=np.float32))
        dynamics = SimpleNamespace(
            models=[FakeLinearModel(np.full((4, 2), 2, dtype=np.float32))]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "training.npz")
            save_training_snapshot(
                path,
                policy,
                dynamics,
                "0123456789abcdef",
                {"episode_id": 7},
            )
            with np.load(path, allow_pickle=False) as snapshot:
                np.testing.assert_array_equal(
                    snapshot["policy_raw_weights"], policy.values
                )
                np.testing.assert_array_equal(
                    snapshot["dynamics_0_raw_weights"], dynamics.models[0].values
                )
                self.assertIn("0123456789abcdef", str(snapshot["metadata_json"]))


if __name__ == "__main__":
    unittest.main()
