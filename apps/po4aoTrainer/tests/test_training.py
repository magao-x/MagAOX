## \file test_training.py
# \brief CPU regression tests for persistent notebook training sessions.
"""Test numerical parity, replay boundaries and cooperative pause without streams."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import threading

import numpy as np
import pytest
import torch

from xapp.po4aoTrainer.training import TrainingSession
from xapp.po4aoTrainer.po4ao_runtime import ReplaySnapshot

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


def test_waits_do_not_advance_or_reinitialize(inputs):
    """Partial and duplicate snapshots never change weights, Adam, history or counters."""
    config, recent, warmup = inputs
    session = make_session(config)
    assert (
        session.step(replace(recent, valid_length=0), warmup, active_model_id="0")
        is None
    )
    assert session.iteration == 0 and session.normalization_metadata is None
    session.step(recent, warmup, active_model_id="0")
    optimizer = session.policy_optimizer
    state = deepcopy(optimizer.state_dict())
    parameters = deepcopy(session.policy.state_dict())
    history_count = session.historical.metrics()["historical_episodes"]
    assert session.step(recent, warmup, active_model_id="0") is None
    assert session.iteration == 1
    assert session.historical.metrics()["historical_episodes"] == history_count
    for key, value in parameters.items():
        torch.testing.assert_close(
            value, session.policy.state_dict()[key], rtol=0, atol=0
        )
    for index, values in state["state"].items():
        for key, value in values.items():
            torch.testing.assert_close(
                value, optimizer.state_dict()["state"][index][key], rtol=0, atol=0
            )
    session.step(replace(recent, counter=2, episode_id=4), warmup, active_model_id="0")
    assert session.policy_optimizer is optimizer
    assert session.iteration == 2
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


def test_recorded_diagnostics_match_direct_plots(inputs, tmp_path):
    """Render identical notebook figures from direct state and matching bounded snapshots."""
    import matplotlib.pyplot as plt
    from xapp.po4aoTrainer.observer import save_diagnostics, load_diagnostics
    from xapp.po4aoTrainer.po4ao_diagnostics import (
        plot_training_diagnostics,
        plot_model_debug_diagnostics,
    )

    config, recent, warmup = inputs
    session = make_session(config)
    session.step(recent, warmup, active_model_id="0")
    direct = (
        plot_training_diagnostics(
            session.metrics_history,
            session.obs_replay,
            session.act_replay,
            sample_rate_hz=2000,
        ),
        plot_model_debug_diagnostics(session.metrics_history),
    )
    snapshot = load_diagnostics(save_diagnostics(session, tmp_path, keep=2))
    recorded = snapshot.plot()
    try:
        assert snapshot.metadata["iteration"] == session.iteration
        assert snapshot.metadata["config_revision"] == session.config_revision
        assert snapshot.age_seconds >= 0
        np.testing.assert_array_equal(snapshot.observations, session.obs_replay)
        for left, right in zip(direct, recorded):
            left.canvas.draw()
            right.canvas.draw()
            np.testing.assert_array_equal(
                np.asarray(left.canvas.buffer_rgba()),
                np.asarray(right.canvas.buffer_rgba()),
            )
        for iteration in (2, 3):
            session.step(
                replace(recent, counter=iteration, episode_id=iteration + 2),
                warmup,
                active_model_id="0",
            )
            save_diagnostics(session, tmp_path, keep=2)
        assert len(list(tmp_path.glob("diagnostics-*.npz"))) == 2
    finally:
        for figure in direct + recorded:
            plt.close(figure)
