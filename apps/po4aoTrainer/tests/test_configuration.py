## \file test_configuration.py
# \brief Public penalty-name compatibility and live numerical update regressions.
# \ingroup po4aoTrainer_unit_test
"""Use real configuration parsing and CPU regularization without instrument I/O."""

from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np
import pytest
import tomli_w
import torch

from test_supervisor import config
from xapp.po4aoTrainer.configuration import (
    apply_hot,
    load_training,
    needs_reset,
    revision,
)
from xapp.po4aoTrainer.training import TrainingSession


@pytest.mark.parametrize("penalty", [0.0, 0.5, [0.1, 0.2, 0.3, 0.4]])
def test_penalty_spelling_preserves_resolved_config(config, penalty):
    """Renaming an installed key alone preserves values, revision and Reset eligibility."""
    path = Path(config.training_config)
    settings = tomllib.loads(path.read_text())
    settings["policy_update"]["weight"] = penalty
    path.write_text(tomli_w.dumps(settings))
    previous = load_training(path, config.runtime_config)
    settings["policy_update"]["penalty"] = settings["policy_update"].pop("weight")
    path.write_text(tomli_w.dumps(settings))
    current = load_training(path, config.runtime_config)
    assert current == previous
    assert revision(current) == revision(previous)
    assert current["policy_update"]["weight"] == penalty
    assert not needs_reset(previous, current)


@pytest.mark.parametrize("penalty", [0.5, 1.0])
def test_penalty_rejects_both_spellings(config, penalty):
    """Do not silently choose either an equal or conflicting duplicate setting."""
    path = Path(config.training_config)
    settings = tomllib.loads(path.read_text())
    settings["policy_update"]["penalty"] = penalty
    path.write_text(tomli_w.dumps(settings))
    with pytest.raises(ValueError, match="not both"):
        load_training(path, config.runtime_config)


@pytest.mark.parametrize(
    "penalty", [-0.5, [0.1, -0.2, 0.3, 0.4], [0.1, 0.2], float("nan")]
)
def test_penalty_keeps_validation(config, penalty):
    """The new spelling still rejects invalid scalars, modal values and modal counts."""
    path = Path(config.training_config)
    settings = tomllib.loads(path.read_text())
    settings["policy_update"].pop("weight")
    settings["policy_update"]["penalty"] = penalty
    path.write_text(tomli_w.dumps(settings))
    with pytest.raises(ValueError):
        load_training(path, config.runtime_config)


def test_penalty_hot_update_reaches_existing_regularizer(config):
    """Hot editing the public key changes the real penalty while retaining the policy and Adam."""
    path = Path(config.training_config)
    previous = load_training(path, config.runtime_config)
    session = TrainingSession(previous, device="cpu")
    warmup = np.ones((96, session.controlled_nmodes), dtype=np.float32)
    session._initialize_normalization((warmup, 0.3 * warmup))
    anchors = torch.ones(8, 2 * session.n_history - 1, 1, session.controlled_nmodes)
    session.policy_update.set_anchors(anchors)
    original_cost = session.policy_update.penalty(session.policy).detach()
    assert original_cost > 0
    policy, optimizer = session.policy, session.policy_optimizer
    parameters = deepcopy(policy.state_dict())
    normalization = deepcopy(session.normalization_metadata)
    optimizer.state["sentinel"] = {"step": 7}

    settings = tomllib.loads(path.read_text())
    settings["policy_update"].pop("weight")
    settings["policy_update"]["penalty"] = 0.5
    settings["RL"]["loss_function_penalty"] = 0.01
    path.write_text(tomli_w.dumps(settings))
    candidate = load_training(path, config.runtime_config)
    assert not needs_reset(previous, candidate)
    apply_hot(session, candidate)
    assert session.policy is policy and session.policy_optimizer is optimizer
    assert optimizer.state["sentinel"]["step"] == 7
    assert session.normalization_metadata == normalization
    assert session.config["RL"]["loss_function_penalty"] == 0.01
    assert session.config_revision == revision(candidate)
    for name, value in parameters.items():
        torch.testing.assert_close(
            session.policy.state_dict()[name], value, rtol=0, atol=0
        )
    assert session.policy_update.anchors is None
    session.policy_update.set_anchors(anchors)
    torch.testing.assert_close(
        session.policy_update.penalty(session.policy).detach(), 0.5 * original_cost
    )
