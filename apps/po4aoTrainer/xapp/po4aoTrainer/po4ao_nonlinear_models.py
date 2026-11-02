## \file po4ao_nonlinear_models.py
# \brief Packaged nonlinear models from the nonlinear notebook.
"""Small shared nonlinear modal models with exact raw-unit identity starts.

Policy features retain the existing current-observation, past-observation,
past-action order. Dynamics features retain past observations/actions followed
by the current observation/action pair. No local hidden layer or output bias is
introduced. The reference modal models are unchanged.

Both training loops may sample elapsed-time historical episodes. History expiry
is checked at batch selection, including during long dynamics pretraining calls.
"""

import numpy as np
import torch
from torch import nn

from .po4ao_models import ModalLocallyConnected, ModalNormalizer
from .po4ao_replay import sample_replay_pair


class NonlinearModalModel(nn.Module):
    """Use shared temporal layers and one bias-free local output per mode."""

    def __init__(
        self,
        n_history,
        n_modes,
        *,
        policy=True,
        architecture="nonlinear_local",
        channels=8,
        shared_layers=2,
        negative_slope=0.1,
        low_order_modes=50,
        global_hidden=4,
        global_features=2,
        control_delay=0,
    ):
        """Construct shared temporal layers with an optional low-order global encoder."""
        super().__init__()
        if n_history < 1 or n_modes < 1 or channels < 2 or shared_layers < 2:
            raise ValueError(
                "Positive history/modes and at least two channels/layers are required"
            )
        if architecture not in {"nonlinear_local", "nonlinear_global"}:
            raise ValueError(f"Unsupported nonlinear architecture: {architecture}")
        if not 0 < negative_slope < 1:
            raise ValueError("negative_slope must be between zero and one")
        if not 0 <= control_delay < n_history:
            raise ValueError("control_delay must be smaller than n_history")
        if min(low_order_modes, global_hidden, global_features) < 1:
            raise ValueError("Global encoder dimensions must be positive")

        self.n_history = n_history
        self.n_modes = n_modes
        self.is_policy = policy
        self.control_delay = control_delay
        self.architecture = architecture
        self.negative_slope = negative_slope
        self.low_order_modes = min(low_order_modes, n_modes)
        self.normalizer = ModalNormalizer(n_history, n_modes)
        input_features = 2 * n_history - int(policy)
        self.current_feature = 0 if policy else input_features - 2
        use_global = architecture == "nonlinear_global"
        self.global_encoder = (
            nn.Sequential(
                nn.Linear(self.low_order_modes * channels, global_hidden),
                nn.LeakyReLU(negative_slope),
                nn.Linear(global_hidden, global_features),
                nn.LeakyReLU(negative_slope),
            )
            if use_global
            else None
        )
        self.shared = nn.ModuleList()
        for index in range(shared_layers):
            in_channels = input_features if index == 0 else channels
            if index == shared_layers - 1 and use_global:
                in_channels += global_features
            self.shared.append(nn.Conv2d(in_channels, channels, kernel_size=1))
        self.activation = nn.LeakyReLU(negative_slope)
        self.output = ModalLocallyConnected(channels, n_modes, bias=False)
        self.register_buffer(
            "zero_features", torch.zeros(1, input_features, 1, n_modes)
        )
        self.initialize_identity()

    def initialize_identity(self):
        """Set the raw output to the current observation for either input sign.

        L(x)-L(-x)=(1+slope)*x supplies an exact identity through every shared
        activation. Unused channels remain small random features that can be learned;
        their initial output weights are zero. Global inputs initially have no effect.
        """
        factor = 1.0 / (1.0 + self.negative_slope)
        with torch.no_grad():
            for index, layer in enumerate(self.shared):
                nn.init.normal_(layer.weight, std=0.02)
                layer.bias.zero_()
                layer.weight[:2].zero_()
                if index == 0:
                    layer.weight[0, self.current_feature, 0, 0] = 1
                    layer.weight[1, self.current_feature, 0, 0] = -1
                else:
                    layer.weight[0, 0, 0, 0] = factor
                    layer.weight[0, 1, 0, 0] = -factor
                    layer.weight[1, 0, 0, 0] = -factor
                    layer.weight[1, 1, 0, 0] = factor
            if self.global_encoder is not None:
                for layer in self.global_encoder:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, std=0.02)
                        layer.bias.zero_()
            ratio = (
                self.normalizer.observation_scale / self.normalizer.action_scale
                if self.is_policy
                else torch.ones_like(self.normalizer.observation_scale)
            )
            self.output.weight.zero_()
            self.output.weight[:, 0:1].copy_(factor * ratio)
            self.output.weight[:, 1:2].copy_(-factor * ratio)

    def set_modal_scales(self, observation_scale, action_scale):
        """Set frozen warm-up scales and reinitialize before any training.

        Changing scales on a trained nonlinear network is not scale-invariant. This
        method deliberately resets to identity; call it only once before training.
        """
        self.normalizer.set_scales(observation_scale, action_scale)
        self.initialize_identity()

    def normalize_action(self, values):
        """Normalize raw commands for the policy objective."""
        return self.normalizer.normalize_action(values)

    def _network(self, features):
        """Fuse low-order context before the last shared nonlinear layer."""
        context_source = None
        for index, layer in enumerate(self.shared):
            if index == len(self.shared) - 1 and self.global_encoder is not None:
                context = (
                    self.global_encoder(
                        context_source[:, :, :, : self.low_order_modes].flatten(1)
                    )
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
                features = torch.cat(
                    [features, context.expand(-1, -1, 1, self.n_modes)], dim=1
                )
            features = self.activation(layer(features))
            if index == 0:
                context_source = features
        return self.output(features)

    def forward(self, features):
        """Return raw commands or next observations with the existing contract."""
        if self.is_policy:
            normalized = self.normalizer.normalize_policy_features(features)
            # Bias-free output alone does not remove offsets from hidden biases.
            output = self._network(normalized) - self._network(self.zero_features)
            return self.normalizer.denormalize_action(output).flatten(start_dim=1)

        normalized = self.normalizer.normalize_dynamics_features(features)
        if self.control_delay:
            normalized = normalized.clone()
            normalized[:, -1:] = 0
            if self.control_delay > 1:
                first = 2 * self.n_history - self.control_delay - 1
                normalized[:, first : 2 * self.n_history - 2] = 0
        return self.normalizer.denormalize_observation(self._network(normalized))


class NonlinearModalDynamics(nn.Module):
    """Expose one nonlinear dynamics model through the existing diagnostics API."""

    def __init__(self, n_history, n_modes, **settings):
        """Wrap a single configured nonlinear predictor for the diagnostics interface."""
        super().__init__()
        self.models = nn.ModuleList(
            [NonlinearModalModel(n_history, n_modes, policy=False, **settings)]
        )

    def set_modal_scales(self, observation_scale, action_scale):
        """Fix warm-up normalization and reset the dynamics model to identity."""
        self.models[0].set_modal_scales(observation_scale, action_scale)

    def normalize_observation(self, values):
        """Normalize raw measurements for supervised and policy losses."""
        return self.models[0].normalizer.normalize_observation(values)

    def forward(self, features):
        """Predict one next measurement per mode, without an ensemble."""
        return self.models[0](features)


def build_nonlinear_models(config):
    """Construct independent identity-initialized dynamics and policy networks."""
    settings = dict(config["nonlinear"])
    n_history = config["MDP"]["n_history"]
    n_modes = config["runtime"]["n_modes_controlled"]
    settings["control_delay"] = config["MDP"]["control_delay"]
    return (
        NonlinearModalDynamics(n_history, n_modes, **settings),
        NonlinearModalModel(n_history, n_modes, **settings),
    )


def sample_training_history(
    observations, actions, *, n_history, episode_length, batch_size, device
):
    """Sample complete histories/targets without crossing or using validation episodes."""
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if observations.ndim != 2 or actions.shape != observations.shape:
        raise ValueError(
            "observations/actions must have matching [samples, modes] shapes"
        )
    if observations.shape[0] % episode_length or episode_length <= n_history:
        raise ValueError(
            "Training data must contain complete episodes longer than the history"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    episode = np.random.randint(
        observations.shape[0] // episode_length, size=batch_size
    )
    current = episode * episode_length + np.random.randint(
        n_history - 1, episode_length - 1, size=batch_size
    )
    indices = current[:, None] + np.arange(-n_history + 1, 1)
    states = torch.as_tensor(observations[indices], device=device).unsqueeze(2)
    commands = torch.as_tensor(actions[indices], device=device).unsqueeze(2)
    target = (
        torch.as_tensor(observations[current + 1], device=device)
        .unsqueeze(1)
        .unsqueeze(1)
    )
    return states, commands, target


def train_nonlinear_dynamics(
    optimizer,
    dynamics,
    replay,
    warmup,
    *,
    settings,
    steps,
    device,
    historical=None,
    checkpoint=None,
):
    """Fit episode-local predictions from recent, warm-up and unexpired history."""
    dynamics.train()
    loss = None
    for _ in range(steps):
        if checkpoint is not None:
            checkpoint()
        observations, actions = sample_replay_pair(
            replay,
            warmup,
            warmup_fraction=settings["replay_buffers"]["train_warmup_percent"],
            historical=historical,
            historical_fraction=settings["replay_buffers"].get(
                "train_historical_percent", 0.0
            ),
        )
        states, commands, target = sample_training_history(
            observations,
            actions,
            n_history=settings["MDP"]["n_history"],
            episode_length=settings["RL"]["episode_length"],
            batch_size=settings["NN_models"]["training_batch"],
            device=device,
        )
        features = torch.cat(
            [states[:, :-1], commands[:, :-1], states[:, -1:], commands[:, -1:]], dim=1
        )
        optimizer.zero_grad(set_to_none=True)
        loss = (
            dynamics.normalize_observation(dynamics(features) - target).square().mean()
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite dynamics loss; no model should be published")
        loss.backward()
        nn.utils.clip_grad_norm_(
            dynamics.parameters(),
            settings["training"]["dynamics_gradient_clip_norm"],
            error_if_nonfinite=True,
        )
        optimizer.step()
    if loss is None:
        raise ValueError("steps must be positive")
    return float(loss.detach())


def train_nonlinear_policy(
    optimizer,
    policy,
    dynamics,
    replay,
    warmup,
    *,
    settings,
    steps,
    device,
    policy_update=None,
    historical=None,
    checkpoint=None,
):
    """Optimize frozen-dynamics rollouts with an optional real-history distance penalty.

    The quadratic penalty backpropagates through the candidate policy only. The
    returned loss remains the rollout objective, not the regularized objective.
    Action-change distance is diagnostic only, not a publication constraint.
    Historical episodes, when supplied, are checked for expiry each gradient step.
    """
    policy.train()
    was_training = dynamics.training
    requires_grad = [parameter.requires_grad for parameter in dynamics.parameters()]
    dynamics.eval()
    for parameter in dynamics.parameters():
        parameter.requires_grad_(False)
    loss = None
    try:
        for _ in range(steps):
            if checkpoint is not None:
                checkpoint()
            observations, actions = sample_replay_pair(
                replay,
                warmup,
                warmup_fraction=settings["replay_buffers"]["train_warmup_percent"],
                historical=historical,
                historical_fraction=settings["replay_buffers"].get(
                    "train_historical_percent", 0.0
                ),
            )
            states, commands, _ = sample_training_history(
                observations,
                actions,
                n_history=settings["MDP"]["n_history"],
                episode_length=settings["RL"]["episode_length"],
                batch_size=settings["NN_models"]["training_batch"],
                device=device,
            )
            past_observations, past_actions, state = (
                states[:, :-1],
                commands[:, :-1],
                states[:, -1:],
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for _ in range(settings["MDP"]["planning_horizon"]):
                history = torch.cat([past_observations, past_actions], dim=1)
                action = (
                    policy(torch.cat([state, history], dim=1)).unsqueeze(1).unsqueeze(1)
                )
                prediction = dynamics(torch.cat([history, state, action], dim=1))
                loss = loss + dynamics.normalize_observation(prediction).square().mean()
                loss = (
                    loss
                    + settings["RL"]["loss_function_penalty"]
                    * policy.normalize_action(action).square().mean()
                )
                past_observations = torch.cat([past_observations[:, 1:], state], dim=1)
                past_actions = torch.cat([past_actions[:, 1:], action], dim=1)
                state = prediction
            objective = (
                loss if policy_update is None else loss + policy_update.penalty(policy)
            )
            if not torch.isfinite(objective):
                raise RuntimeError(
                    "Non-finite policy loss; no model should be published"
                )
            objective.backward()
            nn.utils.clip_grad_norm_(
                policy.parameters(),
                settings["training"]["policy_gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
    finally:
        for parameter, enabled in zip(dynamics.parameters(), requires_grad):
            parameter.requires_grad_(enabled)
        dynamics.train(was_training)
    if loss is None:
        raise ValueError("steps must be positive")
    return float(loss.detach())
