## \file po4ao_models.py
# \brief Configuration-independent modal layers used by the nonlinear trainer.
"""Shared layers extracted without the legacy module's import-time configuration."""

import math
import torch
from torch import nn


class ModalLocallyConnected(nn.Module):
    """Map temporal features to one output with independent weights per mode."""

    def __init__(self, in_features, n_modes, *, bias=True):
        """Initialize independent per-mode weights and optional bias."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, in_features, 1, n_modes))
        self.bias = nn.Parameter(torch.empty(1, 1, 1, n_modes)) if bias else None
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, feats):
        """Reduce temporal features to one weighted output per mode."""
        output = (feats * self.weight).sum(dim=1, keepdim=True)
        return output if self.bias is None else output + self.bias


class ModalNormalizer(nn.Module):
    """Normalize raw modal histories while preserving a raw-unit model interface."""

    def __init__(self, n_history, n_modes):
        """Register fixed modal scaling buffers, initially unity."""
        super().__init__()
        self.n_history = n_history
        self.n_modes = n_modes
        shape = (1, 1, 1, n_modes)
        self.register_buffer("observation_scale", torch.ones(shape))
        self.register_buffer("action_scale", torch.ones(shape))

    def set_scales(self, observation_scale, action_scale):
        """Replace the fixed per-mode scales after validating their contract."""
        observation_scale = self._validated_scale(observation_scale, "observation")
        action_scale = self._validated_scale(action_scale, "action")
        with torch.no_grad():
            self.observation_scale.copy_(observation_scale)
            self.action_scale.copy_(action_scale)

    def _validated_scale(self, values, label):
        """Validate one finite positive scale per mode on the model device."""
        scale = torch.as_tensor(
            values,
            dtype=self.observation_scale.dtype,
            device=self.observation_scale.device,
        ).reshape(-1)
        if scale.numel() != self.n_modes:
            raise ValueError(f"{label} scale must contain one value per mode")
        if not torch.isfinite(scale).all() or not torch.all(scale > 0):
            raise ValueError(f"{label} scale must be finite and positive")
        return scale.reshape(1, 1, 1, self.n_modes)

    def policy_feature_scale(self):
        """Return scales in current/past-observation then past-action order."""
        return torch.cat(
            [
                self.observation_scale.expand(-1, self.n_history, -1, -1),
                self.action_scale.expand(-1, self.n_history - 1, -1, -1),
            ],
            dim=1,
        )

    def dynamics_feature_scale(self):
        """Return scales in past-observation/action then current-pair order."""
        return torch.cat(
            [
                self.observation_scale.expand(-1, self.n_history - 1, -1, -1),
                self.action_scale.expand(-1, self.n_history - 1, -1, -1),
                self.observation_scale,
                self.action_scale,
            ],
            dim=1,
        )

    def normalize_policy_features(self, values):
        """Normalize a raw policy-history tensor."""
        return values / self.policy_feature_scale()

    def normalize_dynamics_features(self, values):
        """Normalize a raw dynamics-history tensor."""
        return values / self.dynamics_feature_scale()

    def normalize_observation(self, values):
        """Normalize values whose final dimension is modal observation."""
        return values / self.observation_scale.reshape(-1)

    def normalize_action(self, values):
        """Normalize values whose final dimension is modal command."""
        return values / self.action_scale.reshape(-1)

    def denormalize_observation(self, values):
        """Restore normalized observations to raw modal units."""
        return values * self.observation_scale

    def denormalize_action(self, values):
        """Restore normalized commands to raw modal units."""
        return values * self.action_scale
