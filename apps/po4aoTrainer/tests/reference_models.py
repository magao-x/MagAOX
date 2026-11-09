## \file reference_models.py
# \brief Legacy linear models retained only for diagnostic regression tests.
"""Extracted from the pinned source; configuration defaults become explicit arguments."""

import torch
from torch import nn
from xapp.po4aoTrainer.po4ao_nonlinear_models import (
    ModalLocallyConnected,
    ModalNormalizer,
)


class ConvDynamics(nn.Module):
    """Independent linear temporal dynamics model for every modal coefficient."""

    def __init__(self, mask, n_history, *, control_delay):
        """Build legacy linear dynamics with an explicit known delay."""
        super().__init__()
        self.n_history = n_history
        self.control_delay = control_delay
        if not 0 <= self.control_delay < n_history:
            raise ValueError("control_delay must be between zero and n_history - 1")
        self.normalizer = ModalNormalizer(n_history, int(mask.numel()))
        self.net = ModalLocallyConnected(n_history * 2, int(mask.numel()), bias=False)

    def set_modal_scales(self, observation_scale, action_scale):
        """Set fixed scales while retaining normalized-coordinate weights."""
        self.normalizer.set_scales(observation_scale, action_scale)

    def raw_linear_weights(self):
        """Return effective feature-major dynamics weights in raw modal units."""
        feature_scale = self.normalizer.dynamics_feature_scale()[0, :, 0]
        output_scale = self.normalizer.observation_scale.reshape(-1)
        weights = self.net.weight[0, :, 0] * output_scale / feature_scale
        weights = weights.clone()
        if self.control_delay > 0:
            weights[-1] = 0
            if self.control_delay > 1:
                first_unavailable = 2 * self.n_history - self.control_delay - 1
                weights[first_unavailable : 2 * self.n_history - 2] = 0
        return weights

    def forward(self, feats):
        """Predict raw measurements while suppressing unavailable action features."""
        feats = self.normalizer.normalize_dynamics_features(feats)
        if self.control_delay == 0:
            return self.normalizer.denormalize_observation(self.net(feats))

        delayed_feats = feats.clone()
        delayed_feats[:, -1:] = 0
        if self.control_delay > 1:
            first_unavailable = 2 * self.n_history - self.control_delay - 1
            delayed_feats[:, first_unavailable : 2 * self.n_history - 2] = 0
        return self.normalizer.denormalize_observation(self.net(delayed_feats))


class LinearLocallyConnectedPolicy(nn.Module):
    """Independent linear temporal predictor for every modal coefficient."""

    def __init__(self, n_history, n_modes, *, integrator_gain):
        """Initialize the legacy linear policy to the supplied integrator gain."""
        super().__init__()
        self.normalizer = ModalNormalizer(n_history, n_modes)
        self.output = ModalLocallyConnected(n_history * 2 - 1, n_modes, bias=False)
        gain = integrator_gain
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.weight[:, 0].fill_(gain)

    def set_modal_scales(self, observation_scale, action_scale):
        """Set normalization while preserving the current raw linear policy."""
        raw_weights = self.raw_linear_weights().detach().clone()
        self.normalizer.set_scales(observation_scale, action_scale)
        feature_scale = self.normalizer.policy_feature_scale()[0, :, 0]
        output_scale = self.normalizer.action_scale.reshape(-1)
        with torch.no_grad():
            self.output.weight[0, :, 0].copy_(
                raw_weights * feature_scale / output_scale
            )

    def raw_linear_weights(self):
        """Return feature-major weights for inference directly in raw units."""
        feature_scale = self.normalizer.policy_feature_scale()[0, :, 0]
        output_scale = self.normalizer.action_scale.reshape(-1)
        return self.output.weight[0, :, 0] * output_scale / feature_scale

    def normalize_action(self, values):
        """Normalize raw commands for the policy loss."""
        return self.normalizer.normalize_action(values)

    def forward(self, feats):
        """Apply the linear policy and restore raw command units."""
        normalized = self.normalizer.normalize_policy_features(feats)
        output = self.normalizer.denormalize_action(self.output(normalized))
        return output.flatten(start_dim=1)
