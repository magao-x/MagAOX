## \file po4ao_policy_update.py
# \brief Packaged policy update from the nonlinear notebook.
"""Fixed quadratic action-change regularization, never real-time inference."""

from copy import deepcopy

import numpy as np
import torch


def policy_history_batch(
    observations, actions, *, n_history, episode_length, samples, device
):
    """Select deterministic real histories without crossing complete episodes.

    Pass training episodes for anchors and a separate held-out episode for the
    diagnostic check. Feature order matches C++: current observation, past
    observations oldest-first, past commands oldest-first.
    """
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if (
        observations.ndim != 2
        or observations.shape != actions.shape
        or not observations.size
        or not np.isfinite(observations).all()
        or not np.isfinite(actions).all()
    ):
        raise ValueError(
            "Policy-update data must be finite matching [time, modes] arrays"
        )
    if (
        not 0 < n_history <= episode_length
        or samples < 1
        or len(observations) % episode_length
    ):
        raise ValueError(
            "Policy-update histories require complete, sufficiently long episodes"
        )
    per_episode = episode_length - n_history + 1
    count = len(observations) // episode_length * per_episode
    chosen = np.linspace(0, count - 1, min(samples, count), dtype=int)
    current = (
        chosen // per_episode * episode_length + chosen % per_episode + n_history - 1
    )
    past = current[:, None] + np.arange(-n_history + 1, 0)
    features = np.concatenate(
        [observations[current, None], observations[past], actions[past]], axis=1
    )
    return torch.as_tensor(features, device=device).unsqueeze(2)


class PolicyUpdatePenalty:
    """Fixed normalized squared action-change loss, without a distance bound.

    Local architectures allow scalar or per-mode weights. Coupled architectures
    use a scalar weight on the dimension-normalized full-vector distance.
    The reference changes only on acknowledgment of a published model ID; an
    unknown active model fails closed. This is not a closed-loop safety proof.
    """

    def __init__(
        self,
        *,
        architecture,
        action_scale,
        baseline_gain,
        weight=1.0,
    ):
        """Freeze physical command scales and non-negative scalar/modal loss weights."""
        if architecture not in {
            "linear_local",
            "nonlinear_local",
            "shared_local_output",
            "nonlinear_global",
        }:
            raise ValueError("Unknown policy-update architecture")
        self.per_mode = architecture != "nonlinear_global"
        self.scale = torch.as_tensor(action_scale, dtype=torch.float32).detach().clone()
        self.weight = (
            torch.as_tensor(weight, dtype=torch.float32).reshape(-1).detach().clone()
        )
        if (
            self.scale.ndim != 1
            or not self.scale.numel()
            or not torch.isfinite(self.scale).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError("Policy-update action scales must be positive and finite")
        allowed = {1, self.scale.numel()} if self.per_mode else {1}
        if (
            self.weight.numel() not in allowed
            or not torch.isfinite(self.weight).all()
            or (self.weight < 0).any()
        ):
            raise ValueError(
                "Use a non-negative scalar weight, or per-mode weights for local policies"
            )
        if not np.isfinite(baseline_gain):
            raise ValueError("Policy-update baseline gain must be finite")
        self.baseline_gain = float(baseline_gain)
        self.reference = None  # Frozen acknowledged policy; None means fallback.
        self.reference_id = "0000000000000000"
        self.retained = (
            None  # Last acknowledged network, retained across temporary fallback.
        )
        self.pending = None  # At most one (model ID, frozen exported policy) pair.
        self.anchors = (
            None  # Fixed real histories, renewed only on acknowledgment/reset.
        )
        self.reference_actions = None

    @staticmethod
    def _model_id(value):
        """Canonicalize the hexadecimal ID used by PO4AO INDI telemetry."""
        number = int(str(value), 16)
        if not 0 <= number < 2**64:
            raise ValueError("Invalid active model ID")
        return f"{number:016x}"

    def observe_active_model(self, model_id):
        """Follow the effective acknowledged policy, including temporary fallback and return."""
        model_id = self._model_id(model_id)
        if self.pending is not None and model_id == self.pending[0]:
            self.reference_id, self.reference = self.pending
            self.retained = self.pending
            self.pending = None
        elif model_id == self.reference_id:
            return
        elif self.retained is not None and model_id == self.retained[0]:
            self.reference_id, self.reference = self.retained
        elif model_id == "0000000000000000":
            self.reference_id, self.reference = model_id, None
        else:
            raise RuntimeError(
                "Active policy is unknown to this trainer; reset_model and start a fresh run"
            )
        self.anchors = self.reference_actions = None

    def _reference_actions(self, features):
        """Evaluate a frozen selected policy, or the configured fallback gain."""
        with torch.no_grad():
            if self.reference is None:
                return self.baseline_gain * features[:, 0, 0]
            return self.reference(features).reshape(features.shape[0], -1)

    def set_anchors(self, features):
        """Fix a real-history batch until the next acknowledged reference change."""
        if features.ndim != 4 or features.shape[2:] != (1, self.scale.numel()):
            raise ValueError(
                "Policy-update anchors must have shape [batch, features, 1, modes]"
            )
        if not features.shape[0] or not torch.isfinite(features).all():
            raise ValueError("Policy-update anchors must be non-empty and finite")
        self.anchors = features.detach().clone()
        self.scale = self.scale.to(features.device)
        self.weight = self.weight.to(features.device)
        self.reference_actions = self._reference_actions(self.anchors)

    def _squared_distance(self, policy, features, reference_actions):
        """Return each mode's normalized mean squared action change."""
        action = policy(features).reshape(features.shape[0], -1)
        return ((action - reference_actions) / self.scale).square().mean(dim=0)

    def penalty(self, policy):
        """Return mean weighted squared distance, differentiable in candidate weights."""
        if self.anchors is None:
            raise RuntimeError("Set policy-update anchors before training")
        distance = self._squared_distance(policy, self.anchors, self.reference_actions)
        return (self.weight * distance).mean()

    def metrics(self, policy, validation_features):
        """Report anchor/held-out distances and finiteness; impose no distance limit."""
        if self.anchors is None:
            raise RuntimeError("Set policy-update anchors before validation")
        with torch.no_grad():
            anchor = self._squared_distance(
                policy, self.anchors, self.reference_actions
            )
            heldout = self._squared_distance(
                policy,
                validation_features,
                self._reference_actions(validation_features),
            )
            penalty = (self.weight * anchor).mean()
            if not self.per_mode:
                anchor, heldout = anchor.mean().reshape(1), heldout.mean().reshape(1)
            distances = torch.maximum(anchor, heldout).sqrt()
        finite = bool(torch.isfinite(distances).all() and torch.isfinite(penalty))
        return {
            "policy_update_metric": "per_mode_rms" if self.per_mode else "joint_rms",
            "policy_update_reference_id": self.reference_id,
            "policy_update_pending_id": (
                None if self.pending is None else self.pending[0]
            ),
            "policy_update_weight": self.weight.tolist(),
            "policy_update_rms": distances.tolist() if finite else None,
            "policy_update_max_rms": float(distances.max()) if finite else None,
            "policy_update_anchor_penalty": float(penalty) if finite else None,
            "policy_update_finite": finite,
        }

    def stage_publication(self, policy, model_id):
        """Retain the exact exported weights until C++ acknowledges their ID."""
        if self.pending is not None:
            raise RuntimeError(
                "Wait for acknowledgment before publishing another policy"
            )
        model_id = self._model_id(model_id)
        if model_id == "0000000000000000":
            raise ValueError("Published models cannot use the fallback ID")
        if model_id == self.reference_id:
            return  # An unchanged artifact does not advance the reference.
        reference = deepcopy(policy).eval()
        reference.requires_grad_(False)
        self.pending = (model_id, reference)
